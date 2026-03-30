import argparse
import asyncio
import logging
import os
from typing import Any

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from agent import Agent
from executor import Executor


# ── Shared eval state for /results endpoint ─────────────────────────
# This is written by the auto-start background task and read by the
# /results HTTP handler. Same contract as the agentbeats gateway.
eval_state: dict[str, Any] = {
    "status": "idle",   # idle → running → completed | failed
    "result": None,
    "error": None,
}


async def results_handler(request: Request) -> JSONResponse:
    """HTTP endpoint that mirrors the gateway's results polling interface.

    Returns:
        {"status": "running"} while evaluation is in progress
        {"status": "completed", ...results...} when done
        {"status": "failed", "error": "..."} on failure
    """
    if eval_state["status"] in ("idle", "running"):
        return JSONResponse({"status": "running"})
    if eval_state["status"] == "failed":
        return JSONResponse({"status": "failed", "error": eval_state["error"]})
    # completed — return the full results dict (which includes status: completed)
    return JSONResponse(eval_state["result"])


async def auto_start_eval(
    agent: Agent,
    coding_agent_url: str,
    shard_index: int = 0,
    num_shards: int = 1,
    num_instances: int | None = None,
):
    """Background task: run evaluation from env config and publish to eval_state.

    Triggered on startup when SHARD_INDEX/NUM_SHARDS env vars are set.
    The /results endpoint serves the state so the CI runner can poll it.
    """
    eval_state["status"] = "running"
    logger.info(
        f"Auto-start evaluation: shard={shard_index}/{num_shards}, "
        f"num_instances={num_instances}, coding_agent={coding_agent_url}"
    )

    try:
        config: dict[str, Any] = {
            "shard_index": shard_index,
            "num_shards": num_shards,
        }
        if num_instances is not None:
            config["num_instances"] = num_instances

        async def on_progress(msg: str):
            logger.info(f"[eval] {msg}")

        result = await agent.run_batch(config, coding_agent_url, on_progress=on_progress)
        eval_state["result"] = result
        eval_state["status"] = "completed"
        logger.info(
            f"Evaluation complete: {result['passed']}/{result['total']} passed "
            f"({result['accuracy']:.1%})"
        )
    except Exception as e:
        logger.exception("Auto-start evaluation failed")
        eval_state["error"] = str(e)
        eval_state["status"] = "failed"


def main():
    parser = argparse.ArgumentParser(description="Run the SWE-bench Pro green agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory containing instances.jsonl and run_scripts/",
    )
    parser.add_argument(
        "--dockerhub-username",
        type=str,
        default=os.environ.get("DOCKERHUB_USERNAME", "jefzda"),
        help="Docker Hub username for SWE-bench Pro images",
    )
    args = parser.parse_args()

    # ── Read auto-start config from env (set by Amber via config_schema) ──
    coding_agent_url = os.environ.get("CODING_AGENT_URL")
    shard_index_raw = os.environ.get("SHARD_INDEX")
    num_shards_raw = os.environ.get("NUM_SHARDS")
    num_instances_raw = os.environ.get("NUM_INSTANCES")

    auto_start = shard_index_raw is not None and num_shards_raw is not None

    if auto_start:
        shard_index = int(shard_index_raw)
        num_shards = int(num_shards_raw)
        num_instances = int(num_instances_raw) if num_instances_raw else None
        logger.info(f"Auto-start mode: shard={shard_index}/{num_shards}, num_instances={num_instances}")
    else:
        logger.info("A2A-only mode: no SHARD_INDEX/NUM_SHARDS set, waiting for A2A messages")

    # ── Build A2A server ──
    skill = AgentSkill(
        id="swe-bench-pro-eval",
        name="SWE-bench Pro Evaluation",
        description=(
            "Evaluates a coding agent on SWE-bench Pro tasks. "
            "Sends the participant a real-world software engineering problem, "
            "then verifies the returned patch against the project's test suite."
        ),
        tags=["swe-bench", "evaluation", "coding", "software-engineering"],
        examples=[
            "Evaluate a coding agent on SWE-bench Pro",
            "Run SWE-bench evaluation for a participant agent",
        ],
    )

    agent_card = AgentCard(
        name="SWE-bench Pro Green Agent",
        description=(
            "An A2A green agent that evaluates coding agents on the SWE-bench Pro benchmark. "
            "It sends real-world software engineering problems to a participant agent, "
            "collects patches, and verifies them against project test suites in Docker containers."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(
            data_dir=args.data_dir,
            dockerhub_username=args.dockerhub_username,
            coding_agent_url=coding_agent_url,
        ),
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = a2a_app.build()

    # ── Mount results endpoint ──
    # Served at both "/" and "/results" so the amber proxy export works with
    # quick-submit-runner (polls "/") and direct access (polls "/results").
    app.routes.insert(0, Route("/results", results_handler, methods=["GET"]))
    app.routes.insert(0, Route("/results/", results_handler, methods=["GET"]))
    app.routes.insert(0, Route("/", results_handler, methods=["GET"]))

    # ── Register auto-start as a startup event ──
    if auto_start:
        agent = Agent(
            data_dir=args.data_dir,
            dockerhub_username=args.dockerhub_username,
            coding_agent_url=coding_agent_url,
        )

        @app.on_event("startup")
        async def _start_eval():
            # Small delay to let the coding agent container start
            await asyncio.sleep(5)
            asyncio.create_task(
                auto_start_eval(agent, coding_agent_url, shard_index, num_shards, num_instances)
            )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
