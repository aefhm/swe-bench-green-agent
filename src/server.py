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
    instance_ids: list[str],
    batch_index: str | None = None,
    total_batches: str | None = None,
):
    """Background task: run evaluation from env config and publish to eval_state.

    Triggered on startup when INSTANCE_IDS or BATCH_INDEX/TOTAL_BATCHES env vars
    are set. The /results endpoint serves the state so the CI runner can poll it.
    """
    eval_state["status"] = "running"
    logger.info(
        f"Auto-start evaluation: instance_ids={len(instance_ids) if instance_ids else 'none'}, "
        f"batch={batch_index}/{total_batches}, coding_agent={coding_agent_url}"
    )

    try:
        config: dict[str, Any] = {}
        if instance_ids:
            config["instances"] = instance_ids
        if batch_index is not None and total_batches is not None:
            config["batch_index"] = batch_index
            config["total_batches"] = total_batches

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


def parse_instance_ids(raw: str) -> list[str]:
    """Parse comma-separated instance IDs from env var."""
    if not raw or not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


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

    # ── Read auto-start config from env (set by Amber via AMBER_CONFIG_GREEN__*) ──
    coding_agent_url = os.environ.get("CODING_AGENT_URL")
    instance_ids_raw = os.environ.get("INSTANCE_IDS", "")
    batch_index = os.environ.get("BATCH_INDEX")
    total_batches = os.environ.get("TOTAL_BATCHES")
    auto_start = bool(instance_ids_raw.strip()) or (batch_index is not None and total_batches is not None)

    if auto_start:
        logger.info(f"Auto-start mode: INSTANCE_IDS={instance_ids_raw[:80] if instance_ids_raw.strip() else 'none'}, BATCH={batch_index}/{total_batches}")
    else:
        logger.info("A2A-only mode: no INSTANCE_IDS or BATCH_INDEX set, waiting for A2A messages")

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
        instance_ids = parse_instance_ids(instance_ids_raw)

        @app.on_event("startup")
        async def _start_eval():
            # Small delay to let the coding agent container start
            await asyncio.sleep(5)
            asyncio.create_task(
                auto_start_eval(agent, coding_agent_url, instance_ids, batch_index, total_batches)
            )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
