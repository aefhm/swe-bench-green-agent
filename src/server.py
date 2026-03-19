import argparse
import logging
import os
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


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

    coding_agent_url = os.environ.get("CODING_AGENT_URL")
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(
            data_dir=args.data_dir,
            dockerhub_username=args.dockerhub_username,
            coding_agent_url=coding_agent_url,
        ),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
