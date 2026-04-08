import json
import logging
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600  # 10 minutes — coding agents need time
TERMINAL_TASK_STATES = {"completed", "failed", "canceled", "rejected"}


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
):
    """Returns dict with context_id, response and status (if exists)"""
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        # Try to fetch the agent card; if the endpoint is behind a gateway
        # proxy that doesn't forward /.well-known/agent-card.json, build a
        # minimal card so the A2A client can still send messages.
        url = base_url.rstrip("/") + "/"
        try:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            agent_card = await resolver.get_agent_card()
            agent_card.url = url
        except Exception:
            logger.info(
                "Agent card fetch failed for %s — using minimal card (gateway proxy?)",
                base_url,
            )
            from a2a.types import AgentCard, AgentCapabilities
            agent_card = AgentCard(
                name="remote-agent",
                description="",
                url=url,
                version="0.0.0",
                capabilities=AgentCapabilities(streaming=True),
                skills=[],
                defaultInputModes=["text"],
                defaultOutputModes=["text"],
            )
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        outbound_msg = create_message(text=message, context_id=context_id)
        outputs = {"response": "", "context_id": None}
        logger.info(
            "Sending A2A message to %s (streaming=%s, context_id=%s)",
            base_url,
            streaming,
            context_id,
        )

        async for event in client.send_message(outbound_msg):
            match event:
                case Message() as msg:
                    outputs["context_id"] = msg.context_id
                    outputs["response"] = merge_parts(msg.parts)
                    outputs["status"] = "completed"
                    logger.info(
                        "Received direct message response from %s (context_id=%s, response_len=%d)",
                        base_url,
                        msg.context_id,
                        len(outputs["response"]),
                    )
                    return outputs

                case (task, _update):
                    outputs["context_id"] = task.context_id
                    outputs["status"] = task.status.state.value
                    logger.info(
                        "Received task event from %s (task_id=%s, context_id=%s, state=%s, artifacts=%d, has_status_message=%s)",
                        base_url,
                        task.id,
                        task.context_id,
                        outputs["status"],
                        len(task.artifacts or []),
                        bool(task.status.message),
                    )

                    if outputs["status"] not in TERMINAL_TASK_STATES:
                        continue

                    parts = []
                    if task.artifacts:
                        for artifact in task.artifacts:
                            parts.extend(artifact.parts)
                    elif task.status.message:
                        parts.extend(task.status.message.parts)

                    outputs["response"] = merge_parts(parts)
                    logger.info(
                        "Returning terminal task event from %s (task_id=%s, state=%s, response_len=%d)",
                        base_url,
                        task.id,
                        outputs["status"],
                        len(outputs["response"]),
                    )
                    return outputs

                case _:
                    logger.info("Received unhandled A2A event from %s: %r", base_url, event)
                    continue

        logger.warning("A2A stream to %s ended without a terminal response", base_url)
        return outputs


class Messenger:
    def __init__(self):
        self._context_ids = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Communicate with another agent by sending a message and receiving their response.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing
            timeout: Timeout in seconds for the request (default: 600)

        Returns:
            str: The agent's response message
        """
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url, None),
            timeout=timeout,
            streaming=True,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id", None)
        return outputs["response"]

    def reset(self):
        self._context_ids = {}
