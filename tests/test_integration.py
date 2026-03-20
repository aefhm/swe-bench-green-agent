"""Integration tests — require a running green agent server.

Run with:
    pytest tests/test_integration.py -m integration --agent-url http://localhost:9009

These tests validate the A2A protocol compliance and /results HTTP endpoint
against a live server. They are skipped by default (no server = skip).
"""

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def agent_url(request):
    """Agent URL fixture. Skips the module if the agent isn't reachable."""
    url = request.config.getoption("--agent-url", default="http://localhost:9009")
    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.skip(f"Agent at {url} returned status {response.status_code}")
    except Exception as e:
        pytest.skip(f"Agent not reachable at {url}: {e}")
    return url


# ── A2A validation helpers ────────────────────────────────────────


def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = frozenset([
        "name", "description", "url", "version",
        "capabilities", "defaultInputModes", "defaultOutputModes", "skills",
    ])
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if "url" in card_data and not (
        card_data["url"].startswith("http://") or card_data["url"].startswith("https://")
    ):
        errors.append("Field 'url' must be an absolute URL.")

    if "capabilities" in card_data and not isinstance(card_data["capabilities"], dict):
        errors.append("Field 'capabilities' must be an object.")

    for field in ["defaultInputModes", "defaultOutputModes"]:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    if "skills" in card_data:
        if not isinstance(card_data["skills"], list):
            errors.append("Field 'skills' must be an array.")
        elif not card_data["skills"]:
            errors.append("Field 'skills' array is empty.")

    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    if "kind" not in data:
        return ["Response missing 'kind' field."]
    kind = data.get("kind")
    if kind == "task":
        errors = []
        if "id" not in data:
            errors.append("Task missing 'id'.")
        if "status" not in data or "state" not in data.get("status", {}):
            errors.append("Task missing 'status.state'.")
        return errors
    elif kind == "status-update":
        if "status" not in data or "state" not in data.get("status", {}):
            return ["StatusUpdate missing 'status.state'."]
        return []
    elif kind == "artifact-update":
        if "artifact" not in data:
            return ["ArtifactUpdate missing 'artifact'."]
        artifact = data.get("artifact", {})
        if not artifact.get("parts"):
            return ["Artifact must have non-empty 'parts'."]
        return []
    elif kind == "message":
        errors = []
        if not data.get("parts"):
            errors.append("Message must have non-empty 'parts'.")
        if data.get("role") != "agent":
            errors.append("Message from agent must have role 'agent'.")
        return errors
    return [f"Unknown kind: '{kind}'."]


async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )
        events = [event async for event in client.send_message(msg)]
    return events


# ── A2A agent card ────────────────────────────────────────────────


def test_agent_card(agent_url):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent_url}/.well-known/agent-card.json")
    assert response.status_code == 200
    card_data = response.json()
    errors = validate_agent_card(card_data)
    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

    # SWE-bench specific checks
    assert "swe-bench" in card_data["name"].lower() or "swe-bench" in card_data["description"].lower(), \
        "Agent card should mention SWE-bench"
    assert len(card_data["skills"]) >= 1, "Agent should have at least one skill"


# ── A2A message handling ─────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent_url, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent_url, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)
            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)
            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_invalid_request_rejected(agent_url, streaming):
    """Test that invalid requests are properly rejected."""
    events = await send_text_message("not valid json", agent_url, streaming=streaming)
    assert events, "Agent should respond with at least one event"

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)
            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)
            case _:
                pass
    assert not all_errors, f"Event validation failed:\n" + "\n".join(all_errors)


# ── /results HTTP endpoint ────────────────────────────────────────


def test_results_endpoint_exists(agent_url):
    """The /results endpoint should exist and return JSON."""
    response = httpx.get(f"{agent_url}/results", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_results_returns_valid_status(agent_url):
    """The /results endpoint should return a known status value."""
    response = httpx.get(f"{agent_url}/results", timeout=5)
    data = response.json()
    assert data["status"] in ("running", "completed", "failed")


def test_results_completed_has_fields(agent_url):
    """If evaluation is complete, results should have accuracy/passed/total."""
    response = httpx.get(f"{agent_url}/results", timeout=5)
    data = response.json()
    if data["status"] == "completed":
        assert "accuracy" in data
        assert "passed" in data
        assert "total" in data
        assert "results" in data
        assert isinstance(data["results"], list)


def test_results_failed_has_error(agent_url):
    """If evaluation failed, results should have an error message."""
    response = httpx.get(f"{agent_url}/results", timeout=5)
    data = response.json()
    if data["status"] == "failed":
        assert "error" in data
        assert isinstance(data["error"], str)
