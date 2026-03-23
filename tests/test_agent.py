"""Unit tests for agent.py — _select_instances, run_batch, _extract_patch, _cleanup_eval_image.

These tests mock the messenger and evaluator so they can run without
Docker, a coding agent, or any network access.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Add src to path so we can import directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Agent
from evaluator import EvalResult


@pytest.fixture
def data_dir(tmp_path):
    """Create a temp data dir with a minimal instances.jsonl."""
    instances = [
        {
            "instance_id": "test__repo-abc123",
            "short_id": "test-001",
            "repo": "test/repo",
            "problem_statement": "Fix the bug",
            "base_commit": "abc123",
            "hints_text": "",
        },
        {
            "instance_id": "test__repo-def456",
            "short_id": "test-002",
            "repo": "test/repo",
            "problem_statement": "Add the feature",
            "base_commit": "def456",
            "hints_text": "",
        },
        {
            "instance_id": "test__repo-ghi789",
            "short_id": "test-003",
            "repo": "test/repo",
            "problem_statement": "Refactor the module",
            "base_commit": "ghi789",
            "hints_text": "",
        },
    ]
    instances_path = tmp_path / "instances.jsonl"
    with open(instances_path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    return str(tmp_path)


@pytest.fixture
def agent(data_dir):
    return Agent(
        data_dir=data_dir,
        dockerhub_username="testuser",
        coding_agent_url="http://fake-agent:9009",
    )


# ── _select_instances ─────────────────────────────────────────────


class TestSelectInstances:
    def test_select_all(self, agent):
        result = agent._select_instances({})
        assert len(result) == 3

    def test_select_by_short_id(self, agent):
        result = agent._select_instances({"instances": ["test-001", "test-003"]})
        assert len(result) == 2
        assert result[0]["short_id"] == "test-001"
        assert result[1]["short_id"] == "test-003"

    def test_select_by_instance_id(self, agent):
        result = agent._select_instances({"instance_ids": ["test__repo-def456"]})
        assert len(result) == 1

    def test_no_match(self, agent):
        result = agent._select_instances({"instances": ["nonexistent"]})
        assert len(result) == 0


# ── run_batch ─────────────────────────────────────────────────────


class TestRunBatch:
    @pytest.mark.asyncio
    async def test_run_batch_success(self, agent):
        """run_batch should return structured results."""
        mock_eval_result = EvalResult(
            instance_id="test__repo-abc123",
            passed=True,
            fail_to_pass_ok=True,
            pass_to_pass_ok=True,
        )

        with patch.object(agent.messenger, "talk_to_agent", new_callable=AsyncMock) as mock_talk, \
             patch("agent.evaluate_patch", return_value=mock_eval_result), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"), \
             patch.object(agent, "_cleanup_eval_image"):

            mock_talk.return_value = '{"patch": "diff --git a/foo b/foo"}'

            result = await agent.run_batch(
                config={"instances": ["test-001"]},
                participant_url="http://fake-agent:9009",
            )

        assert result["status"] == "completed"
        assert result["total"] == 1
        assert result["passed"] == 1
        assert result["accuracy"] == 1.0
        assert len(result["results"]) == 1
        assert result["results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_run_batch_participant_error(self, agent):
        """run_batch should handle participant communication errors gracefully."""
        with patch.object(agent.messenger, "talk_to_agent", new_callable=AsyncMock) as mock_talk, \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"):

            mock_talk.side_effect = RuntimeError("Connection refused")

            result = await agent.run_batch(
                config={"instances": ["test-001"]},
                participant_url="http://fake-agent:9009",
            )

        assert result["status"] == "completed"
        assert result["total"] == 1
        assert result["passed"] == 0
        assert "communication error" in result["results"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_run_batch_empty_patch(self, agent):
        """run_batch should handle empty patches gracefully."""
        with patch.object(agent.messenger, "talk_to_agent", new_callable=AsyncMock) as mock_talk, \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"):

            mock_talk.return_value = ""

            result = await agent.run_batch(
                config={"instances": ["test-001"]},
                participant_url="http://fake-agent:9009",
            )

        assert result["total"] == 1
        assert result["passed"] == 0
        assert "empty" in result["results"][0]["error"].lower()

    @pytest.mark.asyncio
    async def test_run_batch_no_instances(self, agent):
        """run_batch should raise ValueError when no instances match."""
        with pytest.raises(ValueError, match="No matching instances"):
            await agent.run_batch(
                config={"instances": ["nonexistent"]},
                participant_url="http://fake-agent:9009",
            )

    @pytest.mark.asyncio
    async def test_run_batch_progress_callback(self, agent):
        """run_batch should call the progress callback."""
        progress_messages = []

        async def on_progress(msg):
            progress_messages.append(msg)

        mock_eval_result = EvalResult(
            instance_id="test__repo-abc123",
            passed=True,
            fail_to_pass_ok=True,
            pass_to_pass_ok=True,
        )

        with patch.object(agent.messenger, "talk_to_agent", new_callable=AsyncMock) as mock_talk, \
             patch("agent.evaluate_patch", return_value=mock_eval_result), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"), \
             patch.object(agent, "_cleanup_eval_image"):

            mock_talk.return_value = '{"patch": "diff --git a/foo b/foo"}'

            await agent.run_batch(
                config={"instances": ["test-001"]},
                participant_url="http://fake-agent:9009",
                on_progress=on_progress,
            )

        assert len(progress_messages) >= 2  # at least "Starting..." and one instance update
        assert any("Starting" in m for m in progress_messages)

    @pytest.mark.asyncio
    async def test_run_batch_multiple_instances(self, agent):
        """run_batch should process multiple instances."""
        call_count = 0

        async def mock_talk(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return '{"patch": "diff --git a/foo b/foo"}'

        mock_eval_result = EvalResult(
            instance_id="test",
            passed=True,
            fail_to_pass_ok=True,
            pass_to_pass_ok=True,
        )

        with patch.object(agent.messenger, "talk_to_agent", side_effect=mock_talk), \
             patch("agent.evaluate_patch", return_value=mock_eval_result), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"), \
             patch.object(agent, "_cleanup_eval_image"):

            result = await agent.run_batch(
                config={"instances": ["test-001", "test-002"]},
                participant_url="http://fake-agent:9009",
            )

        assert result["total"] == 2
        assert call_count == 2


# ── _extract_patch ────────────────────────────────────────────────


class TestExtractPatch:
    def test_json_with_patch_key(self, agent):
        response = '{"patch": "diff --git a/foo b/foo\\n--- a/foo\\n+++ b/foo"}'
        assert agent._extract_patch(response).startswith("diff --git")

    def test_markdown_code_block(self, agent):
        response = "Here is the fix:\n```diff\ndiff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n```"
        assert agent._extract_patch(response).startswith("diff --git")

    def test_raw_diff(self, agent):
        response = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo"
        assert agent._extract_patch(response).startswith("diff --git")

    def test_empty_response(self, agent):
        assert agent._extract_patch("") is None
        assert agent._extract_patch("   ") is None

    def test_none_response(self, agent):
        assert agent._extract_patch(None) is None

    def test_json_embedded_in_text(self, agent):
        response = 'Status: working on it\n{"patch": "diff --git a/f b/f"}'
        result = agent._extract_patch(response)
        assert result is not None
        assert "diff" in result

    def test_plain_text_fallback(self, agent):
        """Non-diff text should be returned as last resort."""
        response = "some random text"
        assert agent._extract_patch(response) == "some random text"


# ── _cleanup_eval_image ───────────────────────────────────────────


class TestCleanupEvalImage:
    def _mock_docker(self):
        """Create a mock docker module for patching the lazy import."""
        mock_module = MagicMock()
        return mock_module

    def test_cleanup_removes_image(self, agent):
        """_cleanup_eval_image should call docker images.remove."""
        mock_docker = self._mock_docker()
        mock_client = mock_docker.from_env.return_value

        with patch.dict("sys.modules", {"docker": mock_docker}), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test.repo-abc"):

            agent._cleanup_eval_image("test__repo-abc123", "test/repo")

        mock_client.images.remove.assert_called_once_with(
            "testuser/sweap-images:test.repo-abc", force=True
        )

    def test_cleanup_handles_missing_image(self, agent):
        """_cleanup_eval_image should not raise if image doesn't exist."""
        mock_docker = self._mock_docker()
        mock_docker.from_env.return_value.images.remove.side_effect = Exception("No such image")

        with patch.dict("sys.modules", {"docker": mock_docker}), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"):
            # Should not raise
            agent._cleanup_eval_image("test__repo-abc123", "test/repo")

    def test_cleanup_handles_no_docker(self, agent):
        """_cleanup_eval_image should not raise if Docker is unavailable."""
        mock_docker = self._mock_docker()
        mock_docker.from_env.side_effect = Exception("Cannot connect to Docker")

        with patch.dict("sys.modules", {"docker": mock_docker}), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"):
            # Should not raise
            agent._cleanup_eval_image("test__repo-abc123", "test/repo")
