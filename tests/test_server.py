"""Unit tests for server.py — auto_start_eval state machine, results_handler.

These tests mock the agent so they can run without Docker, a coding agent,
or any network access.
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


# ── auto_start_eval state machine ────────────────────────────────


class TestAutoStartEval:
    """Test the eval_state machine used by /results and auto_start_eval."""

    @pytest.fixture(autouse=True)
    def reset_eval_state(self):
        """Reset eval_state before each test."""
        from server import eval_state
        eval_state["status"] = "idle"
        eval_state["result"] = None
        eval_state["error"] = None

    @pytest.mark.asyncio
    async def test_success(self, agent):
        """auto_start_eval should update eval_state on success."""
        from server import auto_start_eval, eval_state

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

            await auto_start_eval(
                agent=agent,
                coding_agent_url="http://fake-agent:9009",
                shard_index=0,
                num_shards=1,
                num_instances=1,
            )

        assert eval_state["status"] == "completed"
        assert eval_state["result"]["passed"] == 1
        assert eval_state["error"] is None

    @pytest.mark.asyncio
    async def test_failure(self, agent):
        """auto_start_eval should set failed status on error."""
        from server import auto_start_eval, eval_state

        # num_instances=0 → empty instance list → ValueError
        await auto_start_eval(
            agent=agent,
            coding_agent_url="http://fake-agent:9009",
            shard_index=0,
            num_shards=1,
            num_instances=0,
        )

        assert eval_state["status"] == "failed"
        assert eval_state["error"] is not None

    @pytest.mark.asyncio
    async def test_shard_config_passed_to_run_batch(self, agent):
        """auto_start_eval should pass shard config to run_batch."""
        from server import auto_start_eval, eval_state

        mock_eval_result = EvalResult(
            instance_id="test__repo-abc123",
            passed=True,
            fail_to_pass_ok=True,
            pass_to_pass_ok=True,
        )

        captured_config = {}

        async def capture_run_batch(config, participant_url, on_progress=None):
            captured_config.update(config)
            return await original_run_batch(config, participant_url, on_progress=on_progress)

        with patch.object(agent.messenger, "talk_to_agent", new_callable=AsyncMock) as mock_talk, \
             patch("agent.evaluate_patch", return_value=mock_eval_result), \
             patch("agent.get_dockerhub_image_uri", return_value="testuser/sweap-images:test"), \
             patch.object(agent, "_cleanup_eval_image"):

            mock_talk.return_value = '{"patch": "diff --git a/foo b/foo"}'
            original_run_batch = agent.run_batch

            with patch.object(agent, "run_batch", side_effect=capture_run_batch):
                await auto_start_eval(
                    agent=agent,
                    coding_agent_url="http://fake-agent:9009",
                    shard_index=0,
                    num_shards=1,
                )

        assert eval_state["status"] == "completed"
        assert captured_config["shard_index"] == 0
        assert captured_config["num_shards"] == 1

    @pytest.mark.asyncio
    async def test_num_instances_passed_when_set(self, agent):
        """auto_start_eval should include num_instances in config when provided."""
        from server import auto_start_eval, eval_state

        captured_config = {}

        async def capture_run_batch(config, participant_url, on_progress=None):
            captured_config.update(config)
            return {"status": "completed", "accuracy": 1.0, "passed": 1, "total": 1, "results": []}

        with patch.object(agent, "run_batch", side_effect=capture_run_batch):
            await auto_start_eval(
                agent=agent,
                coding_agent_url="http://fake-agent:9009",
                shard_index=0,
                num_shards=4,
                num_instances=10,
            )

        assert captured_config["num_instances"] == 10
        assert captured_config["shard_index"] == 0
        assert captured_config["num_shards"] == 4

    @pytest.mark.asyncio
    async def test_num_instances_omitted_when_none(self, agent):
        """auto_start_eval should not include num_instances when None."""
        from server import auto_start_eval, eval_state

        captured_config = {}

        async def capture_run_batch(config, participant_url, on_progress=None):
            captured_config.update(config)
            return {"status": "completed", "accuracy": 1.0, "passed": 1, "total": 1, "results": []}

        with patch.object(agent, "run_batch", side_effect=capture_run_batch):
            await auto_start_eval(
                agent=agent,
                coding_agent_url="http://fake-agent:9009",
                shard_index=0,
                num_shards=1,
            )

        assert "num_instances" not in captured_config

    @pytest.mark.asyncio
    async def test_sets_running_during_eval(self, agent):
        """eval_state should be 'running' while evaluation is in progress."""
        from server import auto_start_eval, eval_state

        observed_status = None

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

            original_run_batch = agent.run_batch

            async def instrumented_run_batch(*args, **kwargs):
                nonlocal observed_status
                observed_status = eval_state["status"]
                return await original_run_batch(*args, **kwargs)

            with patch.object(agent, "run_batch", side_effect=instrumented_run_batch):
                await auto_start_eval(
                    agent=agent,
                    coding_agent_url="http://fake-agent:9009",
                    shard_index=0,
                    num_shards=1,
                    num_instances=1,
                )

        assert observed_status == "running"
        assert eval_state["status"] == "completed"


# ── results_handler ───────────────────────────────────────────────


class TestResultsHandler:
    """Test results_handler directly (no HTTP server needed)."""

    @pytest.mark.asyncio
    async def test_idle_returns_running(self):
        """Idle state should report as running (eval hasn't started yet)."""
        from server import results_handler, eval_state

        eval_state["status"] = "idle"
        request = MagicMock()
        response = await results_handler(request)
        data = json.loads(response.body.decode())
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_running_returns_running(self):
        from server import results_handler, eval_state

        eval_state["status"] = "running"
        request = MagicMock()
        response = await results_handler(request)
        data = json.loads(response.body.decode())
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_completed_returns_result(self):
        from server import results_handler, eval_state

        eval_state["status"] = "completed"
        eval_state["result"] = {
            "status": "completed",
            "accuracy": 1.0,
            "passed": 1,
            "total": 1,
            "results": [],
        }
        request = MagicMock()
        response = await results_handler(request)
        data = json.loads(response.body.decode())
        assert data["status"] == "completed"
        assert data["accuracy"] == 1.0
        assert data["passed"] == 1

    @pytest.mark.asyncio
    async def test_failed_returns_error(self):
        from server import results_handler, eval_state

        eval_state["status"] = "failed"
        eval_state["error"] = "Something went wrong"
        request = MagicMock()
        response = await results_handler(request)
        data = json.loads(response.body.decode())
        assert data["status"] == "failed"
        assert data["error"] == "Something went wrong"
