"""
SWE-bench Pro Green Agent — evaluates coding agents on real-world software engineering tasks.

Flow:
1. Receives an eval request with a participant agent URL and optional instance filter
2. For each SWE-bench instance, sends the participant:
   - problem_statement (the issue to fix)
   - Docker image URI (so the participant can pull and work in the repo)
   - base_commit (the commit the repo is at)
   - repo name
3. Participant works in the container and sends back a patch (git diff)
4. Green agent evaluates the patch by running tests in a clean Docker container
5. Returns structured results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from evaluator import evaluate_patch, get_dockerhub_image_uri, EvalResult

logger = logging.getLogger(__name__)

# Type for optional progress callback: async (message_str) -> None
ProgressCallback = Callable[[str], Awaitable[None]] | None


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to this green agent."""

    participants: dict[str, HttpUrl] = {}  # role -> agent URL (optional if CODING_AGENT_URL set)
    config: dict[str, Any] = {}


class Agent:
    required_roles: list[str] = ["coding_agent"]
    required_config_keys: list[str] = []  # optional: instance_ids, max_instances

    def __init__(self, data_dir: str = "data", dockerhub_username: str = "jefzda",
                 coding_agent_url: str | None = None):
        self.messenger = Messenger()
        self.data_dir = data_dir
        self.coding_agent_url = coding_agent_url
        self.dockerhub_username = dockerhub_username
        self._instances: list[dict] | None = None

    @property
    def instances(self) -> list[dict]:
        if self._instances is None:
            instances_path = Path(self.data_dir) / "instances.jsonl"
            with open(instances_path) as f:
                self._instances = [json.loads(line) for line in f if line.strip()]
        return self._instances

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        # If CODING_AGENT_URL is set (e.g. via Amber slot), don't require it in the request
        if self.coding_agent_url and "coding_agent" in missing_roles:
            missing_roles.discard("coding_agent")
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        return True, "ok"

    def _select_instances(self, config: dict[str, Any]) -> list[dict]:
        """Select which instances to evaluate based on config.

        The 'instances' config key accepts a list of short_id or full instance_id values.
        Falls back to 'instance_ids' (full IDs only) for backwards compatibility.
        """
        instances = self.instances

        # Filter by instances (short_id or instance_id) or legacy instance_ids
        filter_ids = config.get("instances") or config.get("instance_ids")
        if filter_ids:
            target = set(filter_ids)
            instances = [
                i for i in instances
                if i.get("short_id") in target or i["instance_id"] in target
            ]

        # Limit count if specified
        max_instances = config.get("max_instances", len(instances))
        return instances[:max_instances]

    async def run_batch(
        self,
        config: dict[str, Any],
        participant_url: str,
        on_progress: ProgressCallback = None,
    ) -> dict:
        """Run evaluation batch and return structured results dict.

        This is the core evaluation loop, usable from both the A2A handler (run)
        and the auto-start /results path.

        Args:
            config: Instance selection config (instances, instance_ids, max_instances)
            participant_url: URL of the coding agent to evaluate
            on_progress: Optional async callback for status updates

        Returns:
            Dict with accuracy, passed, total, and per-instance results
        """
        instances = self._select_instances(config)
        if not instances:
            raise ValueError("No matching instances found for the given config")

        if on_progress:
            await on_progress(f"Starting evaluation of {len(instances)} instance(s)...")

        results: list[EvalResult] = []

        for i, instance in enumerate(instances):
            uid = instance["instance_id"]
            repo = instance.get("repo", "")
            image_uri = get_dockerhub_image_uri(uid, self.dockerhub_username, repo)

            if on_progress:
                await on_progress(
                    f"[{i + 1}/{len(instances)}] Sending instance {uid} to participant..."
                )

            # Build the problem message for the participant
            problem_payload = json.dumps(
                {
                    "instance_id": uid,
                    "problem_statement": instance["problem_statement"],
                    "docker_image": image_uri,
                    "base_commit": instance["base_commit"],
                    "repo": repo,
                    "hints": instance.get("hints_text", ""),
                },
                indent=2,
            )

            # Send to participant and get patch back
            try:
                response = await self.messenger.talk_to_agent(
                    message=problem_payload,
                    url=participant_url,
                    new_conversation=True,
                    timeout=1800,  # 30 min per instance
                )
            except Exception as e:
                logger.error(f"Participant communication error for {uid}: {e}")
                results.append(
                    EvalResult(
                        instance_id=uid,
                        passed=False,
                        fail_to_pass_ok=False,
                        pass_to_pass_ok=False,
                        error=f"Participant communication error: {e}",
                    )
                )
                continue

            # Extract patch from response
            patch = self._extract_patch(response)
            if not patch:
                results.append(
                    EvalResult(
                        instance_id=uid,
                        passed=False,
                        fail_to_pass_ok=False,
                        pass_to_pass_ok=False,
                        error="Participant returned empty or unparseable patch",
                    )
                )
                continue

            # Evaluate the patch
            if on_progress:
                await on_progress(
                    f"[{i + 1}/{len(instances)}] Evaluating patch for {uid}..."
                )

            result = evaluate_patch(
                instance=instance,
                patch=patch,
                data_dir=self.data_dir,
                dockerhub_username=self.dockerhub_username,
            )
            results.append(result)

            # Remove the pulled evaluation image to reclaim disk space.
            # Each instance uses a different project-specific image (~1-2 GB),
            # so without cleanup they accumulate and can fill the runner disk.
            self._cleanup_eval_image(uid, repo)

        # Compute summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        accuracy = passed / total if total > 0 else 0.0

        return {
            "status": "completed",
            "accuracy": accuracy,
            "passed": passed,
            "total": total,
            "results": [
                {
                    "instance_id": r.instance_id,
                    "passed": r.passed,
                    "fail_to_pass_ok": r.fail_to_pass_ok,
                    "pass_to_pass_ok": r.pass_to_pass_ok,
                    "error": r.error,
                }
                for r in results
            ],
        }

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """A2A message handler — parses request, delegates to run_batch."""
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        participant_url = str(
            request.participants.get("coding_agent")
            or self.coding_agent_url
            or ""
        )
        if not participant_url:
            await updater.reject(
                new_agent_text_message("No coding_agent URL in request and CODING_AGENT_URL not set")
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting evaluation..."),
        )

        async def on_progress(msg_text: str):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(msg_text),
            )

        try:
            structured_results = await self.run_batch(
                request.config, participant_url, on_progress=on_progress
            )
        except ValueError as e:
            await updater.reject(new_agent_text_message(str(e)))
            return

        # Format summary text for the A2A artifact
        summary_text = (
            f"SWE-bench Pro Evaluation Complete\n"
            f"Accuracy: {structured_results['accuracy']:.1%} "
            f"({structured_results['passed']}/{structured_results['total']})\n\n"
        )
        for r in structured_results["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            detail = f"  error: {r['error']}" if r.get("error") else ""
            summary_text += f"  [{status}] {r['instance_id']}{detail}\n"

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=structured_results)),
            ],
            name="SWE-bench Pro Evaluation Results",
        )

    def _cleanup_eval_image(self, uid: str, repo: str) -> None:
        """Remove the pulled sweap evaluation image to reclaim disk space."""
        try:
            import docker as docker_sdk

            client = docker_sdk.from_env()
            image_uri = get_dockerhub_image_uri(uid, self.dockerhub_username, repo)
            client.images.remove(image_uri, force=True)
            logger.info(f"Removed eval image: {image_uri}")
        except Exception as e:
            # Non-fatal — log and continue. The image may already have been
            # removed or may not exist (e.g. pull failed earlier).
            logger.warning(f"Failed to remove eval image for {uid}: {e}")

    def _extract_patch(self, response: str) -> str | None:
        """Extract a git diff patch from the participant's response.

        The response may contain status text concatenated with the actual
        payload (artifact text).  We try several strategies:
        1. Markdown code block containing a diff
        2. JSON with a "patch" key (exact or embedded in surrounding text)
        3. Raw diff starting with "diff " or "---"
        4. Last resort: return the whole response
        """
        import re

        if not response or not response.strip():
            return None

        # 1. Markdown code block
        code_block = re.search(r"```(?:diff)?\s*\n(.*?)```", response, re.DOTALL)
        if code_block:
            candidate = code_block.group(1).strip()
            if candidate.startswith("diff ") or candidate.startswith("---"):
                return candidate

        # 2a. Try full response as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "patch" in data:
                return data["patch"]
        except (json.JSONDecodeError, TypeError):
            pass

        # 2b. Try to find a JSON object embedded in the response
        #     (e.g. status text followed by a JSON payload)
        json_match = re.search(r'\{[^{}]*"patch"\s*:', response)
        if json_match:
            json_candidate = response[json_match.start():]
            try:
                data = json.loads(json_candidate)
                if isinstance(data, dict) and "patch" in data:
                    return data["patch"]
            except (json.JSONDecodeError, TypeError):
                pass

        # 3. Raw diff anywhere in the response
        diff_match = re.search(r'^(diff --git .+)', response, re.MULTILINE | re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()

        if response.strip().startswith("---"):
            return response.strip()

        # 4. Last resort
        return response.strip() if response.strip() else None
