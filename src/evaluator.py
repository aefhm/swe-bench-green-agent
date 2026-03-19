"""
SWE-bench Pro evaluation logic — runs a patch against test suite in a Docker container.

Adapted from SWE-bench_Pro-os/swe_bench_pro_eval.py, local Docker mode only.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestResult:
    name: str
    status: str  # PASSED, FAILED, SKIPPED, ERROR


@dataclass
class EvalResult:
    instance_id: str
    passed: bool
    fail_to_pass_ok: bool
    pass_to_pass_ok: bool
    test_results: list[TestResult] = field(default_factory=list)
    error: str | None = None


def get_dockerhub_image_uri(uid: str, dockerhub_username: str, repo_name: str = "") -> str:
    """Convert instance_id + repo name to Docker Hub image URI."""
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{dockerhub_username}/sweap-images:{tag}"


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _build_entryscript(instance: dict) -> str:
    """Build the bash entry script that resets the repo, applies the patch, and runs tests."""
    before_repo_set_cmd = instance["before_repo_set_cmd"].strip().split("\n")[-1]
    stf = instance["selected_test_files_to_run"]
    selected_test_files = ",".join(stf if isinstance(stf, list) else json.loads(stf))
    base_commit = instance["base_commit"]

    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_key in ("base_dockerfile", "instance_dockerfile"):
        dockerfile_content = instance.get(dockerfile_key, "")
        if not dockerfile_content:
            continue
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                env_cmds.append(line.replace("ENV", "export", 1))

    env_block = "\n".join(env_cmds)

    return f"""{env_block}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


def evaluate_patch(
    instance: dict,
    patch: str,
    data_dir: str,
    dockerhub_username: str = "jefzda",
    docker_platform: str | None = None,
    timeout: int = 3600,
) -> EvalResult:
    """
    Evaluate a patch for a SWE-bench Pro instance using local Docker.

    Args:
        instance: Instance metadata dict (from instances.jsonl)
        patch: The git diff patch to evaluate
        data_dir: Path to data/ directory (unused, kept for API compat)
        dockerhub_username: Docker Hub username for images
        docker_platform: Optional platform override (e.g. 'linux/amd64')
        timeout: Container timeout in seconds

    Returns:
        EvalResult with pass/fail and test details
    """
    import docker as docker_sdk

    uid = instance["instance_id"]
    repo = instance.get("repo", "")

    # Clean the patch
    cleaned_patch = strip_binary_hunks(patch)

    # Read scripts from instance dict (inlined in JSONL)
    run_script = instance.get("run_script", "")
    parser_script = instance.get("parsing_script", "")
    if not run_script or not parser_script:
        return EvalResult(
            instance_id=uid,
            passed=False,
            fail_to_pass_ok=False,
            pass_to_pass_ok=False,
            error=f"Missing run_script or parsing_script in instance data for {uid}",
        )

    entryscript = _build_entryscript(instance)

    # Build workspace files
    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript,
    }

    # Create workspace files and run Docker
    # NOTE: We use put_archive/get_archive instead of volume mounts because
    # in Docker-outside-of-Docker, volume paths are resolved on the HOST,
    # not inside this container.
    image_uri = get_dockerhub_image_uri(uid, dockerhub_username, repo)
    print(f"[evaluator] Using image: {image_uri}")

    client = docker_sdk.from_env()

    # Pull image
    try:
        pull_kwargs = {"platform": docker_platform} if docker_platform else {}
        client.images.pull(image_uri, **pull_kwargs)
    except Exception:
        try:
            client.images.get(image_uri)
            print(f"[evaluator] Using locally cached image: {image_uri}")
        except Exception as e:
            return EvalResult(
                instance_id=uid,
                passed=False,
                fail_to_pass_ok=False,
                pass_to_pass_ok=False,
                error=f"Failed to pull or find image: {e}",
            )

    # Create container (don't start yet — copy files in first)
    run_kwargs: dict = {
        "detach": True,
        "entrypoint": "/bin/bash",
        "command": ["-c", "bash /workspace/entryscript.sh"],
    }
    if docker_platform:
        run_kwargs["platform"] = docker_platform

    try:
        container = client.containers.create(image_uri, **run_kwargs)
    except Exception as e:
        return EvalResult(
            instance_id=uid,
            passed=False,
            fail_to_pass_ok=False,
            pass_to_pass_ok=False,
            error=f"Container creation error: {e}",
        )

    try:
        # Copy workspace files into the container via tar archive
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            for rel_path, content in files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=f"workspace/{rel_path}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        tar_buf.seek(0)
        container.put_archive("/", tar_buf)

        # Start the container and wait for it to finish
        container.start()
        result = container.wait(timeout=timeout)
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
        if status_code != 0:
            print(f"[evaluator] Entryscript exited with code {status_code} for {uid}")

        # Extract output.json from the container
        try:
            bits, _ = container.get_archive("/workspace/output.json")
            out_buf = io.BytesIO()
            for chunk in bits:
                out_buf.write(chunk)
            out_buf.seek(0)
            with tarfile.open(fileobj=out_buf, mode="r") as tar:
                member = tar.getmembers()[0]
                f_obj = tar.extractfile(member)
                if f_obj is None:
                    raise FileNotFoundError("output.json empty in archive")
                output_data = f_obj.read().decode("utf-8")
        except Exception as extract_err:
            # Also grab container logs for debugging
            try:
                logs = container.logs(tail=50).decode("utf-8", errors="replace")
                print(f"[evaluator] Container logs for {uid}:\n{logs}")
            except Exception:
                pass
            return EvalResult(
                instance_id=uid,
                passed=False,
                fail_to_pass_ok=False,
                pass_to_pass_ok=False,
                error=f"output.json not found — tests may not have run: {extract_err}",
            )

        # Parse and evaluate results (inside the try so container is still available)
        output = json.loads(output_data)

        test_results = [
            TestResult(name=t["name"], status=t["status"]) for t in output.get("tests", [])
        ]
        passed_tests = {t.name for t in test_results if t.status == "PASSED"}

        f2p_raw = instance.get("FAIL_TO_PASS") or instance["fail_to_pass"]
        p2p_raw = instance.get("PASS_TO_PASS") or instance["pass_to_pass"]
        f2p = set(f2p_raw if isinstance(f2p_raw, list) else json.loads(f2p_raw))
        p2p = set(p2p_raw if isinstance(p2p_raw, list) else json.loads(p2p_raw))

        fail_to_pass_ok = f2p <= passed_tests
        pass_to_pass_ok = p2p <= passed_tests
        passed = fail_to_pass_ok and pass_to_pass_ok

        return EvalResult(
            instance_id=uid,
            passed=passed,
            fail_to_pass_ok=fail_to_pass_ok,
            pass_to_pass_ok=pass_to_pass_ok,
            test_results=test_results,
        )

    except Exception as e:
        return EvalResult(
            instance_id=uid,
            passed=False,
            fail_to_pass_ok=False,
            pass_to_pass_ok=False,
            error=f"Container execution error: {e}",
        )
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass
