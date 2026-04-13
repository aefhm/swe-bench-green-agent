# SWE-bench Pro Green Agent

The evaluation orchestrator for [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro). This agent is fixed infrastructure — participants do not modify it.

## What it does

1. Receives a batch of SWE-bench instances from the gateway
2. Sends each instance (problem statement + Docker image) to the coding agent via A2A
3. Collects the returned patch
4. Applies the patch and runs the project's test suite in a Docker container
5. Reports pass/fail results back to the gateway

## Quick start

```bash
docker build -t swe-bench-green-agent .

docker run -d -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  swe-bench-green-agent --host 0.0.0.0 --port 9009

curl http://localhost:9009/.well-known/agent-card.json
```

The Docker socket mount is required — the green agent runs sibling containers to evaluate patches.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DOCKERHUB_USERNAME` | `jefzda` | Docker Hub account hosting SWE-bench eval images |
| `DATA_DIR` | `data` | Path to instance data directory |

## Tests

```bash
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```
