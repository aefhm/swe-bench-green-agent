# SWE-bench Pro Green Agent

An A2A green agent that evaluates coding agents on the [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) benchmark.

It sends real-world software engineering problems to a participant agent, collects patches, and verifies them against project test suites in Docker containers.

## Quick start

```bash
# Build
docker build -t swe-bench-green-agent .

# Run
docker run -d -p 9009:9009 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  swe-bench-green-agent --host 0.0.0.0 --port 9009

# Test
curl http://localhost:9009/.well-known/agent-card.json
```

## Tests

```bash
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```
