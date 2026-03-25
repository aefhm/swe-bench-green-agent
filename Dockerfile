FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install Docker CLI (for evaluating patches in sibling containers)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password agent
RUN groupadd -f docker && usermod -aG docker agent

USER agent
WORKDIR /home/agent

COPY --chown=agent pyproject.toml uv.lock README.md ./
COPY --chown=agent src src
COPY --chown=agent data data

RUN uv sync --locked

ENTRYPOINT ["uv", "run", "python", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
