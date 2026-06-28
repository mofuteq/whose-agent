FROM ghcr.io/astral-sh/uv:0.11.14 AS uv-bin

FROM node:24.18.0-bookworm-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# debian:stable-slim intentionally tracks the current Debian stable slim image.
FROM debian:stable-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/tmp \
    UV_PYTHON_INSTALL_DIR=/app/.python \
    UV_LINK_MODE=copy

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /usr/local/bin/

WORKDIR /app
COPY .python-version pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv python install "$(cat .python-version)" \
    && uv sync --locked --no-dev \
    && rm -rf /root/.cache/uv /tmp/.cache \
    && rm -f /usr/local/bin/uv /usr/local/bin/uvx

COPY scenarios/ ./scenarios/
COPY skills/ ./skills/
COPY --from=frontend-builder /build/frontend/dist/ ./frontend/dist/
COPY scripts/container-entrypoint.sh /usr/local/bin/container-entrypoint.sh

RUN groupadd --gid 10001 whose-agent \
    && useradd --uid 10001 --gid whose-agent --home-dir /app --shell /usr/sbin/nologin whose-agent \
    && mkdir -p /data/outputs \
    && chmod 755 /usr/local/bin/container-entrypoint.sh \
    && chown -R whose-agent:whose-agent /app /data

USER whose-agent
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=2).read()"]

ENTRYPOINT ["/usr/local/bin/container-entrypoint.sh"]
