#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker CLI not found. Install Docker Desktop or Docker Engine, then rerun ./scripts/start.sh." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "error: docker compose is not available. Install the Docker Compose plugin, then rerun ./scripts/start.sh." >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p outputs

if [ -z "${WHOSE_AGENT_UID:-}" ] && command -v id >/dev/null 2>&1; then
  WHOSE_AGENT_UID=$(id -u)
  export WHOSE_AGENT_UID
fi

if [ -z "${WHOSE_AGENT_GID:-}" ] && command -v id >/dev/null 2>&1; then
  WHOSE_AGENT_GID=$(id -g)
  export WHOSE_AGENT_GID
fi

exec docker compose up --build "$@"
