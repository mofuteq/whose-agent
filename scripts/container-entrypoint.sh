#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec /app/.venv/bin/whose-agent serve \
  --host "${WHOSE_AGENT_HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --scenarios "${WHOSE_AGENT_SCENARIOS_DIR:-/app/scenarios}" \
  --outputs "${WHOSE_AGENT_OUTPUTS_DIR:-/data/outputs}" \
  --env-file "${WHOSE_AGENT_ENV_FILE:-/app/.env}"
