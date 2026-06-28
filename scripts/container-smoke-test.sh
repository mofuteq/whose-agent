#!/bin/sh
set -eu

WHOSE_AGENT_PORT="${WHOSE_AGENT_PORT:-18000}"
export WHOSE_AGENT_PORT

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    docker compose logs --no-color || true
  fi
  docker compose down --remove-orphans || true
  exit "$status"
}

trap cleanup EXIT INT TERM

./scripts/start.sh -d

python3 - <<'PY'
import json
import os
import time
import urllib.request

port = os.environ["WHOSE_AGENT_PORT"]
base_url = f"http://127.0.0.1:{port}"


def get(path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as response:
        return response.status, response.read().decode("utf-8")


deadline = time.time() + 120
last_error: Exception | None = None
while time.time() < deadline:
    try:
        status, body = get("/health")
        if status == 200 and json.loads(body).get("status") == "ok":
            break
    except Exception as exc:
        last_error = exc
        time.sleep(2)
else:
    raise SystemExit(f"container did not become healthy: {last_error}")

status, root = get("/")
if status != 200 or "<div id=\"root\"" not in root or "whose-agent observation workspace" not in root:
    raise SystemExit("root did not return the built frontend HTML")

status, health = get("/health")
if status != 200 or json.loads(health).get("status") != "ok":
    raise SystemExit("/health did not succeed")

status, scenarios = get("/api/scenarios")
if status != 200 or not isinstance(json.loads(scenarios).get("scenarios"), list):
    raise SystemExit("/api/scenarios did not succeed")
PY

if docker compose exec -T whose-agent node --version; then
  echo "node is present in the runtime container" >&2
  exit 1
fi

if docker compose exec -T whose-agent npm --version; then
  echo "npm is present in the runtime container" >&2
  exit 1
fi

docker compose exec -T whose-agent test -x /app/.venv/bin/whose-agent
