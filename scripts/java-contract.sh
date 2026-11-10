#!/usr/bin/env bash
# Run the published jamjet-sdk EngramClient against a live Python server.
set -euo pipefail
cd "$(dirname "$0")/.."
export ENGRAM_PORT="${ENGRAM_PORT:-18765}"
URL="http://127.0.0.1:${ENGRAM_PORT}"
if curl -s "${URL}/healthz" >/dev/null 2>&1; then
  echo "port ${ENGRAM_PORT} is already in use; set ENGRAM_PORT to a free port" >&2
  exit 1
fi
ENGRAM_AUTH_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ENGRAM_AUTH_TOKEN
uv run python tests/contract/serve.py &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
up=0
for _ in $(seq 1 50); do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "server exited before it became healthy" >&2
    exit 1
  fi
  if curl -sf "${URL}/healthz" >/dev/null; then up=1; break; fi
  sleep 0.2
done
if [ "$up" != 1 ]; then
  echo "server did not become healthy in time" >&2
  exit 1
fi
mvn -q -f tests/contract/java/pom.xml test \
  -Dengram.url="${URL}" \
  -Dengram.token="${ENGRAM_AUTH_TOKEN}"
