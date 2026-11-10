#!/usr/bin/env bash
# Run the published jamjet-sdk EngramClient against a live Python server.
set -euo pipefail
cd "$(dirname "$0")/.."
export ENGRAM_PORT="${ENGRAM_PORT:-18765}"
export ENGRAM_AUTH_TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run python tests/contract/serve.py &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:${ENGRAM_PORT}/healthz" >/dev/null && break
  sleep 0.2
done
curl -sf "http://127.0.0.1:${ENGRAM_PORT}/healthz" >/dev/null || { echo "server did not start"; exit 1; }
mvn -q -f tests/contract/java/pom.xml test \
  -Dengram.url="http://127.0.0.1:${ENGRAM_PORT}" \
  -Dengram.token="${ENGRAM_AUTH_TOKEN}"
