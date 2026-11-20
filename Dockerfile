# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

# System deps for hnswlib compilation and curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime.
RUN useradd --create-home --uid 1000 engram

WORKDIR /app

# Copy minimal install artifacts.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package itself, with the vertex extra (google-auth) so the
# vertex embedder works in the image. No credentials go into the image.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir '.[vertex]'

# Persistent volume for the SQLite DB.
RUN mkdir -p /data && chown engram:engram /data
VOLUME ["/data"]
ENV ENGRAM_DB_PATH=/data/engram.db

USER engram

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8765/healthz || exit 1

# Default to HTTP transport binding all interfaces inside the container.
# ENGRAM_AUTH_TOKEN is required at run time: without it the server exits at
# startup with a message saying so. The token gates every route except
# /healthz. To run without auth on purpose (for example behind a proxy that
# does auth), set ENGRAM_ALLOW_UNAUTHENTICATED=1. Without a token the server
# only answers Host 127.0.0.1, localhost or ::1; set ENGRAM_ALLOWED_HOSTS to the
# name the proxy forwards, or every proxied request gets 421.
# The embedder and the LLM are options too: add --embedder and --llm after the
# command (or set ENGRAM_EMBEDDER and ENGRAM_LLM). The defaults are a synthetic
# embedder and no LLM, which is for trying the server out. The vertex embedder's
# dependencies are installed. Its credentials come from the environment at run
# time: application default credentials (the platform's service account) or a
# mounted workload identity file named by GOOGLE_APPLICATION_CREDENTIALS.
CMD ["engram-server", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
