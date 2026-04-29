"""Run Engram as an HTTP server.

Run: python examples/03_http_server.py
Then: curl http://127.0.0.1:19090/healthz

Wire-protocol parity with Engram Rust v0.5.x — Spring Boot starter, langchain4j,
and Java SDK clients all work against this server unchanged.
"""

import asyncio

import uvicorn

from engram import Engram
from engram.server.http import build_http_app


async def _build_app() -> uvicorn.Server:
    memory = await Engram.open("./engram.db")
    app = build_http_app(memory)
    config = uvicorn.Config(app, host="127.0.0.1", port=19090, log_level="info")
    return uvicorn.Server(config)


def main() -> None:
    server = asyncio.run(_build_app())
    server.run()


if __name__ == "__main__":
    main()
