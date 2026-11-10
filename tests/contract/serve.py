"""Serve Engram over HTTP for the Java contract test.

A scripted LLM stands in for extraction, so the test needs no API key. Each
extraction call returns one fact that names its call number.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os

import uvicorn

from engram import Engram
from engram.llm.base import LLMMessage, LLMResponse
from engram.server.http import build_http_app


class CountingLLM:
    def __init__(self) -> None:
        self._n = itertools.count(1)

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        n = next(self._n)
        fact = {
            "text": f"contract fact {n}: the user likes pizza",
            "category": "user_preference",
            "polarity": "affirmative",
            "confidence": 0.9,
            "entities": [],
            "event_date": None,
        }
        return LLMResponse(content=json.dumps({"facts": [fact]}), finish_reason="stop")


async def main() -> None:
    token = os.environ["ENGRAM_AUTH_TOKEN"]
    port = int(os.environ.get("ENGRAM_PORT", "18765"))
    memory = await Engram.open(":memory:", llm=CountingLLM())
    app = build_http_app(memory, auth_token=token)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    try:
        await server.serve()
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
