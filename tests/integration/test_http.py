"""HTTP API integration tests via FastAPI's TestClient."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from engram import Engram
from engram.llm.base import LLMMessage, LLMResponse
from engram.server.http import build_http_app


@pytest_asyncio.fixture
async def client_engram() -> AsyncIterator[tuple[TestClient, Engram]]:
    memory = await Engram.open(":memory:")
    app = build_http_app(memory, allow_unauthenticated=True)
    client = TestClient(app, base_url="http://127.0.0.1")
    try:
        yield client, memory
    finally:
        await memory.close()


def test_healthz(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_record_then_search(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    r = client.post(
        "/v1/memory/record",
        json={
            "text": "alice prefers espresso",
            "user_id": "alice",
        },
    )
    assert r.status_code == 200
    fact_id = r.json()["fact_id"]
    assert fact_id

    r = client.post(
        "/v1/memory/search",
        json={
            "query": "coffee",
            "user_id": "alice",
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) >= 1
    assert any(h["fact_id"] == fact_id for h in hits)


def test_recall_get_endpoint(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    client.post(
        "/v1/memory/record",
        json={"text": "alice loves coffee", "user_id": "alice"},
    )
    r = client.get("/v1/memory/recall", params={"query": "coffee", "user_id": "alice"})
    assert r.status_code == 200
    assert len(r.json()["hits"]) >= 1


def test_raw_facts_endpoint(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    for i in range(5):
        client.post(
            "/v1/memory/record",
            json={"text": f"fact {i}", "user_id": "alice"},
        )
    r = client.get(
        "/v1/memory/raw_facts", params={"query": "fact", "user_id": "alice", "top_k": 30}
    )
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 5


def test_context_endpoint(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    client.post(
        "/v1/memory/record",
        json={"text": "alice prefers espresso", "user_id": "alice"},
    )
    r = client.get(
        "/v1/memory/context",
        params={"query": "coffee", "user_id": "alice", "token_budget": 100},
    )
    assert r.status_code == 200
    assert "context" in r.json()
    assert "espresso" in r.json()["context"].lower()


def test_extract_without_llm_returns_400(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    r = client.post(
        "/v1/memory/extract",
        json={
            "messages": [
                {
                    "content": "hello",
                    "role": "user",
                    "session_id": "s1",
                }
            ]
        },
    )
    assert r.status_code == 400
    assert "llm" in r.json()["detail"].lower()


def test_session_message_append(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    r = client.post(
        "/v1/sessions/s1/messages",
        json={"content": "hi", "role": "user", "session_id": "s1"},
    )
    assert r.status_code == 200
    assert "message_id" in r.json()


def test_search_respects_scope(client_engram: tuple[TestClient, Engram]) -> None:
    client, _ = client_engram
    client.post(
        "/v1/memory/record",
        json={"text": "alice fact", "user_id": "alice"},
    )
    r = client.post(
        "/v1/memory/search",
        json={"query": "fact", "user_id": "bob", "top_k": 5},
    )
    assert r.status_code == 200
    assert r.json()["hits"] == []


# ── input Engram refuses is a client error ─────────────────────────────


class _StubLLM:
    """Gives the same reply to every prompt."""

    def __init__(self, content: str) -> None:
        self.content = content

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(content=self.content)


@asynccontextmanager
async def _asgi_client(**open_kw: Any) -> AsyncIterator[AsyncClient]:
    memory = await Engram.open(":memory:", **open_kw)
    # A server error comes back as a 500 response, not as an exception here.
    transport = ASGITransport(
        app=build_http_app(memory, allow_unauthenticated=True), raise_app_exceptions=False
    )
    try:
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            yield client
    finally:
        await memory.close()


@pytest.mark.parametrize(
    ("body", "field"),
    [
        (b'{"text": "a\\u0000b"}', "text"),
        (json.dumps({"text": "x" * 70_000}).encode(), "text"),
        # Python's JSON parser, which FastAPI uses, accepts NaN.
        (b'{"text": "a fact", "metadata": {"x": NaN}}', "metadata"),
    ],
    ids=["nul", "over-the-byte-limit", "nan-in-metadata"],
)
async def test_record_of_refused_input_is_400(body: bytes, field: str) -> None:
    async with _asgi_client() as client:
        r = await client.post(
            "/v1/memory/record", content=body, headers={"content-type": "application/json"}
        )
    assert r.status_code == 400
    assert r.json()["detail"].startswith(field)


async def test_record_refused_by_a_model_is_400_without_the_value() -> None:
    user_id = "alice-" + "x" * 130
    async with _asgi_client() as client:
        r = await client.post("/v1/memory/record", json={"text": "a fact", "user_id": user_id})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail.startswith("user_id")
    assert "alice-" not in detail


@pytest.mark.parametrize(
    ("path", "content", "field"),
    [
        ("/v1/sessions/s1/messages", "a\x00b", "content"),
        ("/v1/sessions/%00/messages", "hello", "session_id"),
    ],
    ids=["nul-in-content", "nul-in-session-id"],
)
async def test_session_message_of_refused_input_is_400(path: str, content: str, field: str) -> None:
    async with _asgi_client() as client:
        r = await client.post(path, json={"content": content, "role": "user", "session_id": "s1"})
    assert r.status_code == 400
    assert r.json()["detail"].startswith(field)


async def test_extract_whose_fact_text_is_refused_is_400() -> None:
    reply = json.dumps({"facts": [{"text": "a\u0000b", "confidence": 0.9}]})
    async with _asgi_client(llm=_StubLLM(reply)) as client:
        r = await client.post(
            "/v1/memory/extract",
            json={"messages": [{"content": "hello", "role": "user", "session_id": "s1"}]},
        )
    assert r.status_code == 400
    assert r.json()["detail"].startswith("text")
