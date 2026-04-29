"""HTTP API integration tests via FastAPI's TestClient."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from fastapi.testclient import TestClient

from engram import Engram
from engram.server.http import build_http_app


@pytest_asyncio.fixture
async def client_engram() -> AsyncIterator[tuple[TestClient, Engram]]:
    memory = await Engram.open(":memory:")
    app = build_http_app(memory)
    client = TestClient(app)
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
