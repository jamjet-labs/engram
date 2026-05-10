"""Regression guard: /v1/memory/* must remain unauthed even when /mcp has auth.

This is the asymmetric-auth design choice from the v0.2 spec. If someone later
adds auth to /v1, they must update this test deliberately.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from engram.engram import Engram
from engram.server.http import build_http_app
from engram.server.mcp import build_mcp_server


@pytest_asyncio.fixture
async def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    engram = await Engram.open(path=str(tmp_path / "engram.db"))
    try:
        mcp_server = build_mcp_server(engram, name="engram-test")
        yield build_http_app(engram, mcp_server=mcp_server, auth_token="secret123")
    finally:
        await engram.close()


@pytest.mark.asyncio
async def test_v1_record_works_without_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/memory/record",
            json={"text": "test fact", "user_id": "alice"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "test fact"
        assert "fact_id" in body


@pytest.mark.asyncio
async def test_v1_recall_get_works_without_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/memory/recall", params={"query": "anything", "user_id": "alice"}
        )
        assert resp.status_code == 200
        assert "hits" in resp.json()
