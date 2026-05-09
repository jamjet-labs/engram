"""Integration tests for /mcp Streamable HTTP endpoint.

Drives the MCP handshake via httpx ASGITransport — no real socket binding.
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
async def test_mcp_route_requires_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No auth header → 401
        resp = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert resp.status_code == 401

        # Wrong token → 401
        resp = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_healthz_unauthed(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
