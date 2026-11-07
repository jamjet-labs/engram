"""Origin checks on the HTTP app (MCP Streamable HTTP transport security).

The MCP spec says servers must validate Origin to stop DNS rebinding: a web
page the user visits can point its own hostname at 127.0.0.1 and then talk to
a local server as same-origin. Browsers always send Origin on such requests;
MCP and REST clients outside a browser do not. So: no Origin passes, an Origin
on the allowlist passes, anything else gets 403.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from engram.engram import Engram
from engram.server.http import build_http_app
from engram.server.mcp import build_mcp_server
from tests.integration.test_http_mcp import _start_lifespan

TOKEN = "secret123"
ALLOWED = "https://app.example.com"
INIT: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}
MCP_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


async def _engram(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Engram:
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    return await Engram.open(path=str(tmp_path / "engram.db"))


@pytest_asyncio.fixture
async def app(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    engram = await _engram(tmp_path, monkeypatch)
    task: asyncio.Task[None] | None = None
    try:
        asgi_app = build_http_app(
            engram,
            mcp_server=build_mcp_server(engram, name="t"),
            auth_token=TOKEN,
            allowed_origins=[ALLOWED],
        )
        task = await _start_lifespan(asgi_app)
        yield asgi_app
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await engram.close()


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        # DNS rebinding: the attacker's hostname now resolves to 127.0.0.1, so
        # Host and Origin agree with each other. Only the allowlist catches it.
        "http://rebind.attacker.example:8765",
        "null",
        "https://app.example.com.evil.example",
        "http://app.example.com",
    ],
)
async def test_mcp_rejects_foreign_origin_even_with_token(app: FastAPI, origin: str) -> None:
    async with _client(app) as c:
        resp = await c.post("/mcp", headers={**MCP_HEADERS, "Origin": origin}, json=INIT)
        assert resp.status_code == 403, f"{origin} -> {resp.status_code}"


@pytest.mark.asyncio
async def test_mcp_allows_listed_origin(app: FastAPI) -> None:
    async with _client(app) as c:
        resp = await c.post("/mcp", headers={**MCP_HEADERS, "Origin": ALLOWED}, json=INIT)
        assert resp.status_code == 200
        # Scheme and host are case-insensitive.
        resp = await c.post(
            "/mcp", headers={**MCP_HEADERS, "Origin": "HTTPS://App.Example.COM"}, json=INIT
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_mcp_allows_no_origin(app: FastAPI) -> None:
    async with _client(app) as c:
        resp = await c.post("/mcp", headers=MCP_HEADERS, json=INIT)
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_allowed_origin_still_needs_the_token(app: FastAPI) -> None:
    async with _client(app) as c:
        resp = await c.post("/mcp", headers={"Origin": ALLOWED}, json=INIT)
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_rest_rejects_foreign_origin(app: FastAPI) -> None:
    async with _client(app) as c:
        resp = await c.get(
            "/v1/memory/recall",
            params={"query": "q"},
            headers={"Authorization": f"Bearer {TOKEN}", "Origin": "http://evil.example"},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_app_still_checks_origin(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The --no-auth loopback case is the one DNS rebinding actually reaches.
    engram = await _engram(tmp_path, monkeypatch)
    try:
        app = build_http_app(engram)
        async with _client(app) as c:
            resp = await c.post(
                "/v1/memory/record",
                headers={"Origin": "http://rebind.attacker.example:8765"},
                json={"text": "planted", "user_id": "alice"},
            )
            assert resp.status_code == 403
            resp = await c.post("/v1/memory/record", json={"text": "ok", "user_id": "alice"})
            assert resp.status_code == 200
    finally:
        await engram.close()


@pytest.mark.asyncio
async def test_healthz_ignores_origin(app: FastAPI) -> None:
    async with _client(app) as c:
        resp = await c.get("/healthz", headers={"Origin": "http://evil.example"})
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_invalid_allowlist_entry_is_refused(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engram = await _engram(tmp_path, monkeypatch)
    try:
        for bad in ("*", "null", "app.example.com", "https://app.example.com/path", "ftp://x"):
            with pytest.raises(ValueError):
                build_http_app(engram, allowed_origins=[bad])
    finally:
        await engram.close()
