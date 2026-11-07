"""With a token configured, every HTTP route needs it except /healthz.

Before 0.3.0 only /mcp was gated, so anyone who could reach the port could read
and write any user's memory through /v1/memory/* while the operator believed
the server was protected.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from engram.engram import Engram
from engram.server.http import build_http_app
from engram.server.mcp import build_mcp_server

TOKEN = "secret123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# (method, path, request kwargs) for every non-health route the app serves.
ROUTES: list[tuple[str, str, dict[str, Any]]] = [
    ("POST", "/v1/memory/record", {"json": {"text": "t", "user_id": "alice"}}),
    ("POST", "/v1/memory/search", {"json": {"query": "q", "user_id": "alice"}}),
    (
        "POST",
        "/v1/memory/recall_with_provenance",
        {"json": {"query": "q", "user_id": "alice"}},
    ),
    ("GET", "/v1/memory/recall", {"params": {"query": "q", "user_id": "alice"}}),
    ("GET", "/v1/memory/raw_facts", {"params": {"query": "q", "user_id": "alice"}}),
    ("GET", "/v1/memory/context", {"params": {"query": "q", "user_id": "alice"}}),
    ("POST", "/v1/memory/extract", {"json": {"messages": [], "user_id": "alice"}}),
    (
        "POST",
        "/v1/sessions/s1/messages",
        {"json": {"content": "hi", "role": "user", "session_id": "s1"}},
    ),
    ("POST", "/mcp", {"json": {"jsonrpc": "2.0", "id": 1, "method": "initialize"}}),
    ("GET", "/openapi.json", {}),
    ("GET", "/docs", {}),
    ("GET", "/no/such/route", {}),
]


@pytest_asyncio.fixture(params=["with_mcp", "rest_only"])
async def app(
    request: pytest.FixtureRequest, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[FastAPI]:
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    engram = await Engram.open(path=str(tmp_path / "engram.db"))
    try:
        mcp_server = build_mcp_server(engram, name="t") if request.param == "with_mcp" else None
        yield build_http_app(engram, mcp_server=mcp_server, auth_token=TOKEN)
    finally:
        await engram.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "kwargs"), ROUTES)
async def test_every_route_needs_the_token(
    app: FastAPI, method: str, path: str, kwargs: dict[str, Any]
) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.request(method, path, **kwargs)
        assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}"
        assert resp.json() == {"error": "invalid bearer token"}

        resp = await c.request(method, path, headers={"Authorization": "Bearer nope"}, **kwargs)
        assert resp.status_code == 401, f"{method} {path} wrong token -> {resp.status_code}"


@pytest.mark.asyncio
async def test_rest_works_with_the_token(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/memory/record", headers=AUTH, json={"text": "test fact", "user_id": "alice"}
        )
        assert resp.status_code == 200
        assert resp.json()["text"] == "test fact"

        resp = await c.get(
            "/v1/memory/recall", headers=AUTH, params={"query": "test", "user_id": "alice"}
        )
        assert resp.status_code == 200
        assert "hits" in resp.json()


@pytest.mark.asyncio
async def test_healthz_stays_open(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_healthz_exemption_is_exact(app: FastAPI) -> None:
    # Only the exact path is open; lookalikes fall through to the token check.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for path in ("/healthz/", "/healthzz", "/healthz.json"):
            resp = await c.get(path)
            assert resp.status_code == 401, f"{path} -> {resp.status_code}"


@pytest.mark.asyncio
async def test_empty_token_is_refused(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty string must not quietly mean "no auth".
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    engram = await Engram.open(path=str(tmp_path / "engram.db"))
    try:
        with pytest.raises(ValueError, match="auth_token"):
            build_http_app(engram, auth_token="")
    finally:
        await engram.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    [" secret123", "secret123 ", "   ", "secret123\n", "sec ret123", "sec\nret123", "s\u00e9cret"],
)
async def test_token_a_client_cannot_send_is_refused(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    # Whitespace is stripped from header values, and control or non-ASCII
    # characters are not sent by common clients, so such a token never matches.
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    engram = await Engram.open(path=str(tmp_path / "engram.db"))
    try:
        with pytest.raises(ValueError, match="printable ASCII"):
            build_http_app(engram, auth_token=token)
    finally:
        await engram.close()
