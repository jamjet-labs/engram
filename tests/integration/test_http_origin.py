"""Origin checks on the HTTP app (MCP Streamable HTTP transport security).

The MCP spec says servers must validate Origin to stop DNS rebinding: a web
page the user visits can point its own hostname at 127.0.0.1 and then talk to
a local server. Browsers send Origin on cross-origin requests and on every
POST; MCP and REST clients outside a browser do not send it. So: no Origin
passes, an Origin on the allowlist passes, anything else gets 403.

A rebound page is same-origin with the server, and browsers leave Origin off
same-origin GETs. Those are caught by the Host check, which applies when no
token is configured (with a token, the page cannot authenticate anyway).
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
from engram.server.auth import normalize_host, normalize_origin
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


@pytest.mark.asyncio
async def test_every_origin_header_must_be_allowed(app: FastAPI) -> None:
    # An allowed Origin must not carry a second, foreign one through.
    headers = [*MCP_HEADERS.items(), ("Origin", ALLOWED), ("Origin", "http://evil.example")]
    async with _client(app) as c:
        resp = await c.post("/mcp", headers=headers, json=INIT)
        assert resp.status_code == 403


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("https://App.Example.com/", "https://app.example.com"),
        ("https://app.example.com:443", "https://app.example.com"),
        ("http://app.example.com:80", "http://app.example.com"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://[::1]:3000", "http://[::1]:3000"),
        ("https://xn--bcher-kva.example", "https://xn--bcher-kva.example"),
    ],
)
def test_allowlist_entries_are_stored_as_browsers_send_them(entry: str, expected: str) -> None:
    assert normalize_origin(entry) == expected


@pytest.mark.parametrize(
    "entry",
    [
        "https://app.example.com?",
        "https://app.example.com#",
        "http://[::1",
        "https://x:99999",
        # Python's idna codec (IDNA 2003) maps these to a different domain than
        # browsers send (fass.de instead of xn--fa-hia.de). Require punycode.
        "https://fa\u00df.de",
        "https://a\u200db.example",
    ],
)
def test_allowlist_rejects_entries_that_could_never_match(entry: str) -> None:
    with pytest.raises(ValueError, match="allowed origin"):
        normalize_origin(entry)


# ── Host check (DNS rebinding against servers without a token) ─────────


@pytest_asyncio.fixture
async def open_app(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    engram = await _engram(tmp_path, monkeypatch)
    try:
        await engram.record("alice card 4111", user_id="alice")
        yield build_http_app(engram, allowed_hosts=["engram.internal"])
    finally:
        await engram.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    ["rebind.attacker.example:8765", "rebind.attacker.example", "127.0.0.1.evil.example", ""],
)
async def test_unauthenticated_app_refuses_foreign_host(open_app: FastAPI, host: str) -> None:
    # The rebinding read: same-origin GET, so no Origin header at all.
    async with _client(open_app) as c:
        resp = await c.get(
            "/v1/memory/recall",
            params={"query": "card", "user_id": "alice"},
            headers={"Host": host, "Sec-Fetch-Site": "same-origin"},
        )
        assert resp.status_code == 421, f"{host!r} -> {resp.status_code}"
        assert "4111" not in resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:8765", "localhost:19090", "LOCALHOST", "[::1]:8765", "engram.internal:80"],
)
async def test_unauthenticated_app_serves_loopback_and_listed_hosts(
    open_app: FastAPI, host: str
) -> None:
    async with _client(open_app) as c:
        params = {"query": "card", "user_id": "alice"}
        resp = await c.get("/v1/memory/recall", params=params, headers={"Host": host})
        assert resp.status_code == 200, f"{host!r} -> {resp.status_code}"


@pytest.mark.asyncio
async def test_healthz_ignores_host(open_app: FastAPI) -> None:
    async with _client(open_app) as c:
        resp = await c.get("/healthz", headers={"Host": "rebind.attacker.example"})
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_token_app_does_not_check_host_by_default(app: FastAPI) -> None:
    # A rebound page cannot present the token, so Host is not the gate here,
    # and deployments behind any hostname keep working.
    async with _client(app) as c:
        resp = await c.post(
            "/mcp", headers={**MCP_HEADERS, "Host": "engram.example.com"}, json=INIT
        )
        assert resp.status_code == 200


@pytest.mark.parametrize(
    "entry", ["http://x", "x/y", "*.example.com", "x:80", "", "a b", "fa\u00df.de"]
)
def test_allowed_host_entries_are_validated(entry: str) -> None:
    with pytest.raises(ValueError, match="allowed host"):
        normalize_host(entry)


@pytest.mark.asyncio
async def test_two_host_headers_are_refused(open_app: FastAPI) -> None:
    headers = [("Host", "127.0.0.1:8765"), ("Host", "rebind.attacker.example")]
    async with _client(open_app) as c:
        resp = await c.get("/v1/memory/recall", params={"query": "card"}, headers=headers)
        assert resp.status_code == 421


@pytest.mark.asyncio
async def test_host_check_is_on_by_default_without_a_token(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # build_http_app(engram) as in the README: no token, no allowed_hosts.
    engram = await _engram(tmp_path, monkeypatch)
    try:
        app = build_http_app(engram)
        async with _client(app) as c:
            resp = await c.get(
                "/v1/memory/recall",
                params={"query": "q"},
                headers={"Host": "rebind.attacker.example:19090"},
            )
            assert resp.status_code == 421
            resp = await c.get(
                "/v1/memory/recall", params={"query": "q"}, headers={"Host": "localhost:19090"}
            )
            assert resp.status_code == 200
    finally:
        await engram.close()


@pytest.mark.asyncio
async def test_a_bare_string_is_not_taken_as_a_list(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "engram.internal" iterated would allow hosts "e", "n", "g", ...
    engram = await _engram(tmp_path, monkeypatch)
    try:
        with pytest.raises(TypeError, match="allowed_hosts"):
            build_http_app(engram, allowed_hosts="engram.internal")
        with pytest.raises(TypeError, match="allowed_origins"):
            build_http_app(engram, allowed_origins="https://app.example.com")
    finally:
        await engram.close()


@pytest.mark.asyncio
async def test_non_ascii_digits_are_not_a_port(open_app: FastAPI) -> None:
    async with _client(open_app) as c:
        # Latin-1 superscript two, sent as a raw byte.
        headers = [(b"Host", b"localhost:\xb2")]
        resp = await c.get("/v1/memory/recall", params={"query": "q"}, headers=headers)
        assert resp.status_code == 421
