"""Library default, Fetch Metadata, the server's own origin and HEAD /healthz."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from engram.engram import Engram
from engram.server.http import build_http_app

ALLOWED = "https://app.example.com"


async def _engram(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Engram:
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    return await Engram.open(path=str(tmp_path / "engram.db"))


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:8765")


@pytest_asyncio.fixture
async def open_app(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    engram = await _engram(tmp_path, monkeypatch)
    try:
        await engram.record("alice card 4111", user_id="alice")
        yield build_http_app(engram, allow_unauthenticated=True, allowed_origins=[ALLOWED])
    finally:
        await engram.close()


@pytest_asyncio.fixture
async def token_app(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    engram = await _engram(tmp_path, monkeypatch)
    try:
        yield build_http_app(engram, auth_token="secret123")
    finally:
        await engram.close()


# ── the library default fails closed ───────────────────────────────────


@pytest.mark.asyncio
async def test_library_needs_a_token_or_an_explicit_opt_out(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Host check is DNS-rebinding defence, not access control: a remote
    # client can send "Host: localhost". So no token must be a choice.
    engram = await _engram(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="allow_unauthenticated"):
            build_http_app(engram)
        app = build_http_app(engram, allow_unauthenticated=True)
        async with _client(app) as c:
            assert (await c.get("/v1/memory/recall", params={"query": "q"})).status_code == 200
    finally:
        await engram.close()


# ── Fetch Metadata without a token ─────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["cross-site", "same-site", "CROSS-SITE"])
async def test_no_token_refuses_cross_site_requests_without_origin(
    open_app: FastAPI, site: str
) -> None:
    # A no-cors GET (an <img> or <script> tag) carries no Origin, but
    # browsers do send Sec-Fetch-Site.
    async with _client(open_app) as c:
        resp = await c.get(
            "/v1/memory/recall",
            params={"query": "card", "user_id": "alice"},
            headers={"Sec-Fetch-Site": site},
        )
        assert resp.status_code == 403, f"{site} -> {resp.status_code}"


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["same-origin", "none"])
async def test_no_token_serves_same_origin_and_direct_requests(
    open_app: FastAPI, site: str
) -> None:
    async with _client(open_app) as c:
        resp = await c.get(
            "/v1/memory/recall",
            params={"query": "card", "user_id": "alice"},
            headers={"Sec-Fetch-Site": site},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_no_token_serves_cross_site_from_an_allowed_origin(open_app: FastAPI) -> None:
    async with _client(open_app) as c:
        resp = await c.get(
            "/v1/memory/recall",
            params={"query": "card", "user_id": "alice"},
            headers={"Sec-Fetch-Site": "cross-site", "Origin": ALLOWED},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_token_server_leaves_fetch_metadata_to_the_token(token_app: FastAPI) -> None:
    async with _client(token_app) as c:
        resp = await c.get(
            "/v1/memory/recall",
            params={"query": "q"},
            headers={"Sec-Fetch-Site": "cross-site", "Authorization": "Bearer secret123"},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_healthz_ignores_fetch_metadata(open_app: FastAPI) -> None:
    async with _client(open_app) as c:
        resp = await c.get("/healthz", headers={"Sec-Fetch-Site": "cross-site"})
        assert resp.status_code == 200


# ── HEAD /healthz ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_head_healthz_is_200(open_app: FastAPI, token_app: FastAPI) -> None:
    for app in (open_app, token_app):
        async with _client(app) as c:
            resp = await c.head("/healthz")
            assert resp.status_code == 200
            assert resp.content == b""
