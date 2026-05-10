"""Tests for the bearer-token ASGI auth wrapper."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from engram.server.auth import auth_asgi_wrapper


def _ok_app() -> Starlette:
    async def hello(request):
        return PlainTextResponse("ok")

    return Starlette(routes=[Route("/", hello)])


def test_no_token_disables_auth():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token=None)
    client = TestClient(wrapped)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.text == "ok"


def test_correct_token_passes():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    resp = client.get("/", headers={"Authorization": "Bearer secret123"})
    assert resp.status_code == 200


def test_missing_header_rejected():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    resp = client.get("/")
    assert resp.status_code == 401
    assert resp.json() == {"error": "invalid bearer token"}


def test_wrong_scheme_rejected():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    resp = client.get("/", headers={"Authorization": "Basic secret123"})
    assert resp.status_code == 401


def test_wrong_token_rejected():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    resp = client.get("/", headers={"Authorization": "Bearer not-the-token"})
    assert resp.status_code == 401


def test_constant_time_compare_used(monkeypatch):
    calls = []
    import secrets as _secrets

    real_compare = _secrets.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr("engram.server.auth.compare_digest", spy)
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    client.get("/", headers={"Authorization": "Bearer secret123"})
    assert calls, "compare_digest should have been called for token comparison"


def test_websocket_scope_passes_through():
    # Non-HTTP scopes (websocket, lifespan) should pass through unchanged so the
    # underlying app handles them.
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    wrapped = auth_asgi_wrapper(inner, expected_token="secret123")

    import anyio

    async def drive():
        await wrapped({"type": "lifespan"}, lambda: None, lambda m: None)

    anyio.run(drive)
    assert seen == ["lifespan"]
