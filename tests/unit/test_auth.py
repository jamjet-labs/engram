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


def test_lowercase_bearer_accepted():
    # RFC 9110 §11: auth-scheme tokens are case-insensitive. Some clients
    # (Go oauth2, curl with explicit casing) send "bearer" lowercase.
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    resp = client.get("/", headers={"Authorization": "bearer secret123"})
    assert resp.status_code == 200


def test_uppercase_bearer_accepted():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123")
    client = TestClient(wrapped)
    resp = client.get("/", headers={"Authorization": "BEARER secret123"})
    assert resp.status_code == 200


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


def test_lifespan_scope_passes_through():
    # Lifespan carries no request, so there is nothing to authenticate.
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    wrapped = auth_asgi_wrapper(inner, expected_token="secret123")

    import anyio

    async def drive():
        await wrapped({"type": "lifespan"}, lambda: None, lambda m: None)

    anyio.run(drive)
    assert seen == ["lifespan"]


def _drive_websocket(headers):
    import anyio

    reached = []
    sent = []

    async def inner(scope, receive, send):
        reached.append(scope["type"])

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "websocket.connect"}

    wrapped = auth_asgi_wrapper(inner, expected_token="secret123")

    async def drive():
        await wrapped({"type": "websocket", "path": "/ws", "headers": headers}, receive, send)

    anyio.run(drive)
    return reached, sent


def test_websocket_without_token_is_closed():
    # No websocket routes exist today; if one is added it must not skip auth.
    reached, sent = _drive_websocket([])
    assert reached == []
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_websocket_with_token_passes():
    reached, _ = _drive_websocket([(b"authorization", b"Bearer secret123")])
    assert reached == ["websocket"]


def test_non_ascii_token_is_401_not_500():
    # compare_digest on str raises TypeError for non-ASCII input, which used to
    # surface as a 500. Compare bytes instead.
    import anyio

    sent = []

    async def inner(scope, receive, send):
        raise AssertionError("must not reach the app")

    async def send(message):
        sent.append(message)

    wrapped = auth_asgi_wrapper(inner, expected_token="secret123")
    scope = {
        "type": "http",
        "path": "/",
        "headers": [(b"authorization", "Bearer s\u00e9cret123".encode("utf-8"))],
    }

    async def drive():
        await wrapped(scope, lambda: None, send)

    anyio.run(drive)
    assert sent[0]["status"] == 401


def test_exempt_path_skips_auth():
    wrapped = auth_asgi_wrapper(_ok_app(), expected_token="secret123", exempt_paths={"/"})
    client = TestClient(wrapped)
    assert client.get("/").status_code == 200
