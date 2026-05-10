"""Bearer-token ASGI auth wrapper.

Wraps an ASGI app so HTTP requests must carry `Authorization: Bearer <token>`.
Non-HTTP scopes (lifespan, websocket) pass through unchanged.

Used to gate the `/mcp` mount in `build_http_app`. The `/v1/memory/*` REST
routes are intentionally NOT gated — see the v0.2 design spec.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from secrets import compare_digest
from typing import Any

# Type aliases to stay within line length limits
ReceiveFn = Callable[[], Awaitable[Any]]
SendFn = Callable[[Any], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ReceiveFn, SendFn], Awaitable[None]]


_UNAUTH_BODY = json.dumps({"error": "invalid bearer token"}).encode("utf-8")


async def _send_401(send: SendFn) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_UNAUTH_BODY)).encode("ascii")),
                (b"www-authenticate", b'Bearer realm="engram"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _UNAUTH_BODY})


def auth_asgi_wrapper(app: ASGIApp, expected_token: str | None) -> ASGIApp:
    """Return `app` wrapped with bearer-token auth.

    If `expected_token` is None, returns `app` unchanged (used for --no-auth mode).
    Otherwise, HTTP requests without `Authorization: Bearer <token>` matching
    `expected_token` (constant-time compared) get a 401 JSON response.
    Non-HTTP scopes pass through.
    """
    if expected_token is None:
        return app

    async def wrapped(scope: dict[str, Any], receive: ReceiveFn, send: SendFn) -> None:
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return

        header_value = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                header_value = value
                break

        # RFC 9110 §11: auth-scheme tokens are case-insensitive (e.g. Go's oauth2
        # client and curl can emit "bearer" lowercase). Match scheme tolerantly.
        scheme, _, rest = header_value.partition(b" ")
        if scheme.lower() == b"bearer" and rest:
            token = rest.decode("utf-8", errors="replace")
            if compare_digest(token, expected_token):
                await app(scope, receive, send)
                return

        await _send_401(send)

    return wrapped
