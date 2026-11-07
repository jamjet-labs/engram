"""ASGI guards for the Engram HTTP server: bearer-token auth and Origin checks.

`build_http_app` wraps the whole app, REST routes and `/mcp` alike, in both.
Only `/healthz` is exempt. One token gates everything the server can reach, so
one token is one trust domain: whoever holds it can read and write every
`(org_id, user_id)` scope in that database.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from secrets import compare_digest
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

HEALTHZ_PATH = "/healthz"

_UNAUTH_BODY = json.dumps({"error": "invalid bearer token"}).encode("utf-8")
_FORBIDDEN_ORIGIN_BODY = json.dumps({"error": "origin not allowed"}).encode("utf-8")

# RFC 6455 7.4.1: 1008 = policy violation.
_WS_POLICY_VIOLATION = 1008


async def _reject(scope: Scope, send: Send, status: int, body: bytes) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
        return
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if status == 401:
        headers.append((b"www-authenticate", b'Bearer realm="engram"'))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _is_exempt(scope: Scope, exempt_paths: frozenset[str]) -> bool:
    return scope["type"] == "http" and scope.get("path") in exempt_paths


class BearerAuthMiddleware:
    """Require `Authorization: Bearer <token>` on every HTTP and websocket request.

    Lifespan events pass through. Paths in `exempt_paths` (exact match) pass
    through. Everything else without the token gets a 401, or a websocket close
    with code 1008.
    """

    def __init__(
        self, app: ASGIApp, *, expected_token: str, exempt_paths: Iterable[str] = ()
    ) -> None:
        if not expected_token:
            raise ValueError("expected_token must be a non-empty string")
        self.app = app
        self._token = expected_token.encode("utf-8")
        self._exempt = frozenset(exempt_paths)

    def _authorized(self, scope: Scope) -> bool:
        header_value = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                header_value = value
                break
        # RFC 9110 §11: auth-scheme tokens are case-insensitive (Go's oauth2
        # client and some curl invocations send "bearer").
        scheme, _, presented = header_value.partition(b" ")
        if scheme.lower() != b"bearer" or not presented:
            return False
        # Compare bytes: compare_digest on str raises TypeError for non-ASCII
        # input, which would turn a bad token into a 500.
        return compare_digest(presented, self._token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or _is_exempt(scope, self._exempt):
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await _reject(scope, send, 401, _UNAUTH_BODY)


def normalize_origin(origin: str) -> str:
    """Validate one allowlist entry and return it in the form browsers send.

    Accepts `scheme://host[:port]` with scheme http or https. Rejects
    wildcards, `null`, paths, queries, fragments and userinfo, so a typo cannot
    widen the allowlist.
    """
    value = origin.strip().rstrip("/").lower()
    parts = urlsplit(value)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or "*" in value
        or parts.path
        or parts.query
        or parts.fragment
        or "@" in parts.netloc
    ):
        raise ValueError(
            f"invalid allowed origin {origin!r}: expected scheme://host[:port], "
            "for example https://app.example.com"
        )
    try:
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError as e:
        raise ValueError(f"invalid allowed origin {origin!r}: bad port") from e
    return value


class OriginGuardMiddleware:
    """Reject browser requests whose `Origin` is not on the allowlist (403).

    Requests without an Origin header pass: MCP and REST clients outside a
    browser do not send one. This is the DNS-rebinding check the MCP Streamable
    HTTP transport requires. The allowlist is never derived from the Host
    header, because a rebinding attacker controls both.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_origins: Iterable[str] = (),
        exempt_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self._allowed = frozenset(normalize_origin(o) for o in allowed_origins)
        self._exempt = frozenset(exempt_paths)

    def _allowed_request(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", []):
            if name != b"origin":
                continue
            origin = value.decode("latin-1").strip().lower()
            if origin not in self._allowed:
                return False
        return True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "lifespan"
            or _is_exempt(scope, self._exempt)
            or self._allowed_request(scope)
        ):
            await self.app(scope, receive, send)
            return
        await _reject(scope, send, 403, _FORBIDDEN_ORIGIN_BODY)


def auth_asgi_wrapper(
    app: ASGIApp, expected_token: str | None, *, exempt_paths: Iterable[str] = ()
) -> ASGIApp:
    """Return `app` wrapped with bearer-token auth, or `app` itself if no token.

    `expected_token=None` means no auth (the CLI only allows that on loopback
    unless `--allow-unauthenticated` is given). An empty string is an error.
    """
    if expected_token is None:
        return app
    return BearerAuthMiddleware(app, expected_token=expected_token, exempt_paths=exempt_paths)
