"""ASGI guards for the Engram HTTP server: bearer token, Origin and Host checks.

`build_http_app` wraps the whole app, REST routes and `/mcp` alike. Only
`/healthz` is exempt. One token gates everything the server can reach, so one
token is one trust domain: whoever holds it can read and write every
`(org_id, user_id)` scope in that database.

DNS rebinding: a web page can point its own hostname at 127.0.0.1 and then talk
to a local server. Its cross-origin requests and POSTs carry an `Origin` header
(checked by `OriginGuardMiddleware`). Its same-origin GETs carry none, but
their `Host` is the attacker's name (checked by `HostGuardMiddleware`, which
`build_http_app` turns on when there is no token).
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
_MISDIRECTED_BODY = json.dumps({"error": "host not allowed"}).encode("utf-8")

# Hostnames that always reach a server bound to loopback.
LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

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
        check_token(expected_token)
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


def check_token(token: str) -> None:
    """Raise ValueError for a token no client could reliably present."""
    if not token:
        raise ValueError("auth token must be a non-empty string")
    if not all("!" <= ch <= "~" for ch in token):
        # Servers strip whitespace around header values, and common clients
        # will not send control or non-ASCII characters. Such a token (say, one
        # with a newline from a secrets file) would get 401 on every request.
        raise ValueError("auth token must be printable ASCII with no spaces or control characters")


def normalize_origin(origin: str) -> str:
    """Validate one allowlist entry and return it in the form browsers send.

    Accepts `scheme://host[:port]` with scheme http or https. Rejects
    wildcards, `null`, paths, queries, fragments and userinfo, so a typo cannot
    widen the allowlist.
    """

    def bad(why: str) -> ValueError:
        return ValueError(
            f"invalid allowed origin {origin!r}: {why}; expected scheme://host[:port], "
            "for example https://app.example.com"
        )

    value = origin.strip().rstrip("/")
    if "*" in value or "?" in value or "#" in value or "@" in value:
        raise bad("no wildcards, query, fragment or userinfo")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as e:
        raise bad(str(e)) from e
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname or parts.path:
        raise bad("http or https with a host and no path")
    host = parts.hostname
    if not host.isascii():
        # Python's idna codec is IDNA 2003, which maps some names to a
        # different domain than browsers send (faß.de -> fass.de). Ask for
        # the punycode form instead of guessing.
        raise bad("write a non-ASCII host in its punycode (xn--) form")
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        host = f"{host}:{port}"
    return f"{scheme}://{host}"


def normalize_host(host: str) -> str:
    """Validate one allowed-host entry: a bare hostname or IP, matched on any port."""
    value = host.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    bad_chars = set("/*@?#[] \t")
    if (
        not value
        or not value.isascii()
        or bad_chars & set(value)
        or (":" in value and value.count(":") < 2)
    ):
        raise ValueError(
            f"invalid allowed host {host!r}: expected an ASCII hostname or IP without "
            "scheme, port or path (punycode for non-ASCII names), for example engram.internal"
        )
    return value


def _request_hostname(raw: bytes) -> str | None:
    """Hostname part of a Host header, lowercased, or None if malformed."""
    value = raw.decode("latin-1").strip().lower()
    if value.startswith("["):
        end = value.find("]")
        rest = value[end + 1 :]
        if end == -1 or (rest and not (rest[:1] == ":" and _is_port(rest[1:]))):
            return None
        return value[1:end] or None
    name, sep, port = value.rpartition(":")
    if not sep:
        return value or None
    if not _is_port(port):
        return None
    return name or None


def _is_port(value: str) -> bool:
    # str.isdigit alone accepts superscripts and other non-ASCII digits.
    return value.isascii() and value.isdigit()


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


class HostGuardMiddleware:
    """Reject requests whose `Host` hostname is not on the allowlist (421).

    Stops DNS-rebinding reads against a server without a token: the rebound
    page is same-origin with the server, so its GETs carry no `Origin`, but
    their `Host` is still the attacker's hostname. Any port is accepted.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_hosts: Iterable[str],
        exempt_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self._allowed = frozenset(normalize_host(h) for h in allowed_hosts)
        self._exempt = frozenset(exempt_paths)

    def _allowed_request(self, scope: Scope) -> bool:
        hosts = [value for name, value in scope.get("headers", []) if name == b"host"]
        if len(hosts) != 1:
            return False
        return _request_hostname(hosts[0]) in self._allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "lifespan"
            or _is_exempt(scope, self._exempt)
            or self._allowed_request(scope)
        ):
            await self.app(scope, receive, send)
            return
        await _reject(scope, send, 421, _MISDIRECTED_BODY)


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
