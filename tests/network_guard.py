"""The test network guard, a pytest plugin for every test path.

The root conftest.py registers it, so tests under tests/ and training/ both
get it. tests/unit/test_vertex_embedding.py and
training/cross_encoder/tests/test_network_guard.py prove it works.
"""

from __future__ import annotations

import ipaddress
from typing import NoReturn

import httpx
import pytest


class NetworkCallBlocked(RuntimeError):
    """A test tried to reach the network or look up Google credentials."""


def _is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _refuse_unless_loopback(request: httpx.Request) -> None:
    if not _is_loopback(request.url.host):
        raise NetworkCallBlocked(
            f"the test network guard blocked a request to {request.url.host}; "
            "give the client an httpx.MockTransport"
        )


def _refuse_adc(*args: object, **kwargs: object) -> NoReturn:
    raise NetworkCallBlocked(
        "the test network guard blocked google.auth.default(); pass fake credentials"
    )


@pytest.fixture(autouse=True)
def _network_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the network or reads Google credentials.

    Real httpx transports refuse every host but loopback, so a local server
    (Ollama) still answers and a refused port still fails as it should.
    `google.auth.default` raises. MockTransport, ASGI and WSGI transports are
    not touched.
    """
    sync_send = httpx.HTTPTransport.handle_request
    async_send = httpx.AsyncHTTPTransport.handle_async_request

    def handle_request(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        _refuse_unless_loopback(request)
        return sync_send(self, request)

    async def handle_async_request(
        self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        _refuse_unless_loopback(request)
        return await async_send(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request)
    try:
        import google.auth
    except ImportError:  # the vertex extra is not installed
        return
    monkeypatch.setattr(google.auth, "default", _refuse_adc)
