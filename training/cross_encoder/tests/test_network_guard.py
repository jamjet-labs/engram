"""The test network guard covers the training tests too.

The guard lives in tests/network_guard.py and the root conftest.py registers
it, so every test path gets it, not only tests/.
"""

from __future__ import annotations

from typing import Any

import httpcore
import httpx
import pytest

URL = "https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L-6-v2"


class ReachedTheConnectionPool(Exception):
    pass


async def test_the_network_guard_blocks_a_real_transport_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real transport to the hub is refused before it opens a socket.

    Backstop: the connection pools raise before any socket opens, so without
    the guard this test fails on them and nothing leaves the machine.
    """

    def sync_pool(*args: Any, **kw: Any) -> Any:
        raise ReachedTheConnectionPool

    async def async_pool(*args: Any, **kw: Any) -> Any:
        raise ReachedTheConnectionPool

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", sync_pool)
    monkeypatch.setattr(httpcore.AsyncConnectionPool, "handle_async_request", async_pool)
    guard = "network guard"

    def blocked_without_a_loop() -> None:
        with pytest.raises(RuntimeError, match=guard):
            httpx.HTTPTransport().handle_request(httpx.Request("GET", URL))
        with httpx.Client(trust_env=False) as client, pytest.raises(RuntimeError, match=guard):
            client.get(URL)

    with pytest.raises(RuntimeError, match=guard):
        await httpx.AsyncHTTPTransport().handle_async_request(httpx.Request("GET", URL))
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(RuntimeError, match=guard):
            await client.get(URL)
    blocked_without_a_loop()
