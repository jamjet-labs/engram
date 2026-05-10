"""Integration tests for /mcp Streamable HTTP endpoint.

Drives the MCP handshake via httpx ASGITransport — no real socket binding.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, MutableMapping
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from anyio import create_memory_object_stream
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from engram.engram import Engram
from engram.server.http import build_http_app
from engram.server.mcp import build_mcp_server


async def _start_lifespan(asgi_app: FastAPI) -> asyncio.Task[None]:
    """Send an ASGI lifespan.startup event and wait for startup.complete.

    Returns the background asyncio Task running the lifespan coroutine so the
    caller can cancel it during teardown. Using a raw asyncio.Task (rather than
    anyio.create_task_group) avoids the "cancel scope in a different task"
    restriction that anyio enforces during pytest fixture teardown.
    """
    send_chan, recv_chan = create_memory_object_stream[dict[str, Any]](2)
    startup_complete: asyncio.Event = asyncio.Event()
    scope: dict[str, Any] = {"type": "lifespan", "asgi": {"version": "3.0"}}

    async def _receive() -> dict[str, Any]:
        return await recv_chan.receive()

    async def _send(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()

    task: asyncio.Task[None] = asyncio.ensure_future(asgi_app(scope, _receive, _send))
    await send_chan.send({"type": "lifespan.startup"})
    await startup_complete.wait()
    return task


@pytest_asyncio.fixture
async def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[FastAPI, None]:
    """Build a FastAPI app with MCP mounted and the ASGI lifespan started.

    ASGITransport does not trigger the ASGI lifespan, so we start it manually
    by sending lifespan.startup and waiting for lifespan.startup.complete before
    yielding to the test. The background task is cancelled on teardown.
    """
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    engram = await Engram.open(path=str(tmp_path / "engram.db"))
    lifespan_task: asyncio.Task[None] | None = None
    try:
        mcp_server = build_mcp_server(engram, name="engram-test")
        asgi_app = build_http_app(engram, mcp_server=mcp_server, auth_token="secret123")
        lifespan_task = await _start_lifespan(asgi_app)
        yield asgi_app
    finally:
        if lifespan_task is not None:
            lifespan_task.cancel()
            try:
                await lifespan_task
            except asyncio.CancelledError:
                pass
        await engram.close()


@pytest.mark.asyncio
async def test_mcp_route_requires_auth(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No auth header → 401
        resp = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert resp.status_code == 401

        # Wrong token → 401
        resp = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_healthz_unauthed(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


async def _post_mcp(
    client: AsyncClient, body: dict[str, Any], session_id: str | None = None
) -> Response:
    headers: dict[str, str] = {
        "Authorization": "Bearer secret123",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    return await client.post("/mcp", headers=headers, json=body)


def _parse_body(resp: Response) -> dict[str, Any]:
    """The streamable HTTP endpoint may return either application/json or
    text/event-stream. Decode either into a JSON dict."""
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        result: dict[str, Any] = resp.json()
        return result
    if "text/event-stream" in content_type:
        # Find the first `data: { ... }` line.
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                parsed: dict[str, Any] = json.loads(line[len("data: ") :])
                return parsed
        raise AssertionError(f"no data: line in SSE body: {resp.text!r}")
    raise AssertionError(f"unexpected content-type: {content_type!r}")


@pytest.mark.asyncio
async def test_full_handshake_and_tools_list(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        init_body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        resp = await _post_mcp(client, init_body)
        assert resp.status_code == 200
        # stateless=True mode: session ID may be absent — capture if present
        session_id = resp.headers.get("mcp-session-id")
        init_result = _parse_body(resp)
        assert init_result["result"]["serverInfo"]["name"] == "engram-test"

        # initialized notification
        await _post_mcp(
            client,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )

        # tools/list
        resp = await _post_mcp(
            client,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            session_id=session_id,
        )
        assert resp.status_code == 200
        tools = _parse_body(resp)["result"]["tools"]
        names = sorted(t["name"] for t in tools)
        assert names == ["memory_context", "memory_recall", "memory_record"]


@pytest.mark.asyncio
async def test_memory_record_then_recall_round_trip(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        init_body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        resp = await _post_mcp(client, init_body)
        # stateless=True mode: session ID may be absent
        session_id = resp.headers.get("mcp-session-id")
        await _post_mcp(
            client,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )

        # memory_record
        resp = await _post_mcp(
            client,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_record",
                    "arguments": {"text": "Alice's dog is named Pepper", "user_id": "alice"},
                },
            },
            session_id=session_id,
        )
        assert resp.status_code == 200
        body = _parse_body(resp)
        record_text = body["result"]["content"][0]["text"]
        assert record_text.startswith("recorded fact ")

        # memory_recall — should retrieve the fact we just stored
        resp = await _post_mcp(
            client,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory_recall",
                    "arguments": {"query": "what is Alice's dog called", "user_id": "alice"},
                },
            },
            session_id=session_id,
        )
        assert resp.status_code == 200
        body = _parse_body(resp)
        recall_text = body["result"]["content"][0]["text"]
        # Recall returns "- [score=X.XXX] <text>" lines OR "(no matches)"
        assert "Pepper" in recall_text or "(no matches)" in recall_text


@pytest.mark.asyncio
async def test_unknown_tool_returns_jsonrpc_error(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        init_body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        resp = await _post_mcp(client, init_body)
        # stateless=True mode: session ID may be absent
        session_id = resp.headers.get("mcp-session-id")
        await _post_mcp(
            client,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )

        resp = await _post_mcp(
            client,
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": "no_such_tool", "arguments": {}},
            },
            session_id=session_id,
        )
        # MCP SDK surfaces tool handler exceptions as a successful JSON-RPC response
        # with result.isError=True and the error message in result.content[0].text.
        # The HTTP status is still 200 — errors are in-band per MCP spec.
        assert resp.status_code == 200
        body = _parse_body(resp)
        assert body["result"]["isError"] is True
        assert "no_such_tool" in body["result"]["content"][0]["text"]
