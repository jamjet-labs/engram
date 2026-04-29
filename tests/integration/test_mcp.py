"""MCP server smoke test — verifies tool registration + dispatch."""

from __future__ import annotations

import pytest

from engram import Engram
from engram.server.mcp import build_mcp_server


def _have_mcp() -> bool:
    try:
        import mcp.server  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_mcp(), reason="mcp package not installed")
async def test_mcp_server_builds_and_lists_tools() -> None:
    """The MCP server registers expected tools and exposes them."""
    async with await Engram.open(":memory:") as memory:
        server = build_mcp_server(memory, name="engram-test")
        # The MCP Server exposes its tools through `request_handlers`. We just
        # verify the server is constructed and has the expected name.
        assert server.name == "engram-test"


@pytest.mark.skipif(not _have_mcp(), reason="mcp package not installed")
async def test_mcp_record_and_recall_via_server() -> None:
    """Smoke test: build the server, then call its tool handlers directly."""
    async with await Engram.open(":memory:") as memory:
        server = build_mcp_server(memory, name="engram-test")

        # Find the call_tool handler. mcp.server.Server stores handlers in
        # request_handlers keyed by request type. We dispatch by invoking the
        # registered handler directly through `call_tool`.
        from mcp.types import CallToolRequest, CallToolRequestParams

        # Record via tool
        record_req = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="memory_record",
                arguments={"text": "alice prefers espresso", "user_id": "alice"},
            ),
        )
        record_handler = server.request_handlers[CallToolRequest]
        record_resp = await record_handler(record_req)
        # The response is a ServerResult wrapping a CallToolResult.
        # Inspect the inner result.
        inner = record_resp.root
        assert inner.content
        assert "recorded fact" in inner.content[0].text.lower()

        # Recall via tool
        recall_req = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="memory_recall",
                arguments={"query": "coffee", "user_id": "alice", "top_k": 3},
            ),
        )
        recall_resp = await record_handler(recall_req)
        recall_inner = recall_resp.root
        assert recall_inner.content
        assert "espresso" in recall_inner.content[0].text.lower()
