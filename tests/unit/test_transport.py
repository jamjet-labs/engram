"""Unit tests for the Streamable HTTP transport adapter."""

from __future__ import annotations

import pytest
import pytest_asyncio

from engram.engram import Engram
from engram.server.mcp import build_mcp_server
from engram.server.transport import build_streamable_http_mount


@pytest_asyncio.fixture
async def engram_instance(tmp_path, monkeypatch):
    """Create an Engram instance with mock LLM for testing."""
    monkeypatch.setenv("ENGRAM_LLM_PROVIDER", "mock")
    engram = await Engram.open(path=str(tmp_path / "engram.db"))
    try:
        yield engram
    finally:
        await engram.close()


@pytest.mark.asyncio
async def test_returns_manager_and_handler(engram_instance):
    """Test that build_streamable_http_mount returns manager and handler."""
    mcp_server = build_mcp_server(engram_instance, name="engram-test")
    manager, handler = build_streamable_http_mount(mcp_server)

    # The handler is the manager's handle_request method.
    assert handler == manager.handle_request

    # The manager should be configured stateless per the v0.2 spec.
    assert manager.stateless is True


@pytest.mark.asyncio
async def test_idempotent_singleton_warning(engram_instance):
    """Per StreamableHTTPSessionManager docs, only one instance per app — but our
    builder is called once at app-build time, so this just verifies the builder
    returns a fresh instance on each call (callers must not call it twice)."""
    mcp_server = build_mcp_server(engram_instance, name="engram-test")

    m1, _ = build_streamable_http_mount(mcp_server)
    m2, _ = build_streamable_http_mount(mcp_server)
    assert m1 is not m2
