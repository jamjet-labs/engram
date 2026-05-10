"""Streamable HTTP transport adapter for Engram's MCP server.

This is the only module that imports from `mcp.server.streamable_http_manager`.
If the MCP SDK changes its transport API, only this file changes.

Usage:
    manager, handler = build_streamable_http_mount(mcp_server)
    # `manager.run()` must be entered as a lifespan context in the host app.
    # `handler` is an ASGI handler (callable scope/receive/send) suitable for
    # mounting.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

ASGIHandler = Callable[
    [dict[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]


def build_streamable_http_mount(
    mcp_server: Server,
) -> tuple[StreamableHTTPSessionManager, ASGIHandler]:
    """Build a Streamable HTTP session manager and ASGI handler for `mcp_server`.

    Returns `(manager, handler)`. The caller is responsible for:
      1. Entering `manager.run()` as a lifespan context manager on the host app.
      2. Mounting `handler` (or an auth-wrapped version of it) at the desired path.

    The manager is created with `stateless=True` because Engram tools have no
    per-session state — each call hits the same Engram instance. This means
    multiple replicas can sit behind a load balancer without sticky sessions.
    """
    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        stateless=True,
        json_response=False,
    )
    return manager, manager.handle_request
