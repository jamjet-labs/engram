"""MCP server exposing Engram's core operations as MCP tools."""

# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

import logging
from typing import Any

from engram.engram import Engram

logger = logging.getLogger(__name__)


def build_mcp_server(engram: Engram, name: str = "engram") -> Any:
    """Build an MCP `Server` exposing memory_* tools, bound to one Engram instance."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as e:
        raise RuntimeError("mcp package not installed; pip install 'jamjet-engram[mcp]'") from e

    server: Any = Server(name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        typed_recall_props = {
            "memory_systems": {"type": "array", "items": {"type": "string"}},
            "memory_subtypes": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "infer_memory_filters": {"type": "boolean", "default": False},
            "allow_sensitive_recall": {"type": "boolean", "default": False},
        }
        return [
            Tool(
                name="memory_record",
                description="Persist a single fact into Engram memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "user_id": {"type": "string", "default": "default"},
                        "org_id": {"type": "string", "default": "default"},
                        "category": {"type": "string"},
                        "memory_system": {"type": "string"},
                        "memory_subtype": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "route_memory": {"type": "boolean", "default": False},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="memory_recall",
                description=(
                    "Semantic + keyword recall over stored facts. Returns top-K scored hits."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "string", "default": "default"},
                        "org_id": {"type": "string", "default": "default"},
                        "top_k": {"type": "integer", "default": 10},
                        **typed_recall_props,
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="memory_context",
                description=(
                    "Assemble a token-budgeted context string from facts relevant to the query."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "string", "default": "default"},
                        "org_id": {"type": "string", "default": "default"},
                        "token_budget": {"type": "integer", "default": 2000},
                        **typed_recall_props,
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "memory_record":
            fact = await engram.record(
                text=arguments["text"],
                user_id=arguments.get("user_id", "default"),
                org_id=arguments.get("org_id", "default"),
                category=arguments.get("category"),
                memory_system=arguments.get("memory_system"),
                memory_subtype=arguments.get("memory_subtype"),
                tags=_tuple_arg(arguments.get("tags")),
                route_memory=arguments.get("route_memory", False),
            )
            return [TextContent(type="text", text=f"recorded fact {fact.id}")]

        if name == "memory_recall":
            results = await engram.recall(
                query=arguments["query"],
                user_id=arguments.get("user_id", "default"),
                org_id=arguments.get("org_id", "default"),
                top_k=arguments.get("top_k", 10),
                memory_systems=_tuple_arg(arguments.get("memory_systems")),
                memory_subtypes=_tuple_arg(arguments.get("memory_subtypes")),
                tags=_tuple_arg(arguments.get("tags")),
                infer_memory_filters=arguments.get("infer_memory_filters", False),
                allow_sensitive_recall=arguments.get("allow_sensitive_recall", False),
            )
            lines = [f"- [score={sf.score:.3f}] {sf.fact.text}" for sf in results]
            return [TextContent(type="text", text="\n".join(lines) or "(no matches)")]

        if name == "memory_context":
            ctx = await engram.context(
                query=arguments["query"],
                user_id=arguments.get("user_id", "default"),
                org_id=arguments.get("org_id", "default"),
                token_budget=arguments.get("token_budget", 2000),
                memory_systems=_tuple_arg(arguments.get("memory_systems")),
                memory_subtypes=_tuple_arg(arguments.get("memory_subtypes")),
                tags=_tuple_arg(arguments.get("tags")),
                infer_memory_filters=arguments.get("infer_memory_filters", False),
                allow_sensitive_recall=arguments.get("allow_sensitive_recall", False),
            )
            return [TextContent(type="text", text=ctx)]

        raise ValueError(f"unknown tool: {name}")

    return server


def _tuple_arg(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)
