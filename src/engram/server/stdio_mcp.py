"""Stdio MCP server entry point with typed-memory profile support."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool, ToolsCapability

from engram import Engram


def _tuple_arg(val: list | None) -> tuple | None:
    return tuple(val) if val is not None else None


async def serve(db_path: str) -> None:
    profile_name = os.environ.get("ENGRAM_PROFILE", "")
    profile_tag = f"profile:{profile_name}" if profile_name else None
    server = Server("engram-typed")

    async with await Engram.open(db_path) as memory:
        typed_recall_props = {
            "memory_systems": {"type": "array", "items": {"type": "string"}},
            "memory_subtypes": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "infer_memory_filters": {"type": "boolean", "default": False},
            "allow_sensitive_recall": {"type": "boolean", "default": False},
        }

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="memory_record",
                    description="Persist a fact with typed memory routing.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "user_id": {"type": "string", "default": "default"},
                            "memory_system": {"type": "string"},
                            "memory_subtype": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "route_memory": {"type": "boolean", "default": False},
                            "lifecycle_state": {"type": "string"},
                            "promotion_state": {"type": "string"},
                            "retrieval_policy": {"type": "string"},
                        },
                        "required": ["text"],
                    },
                ),
                Tool(
                    name="memory_recall",
                    description="Typed recall over stored facts.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "user_id": {"type": "string", "default": "default"},
                            "top_k": {"type": "integer", "default": 10},
                            **typed_recall_props,
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="memory_context",
                    description="Assemble token-budgeted typed context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "user_id": {"type": "string", "default": "default"},
                            "token_budget": {"type": "integer", "default": 2000},
                            **typed_recall_props,
                        },
                        "required": ["query"],
                    },
                ),
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "memory_record":
                tags = list(_tuple_arg(arguments.get("tags")) or ())
                if profile_tag and profile_tag not in tags:
                    tags.append(profile_tag)
                f = await memory.record(
                    text=arguments["text"],
                    user_id=arguments.get("user_id", "default"),
                    memory_system=arguments.get("memory_system"),
                    memory_subtype=arguments.get("memory_subtype"),
                    tags=tuple(tags) if tags else None,
                    route_memory=arguments.get("route_memory", False),
                    lifecycle_state=arguments.get("lifecycle_state"),
                    promotion_state=arguments.get("promotion_state"),
                    retrieval_policy=arguments.get("retrieval_policy"),
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "id": str(f.id),
                                "memory_system": f.memory_system.value if f.memory_system else None,
                                "memory_subtype": f.memory_subtype,
                                "lifecycle_state": (
                                    f.lifecycle_state.value if f.lifecycle_state else None
                                ),
                            }
                        ),
                    )
                ]

            if name == "memory_recall":
                results = await memory.recall(
                    query=arguments["query"],
                    user_id=arguments.get("user_id", "default"),
                    top_k=arguments.get("top_k", 10),
                    memory_systems=_tuple_arg(arguments.get("memory_systems")),
                    memory_subtypes=_tuple_arg(arguments.get("memory_subtypes")),
                    tags=_tuple_arg(arguments.get("tags")),
                    infer_memory_filters=arguments.get("infer_memory_filters", False),
                    allow_sensitive_recall=arguments.get("allow_sensitive_recall", False),
                )
                lines = [
                    json.dumps(
                        {
                            "text": sf.fact.text,
                            "score": round(sf.score, 3),
                            "memory_system": (
                                sf.fact.memory_system.value if sf.fact.memory_system else None
                            ),
                            "memory_subtype": sf.fact.memory_subtype,
                            "tags": list(sf.fact.tags or ()),
                        }
                    )
                    for sf in results
                ]
                return [TextContent(type="text", text="\n".join(lines) or "(no matches)")]

            if name == "memory_context":
                ctx = await memory.context(
                    query=arguments["query"],
                    user_id=arguments.get("user_id", "default"),
                    token_budget=arguments.get("token_budget", 2000),
                    memory_systems=_tuple_arg(arguments.get("memory_systems")),
                    memory_subtypes=_tuple_arg(arguments.get("memory_subtypes")),
                    tags=_tuple_arg(arguments.get("tags")),
                    infer_memory_filters=arguments.get("infer_memory_filters", False),
                    allow_sensitive_recall=arguments.get("allow_sensitive_recall", False),
                )
                return [TextContent(type="text", text=ctx)]

            raise ValueError(f"unknown tool: {name}")

        async with stdio_server() as (read, write):
            await server.run(
                read,
                write,
                InitializationOptions(
                    server_name="engram-typed",
                    server_version="0.2.0",
                    capabilities=ServerCapabilities(tools=ToolsCapability(listChanged=False)),
                ),
            )


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".engram" / "engram.db")
    asyncio.run(serve(db))


if __name__ == "__main__":
    main()
