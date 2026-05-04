"""SVO event-calendar search tool."""

from __future__ import annotations

from typing import Any, ClassVar

from engram.scope import Scope
from engram.tools.base import ToolResult


class SearchEventsTool:
    name: ClassVar[str] = "search_events"
    description: ClassVar[str] = (
        "Search the SVO event calendar by free-text query. "
        "Returns matching events with their dates."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }

    def __init__(self, engram: Any, scope: Scope) -> None:
        self._engram = engram
        self._scope = scope

    async def __call__(self, query: str, limit: int = 10) -> ToolResult:
        events = await self._engram.search_events(
            query=query,
            user_id=self._scope.user_id,
            org_id=self._scope.org_id,
            limit=limit,
        )
        if not events:
            return ToolResult(content="(no events found)", raw=[])
        lines = [
            f"- [{e.time_start.date().isoformat()}] "
            f"{e.subject_canonical} {e.verb} {e.object_canonical}"
            for e in events
        ]
        return ToolResult(content="\n".join(lines), raw=[str(e.id) for e in events])
