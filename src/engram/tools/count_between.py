"""Deterministic event-count between two ISO dates, optionally filtered."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from engram.scope import Scope
from engram.tools.base import ToolResult


class CountBetweenTool:
    name: ClassVar[str] = "count_between"
    description: ClassVar[str] = (
        "Count events between two ISO dates, optionally filtered by verb "
        "or by an object-substring match."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "Start ISO date YYYY-MM-DD"},
            "end": {"type": "string", "description": "End ISO date YYYY-MM-DD"},
            "verb": {"type": "string", "description": "Optional verb filter (exact match)"},
            "object_substring": {
                "type": "string",
                "description": "Optional substring to match against object_canonical",
            },
        },
        "required": ["start", "end"],
    }

    def __init__(self, engram: Any, scope: Scope) -> None:
        self._engram = engram
        self._scope = scope

    async def __call__(
        self,
        start: str,
        end: str,
        verb: str | None = None,
        object_substring: str | None = None,
    ) -> ToolResult:
        time_start = _to_utc(start)
        time_end = _to_utc(end)
        events = await self._engram.search_events(
            query=verb or "",
            user_id=self._scope.user_id,
            org_id=self._scope.org_id,
            time_start=time_start,
            time_end=time_end,
            limit=500,
        )
        if verb:
            events = [e for e in events if e.verb.lower() == verb.lower()]
        if object_substring:
            o = object_substring.lower()
            events = [e for e in events if o in e.object_canonical.lower()]
        return ToolResult(content=str(len(events)), raw=len(events))


def _to_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.split("T")[0]).replace(tzinfo=UTC)
