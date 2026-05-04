"""Pure date-arithmetic tools — no LLM, deterministic. Saves the model from
doing mental math on temporal questions.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, ClassVar

from engram.tools.base import ToolResult


class AddDaysTool:
    name: ClassVar[str] = "add_days"
    description: ClassVar[str] = (
        "Add N days to an ISO date (YYYY-MM-DD). Returns the resulting ISO date."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "n": {"type": "integer", "description": "Days to add (negative to subtract)"},
        },
        "required": ["date", "n"],
    }

    async def __call__(self, date: str, n: int) -> ToolResult:
        d = _date_from_iso(date) + timedelta(days=n)
        return ToolResult(content=d.isoformat(), raw=d.isoformat())


class DaysBetweenTool:
    name: ClassVar[str] = "days_between"
    description: ClassVar[str] = (
        "Count whole days between two ISO dates (start to end). "
        "Returns an integer; positive when end > start."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "Start ISO date"},
            "end": {"type": "string", "description": "End ISO date"},
        },
        "required": ["start", "end"],
    }

    async def __call__(self, start: str, end: str) -> ToolResult:
        delta = (_date_from_iso(end) - _date_from_iso(start)).days
        return ToolResult(content=str(delta), raw=delta)


def _date_from_iso(s: str) -> date:
    """Parse an ISO date, accepting either YYYY-MM-DD or full ISO datetimes."""
    return date.fromisoformat(s.split("T")[0])
