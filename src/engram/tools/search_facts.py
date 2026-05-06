"""Semantic search over the user's stored facts."""

from __future__ import annotations

from typing import Any, ClassVar

from engram.scope import Scope
from engram.tools.base import ToolResult


class SearchFactsTool:
    name: ClassVar[str] = "search_facts"
    description: ClassVar[str] = (
        "Search the user's stored facts by semantic query. "
        "Returns up to top_k matching facts as a bullet list."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, engram: Any, scope: Scope) -> None:
        self._engram = engram
        self._scope = scope

    async def __call__(self, query: str, top_k: int = 5) -> ToolResult:
        scored = await self._engram.recall(
            query=query,
            user_id=self._scope.user_id,
            org_id=self._scope.org_id,
            top_k=top_k,
        )
        if not scored:
            return ToolResult(content="(no facts found)", raw=[])
        lines = [f"- {sf.fact.text}" for sf in scored]
        return ToolResult(content="\n".join(lines), raw=[str(sf.fact.id) for sf in scored])
