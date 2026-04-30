"""Tool wrapper around the programmatic temporal solver (item 3 / N1)."""

from __future__ import annotations

from typing import Any, ClassVar

from engram.scope import Scope
from engram.solve.temporal import TemporalSolver
from engram.tools.base import ToolResult


class SolveTemporalTool:
    name: ClassVar[str] = "solve_temporal"
    description: ClassVar[str] = (
        "Solve a structured temporal question against the user's event calendar. "
        "Best for 'how many X before/after Y', 'how long since X', "
        "'when did X relative to Y' style questions."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self, solver: TemporalSolver, scope: Scope) -> None:
        self._solver = solver
        self._scope = scope

    async def __call__(self, query: str) -> ToolResult:
        tq = await self._solver.parse(query)
        if tq is None:
            return ToolResult(content="(could not solve as a structured query)")
        res = await self._solver.solve(tq, self._scope)
        if res is None:
            return ToolResult(content="(no events found to satisfy this query)")
        return ToolResult(content=str(res.answer), raw=res.answer)
