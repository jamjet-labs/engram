"""Programmatic temporal query solver — deterministic SQL execution against the
SVO event calendar. Falls through (returns None) for questions that don't fit
the supported ops, letting the LLM reader handle them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from engram.llm.base import LLMClient
from engram.scope import Scope
from engram.store.base import EngramStore

Op = Literal["count", "duration", "ordering", "before_after", "elapsed"]


class TemporalQuery(BaseModel):
    op: Op
    subject: str | None = None
    verb: str | None = None
    object: str | None = None
    anchor_event: str | None = None
    bound: Literal["before", "after", "between"] | None = None
    window: tuple[datetime, datetime] | None = None


class SolverResult(BaseModel):
    answer: str | int | float
    confidence: float
    evidence_event_ids: list[UUID] = []


class TemporalSolver:
    """Parse + solve structured temporal queries.

    ``parse`` returns ``None`` when the question doesn't look like a structured
    temporal query. ``solve`` returns ``None`` when the parsed query lacks
    enough information (e.g. unknown anchor event). Both cases fall through
    to the LLM reader.
    """

    def __init__(self, store: EngramStore, llm: LLMClient) -> None:
        self._store = store
        self._llm = llm

    async def parse(
        self, question: str, today: datetime | None = None
    ) -> TemporalQuery | None:
        raise NotImplementedError("implemented in 3.2")

    async def solve(self, q: TemporalQuery, scope: Scope) -> SolverResult | None:
        raise NotImplementedError("implemented in 3.3")
