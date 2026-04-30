"""Programmatic temporal query solver — deterministic SQL execution against the
SVO event calendar. Falls through (returns None) for questions that don't fit
the supported ops, letting the LLM reader handle them.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ValidationError

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.scope import Scope
from engram.store.base import EngramStore

logger = logging.getLogger(__name__)

PARSE_PROMPT = """You are a query parser. The user's question MIGHT be a structured temporal query
about events in their personal history. If so, extract:

  op: one of "count", "duration", "ordering", "before_after", "elapsed",
      or null if not a temporal/structured query
  subject: who did the event (or null)
  verb: the action verb in lowercase infinitive (or null)
  object: the target (or null)
  anchor_event: a named event the question references as a reference point (or null)
  bound: "before" | "after" | "between" | null

Reply ONLY with strict JSON, no markdown, no explanation.
If the question isn't a structured temporal query, reply {{"op": null}}.

Today's date (for relative date resolution): {today}

Question: {question}"""

_VALID_OPS = ("count", "duration", "ordering", "before_after", "elapsed")

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
        today_str = today.date().isoformat() if today else "unknown"
        prompt = PARSE_PROMPT.format(today=today_str, question=question)
        try:
            resp = await self._llm.generate(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=200,
                json_mode=True,
            )
        except ExtractionError as e:
            logger.warning("temporal parse LLM failed: %s", e)
            return None

        try:
            data = json.loads(resp.content.strip())
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("temporal parse JSON malformed: %s", e)
            return None

        if not isinstance(data, dict) or data.get("op") is None:
            return None
        if data["op"] not in _VALID_OPS:
            return None

        try:
            return TemporalQuery(
                op=data["op"],
                subject=data.get("subject"),
                verb=data.get("verb"),
                object=data.get("object"),
                anchor_event=data.get("anchor_event"),
                bound=data.get("bound"),
            )
        except ValidationError as e:
            logger.warning("temporal parse model construction failed: %s", e)
            return None

    async def solve(self, q: TemporalQuery, scope: Scope) -> SolverResult | None:
        raise NotImplementedError("implemented in 3.3")
