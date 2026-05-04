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
        # Fail-closed: if the SVO event calendar is empty for this scope,
        # we cannot answer ANY temporal query. Returning 0 from a count would
        # be a confident-but-wrong answer that the Reader trusts as final.
        # Cheap probe — limit=1 short-circuits.
        any_events = await self._store.search_events("", scope, limit=1)
        if not any_events:
            return None

        # Resolve anchor → time bound
        anchor_dt: datetime | None = None
        if q.anchor_event:
            anchor_events = await self._store.search_events(
                q.anchor_event, scope, limit=5
            )
            # Sanity check: store can fall back to "list all" when the FTS query
            # is filtered out by sanitisation (e.g. all-hyphen tokens). Require at
            # least one shared word between the requested anchor name and the
            # candidate event's subject/verb/object/aliases — otherwise treat as
            # unknown anchor.
            anchor_words = {
                w.lower() for w in q.anchor_event.replace("-", " ").split() if len(w) > 2
            }
            anchor_events = [
                e
                for e in anchor_events
                if anchor_words
                & {
                    w.lower()
                    for piece in (e.subject_canonical, e.verb, e.object_canonical, *e.aliases)
                    for w in piece.replace("-", " ").split()
                }
            ]
            if not anchor_events:
                return None  # unknown anchor → fall through to LLM
            # Highest-confidence match wins; ties broken by earliest time.
            anchor_dt = max(
                anchor_events, key=lambda e: (e.confidence, -e.time_start.timestamp())
            ).time_start

        if q.op == "count":
            # Need at least a verb or object to filter on
            if not (q.verb or q.object):
                return None
            # FTS retrieval — fetch a wide candidate set, then post-filter in Python.
            # We do NOT push the anchor bound into the store's `time_end` parameter
            # because that filters by the event's own time_end (which is NULL for
            # instantaneous events) — semantically different from "events before X".
            query_text = " ".join(filter(None, [q.verb, q.object]))
            events = await self._store.search_events(query_text, scope, limit=200)
            # Strict verb/object match — FTS gives candidates, exact match enforces semantics.
            if q.verb:
                events = [e for e in events if e.verb.lower() == q.verb.lower()]
            if q.object:
                obj_lc = q.object.lower()
                events = [e for e in events if obj_lc in e.object_canonical.lower()]
            # Anchor bound + don't count the anchor itself.
            if anchor_dt:
                events = [e for e in events if e.time_start != anchor_dt]
                if q.bound == "before":
                    events = [e for e in events if e.time_start < anchor_dt]
                elif q.bound == "after":
                    events = [e for e in events if e.time_start > anchor_dt]
            # Optional explicit window
            if q.window:
                lo, hi = q.window
                events = [e for e in events if lo <= e.time_start <= hi]
            # Defer to the LLM when we'd answer 0 — in LongMemEval-style questions
            # the asker has usually done the thing they're asking about, so 0
            # almost always means our SVO extraction missed the relevant events
            # (the canonical verb/object differed from what the parser produced).
            if len(events) == 0:
                return None
            return SolverResult(
                answer=len(events),
                confidence=0.95,
                evidence_event_ids=[e.id for e in events],
            )

        # Other ops fall through to LLM for now — parser still parses them
        # so downstream can route to richer solvers in a follow-up.
        return None
