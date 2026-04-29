"""SVO event extractor (Phase 11).

Pulls structured events from a chat segment. Each event is `(subject, verb,
object, time_start, time_end?, aliases)` — same shape Chronos uses, just with
JSON output for tolerant parsing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.models import ChatMessage, Event
from engram.scope import Scope

logger = logging.getLogger(__name__)


EVENT_EXTRACTION_SYSTEM_PROMPT = """\
You extract durable EVENTS from a conversation. An event is something that
happened (or is happening, or will happen) — distinct from a static fact.

For each event, output:
  - subject_canonical: the canonical entity that performed the event
  - verb: a single canonical verb (lowercase, infinitive form preferred)
  - object_canonical: the canonical target/object/destination
  - time_start: ISO 8601 datetime when the event began (or single point in time)
  - time_end: ISO 8601 datetime when the event ended; null if instantaneous or unknown
  - aliases: 2-4 surface variants the user/assistant might phrase this event as
  - confidence: float in [0,1]

Resolve relative dates against the session date when given. Use canonical entity
names (e.g. "the user" -> "user", "my brother" -> "user's brother", "she" -> the
named entity she refers to).

Output STRICTLY valid JSON of shape:
{"events": [{"subject_canonical": "...", "verb": "...", "object_canonical": "...",
             "time_start": "...", "time_end": null, "aliases": ["...","..."],
             "confidence": 0.9}, ...]}

If no events, return {"events": []}. No markdown. No explanation."""


def build_event_user_prompt(turns: list[dict[str, str]], session_date: str | None) -> str:
    header = []
    if session_date:
        header.append(
            f"Session date (treat as 'today' for relative time resolution): {session_date}"
        )
    header.append("Conversation:")
    body = "\n".join(f"[{t['role']}] {t['content']}" for t in turns)
    return "\n\n".join([*header, body, "Extract events now."])


class EventExtractor:
    """LLM-driven SVO event extractor with tolerant JSON parsing.

    Stays defensive: malformed entries are dropped with a warning rather than
    failing the whole batch.
    """

    def __init__(self, llm: LLMClient, max_retries: int = 1) -> None:
        self._llm = llm
        self._max_retries = max_retries

    async def extract(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
    ) -> list[Event]:
        if not messages:
            return []
        scope = messages[0].scope
        turns = [{"role": m.role, "content": m.content} for m in messages]
        user_prompt = build_event_user_prompt(
            turns,
            session_date.date().isoformat() if session_date else None,
        )
        prompt = [
            LLMMessage(role="system", content=EVENT_EXTRACTION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]
        last_err: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                resp = await self._llm.generate(prompt, temperature=0.0, json_mode=True)
                return _parse_events(resp.content, scope)
            except (ExtractionError, json.JSONDecodeError, ValueError) as e:
                last_err = e
                logger.warning("event extraction attempt failed: %s", e)
        raise ExtractionError(f"event extraction failed: {last_err}")


def _parse_events(content: str, scope: Scope) -> list[Event]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    raw_events = obj.get("events", [])
    if not isinstance(raw_events, list):
        return []
    out: list[Event] = []
    for entry in raw_events:
        if not isinstance(entry, dict):
            continue
        try:
            evt = Event(
                id=uuid4(),
                scope=scope,
                subject_canonical=entry.get("subject_canonical", ""),
                verb=entry.get("verb", ""),
                object_canonical=entry.get("object_canonical", ""),
                time_start=_parse_iso(entry.get("time_start")),
                time_end=_parse_iso(entry.get("time_end")) if entry.get("time_end") else None,
                confidence=float(entry.get("confidence", 1.0)),
                aliases=list(entry.get("aliases") or []),
            )
            out.append(evt)
        except (ValidationError, ValueError, TypeError) as e:
            logger.debug("dropped malformed event %s: %s", entry, e)
            continue
    return out


def _parse_iso(s: str | None) -> datetime:
    if s is None:
        raise ValueError("missing time_start")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
