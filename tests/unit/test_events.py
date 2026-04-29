"""Phase 11a: SVO event calendar — storage + extractor tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from engram.errors import ExtractionError
from engram.extract.event_extractor import EventExtractor
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Event
from engram.scope import Scope
from engram.store.sqlite import SqliteStore


def _now() -> datetime:
    return datetime(2024, 3, 12, tzinfo=UTC)


SCOPE = Scope(org_id="acme", user_id="alice")


# ── Event store CRUD ────────────────────────────────────────────────


async def test_event_upsert_then_get() -> None:
    store = await SqliteStore.open(":memory:")
    try:
        e = Event(
            scope=SCOPE,
            subject_canonical="user",
            verb="visited",
            object_canonical="paris",
            time_start=_now(),
        )
        await store.upsert_event(e)
        got = await store.get_event(e.id, SCOPE)
        assert got is not None
        assert got.subject_canonical == "user"
        assert got.verb == "visited"
        assert got.object_canonical == "paris"
    finally:
        await store.close()


async def test_event_get_respects_scope() -> None:
    store = await SqliteStore.open(":memory:")
    try:
        e = Event(
            scope=SCOPE,
            subject_canonical="user",
            verb="went",
            object_canonical="x",
            time_start=_now(),
        )
        await store.upsert_event(e)
        other = Scope(org_id="acme", user_id="bob")
        assert await store.get_event(e.id, other) is None
    finally:
        await store.close()


async def test_event_search_keyword() -> None:
    store = await SqliteStore.open(":memory:")
    try:
        await store.upsert_event(
            Event(
                scope=SCOPE,
                subject_canonical="user",
                verb="visited",
                object_canonical="paris",
                time_start=_now(),
            )
        )
        await store.upsert_event(
            Event(
                scope=SCOPE,
                subject_canonical="user",
                verb="bought",
                object_canonical="book",
                time_start=_now(),
            )
        )
        hits = await store.search_events("paris", SCOPE)
        assert len(hits) == 1
        assert hits[0].object_canonical == "paris"
    finally:
        await store.close()


async def test_event_search_time_window() -> None:
    store = await SqliteStore.open(":memory:")
    try:
        recent = Event(
            scope=SCOPE,
            subject_canonical="user",
            verb="visited",
            object_canonical="rome",
            time_start=_now(),
        )
        old = Event(
            scope=SCOPE,
            subject_canonical="user",
            verb="visited",
            object_canonical="rome",
            time_start=_now() - timedelta(days=400),
        )
        await store.upsert_event(recent)
        await store.upsert_event(old)
        hits = await store.search_events("rome", SCOPE, time_start=_now() - timedelta(days=30))
        assert len(hits) == 1
        assert hits[0].id == recent.id
    finally:
        await store.close()


async def test_event_search_aliases_match() -> None:
    """An alias should be discoverable via FTS."""
    store = await SqliteStore.open(":memory:")
    try:
        await store.upsert_event(
            Event(
                scope=SCOPE,
                subject_canonical="user",
                verb="had",
                object_canonical="dinner",
                time_start=_now(),
                aliases=["family dinner", "dinner with mom"],
            )
        )
        hits = await store.search_events("family", SCOPE)
        assert len(hits) == 1
    finally:
        await store.close()


async def test_event_no_query_lists_by_time() -> None:
    store = await SqliteStore.open(":memory:")
    try:
        e1 = Event(
            scope=SCOPE,
            subject_canonical="user",
            verb="a",
            object_canonical="x",
            time_start=_now() - timedelta(days=2),
        )
        e2 = Event(
            scope=SCOPE, subject_canonical="user", verb="b", object_canonical="y", time_start=_now()
        )
        await store.upsert_event(e1)
        await store.upsert_event(e2)
        hits = await store.search_events("", SCOPE)
        # Most recent first
        assert hits[0].id == e2.id
        assert hits[1].id == e1.id
    finally:
        await store.close()


# ── EventExtractor (mock LLM) ───────────────────────────────────────


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(content=self._response, finish_reason="stop")


def _msg(text: str) -> ChatMessage:
    return ChatMessage(scope=SCOPE, session_id="s1", role="user", content=text, timestamp=_now())


async def test_event_extractor_parses_well_formed_events() -> None:
    response = json.dumps(
        {
            "events": [
                {
                    "subject_canonical": "user",
                    "verb": "visited",
                    "object_canonical": "paris",
                    "time_start": "2024-03-05T00:00:00+00:00",
                    "time_end": None,
                    "aliases": ["trip to paris", "paris vacation"],
                    "confidence": 0.92,
                }
            ]
        }
    )
    extractor = EventExtractor(_FakeLLM(response))
    events = await extractor.extract([_msg("I went to Paris last Tuesday.")], session_date=_now())
    assert len(events) == 1
    assert events[0].verb == "visited"
    assert events[0].object_canonical == "paris"
    assert "trip to paris" in events[0].aliases


async def test_event_extractor_strips_markdown_fences() -> None:
    response = '```json\n{"events": []}\n```'
    extractor = EventExtractor(_FakeLLM(response))
    out = await extractor.extract([_msg("nothing here")], session_date=_now())
    assert out == []


async def test_event_extractor_drops_malformed_entries() -> None:
    response = json.dumps(
        {
            "events": [
                {
                    "subject_canonical": "user",
                    "verb": "ate",
                    "object_canonical": "pizza",
                    "time_start": "2024-03-05T00:00:00+00:00",
                    "confidence": 1.0,
                },
                {
                    # missing time_start -> should be dropped
                    "subject_canonical": "user",
                    "verb": "ran",
                    "object_canonical": "marathon",
                    "confidence": 1.0,
                },
                {
                    "subject_canonical": "user",
                    "verb": "watched",
                    "object_canonical": "movie",
                    "time_start": "2024-03-04T12:00:00+00:00",
                    "confidence": 0.8,
                },
            ]
        }
    )
    extractor = EventExtractor(_FakeLLM(response))
    out = await extractor.extract([_msg("...")])
    assert len(out) == 2
    assert {e.verb for e in out} == {"ate", "watched"}


async def test_event_extractor_invalid_json_raises() -> None:
    extractor = EventExtractor(_FakeLLM("not json"), max_retries=0)
    with pytest.raises(ExtractionError):
        await extractor.extract([_msg("...")])


async def test_event_extractor_empty_messages() -> None:
    extractor = EventExtractor(_FakeLLM('{"events": []}'))
    assert await extractor.extract([]) == []
