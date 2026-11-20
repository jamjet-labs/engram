"""Rules every store keeps: text, JSON and limits, ids and scopes, and time.

These run on every backend through the `open_store` and `open_memory`
fixtures. A store refuses bad input in its own write methods, because a caller
can reach a store without the facade. The facade refuses it too, before any
embedder or LLM call.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from engram.embedding.base import EmbedderIdentity
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import ScopeError
from engram.models import ChatMessage, Event, Fact
from engram.retrieve.hybrid import HybridRetriever
from engram.scope import Scope
from engram.validate import MAX_FACT_TEXT_BYTES, MAX_METADATA_BYTES, MAX_STORE_LIMIT, MAX_TOP_K
from engram.vector.hnsw import HnswVectorStore
from tests.conformance.backends import OpenMemory, OpenStore

ACME_ALICE = Scope(org_id="acme", user_id="alice")
ACME_BOB = Scope(org_id="acme", user_id="bob")
WHO = {"user_id": "alice", "org_id": "acme"}

NOW = datetime(2026, 3, 4, 10, 30, tzinfo=UTC)


def _fact(text: str = "Alice keeps her bike in the blue shed.", **kw: Any) -> Fact:
    kw.setdefault("scope", ACME_ALICE)
    kw.setdefault("valid_from", NOW)
    return Fact(text=text, **kw)


def _message(content: str = "I bought a new helmet today.", **kw: Any) -> ChatMessage:
    kw.setdefault("scope", ACME_ALICE)
    kw.setdefault("session_id", "s1")
    kw.setdefault("timestamp", NOW)
    return ChatMessage(role="user", content=content, **kw)


def _event(**kw: Any) -> Event:
    kw.setdefault("scope", ACME_ALICE)
    kw.setdefault("subject_canonical", "alice")
    kw.setdefault("time_start", NOW)
    return Event(verb="painted", object_canonical="shed", **kw)


# ── Text ────────────────────────────────────────────────────────────────

_UNICODE_TEXTS = {
    "emoji with zero-width joiner": (
        "Alice is a \U0001f469\u200d\U0001f4bb and a \U0001f468\u200d\U0001f469\u200d\U0001f467"
    ),
    "right to left": "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd and \u0645\u0631",
    "combining marks": "Cafe\u0301 and u\u0308ber and n\u0303 stay as typed",
    "cjk": "\u4f60\u597d\u4e16\u754c\u3001\u3053\u3093\u306b\u3061\u306f\u3001\uc548\ub155",
    "line separator": "first line\u2028second line\u2029third",
    "four byte at both ends": "\U0001d49c the middle \U0001d49c",
}


@pytest.mark.parametrize("text", list(_UNICODE_TEXTS.values()), ids=list(_UNICODE_TEXTS))
async def test_text_round_trips_unicode_exactly(open_store: OpenStore, text: str) -> None:
    store = await open_store()
    fact = _fact(text)
    message = _message(text)
    event = _event(subject_canonical=text, aliases=[text])
    await store.upsert_fact(fact)
    await store.upsert_message(message)
    await store.upsert_event(event)

    got = await store.get_fact(fact.id, ACME_ALICE)
    assert got is not None
    assert got.text == text
    messages = await store.list_messages("s1", ACME_ALICE)
    assert [m.content for m in messages] == [text]
    got_event = await store.get_event(event.id, ACME_ALICE)
    assert got_event is not None
    assert got_event.subject_canonical == text
    assert got_event.aliases == [text]


_BAD_FIELDS: dict[str, Callable[[str], Any]] = {
    "fact text": lambda bad: _fact(f"Alice keeps {bad} a bike."),
    "fact session id": lambda bad: _fact(session_id=f"s{bad}1"),
    "fact metadata key": lambda bad: _fact(metadata={f"k{bad}": "v"}),
    "fact metadata value": lambda bad: _fact(metadata={"k": [f"v{bad}"]}),
    "message content": lambda bad: _message(f"I bought {bad} a helmet."),
    "message session id": lambda bad: _message(session_id=f"s{bad}1"),
    "message metadata value": lambda bad: _message(metadata={"k": f"v{bad}"}),
    "event subject": lambda bad: _event(subject_canonical=f"al{bad}ice"),
    "event alias": lambda bad: _event(aliases=["ok", f"al{bad}ias"]),
    "event metadata key": lambda bad: _event(metadata={f"k{bad}": 1}),
}


async def _write(store: Any, item: Fact | ChatMessage | Event) -> None:
    if isinstance(item, Fact):
        await store.upsert_fact(item)
    elif isinstance(item, ChatMessage):
        await store.upsert_message(item)
    else:
        await store.upsert_event(item)


async def _stored(store: Any, item: Fact | ChatMessage | Event) -> bool:
    if isinstance(item, Fact):
        return await store.get_fact(item.id, ACME_ALICE) is not None
    if isinstance(item, ChatMessage):
        return bool(await store.list_messages(item.session_id, ACME_ALICE))
    return await store.get_event(item.id, ACME_ALICE) is not None


@pytest.mark.parametrize("make", list(_BAD_FIELDS.values()), ids=list(_BAD_FIELDS))
async def test_nul_is_refused_and_nothing_is_stored(
    open_store: OpenStore, make: Callable[[str], Any]
) -> None:
    store = await open_store()
    item = make("\x00")
    with pytest.raises(ValueError, match="NUL"):
        await _write(store, item)
    assert not await _stored(store, item)


@pytest.mark.parametrize("make", list(_BAD_FIELDS.values()), ids=list(_BAD_FIELDS))
async def test_lone_surrogate_is_refused(open_store: OpenStore, make: Callable[[str], Any]) -> None:
    store = await open_store()
    try:
        item = make("\ud800")
    except ValidationError:
        return  # the model refuses it before a store sees it (a field with a length rule)
    with pytest.raises(ValueError, match="surrogate"):
        await _write(store, item)
    assert not await _stored(store, item)


async def test_blank_fact_text_is_refused_by_the_facade(open_memory: OpenMemory) -> None:
    memory = await open_memory()
    for blank in ("", "   ", "\n\t "):
        with pytest.raises(ValueError, match="text"):
            await memory.record(blank, **WHO)
    assert await memory.recall("anything", **WHO) == []


async def test_the_facade_refuses_bad_text_before_it_embeds(open_memory: OpenMemory) -> None:
    embedder = _SpyEmbedder()
    memory = await open_memory(embedder=embedder)
    with pytest.raises(ValueError, match="NUL"):
        await memory.record("Alice keeps\x00 a bike.", **WHO)
    with pytest.raises(ValueError, match="surrogate"):
        await memory.record("Alice keeps \ud800 a bike.", **WHO)
    with pytest.raises(ValueError, match="NUL"):
        await memory.record_message("I bought\x00 a helmet.", **WHO)
    with pytest.raises(ValueError, match="metadata"):
        await memory.record("Alice keeps a bike.", metadata={"k": float("nan")}, **WHO)
    assert embedder.calls == []


@pytest.mark.parametrize("field", ["reason", "actor"])
@pytest.mark.parametrize("call", ["revoke", "reinstate", "erase"])
async def test_nul_in_a_reason_or_actor_is_refused_and_nothing_changes(
    open_store: OpenStore, call: str, field: str
) -> None:
    store = await open_store()
    fact = _fact()
    await store.upsert_fact(fact)
    kwargs = {"reason": "a request", "actor": "a reviewer", field: "bad\x00value"}
    with pytest.raises(ValueError, match=field):
        await getattr(store, call)(fact.id, ACME_ALICE, **kwargs)
    assert await store.get_fact(fact.id, ACME_ALICE) is not None
    assert await store.list_revocations(ACME_ALICE, include_reinstated=True) == []
    assert await store.list_erasures(ACME_ALICE) == []


@pytest.mark.parametrize("field", ["reason", "actor"])
async def test_nul_in_a_scope_erasure_reason_or_actor_is_refused_and_nothing_changes(
    open_store: OpenStore, field: str
) -> None:
    store = await open_store()
    await store.upsert_message(_message())
    kwargs = {"reason": "a request", "actor": "a reviewer", field: "bad\x00value"}
    with pytest.raises(ValueError, match=field):
        await store.erase_scope_residue(ACME_ALICE, **kwargs)
    assert len(await store.list_messages("s1", ACME_ALICE)) == 1
    assert await store.list_erasures(ACME_ALICE) == []


def _text_of_bytes(size: int) -> str:
    # A 2-byte character ends the text, so a text one byte over the limit
    # puts that character across the boundary.
    return "a" * (size - 2) + "\u00e9"


async def test_fact_text_at_the_byte_limit_is_stored_and_one_byte_over_is_refused(
    open_store: OpenStore,
) -> None:
    store = await open_store()
    at_limit = _text_of_bytes(MAX_FACT_TEXT_BYTES)
    assert len(at_limit.encode("utf-8")) == MAX_FACT_TEXT_BYTES
    fact = _fact(at_limit)
    await store.upsert_fact(fact)
    got = await store.get_fact(fact.id, ACME_ALICE)
    assert got is not None
    assert got.text == at_limit

    over = _text_of_bytes(MAX_FACT_TEXT_BYTES + 1)
    assert len(over.encode("utf-8")) == MAX_FACT_TEXT_BYTES + 1
    assert len(over) == MAX_FACT_TEXT_BYTES  # as many characters as the limit has bytes
    too_long = _fact(over)
    with pytest.raises(ValueError, match="limit"):
        await store.upsert_fact(too_long)
    assert await store.get_fact(too_long.id, ACME_ALICE) is None


async def test_metadata_at_the_size_limit_is_stored_and_over_is_refused(
    open_store: OpenStore,
) -> None:
    store = await open_store()
    overhead = len('{"k": ""}')
    ok = _fact(metadata={"k": "x" * (MAX_METADATA_BYTES - overhead)})
    await store.upsert_fact(ok)
    assert await store.get_fact(ok.id, ACME_ALICE) is not None
    over = _fact(metadata={"k": "x" * (MAX_METADATA_BYTES - overhead + 1)})
    with pytest.raises(ValueError, match="limit"):
        await store.upsert_fact(over)
    assert await store.get_fact(over.id, ACME_ALICE) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=str)
async def test_metadata_nan_and_infinity_are_refused(open_store: OpenStore, bad: float) -> None:
    store = await open_store()
    for item in (
        _fact(metadata={"score": bad}),
        _message(metadata={"nested": {"scores": [1.0, bad]}}),
        _event(metadata={"score": bad}),
    ):
        with pytest.raises(ValueError, match="metadata"):
            await _write(store, item)
        assert not await _stored(store, item)


# ── Limits ──────────────────────────────────────────────────────────────


class _SpyEmbedder:
    """Synthetic vectors, and a record of every batch it was asked to embed."""

    def __init__(self) -> None:
        self._inner = SyntheticEmbedding(dim=384)
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def identity(self) -> EmbedderIdentity:
        return self._inner.identity

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await self._inner.embed(texts)


class _SpyStore:
    """Passes every call to a real store and records its name."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return attr(*args, **kwargs)

        return call


async def test_recall_top_k_zero_returns_empty_without_store_calls(open_store: OpenStore) -> None:
    store = await open_store()
    await store.upsert_fact(_fact())
    spy = _SpyStore(store)
    embedder = _SpyEmbedder()
    retriever = HybridRetriever(spy, HnswVectorStore(dim=embedder.dim), embedder)  # type: ignore[arg-type]
    assert await retriever.search("blue shed", ACME_ALICE, top_k=0) == []
    assert spy.calls == []
    assert embedder.calls == []
    # The same query with a real top_k does reach the store and finds the fact.
    assert len(await retriever.search("blue shed", ACME_ALICE, top_k=3)) == 1
    assert spy.calls != []


@pytest.mark.parametrize("top_k", [MAX_TOP_K + 1, -1, True, False, 2.5], ids=repr)
async def test_recall_top_k_over_max_or_negative_or_bool_is_refused(
    open_memory: OpenMemory, top_k: Any
) -> None:
    memory = await open_memory()
    await memory.record("Alice keeps her bike in the blue shed.", **WHO)
    with pytest.raises(ValueError, match="top_k"):
        await memory.recall("blue shed", top_k=top_k, **WHO)


async def test_recall_top_k_at_max_is_served(open_memory: OpenMemory) -> None:
    # The retriever asks the store for several times top_k candidates. That
    # must not trip the store's own limit.
    memory = await open_memory()
    fact = await memory.record("Alice keeps her bike in the blue shed.", **WHO)
    hits = await memory.recall("blue shed", top_k=MAX_TOP_K, **WHO)
    assert [h.fact.id for h in hits] == [fact.id]


async def test_keyword_search_limit_zero_returns_empty(open_store: OpenStore) -> None:
    store = await open_store()
    await store.upsert_fact(_fact(session_id="s1"))
    await store.upsert_message(_message())
    await store.upsert_event(_event())
    assert [f.text for f in await store.keyword_search("blue shed", ACME_ALICE, limit=1)]
    assert await store.keyword_search("blue shed", ACME_ALICE, limit=0) == []
    assert await store.list_facts_by_session("s1", ACME_ALICE, limit=1)
    assert await store.list_facts_by_session("s1", ACME_ALICE, limit=0) == []
    assert await store.list_messages("s1", ACME_ALICE, limit=1)
    assert await store.list_messages("s1", ACME_ALICE, limit=0) == []
    assert await store.search_events("alice", ACME_ALICE, limit=1)
    assert await store.search_events("alice", ACME_ALICE, limit=0) == []


@pytest.mark.parametrize("limit", [-1, MAX_STORE_LIMIT + 1, True], ids=repr)
async def test_a_store_limit_that_is_negative_or_too_large_is_refused(
    open_store: OpenStore, limit: Any
) -> None:
    store = await open_store()
    with pytest.raises(ValueError, match="limit"):
        await store.keyword_search("blue shed", ACME_ALICE, limit=limit)


# ── Ids and scopes ──────────────────────────────────────────────────────


async def test_same_fact_id_in_two_scopes_is_refused(open_store: OpenStore) -> None:
    store = await open_store()
    mine = _fact("Alice keeps her bike in the blue shed.")
    await store.upsert_fact(mine)
    theirs = _fact("Bob keeps his bike in the red garage.", id=mine.id, scope=ACME_BOB)
    with pytest.raises(ScopeError):
        await store.upsert_fact(theirs)
    kept = await store.get_fact(mine.id, ACME_ALICE)
    assert kept is not None
    assert kept.text == mine.text
    assert await store.get_fact(mine.id, ACME_BOB) is None
    assert await store.keyword_search("red garage", ACME_BOB) == []


async def test_same_fact_id_twice_in_one_scope_updates(open_store: OpenStore) -> None:
    store = await open_store()
    first = _fact("Alice keeps her bike in the blue shed.")
    await store.upsert_fact(first)
    second = _fact("Alice moved her bike to the red garage.", id=first.id)
    await store.upsert_fact(second)
    got = await store.get_fact(first.id, ACME_ALICE)
    assert got is not None
    assert got.text == second.text
    assert [f.id for f in await store.keyword_search("garage", ACME_ALICE)] == [first.id]
    assert await store.keyword_search("shed", ACME_ALICE) == []


async def test_same_message_id_in_two_scopes_is_refused(open_store: OpenStore) -> None:
    store = await open_store()
    mine = _message("I bought a new helmet today.")
    await store.upsert_message(mine)
    theirs = _message("Bob sold a helmet today.", id=mine.id, scope=ACME_BOB)
    with pytest.raises(ScopeError):
        await store.upsert_message(theirs)
    assert [m.content for m in await store.list_messages("s1", ACME_ALICE)] == [mine.content]
    assert await store.list_messages("s1", ACME_BOB) == []


# ── Time ────────────────────────────────────────────────────────────────


async def test_naive_and_aware_times_round_trip_to_the_same_instant(open_store: OpenStore) -> None:
    store = await open_store()
    naive = datetime(2026, 3, 4, 10, 30, 15)
    utc = naive.replace(tzinfo=UTC)
    plus_five = utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

    for stamp in (naive, utc, plus_five):
        fact = _fact(valid_from=stamp, event_date=stamp)
        message = _message(timestamp=stamp, session_id=f"s-{uuid4().hex[:6]}")
        event = _event(time_start=stamp, time_end=stamp)
        await store.upsert_fact(fact)
        await store.upsert_message(message)
        await store.upsert_event(event)

        got = await store.get_fact(fact.id, ACME_ALICE)
        assert got is not None
        assert got.valid_from == utc
        assert got.event_date == utc
        [got_message] = await store.list_messages(message.session_id, ACME_ALICE)
        assert got_message.timestamp == utc
        got_event = await store.get_event(event.id, ACME_ALICE)
        assert got_event is not None
        assert got_event.time_start == utc
        assert got_event.time_end == utc


async def test_a_naive_time_window_finds_events_stored_with_aware_times(
    open_store: OpenStore,
) -> None:
    store = await open_store()
    event = _event(time_start=NOW)
    await store.upsert_event(event)
    naive_start = NOW.replace(tzinfo=None) - timedelta(hours=1)
    naive_end = NOW.replace(tzinfo=None) + timedelta(hours=1)
    found = await store.search_events(
        "alice", ACME_ALICE, time_start=naive_start, time_end=naive_end
    )
    assert [e.id for e in found] == [event.id]
