"""ScopedEngram: a handle on one scope that cannot reach another.

`Engram.scoped(org_id=..., user_id=...)` checks the scope once. The handle's
methods take no scope. A fact id from another scope reads exactly like an
unknown id. A `Fact` or a message from another scope is refused with
ScopeError. Every fact, event, message, revocation and erasure record the
store hands back is checked against the handle's scope, so a store that
returned another scope's data fails with ScopeError instead of leaking it.
The root methods run on a handle, so the mixed-scope refusals hold there too.

Every test that needs memory takes `open_memory` or `open_store`, so it runs
on every backend. The file is not in tests/conformance/FILES: the tests that
put a fake store in front of the backend's store pass it to
`Engram.open(store=...)`, which the FILES guard does not allow.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from engram import Engram, ScopedEngram
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import ReadOnlyStoreError, ScopeError, ScopeRequiredError
from engram.extract.event_extractor import EventExtractor
from engram.governance import ErasureRecord, Provenance, Revocation
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Event, Fact
from engram.scope import Scope
from engram.store.base import EngramStore
from tests.conformance.backends import OpenMemory, OpenStore

ALICE = Scope(org_id="acme", user_id="alice")
BOB = Scope(org_id="acme", user_id="bob")

BIKE = "Alice keeps her bike in the blue shed."
HELMET = "Alice bought a red helmet for the bike."
KEY = "Alice hides the spare key under the doormat."
BOB_BIKE = "Bob keeps his bike at the office."
BOB_KEY = "Bob hides his spare key in the mailbox."
BOB_CAR = "Bob parks his car behind the office."

_EXTRACTED = (
    '{"facts": [{"text": "The user keeps a bike in the shed.", "confidence": 0.9}],'
    ' "events": [{"subject_canonical": "user", "verb": "bought",'
    ' "object_canonical": "bike", "time_start": "2026-01-05T00:00:00+00:00",'
    ' "time_end": null, "aliases": [], "confidence": 0.9}]}'
)


def _now() -> datetime:
    return datetime.now(UTC)


def _turn(scope: Scope, text: str = "I keep my bike in the shed.") -> ChatMessage:
    return ChatMessage(scope=scope, session_id="s1", role="user", content=text, timestamp=_now())


def _event(scope: Scope) -> Event:
    return Event(
        scope=scope,
        subject_canonical="user",
        verb="painted",
        object_canonical="shed",
        time_start=_now(),
    )


class SpyEmbedder(SyntheticEmbedding):
    """Synthetic vectors, and a record of every batch it was asked to embed."""

    def __init__(self) -> None:
        super().__init__(dim=384)
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await super().embed(texts)


class SpyLLM:
    """Answers every prompt with one fact and one event, and counts the calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=_EXTRACTED, finish_reason="stop")


def _handles(memory: Engram) -> tuple[ScopedEngram, ScopedEngram]:
    return (
        memory.scoped(org_id=ALICE.org_id, user_id=ALICE.user_id),
        memory.scoped(org_id=BOB.org_id, user_id=BOB.user_id),
    )


async def _state(handle: ScopedEngram) -> tuple[object, ...]:
    """What a refused call must leave unchanged in a scope."""
    revocations = await handle.revocations(include_reinstated=True)
    return (
        await handle.fact_counts(),
        sorted((str(r.fact_id), r.reinstated_at is None) for r in revocations),
        len(await handle.erasures()),
        len(await handle.search_events("")),
        await handle.list_conversations(),
    )


async def _outcome(call: Callable[[], Awaitable[object]], fact_id: UUID) -> tuple[str, str]:
    """The error type and message, or the value, with the id written as <id>."""
    try:
        result = await call()
    except Exception as e:
        return type(e).__name__, str(e).replace(str(fact_id), "<id>")
    return "returned", repr(result)


# ── The handle's surface ───────────────────────────────────────────────

# An allowlist: a new public member fails the test until it is added here.
HANDLE_MEMBERS = {
    "scope",
    "record",
    "record_message",
    "extract",
    "extract_events",
    "recall",
    "context_entries",
    "context",
    "search_events",
    "supersede",
    "revoke",
    "reinstate",
    "erase",
    "forget",
    "explain",
    "revocations",
    "erasures",
    "list_messages",
    "list_conversations",
    "delete_conversation",
    "fact_counts",
    "erase_all",
}

# Calls that reach more than one scope. They stay on the root Engram.
CROSS_SCOPE = {"forget_in_org", "fact_counts_in_org", "reembed"}


def _public_functions() -> dict[str, Callable[..., Any]]:
    return {
        name: member
        for name, member in inspect.getmembers(ScopedEngram, inspect.isfunction)
        if not name.startswith("_")
    }


def test_scoped_methods_take_no_scope_parameters() -> None:
    methods = _public_functions()
    assert set(methods) == HANDLE_MEMBERS - {"scope"}
    for name, method in methods.items():
        named = set(inspect.signature(method).parameters) & {"user_id", "org_id", "scope"}
        assert named == set(), f"ScopedEngram.{name} takes {named}"
    # Engram.scoped names both ids, by keyword.
    params = inspect.signature(Engram.scoped).parameters
    assert [(p.name, p.kind) for p in params.values() if p.name != "self"] == [
        ("org_id", inspect.Parameter.KEYWORD_ONLY),
        ("user_id", inspect.Parameter.KEYWORD_ONLY),
    ]


async def test_scoped_handle_exposes_only_the_allowlisted_methods(
    open_memory: OpenMemory,
) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice, _ = _handles(memory)
    public = {name for name in dir(alice) if not name.startswith("_")}
    assert public == HANDLE_MEMBERS
    assert {name for name in dir(ScopedEngram) if not name.startswith("_")} == HANDLE_MEMBERS
    assert public & CROSS_SCOPE == set()
    assert alice.scope == ALICE
    with pytest.raises(AttributeError):
        alice.scope = BOB  # type: ignore[misc]
    assert alice.scope == ALICE


async def test_scoped_checks_the_scope_by_the_root_rules(open_memory: OpenMemory) -> None:
    strict = await open_memory(allow_default_scope=False)
    refused: list[dict[str, Any]] = [
        {"org_id": "default", "user_id": "alice"},
        {"org_id": "acme", "user_id": "default"},
        {"org_id": "acme", "user_id": ""},
        {"org_id": " ", "user_id": "alice"},
        {"org_id": None, "user_id": "alice"},
        {"org_id": "acme", "user_id": None},
        {"org_id": b"acme", "user_id": "alice"},
    ]
    for ids in refused:
        with pytest.raises(ScopeRequiredError):
            strict.scoped(**ids)
    with pytest.raises(ValueError, match="control character"):
        strict.scoped(org_id="acme", user_id="ali\x00ce")

    allowed = await open_memory()  # allow_default_scope=True
    assert allowed.scoped(org_id="default", user_id="default").scope == Scope()
    # A handle always names its scope: None does not mean "default" here.
    for ids in ({"org_id": None, "user_id": None}, {"org_id": "", "user_id": "alice"}):
        with pytest.raises(ScopeRequiredError):
            allowed.scoped(**ids)


async def test_each_handle_reads_and_writes_only_its_own_scope(open_memory: OpenMemory) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice, bob = _handles(memory)
    mine = await alice.record(BIKE, session_id="s1")
    theirs = await bob.record(BOB_BIKE, session_id="s1")
    await alice.record_message("Where is my bike?", session_id="s1")
    await bob.record_message("Where is my bike?", session_id="s2")

    assert [h.fact.id for h in await alice.recall("bike")] == [mine.id]
    assert [h.fact.id for h in await bob.recall("bike")] == [theirs.id]
    assert [e.fact.id for e in await alice.context_entries("bike")] == [mine.id]
    assert await alice.context("bike") == f"- {BIKE}"
    assert await alice.list_conversations() == ["s1"]
    assert await bob.list_conversations() == ["s2"]
    assert await alice.list_messages("s2") == []
    assert await alice.fact_counts() == (1, 1)
    assert await bob.fact_counts() == (1, 1)

    # forget, revoke and erase_all act on the handle's scope only.
    assert await alice.forget(theirs.id, reason="wrong", actor="test") is False
    assert await alice.forget(mine.id, reason="wrong", actor="test") is True
    assert await alice.fact_counts() == (1, 0)
    assert await bob.fact_counts() == (1, 1)
    assert await alice.erase_all(reason="asked to", actor="test") == 1
    assert await alice.fact_counts() == (0, 0)
    assert await alice.list_conversations() == []
    assert await bob.fact_counts() == (1, 1)
    assert await bob.list_conversations() == ["s2"]
    # The root reads the same scopes through its ids.
    found = await memory.recall("bike", org_id="acme", user_id="bob")
    assert [h.fact.id for h in found] == [theirs.id]


# ── Facts and fact ids from another scope ──────────────────────────────


@pytest.mark.parametrize("method", ["revoke", "reinstate", "erase", "supersede"])
async def test_fact_from_another_scope_is_refused(open_memory: OpenMemory, method: str) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice, bob = _handles(memory)
    mine = await alice.record(BIKE)
    theirs = await bob.record(BOB_BIKE)
    their_revoked = await bob.record(BOB_KEY)
    await bob.revoke(their_revoked, reason="no longer true", actor="test")
    before = (await _state(alice), await _state(bob))

    calls: dict[str, Callable[[], Awaitable[object]]] = {
        "revoke": lambda: alice.revoke(theirs, reason="wrong", actor="test"),
        "reinstate": lambda: alice.reinstate(their_revoked, reason="right", actor="test"),
        "erase": lambda: alice.erase(theirs, reason="asked to", actor="test"),
        # supersede takes ids, and another scope's id reads like an unknown one.
        "supersede": lambda: alice.supersede(theirs.id, mine.id),
    }
    expected = ValueError if method == "supersede" else ScopeError
    with pytest.raises(expected):
        await calls[method]()
    assert (await _state(alice), await _state(bob)) == before


IdCall = Callable[[ScopedEngram, UUID, Fact], Awaitable[object]]

ID_CALLS: dict[str, IdCall] = {
    "revoke": lambda h, fid, mine: h.revoke(fid, reason="wrong", actor="test"),
    "reinstate": lambda h, fid, mine: h.reinstate(fid, reason="right", actor="test"),
    "erase": lambda h, fid, mine: h.erase(fid, reason="asked to", actor="test"),
    "supersede": lambda h, fid, mine: h.supersede(fid, mine.id),
    "explain": lambda h, fid, mine: h.explain(fid),
    "forget": lambda h, fid, mine: h.forget(fid, reason="wrong", actor="test"),
}

NOT_FOUND = {
    "revoke": "NotFoundError",
    "reinstate": "NotFoundError",
    "erase": "NotFoundError",
    "supersede": "ValueError",  # as the root's supersede has always raised
    "explain": "returned",
    "forget": "returned",
}


@pytest.mark.parametrize("method", sorted(ID_CALLS))
async def test_fact_id_from_another_scope_reads_as_not_found(
    open_memory: OpenMemory, method: str
) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice, bob = _handles(memory)
    mine = await alice.record(BIKE)
    theirs = await bob.record(BOB_BIKE)
    their_revoked = await bob.record(BOB_KEY)
    await bob.revoke(their_revoked, reason="no longer true", actor="test")
    target = their_revoked if method == "reinstate" else theirs
    before = (await _state(alice), await _state(bob))

    call = ID_CALLS[method]
    got = await _outcome(lambda: call(alice, target.id, mine), target.id)
    unknown = uuid4()
    want = await _outcome(lambda: call(alice, unknown, mine), unknown)
    assert got == want
    assert got[0] == NOT_FOUND[method]
    assert (await _state(alice), await _state(bob)) == before


@pytest.mark.parametrize("link", ["derived_from", "source_message_id"])
async def test_derived_from_or_source_message_in_another_scope_is_not_found(
    open_memory: OpenMemory, link: str
) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice, bob = _handles(memory)
    theirs = await bob.record(BOB_BIKE)
    their_turn = await bob.record_message("I keep my bike at the office.", session_id="s1")
    before = (await _state(alice), await _state(bob))

    def call(linked: UUID) -> Awaitable[Fact]:
        if link == "derived_from":
            return alice.record(HELMET, derived_from=[linked])
        return alice.record(HELMET, source_message_id=linked)

    their_id = theirs.id if link == "derived_from" else their_turn.id
    got = await _outcome(lambda: call(their_id), their_id)
    unknown = uuid4()
    want = await _outcome(lambda: call(unknown), unknown)
    assert got == want
    assert got[0] == "NotFoundError"
    assert (await _state(alice), await _state(bob)) == before


async def test_supersede_with_a_replacement_in_another_scope_is_refused(
    open_memory: OpenMemory,
) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice, bob = _handles(memory)
    mine = await alice.record(BIKE)
    theirs = await bob.record(BOB_BIKE)
    before = (await _state(alice), await _state(bob))

    got = await _outcome(lambda: alice.supersede(mine.id, theirs.id), theirs.id)
    unknown = uuid4()
    want = await _outcome(lambda: alice.supersede(mine.id, unknown), unknown)
    assert got == want
    assert got[0] == "ValueError"
    assert (await _state(alice), await _state(bob)) == before
    their_provenance = await bob.explain(theirs.id)
    assert their_provenance is not None and their_provenance.supersedes is None
    assert [h.fact.id for h in await alice.recall("bike")] == [mine.id]


# ── Messages from more than one scope ──────────────────────────────────


@pytest.mark.parametrize("persist", [True, False], ids=["persist", "preview"])
@pytest.mark.parametrize("via", ["handle", "root"])
async def test_extract_refuses_messages_from_two_scopes(
    open_memory: OpenMemory, via: str, persist: bool
) -> None:
    llm, embedder = SpyLLM(), SpyEmbedder()
    memory = await open_memory(allow_default_scope=False, llm=llm, embedder=embedder)
    alice, bob = _handles(memory)
    before = (await _state(alice), await _state(bob))
    messages = [_turn(ALICE), _turn(BOB, "I keep my bike at the office.")]
    extract = alice.extract if via == "handle" else memory.extract
    with pytest.raises(ScopeError):
        await extract(messages, persist=persist)
    assert llm.calls == 0
    assert embedder.calls == []
    assert (await _state(alice), await _state(bob)) == before


@pytest.mark.parametrize("persist", [True, False], ids=["persist", "preview"])
@pytest.mark.parametrize("via", ["handle", "root"])
async def test_extract_events_refuses_messages_from_two_scopes(
    open_memory: OpenMemory, via: str, persist: bool
) -> None:
    llm = SpyLLM()
    memory = await open_memory(allow_default_scope=False, llm=llm)
    alice, bob = _handles(memory)
    before = (await _state(alice), await _state(bob))
    messages = [_turn(ALICE), _turn(BOB, "I keep my bike at the office.")]
    extract_events = alice.extract_events if via == "handle" else memory.extract_events
    with pytest.raises(ScopeError):
        await extract_events(messages, persist=persist)
    assert llm.calls == 0
    assert (await _state(alice), await _state(bob)) == before


async def test_event_extractor_refuses_messages_from_two_scopes() -> None:
    llm = SpyLLM()
    extractor = EventExtractor(llm)  # type: ignore[arg-type]
    with pytest.raises(ScopeError):
        await extractor.extract([_turn(ALICE), _turn(BOB)])
    assert llm.calls == 0
    assert len(await extractor.extract([_turn(ALICE), _turn(ALICE)])) == 1


@pytest.mark.parametrize("method", ["extract", "extract_events"])
@pytest.mark.parametrize("persist", [True, False], ids=["persist", "preview"])
async def test_a_handle_refuses_messages_from_another_scope(
    open_memory: OpenMemory, method: str, persist: bool
) -> None:
    # All from one scope, but not the handle's: the event extractor alone
    # cannot tell, so the handle must.
    llm = SpyLLM()
    memory = await open_memory(allow_default_scope=False, llm=llm)
    alice, bob = _handles(memory)
    before = (await _state(alice), await _state(bob))
    with pytest.raises(ScopeError):
        await getattr(alice, method)([_turn(BOB), _turn(BOB)], persist=persist)
    assert llm.calls == 0
    assert (await _state(alice), await _state(bob)) == before

    # Its own scope's messages go through.
    assert len(await getattr(alice, method)([_turn(ALICE)], persist=persist)) == 1


@pytest.mark.parametrize("method", ["extract", "extract_events"])
@pytest.mark.parametrize("via", ["handle", "root"])
async def test_extract_with_no_messages_still_needs_an_llm(
    open_memory: OpenMemory, method: str, via: str
) -> None:
    # No message names a scope, so the root builds no handle; both say the same.
    with_llm = await open_memory(allow_default_scope=False, llm=SpyLLM())
    target = _handles(with_llm)[0] if via == "handle" else with_llm
    assert await getattr(target, method)([]) == []
    without = await open_memory(allow_default_scope=False)
    target = _handles(without)[0] if via == "handle" else without
    with pytest.raises(RuntimeError, match="requires an LLM"):
        await getattr(target, method)([])


# ── A store that answers with another scope's data ─────────────────────

Leak = Callable[[Any], Any]


class LeakyStore:
    """Passes every call to a real store, and can change what some methods answer.

    It stands for a store with a scope bug. For facts it also answers
    `provenance` for the leaked fact, as a store that ignored the scope in
    every read would, so the retriever's own visibility check lets it through.
    """

    def __init__(self, inner: EngramStore) -> None:
        self._inner = inner
        self.leaks: dict[str, Leak] = {}
        self.calls: list[str] = []

    @property
    def read_only(self) -> bool:
        return self._inner.read_only

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._inner, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            answer = await method(*args, **kwargs)
            leak = self.leaks.get(name)
            return answer if leak is None else leak(answer)

        return call


@dataclass
class TwoScopes:
    memory: Engram
    store: LeakyStore
    embedder: SpyEmbedder
    alice: ScopedEngram
    mine: Fact
    mine_revoked: Fact
    theirs: Fact
    their_provenance: Provenance
    their_turn: ChatMessage
    their_event: Event
    their_revocation: Revocation
    their_erasure: ErasureRecord


async def _two_scopes(open_store: OpenStore) -> TwoScopes:
    inner = await open_store()
    store = LeakyStore(inner)
    embedder = SpyEmbedder()
    memory = await Engram.open(store=store, embedder=embedder)  # type: ignore[arg-type]
    alice, bob = _handles(memory)
    mine = await alice.record(BIKE, session_id="s1")
    mine_revoked = await alice.record(KEY, session_id="s1")
    await alice.revoke(mine_revoked, reason="no longer true", actor="test")
    await alice.record_message("Where is my bike?", session_id="s1")
    await inner.upsert_event(_event(ALICE))

    theirs = await bob.record(BOB_BIKE, session_id="s1")
    their_turn = await bob.record_message("Where is my bike?", session_id="s1")
    their_event = _event(BOB)
    await inner.upsert_event(their_event)
    their_revocation = await bob.revoke(
        await bob.record(BOB_KEY), reason="no longer true", actor="test"
    )
    their_erasure = await bob.erase(await bob.record(BOB_CAR), reason="asked to", actor="test")
    their_provenance = (await inner.provenance([theirs.id], BOB))[theirs.id]
    stored = await inner.get_fact(theirs.id, BOB)
    assert stored is not None
    return TwoScopes(
        memory=memory,
        store=store,
        embedder=embedder,
        alice=alice,
        mine=mine,
        mine_revoked=mine_revoked,
        theirs=stored,
        their_provenance=their_provenance,
        their_turn=their_turn,
        their_event=their_event,
        their_revocation=their_revocation,
        their_erasure=their_erasure,
    )


def _plus(item: object) -> Leak:
    return lambda answer: [*answer, item]


def _with_their_fact(s: TwoScopes, method: str) -> dict[str, Leak]:
    return {
        method: _plus(s.theirs),
        "provenance": lambda answer: {**answer, s.theirs.id: s.their_provenance},
    }


HandleCall = Callable[[ScopedEngram], Awaitable[object]]

READ_LEAKS: dict[str, Callable[[TwoScopes], tuple[dict[str, Leak], HandleCall]]] = {
    "get_facts-recall": lambda s: (_with_their_fact(s, "get_facts"), lambda h: h.recall("bike")),
    "keyword_search-recall": lambda s: (
        _with_their_fact(s, "keyword_search"),
        lambda h: h.recall("bike"),
    ),
    "keyword_search-context_entries": lambda s: (
        _with_their_fact(s, "keyword_search"),
        lambda h: h.context_entries("bike"),
    ),
    "keyword_search-context": lambda s: (
        _with_their_fact(s, "keyword_search"),
        lambda h: h.context("bike"),
    ),
    # Their provenance on our fact: the scored fact and its provenance disagree.
    "provenance-recall": lambda s: (
        {"provenance": lambda answer: dict.fromkeys(answer, s.their_provenance)},
        lambda h: h.recall("bike"),
    ),
    "list_session_messages-list_messages": lambda s: (
        {"list_session_messages": _plus(s.their_turn)},
        lambda h: h.list_messages("s1"),
    ),
    "search_events-search_events": lambda s: (
        {"search_events": _plus(s.their_event)},
        lambda h: h.search_events(""),
    ),
    "list_revocations-revocations": lambda s: (
        {"list_revocations": _plus(s.their_revocation)},
        lambda h: h.revocations(),
    ),
    "list_erasures-erasures": lambda s: (
        {"list_erasures": _plus(s.their_erasure)},
        lambda h: h.erasures(),
    ),
    "explain-explain": lambda s: (
        {"explain": lambda answer: s.their_provenance},
        lambda h: h.explain(s.mine.id),
    ),
}


@pytest.mark.parametrize("case", sorted(READ_LEAKS))
async def test_store_returning_another_scopes_fact_raises_scope_error(
    open_store: OpenStore, case: str
) -> None:
    s = await _two_scopes(open_store)
    leaks, call = READ_LEAKS[case](s)
    await call(s.alice)  # without the leak the call answers
    s.store.leaks = leaks
    with pytest.raises(ScopeError):
        await call(s.alice)


def _as_bob[T: (Revocation, ErasureRecord)](record: T) -> T:
    return record.model_copy(update={"org_id": BOB.org_id, "user_id": BOB.user_id})


WRITE_LEAKS: dict[str, Callable[[TwoScopes], tuple[dict[str, Leak], HandleCall]]] = {
    "revoke": lambda s: (
        {"revoke": _as_bob},
        lambda h: h.revoke(s.mine, reason="wrong", actor="test"),
    ),
    "supersede": lambda s: (
        {"revoke": _as_bob},
        lambda h: h.supersede(s.mine_revoked.id, s.mine.id),
    ),
    # Their fact among the ones made visible again: the embedder must not see it.
    "reinstate-visible": lambda s: (
        {"reinstate": lambda answer: (answer[0], [*answer[1], s.theirs])},
        lambda h: h.reinstate(s.mine_revoked, reason="right", actor="test"),
    ),
    "reinstate-lifted": lambda s: (
        {"reinstate": lambda answer: ([*answer[0], s.their_revocation], answer[1])},
        lambda h: h.reinstate(s.mine_revoked, reason="right", actor="test"),
    ),
    # Their record from erase: finalize must not write to it.
    "erase": lambda s: (
        {"erase": _as_bob},
        lambda h: h.erase(s.mine, reason="asked to", actor="test"),
    ),
    "erase-finalize": lambda s: (
        {"finalize_erasure": _as_bob},
        lambda h: h.erase(s.mine, reason="asked to", actor="test"),
    ),
    "erase_all": lambda s: (
        {"erase_scope_residue": _as_bob},
        lambda h: h.erase_all(reason="asked to", actor="test"),
    ),
}

# Cases where the store's answer is a record that finalize_erasure would then write to.
REFUSED_BEFORE_FINALIZE = {"erase", "erase_all"}


@pytest.mark.parametrize("case", sorted(WRITE_LEAKS))
async def test_store_answering_a_write_for_another_scope_raises_scope_error(
    open_store: OpenStore, case: str
) -> None:
    s = await _two_scopes(open_store)
    leaks, call = WRITE_LEAKS[case](s)
    s.store.leaks = leaks
    s.embedder.calls.clear()
    finalized = s.store.calls.count("finalize_erasure")
    with pytest.raises(ScopeError):
        await call(s.alice)
    # Nothing of the other scope reached the embedder or this scope's index.
    assert s.theirs.text not in [t for batch in s.embedder.calls for t in batch]
    if case == "erase_all":
        # Each fact's erase finalizes its own record; the residue's is never finalized.
        assert s.store.calls[-1] == "erase_scope_residue"
    elif case in REFUSED_BEFORE_FINALIZE:
        assert s.store.calls.count("finalize_erasure") == finalized


class ForeignEvents:
    """An event extractor that files every event under Bob's scope."""

    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self, messages: list[ChatMessage], session_date: datetime | None = None
    ) -> list[Event]:
        self.calls += 1
        return [_event(BOB)]


@pytest.mark.parametrize("persist", [True, False], ids=["persist", "preview"])
async def test_events_the_extractor_files_under_another_scope_are_refused(
    open_memory: OpenMemory, persist: bool
) -> None:
    memory = await open_memory(allow_default_scope=False, llm=SpyLLM())
    memory._events = ForeignEvents()  # type: ignore[assignment]
    alice, bob = _handles(memory)
    with pytest.raises(ScopeError):
        await alice.extract_events([_turn(ALICE)], persist=persist)
    assert await bob.search_events("") == []
    assert await alice.search_events("") == []


async def test_explain_checks_the_id_is_in_the_scope_before_it_asks(
    open_store: OpenStore,
) -> None:
    # Provenance names no scope, so the handle checks the id is its own
    # before it asks for it. A store whose explain ignored the scope then
    # still answers nothing for another scope's id.
    s = await _two_scopes(open_store)
    explain = s.store._inner.explain

    async def ignores_the_scope(fact_id: UUID, scope: Scope) -> Provenance | None:
        return await explain(fact_id, BOB)  # finds the fact where it is: Bob's scope

    s.store.explain = ignores_the_scope  # type: ignore[method-assign]
    assert await s.alice.explain(s.theirs.id) is None


# ── Read-only ──────────────────────────────────────────────────────────


class ClaimsReadOnly:
    """Says it is read-only but passes every call on and records its name."""

    def __init__(self, inner: EngramStore) -> None:
        self._inner = inner
        self.calls: list[str] = []

    @property
    def read_only(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._inner, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return await method(*args, **kwargs)

        return call


HANDLE_WRITES: dict[str, HandleCall] = {
    "record": lambda h: h.record(HELMET),
    "record_message": lambda h: h.record_message("Thanks!", session_id="s1"),
    "extract": lambda h: h.extract([_turn(ALICE)], persist=True),
    "extract_events": lambda h: h.extract_events([_turn(ALICE)], persist=True),
    "supersede": lambda h: h.supersede(uuid4(), uuid4()),
    "revoke": lambda h: h.revoke(uuid4(), reason="wrong", actor="test"),
    "reinstate": lambda h: h.reinstate(uuid4(), reason="right", actor="test"),
    "erase": lambda h: h.erase(uuid4(), reason="asked to", actor="test"),
    "forget": lambda h: h.forget(uuid4(), reason="wrong", actor="test"),
    "erase_all": lambda h: h.erase_all(reason="asked to", actor="test"),
    "delete_conversation": lambda h: h.delete_conversation("s1"),
}

# extract and extract_events with persist=False store nothing, so they are reads.
HANDLE_READS: dict[str, HandleCall] = {
    "recall": lambda h: h.recall("bike"),
    "context": lambda h: h.context("where is the bike"),
    "context_entries": lambda h: h.context_entries("where is the bike"),
    "search_events": lambda h: h.search_events(""),
    "explain": lambda h: h.explain(uuid4()),
    "revocations": lambda h: h.revocations(),
    "erasures": lambda h: h.erasures(),
    "extract": lambda h: h.extract([_turn(ALICE)], persist=False),
    "extract_events": lambda h: h.extract_events([_turn(ALICE)], persist=False),
    "list_conversations": lambda h: h.list_conversations(),
    "list_messages": lambda h: h.list_messages("s1"),
    "fact_counts": lambda h: h.fact_counts(),
}


def test_every_handle_method_is_listed_as_a_read_or_a_write() -> None:
    coroutines = {
        name for name, member in _public_functions().items() if inspect.iscoroutinefunction(member)
    }
    assert coroutines == set(HANDLE_WRITES) | set(HANDLE_READS)


@pytest.mark.parametrize("method", sorted(HANDLE_WRITES))
async def test_a_handle_refuses_every_write_when_read_only(
    open_store: OpenStore, method: str
) -> None:
    store = ClaimsReadOnly(await open_store())
    embedder, llm = SpyEmbedder(), SpyLLM()
    memory = await Engram.open(store=store, embedder=embedder, llm=llm)  # type: ignore[arg-type]
    alice, _ = _handles(memory)
    with pytest.raises(ReadOnlyStoreError):
        await HANDLE_WRITES[method](alice)
    assert store.calls == []
    assert embedder.calls == []
    assert llm.calls == 0


@pytest.mark.parametrize("method", sorted(HANDLE_READS))
async def test_a_handle_reads_when_read_only(open_store: OpenStore, method: str) -> None:
    store = ClaimsReadOnly(await open_store())
    memory = await Engram.open(store=store, llm=SpyLLM())  # type: ignore[arg-type]
    alice, _ = _handles(memory)
    await HANDLE_READS[method](alice)
