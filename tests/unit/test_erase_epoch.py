"""The erase epoch: a write that started before a scope-wide erase never lands after it.

Each scope has an erase epoch, 0 until its first `erase_all`. A write takes a
guard, the epoch it saw, when it starts, and the store compares the guard
inside the write transaction. `erase_all` moves the epoch in the transaction
that removes the scope's residue. No clock is read on either side, so two
processes whose clocks differ still agree on what came first.

Every test that needs memory takes `open_memory`, so it runs on every backend.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

import pytest
import pytest_asyncio

from engram import Engram, ScopedEngram
from engram.errors import ScopeErasedError, ScopeError, ScopeResidueRefusedError
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Fact
from engram.scope import Scope, ScopeWriteGuard
from tests.conformance.backends import OpenMemory

ALICE = Scope(org_id="acme", user_id="alice")
BOB = Scope(org_id="acme", user_id="bob")
ALICE_AT_GLOBEX = Scope(org_id="globex", user_id="alice")
SECRET = "Alice's locker code is 4417."
WRITERS = ["record", "extract"]


class Gate:
    """Holds the first call that reaches it until released."""

    def __init__(self) -> None:
        self.armed = True
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def hold(self) -> None:
        if self.armed:
            self.armed = False
            self.entered.set()
            await self.release.wait()

    async def wait_entered(self) -> None:
        await asyncio.wait_for(self.entered.wait(), timeout=10)


@pytest_asyncio.fixture
async def gate() -> AsyncIterator[Gate]:
    g = Gate()
    yield g
    g.release.set()


class OneFactLLM:
    """Extracts the same one fact from any conversation, and counts the calls."""

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
        fact = {
            "text": SECRET,
            "category": "personal",
            "polarity": "affirmative",
            "confidence": 0.9,
            "entities": [],
            "event_date": None,
        }
        return LLMResponse(content=json.dumps({"facts": [fact]}), finish_reason="stop")


def _now() -> datetime:
    return datetime.now(UTC)


def _turn(scope: Scope) -> ChatMessage:
    return ChatMessage(
        scope=scope,
        session_id="s1",
        role="user",
        content="My locker code is 4417.",
        timestamp=_now(),
    )


def _handle(memory: Engram, scope: Scope) -> ScopedEngram:
    return memory.scoped(org_id=scope.org_id, user_id=scope.user_id)


def _alice(memory: Engram) -> ScopedEngram:
    return _handle(memory, ALICE)


def _skew_writer_clock(monkeypatch: pytest.MonkeyPatch, by: timedelta) -> None:
    """Make the clock a write reads when it starts run ``by`` away from the eraser's."""

    class Skewed(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return datetime.now(tz) + by

    monkeypatch.setattr("engram.scoped.datetime", Skewed)


def _hold_first_upsert(monkeypatch: pytest.MonkeyPatch, memory: Engram, gate: Gate) -> None:
    """The first fact write waits at the gate: the write has started, nothing is stored."""
    upsert = memory._store.upsert_fact

    async def held(fact: Fact, **kw: Any) -> None:
        await gate.hold()
        await upsert(fact, **kw)

    monkeypatch.setattr(memory._store, "upsert_fact", held)


async def _write(handle: ScopedEngram, writer: str) -> bool:
    """One write in the handle's scope. True if it stored its fact, False if refused."""
    if writer == "record":
        try:
            await handle.record(SECRET)
        except ScopeErasedError:
            return False
        return True
    return bool(await handle.extract([_turn(handle.scope)]))


async def _assert_nothing_stored(memory: Engram) -> None:
    assert await memory._store.list_fact_ids(ALICE) == []
    assert await _alice(memory).recall("locker code") == []
    assert await memory._vec.count(ALICE) == 0


async def _started_before_the_erase(
    open_memory: OpenMemory,
    monkeypatch: pytest.MonkeyPatch,
    gate: Gate,
    writer: str,
    skew: timedelta,
) -> None:
    memory = await open_memory(allow_default_scope=False, llm=OneFactLLM())
    alice = _alice(memory)
    _skew_writer_clock(monkeypatch, skew)
    _hold_first_upsert(monkeypatch, memory, gate)
    write = asyncio.create_task(_write(alice, writer))
    await gate.wait_entered()  # the write has started; its fact is not stored yet
    assert await alice.erase_all(reason="asked to", actor="admin") == 0
    gate.release.set()
    assert await write is False
    await _assert_nothing_stored(memory)


# ── Clocks ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("writer", WRITERS)
async def test_write_started_before_erase_is_refused_when_the_writer_clock_runs_ahead(
    open_memory: OpenMemory, monkeypatch: pytest.MonkeyPatch, gate: Gate, writer: str
) -> None:
    await _started_before_the_erase(open_memory, monkeypatch, gate, writer, timedelta(minutes=2))


@pytest.mark.parametrize("writer", WRITERS)
async def test_write_started_before_erase_is_refused_when_the_writer_clock_runs_behind(
    open_memory: OpenMemory, monkeypatch: pytest.MonkeyPatch, gate: Gate, writer: str
) -> None:
    await _started_before_the_erase(open_memory, monkeypatch, gate, writer, -timedelta(minutes=2))


@pytest.mark.parametrize("writer", WRITERS)
async def test_write_started_after_erase_is_stored_when_the_writer_clock_runs_behind(
    open_memory: OpenMemory, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    memory = await open_memory(allow_default_scope=False, llm=OneFactLLM())
    alice = _alice(memory)
    await alice.record("Alice used to park on level 2.")
    assert await alice.erase_all(reason="asked to", actor="admin") == 1
    _skew_writer_clock(monkeypatch, -timedelta(minutes=2))
    assert await _write(alice, writer) is True
    assert len(await memory._store.list_fact_ids(ALICE)) == 1
    assert [h.fact.text for h in await alice.recall("locker code")] == [SECRET]


# ── Guards ──────────────────────────────────────────────────────────────


def _alice_fact() -> Fact:
    return Fact(text=SECRET, scope=ALICE, valid_from=_now())


# Each write that takes a guard, given one for another scope.
GuardedWrite = Callable[[Engram, Any], Awaitable[object]]

GUARDED_WRITES: dict[str, GuardedWrite] = {
    "store.upsert_fact": lambda m, g: m._store.upsert_fact(_alice_fact(), guard=g),
    "store.upsert_message": lambda m, g: m._store.upsert_message(_turn(ALICE), guard=g),
    "scoped.record_message": lambda m, g: _alice(m).record_message(
        SECRET, session_id="s1", guard=g
    ),
    "scoped.extract": lambda m, g: _alice(m).extract([_turn(ALICE)], guard=g),
    "root.record_message": lambda m, g: m.record_message(
        SECRET, session_id="s1", user_id=ALICE.user_id, org_id=ALICE.org_id, guard=g
    ),
    "root.extract": lambda m, g: m.extract([_turn(ALICE)], guard=g),
}


@pytest.mark.parametrize("site", sorted(GUARDED_WRITES))
@pytest.mark.parametrize("other", [BOB, ALICE_AT_GLOBEX], ids=["other-user", "other-org"])
async def test_guard_for_another_scope_is_refused(
    open_memory: OpenMemory, site: str, other: Scope
) -> None:
    llm = OneFactLLM()
    memory = await open_memory(allow_default_scope=False, llm=llm)
    theirs = await _handle(memory, other).guard()
    # The same epoch as Alice's: only the scope tells them apart.
    assert theirs.erase_epoch == (await _alice(memory).guard()).erase_epoch == 0
    with pytest.raises(ScopeError):
        await GUARDED_WRITES[site](memory, theirs)
    assert llm.calls == 0  # refused before the LLM saw the conversation
    assert await memory._store.list_fact_ids(ALICE) == []
    assert await _alice(memory).list_messages("s1") == []
    assert await _handle(memory, other).list_messages("s1") == []


@pytest.mark.persistent
async def test_epoch_survives_reopen(open_memory: OpenMemory) -> None:
    first = await open_memory(allow_default_scope=False)
    before = await _alice(first).guard()
    assert before.erase_epoch == 0
    await _alice(first).erase_all(reason="asked to", actor="admin")
    await first.close()

    alice = _alice(await open_memory.reopen(allow_default_scope=False))
    after = await alice.guard()
    assert (after.scope, after.erase_epoch) == (ALICE, 1)
    # A guard from before the erase is still refused after the restart.
    with pytest.raises(ScopeErasedError):
        await alice.record_message(SECRET, session_id="s1", guard=before)
    await alice.record_message("Alice changed the code.", session_id="s1", guard=after)
    assert [m.content for m in await alice.list_messages("s1")] == ["Alice changed the code."]


async def test_guard_taken_after_erase_allows_writes(open_memory: OpenMemory) -> None:
    memory = await open_memory(allow_default_scope=False, llm=OneFactLLM())
    alice = _alice(memory)
    stale = await alice.guard()
    await alice.record("Alice used to park on level 2.")
    assert await alice.erase_all(reason="asked to", actor="admin") == 1
    fresh = await alice.guard()
    assert (stale.erase_epoch, fresh.erase_epoch) == (0, 1)

    # Control: the guard from before the erase is refused, so the writes below mean something.
    with pytest.raises(ScopeErasedError):
        await memory._store.upsert_fact(_alice_fact(), guard=stale)
    with pytest.raises(ScopeErasedError):
        await alice.record_message(SECRET, session_id="s1", guard=stale)
    assert await alice.extract([_turn(ALICE)], guard=stale) == []
    assert await memory._store.list_fact_ids(ALICE) == []

    stored = _alice_fact()
    await memory._store.upsert_fact(stored, guard=fresh)
    await alice.record_message(SECRET, session_id="s1", guard=fresh)
    [extracted] = await alice.extract([_turn(ALICE)], guard=fresh)
    assert set(await memory._store.list_fact_ids(ALICE)) == {stored.id, extracted.id}
    assert [m.content for m in await alice.list_messages("s1")] == [SECRET]


async def test_erase_of_one_fact_does_not_refuse_other_writes(open_memory: OpenMemory) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice = _alice(memory)
    guard = await alice.guard()
    fact = await alice.record("Alice parks on level 2.")
    await alice.erase(fact, reason="asked to", actor="admin")
    assert (await alice.guard()).erase_epoch == guard.erase_epoch == 0

    late = _alice_fact()
    await memory._store.upsert_fact(late, guard=guard)
    await alice.record_message(SECRET, session_id="s1", guard=guard)
    assert await memory._store.list_fact_ids(ALICE) == [late.id]
    assert [m.content for m in await alice.list_messages("s1")] == [SECRET]


async def test_each_scope_has_its_own_epoch(open_memory: OpenMemory) -> None:
    memory = await open_memory(allow_default_scope=False)
    # Another user in Alice's org, Alice in another org, and two scopes whose
    # ids run together the same way.
    split_a, split_b = Scope(org_id="a", user_id="bc"), Scope(org_id="ab", user_id="c")
    for scope in (BOB, BOB, ALICE_AT_GLOBEX, split_b):
        await _handle(memory, scope).erase_all(reason="asked to", actor="admin")
    epochs = {
        (s.org_id, s.user_id): (await _handle(memory, s).guard()).erase_epoch
        for s in (ALICE, BOB, ALICE_AT_GLOBEX, split_a, split_b)
    }
    assert epochs == {
        ("acme", "alice"): 0,
        ("acme", "bob"): 2,
        ("globex", "alice"): 1,
        ("a", "bc"): 0,
        ("ab", "c"): 1,
    }


async def test_a_refused_residue_erase_leaves_the_epoch(open_memory: OpenMemory) -> None:
    memory = await open_memory(allow_default_scope=False)
    alice = _alice(memory)
    guard = await alice.guard()
    await alice.record("Alice parks on level 2.")
    with pytest.raises(ScopeResidueRefusedError):
        await memory._store.erase_scope_residue(ALICE, reason="asked to", actor="admin")
    assert await alice.guard() == guard
    await alice.record_message(SECRET, session_id="s1", guard=guard)


def test_scope_write_guard_is_a_checked_value() -> None:
    guard = ScopeWriteGuard(scope=ALICE, erase_epoch=3)
    assert (guard.scope, guard.erase_epoch) == (ALICE, 3)
    assert guard == ScopeWriteGuard(scope=ALICE, erase_epoch=3)
    assert guard != ScopeWriteGuard(scope=BOB, erase_epoch=3)
    with pytest.raises(FrozenInstanceError):
        guard.erase_epoch = 4  # type: ignore[misc]
    for scope, epoch, error in [
        (("acme", "alice"), 0, TypeError),
        (ALICE, True, TypeError),
        (ALICE, "1", TypeError),
        (ALICE, 1.0, TypeError),
        (ALICE, -1, ValueError),
    ]:
        with pytest.raises(error):
            ScopeWriteGuard(scope=scope, erase_epoch=epoch)  # type: ignore[arg-type]
