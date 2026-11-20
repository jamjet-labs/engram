"""Scope ids that look like SQL, patterns or each other stay in their own scope.

Every id here is legal under the scope rules. For each pair, one scope holds a
fact, a message, an event, a revocation and an erasure record. The other scope
must see none of it through any read, and erasing one must leave the other
intact. These run on every backend.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.engram import Engram
from engram.models import ChatMessage, Event
from engram.scope import Scope
from tests.conformance.backends import OpenMemory

NOW = datetime(2026, 3, 4, 10, 30, tzinfo=UTC)

ScopePair = tuple[tuple[str, str], tuple[str, str]]


def _by_user(a: str, b: str) -> ScopePair:
    return ("acme", a), ("acme", b)


def _by_org(a: str, b: str) -> ScopePair:
    return (a, "alice"), (b, "alice")


_ID_PAIRS = [
    ("o'r", "or"),
    ("a\\'b", "a'b"),
    ("%", "x"),
    ("_", "a"),
    ("a%", "ab"),
    ("x' OR '1'='1", "x"),
    ("); DROP TABLE facts;--", "a"),
    ("ü", "ü"),
    ("ali", "alice"),
    ("a" * 128, "a" * 127 + "b"),
    ("\U0001f600", "\U0001f601"),
]

_PAIRS: list[tuple[str, ScopePair]] = []
for _a, _b in _ID_PAIRS:
    _PAIRS.append((f"user-{_a[:12]!r}", _by_user(_a, _b)))
    _PAIRS.append((f"org-{_a[:12]!r}", _by_org(_a, _b)))
_PAIRS.append(("org-user-split", (("a", "bc"), ("ab", "c"))))


def _text(tag: str) -> str:
    return f"The zebra crossing note for {tag} is kept in the grey cabinet."


async def _fill(memory: Engram, scope: Scope, tag: str) -> None:
    """A fact, a message, an event, a revocation and an erasure in one scope."""
    handle = memory.scoped(org_id=scope.org_id, user_id=scope.user_id)
    await handle.record(_text(tag), session_id="s1")
    revoked = await handle.record(f"The zebra crossing was repainted for {tag}.")
    erased = await handle.record(f"A zebra crossing secret about {tag} lives here.")
    await handle.revoke(revoked, reason="out of date", actor="tester")
    await handle.erase(erased.id, reason="asked to remove", actor="tester")
    await memory._store.upsert_message(
        ChatMessage(
            role="user",
            content=f"The zebra crossing is by the school, says {tag}.",
            scope=scope,
            session_id="s1",
            timestamp=NOW,
        )
    )
    await memory._store.upsert_event(
        Event(
            verb="painted",
            object_canonical="crossing",
            subject_canonical="zebra",
            scope=scope,
            time_start=NOW,
        )
    )


async def _assert_sees_nothing(memory: Engram, scope: Scope) -> None:
    handle = memory.scoped(org_id=scope.org_id, user_id=scope.user_id)
    store = memory._store
    query = "zebra crossing"
    assert await handle.recall(query) == []
    assert await handle.context(query) == ""
    assert await store.keyword_search(query, scope) == []
    assert await store.aggregate_sessions(query, scope) == []
    assert await handle.list_messages("s1") == []
    assert await handle.list_conversations() == []
    assert await handle.search_events("zebra") == []
    assert await handle.revocations(include_reinstated=True) == []
    assert await handle.erasures() == []
    assert await handle.fact_counts() == (0, 0)
    assert await store.list_fact_ids(scope) == []


@pytest.mark.parametrize(("pair"), [p for _, p in _PAIRS], ids=[i for i, _ in _PAIRS])
async def test_hostile_scope_ids_stay_isolated(open_memory: OpenMemory, pair: ScopePair) -> None:
    memory = await open_memory(allow_default_scope=False)
    one = Scope(org_id=pair[0][0], user_id=pair[0][1])
    other = Scope(org_id=pair[1][0], user_id=pair[1][1])
    assert one != other

    await _fill(memory, one, "one")
    await _assert_sees_nothing(memory, other)

    # The first scope does see its own data, so the empty reads above mean something.
    handle_one = memory.scoped(org_id=one.org_id, user_id=one.user_id)
    assert [sf.fact.text for sf in await handle_one.recall("zebra crossing")] == [_text("one")]
    assert await store_has_message(memory, one)

    # The same data in the other scope, then an erase in one leaves the other intact.
    await _fill(memory, other, "other")
    handle_other = memory.scoped(org_id=other.org_id, user_id=other.user_id)
    before = await handle_other.fact_counts()
    assert before == (2, 1)

    await handle_one.erase_all(reason="account closed", actor="tester")
    assert await handle_one.recall("zebra crossing") == []
    assert await handle_one.fact_counts() == (0, 0)
    assert await handle_one.list_messages("s1") == []

    assert await handle_other.fact_counts() == before
    assert [sf.fact.text for sf in await handle_other.recall("zebra crossing")] == [_text("other")]
    assert len(await handle_other.list_messages("s1")) == 1
    assert len(await handle_other.search_events("zebra")) == 1
    assert len(await handle_other.revocations()) == 1


async def store_has_message(memory: Engram, scope: Scope) -> bool:
    handle = memory.scoped(org_id=scope.org_id, user_id=scope.user_id)
    return len(await handle.list_messages("s1")) == 1
