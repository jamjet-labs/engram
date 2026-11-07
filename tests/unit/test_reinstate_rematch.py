"""Reinstating one fact must not bring back content that quotes another revoked fact.

The bypass: revoke A, revoke B, reinstate A, and C (which quotes B) became
visible. When B was revoked, C was already hidden by A, and the scan that
records "C quotes B" only looked at visible rows. So C had no record of B,
and lifting A's tombstone left nothing hiding it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from engram import Engram
from engram.models import Event, Fact
from engram.scope import Scope

A = "The vault code for building seven is four four one two."
B = "The backup server root password is purple elephant ninety."
OPS = Scope(org_id="default", user_id="ops")


async def _visible_texts(memory: Engram, query: str) -> list[str]:
    return [r.fact.text for r in await memory.recall(query, user_id="ops", top_k=50)]


async def test_revoke_a_revoke_b_reinstate_a_keeps_c_hidden() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Handover notes: {A} Also, {B}", user_id="ops")

        await memory.revoke(a, reason="rotated", actor="sec")
        await memory.revoke(b, reason="rotated", actor="sec")
        await memory.reinstate(a, reason="vault code was not rotated after all", actor="sec")

        visible = await _visible_texts(memory, "backup server root password vault code")
        assert A in visible
        assert not any("purple elephant" in t for t in visible), visible
        assert c.id in await memory._store.hidden_fact_ids(OPS)
        explained = await memory.explain(c.id, user_id="ops")
        assert explained is not None
        assert explained.revocation_status == "revoked"
        [rev_b] = await memory.revocations(user_id="ops")
        assert explained.hidden_by == [rev_b.id]


async def test_revoking_b_records_c_even_while_c_is_hidden_by_a() -> None:
    # explain() must list every tombstone that covers a fact, not only the first.
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Handover notes: {A} Also, {B}", user_id="ops")
        rev_a = await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        explained = await memory.explain(c.id, user_id="ops")
        assert explained is not None
        assert explained.hidden_by == [rev_a.id, rev_b.id]


async def test_fact_derived_from_a_that_quotes_b_stays_hidden() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Ticket 88 says: {B}", user_id="ops", derived_from=[a.id])
        await memory.revoke(a, reason="r", actor="sec")
        await memory.revoke(b, reason="r", actor="sec")
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert c.id in await memory._store.hidden_fact_ids(OPS)


async def test_transcript_turn_and_event_quoting_b_stay_hidden() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        await memory.record_message(f"{A} and {B}", session_id="s1", user_id="ops")
        event = Event(
            scope=OPS,
            subject_canonical="ops",
            verb="noted",
            object_canonical=A,
            aliases=[B],
            time_start=datetime(2026, 3, 1, tzinfo=UTC),
        )
        await memory._store.upsert_event(event)

        await memory.revoke(a, reason="r", actor="sec")
        await memory.revoke(b, reason="r", actor="sec")
        await memory.reinstate(a, reason="mistake", actor="sec")

        assert await memory._store.list_messages("s1", OPS) == []
        assert await memory._store.get_event(event.id, OPS) is None


async def test_reinstate_repairs_missing_matches_from_older_writes() -> None:
    # A database written before this fix can lack the "C quotes B" row.
    # Reinstating A must re-check C against B before C becomes visible.
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Handover notes: {A} Also, {B}", user_id="ops")
        await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        await memory._store._conn.execute(
            "DELETE FROM content_matches WHERE revocation_id = ?", (str(rev_b.id),)
        )
        await memory._store._conn.commit()

        await memory.reinstate(a, reason="mistake", actor="sec")

        assert c.id in await memory._store.hidden_fact_ids(OPS)
        visible = await _visible_texts(memory, "backup server root password")
        assert not any("purple elephant" in t for t in visible), visible


async def test_reinstate_still_brings_back_what_only_a_hid() -> None:
    # The fix must not over-hide: a quote of A alone comes back with A.
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        quote_a = await memory.record(f"Reminder: {A}", user_id="ops")
        await memory.revoke(a, reason="r", actor="sec")
        await memory.revoke(b, reason="r", actor="sec")
        await memory.reinstate(a, reason="mistake", actor="sec")
        hidden = await memory._store.hidden_fact_ids(OPS)
        assert quote_a.id not in hidden
        assert a.id not in hidden
        assert b.id in hidden


async def test_c_is_never_visible_inside_the_reinstate_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reads share the connection, so they see uncommitted state. The re-check
    # has to land before the tombstone is lifted, not after.
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Handover notes: {A} Also, {B}", user_id="ops")
        await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        await memory._store._conn.execute(
            "DELETE FROM content_matches WHERE revocation_id = ?", (str(rev_b.id),)
        )
        await memory._store._conn.commit()

        seen: list[bool] = []
        real = memory._store._match_scope

        async def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(c.id in await memory._store.hidden_fact_ids(OPS))
            result = await real(*args, **kwargs)
            seen.append(c.id in await memory._store.hidden_fact_ids(OPS))
            return result

        real_commit = memory._store._conn.commit

        async def commit_spy() -> None:
            seen.append(c.id in await memory._store.hidden_fact_ids(OPS))
            await real_commit()

        monkeypatch.setattr(memory._store, "_match_scope", spy)
        monkeypatch.setattr(memory._store._conn, "commit", commit_spy)
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert seen and all(seen), seen
        assert c.id in await memory._store.hidden_fact_ids(OPS)


async def test_reinstate_only_rechecks_what_the_tombstone_was_hiding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-checking the whole scope took seconds at 20k facts x 200 tombstones,
    # on the event loop, holding the write lock. Only items the lifted
    # tombstone hid (and what was derived from them) can change.
    import engram.store.sqlite as store_mod

    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        child = await memory.record("Child note", user_id="ops", derived_from=[a.id])
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Handover notes: {A} Also, {B}", user_id="ops")
        now = datetime.now(UTC)
        for i in range(300):
            await memory._store.upsert_fact(
                Fact(scope=OPS, text=f"filler note {i} about lunch", valid_from=now)
            )
        await memory.revoke(a, reason="r", actor="sec")
        await memory.revoke(b, reason="r", actor="sec")

        calls = 0
        real = store_mod.normalize_text

        def counting(text: str) -> str:
            nonlocal calls
            calls += 1
            return real(text)

        monkeypatch.setattr(store_mod, "normalize_text", counting)
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert calls < 20, calls
        hidden = await memory._store.hidden_fact_ids(OPS)
        assert a.id not in hidden
        assert child.id not in hidden
        assert c.id in hidden


async def test_reinstate_restores_vectors_of_derived_facts() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        await memory.record("Child note about the vault", user_id="ops", derived_from=[a.id])
        assert await memory._vec.count(OPS) == 2
        await memory.revoke(a, reason="r", actor="sec")
        assert await memory._vec.count(OPS) == 0
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert await memory._vec.count(OPS) == 2


async def _drop_matches(memory: Engram, rev_id: object) -> None:
    await memory._store._conn.execute(
        "DELETE FROM content_matches WHERE revocation_id = ?", (str(rev_id),)
    )
    await memory._store._conn.commit()


async def test_repair_covers_turns_hidden_by_the_session_rule() -> None:
    # A revoked by id with a session and no message link hides that session's
    # turns. Lifting A must re-check them, not just the facts.
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops", session_id="s1")
        b = await memory.record(B, user_id="ops")
        await memory.record_message(f"we said {B}", session_id="s1", user_id="ops")
        await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        await _drop_matches(memory, rev_b.id)
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert await memory._store.list_messages("s1", OPS) == []


async def test_repair_covers_events_linked_to_the_reinstated_fact() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        event = Event(
            scope=OPS,
            subject_canonical="ops",
            verb="noted",
            object_canonical=B,
            source_fact_ids=[a.id],
            time_start=datetime(2026, 3, 1, tzinfo=UTC),
        )
        await memory._store.upsert_event(event)
        await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        await _drop_matches(memory, rev_b.id)
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert await memory._store.get_event(event.id, OPS) is None


async def test_repair_covers_facts_derived_from_the_reinstated_fact() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Ticket 88 says: {B}", user_id="ops", derived_from=[a.id])
        await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        await _drop_matches(memory, rev_b.id)
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert c.id in await memory._store.hidden_fact_ids(OPS)


async def test_repair_covers_turns_cited_by_the_reinstated_fact() -> None:
    async with await Engram.open(":memory:") as memory:
        turn = await memory.record_message(f"we said {B}", session_id="s1", user_id="ops")
        a = await memory.record(A, user_id="ops", session_id="s1", source_message_id=turn.id)
        b = await memory.record(B, user_id="ops")
        await memory.revoke(a, reason="r", actor="sec")
        rev_b = await memory.revoke(b, reason="r", actor="sec")
        await _drop_matches(memory, rev_b.id)
        await memory.reinstate(a, reason="mistake", actor="sec")
        assert await memory._store.list_messages("s1", OPS) == []
