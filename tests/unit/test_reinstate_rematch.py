"""Reinstating one fact must not bring back content that quotes another revoked fact.

The bypass: revoke A, revoke B, reinstate A, and C (which quotes B) became
visible. When B was revoked, C was already hidden by A, and the scan that
records "C quotes B" only looked at visible rows. So C had no record of B,
and lifting A's tombstone left nothing hiding it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from engram import Engram
from engram.models import Event
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
