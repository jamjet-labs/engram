"""Store methods behind the v1 REST routes the Java client calls."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from engram import Engram
from engram.models import Event, Fact
from engram.scope import Scope

ALICE = Scope(org_id="default", user_id="alice")
BOB = Scope(org_id="default", user_id="bob")


@pytest.fixture
async def memory():
    async with await Engram.open(":memory:") as m:
        yield m


async def test_find_fact_scope(memory: Engram) -> None:
    fact = await memory.record("alice likes tea", user_id="alice")
    assert await memory._store.find_fact_scope(fact.id) == ALICE
    assert await memory._store.find_fact_scope(uuid4()) is None


async def test_list_fact_ids_includes_revoked(memory: Engram) -> None:
    a = await memory.record("alice likes tea", user_id="alice")
    b = await memory.record("alice likes jazz", user_id="alice")
    await memory.record("bob likes chess", user_id="bob")
    await memory.revoke(b.id, reason="test", actor="t", user_id="alice")
    assert set(await memory._store.list_fact_ids(ALICE)) == {a.id, b.id}


async def test_list_session_ids_and_delete_session(memory: Engram) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    await memory.record_message("hi", session_id="s1", user_id="alice", timestamp=t)
    await memory.record_message("again", session_id="s2", user_id="alice", timestamp=t)
    await memory.record_message("hey", session_id="s1", user_id="bob", timestamp=t)
    assert await memory._store.list_session_ids(ALICE) == ["s1", "s2"]
    assert await memory._store.delete_session_messages("s1", ALICE) == 1
    assert await memory._store.list_session_ids(ALICE) == ["s2"]
    assert await memory._store.list_session_ids(BOB) == ["s1"]  # other scope untouched


async def test_fact_counts(memory: Engram) -> None:
    await memory.record("one", user_id="alice")
    two = await memory.record("two", user_id="bob")
    await memory.revoke(two.id, reason="test", actor="t", user_id="bob")
    assert await memory._store.fact_counts() == (2, 1)


async def test_erase_scope_residue_refuses_while_facts_remain(memory: Engram) -> None:
    await memory.record("alice likes tea", user_id="alice")
    with pytest.raises(Exception, match="facts remain"):
        await memory._store.erase_scope_residue(ALICE, reason="test", actor="t")


async def test_erase_scope_residue_refuses_while_only_a_revoked_fact_remains(
    memory: Engram,
) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    fact = await memory.record("alice likes tea", user_id="alice")
    await memory.record_message("hi", session_id="s1", user_id="alice", timestamp=t)
    await memory.revoke(fact.id, reason="test", actor="t", user_id="alice")
    with pytest.raises(Exception, match="facts remain"):
        await memory._store.erase_scope_residue(ALICE, reason="test", actor="t")
    conn = memory._store._conn
    async with conn.execute("SELECT COUNT(*) FROM messages WHERE user_id='alice'") as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == 1  # nothing deleted by the refusal
    assert await memory._store.list_erasures(ALICE) == []  # and nothing recorded


async def test_delete_session_then_reuse_id_is_withheld_after_revoked_fact(
    memory: Engram,
) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    msg = await memory.record_message("hi", session_id="s1", user_id="alice", timestamp=t)
    fact = Fact(
        text="alice likes tea", scope=ALICE, valid_from=t, session_id="s1", source_message_id=msg.id
    )
    await memory._store.upsert_fact(fact)
    await memory.revoke(fact.id, reason="test", actor="t", user_id="alice")
    assert await memory._store.delete_session_messages("s1", ALICE) == 1
    await memory.record_message("new", session_id="s1", user_id="alice", timestamp=t)
    # source_message_id is now NULL; the view withholds every turn in that session id.
    assert await memory._store.list_session_ids(ALICE) == []


async def _count(memory: Engram, table: str, user: str) -> int:
    conn = memory._store._conn
    async with conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id=?", (user,)) as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def test_erase_scope_residue_clears_scope_only(memory: Engram) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    fact = await memory.record("alice likes tea", user_id="alice")
    gone = await memory.record("alice likes jazz", user_id="alice")
    await memory.revoke(gone.id, reason="test", actor="t", user_id="alice")
    await memory.record_message("hi", session_id="s1", user_id="alice", timestamp=t)
    await memory.record_message("hey", session_id="s1", user_id="bob", timestamp=t)
    await memory._store.upsert_event(
        Event(
            scope=ALICE, subject_canonical="alice", verb="met", object_canonical="bob", time_start=t
        )
    )
    conn = memory._store._conn
    for user, ent in (("alice", "e-a"), ("bob", "e-b")):
        await conn.execute(
            "INSERT INTO entities (id, org_id, user_id, name) VALUES (?,?,?,?)",
            (ent, "default", user, "x"),
        )
        # No store upsert exists for relationships, so seed the table directly.
        await conn.execute(
            "INSERT INTO relationships (id, org_id, user_id, source, relation, target) "
            "VALUES (?,?,?,?,?,?)",
            (f"r-{user}", "default", user, "a", "knows", "b"),
        )
    await conn.execute(
        "INSERT INTO fact_entity_refs (fact_id, entity_id) VALUES (?,?)", (str(fact.id), "e-a")
    )
    await conn.commit()
    await memory.erase(fact, reason="test", actor="t", user_id="alice")
    await memory.erase(gone, reason="test", actor="t", user_id="alice")
    erasures_before = await memory._store.list_erasures(ALICE)
    revocations_before = await _count(memory, "revocations", "alice")
    assert erasures_before and revocations_before >= 1

    residue = await memory._store.erase_scope_residue(ALICE, reason="test", actor="t")

    removed = residue.artifacts
    assert removed["messages"] == 1
    assert removed["entities"] == 1
    assert removed["events"] == 1
    assert removed["relationships"] == 1
    for table in ("events", "relationships", "fact_meta", "messages", "entities"):
        assert await _count(memory, table, "alice") == 0, table
    # content-free audit trail stays, plus one record for the residue
    after = await memory._store.list_erasures(ALICE)
    assert [e.id for e in after] == [e.id for e in erasures_before] + [residue.id]
    assert after[-1].fact_ids == [] and after[-1].artifacts == removed
    assert (after[-1].reason, after[-1].actor) == ("test", "t")
    assert await _count(memory, "revocations", "alice") == revocations_before
    # other scope untouched
    assert await memory._store.list_session_ids(BOB) == ["s1"]
    assert await _count(memory, "entities", "bob") == 1
    assert await _count(memory, "relationships", "bob") == 1
    async with conn.execute("SELECT COUNT(*) FROM fact_entity_refs") as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == 0


async def test_erase_scope_residue_scrubs_every_tombstone_in_scope(memory: Engram) -> None:
    conn = memory._store._conn
    # A tombstone whose fact is already gone, still carrying its hash and text.
    for user, rid in (("alice", "rev-a"), ("bob", "rev-b")):
        await conn.execute(
            "INSERT INTO revocations (id, org_id, user_id, fact_id, content_hash, content_norm,"
            " reason, actor, revoked_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rid,
                "default",
                user,
                str(uuid4()),
                "h",
                "n",
                "test",
                "t",
                "2026-11-10T00:00:00+00:00",
            ),
        )
    await conn.commit()

    residue = await memory._store.erase_scope_residue(ALICE, reason="test", actor="t")

    assert residue.artifacts["revocations_scrubbed"] == 1
    async with conn.execute(
        "SELECT user_id, content_hash, content_norm FROM revocations ORDER BY user_id"
    ) as cur:
        rows = [tuple(r) for r in await cur.fetchall()]
    assert rows == [("alice", None, None), ("bob", "h", "n")]  # row kept, content gone
