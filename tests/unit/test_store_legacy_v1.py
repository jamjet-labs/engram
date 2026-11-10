"""Store methods behind the v1 REST routes the Java client calls."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from engram import Engram
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
        await memory._store.erase_scope_residue(ALICE)


async def test_erase_scope_residue_clears_scope_only(memory: Engram) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    fact = await memory.record("alice likes tea", user_id="alice")
    await memory.record_message("hi", session_id="s1", user_id="alice", timestamp=t)
    await memory.record_message("hey", session_id="s1", user_id="bob", timestamp=t)
    conn = memory._store._conn
    for scope_user, ent in (("alice", "e-a"), ("bob", "e-b")):
        await conn.execute(
            "INSERT INTO entities (id, org_id, user_id, name) VALUES (?,?,?,?)",
            (ent, "default", scope_user, "x"),
        )
    await conn.execute(
        "INSERT INTO fact_entity_refs (fact_id, entity_id) VALUES (?,?)", (str(fact.id), "e-a")
    )
    await conn.commit()
    await memory.erase(fact, reason="test", actor="t", user_id="alice")

    removed = await memory._store.erase_scope_residue(ALICE)

    assert removed["messages"] == 1
    assert removed["entities"] == 1
    assert await memory._store.list_session_ids(ALICE) == []
    assert await memory._store.list_session_ids(BOB) == ["s1"]
    async with conn.execute("SELECT COUNT(*) FROM entities WHERE user_id='bob'") as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == 1
    async with conn.execute("SELECT COUNT(*) FROM fact_entity_refs") as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == 0
