"""Engram methods behind the v1 REST routes."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from engram import Engram
from engram.scope import Scope


@pytest.fixture
async def memory():
    async with await Engram.open(":memory:") as m:
        yield m


async def test_forget_revokes_without_naming_the_scope(memory: Engram) -> None:
    fact = await memory.record("alice likes tea", user_id="alice")
    assert await memory.forget(fact.id, reason="user asked", actor="rest:v1") is True
    assert await memory.recall("tea", user_id="alice") == []
    revs = await memory.revocations(user_id="alice")
    assert [r.fact_id for r in revs] == [fact.id]  # soft: a tombstone, not an erasure


async def test_forget_unknown_id_is_false(memory: Engram) -> None:
    from uuid import uuid4

    assert await memory.forget(uuid4(), reason="x", actor="rest:v1") is False


async def test_erase_user_leaves_no_rows_in_scope(memory: Engram) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    await memory.record("alice likes tea", user_id="alice")
    revoked = await memory.record("alice likes jazz", user_id="alice")
    await memory.revoke(revoked.id, reason="test", actor="t", user_id="alice")
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=t)
    keep = await memory.record("bob likes chess", user_id="bob")
    await memory.record_message("hey", session_id="s1", user_id="bob", timestamp=t)

    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 2

    store = memory._store
    async with store._reading() as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%fts%'"
        ) as cur:
            tables = [r[0] for r in await cur.fetchall()]
        for table in tables:
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                cols = {r[1] for r in await cur.fetchall()}
            if not {"org_id", "user_id"} <= cols:
                continue
            async with db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE org_id='default' AND user_id='alice'"
            ) as cur:
                n = (await cur.fetchone())[0]
            if table in ("erasures", "revocations"):
                continue  # content-free audit records, checked below
            assert n == 0, f"{table} still has {n} rows for alice"
        async with db.execute(
            "SELECT content_hash, content_norm FROM revocations WHERE user_id='alice'"
        ) as cur:
            assert all(r[0] is None and r[1] is None for r in await cur.fetchall())

    assert [sf.fact.id for sf in await memory.recall("chess", user_id="bob")] == [keep.id]
    assert len(await memory.list_messages("s1", user_id="bob")) == 1


async def test_conversations(memory: Engram) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    await memory.record_message("hi", session_id="c1", user_id="alice", timestamp=t)
    assert await memory.list_conversations(user_id="alice") == ["c1"]
    assert [m.content for m in await memory.list_messages("c1", user_id="alice")] == ["hi"]
    assert await memory.delete_conversation("c1", user_id="alice") == 1
    assert await memory.list_conversations(user_id="alice") == []


async def test_forget_refuses_a_blank_reason_before_looking_up_the_id(memory: Engram) -> None:
    from uuid import uuid4

    with pytest.raises(ValueError, match="reason"):
        await memory.forget(uuid4(), reason=" ", actor="rest:v1")


async def test_erase_user_skips_ids_already_erased_as_copies(memory: Engram) -> None:
    await memory.record("alice likes tea", user_id="alice")
    await memory.record("alice likes tea", user_id="alice")  # a copy, erased with the first
    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 2
    assert len(await memory.erasures(user_id="alice")) == 1


async def test_erase_user_refuses_a_blank_actor_before_deleting(memory: Engram) -> None:
    t = datetime(2026, 11, 10, tzinfo=UTC)
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=t)
    with pytest.raises(ValueError, match="actor"):
        await memory.erase_user(user_id="alice", reason="gdpr request", actor=" ")
    assert len(await memory.list_messages("s1", user_id="alice")) == 1


async def test_erase_user_stays_inside_its_org(memory: Engram) -> None:
    home = await memory.record("alice likes tea", user_id="alice")
    await memory.record("alice likes tea", user_id="alice", org_id="acme")
    assert (
        await memory.erase_user(
            user_id="alice", org_id="acme", reason="gdpr request", actor="rest:v1"
        )
        == 1
    )
    assert [sf.fact.id for sf in await memory.recall("tea", user_id="alice")] == [home.id]
    assert await memory.recall("tea", user_id="alice", org_id="acme") == []


async def test_erase_user_refuses_the_residue_step_if_a_fact_lands_mid_way(
    memory: Engram, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engram.errors import StoreError

    t = datetime(2026, 11, 10, tzinfo=UTC)
    await memory.record("alice likes tea", user_id="alice")
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=t)
    listed = memory._store.list_fact_ids

    async def list_then_write(scope: Scope) -> list[UUID]:
        ids = await listed(scope)
        await memory.record("alice likes jazz", user_id="alice")
        return ids

    monkeypatch.setattr(memory._store, "list_fact_ids", list_then_write)
    with pytest.raises(StoreError, match="facts remain"):
        await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1")
    assert len(await memory.list_messages("s1", user_id="alice")) == 1  # residue untouched

    monkeypatch.setattr(memory._store, "list_fact_ids", listed)
    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 1
    assert await memory.list_messages("s1", user_id="alice") == []
