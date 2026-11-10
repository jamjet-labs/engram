"""Engram methods behind the v1 REST routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import aiosqlite
import pytest

from engram import Engram
from engram.errors import ScopeResidueRefusedError, StoreError
from engram.models import Event
from engram.scope import Scope

ALICE = Scope(org_id="default", user_id="alice")
BOB = Scope(org_id="default", user_id="bob")
T = datetime(2026, 11, 10, tzinfo=UTC)

# Every table, so a new one fails the erasure test until it is classified.
SCOPED_TABLES = {
    "entities",
    "erasures",
    "events",
    "events_fts",
    "fact_meta",
    "facts",
    "facts_fts",
    "messages",
    "relationships",
    "revocations",
}
UNSCOPED_TABLES = {"content_matches", "fact_derivations", "fact_entity_refs", "schema_version"}
FTS_SHADOW_PREFIXES = ("facts_fts_", "events_fts_")


@pytest.fixture
async def memory():
    async with await Engram.open(":memory:") as m:
        yield m


async def _scalar(db: aiosqlite.Connection, sql: str, args: tuple[Any, ...] = ()) -> int:
    async with db.execute(sql, args) as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def _ids(db: aiosqlite.Connection, sql: str, args: tuple[Any, ...] = ()) -> set[str]:
    async with db.execute(sql, args) as cur:
        return {r[0] for r in await cur.fetchall()}


async def _scoped_count(db: aiosqlite.Connection, table: str, user: str) -> int:
    return await _scalar(
        db, f"SELECT COUNT(*) FROM {table} WHERE org_id='default' AND user_id=?", (user,)
    )


async def _seed_graph(memory: Engram, user: str, fact_id: UUID | None) -> None:
    """Entities and relationships have no store upsert, so seed the tables directly."""
    conn = memory._store._conn
    await conn.execute(
        "INSERT INTO entities (id, org_id, user_id, name) VALUES (?,?,?,?)",
        (f"e-{user}", "default", user, "x"),
    )
    await conn.execute(
        "INSERT INTO relationships (id, org_id, user_id, source, relation, target) "
        "VALUES (?,?,?,?,?,?)",
        (f"r-{user}", "default", user, "a", "knows", "b"),
    )
    if fact_id is not None:
        await conn.execute(
            "INSERT INTO fact_entity_refs (fact_id, entity_id) VALUES (?,?)",
            (str(fact_id), f"e-{user}"),
        )
    await conn.commit()


# ── forget ───────────────────────────────────────────────────────────────


async def test_forget_revokes_without_naming_the_scope(memory: Engram) -> None:
    fact = await memory.record("alice likes tea", user_id="alice")
    assert await memory.forget(fact.id, reason="user asked", actor="rest:v1") is True
    assert await memory.recall("tea", user_id="alice") == []
    revs = await memory.revocations(user_id="alice")
    assert [r.fact_id for r in revs] == [fact.id]  # soft: a tombstone, not an erasure


async def test_forget_unknown_id_is_false(memory: Engram) -> None:
    assert await memory.forget(uuid4(), reason="x", actor="rest:v1") is False


async def test_forget_revokes_in_the_facts_own_org(memory: Engram) -> None:
    fact = await memory.record("alice likes tea", user_id="alice", org_id="acme")
    assert await memory.forget(fact.id, reason="user asked", actor="rest:v1") is True
    assert [r.fact_id for r in await memory.revocations(user_id="alice", org_id="acme")] == [
        fact.id
    ]


async def test_forget_is_false_when_the_fact_is_erased_before_the_revoke(
    memory: Engram, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stale_lookup(fact_id: UUID) -> Scope:
        return ALICE  # the owner, as read just before a concurrent erase

    monkeypatch.setattr(memory._store, "find_fact_scope", stale_lookup)
    assert await memory.forget(uuid4(), reason="user asked", actor="rest:v1") is False


@pytest.mark.parametrize(
    ("reason", "actor", "field"), [(" ", "rest:v1", "reason"), ("x", " ", "actor")]
)
async def test_forget_refuses_a_blank_field_before_looking_up_the_id(
    memory: Engram, reason: str, actor: str, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        await memory.forget(uuid4(), reason=reason, actor=actor)


# ── erase_user ───────────────────────────────────────────────────────────


async def test_erase_user_leaves_no_rows_in_scope(memory: Engram) -> None:
    tea = await memory.record("alice likes tea", user_id="alice")
    revoked = await memory.record("alice likes jazz music", user_id="alice")
    await memory.revoke(revoked.id, reason="test", actor="t", user_id="alice")
    derived = await memory.record("alice drinks oolong", user_id="alice", derived_from=[tea.id])
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=T)
    await memory.record_message(
        "she said alice likes jazz music", session_id="s1", user_id="alice", timestamp=T
    )
    await memory._store.upsert_event(
        Event(
            scope=ALICE,
            subject_canonical="alice",
            verb="met",
            object_canonical="carol",
            time_start=T,
        )
    )
    await _seed_graph(memory, "alice", tea.id)
    keep = await memory.record("bob likes chess", user_id="bob")
    await memory.record_message("hey", session_id="s1", user_id="bob", timestamp=T)
    await memory._store.upsert_event(
        Event(scope=BOB, subject_canonical="bob", verb="met", object_canonical="dave", time_start=T)
    )
    await _seed_graph(memory, "bob", keep.id)

    store = memory._store
    async with store._reading() as db:
        tables = await _ids(db, "SELECT name FROM sqlite_master WHERE type='table'")
        tables = {t for t in tables if not t.startswith(FTS_SHADOW_PREFIXES)}
        assert tables == SCOPED_TABLES | UNSCOPED_TABLES
        for table in sorted(tables):
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                cols = {r[1] for r in await cur.fetchall()}
            assert ({"org_id", "user_id"} <= cols) == (table in SCOPED_TABLES), table
        # The seed reaches every scoped table, so an empty table cannot pass for erased.
        for table in sorted(SCOPED_TABLES - {"erasures"}):
            assert await _scoped_count(db, table, "alice") > 0, table
        fact_ids = {str(i) for i in await store.list_fact_ids(ALICE)}
        assert fact_ids == {str(tea.id), str(revoked.id), str(derived.id)}
        message_ids = await _ids(db, "SELECT id FROM messages WHERE user_id='alice'")
        event_ids = await _ids(db, "SELECT id FROM events WHERE user_id='alice'")
        rev_ids = await _ids(db, "SELECT id FROM revocations WHERE user_id='alice'")
        item_ids = fact_ids | message_ids | event_ids
        assert await _scalar(db, "SELECT COUNT(*) FROM fact_derivations") == 1
        assert await _scalar(db, "SELECT COUNT(*) FROM fact_entity_refs") == 2
        assert await _scalar(db, "SELECT COUNT(*) FROM content_matches") > 0
        for token in ("alice", "jazz", "oolong"):
            n = await _scalar(
                db, "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", (token,)
            )
            assert n > 0, token

    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 3

    async with store._reading() as db:
        for table in sorted(SCOPED_TABLES - {"erasures", "revocations", "facts_fts"}):
            assert await _scoped_count(db, table, "alice") == 0, table
        for token in ("alice", "jazz", "oolong", "tea"):
            n = await _scalar(
                db, "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", (token,)
            )
            assert n == 0, token
        assert (
            await _scalar(db, "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH 'chess'") == 1
        )
        assert await _scoped_count(db, "events_fts", "bob") == 1

        marks = ",".join("?" * len(item_ids))
        refs = await _scalar(
            db,
            f"SELECT COUNT(*) FROM fact_entity_refs WHERE fact_id IN ({marks}) OR entity_id=?",
            (*item_ids, "e-alice"),
        )
        derivations = await _scalar(
            db,
            f"SELECT COUNT(*) FROM fact_derivations"
            f" WHERE fact_id IN ({marks}) OR source_fact_id IN ({marks})",
            (*item_ids, *item_ids),
        )
        rev_marks = ",".join("?" * len(rev_ids))
        matches = await _scalar(
            db,
            f"SELECT COUNT(*) FROM content_matches"
            f" WHERE item_id IN ({marks}) OR revocation_id IN ({rev_marks})",
            (*item_ids, *rev_ids),
        )
        assert (refs, derivations, matches) == (0, 0, 0)

        # The content-free audit trail: one tombstone, three erasure records.
        async with db.execute(
            "SELECT content_hash, content_norm FROM revocations"
            " WHERE org_id='default' AND user_id='alice'"
        ) as cur:
            assert [tuple(r) for r in await cur.fetchall()] == [(None, None)]
    erasures = await memory.erasures(user_id="alice")
    assert len(erasures) == 3
    assert {i for e in erasures for i in e.fact_ids} == {tea.id, revoked.id, derived.id}
    [residue] = [e for e in erasures if not e.fact_ids]
    assert residue.artifacts == {
        "messages": 1,
        "events": 1,
        "relationships": 1,
        "entities": 1,
        "fact_meta": 0,
        "revocations_scrubbed": 0,
        "scope_erasure": 1,
    }

    assert [sf.fact.id for sf in await memory.recall("chess", user_id="bob")] == [keep.id]
    assert len(await memory.list_messages("s1", user_id="bob")) == 1
    async with store._reading() as db:
        for table in ("entities", "relationships", "events"):
            assert await _scoped_count(db, table, "bob") == 1, table
        assert await _scalar(db, "SELECT COUNT(*) FROM fact_entity_refs") == 1


async def test_erase_user_clears_a_hash_an_earlier_erase_kept(memory: Engram) -> None:
    fact = await memory.record("alice likes jazz", user_id="alice")
    await memory.revoke(fact.id, reason="test", actor="t", user_id="alice")
    await memory.erase(fact.id, reason="first pass", actor="t", user_id="alice")
    await memory.record("alice likes tea", user_id="alice")
    sql = "SELECT content_hash IS NOT NULL, content_norm IS NOT NULL FROM revocations"
    async with memory._store._reading() as db, db.execute(sql) as cur:
        assert [tuple(r) for r in await cur.fetchall()] == [(1, 0)]  # block_reinsertion kept it

    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 1

    async with memory._store._reading() as db, db.execute(sql) as cur:
        assert [tuple(r) for r in await cur.fetchall()] == [(0, 0)]
    [_, _, residue] = await memory.erasures(user_id="alice")
    assert residue.fact_ids == [] and residue.artifacts["revocations_scrubbed"] == 1


async def test_erase_user_records_the_residue_it_removes(memory: Engram) -> None:
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=T)
    await memory.record_message("again", session_id="s2", user_id="alice", timestamp=T)

    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 0

    [record] = await memory.erasures(user_id="alice")
    assert record.fact_ids == []
    assert (record.reason, record.actor) == ("gdpr request", "rest:v1")
    assert record.artifacts == {
        "messages": 2,
        "events": 0,
        "relationships": 0,
        "entities": 0,
        "fact_meta": 0,
        "revocations_scrubbed": 0,
        "scope_erasure": 1,
    }
    assert "sqlite_secure_delete" in record.guarantees  # finalized like a per-fact erasure
    assert await memory.list_conversations(user_id="alice") == []


@pytest.mark.parametrize(
    ("short_fact", "hidden", "reason", "actor", "field"),
    [
        ("request", False, "gdpr request", "rest:v1", "reason"),
        ("v1", False, "gdpr request", "rest:v1", "actor"),
        ("request", True, "gdpr request", "rest:v1", "reason"),
    ],
)
async def test_erase_user_refuses_a_field_quoting_any_fact_before_deleting(
    memory: Engram, short_fact: str, hidden: bool, reason: str, actor: str, field: str
) -> None:
    # The quoted fact comes second, so a per-fact check alone would erase the first.
    first = await memory.record("alice likes tea", user_id="alice")
    second = await memory.record(short_fact, user_id="alice")
    if hidden:
        await memory.revoke(second.id, reason="test", actor="t", user_id="alice")
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=T)

    with pytest.raises(ValueError, match=field):
        await memory.erase_user(user_id="alice", reason=reason, actor=actor)

    assert await memory._store.list_fact_ids(ALICE) == [first.id, second.id]
    assert len(await memory.list_messages("s1", user_id="alice")) == 1
    assert await memory.erasures(user_id="alice") == []


@pytest.mark.parametrize(
    ("reason", "actor", "field"), [(" ", "rest:v1", "reason"), ("gdpr request", " ", "actor")]
)
async def test_erase_user_refuses_a_blank_field_before_deleting(
    memory: Engram, reason: str, actor: str, field: str
) -> None:
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=T)
    with pytest.raises(ValueError, match=field):
        await memory.erase_user(user_id="alice", reason=reason, actor=actor)
    assert len(await memory.list_messages("s1", user_id="alice")) == 1


async def test_erase_user_skips_ids_already_erased_as_copies(memory: Engram) -> None:
    await memory.record("alice likes tea", user_id="alice")
    await memory.record("alice likes tea", user_id="alice")  # a copy, erased with the first
    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1") == 2
    assert [len(e.fact_ids) for e in await memory.erasures(user_id="alice")] == [2, 0]


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
    await memory.record("alice likes tea", user_id="alice")
    await memory.record_message("hello", session_id="s1", user_id="alice", timestamp=T)
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


# ── conversations and messages ───────────────────────────────────────────


async def test_conversations(memory: Engram) -> None:
    await memory.record_message("hi", session_id="c1", user_id="alice", timestamp=T)
    assert await memory.list_conversations(user_id="alice") == ["c1"]
    assert [m.content for m in await memory.list_messages("c1", user_id="alice")] == ["hi"]
    assert await memory.delete_conversation("c1", user_id="alice") == 1
    assert await memory.list_conversations(user_id="alice") == []


async def test_conversations_stay_inside_their_user_and_org(memory: Engram) -> None:
    for user, org in (("alice", "default"), ("bob", "default"), ("alice", "acme")):
        await memory.record_message("hi", session_id="c1", user_id=user, org_id=org, timestamp=T)
    assert await memory.delete_conversation("c1", user_id="alice", org_id="acme") == 1
    assert await memory.list_conversations(user_id="alice", org_id="acme") == []
    assert await memory.list_conversations(user_id="alice") == ["c1"]
    assert await memory.list_conversations(user_id="bob") == ["c1"]
    assert await memory.list_messages("c1", user_id="alice", org_id="acme") == []
    assert len(await memory.list_messages("c1", user_id="alice")) == 1
    assert len(await memory.list_messages("c1", user_id="bob")) == 1


async def test_list_messages_last_n(memory: Engram) -> None:
    for minute in (2, 3, 1):  # recorded out of order
        await memory.record_message(
            f"m{minute}", session_id="s1", user_id="alice", timestamp=T + timedelta(minutes=minute)
        )

    async def contents(last_n: int | None) -> list[str]:
        return [m.content for m in await memory.list_messages("s1", user_id="alice", last_n=last_n)]

    assert await contents(None) == ["m1", "m2", "m3"]
    assert await contents(2) == ["m2", "m3"]
    assert await contents(1) == ["m3"]
    for bad in (0, -1):
        with pytest.raises(ValueError, match="last_n"):
            await memory.list_messages("s1", user_id="alice", last_n=bad)


async def test_list_messages_has_no_cap_by_default(memory: Engram) -> None:
    for i in range(1001):
        await memory.record_message(
            f"m{i}", session_id="s1", user_id="alice", timestamp=T + timedelta(seconds=i)
        )
    assert len(await memory.list_messages("s1", user_id="alice")) == 1001


async def test_fact_counts(memory: Engram) -> None:
    await memory.record("one", user_id="alice")
    two = await memory.record("two", user_id="bob", org_id="acme")
    await memory.revoke(two.id, reason="test", actor="t", user_id="bob", org_id="acme")
    assert await memory.fact_counts() == (2, 1)


async def test_only_the_residue_refusal_is_a_scope_residue_refused_error(
    memory: Engram, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert issubclass(ScopeResidueRefusedError, StoreError)
    await memory.record("alice likes tea", user_id="alice")
    listed = memory._store.list_fact_ids

    async def list_then_write(scope: Scope) -> list[UUID]:
        ids = await listed(scope)
        await memory.record("alice likes jazz", user_id="alice")
        return ids

    monkeypatch.setattr(memory._store, "list_fact_ids", list_then_write)
    with pytest.raises(ScopeResidueRefusedError, match="facts remain"):
        await memory.erase_user(user_id="alice", reason="gdpr request", actor="rest:v1")
