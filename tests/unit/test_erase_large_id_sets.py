"""Erase and session delete must not bind one SQL variable per id.

SQLite refuses a statement with more than 32,766 variables. These tests lower
that limit on the write connection, so a set a little over it is enough to
show a statement that binds one "?" per id. The last test builds the real
size, 33,000 turns quoting one fact, with direct inserts (about 2 seconds).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import aiosqlite
import pytest

from engram import Engram
from engram.models import Event
from engram.scope import Scope

ALICE = Scope(org_id="default", user_id="alice")
T = datetime(2026, 11, 10, tzinfo=UTC)
LIMIT = 20
OVER = 30  # ids in each set: more than LIMIT, and 2 * OVER for a statement that binds twice
ADDRESS = "alice's home address is 12 Main Street"


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[Engram]:
    async with await Engram.open(tmp_path / "e.db") as m:
        yield m


async def _lower_variable_limit(memory: Engram, n: int = LIMIT) -> None:
    conn = memory._store._conn
    await conn._execute(conn._conn.setlimit, sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, n)


async def test_the_lowered_limit_is_in_effect(memory: Engram) -> None:
    """Control: without it, the tests below would pass against a per-id list too."""
    await _lower_variable_limit(memory)
    marks = ",".join("?" * (LIMIT + 1))
    with pytest.raises(aiosqlite.OperationalError, match="too many SQL variables"):
        await memory._store._conn.execute(f"SELECT 1 WHERE 1 IN ({marks})", range(LIMIT + 1))


async def _quoting_turns(memory: Engram, n: int, session: str = "s1") -> None:
    for i in range(n):
        await memory.record_message(
            f"turn {i}: {ADDRESS}",
            session_id=session,
            user_id="alice",
            timestamp=T + timedelta(seconds=i),
        )


async def test_erase_user_with_more_quoting_turns_than_the_limit(memory: Engram) -> None:
    await memory.record(ADDRESS, user_id="alice")
    await _quoting_turns(memory, OVER)
    await _lower_variable_limit(memory)
    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="dpo") == 1
    assert await memory.list_messages("s1", user_id="alice") == []
    [per_fact, _] = await memory.erasures(user_id="alice")
    assert per_fact.artifacts["messages"] == OVER


async def test_erase_with_more_quoting_facts_than_the_limit(memory: Engram) -> None:
    root = await memory.record(ADDRESS, user_id="alice")
    for i in range(OVER):
        await memory.record(f"note {i}: {ADDRESS}", user_id="alice")
    kept = await memory.record("alice likes tea", user_id="alice")
    await _lower_variable_limit(memory)
    record = await memory.erase(root.id, reason="asked", actor="dpo", user_id="alice")
    assert len(record.fact_ids) == OVER + 1
    assert await memory._store.list_fact_ids(ALICE) == [kept.id]
    async with memory._store._reading() as db:
        for table in ("fact_meta", "fact_derivations"):
            async with db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE fact_id != ?", (str(kept.id),)
            ) as cur:
                row = await cur.fetchone()
            assert row is not None and row[0] == 0, table


async def test_erase_with_more_linked_events_than_the_limit(memory: Engram) -> None:
    root = await memory.record(ADDRESS, user_id="alice")
    for i in range(OVER):
        await memory._store.upsert_event(
            Event(
                scope=ALICE,
                subject_canonical="alice",
                verb="visited",
                object_canonical=f"place {i}",
                time_start=T,
                source_fact_ids=[root.id],
            )
        )
    await _lower_variable_limit(memory)
    record = await memory.erase(root.id, reason="asked", actor="dpo", user_id="alice")
    assert record.artifacts["events"] == OVER
    assert await memory.search_events("place", user_id="alice") == []


async def test_delete_conversation_with_more_turns_than_the_limit(memory: Engram) -> None:
    # A revoked fact the turns quote, so every turn has a revocation match row.
    fact = await memory.record(ADDRESS, user_id="alice")
    await memory.revoke(fact.id, reason="asked", actor="dpo", user_id="alice")
    await _quoting_turns(memory, OVER)
    await memory.record_message("other chat", session_id="s2", user_id="alice", timestamp=T)
    sql = "SELECT COUNT(*) FROM content_matches WHERE kind = 'message'"
    async with memory._store._reading() as db, db.execute(sql) as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == OVER
    await _lower_variable_limit(memory)

    assert await memory.delete_conversation("s1", user_id="alice") == OVER

    async with memory._store._reading() as db, db.execute(sql) as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == 0
    assert [m.content for m in await memory.list_messages("s2", user_id="alice")] == ["other chat"]
    assert await memory.delete_conversation("s1", user_id="alice") == 0


async def _insert_turns(memory: Engram, n: int, session: str) -> None:
    """Direct inserts: the per-turn API takes minutes at this size."""
    rows = [
        (
            str(uuid4()),
            "default",
            "alice",
            session,
            "user",
            f"turn {i}: {ADDRESS}",
            (T + timedelta(microseconds=i)).isoformat(),
        )
        for i in range(n)
    ]
    async with memory._store._transaction():
        await memory._store._conn.executemany(
            "INSERT INTO messages (id, org_id, user_id, session_id, role, content, timestamp)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )


async def test_real_size_erase_user_and_delete_conversation(
    memory: Engram, capsys: pytest.CaptureFixture[str]
) -> None:
    n = 33_000  # over SQLite's default limit of 32,766 variables
    await memory.record(ADDRESS, user_id="alice")
    await _insert_turns(memory, n, "flood")
    started = time.perf_counter()
    assert await memory.erase_user(user_id="alice", reason="gdpr request", actor="dpo") == 1
    erase_s = time.perf_counter() - started
    assert await memory.list_messages("flood", user_id="alice") == []

    await _insert_turns(memory, n, "flood2")
    started = time.perf_counter()
    assert await memory.delete_conversation("flood2", user_id="alice") == n
    delete_s = time.perf_counter() - started
    with capsys.disabled():
        print(f"\n{n} turns: erase_user {erase_s:.1f}s, delete_conversation {delete_s:.1f}s")
