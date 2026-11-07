"""The store records its schema version and refuses databases from a newer one.

Opening a database recreates the visibility views from this library's own
schema.sql. If an older library did that to a newer database, it would swap
the newer revocation rules for its own older ones. So the check runs before
anything is written.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from engram.errors import StoreError
from engram.store.sqlite import SCHEMA_VERSION, SqliteStore


def _versions(path: Path) -> list[int]:
    with sqlite3.connect(path) as conn:
        return [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY 1")]


async def test_open_records_the_current_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    store = await SqliteStore.open(db)
    await store.close()
    assert SCHEMA_VERSION >= 2
    assert max(_versions(db)) == SCHEMA_VERSION


async def test_database_from_a_0_2_library_opens_and_is_upgraded(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version VALUES (1)")
    store = await SqliteStore.open(db)
    await store.close()
    assert _versions(db) == [1, SCHEMA_VERSION]


async def test_newer_database_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    db = tmp_path / "new.db"
    store = await SqliteStore.open(db)
    await store.close()
    sentinel = "CREATE VIEW hidden_facts AS SELECT 'from-a-newer-engram' AS id"
    with sqlite3.connect(db) as conn:
        conn.execute("DROP VIEW live_messages")
        conn.execute("DROP VIEW live_events")
        conn.execute("DROP VIEW live_facts")
        conn.execute("DROP VIEW hidden_facts")
        conn.execute(sentinel)
        conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION + 1,))

    with pytest.raises(StoreError, match=r"newer version of Engram"):
        await SqliteStore.open(db)

    with sqlite3.connect(db) as conn:
        [(sql,)] = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'hidden_facts'"
        ).fetchall()
    assert sql == sentinel


async def test_fresh_and_in_memory_databases_open(tmp_path: Path) -> None:
    for target in (":memory:", tmp_path / "fresh.db"):
        store = await SqliteStore.open(target)
        await store.close()
