"""Memory stored under scope ids the rules now refuse.

0.2 took any id of 1 to 128 characters, so a database it wrote can hold
memory under ids such as "alice " or "a\\x00b". No normal call can name such a
scope. `Engram.refused_scopes` lists them and `Engram.erase_refused_scope`
erases one, taking the ids as stored. Every other call still refuses them.

The tests stand for a 0.2 row by writing under a valid scope and renaming its
ids in SQL, in every table, so they need SQLite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import aiosqlite
import pytest

from engram import Engram
from engram.errors import ReadOnlyStoreError, ScopeRequiredError
from engram.models import Event
from engram.scope import Scope
from tests.conformance.backends import OpenMemory

pytestmark = pytest.mark.sqlite_only

T = datetime(2026, 11, 10, tzinfo=UTC)
WRITTEN = ("acme", "alice")

# Ids 0.2 accepted and the rules refuse, as (org_id, user_id).
REFUSED = [
    pytest.param(("acme", "alice "), id="trailing-space"),
    pytest.param(("acme", " "), id="blank"),
    pytest.param(("acme", "\talice"), id="leading-tab"),
    pytest.param(("acme", "alice\n"), id="newline"),
    pytest.param(("acme", "a\x00b"), id="nul"),
    pytest.param(("acme ", "alice"), id="org-trailing-space"),
]

# Scoped tables that hold memory, which an erase of the scope empties.
MEMORY_TABLES = ("facts", "fact_meta", "messages", "events", "entities", "relationships")
# Scoped tables that stay after an erase as the content-free audit trail.
# A tombstone counts as memory while it still holds a content hash or text.
AUDIT_TABLES = ("revocations", "erasures", "scope_state")


@pytest.fixture
async def memory(open_memory: OpenMemory) -> Engram:
    return await open_memory()


def _conn(memory: Engram) -> aiosqlite.Connection:
    conn: aiosqlite.Connection = memory._store._conn  # type: ignore[attr-defined]
    return conn


async def _scalar(memory: Engram, sql: str, args: tuple[Any, ...] = ()) -> int:
    async with _conn(memory).execute(sql, args) as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def _scoped_tables(memory: Engram) -> set[str]:
    """Every table with org_id and user_id columns, FTS tables left out."""
    async with _conn(memory).execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
        names = [r[0] for r in await cur.fetchall()]
    out = set()
    for name in names:
        if "_fts" in name:
            continue
        async with _conn(memory).execute(f"PRAGMA table_info({name})") as cur:
            cols = {r[1] for r in await cur.fetchall()}
        if {"org_id", "user_id"} <= cols:
            out.add(name)
    return out


async def _seed(memory: Engram) -> None:
    """Memory in every scoped table of the WRITTEN scope."""
    org, user = WRITTEN
    tea = await memory.record("alice likes green tea", org_id=org, user_id=user)
    await memory.record(
        "alice drinks oolong daily", org_id=org, user_id=user, derived_from=[tea.id]
    )
    jazz = await memory.record("alice plays jazz piano", org_id=org, user_id=user)
    await memory.revoke(jazz.id, reason="no longer true", actor="t", org_id=org, user_id=user)
    await memory.record_message(
        "my locker code is 4417", session_id="s1", org_id=org, user_id=user, timestamp=T
    )
    await memory._store.upsert_event(
        Event(
            scope=Scope(org_id=org, user_id=user),
            subject_canonical="alice",
            verb="met",
            object_canonical="carol",
            time_start=T,
        )
    )
    conn = _conn(memory)
    await conn.execute(
        "INSERT INTO entities (id, org_id, user_id, name) VALUES (?,?,?,?)",
        ("e-alice", org, user, "alice"),
    )
    await conn.execute(
        "INSERT INTO relationships (id, org_id, user_id, source, relation, target) "
        "VALUES (?,?,?,?,?,?)",
        ("r-alice", org, user, "alice", "knows", "carol"),
    )
    await conn.commit()


async def _store_under(memory: Engram, stored: tuple[str, str], *tables: str) -> None:
    """Rename the WRITTEN scope's ids to ``stored`` in ``tables`` (all scoped ones by default)."""
    conn = _conn(memory)
    for table in tables or sorted(await _scoped_tables(memory)):
        await conn.execute(
            f"UPDATE {table} SET org_id = ?, user_id = ? WHERE org_id = ? AND user_id = ?",
            (*stored, *WRITTEN),
        )
    await conn.commit()


async def _rows_under(memory: Engram, stored: tuple[str, str], table: str) -> int:
    return await _scalar(
        memory, f"SELECT COUNT(*) FROM {table} WHERE org_id = ? AND user_id = ?", stored
    )


async def _neighbours(memory: Engram) -> dict[tuple[str, str], str]:
    """Valid scopes written after the rename, a look-alike of the stored ids among them."""
    out = {}
    for org, user, text in (
        ("acme", "alice", "alice likes chess"),
        ("acme", "bob", "bob likes green tea"),
        ("globex", "alice", "alice likes green tea"),
    ):
        fact = await memory.record(text, org_id=org, user_id=user)
        await memory.record_message(text, session_id="s1", org_id=org, user_id=user, timestamp=T)
        out[(org, user)] = str(fact.id)
    return out


async def _neighbours_intact(memory: Engram, neighbours: dict[tuple[str, str], str]) -> None:
    for (org, user), fact_id in neighbours.items():
        handle = memory.scoped(org_id=org, user_id=user)
        assert [str(i) for i in await memory._store.list_fact_ids(handle.scope)] == [fact_id]
        assert len(await handle.list_messages("s1")) == 1
        assert await handle.erasures() == []


# ── Listing ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stored", REFUSED)
async def test_refused_scopes_lists_memory_stored_under_ids_the_rules_refuse(
    memory: Engram, stored: tuple[str, str]
) -> None:
    with pytest.raises(ValueError):
        Scope(org_id=stored[0], user_id=stored[1])  # the rules refuse it
    await _seed(memory)
    await _store_under(memory, stored)
    await _neighbours(memory)
    assert await memory.refused_scopes() == [stored]


async def test_refused_scopes_is_empty_when_every_id_is_accepted(memory: Engram) -> None:
    await _seed(memory)
    await _neighbours(memory)
    await memory.record("the default user likes tea")
    assert await memory.refused_scopes() == []


async def test_refused_scopes_covers_every_scoped_table(memory: Engram) -> None:
    # A new scoped table fails here until it is sorted into one of the two lists.
    assert await _scoped_tables(memory) == set(MEMORY_TABLES) | set(AUDIT_TABLES)


@pytest.mark.parametrize("table", MEMORY_TABLES)
async def test_refused_scopes_finds_memory_in_any_one_table(memory: Engram, table: str) -> None:
    stored = ("acme", "alice ")
    await _seed(memory)
    await _store_under(memory, stored, table)
    assert await _rows_under(memory, stored, table) > 0
    assert await memory.refused_scopes() == [stored]


async def test_a_tombstone_counts_while_it_holds_content(memory: Engram) -> None:
    stored = ("acme", "alice ")
    await _seed(memory)
    await _store_under(memory, stored, "revocations", "erasures", "scope_state")
    assert await memory.refused_scopes() == [stored]  # the tombstone keeps a hash and text
    await _conn(memory).execute(
        "UPDATE revocations SET content_hash = NULL, content_norm = NULL"
        " WHERE org_id = ? AND user_id = ?",
        stored,
    )
    await _conn(memory).execute(
        "INSERT INTO scope_state (org_id, user_id, erase_epoch) VALUES (?, ?, 1)", stored
    )
    await _conn(memory).commit()
    assert await memory.refused_scopes() == []  # audit rows only


# ── Erasing ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stored", REFUSED)
async def test_erase_refused_scope_erases_it_and_nothing_else(
    memory: Engram, stored: tuple[str, str]
) -> None:
    await _seed(memory)
    await _store_under(memory, stored)
    neighbours = await _neighbours(memory)
    for table in MEMORY_TABLES:
        assert await _rows_under(memory, stored, table) > 0, table

    org, user = stored
    erased = await memory.erase_refused_scope(
        org_id=org, user_id=user, reason="erasure request", actor="ops"
    )

    assert erased == 3
    for table in MEMORY_TABLES:
        assert await _rows_under(memory, stored, table) == 0, table
    match = "SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?"
    for token in ("oolong", "jazz", "piano"):
        assert await _scalar(memory, match, (token,)) == 0, token
    assert await _rows_under(memory, stored, "events_fts") == 0
    # The audit trail, as erase_user leaves it: a tombstone without its
    # content, an erasure record per erase and one for the residue, the epoch moved.
    assert (
        await _scalar(
            memory,
            "SELECT COUNT(*) FROM revocations WHERE org_id = ? AND user_id = ?"
            " AND content_hash IS NULL AND content_norm IS NULL",
            stored,
        )
        == 1
    )
    assert await _rows_under(memory, stored, "erasures") == 3
    assert (
        await _scalar(
            memory,
            "SELECT erase_epoch FROM scope_state WHERE org_id = ? AND user_id = ?",
            stored,
        )
        == 1
    )
    assert await memory.refused_scopes() == []
    await _neighbours_intact(memory, neighbours)


async def test_erasing_a_refused_scope_twice_is_safe(memory: Engram) -> None:
    stored = ("acme", "alice ")
    await _seed(memory)
    await _store_under(memory, stored)
    kw: dict[str, str] = {"reason": "erasure request", "actor": "ops"}
    assert await memory.erase_refused_scope(org_id="acme", user_id="alice ", **kw) == 3
    assert await memory.erase_refused_scope(org_id="acme", user_id="alice ", **kw) == 0
    assert await memory.refused_scopes() == []


async def test_erase_refused_scope_refuses_a_reason_that_quotes_the_memory(
    memory: Engram,
) -> None:
    stored = ("acme", "alice ")
    await _seed(memory)
    await _store_under(memory, stored)
    with pytest.raises(ValueError, match="must not quote"):
        await memory.erase_refused_scope(
            org_id="acme", user_id="alice ", reason="alice likes green tea", actor="ops"
        )
    assert await _rows_under(memory, stored, "facts") == 3
    assert await _rows_under(memory, stored, "erasures") == 0


# ── Every other call still refuses the ids ───────────────────────────────


async def test_normal_calls_still_refuse_a_refused_id(memory: Engram) -> None:
    stored = ("acme", "alice ")
    await _seed(memory)
    await _store_under(memory, stored)
    async with _conn(memory).execute(
        "SELECT id FROM facts WHERE org_id = ? AND user_id = ?", stored
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    fact_id = UUID(row[0])
    ids = {"org_id": "acme", "user_id": "alice "}
    calls: dict[str, Any] = {
        "erase_user": lambda: memory.erase_user(reason="erasure request", actor="ops", **ids),
        "scoped": lambda: memory.scoped(**ids),
        "recall": lambda: memory.recall("tea", **ids),
        "record": lambda: memory.record("alice likes coffee", **ids),
        "erasures": lambda: memory.erasures(**ids),
        "forget_in_org": lambda: memory.forget_in_org(
            fact_id, org_id="acme", reason="wrong", actor="ops"
        ),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError):
            result = call()
            if hasattr(result, "__await__"):
                await result
            pytest.fail(f"{name} accepted a refused id")
    # The look-alike valid scope is another scope: erasing it leaves the stored one.
    assert await memory.erase_user(org_id="acme", user_id="alice", reason="r", actor="ops") == 0
    assert await _rows_under(memory, stored, "facts") == 3
    assert await _rows_under(memory, stored, "erasures") == 0
    assert await _rows_under(memory, stored, "revocations") == 1
    assert await memory.refused_scopes() == [stored]


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param({"org_id": "acme", "user_id": "alice"}, id="valid"),
        pytest.param({"org_id": "acme", "user_id": "default"}, id="default-user"),
        pytest.param({"org_id": "default", "user_id": "default"}, id="both-default"),
    ],
)
async def test_erase_refused_scope_refuses_ids_the_rules_accept(
    memory: Engram, ids: dict[str, str]
) -> None:
    fact = await memory.record("someone likes green tea", **ids)
    with pytest.raises(ValueError, match="erase_user"):
        await memory.erase_refused_scope(reason="erasure request", actor="ops", **ids)
    assert [h.fact.id for h in await memory.recall("green tea", **ids)] == [fact.id]
    assert await memory.erasures(**ids) == []


@pytest.mark.persistent
async def test_a_read_only_engram_refuses_before_it_checks_the_ids(
    open_memory: OpenMemory,
) -> None:
    writer = await open_memory()
    await _seed(writer)
    await _store_under(writer, ("acme", "alice "))
    await writer.close()
    reader = await open_memory.reopen(read_only=True)
    assert await reader.refused_scopes() == [("acme", "alice ")]  # a read
    for user in ("alice ", "alice", ""):
        with pytest.raises(ReadOnlyStoreError):
            await reader.erase_refused_scope(
                org_id="acme", user_id=user, reason="erasure request", actor="ops"
            )
    assert await reader.refused_scopes() == [("acme", "alice ")]


class _Str(str):
    pass


@pytest.mark.parametrize(
    ("ids", "error", "message"),
    [
        pytest.param({"user_id": "alice "}, TypeError, "org_id", id="no-org"),
        pytest.param({"org_id": "acme"}, TypeError, "user_id", id="no-user"),
        pytest.param(
            {"org_id": None, "user_id": "alice "}, ScopeRequiredError, "string", id="none-org"
        ),
        pytest.param(
            {"org_id": "acme", "user_id": b"alice "}, ScopeRequiredError, "string", id="bytes"
        ),
        pytest.param(
            {"org_id": "acme", "user_id": _Str("alice ")},
            ScopeRequiredError,
            "string",
            id="str-subclass",
        ),
        pytest.param({"org_id": "", "user_id": "alice "}, ScopeRequiredError, "empty", id="empty"),
        pytest.param(
            {"org_id": "acme", "user_id": "x" * 128 + " "}, ValueError, "128", id="too-long"
        ),
        pytest.param(
            {"org_id": "acme", "user_id": "alice\ud800"}, ValueError, "surrogate", id="surrogate"
        ),
    ],
)
async def test_erase_refused_scope_refuses_ids_no_store_holds(
    memory: Engram, ids: dict[str, Any], error: type[Exception], message: str
) -> None:
    await _seed(memory)
    await _store_under(memory, ("acme", "alice "))
    with pytest.raises(error, match=message):
        await memory.erase_refused_scope(reason="erasure request", actor="ops", **ids)
    assert await _rows_under(memory, ("acme", "alice "), "facts") == 3
    assert await _rows_under(memory, ("acme", "alice "), "erasures") == 0
