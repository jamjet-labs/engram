"""The store seam: one protocol, a store passed in, read-only opens, numbered migrations.

`Engram` talks to its store only through `EngramStore`, so any store that
meets the protocol can sit behind it. A store opened read-only serves every
read and refuses every write, before any embedder or LLM call and before it
touches the database. SQLite files are upgraded by numbered migrations, all in
the one write transaction that opens the file.

The facade and store tests take the `open_memory` and `open_store` fixtures,
so they run on every backend; those that reopen a target skip a backend that
keeps nothing between opens. The migration and file tests are about SQLite
files and open them directly.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

import pytest

from engram import Engram
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import ReadOnlyStoreError, SchemaVersionError, StoreError
from engram.governance import ErasureRecord, Provenance, Revocation
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Event, Fact
from engram.scope import Scope
from engram.store import EngramStore, SqliteStore, sqlite_migrations
from engram.store import open_store as open_store_fn
from engram.store import sqlite as sqlite_mod
from engram.store.sqlite_migrations import (
    BASELINE_VERSION,
    MIGRATIONS,
    SCHEMA_VERSION,
    Migration,
)
from tests.conformance.backends import OpenMemory, OpenStore

ACME_ALICE = Scope(org_id="acme", user_id="alice")


class Who(TypedDict):
    user_id: str
    org_id: str


WHO: Who = {"user_id": "alice", "org_id": "acme"}

KEPT = "Alice keeps her bike in the blue shed."
NEWER = "Alice moved her bike to the red garage."
REVOKED = "Alice left her bike at the train station."
ERASED = "Alice hides the spare key under the doormat."

_EXTRACTED = (
    '{"facts": [{"text": "Alice bought a new helmet.", "confidence": 0.9}],'
    ' "events": [{"subject_canonical": "alice", "verb": "bought",'
    ' "object_canonical": "helmet", "time_start": "2026-01-05T00:00:00+00:00",'
    ' "time_end": null, "aliases": [], "confidence": 0.9}]}'
)


def _now() -> datetime:
    return datetime.now(UTC)


def _message(text: str = "I bought a new helmet today.") -> ChatMessage:
    return ChatMessage(
        scope=ACME_ALICE, session_id="s1", role="user", content=text, timestamp=_now()
    )


def _event() -> Event:
    return Event(
        scope=ACME_ALICE,
        subject_canonical="alice",
        verb="painted",
        object_canonical="shed",
        time_start=_now(),
    )


class SpyEmbedder:
    """Synthetic vectors, and a record of every batch it was asked to embed."""

    def __init__(self) -> None:
        self._inner = SyntheticEmbedding(dim=384)
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await self._inner.embed(texts)


class SpyLLM:
    """Answers every prompt with one fact and one event, and counts the calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=_EXTRACTED, finish_reason="stop")


@dataclass
class Seeded:
    kept: Fact
    newer: Fact
    revoked: Fact
    erasure: ErasureRecord | None = None
    message: ChatMessage | None = None
    event: Event | None = None


# ── Any store that meets the protocol ───────────────────────────────────

PROTOCOL_METHODS = sorted(
    name
    for name, member in vars(EngramStore).items()
    if not name.startswith("_") and inspect.iscoroutinefunction(member)
)


class DelegatingStore:
    """Has exactly the protocol's members and passes each call to another store.

    If the facade called anything outside `EngramStore`, it would get an
    AttributeError here.
    """

    def __init__(self, inner: EngramStore) -> None:
        self._inner = inner
        self.calls: list[str] = []

    @property
    def read_only(self) -> bool:
        return self._inner.read_only


def _delegate(name: str) -> Callable[..., Awaitable[Any]]:
    async def call(self: DelegatingStore, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(name)
        return await getattr(self._inner, name)(*args, **kwargs)

    call.__name__ = name
    return call


for _name in PROTOCOL_METHODS:
    setattr(DelegatingStore, _name, _delegate(_name))


async def test_engram_accepts_any_store_that_meets_the_protocol(open_store: OpenStore) -> None:
    store = DelegatingStore(await open_store())
    assert isinstance(store, EngramStore)
    memory = await Engram.open(store=store)
    assert memory.read_only is False

    kept = await memory.record(KEPT, **WHO)
    dropped = await memory.record(REVOKED, **WHO)
    gone = await memory.record(ERASED, **WHO)
    found = {h.fact.id for h in await memory.recall("bike", **WHO)}
    assert {kept.id, dropped.id} <= found

    await memory.revoke(dropped, reason="no longer true", actor="test")
    found = {h.fact.id for h in await memory.recall("bike", **WHO)}
    assert kept.id in found
    assert dropped.id not in found

    record = await memory.erase(gone, reason="asked to", actor="test")
    assert record.fact_ids == [gone.id]
    assert gone.id not in {h.fact.id for h in await memory.recall("spare key doormat", **WHO)}
    assert await memory.explain(gone.id, **WHO) is None

    await memory.close()
    # The Engram owns a store it was given and closes it.
    assert store.calls[-1] == "close"
    assert {"upsert_fact", "keyword_search", "revoke", "erase", "finalize_erasure"} <= set(
        store.calls
    )


async def test_store_and_path_together_are_refused(open_store: OpenStore, tmp_path: Path) -> None:
    store = await open_store()
    other = tmp_path / "other.db"
    with pytest.raises(ValueError, match="path or a store"):
        await Engram.open(other, store=store)
    assert not other.exists()


async def test_read_only_needs_a_store_opened_read_only(open_store: OpenStore) -> None:
    store = await open_store()
    with pytest.raises(ValueError, match="read-only"):
        await Engram.open(store=store, read_only=True)


@pytest.mark.persistent
async def test_a_read_only_store_makes_a_read_only_engram(open_store: OpenStore) -> None:
    await (await open_store()).close()
    store = await open_store.reopen(read_only=True)
    memory = await Engram.open(store=store, read_only=True)
    assert memory.read_only is True
    assert (await Engram.open(store=store)).read_only is True


# ── Read-only Engram ────────────────────────────────────────────────────

EngramCall = Callable[[Engram, Seeded], Awaitable[object]]

ENGRAM_WRITES: dict[str, EngramCall] = {
    "record": lambda m, s: m.record("Alice bought a new helmet.", **WHO),
    "record_message": lambda m, s: m.record_message("Thanks!", session_id="s1", **WHO),
    "extract": lambda m, s: m.extract([_message()], persist=True),
    "extract_events": lambda m, s: m.extract_events([_message()], persist=True),
    "supersede": lambda m, s: m.supersede(s.kept.id, s.newer.id, **WHO),
    "revoke": lambda m, s: m.revoke(s.kept, reason="wrong", actor="test"),
    # Without the check this would embed the facts it brings back.
    "reinstate": lambda m, s: m.reinstate(s.revoked, reason="it was right", actor="test"),
    "erase": lambda m, s: m.erase(s.kept, reason="asked to", actor="test"),
    "forget_in_org": lambda m, s: m.forget_in_org(
        s.kept.id, org_id=WHO["org_id"], reason="wrong", actor="test"
    ),
    "erase_user": lambda m, s: m.erase_user(reason="asked to", actor="test", **WHO),
    "erase_refused_scope": lambda m, s: m.erase_refused_scope(
        org_id=WHO["org_id"], user_id="alice ", reason="asked to", actor="test"
    ),
    "delete_conversation": lambda m, s: m.delete_conversation("s1", **WHO),
}

# extract and extract_events with persist=False store nothing, so they are reads.
ENGRAM_READS: dict[str, EngramCall] = {
    "recall": lambda m, s: m.recall("bike", **WHO),
    "context": lambda m, s: m.context("where is the bike", **WHO),
    "context_entries": lambda m, s: m.context_entries("where is the bike", **WHO),
    "search_events": lambda m, s: m.search_events("", **WHO),
    "explain": lambda m, s: m.explain(s.revoked.id, **WHO),
    "revocations": lambda m, s: m.revocations(**WHO),
    "erasures": lambda m, s: m.erasures(**WHO),
    "extract": lambda m, s: m.extract([_message()], persist=False),
    "extract_events": lambda m, s: m.extract_events([_message()], persist=False),
    "list_conversations": lambda m, s: m.list_conversations(**WHO),
    "list_messages": lambda m, s: m.list_messages("s1", **WHO),
    "fact_counts_in_org": lambda m, s: m.fact_counts_in_org(org_id=WHO["org_id"]),
    "refused_scopes": lambda m, s: m.refused_scopes(),
}

ENGRAM_LIFECYCLE = {"open", "close"}


def test_every_public_engram_method_is_listed_as_a_read_or_a_write() -> None:
    public = {
        name
        for name, member in inspect.getmembers(Engram)
        if not name.startswith("_") and inspect.iscoroutinefunction(member)
    }
    # A new method must be added to one of the lists, so the tests below cover it.
    assert public == set(ENGRAM_WRITES) | set(ENGRAM_READS) | ENGRAM_LIFECYCLE


async def _seed_engram(open_memory: OpenMemory) -> Seeded:
    writer = await open_memory()
    seeded = Seeded(
        kept=await writer.record(KEPT, **WHO),
        newer=await writer.record(NEWER, **WHO),
        revoked=await writer.record(REVOKED, **WHO),
    )
    await writer.revoke(seeded.revoked, reason="no longer true", actor="test")
    await writer.record_message("Where is the bike?", session_id="s1", **WHO)
    await writer.close()
    return seeded


async def _visible(memory: Engram) -> tuple[list[UUID], list[UUID], int]:
    hits = await memory.recall("bike", **WHO)
    revocations = await memory.revocations(**WHO, include_reinstated=True)
    return (
        sorted(h.fact.id for h in hits),
        sorted(r.id for r in revocations),
        len(await memory.erasures(**WHO)),
    )


@pytest.mark.persistent
@pytest.mark.parametrize("method", sorted(ENGRAM_WRITES))
async def test_read_only_open_reads_and_refuses_every_write(
    open_memory: OpenMemory, method: str
) -> None:
    seeded = await _seed_engram(open_memory)
    embedder, llm = SpyEmbedder(), SpyLLM()
    memory = await open_memory.reopen(read_only=True, embedder=embedder, llm=llm)
    assert memory.read_only is True

    # The revoked fact stays hidden; the others are found by keyword.
    hits = await memory.recall("bike", **WHO)
    assert sorted(h.fact.id for h in hits) == sorted([seeded.kept.id, seeded.newer.id])
    before = await _visible(memory)
    embedder.calls.clear()

    with pytest.raises(ReadOnlyStoreError):
        await ENGRAM_WRITES[method](memory, seeded)
    assert embedder.calls == []
    assert llm.calls == 0
    assert await _visible(memory) == before


class ClaimsReadOnly(DelegatingStore):
    """Says it is read-only but passes writes on. Only the facade's own check stops them."""

    @property
    def read_only(self) -> bool:
        return True


@pytest.mark.parametrize("method", sorted(ENGRAM_WRITES))
async def test_the_facade_refuses_a_write_before_any_store_or_model_call(
    open_store: OpenStore, method: str
) -> None:
    store = ClaimsReadOnly(await open_store())
    assert isinstance(store, EngramStore)
    seeded = await _seed_store(store)
    store.calls.clear()
    embedder, llm = SpyEmbedder(), SpyLLM()
    memory = await Engram.open(store=store, embedder=embedder, llm=llm, read_only=True)

    with pytest.raises(ReadOnlyStoreError):
        await ENGRAM_WRITES[method](memory, seeded)
    assert store.calls == []
    assert embedder.calls == []
    assert llm.calls == 0


@pytest.mark.persistent
async def test_read_only_open_serves_every_read(open_memory: OpenMemory) -> None:
    seeded = await _seed_engram(open_memory)
    memory = await open_memory.reopen(read_only=True, llm=SpyLLM())
    for name, call in ENGRAM_READS.items():
        try:
            await call(memory, seeded)
        except Exception as e:
            pytest.fail(f"{name} failed on a read-only Engram: {e!r}")


@pytest.mark.persistent
async def test_read_only_recall_does_not_count_access(
    open_memory: OpenMemory, caplog: pytest.LogCaptureFixture
) -> None:
    writer = await open_memory()
    fact = await writer.record(KEPT, **WHO)
    await writer.close()

    caplog.clear()
    reader = await open_memory.reopen(read_only=True)
    hits = await reader.recall("blue shed", **WHO)
    assert [h.fact.id for h in hits] == [fact.id]
    await reader.close()  # waits for any queued access count
    # Nothing tried to write: a refused write would have been logged.
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    writer = await open_memory.reopen()
    [hit] = await writer.recall("blue shed", **WHO)
    assert (hit.fact.access_count, hit.fact.last_accessed) == (0, None)
    await writer.close()
    # Control: a writable recall does count, so the zero above means something.
    [hit] = await (await open_memory.reopen()).recall("blue shed", **WHO)
    assert hit.fact.access_count == 1


# ── Read-only store ─────────────────────────────────────────────────────

StoreCall = Callable[[EngramStore, Seeded], Awaitable[object]]


def _new_fact() -> Fact:
    return Fact(text="Alice bought a new helmet.", scope=ACME_ALICE, valid_from=_now())


def _seeded_erasure(s: Seeded) -> ErasureRecord:
    assert s.erasure is not None
    return s.erasure


def _seeded_event(s: Seeded) -> Event:
    assert s.event is not None
    return s.event


STORE_WRITES: dict[str, StoreCall] = {
    "upsert_fact": lambda st, s: st.upsert_fact(_new_fact()),
    "upsert_message": lambda st, s: st.upsert_message(_message()),
    "upsert_event": lambda st, s: st.upsert_event(_event()),
    "revoke": lambda st, s: st.revoke(s.kept.id, ACME_ALICE, reason="wrong", actor="test"),
    "reinstate": lambda st, s: st.reinstate(
        s.revoked.id, ACME_ALICE, reason="it was right", actor="test"
    ),
    "erase": lambda st, s: st.erase(s.kept.id, ACME_ALICE, reason="asked to", actor="test"),
    "finalize_erasure": lambda st, s: st.finalize_erasure(_seeded_erasure(s), {"vectors": 0}, {}),
    "delete_session_messages": lambda st, s: st.delete_session_messages("s1", ACME_ALICE),
    "erase_scope_residue": lambda st, s: st.erase_scope_residue(
        ACME_ALICE, reason="asked to", actor="test"
    ),
}

STORE_READS: dict[str, StoreCall] = {
    "get_fact": lambda st, s: st.get_fact(s.kept.id, ACME_ALICE),
    "get_facts": lambda st, s: st.get_facts([s.kept.id, s.revoked.id], ACME_ALICE),
    "list_facts_by_session": lambda st, s: st.list_facts_by_session("s1", ACME_ALICE),
    "keyword_search": lambda st, s: st.keyword_search("bike", ACME_ALICE),
    "aggregate_sessions": lambda st, s: st.aggregate_sessions("bike", ACME_ALICE),
    "list_messages": lambda st, s: st.list_messages("s1", ACME_ALICE),
    "get_event": lambda st, s: st.get_event(_seeded_event(s).id, ACME_ALICE),
    "search_events": lambda st, s: st.search_events("", ACME_ALICE),
    "is_content_revoked": lambda st, s: st.is_content_revoked(REVOKED, ACME_ALICE),
    "provenance": lambda st, s: st.provenance([s.kept.id, s.revoked.id], ACME_ALICE),
    "explain": lambda st, s: st.explain(s.revoked.id, ACME_ALICE),
    "derivation_targets_in_scope": lambda st, s: st.derivation_targets_in_scope(
        [s.kept.id, s.revoked.id], ACME_ALICE
    ),
    "list_revocations": lambda st, s: st.list_revocations(ACME_ALICE),
    "list_erasures": lambda st, s: st.list_erasures(ACME_ALICE),
    "hidden_fact_ids": lambda st, s: st.hidden_fact_ids(ACME_ALICE),
    "fact_row_committed": lambda st, s: st.fact_row_committed(s.kept.id, ACME_ALICE),
    "find_fact_owner_in_org": lambda st, s: st.find_fact_owner_in_org(s.kept.id, "acme"),
    "list_fact_ids": lambda st, s: st.list_fact_ids(ACME_ALICE),
    "fact_text_quoted_by": lambda st, s: st.fact_text_quoted_by(KEPT, ACME_ALICE),
    "fact_counts": lambda st, s: st.fact_counts(ACME_ALICE),
    "fact_counts_in_org": lambda st, s: st.fact_counts_in_org("acme"),
    "refused_scopes": lambda st, s: st.refused_scopes(),
    "list_session_ids": lambda st, s: st.list_session_ids(ACME_ALICE),
    "list_session_messages": lambda st, s: st.list_session_messages("s1", ACME_ALICE),
    "scope_guard": lambda st, s: st.scope_guard(ACME_ALICE),
}

# Accepted on a read-only store, and write nothing there.
STORE_NO_WRITE = {"record_access", "flush_access"}
STORE_LIFECYCLE = {"close"}


def test_every_store_protocol_member_is_listed_as_a_read_or_a_write() -> None:
    members = {name for name in vars(EngramStore) if not name.startswith("_")}
    # A new protocol member must be added to one of the lists.
    assert members == (
        set(STORE_WRITES) | set(STORE_READS) | STORE_NO_WRITE | STORE_LIFECYCLE | {"read_only"}
    )


async def _seed_store(store: EngramStore) -> Seeded:
    seeded = Seeded(
        kept=Fact(text=KEPT, scope=ACME_ALICE, valid_from=_now(), session_id="s1"),
        newer=Fact(text=NEWER, scope=ACME_ALICE, valid_from=_now(), session_id="s1"),
        revoked=Fact(text=REVOKED, scope=ACME_ALICE, valid_from=_now(), session_id="s1"),
        message=_message("Where is the bike?"),
        event=_event(),
    )
    erased = Fact(text=ERASED, scope=ACME_ALICE, valid_from=_now())
    for fact in (seeded.kept, seeded.newer, seeded.revoked, erased):
        await store.upsert_fact(fact)
    assert seeded.message is not None and seeded.event is not None
    await store.upsert_message(seeded.message)
    await store.upsert_event(seeded.event)
    await store.revoke(seeded.revoked.id, ACME_ALICE, reason="no longer true", actor="test")
    seeded.erasure = await store.erase(erased.id, ACME_ALICE, reason="asked to", actor="test")
    return seeded


def _summary(value: object) -> Any:
    """What a read returned, as ids and plain values, so two stores' answers compare."""
    if isinstance(value, Fact | ChatMessage | Event | Revocation | ErasureRecord):
        return value.id
    if isinstance(value, Provenance):
        return (value.fact_id, value.revocation_status, value.hidden_reason)
    if isinstance(value, dict):
        return {k: _summary(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted((_summary(v) for v in value), key=str)
    if isinstance(value, list | tuple):
        return [_summary(v) for v in value]
    return value


@pytest.mark.persistent
@pytest.mark.parametrize("method", sorted(STORE_WRITES))
async def test_read_only_store_refuses_every_write(open_store: OpenStore, method: str) -> None:
    writer = await open_store()
    seeded = await _seed_store(writer)
    await writer.close()

    store = await open_store.reopen(read_only=True)
    assert store.read_only is True
    before = {name: _summary(await call(store, seeded)) for name, call in STORE_READS.items()}
    with pytest.raises(ReadOnlyStoreError):
        await STORE_WRITES[method](store, seeded)
    after = {name: _summary(await call(store, seeded)) for name, call in STORE_READS.items()}
    assert after == before


@pytest.mark.persistent
async def test_read_only_store_serves_every_read(open_store: OpenStore) -> None:
    writer = await open_store()
    seeded = await _seed_store(writer)
    expected = {name: _summary(await call(writer, seeded)) for name, call in STORE_READS.items()}
    await writer.close()

    reader = await open_store.reopen(read_only=True)
    got = {name: _summary(await call(reader, seeded)) for name, call in STORE_READS.items()}
    assert got == expected
    # The answers are not empty by accident.
    assert set(expected["keyword_search"]) == {seeded.kept.id, seeded.newer.id}
    assert expected["hidden_fact_ids"] == [seeded.revoked.id]
    assert expected["is_content_revoked"] is True


# ── Read-only SQLite files ──────────────────────────────────────────────


def _ro(db: Path) -> closing[sqlite3.Connection]:
    return closing(sqlite3.connect(db.absolute().as_uri() + "?mode=ro", uri=True))


def _snapshot(db: Path) -> tuple[int, str, list[tuple[Any, ...]]]:
    mtime = db.stat().st_mtime_ns
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    with _ro(db) as conn:
        master = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
    return mtime, digest, master


def _versions(db: Path) -> list[int]:
    with _ro(db) as conn:
        return [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY 1")]


def _tables(db: Path) -> set[str]:
    with _ro(db) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


async def _make_file(db: Path) -> Fact:
    store = await SqliteStore.open(db)
    fact = Fact(text=KEPT, scope=ACME_ALICE, valid_from=_now())
    await store.upsert_fact(fact)
    await store.close()
    return fact


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


# A test-only migration goes after the real ones, so these tests stay true as
# real migrations are added.
PROBE = SCHEMA_VERSION + 1
CURRENT_ROWS = list(range(1, SCHEMA_VERSION + 1))  # each version records its own row
PROBED_ROWS = [*CURRENT_ROWS, PROBE]


def _with_probe(*statements: str) -> tuple[Migration, ...]:
    return (*MIGRATIONS, Migration(version=PROBE, statements=statements))


def _probed_migrations() -> tuple[Migration, ...]:
    return _with_probe("CREATE TABLE seam_probe (id INTEGER PRIMARY KEY, note TEXT NOT NULL)")


async def test_read_only_open_runs_no_ddl(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    fact = await _make_file(db)
    before = _snapshot(db)

    store = await open_store_fn(db, read_only=True)
    assert store.read_only is True
    assert [f.id for f in await store.keyword_search("shed", ACME_ALICE)] == [fact.id]
    await store.close()
    assert _snapshot(db) == before


async def test_read_only_open_refuses_an_older_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "e.db"
    await _make_file(db)
    before = _snapshot(db)
    # This library now knows one more migration than the file has.
    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", _probed_migrations())

    with pytest.raises(SchemaVersionError, match=r"(?i)open it writable once to upgrade"):
        await SqliteStore.open(db, read_only=True)
    assert _snapshot(db) == before

    await (await SqliteStore.open(db)).close()
    assert _versions(db) == PROBED_ROWS
    await (await SqliteStore.open(db, read_only=True)).close()


async def test_read_only_open_refuses_a_newer_schema(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    await _make_file(db)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION + 1,))
        conn.commit()
    before = _snapshot(db)

    with pytest.raises(SchemaVersionError, match="newer version of Engram"):
        await SqliteStore.open(db, read_only=True)
    with pytest.raises(SchemaVersionError, match="newer version of Engram"):
        await SqliteStore.open(db)
    assert _snapshot(db) == before


async def test_read_only_open_refuses_a_file_without_an_engram_schema(tmp_path: Path) -> None:
    db = tmp_path / "other.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE notes (body TEXT)")
        conn.commit()
    before = _snapshot(db)
    with pytest.raises(SchemaVersionError, match=r"(?i)open it writable once"):
        await SqliteStore.open(db, read_only=True)
    assert _snapshot(db) == before


async def test_read_only_open_of_a_missing_file_creates_nothing(tmp_path: Path) -> None:
    db = tmp_path / "missing.db"
    with pytest.raises(StoreError, match="read-only"):
        await SqliteStore.open(db, read_only=True)
    assert _names(tmp_path) == []


async def test_read_only_in_memory_is_refused() -> None:
    with pytest.raises(ValueError, match="in-memory"):
        await SqliteStore.open(":memory:", read_only=True)
    with pytest.raises(ValueError, match="in-memory"):
        await Engram.open(read_only=True)


async def test_read_only_checkpoint_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    await _make_file(db)
    store = await SqliteStore.open(db, read_only=True)
    try:
        with pytest.raises(ReadOnlyStoreError):
            await store.checkpoint()
    finally:
        await store.close()


async def test_connections_wait_out_a_lock_for_30_seconds(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    writer = await SqliteStore.open(db)
    reader = await SqliteStore.open(db, read_only=True)
    try:
        conns = [writer._conn, writer._rconn, reader._conn]
        timeouts = []
        for conn in conns:
            assert conn is not None
            async with conn.execute("PRAGMA busy_timeout") as cur:
                row = await cur.fetchone()
            assert row is not None
            timeouts.append(row[0])
        assert timeouts == [30_000, 30_000, 30_000]
    finally:
        await reader.close()
        await writer.close()


# ── Numbered migrations ─────────────────────────────────────────────────


def _migration_problems(migrations: tuple[Migration, ...], baseline: int) -> list[str]:
    problems = []
    for expected, m in enumerate(migrations, start=baseline + 1):
        if m.version != expected:
            problems.append(f"migration {m.version} where {expected} was expected")
        if not m.statements or not all(isinstance(s, str) and s.strip() for s in m.statements):
            problems.append(f"migration {m.version} has an empty statement")
    return problems


def test_migrations_are_consecutive_and_schema_version_is_derived() -> None:
    assert BASELINE_VERSION == 2
    assert _migration_problems(MIGRATIONS, BASELINE_VERSION) == []
    expected = MIGRATIONS[-1].version if MIGRATIONS else BASELINE_VERSION
    # The two places the version is written agree, and match the list.
    assert SCHEMA_VERSION == expected
    assert sqlite_mod.SCHEMA_VERSION == expected
    assert sqlite_migrations.target_version() == expected
    # Control: the check sees a wrong start, a gap, a repeat and an empty migration.
    sql = ("CREATE TABLE a (x)",)
    assert _migration_problems((Migration(4, sql),), 2) == ["migration 4 where 3 was expected"]
    assert _migration_problems((Migration(3, sql), Migration(5, sql)), 2) == [
        "migration 5 where 4 was expected"
    ]
    assert _migration_problems((Migration(3, sql), Migration(3, sql)), 2) == [
        "migration 3 where 4 was expected"
    ]
    assert _migration_problems((Migration(3, ()), Migration(4, (" ",))), 2) == [
        "migration 3 has an empty statement",
        "migration 4 has an empty statement",
    ]


async def test_a_failing_migration_leaves_the_file_at_the_old_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "e.db"
    fact = await _make_file(db)
    broken = _with_probe("CREATE TABLE seam_probe (id INTEGER PRIMARY KEY)", "THIS IS NOT SQL")
    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", broken)

    with pytest.raises(StoreError, match=f"migration {PROBE} failed"):
        await SqliteStore.open(db)
    assert _versions(db) == CURRENT_ROWS
    assert "seam_probe" not in _tables(db)

    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", MIGRATIONS)
    store = await SqliteStore.open(db)
    try:
        assert [f.id for f in await store.keyword_search("shed", ACME_ALICE)] == [fact.id]
    finally:
        await store.close()
    assert _versions(db) == CURRENT_ROWS


async def test_a_new_migration_runs_once_and_older_libraries_then_refuse_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, fresh = tmp_path / "old.db", tmp_path / "fresh.db"
    await _make_file(old)
    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", _probed_migrations())

    # The second open neither reruns the migration nor takes its own row as newer.
    for db in (old, fresh, old, fresh):
        await (await SqliteStore.open(db)).close()
    for db in (old, fresh):
        assert _versions(db) == PROBED_ROWS
        assert "seam_probe" in _tables(db)

    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", MIGRATIONS)
    with pytest.raises(SchemaVersionError, match="newer version of Engram"):
        await SqliteStore.open(old)


async def _migrate_with_a_newer_library(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the file with one more migration than this library knows, then go back."""
    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", _probed_migrations())
    await (await SqliteStore.open(db)).close()
    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", MIGRATIONS)
    assert _versions(db) == PROBED_ROWS


def _table_rows(db: Path) -> dict[str, list[str]]:
    """Every row of every table, so a test can tell that nothing was written."""
    with _ro(db) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {
            name: sorted(repr(row) for row in conn.execute(f'SELECT * FROM "{name}"'))
            for name in names
        }


@pytest.mark.parametrize("method", sorted(STORE_WRITES))
async def test_an_open_store_stops_writing_once_a_newer_library_migrates_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    db = tmp_path / "e.db"
    store = await SqliteStore.open(db)
    try:
        seeded = await _seed_store(store)
        await _migrate_with_a_newer_library(db, monkeypatch)
        before = _table_rows(db)
        assert before["facts"] and before["revocations"] and before["erasures"]

        with pytest.raises(SchemaVersionError, match="newer version of Engram"):
            await STORE_WRITES[method](store, seeded)
        assert _table_rows(db) == before
    finally:
        await store.close()
    assert _versions(db) == PROBED_ROWS


async def test_access_counts_are_dropped_once_a_newer_library_migrates_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "e.db"
    store = await SqliteStore.open(db)
    try:
        fact = Fact(text=KEPT, scope=ACME_ALICE, valid_from=_now())
        await store.upsert_fact(fact)
        await _migrate_with_a_newer_library(db, monkeypatch)
        caplog.clear()
        await store.record_access(fact.id)
    finally:
        # Waits for the queued count. A refused count is logged, not raised here.
        await store.close()
    with _ro(db) as conn:
        row = conn.execute("SELECT access_count FROM facts WHERE id = ?", (str(fact.id),))
        assert row.fetchone() == (0,)
    assert "could not record access counts; dropped 1" in caplog.text
    assert "newer version of Engram" in caplog.text
