"""SQLite-backed implementation of `EngramStore`."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections.abc import AsyncIterator, Collection, Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID

import aiosqlite

from engram.errors import (
    NotFoundError,
    ReadOnlyStoreError,
    SchemaVersionError,
    ScopeErasedError,
    ScopeError,
    ScopeResidueRefusedError,
    StoreError,
)
from engram.governance import (
    ErasureRecord,
    HiddenReason,
    Provenance,
    Revocation,
    content_hash,
    normalize_text,
    quotes,
    quotes_any_run,
)
from engram.models import ChatMessage, Event, Fact, MemoryTier, Polarity
from engram.scope import Scope, ScopeWriteGuard, is_refused, stored_scope
from engram.store import _rules, sqlite_migrations
from engram.store.sqlite_migrations import BASELINE_VERSION, Migration

# Re-exported for callers that read the version from here. The store itself
# calls sqlite_migrations.target_version() at open time.
from engram.store.sqlite_migrations import SCHEMA_VERSION as SCHEMA_VERSION
from engram.validate import (
    MAX_FACT_TEXT_BYTES,
    MAX_MESSAGE_BYTES,
    MAX_STORE_LIMIT,
    check_json,
    check_limit,
    check_text,
)

logger = logging.getLogger(__name__)

_FTS_TABLES = ("facts_fts", "events_fts")
# Schema versions: 1 is 0.1/0.2, 2 is 0.3's revocation views and side tables
# (schema.sql), and each later one is a numbered migration. A library refuses
# a database whose recorded version is higher than its own, before writing to
# it. 0.2 never reads this table, so it cannot be stopped from opening a 0.3 file.

# How long a connection waits for another connection's lock before it fails,
# so a second process waits out a migration or a long write instead of erroring.
_BUSY_TIMEOUT_S = 30.0

_INSERT_MATCH = (
    "INSERT OR IGNORE INTO content_matches (kind, item_id, revocation_id) VALUES (?,?,?)"
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _load_schema() -> str:
    return (files("engram.store") / "schema.sql").read_text(encoding="utf-8")


_LEADING_COMMENTS = re.compile(r"\A(?:\s*--[^\n]*\n)*\s*")


def _schema_statements() -> tuple[list[str], list[str]]:
    """Split schema.sql into PRAGMAs and everything else.

    PRAGMAs such as journal_mode and foreign_keys do nothing (or fail) inside
    a transaction, so they run first; the rest runs in one transaction with
    the version check.
    """
    pragmas: list[str] = []
    statements: list[str] = []
    buf = ""
    for line in _load_schema().splitlines(keepends=True):
        buf += line
        if not sqlite3.complete_statement(buf):
            continue
        stmt = _LEADING_COMMENTS.sub("", buf).strip()
        buf = ""
        if stmt:
            (pragmas if stmt.upper().startswith("PRAGMA") else statements).append(stmt)
    if _LEADING_COMMENTS.sub("", buf).strip():
        raise StoreError("schema.sql ends with an incomplete statement")
    return pragmas, statements


def _is_private_memory(path: str | Path) -> bool:
    """True for databases a second connection cannot open (in-memory, temporary)."""
    text = str(path)
    return text in ("", ":memory:") or "mode=memory" in text


def _read_only_uri(path: str | Path) -> str:
    """A ``file:`` URI with ``mode=ro`` for an existing database file.

    In this mode SQLite itself refuses every write, and a missing file is an
    error rather than a new empty database. ``as_uri`` escapes ``?``, ``#``
    and ``%`` in the path.
    """
    return Path(path).absolute().as_uri() + "?mode=ro"


async def _recorded_version(conn: aiosqlite.Connection) -> int | None:
    """The highest schema version the database records; None if it records none."""
    async with conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ) as cur:
        if await cur.fetchone() is None:
            return None
    async with conn.execute("SELECT MAX(version) FROM schema_version") as cur:
        row = await cur.fetchone()
    return None if row is None or row[0] is None else int(row[0])


def _newer_schema_message(path: str | Path, found: int, target: int) -> str:
    return (
        f"database {path} was written by a newer version of Engram "
        f"(schema {found}; this version supports up to {target}). "
        "Upgrade jamjet-engram to open it. Nothing was written."
    )


async def _refuse_if_newer(conn: aiosqlite.Connection, path: str | Path) -> None:
    """Raise `SchemaVersionError` if the file records a schema above this library's."""
    found = await _recorded_version(conn)
    target = sqlite_migrations.target_version()
    if found is not None and found > target:
        raise SchemaVersionError(_newer_schema_message(path, found, target))


async def _run_pragma(conn: aiosqlite.Connection, pragma: str) -> None:
    """Run one PRAGMA, waiting out another process that holds the file.

    SQLite does not call the busy handler for every lock a PRAGMA needs. The
    first journal_mode=WAL on a new file fails at once with "database is
    locked" when another process opens the same file at the same moment. So
    wait and retry for as long as the busy timeout.
    """
    deadline = time.monotonic() + _BUSY_TIMEOUT_S
    while True:
        try:
            await conn.execute(pragma)
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() >= deadline:
                raise
            await asyncio.sleep(0.02)


async def _apply_migration(conn: aiosqlite.Connection, migration: Migration) -> None:
    """Run one migration's statements and record its version. No commit.

    Called inside the open's write transaction, so a failure part way
    leaves nothing of it behind.
    """
    for stmt in migration.statements:
        await conn.execute(stmt)
    await conn.execute("INSERT INTO schema_version (version) VALUES (?)", (migration.version,))


async def _shielded_rollback(conn: aiosqlite.Connection) -> None:
    """Roll back, even if the caller is cancelled again while waiting.

    If a second cancellation interrupts the wait, the rollback still runs in
    aiosqlite's queue; its outcome is logged from a done-callback instead of
    surfacing later as an unretrieved task exception.
    """
    future = asyncio.ensure_future(conn.rollback())

    def log_failure(done: asyncio.Future[None]) -> None:
        if not done.cancelled() and done.exception() is not None:
            logger.error("rollback failed", exc_info=done.exception())

    future.add_done_callback(log_failure)
    try:
        await asyncio.shield(future)
    except Exception:
        pass  # already logged by log_failure


def _ids_json(ids: Collection[str]) -> str:
    """Bind a set of ids as one parameter: `x IN (SELECT value FROM json_each(?))`."""
    return json.dumps(sorted(ids))


def _utc(dt: datetime) -> datetime:
    """A naive time is read as UTC, so every stored time names one instant."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _dt(dt: datetime | None) -> str | None:
    return _utc(dt).isoformat() if dt else None


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return _parse_required_dt(s)


def _parse_required_dt(s: str) -> datetime:
    return _utc(datetime.fromisoformat(s))


def _check_fact(fact: Fact) -> None:
    """Refuse a fact the stores cannot hold alike. Blank text is the facade's to refuse."""
    check_text("text", fact.text, max_bytes=MAX_FACT_TEXT_BYTES, allow_blank=True)
    if fact.session_id is not None:
        check_text("session_id", fact.session_id, max_bytes=MAX_FACT_TEXT_BYTES)
    for name, value in (("category", fact.category), ("source_event_id", fact.source_event_id)):
        if value is not None:
            check_text(name, value, max_bytes=MAX_FACT_TEXT_BYTES, allow_blank=True)
    check_json("metadata", fact.metadata)


def _check_message(message: ChatMessage) -> None:
    check_text("content", message.content, max_bytes=MAX_MESSAGE_BYTES, allow_blank=True)
    check_text("session_id", message.session_id, max_bytes=MAX_FACT_TEXT_BYTES)
    check_json("metadata", message.metadata)


def _check_event(event: Event) -> None:
    for name, value in (
        ("subject_canonical", event.subject_canonical),
        ("verb", event.verb),
        ("object_canonical", event.object_canonical),
    ):
        check_text(name, value, max_bytes=MAX_FACT_TEXT_BYTES)
    for alias in event.aliases:
        check_text("alias", alias, max_bytes=MAX_FACT_TEXT_BYTES)
    check_json("metadata", event.metadata)


def _check_audit(reason: str, actor: str) -> None:
    check_text("reason", reason, max_bytes=MAX_FACT_TEXT_BYTES)
    check_text("actor", actor, max_bytes=MAX_FACT_TEXT_BYTES)


def _check_guard(guard: ScopeWriteGuard | None, scope: Scope) -> None:
    """Refuse a guard that is not a ScopeWriteGuard, or that is for another scope."""
    if guard is None:
        return
    value: object = guard
    if not isinstance(value, ScopeWriteGuard):
        raise TypeError("guard must be a ScopeWriteGuard, from ScopedEngram.guard()")
    if (value.scope.org_id, value.scope.user_id) != (scope.org_id, scope.user_id):
        raise ScopeError("the guard is for another scope than this write")


class SqliteStore:
    """Async SQLite + FTS5 store. Open with `await SqliteStore.open(path)`."""

    def __init__(
        self,
        conn: aiosqlite.Connection,
        read_conn: aiosqlite.Connection | None = None,
        *,
        read_only: bool = False,
        path: str | Path = ":memory:",
    ) -> None:
        # Writes go through `_conn`, one transaction at a time (see
        # _transaction). Reads go through `_reading()`: a second connection for
        # file databases, which in WAL mode only ever sees committed data, or,
        # for an in-memory database, the write connection under the same lock.
        # A read-only store has the one connection, opened read-only.
        self._conn = conn
        self._rconn = read_conn
        self._read_only = read_only
        self._path = path
        self._fts_secure_delete = False
        # One Engram shares its connections across tasks, and SQLite has one
        # transaction per connection. Every write transaction holds this lock,
        # so two tasks never interleave inside one.
        self._write_lock = asyncio.Lock()
        self._lock_owner: asyncio.Task[Any] | None = None
        # Access counts are advisory, so they are queued and applied in their
        # own short transaction instead of making recall() wait for the lock.
        self._pending_access: dict[str, tuple[int, str | None]] = {}
        self._access_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(cls, path: str | Path, *, read_only: bool = False) -> SqliteStore:
        """Open, or create, the store at ``path``: a file, or ":memory:".

        A writable open brings the file to this library's schema in one write
        transaction: it refuses a file from a newer Engram, runs schema.sql,
        then every numbered migration the file lacks
        (`engram.store.sqlite_migrations`). Each later write transaction
        checks the version again, so if a newer Engram migrates the file while
        this store is open, its writes raise `SchemaVersionError`.

        ``read_only=True`` opens an existing file for reads only. It runs no
        schema statement and writes nothing, so the file must already be at
        this library's schema version (`SchemaVersionError` otherwise). Every
        write method then raises `ReadOnlyStoreError`. ":memory:" cannot be
        opened read-only. A read-only store checks the version only at open:
        if a newer Engram migrates the file later, its reads go on through the
        newer views.
        """
        if read_only:
            return await cls._open_read_only(path)
        conn = await aiosqlite.connect(str(path), timeout=_BUSY_TIMEOUT_S)
        conn.row_factory = aiosqlite.Row
        try:
            # Cheap early exit before even the PRAGMAs touch the file.
            await cls._refuse_newer_schema(conn, path)
            pragmas, statements = _schema_statements()
            for pragma in pragmas:
                await _run_pragma(conn, pragma)
            # Overwrite freed pages with zeros so deleted content does not linger
            # in the database file (per connection; see erase()).
            await _run_pragma(conn, "PRAGMA secure_delete = ON")
            # Check, create and migrate in one write transaction, so a newer
            # library cannot record its version between our check and our
            # views, and a failed migration leaves the file as it was.
            await conn.execute("BEGIN IMMEDIATE")
            try:
                await cls._refuse_newer_schema(conn, path)
                for stmt in statements:
                    await conn.execute(stmt)
                await conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                    (BASELINE_VERSION,),
                )
                await cls._migrate(conn, path)
                await conn.commit()
            except BaseException:
                await _shielded_rollback(conn)
                raise
        except BaseException:
            await conn.close()
            raise
        read_conn: aiosqlite.Connection | None = None
        if not _is_private_memory(path):
            read_conn = await aiosqlite.connect(str(path), timeout=_BUSY_TIMEOUT_S)
            read_conn.row_factory = aiosqlite.Row
            # Fail closed if a write is ever routed to the read connection.
            await read_conn.execute("PRAGMA query_only = ON")
        store = cls(conn, read_conn, path=path)
        try:
            store._fts_secure_delete = await store._enable_fts_secure_delete()
            await store._backfill_fact_meta()
        except BaseException:
            await store.close()
            raise
        return store

    @classmethod
    async def _open_read_only(cls, path: str | Path) -> SqliteStore:
        if _is_private_memory(path):
            raise ValueError("an in-memory database cannot be opened read-only: it starts empty")
        try:
            conn = await aiosqlite.connect(_read_only_uri(path), uri=True, timeout=_BUSY_TIMEOUT_S)
        except aiosqlite.Error as e:
            raise StoreError(f"cannot open {path} read-only: {e}") from e
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA query_only = ON")
            await cls._require_current_schema(conn, path)
        except aiosqlite.Error as e:
            # For example a WAL file whose -shm cannot be created next to it.
            await conn.close()
            raise StoreError(f"cannot open {path} read-only: {e}") from e
        except BaseException:
            await conn.close()
            raise
        return cls(conn, read_only=True, path=path)

    @staticmethod
    async def _migrate(conn: aiosqlite.Connection, path: str | Path) -> None:
        """Apply, in order, each migration above the file's version. No commit."""
        found = await _recorded_version(conn)
        start = BASELINE_VERSION if found is None else found
        for migration in sqlite_migrations.pending(start):
            try:
                await _apply_migration(conn, migration)
            except aiosqlite.Error as e:
                raise StoreError(
                    f"schema migration {migration.version} failed on {path}: {e}. "
                    "The open was rolled back; the file keeps its old schema."
                ) from e

    @property
    def read_only(self) -> bool:
        """True when opened with ``read_only=True``: every write is refused."""
        return self._read_only

    def _require_writable(self) -> None:
        """Raise before a write method reads or writes anything on a read-only store."""
        if self._read_only:
            raise ReadOnlyStoreError("this store was opened read-only; writes are refused")

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        """Run the body as one write transaction, alone on this connection.

        Commits on success. On any other exit, cancellation included, rolls
        back before releasing the lock, so no later write can commit half of
        this one. The rollback is shielded from a second cancellation;
        sqlite3's rollback is a no-op outside a transaction, and aiosqlite runs
        calls in order, so it always lands after the BEGIN.

        BEGIN IMMEDIATE takes SQLite's write lock up front, so reads made at
        the start of the body cannot go stale under another process's write.

        Under that lock it reads the file's schema version again: a newer
        Engram may have migrated the file since this store opened it, and this
        library never writes to a schema newer than it knows.
        """
        async with self._write_lock:
            self._lock_owner = asyncio.current_task()
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
                await _refuse_if_newer(self._conn, self._path)
                yield
                await self._conn.commit()
            except BaseException:
                await _shielded_rollback(self._conn)
                raise
            finally:
                self._lock_owner = None

    @asynccontextmanager
    async def _reading(self) -> AsyncIterator[aiosqlite.Connection]:
        """A connection that shows only committed data, for one read.

        Inside this task's own write transaction (or an enclosing read on an
        in-memory database), the write connection, so a write sees its own
        rows. Otherwise the read connection, or, for an in-memory database,
        the write connection once no write transaction is open.
        """
        task = asyncio.current_task()
        if task is not None and self._lock_owner is task:
            yield self._conn
        elif self._rconn is not None:
            yield self._rconn
        else:
            async with self._write_lock:
                self._lock_owner = task
                try:
                    yield self._conn
                finally:
                    self._lock_owner = None

    @staticmethod
    async def _refuse_newer_schema(conn: aiosqlite.Connection, path: str | Path) -> None:
        """Raise before touching a database written by a newer schema.

        Opening recreates the visibility views from this library's schema.sql,
        which would replace a newer version's revocation rules with older ones.
        """
        await _refuse_if_newer(conn, path)

    @staticmethod
    async def _require_current_schema(conn: aiosqlite.Connection, path: str | Path) -> None:
        """Raise unless the database is at exactly this library's schema version.

        A read-only open cannot upgrade a file, and must not read one whose
        visibility rules are newer than its own.
        """
        found = await _recorded_version(conn)
        target = sqlite_migrations.target_version()
        if found is not None and found > target:
            raise SchemaVersionError(_newer_schema_message(path, found, target))
        if found != target:
            shown = "none" if found is None else str(found)
            raise SchemaVersionError(
                f"database {path} is at schema {shown}; this version of Engram reads "
                f"schema {target}. Open it writable once to upgrade it, then open it "
                "read-only. Nothing was written."
            )

    async def _enable_fts_secure_delete(self) -> bool:
        """Make FTS5 drop deleted tokens from its index (SQLite >= 3.44)."""
        try:
            async with self._transaction():
                for table in _FTS_TABLES:
                    await self._conn.execute(
                        f"INSERT INTO {table}({table}, rank) VALUES('secure-delete', 1)"
                    )
            return True
        except aiosqlite.Error as e:
            logger.warning("FTS5 secure-delete unavailable (%s); erased tokens may persist", e)
            return False

    async def _backfill_fact_meta(self) -> None:
        """Give rows written before 0.3 a content hash so tombstones can match them."""
        async with self._transaction():
            async with self._conn.execute(
                """SELECT id, org_id, user_id, text FROM facts
                   WHERE id NOT IN (SELECT fact_id FROM fact_meta)"""
            ) as cur:
                rows = await cur.fetchall()
            if rows:
                await self._conn.executemany(
                    """INSERT OR IGNORE INTO fact_meta
                       (fact_id, org_id, user_id, content_hash, writer, source, recorded_at)
                       VALUES (?,?,?,?,NULL,'legacy',NULL)""",
                    [(r["id"], r["org_id"], r["user_id"], content_hash(r["text"])) for r in rows],
                )

    async def close(self) -> None:
        await self.flush_access()
        # Wait for an in-flight write transaction rather than cut it off.
        async with self._write_lock:
            if self._rconn is not None:
                await self._rconn.close()
            await self._conn.close()

    async def __aenter__(self) -> SqliteStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ── Facts ────────────────────────────────────────────────────────────

    async def upsert_fact(
        self,
        fact: Fact,
        *,
        writer: str | None = None,
        source: str = "upsert",
        derived_from: Iterable[UUID] | None = None,
        guard: ScopeWriteGuard | None = None,
    ) -> None:
        """Insert or update a fact, its provenance row and its derivation links.

        Writer, source and recorded_at are fixed by the first write of an id.
        The content hash follows the text, so rewriting an active fact into
        revoked text hides it. Nothing here touches the revocations table: a
        write cannot lift a tombstone.

        ``guard`` is the scope's erase epoch when the caller's write started
        (:meth:`scope_guard`). If `erase_scope_residue` has moved the epoch
        since, this raises ScopeErasedError and writes nothing: the fact was
        drawn from data that erase removed. A guard for another scope than the
        fact's raises ScopeError, and anything but a ScopeWriteGuard TypeError.
        """
        self._require_writable()
        _check_fact(fact)
        _check_guard(guard, fact.scope)
        sources = list(derived_from or [])
        try:
            async with self._transaction():
                if guard is not None:
                    await self._refuse_if_erased(guard, "the fact")
                # Checked inside the transaction, so a concurrent erase cannot
                # remove a source between the check and the insert.
                if sources:
                    found = await self.derivation_targets_in_scope(sources, fact.scope)
                    missing = [s for s in sources if s not in found]
                    if missing:
                        raise NotFoundError(f"derived_from facts not found in scope: {missing}")
                if fact.source_message_id is not None and not await self._message_in_scope(
                    fact.source_message_id, fact.scope
                ):
                    raise NotFoundError(
                        f"source message {fact.source_message_id} not found in scope"
                    )
                # Side rows first: the content hash (equality tombstones), the
                # derivation links and any new quote matches exist before the
                # row they hide, and stale matches go only after it changes.
                await self._check_owner("fact", str(fact.id), fact.scope)
                await self._write_fact_meta(fact, writer=writer, source=source)
                if sources:
                    await self._conn.executemany(
                        "INSERT OR IGNORE INTO fact_derivations (fact_id, source_fact_id) "
                        "VALUES (?,?)",
                        [(str(fact.id), str(s)) for s in sources],
                    )
                keep = await self._add_matches("fact", str(fact.id), fact.text, fact.scope)
                cur = await self._conn.execute(
                    """
                    INSERT INTO facts (
                        id, org_id, user_id, text, valid_from, invalid_at,
                        confidence, category, polarity, tier,
                        event_date, mention_date, source_event_id,
                        source_message_id, source_span_start, source_span_end,
                        supersedes, superseded_by,
                        access_count, last_accessed, metadata, session_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        text=excluded.text,
                        valid_from=excluded.valid_from,
                        invalid_at=excluded.invalid_at,
                        confidence=excluded.confidence,
                        category=excluded.category,
                        polarity=excluded.polarity,
                        tier=excluded.tier,
                        event_date=excluded.event_date,
                        mention_date=excluded.mention_date,
                        source_event_id=excluded.source_event_id,
                        source_message_id=excluded.source_message_id,
                        source_span_start=excluded.source_span_start,
                        source_span_end=excluded.source_span_end,
                        supersedes=excluded.supersedes,
                        superseded_by=excluded.superseded_by,
                        access_count=excluded.access_count,
                        last_accessed=excluded.last_accessed,
                        metadata=excluded.metadata,
                        session_id=excluded.session_id
                    WHERE facts.org_id = excluded.org_id AND facts.user_id = excluded.user_id
                    """,
                    (
                        str(fact.id),
                        fact.scope.org_id,
                        fact.scope.user_id,
                        fact.text,
                        _dt(fact.valid_from),
                        _dt(fact.invalid_at),
                        fact.confidence,
                        fact.category,
                        fact.polarity.value,
                        fact.tier.value,
                        _dt(fact.event_date),
                        _dt(fact.mention_date),
                        fact.source_event_id,
                        str(fact.source_message_id) if fact.source_message_id else None,
                        fact.source_span[0] if fact.source_span else None,
                        fact.source_span[1] if fact.source_span else None,
                        str(fact.supersedes) if fact.supersedes else None,
                        str(fact.superseded_by) if fact.superseded_by else None,
                        fact.access_count,
                        _dt(fact.last_accessed),
                        json.dumps(fact.metadata),
                        fact.session_id,
                    ),
                )
                if cur.rowcount == 0:
                    # The id exists in another scope: the conditional update did nothing.
                    raise ScopeError(f"fact {fact.id} belongs to another scope")
                await self._drop_stale_matches("fact", str(fact.id), keep)
        except aiosqlite.Error as e:
            raise StoreError(f"upsert_fact: {e}") from e

    async def _write_fact_meta(self, fact: Fact, *, writer: str | None, source: str) -> None:
        # org/user come from the stored row, not the argument: an upsert on an
        # existing id never moves it to another scope.
        await self._conn.execute(
            """INSERT INTO fact_meta
               (fact_id, org_id, user_id, content_hash, writer, source, recorded_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(fact_id) DO UPDATE SET content_hash = excluded.content_hash""",
            (
                str(fact.id),
                fact.scope.org_id,
                fact.scope.user_id,
                content_hash(fact.text),
                writer,
                source,
                _dt(_now()),
            ),
        )

    @staticmethod
    async def _erase_epoch(db: aiosqlite.Connection, scope: Scope) -> int:
        """The scope's erase epoch as ``db`` sees it; 0 when it has no row."""
        async with db.execute(
            "SELECT erase_epoch FROM scope_state WHERE org_id = ? AND user_id = ?",
            (scope.org_id, scope.user_id),
        ) as cur:
            row = await cur.fetchone()
        return 0 if row is None else int(row[0])

    async def scope_guard(self, scope: Scope) -> ScopeWriteGuard:
        """The scope's erase epoch now, as of the latest commit. For a write that starts now.

        A read: it works on a read-only store too.
        """
        async with self._reading() as db:
            epoch = await self._erase_epoch(db, scope)
        return ScopeWriteGuard(scope=scope, erase_epoch=epoch)

    async def _refuse_if_erased(self, guard: ScopeWriteGuard, what: str) -> None:
        """Raise ScopeErasedError if the scope's erase epoch moved since ``guard``.

        Called inside the write transaction. It holds SQLite's write lock, so
        no erase can commit between this read and the write's own commit.
        """
        if await self._erase_epoch(self._conn, guard.scope) != guard.erase_epoch:
            raise ScopeErasedError(
                f"the scope was erased while this write ran; {what} was not stored"
            )

    async def _bump_erase_epoch(self, scope: Scope) -> None:
        """Move the scope's erase epoch on by one. No commit: the erase commits it."""
        await self._conn.execute(
            """INSERT INTO scope_state (org_id, user_id, erase_epoch) VALUES (?, ?, 1)
               ON CONFLICT(org_id, user_id) DO UPDATE SET erase_epoch = erase_epoch + 1""",
            (scope.org_id, scope.user_id),
        )

    async def _message_in_scope(self, message_id: UUID, scope: Scope) -> bool:
        async with self._reading() as db:
            async with db.execute(
                "SELECT 1 FROM messages WHERE id = ? AND org_id = ? AND user_id = ?",
                (str(message_id), scope.org_id, scope.user_id),
            ) as cur:
                return await cur.fetchone() is not None

    async def _active_revoked_norms(self, scope: Scope) -> list[tuple[str, str]]:
        async with self._reading() as db:
            async with db.execute(
                """SELECT id, content_norm FROM revocations
                    WHERE org_id = ? AND user_id = ? AND reinstated_at IS NULL
                      AND content_norm IS NOT NULL""",
                (scope.org_id, scope.user_id),
            ) as cur:
                return [(r["id"], r["content_norm"]) for r in await cur.fetchall()]

    async def _check_owner(self, kind: str, item_id: str, scope: Scope) -> None:
        """Raise ScopeError if this id already belongs to another scope. No text."""
        sql = {
            "fact": "SELECT org_id, user_id FROM facts WHERE id = ?",
            "message": "SELECT org_id, user_id FROM messages WHERE id = ?",
            "event": "SELECT org_id, user_id FROM events WHERE id = ?",
        }[kind]
        async with self._conn.execute(sql, (item_id,)) as cur:
            row = await cur.fetchone()
        if row is not None and (row["org_id"], row["user_id"]) != (scope.org_id, scope.user_id):
            raise ScopeError(f"{kind} {item_id} belongs to another scope")

    async def _add_matches(self, kind: str, item_id: str, text: str, scope: Scope) -> set[str]:
        """Record which active revocations this text quotes; return their ids. No commit."""
        norm = normalize_text(text)
        matched = {
            rev_id
            for rev_id, rev_norm in await self._active_revoked_norms(scope)
            if quotes(norm, rev_norm)
        }
        if matched:
            await self._conn.executemany(
                _INSERT_MATCH, [(kind, item_id, rev_id) for rev_id in matched]
            )
        return matched

    async def _drop_stale_matches(self, kind: str, item_id: str, keep: Collection[str]) -> None:
        """Remove match rows the item's new text no longer earns. No commit."""
        await self._conn.execute(
            """DELETE FROM content_matches
                WHERE kind = ? AND item_id = ?
                  AND revocation_id NOT IN (SELECT value FROM json_each(?))""",
            (kind, item_id, _ids_json(keep)),
        )

    async def derivation_targets_in_scope(self, ids: Iterable[UUID], scope: Scope) -> set[UUID]:
        """Which of ``ids`` exist in ``scope`` (visible or not). Returns ids only."""
        async with self._reading() as db:
            wanted = [str(i) for i in ids]
            if not wanted:
                return set()
            placeholders = ",".join("?" * len(wanted))
            async with db.execute(
                f"SELECT id FROM facts WHERE org_id=? AND user_id=? AND id IN ({placeholders})",
                (scope.org_id, scope.user_id, *wanted),
            ) as cur:
                rows = await cur.fetchall()
            return {UUID(r["id"]) for r in rows}

    async def get_fact(self, fact_id: UUID, scope: Scope) -> Fact | None:
        async with self._reading() as db:
            async with db.execute(
                "SELECT * FROM live_facts WHERE id=? AND org_id=? AND user_id=?",
                (str(fact_id), scope.org_id, scope.user_id),
            ) as cur:
                row = await cur.fetchone()
            return self._row_to_fact(row) if row else None

    async def get_facts(self, fact_ids: Iterable[UUID], scope: Scope) -> list[Fact]:
        """Batch form of :meth:`get_fact`: one visibility evaluation for many ids."""
        async with self._reading() as db:
            wanted = [str(i) for i in fact_ids]
            if not wanted:
                return []
            placeholders = ",".join("?" * len(wanted))
            async with db.execute(
                f"SELECT * FROM live_facts WHERE org_id=? AND user_id=? AND id IN ({placeholders})",
                (scope.org_id, scope.user_id, *wanted),
            ) as cur:
                rows = await cur.fetchall()
            return [self._row_to_fact(r) for r in rows]

    async def list_facts_by_session(
        self, session_id: str, scope: Scope, limit: int = 100
    ) -> list[Fact]:
        if check_limit("limit", limit, maximum=MAX_STORE_LIMIT) == 0:
            return []
        async with self._reading() as db:
            async with db.execute(
                """SELECT * FROM live_facts
                   WHERE org_id=? AND user_id=? AND session_id=?
                   ORDER BY valid_from DESC LIMIT ?""",
                (scope.org_id, scope.user_id, session_id, limit),
            ) as cur:
                rows = await cur.fetchall()
            return [self._row_to_fact(r) for r in rows]

    async def keyword_search(self, query: str, scope: Scope, limit: int = 30) -> list[Fact]:
        if check_limit("limit", limit, maximum=MAX_STORE_LIMIT) == 0:
            return []
        async with self._reading() as db:
            sanitized = self._sanitize_fts(query)
            if not sanitized:
                return []
            try:
                async with db.execute(
                    """SELECT live_facts.*, bm25(facts_fts) AS rank
                       FROM facts_fts
                       JOIN live_facts ON live_facts.fact_rowid = facts_fts.rowid
                       WHERE facts_fts MATCH ?
                         AND facts_fts.org_id = ?
                         AND facts_fts.user_id = ?
                       ORDER BY rank LIMIT ?""",
                    (sanitized, scope.org_id, scope.user_id, limit),
                ) as cur:
                    rows = await cur.fetchall()
                return [self._row_to_fact(r) for r in rows]
            except aiosqlite.Error as e:
                raise StoreError(f"keyword_search: {e}") from e

    @staticmethod
    def _sanitize_fts(q: str) -> str:
        # Strip FTS5 metacharacters; keep alphanumerics + spaces.
        return " ".join(t for t in q.split() if t.replace("_", "").isalnum())

    async def aggregate_sessions(
        self, query: str, scope: Scope, top_sessions: int = 5
    ) -> list[tuple[str, float]]:
        """Stage-1 retrieval: rank sessions by aggregate fact relevance.

        FTS5's `bm25()` aux function is only valid in a SELECT directly against
        the FTS5 virtual table — joining or CTE-wrapping breaks it. So we do
        FTS5 + scoring in one query, then aggregate in Python.
        """
        if check_limit("top_sessions", top_sessions, maximum=MAX_STORE_LIMIT) == 0:
            return []
        async with self._reading() as db:
            sanitized = self._sanitize_fts(query)
            if not sanitized:
                return []
            try:
                async with db.execute(
                    """SELECT rowid, -bm25(facts_fts) AS row_score
                       FROM facts_fts
                       WHERE facts_fts MATCH ?
                         AND facts_fts.org_id = ?
                         AND facts_fts.user_id = ?""",
                    (sanitized, scope.org_id, scope.user_id),
                ) as cur:
                    rows = await cur.fetchall()
                if not rows:
                    return []
                # Now join to facts to get session_id (fts rowid == facts rowid)
                row_scores = {int(r["rowid"]): float(r["row_score"]) for r in rows}
                placeholders = ",".join("?" * len(row_scores))
                async with db.execute(
                    f"SELECT fact_rowid AS rowid, session_id FROM live_facts "
                    f"WHERE fact_rowid IN ({placeholders})",
                    tuple(row_scores.keys()),
                ) as cur:
                    fact_rows = await cur.fetchall()
            except aiosqlite.Error as e:
                raise StoreError(f"aggregate_sessions: {e}") from e

            # Aggregate by session_id, skipping null
            agg: dict[str, float] = {}
            for fr in fact_rows:
                sid = fr["session_id"]
                if sid is None:
                    continue
                agg[sid] = agg.get(sid, 0.0) + row_scores[int(fr["rowid"])]
            ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
            return ranked[:top_sessions]

    async def record_access(self, fact_id: UUID) -> None:
        """Count a read of this fact. Applied shortly after, in its own transaction.

        Queued so a recall() never waits behind a long write; call
        flush_access() when the counts must be on disk now. Does nothing on a
        read-only store.
        """
        if self._read_only:
            return
        key = str(fact_id)
        count, _ = self._pending_access.get(key, (0, None))
        self._pending_access[key] = (count + 1, _dt(datetime.now().astimezone()))
        if self._access_task is None or self._access_task.done():
            self._access_task = asyncio.create_task(self._apply_access())

    async def _apply_access(self) -> None:
        while self._pending_access:
            batch, self._pending_access = self._pending_access, {}
            try:
                async with self._transaction():
                    await self._conn.executemany(
                        """UPDATE facts
                              SET access_count = access_count + ?, last_accessed = ?
                            WHERE id = ?""",
                        [(n, at, fid) for fid, (n, at) in batch.items()],
                    )
            except (aiosqlite.Error, SchemaVersionError):
                # A newer Engram may have migrated the file since this store
                # opened it; the counts are advisory, so drop them, never raise
                # from close().
                logger.exception("could not record access counts; dropped %d", len(batch))

    async def flush_access(self) -> None:
        """Wait until every queued access count is written."""
        while self._access_task is not None and not self._access_task.done():
            await asyncio.shield(self._access_task)

    # ── Messages ─────────────────────────────────────────────────────────

    async def upsert_message(
        self, message: ChatMessage, *, guard: ScopeWriteGuard | None = None
    ) -> None:
        """Insert or update a transcript turn.

        ``guard`` works as in :meth:`upsert_fact`: if the scope's erase epoch
        has moved since it was taken, raises ScopeErasedError and writes
        nothing.
        """
        self._require_writable()
        _check_message(message)
        _check_guard(guard, message.scope)
        async with self._transaction():
            if guard is not None:
                await self._refuse_if_erased(guard, "the turn")
            await self._check_owner("message", str(message.id), message.scope)
            keep = await self._add_matches(
                "message", str(message.id), message.content, message.scope
            )
            cur = await self._conn.execute(
                """INSERT INTO messages (
                    id, org_id, user_id, session_id, role, content, timestamp, metadata
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id=excluded.session_id,
                    role=excluded.role,
                    content=excluded.content,
                    timestamp=excluded.timestamp,
                    metadata=excluded.metadata
                WHERE messages.org_id = excluded.org_id AND messages.user_id = excluded.user_id""",
                (
                    str(message.id),
                    message.scope.org_id,
                    message.scope.user_id,
                    message.session_id,
                    message.role,
                    message.content,
                    _dt(message.timestamp),
                    json.dumps(message.metadata),
                ),
            )
            if cur.rowcount == 0:
                raise ScopeError(f"message {message.id} belongs to another scope")
            await self._drop_stale_matches("message", str(message.id), keep)

    # ── Events (Phase 11) ────────────────────────────────────────────────

    async def upsert_event(self, event: Event, *, guard: ScopeWriteGuard | None = None) -> None:
        """Insert or update an event.

        ``guard`` works as in :meth:`upsert_fact`: if the scope's erase epoch
        has moved since it was taken, raises ScopeErasedError and writes
        nothing.
        """
        self._require_writable()
        _check_event(event)
        _check_guard(guard, event.scope)
        try:
            async with self._transaction():
                if guard is not None:
                    await self._refuse_if_erased(guard, "the event")
                await self._check_owner("event", str(event.id), event.scope)
                keep = await self._add_matches(
                    "event", str(event.id), _rules.event_text(event), event.scope
                )
                cur = await self._conn.execute(
                    """INSERT INTO events (
                        id, org_id, user_id, subject_canonical, verb, object_canonical,
                        time_start, time_end, confidence, aliases, source_fact_ids, metadata
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        subject_canonical=excluded.subject_canonical,
                        verb=excluded.verb,
                        object_canonical=excluded.object_canonical,
                        time_start=excluded.time_start,
                        time_end=excluded.time_end,
                        confidence=excluded.confidence,
                        aliases=excluded.aliases,
                        source_fact_ids=excluded.source_fact_ids,
                        metadata=excluded.metadata
                    WHERE events.org_id = excluded.org_id AND events.user_id = excluded.user_id""",
                    (
                        str(event.id),
                        event.scope.org_id,
                        event.scope.user_id,
                        event.subject_canonical,
                        event.verb,
                        event.object_canonical,
                        _dt(event.time_start),
                        _dt(event.time_end),
                        event.confidence,
                        json.dumps(event.aliases),
                        json.dumps([str(fid) for fid in event.source_fact_ids]),
                        json.dumps(event.metadata),
                    ),
                )
                if cur.rowcount == 0:
                    raise ScopeError(f"event {event.id} belongs to another scope")
                await self._drop_stale_matches("event", str(event.id), keep)
        except aiosqlite.Error as e:
            raise StoreError(f"upsert_event: {e}") from e

    async def get_event(self, event_id: UUID, scope: Scope) -> Event | None:
        async with self._reading() as db:
            async with db.execute(
                "SELECT * FROM live_events WHERE id=? AND org_id=? AND user_id=?",
                (str(event_id), scope.org_id, scope.user_id),
            ) as cur:
                row = await cur.fetchone()
            return self._row_to_event(row) if row else None

    async def search_events(
        self,
        query: str,
        scope: Scope,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        limit: int = 10,
    ) -> list[Event]:
        if check_limit("limit", limit, maximum=MAX_STORE_LIMIT) == 0:
            return []
        async with self._reading() as db:
            sanitized = self._sanitize_fts(query) if query else ""
            try:
                if sanitized:
                    # Step 1: FTS5 (bm25 only valid against the FTS5 table directly)
                    async with db.execute(
                        """SELECT rowid, -bm25(events_fts) AS score
                           FROM events_fts
                           WHERE events_fts MATCH ?
                             AND events_fts.org_id = ?
                             AND events_fts.user_id = ?""",
                        (sanitized, scope.org_id, scope.user_id),
                    ) as cur:
                        fts_rows = await cur.fetchall()
                    if not fts_rows:
                        return []
                    row_scores = {int(r["rowid"]): float(r["score"]) for r in fts_rows}

                    # Step 2: fetch events including their rowids so we can sort in Python
                    placeholders = ",".join("?" * len(row_scores))
                    params: list[object] = [scope.org_id, scope.user_id]
                    params += list(row_scores.keys())
                    clauses = ["org_id=?", "user_id=?", f"event_rowid IN ({placeholders})"]
                    if time_start is not None:
                        clauses.append("time_start >= ?")
                        params.append(_dt(time_start))
                    if time_end is not None:
                        clauses.append("(time_end IS NULL OR time_end <= ?)")
                        params.append(_dt(time_end))
                    sql = (
                        f"SELECT live_events.*, event_rowid AS _rowid "
                        f"FROM live_events WHERE {' AND '.join(clauses)} LIMIT ?"
                    )
                    params.append(limit * 4)
                    async with db.execute(sql, params) as cur:
                        rows = await cur.fetchall()
                    scored: list[tuple[float, Event]] = [
                        (row_scores.get(int(r["_rowid"]), 0.0), self._row_to_event(r)) for r in rows
                    ]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [e for _, e in scored[:limit]]

                # No query: time-window only listing
                clauses = ["org_id=?", "user_id=?"]
                params2: list[object] = [scope.org_id, scope.user_id]
                if time_start is not None:
                    clauses.append("time_start >= ?")
                    params2.append(_dt(time_start))
                if time_end is not None:
                    clauses.append("(time_end IS NULL OR time_end <= ?)")
                    params2.append(_dt(time_end))
                sql = (
                    f"SELECT * FROM live_events WHERE {' AND '.join(clauses)} "
                    f"ORDER BY time_start DESC LIMIT ?"
                )
                params2.append(limit)
                async with db.execute(sql, params2) as cur:
                    rows = await cur.fetchall()
                return [self._row_to_event(r) for r in rows]
            except aiosqlite.Error as e:
                raise StoreError(f"search_events: {e}") from e

    @staticmethod
    def _row_to_event(row: aiosqlite.Row) -> Event:
        return Event(
            id=UUID(row["id"]),
            scope=stored_scope(row["org_id"], row["user_id"]),
            subject_canonical=row["subject_canonical"],
            verb=row["verb"],
            object_canonical=row["object_canonical"],
            time_start=_parse_required_dt(row["time_start"]),
            time_end=_parse_dt(row["time_end"]),
            confidence=row["confidence"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else [],
            source_fact_ids=[UUID(s) for s in json.loads(row["source_fact_ids"] or "[]")],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # ── Messages ─────────────────────────────────────────────────────────

    async def list_messages(
        self, session_id: str, scope: Scope, limit: int = 1000
    ) -> list[ChatMessage]:
        if check_limit("limit", limit, maximum=MAX_STORE_LIMIT) == 0:
            return []
        async with self._reading() as db:
            async with db.execute(
                """SELECT * FROM live_messages
                   WHERE org_id=? AND user_id=? AND session_id=?
                   ORDER BY timestamp ASC LIMIT ?""",
                (scope.org_id, scope.user_id, session_id, limit),
            ) as cur:
                rows = await cur.fetchall()
            return [self._row_to_message(r) for r in rows]

    async def list_session_ids(self, scope: Scope) -> list[str]:
        """Sessions with a visible transcript turn in scope, by first turn."""
        async with self._reading() as db:
            async with db.execute(
                """SELECT session_id FROM live_messages
                   WHERE org_id=? AND user_id=?
                   GROUP BY session_id ORDER BY MIN(timestamp) ASC, session_id ASC""",
                (scope.org_id, scope.user_id),
            ) as cur:
                rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def list_session_messages(
        self, session_id: str, scope: Scope, *, last_n: int | None = None
    ) -> list[ChatMessage]:
        """A session's visible turns, oldest first. For the v1 REST contract.

        ``last_n=None`` returns every visible turn, with no cap. ``last_n=N``
        returns the newest N, still oldest first. ``last_n`` below 1 is
        refused. Ties on timestamp are broken by id.
        """
        if last_n is not None and not last_n >= 1:
            raise ValueError("last_n must be at least 1")
        where = "org_id=? AND user_id=? AND session_id=?"
        args: tuple[Any, ...] = (scope.org_id, scope.user_id, session_id)
        if last_n is None:
            sql = f"SELECT * FROM live_messages WHERE {where} ORDER BY timestamp ASC, id ASC"
        else:
            sql = f"""SELECT * FROM (
                          SELECT * FROM live_messages WHERE {where}
                           ORDER BY timestamp DESC, id DESC LIMIT ?
                      ) ORDER BY timestamp ASC, id ASC"""
            args = (*args, last_n)
        async with self._reading() as db:
            async with db.execute(sql, args) as cur:
                rows = await cur.fetchall()
        return [self._row_to_message(r) for r in rows]

    # ── Governance: provenance, revocation, erasure ──────────────────────
    #
    # Visibility is decided by the `hidden_facts` view (schema.sql). The read
    # methods above select from `live_*` views built on it. The methods in this
    # section are the only ones allowed to read the base tables, and none of
    # them returns fact text to a caller.

    async def provenance(self, fact_ids: Iterable[UUID], scope: Scope) -> dict[UUID, Provenance]:
        """Provenance for the ids that are visible right now.

        Ids that are hidden (or absent) are left out of the result, so callers
        use this as the last visibility check before returning memories.
        """
        async with self._reading() as db:
            wanted = [str(i) for i in fact_ids]
            if not wanted:
                return {}
            checked_at = _now()
            placeholders = ",".join("?" * len(wanted))
            async with db.execute(
                f"""SELECT lf.id, lf.session_id, lf.source_message_id, lf.source_event_id,
                           lf.supersedes, lf.superseded_by,
                           m.writer, m.source, m.recorded_at
                      FROM live_facts lf
                      LEFT JOIN fact_meta m ON m.fact_id = lf.id
                     WHERE lf.org_id = ? AND lf.user_id = ? AND lf.id IN ({placeholders})""",
                (scope.org_id, scope.user_id, *wanted),
            ) as cur:
                rows = await cur.fetchall()
            derived = await self._derived_from([r["id"] for r in rows])
            return {
                UUID(r["id"]): self._row_to_provenance(
                    r, derived.get(r["id"], []), "active", None, [], checked_at
                )
                for r in rows
            }

    async def _derived_from(self, ids: list[str]) -> dict[str, list[UUID]]:
        async with self._reading() as db:
            if not ids:
                return {}
            placeholders = ",".join("?" * len(ids))
            async with db.execute(
                f"""SELECT fact_id, source_fact_id FROM fact_derivations
                     WHERE fact_id IN ({placeholders}) ORDER BY rowid""",
                tuple(ids),
            ) as cur:
                rows = await cur.fetchall()
            out: dict[str, list[UUID]] = {}
            for r in rows:
                out.setdefault(r["fact_id"], []).append(UUID(r["source_fact_id"]))
            return out

    @staticmethod
    def _row_to_provenance(
        row: aiosqlite.Row,
        derived: list[UUID],
        status: str,
        hidden_reason: HiddenReason | None,
        hidden_by: list[UUID],
        checked_at: datetime,
    ) -> Provenance:
        return Provenance(
            fact_id=UUID(row["id"]),
            writer=row["writer"],
            source=row["source"] or "legacy",
            recorded_at=_parse_dt(row["recorded_at"]),
            session_id=row["session_id"],
            source_message_id=UUID(row["source_message_id"]) if row["source_message_id"] else None,
            source_event_id=row["source_event_id"],
            derived_from=derived,
            supersedes=UUID(row["supersedes"]) if row["supersedes"] else None,
            superseded_by=UUID(row["superseded_by"]) if row["superseded_by"] else None,
            revocation_status="active" if status == "active" else "revoked",
            hidden_reason=hidden_reason,
            hidden_by=hidden_by,
            checked_at=checked_at,
        )

    async def explain(self, fact_id: UUID, scope: Scope) -> Provenance | None:
        """Provenance and revocation status for one fact, visible or not. No text."""
        async with self._reading() as db:
            checked_at = _now()
            async with db.execute(
                """SELECT f.id, f.session_id, f.source_message_id, f.source_event_id,
                          f.supersedes, f.superseded_by,
                          m.writer, m.source, m.recorded_at,
                          EXISTS (SELECT 1 FROM hidden_facts h WHERE h.id = f.id) AS hidden
                     FROM facts f
                     LEFT JOIN fact_meta m ON m.fact_id = f.id
                    WHERE f.id = ? AND f.org_id = ? AND f.user_id = ?""",
                (str(fact_id), scope.org_id, scope.user_id),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            derived = await self._derived_from([row["id"]])
            if not row["hidden"]:
                return self._row_to_provenance(
                    row, derived.get(row["id"], []), "active", None, [], checked_at
                )
            reason, by = await self._hidden_reason(fact_id)
            return self._row_to_provenance(
                row, derived.get(row["id"], []), "revoked", reason, by, checked_at
            )

    async def _hidden_reason(self, fact_id: UUID) -> tuple[HiddenReason, list[UUID]]:
        async with self._reading() as db:
            fid = str(fact_id)
            async with db.execute(
                """SELECT id FROM revocations
                    WHERE fact_id = ? AND reinstated_at IS NULL ORDER BY revoked_at""",
                (fid,),
            ) as cur:
                direct = [UUID(r["id"]) for r in await cur.fetchall()]
            if direct:
                return "revoked", direct
            async with db.execute(
                """SELECT r.id FROM revocations r
                    WHERE r.reinstated_at IS NULL
                      AND (r.id IN (SELECT revocation_id FROM content_matches
                                     WHERE kind = 'fact' AND item_id = ?)
                           OR EXISTS (SELECT 1 FROM fact_meta m
                                       WHERE m.fact_id = ? AND m.content_hash = r.content_hash
                                         AND m.org_id = r.org_id AND m.user_id = r.user_id))
                    ORDER BY r.revoked_at""",
                (fid, fid),
            ) as cur:
                by_content = [UUID(r["id"]) for r in await cur.fetchall()]
            if by_content:
                return "content_tombstone", by_content
            async with db.execute(
                """WITH RECURSIVE anc(id) AS (
                       SELECT source_fact_id FROM fact_derivations WHERE fact_id = ?
                       UNION
                       SELECT d.source_fact_id
                         FROM fact_derivations d JOIN anc ON d.fact_id = anc.id
                   )
                   SELECT DISTINCT r.id, r.revoked_at FROM revocations r
                    WHERE r.reinstated_at IS NULL
                      AND (r.fact_id IN (SELECT id FROM anc)
                           OR r.id IN (SELECT revocation_id FROM content_matches
                                        WHERE kind = 'fact' AND item_id IN (SELECT id FROM anc))
                           OR EXISTS (SELECT 1 FROM fact_meta m
                                       WHERE m.fact_id IN (SELECT id FROM anc)
                                         AND m.content_hash = r.content_hash
                                         AND m.org_id = r.org_id AND m.user_id = r.user_id))
                    ORDER BY r.revoked_at""",
                (fid,),
            ) as cur:
                via = [UUID(r["id"]) for r in await cur.fetchall()]
            return "derived_from_revoked", via

    async def _governance_fact(self, fact_id: UUID, scope: Scope) -> Fact | None:
        """Unfiltered lookup for governance operations inside this class only."""
        async with self._reading() as db:
            async with db.execute(
                "SELECT * FROM facts WHERE id=? AND org_id=? AND user_id=?",
                (str(fact_id), scope.org_id, scope.user_id),
            ) as cur:
                row = await cur.fetchone()
            return self._row_to_fact(row) if row else None

    async def fact_exists(self, fact_id: UUID, scope: Scope) -> bool:
        return await self.explain(fact_id, scope) is not None

    async def fact_row_committed(self, fact_id: UUID, scope: Scope) -> bool:
        """Is the fact's row there as of the latest commit? Returns a bool.

        Reads on the write connection under the write lock. The read
        connection can hold an older snapshot while another task has a
        statement open on it, and then still show a fact an erase removed.
        """
        async with self._write_lock:
            async with self._conn.execute(
                "SELECT 1 FROM facts WHERE id = ? AND org_id = ? AND user_id = ?",
                (str(fact_id), scope.org_id, scope.user_id),
            ) as cur:
                return await cur.fetchone() is not None

    async def content_hash_of(self, fact_id: UUID, scope: Scope) -> str | None:
        async with self._reading() as db:
            async with db.execute(
                """SELECT content_hash FROM fact_meta
                    WHERE fact_id = ? AND org_id = ? AND user_id = ?""",
                (str(fact_id), scope.org_id, scope.user_id),
            ) as cur:
                row = await cur.fetchone()
            return str(row["content_hash"]) if row else None

    async def is_content_revoked(self, text: str, scope: Scope) -> bool:
        """Would this text be withheld by an active tombstone in scope?

        True when the text equals a revoked text after normalization (including
        tombstones whose text was erased but whose hash was kept), or quotes a
        revoked text long enough for containment to count.
        """
        async with self._reading() as db:
            norm = normalize_text(text)
            if not norm:
                return False
            async with db.execute(
                """SELECT 1 FROM revocations
                    WHERE org_id = ? AND user_id = ? AND content_hash = ?
                      AND reinstated_at IS NULL LIMIT 1""",
                (scope.org_id, scope.user_id, content_hash(text)),
            ) as cur:
                if await cur.fetchone() is not None:
                    return True
            return any(quotes(norm, rev) for _, rev in await self._active_revoked_norms(scope))

    async def hidden_fact_ids(self, scope: Scope) -> set[UUID]:
        """Ids of facts in scope that are currently withheld. Ids only."""
        async with self._reading() as db:
            async with db.execute(
                """SELECT f.id FROM facts f
                    WHERE f.org_id = ? AND f.user_id = ?
                      AND f.id IN (SELECT id FROM hidden_facts)""",
                (scope.org_id, scope.user_id),
            ) as cur:
                return {UUID(r["id"]) for r in await cur.fetchall()}

    async def revoke(
        self,
        fact_id: UUID,
        scope: Scope,
        *,
        reason: str,
        actor: str,
        replaced_by: UUID | None = None,
    ) -> Revocation:
        """Tombstone a fact, every copy of its text in scope, and items quoting it.

        Idempotent per fact. Calling it with a replacement on a fact that is
        already revoked without one attaches the replacement; a different
        replacement than the one recorded is refused.
        """
        self._require_writable()
        _check_audit(reason, actor)
        try:
            async with self._transaction():
                fact = await self._governance_fact(fact_id, scope)
                if fact is None:
                    raise NotFoundError(f"fact {fact_id} not found in scope")
                if replaced_by is not None:
                    await self._check_replacement(fact, replaced_by, scope)
                async with self._conn.execute(
                    """SELECT * FROM revocations
                        WHERE fact_id = ? AND reinstated_at IS NULL ORDER BY revoked_at LIMIT 1""",
                    (str(fact_id),),
                ) as cur:
                    existing_row = await cur.fetchone()
                if existing_row is not None:
                    existing = self._row_to_revocation(existing_row)
                    if replaced_by is not None and existing.replaced_by not in (None, replaced_by):
                        raise ValueError(
                            f"fact {fact_id} is already superseded by {existing.replaced_by}"
                        )
                    if replaced_by is not None and existing.replaced_by is None:
                        await self._conn.execute(
                            "UPDATE revocations SET replaced_by = ? WHERE id = ?",
                            (str(replaced_by), str(existing.id)),
                        )
                        await self._link_supersession(fact_id, replaced_by)
                        existing = existing.model_copy(update={"replaced_by": replaced_by})
                    return existing
                rev = Revocation(
                    org_id=scope.org_id,
                    user_id=scope.user_id,
                    fact_id=fact_id,
                    reason=reason,
                    actor=actor,
                    revoked_at=_now(),
                    replaced_by=replaced_by,
                )
                norm = normalize_text(fact.text)
                await self._conn.execute(
                    """INSERT INTO revocations
                       (id, org_id, user_id, fact_id, content_hash, content_norm, reason, actor,
                        revoked_at, replaced_by)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(rev.id),
                        rev.org_id,
                        rev.user_id,
                        str(fact_id),
                        content_hash(fact.text) if norm else None,
                        norm or None,
                        reason,
                        actor,
                        _dt(rev.revoked_at),
                        str(replaced_by) if replaced_by else None,
                    ),
                )
                if norm:
                    await self._match_scope(scope, [(str(rev.id), norm)])
                if replaced_by is not None:
                    await self._link_supersession(fact_id, replaced_by)
        except aiosqlite.Error as e:
            raise StoreError(f"revoke: {e}") from e
        return rev

    async def _check_replacement(self, old: Fact, new_id: UUID, scope: Scope) -> None:
        new = await self.get_fact(new_id, scope)
        if new is None:
            if await self._governance_fact(new_id, scope) is not None:
                raise ValueError(f"new fact {new_id} is revoked and cannot be a replacement")
            raise ValueError(f"new fact {new_id} not found in scope")
        if quotes(normalize_text(new.text), normalize_text(old.text)):
            raise ValueError(
                "replacement repeats the text of the fact it replaces and would be withheld"
            )

    async def _link_supersession(self, old_id: UUID, new_id: UUID) -> None:
        await self._conn.execute(
            "UPDATE facts SET superseded_by = ? WHERE id = ?", (str(new_id), str(old_id))
        )
        await self._conn.execute(
            "UPDATE facts SET supersedes = ? WHERE id = ?", (str(old_id), str(new_id))
        )

    async def _match_scope(
        self,
        scope: Scope,
        revoked: Sequence[tuple[str, str]],
        *,
        facts: Collection[str] | None = None,
        messages: Collection[str] | None = None,
        events: Collection[str] | None = None,
    ) -> None:
        """Record every fact, turn and event in scope that quotes a revoked text.

        `revoked` is `(revocation_id, normalized_text)` pairs. `facts`,
        `messages` and `events` limit the scan to those ids; None means all of
        that kind in scope. Scans the base tables, not the live views: an item
        already hidden by another tombstone or a derivation still needs a row
        for this one, or lifting the other tombstone later brings it back
        while it quotes this text. Only adds rows, so it can only hide more.
        Returns nothing. Call inside a write transaction.
        """
        if not revoked:
            return
        items: list[tuple[str, str, str]] = []
        sc = (scope.org_id, scope.user_id)
        only = " AND id IN (SELECT value FROM json_each(?))"
        scans = (
            ("fact", "SELECT id, text AS body FROM facts WHERE org_id = ? AND user_id = ?", facts),
            (
                "message",
                "SELECT id, content AS body FROM messages WHERE org_id = ? AND user_id = ?",
                messages,
            ),
            ("event", "SELECT * FROM events WHERE org_id = ? AND user_id = ?", events),
        )
        for kind, sql, ids in scans:
            if ids is not None and not ids:
                continue
            params: tuple[str, ...] = sc if ids is None else (*sc, _ids_json(ids))
            async with self._conn.execute(sql if ids is None else sql + only, params) as cur:
                for r in await cur.fetchall():
                    if kind == "event":
                        items.append((kind, r["id"], _rules.event_text(self._row_to_event(r))))
                    else:
                        items.append((kind, r["id"], r["body"]))
        # Normalizing and matching a whole scope is CPU work (about a second at
        # 20k facts); run it in a thread so the event loop keeps serving reads.
        rows = await asyncio.to_thread(_rules.quote_rows, items, list(revoked))
        if rows:
            await self._conn.executemany(_INSERT_MATCH, rows)

    async def _lift_candidates(
        self, scope: Scope, revocation_ids: Collection[str]
    ) -> tuple[set[str], set[str], set[str]]:
        """Ids of facts, turns and events these tombstones may be hiding. Ids only.

        Mirrors the hiding rules in schema.sql: the revoked fact, facts with its
        content hash, items with a quote match, everything derived from those
        facts, turns they cite or that share a session with a fact revoked by
        id, and events linked to or quoting them. Nothing outside this set can
        change visibility when these tombstones are lifted.
        """
        revs = _ids_json(revocation_ids)
        sc = (scope.org_id, scope.user_id)
        in_revs = "IN (SELECT value FROM json_each(?))"
        async with self._conn.execute(
            f"""WITH RECURSIVE seed(id) AS (
                    SELECT fact_id FROM revocations WHERE id {in_revs}
                    UNION
                    SELECT m.fact_id FROM fact_meta m
                      JOIN revocations r
                        ON r.content_hash = m.content_hash
                       AND r.org_id = m.org_id AND r.user_id = m.user_id
                     WHERE r.id {in_revs}
                    UNION
                    SELECT item_id FROM content_matches
                     WHERE kind = 'fact' AND revocation_id {in_revs}
                ), fam(id) AS (
                    SELECT id FROM seed
                    UNION
                    SELECT d.fact_id FROM fact_derivations d JOIN fam ON d.source_fact_id = fam.id
                )
                SELECT f.id, f.source_message_id FROM facts f
                 WHERE f.id IN (SELECT id FROM fam) AND f.org_id = ? AND f.user_id = ?""",
            (revs, revs, revs, *sc),
        ) as cur:
            fact_rows = await cur.fetchall()
        facts = {r["id"] for r in fact_rows}
        messages = {r["source_message_id"] for r in fact_rows if r["source_message_id"]}
        async with self._conn.execute(
            f"""SELECT item_id AS id FROM content_matches
                 WHERE kind = 'message' AND revocation_id {in_revs}
                UNION
                SELECT m.id FROM messages m
                  JOIN facts f
                    ON f.session_id = m.session_id
                   AND f.org_id = m.org_id AND f.user_id = m.user_id
                  JOIN revocations r ON r.fact_id = f.id
                 WHERE r.id {in_revs} AND f.source_message_id IS NULL""",
            (revs, revs),
        ) as cur:
            messages |= {r["id"] for r in await cur.fetchall()}
        async with self._conn.execute(
            f"""SELECT item_id AS id FROM content_matches
                 WHERE kind = 'event' AND revocation_id {in_revs}
                UNION
                SELECT e.id FROM events e
                 WHERE e.org_id = ? AND e.user_id = ?
                   AND EXISTS (SELECT 1 FROM json_each(e.source_fact_ids) j
                                WHERE j.value IN (SELECT value FROM json_each(?)))""",
            (revs, *sc, _ids_json(facts)),
        ) as cur:
            events = {r["id"] for r in await cur.fetchall()}
        return facts, messages, events

    async def reinstate(
        self, fact_id: UUID, scope: Scope, *, reason: str, actor: str
    ) -> tuple[list[Revocation], list[Fact]]:
        """Lift the tombstones placed on this fact itself.

        Returns the lifted revocations and the facts that are visible again
        (the fact and copies hidden only by those tombstones). A fact hidden
        because its text matches another fact's tombstone, or because it was
        derived from a revoked fact, cannot be reinstated on its own: reinstate
        the revoked fact named in ``explain().hidden_by`` instead. That keeps a
        write of revoked text from being a handle to un-revoke the original.
        """
        self._require_writable()
        _check_audit(reason, actor)
        try:
            async with self._transaction():
                if await self._governance_fact(fact_id, scope) is None:
                    raise NotFoundError(f"fact {fact_id} not found in scope")
                explained = await self.explain(fact_id, scope)
                if explained is None or explained.revocation_status == "active":
                    return [], []
                if explained.hidden_reason != "revoked":
                    raise ValueError(
                        f"fact {fact_id} is hidden by tombstones on other facts "
                        f"({explained.hidden_reason}, revocations {explained.hidden_by}); "
                        "reinstate the revoked fact itself"
                    )
                lifted_ids = {str(r) for r in explained.hidden_by}
                facts, messages, events = await self._lift_candidates(scope, lifted_ids)
                # Before anything is lifted, re-check what these tombstones were
                # hiding against every tombstone that stays in force. Content that
                # also quotes another revoked fact stays hidden, even if its match
                # row was never written (databases from before _match_scope
                # scanned hidden rows). Doing this first means no read on this
                # connection ever sees such content visible mid-transaction.
                staying = [
                    (rev_id, norm)
                    for rev_id, norm in await self._active_revoked_norms(scope)
                    if rev_id not in lifted_ids
                ]
                await self._match_scope(
                    scope, staying, facts=facts, messages=messages, events=events
                )
                await self._conn.execute(
                    """UPDATE revocations
                          SET reinstated_at = ?, reinstated_by = ?, reinstate_reason = ?
                        WHERE id IN (SELECT value FROM json_each(?)) AND reinstated_at IS NULL""",
                    (_dt(_now()), actor, reason, _ids_json(lifted_ids)),
                )
        except aiosqlite.Error as e:
            raise StoreError(f"reinstate: {e}") from e
        async with self._reading() as db:
            async with db.execute(
                "SELECT * FROM revocations WHERE id IN (SELECT value FROM json_each(?))",
                (_ids_json(lifted_ids),),
            ) as cur:
                lifted = [self._row_to_revocation(r) for r in await cur.fetchall()]
            # Everything that is visible again, including facts derived from the
            # reinstated one, so the caller can restore their vectors.
            async with db.execute(
                """SELECT * FROM live_facts
                    WHERE org_id = ? AND user_id = ? AND id IN (SELECT value FROM json_each(?))""",
                (scope.org_id, scope.user_id, _ids_json(facts)),
            ) as cur:
                visible = [self._row_to_fact(r) for r in await cur.fetchall()]
        return lifted, visible

    async def list_revocations(
        self, scope: Scope, include_reinstated: bool = False
    ) -> list[Revocation]:
        async with self._reading() as db:
            sql = "SELECT * FROM revocations WHERE org_id = ? AND user_id = ?"
            if not include_reinstated:
                sql += " AND reinstated_at IS NULL"
            async with db.execute(
                sql + " ORDER BY revoked_at", (scope.org_id, scope.user_id)
            ) as cur:
                return [self._row_to_revocation(r) for r in await cur.fetchall()]

    @staticmethod
    def _row_to_revocation(row: aiosqlite.Row) -> Revocation:
        return Revocation(
            id=UUID(row["id"]),
            org_id=row["org_id"],
            user_id=row["user_id"],
            fact_id=UUID(row["fact_id"]),
            reason=row["reason"],
            actor=row["actor"],
            revoked_at=_parse_required_dt(row["revoked_at"]),
            replaced_by=UUID(row["replaced_by"]) if row["replaced_by"] else None,
            reinstated_at=_parse_dt(row["reinstated_at"]),
            reinstated_by=row["reinstated_by"],
            reinstate_reason=row["reinstate_reason"],
        )

    async def erase(
        self,
        fact_id: UUID,
        scope: Scope,
        *,
        reason: str,
        actor: str,
        block_reinsertion: bool = True,
    ) -> ErasureRecord:
        """Hard-delete a fact and what was copied out of or derived from it.

        Removed, in one write transaction held from discovery to commit:

        - the fact, every fact in scope whose normalized text equals it or
          quotes it (see :func:`engram.governance.quotes`), and facts recorded
          with ``derived_from`` pointing at any of those, repeated to a fixpoint;
        - events listing any of them in ``source_fact_ids`` or whose text
          quotes an erased text;
        - transcript turns cited through ``source_message_id`` or whose text
          quotes an erased text;
        - the free text of any tombstone or reinstatement note that quotes it.

        Tombstones on the erased facts lose their normalized text. Their
        content hash is kept when ``block_reinsertion`` is true, so revoked
        content stays blocked after erasure; the hash of a short text can be
        confirmed by guessing, so pass False when that matters more.

        ``reason`` and ``actor`` are refused if they contain any three-token
        run of an erased text. The caller (``Engram.erase``) drops the vectors
        and then calls :meth:`finalize_erasure`.
        """
        self._require_writable()
        _check_audit(reason, actor)
        try:
            async with self._transaction():
                record = await self._erase_in_tx(
                    fact_id, scope, reason=reason, actor=actor, block_reinsertion=block_reinsertion
                )
        except aiosqlite.Error as e:
            raise StoreError(f"erase: {e}") from e
        return record

    async def _erase_in_tx(
        self,
        fact_id: UUID,
        scope: Scope,
        *,
        reason: str,
        actor: str,
        block_reinsertion: bool,
    ) -> ErasureRecord:
        if await self._governance_fact(fact_id, scope) is None:
            raise NotFoundError(f"fact {fact_id} not found in scope")
        sc = (scope.org_id, scope.user_id)
        async with self._conn.execute(
            "SELECT id, text, source_message_id FROM facts WHERE org_id = ? AND user_id = ?", sc
        ) as cur:
            facts: dict[str, _rules.FactRow] = {
                r["id"]: (r["text"], normalize_text(r["text"]), r["source_message_id"])
                for r in await cur.fetchall()
            }
        async with self._conn.execute(
            "SELECT fact_id, source_fact_id FROM fact_derivations"
        ) as cur:
            children: dict[str, list[str]] = {}
            for r in await cur.fetchall():
                children.setdefault(r["source_fact_id"], []).append(r["fact_id"])

        erased = _rules.erase_closure(facts, children, str(fact_id))
        norms, hashes = _rules.erased_norms_and_hashes(erased, facts)
        _rules.refuse_quoting_audit_fields(reason, actor, norms)
        ids = sorted(erased)

        async with self._conn.execute(
            "SELECT id, content FROM messages WHERE org_id = ? AND user_id = ?", sc
        ) as cur:
            message_ids = _rules.messages_to_erase(
                erased, facts, [(r["id"], r["content"]) for r in await cur.fetchall()], norms
            )
        async with self._conn.execute(
            "SELECT * FROM events WHERE org_id = ? AND user_id = ?", sc
        ) as cur:
            events = [(r["id"], self._row_to_event(r)) for r in await cur.fetchall()]
        event_ids = _rules.events_to_erase(
            erased,
            [
                (row_id, [str(s) for s in ev.source_fact_ids], _rules.event_text(ev))
                for row_id, ev in events
            ],
            norms,
        )

        record = ErasureRecord(
            org_id=scope.org_id,
            user_id=scope.user_id,
            fact_ids=[UUID(i) for i in ids],
            actor=actor,
            reason=reason,
            erased_at=_now(),
            artifacts={
                "facts": len(ids),
                "events": len(event_ids),
                "messages": len(message_ids),
            },
        )
        scrubbed = f"[scrubbed by erasure {record.id}]"
        # Each id set is bound as one JSON parameter, not one "?" per id:
        # SQLite refuses more than 32,766 variables in one statement.
        in_ids = "IN (SELECT value FROM json_each(?))"
        ids_json = _ids_json(ids)

        # Base rows first. Until they are gone, the side rows below (match
        # rows, content hashes, tombstone text) are what keeps them hidden.
        await self._conn.execute(f"DELETE FROM facts WHERE id {in_ids}", (ids_json,))
        if message_ids:
            await self._conn.execute(
                f"DELETE FROM messages WHERE id {in_ids}", (_ids_json(message_ids),)
            )
        if event_ids:
            await self._conn.execute(
                f"DELETE FROM events WHERE id {in_ids}", (_ids_json(event_ids),)
            )

        async with self._conn.execute(
            "SELECT * FROM revocations WHERE org_id = ? AND user_id = ?", sc
        ) as cur:
            revs = [dict(r) for r in await cur.fetchall()]
        scrubs = _rules.revocations_to_scrub(
            revs, erased, hashes, norms, block_reinsertion=block_reinsertion
        )
        for s in scrubs:
            await self._conn.execute(
                """UPDATE revocations
                      SET reason = CASE WHEN ? THEN ? ELSE reason END,
                          reinstate_reason = CASE WHEN ? AND reinstate_reason IS NOT NULL
                                                  THEN ? ELSE reinstate_reason END,
                          content_norm = CASE WHEN ? THEN NULL ELSE content_norm END,
                          content_hash = CASE WHEN ? THEN NULL ELSE content_hash END
                    WHERE id = ?""",
                (
                    s.scrub_text,
                    scrubbed,
                    s.scrub_text,
                    scrubbed,
                    s.clear_norm,
                    s.drop_hash,
                    s.revocation_id,
                ),
            )
        record.guarantees["reinsertion_hashes_kept"] = sum(s.keep_hash for s in scrubs)

        for kind, items in (("fact", ids), ("message", message_ids), ("event", event_ids)):
            if items:
                await self._conn.execute(
                    f"DELETE FROM content_matches WHERE kind = ? AND item_id {in_ids}",
                    (kind, _ids_json(items)),
                )
        await self._conn.execute(
            f"DELETE FROM fact_derivations WHERE fact_id {in_ids} OR source_fact_id {in_ids}",
            (ids_json, ids_json),
        )
        await self._conn.execute(f"DELETE FROM fact_meta WHERE fact_id {in_ids}", (ids_json,))
        await self._conn.execute(
            """INSERT INTO erasures
               (id, org_id, user_id, fact_ids, actor, reason, erased_at, artifacts, guarantees)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                str(record.id),
                record.org_id,
                record.user_id,
                json.dumps(ids),
                actor,
                reason,
                _dt(record.erased_at),
                json.dumps(record.artifacts),
                json.dumps(record.guarantees),
            ),
        )
        return record

    async def finalize_erasure(
        self, record: ErasureRecord, extra_artifacts: dict[str, int], extra: dict[str, Any]
    ) -> ErasureRecord:
        """Record what else was removed, then flush the WAL so old pages leave the -wal file."""
        self._require_writable()
        async with self._conn.execute("PRAGMA secure_delete") as cur:
            row = await cur.fetchone()
        secure = bool(row and row[0])
        guarantees: dict[str, Any] = {
            **record.guarantees,
            "sqlite_secure_delete": secure,
            "fts_secure_delete": self._fts_secure_delete,
            **extra,
        }
        artifacts = {**record.artifacts, **extra_artifacts}
        async with self._transaction():
            await self._conn.execute(
                "UPDATE erasures SET artifacts = ?, guarantees = ? WHERE id = ?",
                (json.dumps(artifacts), json.dumps(guarantees), str(record.id)),
            )
        guarantees["wal_checkpointed"] = await self.checkpoint()
        async with self._transaction():
            await self._conn.execute(
                "UPDATE erasures SET guarantees = ? WHERE id = ?",
                (json.dumps(guarantees), str(record.id)),
            )
        await self.checkpoint()
        return record.model_copy(update={"artifacts": artifacts, "guarantees": guarantees})

    async def checkpoint(self) -> bool:
        """Checkpoint and truncate the WAL. True unless a reader blocked it."""
        # A read-only connection reports a checkpoint it cannot do as done.
        self._require_writable()
        # Under the write lock: a checkpoint cannot run inside a transaction.
        async with self._write_lock, self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)") as cur:
            row = await cur.fetchone()
        return row is not None and int(row[0]) == 0

    async def list_erasures(self, scope: Scope) -> list[ErasureRecord]:
        async with self._reading() as db:
            async with db.execute(
                "SELECT * FROM erasures WHERE org_id = ? AND user_id = ? ORDER BY erased_at",
                (scope.org_id, scope.user_id),
            ) as cur:
                rows = await cur.fetchall()
            return [
                ErasureRecord(
                    id=UUID(r["id"]),
                    org_id=r["org_id"],
                    user_id=r["user_id"],
                    fact_ids=[UUID(i) for i in json.loads(r["fact_ids"])],
                    actor=r["actor"],
                    reason=r["reason"],
                    erased_at=_parse_required_dt(r["erased_at"]),
                    artifacts=json.loads(r["artifacts"]),
                    guarantees=json.loads(r["guarantees"]),
                )
                for r in rows
            ]

    # ── v1 REST support (ids and counts only, no text) ───────────────────

    async def find_fact_owner_in_org(self, fact_id: UUID, org_id: str) -> Scope | None:
        """The scope of the fact with this id in ``org_id``, for `Engram.forget_in_org`.

        Hidden facts included, so forgetting one twice finds it again. None
        when no fact in that org has the id: another org's id reads like an
        unknown one. Returns ids only, never text.
        """
        async with self._reading() as db:
            async with db.execute(
                "SELECT org_id, user_id FROM facts WHERE id = ? AND org_id = ?",
                (str(fact_id), org_id),
            ) as cur:
                row = await cur.fetchone()
        return None if row is None else Scope(org_id=row[0], user_id=row[1])

    async def list_fact_ids(self, scope: Scope) -> list[UUID]:
        """Every fact id in scope, hidden ones included. For erasure only."""
        async with self._reading() as db:
            async with db.execute(
                "SELECT id FROM facts WHERE org_id=? AND user_id=? ORDER BY rowid",
                (scope.org_id, scope.user_id),
            ) as cur:
                rows = await cur.fetchall()
        return [UUID(r[0]) for r in rows]

    async def fact_text_quoted_by(self, value: str, scope: Scope) -> bool:
        """Does ``value`` quote any fact text in scope, hidden facts included?

        The rule :meth:`erase` applies to its reason and actor: any three-token
        run of a text, or all of a text of three tokens or fewer. Lets
        `Engram.erase_user` refuse before it deletes anything. Returns a bool,
        never text.
        """
        value_norm = normalize_text(value)
        async with self._reading() as db:
            async with db.execute(
                "SELECT text FROM facts WHERE org_id=? AND user_id=?",
                (scope.org_id, scope.user_id),
            ) as cur:
                rows = await cur.fetchall()
        return any(quotes_any_run(value_norm, normalize_text(r[0])) for r in rows)

    async def fact_counts(self, scope: Scope) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) in ``scope``."""
        return await self._counts(
            "SELECT COUNT(*) FROM facts WHERE org_id = ? AND user_id = ?",
            "SELECT COUNT(*) FROM live_facts"
            " WHERE invalid_at IS NULL AND org_id = ? AND user_id = ?",
            (scope.org_id, scope.user_id),
        )

    async def fact_counts_in_org(self, org_id: str) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) of every user in ``org_id``."""
        return await self._counts(
            "SELECT COUNT(*) FROM facts WHERE org_id = ?",
            "SELECT COUNT(*) FROM live_facts WHERE invalid_at IS NULL AND org_id = ?",
            (org_id,),
        )

    async def _counts(
        self, total_sql: str, valid_sql: str, args: tuple[str, ...]
    ) -> tuple[int, int]:
        # Both statements are literals from the two callers above.
        async with self._reading() as db:
            async with db.execute(total_sql, args) as cur:
                total_row = await cur.fetchone()
            async with db.execute(valid_sql, args) as cur:
                valid_row = await cur.fetchone()
        if total_row is None or valid_row is None:
            raise StoreError("fact_counts: COUNT returned no row")
        return int(total_row[0]), int(valid_row[0])

    async def refused_scopes(self) -> list[tuple[str, str]]:
        """(org_id, user_id) of each scope that holds memory under ids Scope refuses.

        For `Engram.refused_scopes`. Such rows were written before ids were
        checked: 0.2 took any id of 1 to 128 characters. Memory here is what
        erasing the scope removes: facts, fact_meta rows, transcript turns,
        events, entities, relationships, and tombstones that still hold a
        content hash or text. Ids only, sorted. It reads the ids of every row.
        """
        async with self._reading() as db:
            async with db.execute(
                """SELECT org_id, user_id FROM facts
                   UNION SELECT org_id, user_id FROM fact_meta
                   UNION SELECT org_id, user_id FROM messages
                   UNION SELECT org_id, user_id FROM events
                   UNION SELECT org_id, user_id FROM entities
                   UNION SELECT org_id, user_id FROM relationships
                   UNION SELECT org_id, user_id FROM revocations
                          WHERE content_hash IS NOT NULL OR content_norm IS NOT NULL"""
            ) as cur:
                rows = await cur.fetchall()
        return sorted((r[0], r[1]) for r in rows if is_refused(r[0], r[1]))

    async def delete_session_messages(self, session_id: str, scope: Scope) -> int:
        """Hard-delete one session's transcript in scope. Returns turns removed.

        Facts extracted from these turns stay, as on the Rust server. With
        foreign keys on, `facts.source_message_id` is ON DELETE SET NULL, so
        those facts lose their message link. Revocation matches on the removed
        turns go too. Consequence: a revoked fact from this session no longer
        links to a message, so rule (c) of the `live_messages` view withholds
        every turn later recorded under the same session id. That is
        fail-closed and accepted; reusing the id shows an empty session.
        """
        self._require_writable()
        # By predicate, not by a list of ids: a list binds one variable per
        # turn, and SQLite refuses more than 32,766 in one statement.
        session = "org_id=? AND user_id=? AND session_id=?"
        args = (scope.org_id, scope.user_id, session_id)
        try:
            async with self._transaction():
                await self._conn.execute(
                    f"DELETE FROM content_matches WHERE kind='message' AND item_id IN "
                    f"(SELECT id FROM messages WHERE {session})",
                    args,
                )
                cur = await self._conn.execute(f"DELETE FROM messages WHERE {session}", args)
                removed = cur.rowcount
        except aiosqlite.Error as e:
            raise StoreError(f"delete_session_messages: {e}") from e
        return removed

    async def erase_scope_residue(self, scope: Scope, *, reason: str, actor: str) -> ErasureRecord:
        """Hard-delete what per-fact erasure leaves behind in a scope.

        Called by `Engram.erase_user` after every fact in the scope has been
        erased: transcript turns, events, entities, relationships and fact_meta
        rows. Refuses while any fact remains, raising ScopeResidueRefusedError,
        so it can never stand in for `erase`. Every tombstone in the scope
        loses its content hash and normalized text, including a hash an earlier
        ``erase(block_reinsertion=True)`` kept, so the erased texts are no
        longer blocked from re-insertion. Revocation and erasure rows stay:
        they are the content-free audit trail.

        In the same transaction, writes one erasure record with no fact ids,
        ``reason`` and ``actor`` as given, and the per-table counts in
        ``artifacts`` (``revocations_scrubbed`` counts tombstones cleared).
        ``artifacts["scope_erasure"] = 1`` marks it as the scope-wide record.
        Returns it; the caller passes it to :meth:`finalize_erasure`, which
        keeps the marker.

        Last in that transaction, it moves the scope's erase epoch on by one,
        so the erase and the epoch commit together or not at all. A write
        whose `ScopeWriteGuard` was taken before the commit is then refused.
        A refusal moves nothing.

        `fact_entity_refs` rows go through ON DELETE CASCADE (foreign_keys is ON,
        set from schema.sql in `open`).
        """
        self._require_writable()
        _check_audit(reason, actor)
        where = "org_id=? AND user_id=?"
        args = (scope.org_id, scope.user_id)
        removed: dict[str, int] = {}
        try:
            async with self._transaction():
                async with self._conn.execute(
                    f"SELECT COUNT(*) FROM facts WHERE {where}", args
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    raise StoreError("erase_scope_residue: COUNT returned no row")
                if row[0]:
                    raise ScopeResidueRefusedError(
                        f"erase_scope_residue: {row[0]} facts remain in scope"
                    )
                for kind, table in (("message", "messages"), ("event", "events")):
                    await self._conn.execute(
                        f"DELETE FROM content_matches WHERE kind=? AND item_id IN "
                        f"(SELECT id FROM {table} WHERE {where})",
                        (kind, *args),
                    )
                for table in ("messages", "events", "relationships", "entities", "fact_meta"):
                    cur2 = await self._conn.execute(f"DELETE FROM {table} WHERE {where}", args)
                    removed[table] = cur2.rowcount
                cur2 = await self._conn.execute(
                    f"""UPDATE revocations SET content_hash = NULL, content_norm = NULL
                         WHERE {where}
                           AND (content_hash IS NOT NULL OR content_norm IS NOT NULL)""",
                    args,
                )
                removed["revocations_scrubbed"] = cur2.rowcount
                removed["scope_erasure"] = 1
                record = ErasureRecord(
                    org_id=scope.org_id,
                    user_id=scope.user_id,
                    fact_ids=[],
                    actor=actor,
                    reason=reason,
                    erased_at=_now(),
                    artifacts=removed,
                )
                await self._conn.execute(
                    """INSERT INTO erasures
                       (id, org_id, user_id, fact_ids, actor, reason, erased_at, artifacts,
                        guarantees)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        str(record.id),
                        record.org_id,
                        record.user_id,
                        json.dumps([]),
                        actor,
                        reason,
                        _dt(record.erased_at),
                        json.dumps(record.artifacts),
                        json.dumps(record.guarantees),
                    ),
                )
                await self._bump_erase_epoch(scope)
        except aiosqlite.Error as e:
            raise StoreError(f"erase_scope_residue: {e}") from e
        return record

    # ── Embedder identity ────────────────────────────────────────────────

    @staticmethod
    async def _recorded_embedder_key(db: aiosqlite.Connection) -> str | None:
        """The embedder key as ``db`` sees it; None when none is recorded."""
        async with db.execute(
            "SELECT value FROM engram_meta WHERE key = ?", (_rules.EMBEDDER_META_KEY,)
        ) as cur:
            row = await cur.fetchone()
        return None if row is None else str(row[0])

    async def embedder_key(self) -> str | None:
        """The recorded embedder key, as of the latest commit. A read: it works read-only."""
        async with self._reading() as db:
            return await self._recorded_embedder_key(db)

    async def set_embedder_key(self, key: str, *, replace: bool) -> None:
        """Record ``key`` as this file's embedder (see `EngramStore.set_embedder_key`).

        The read and the write are one BEGIN IMMEDIATE transaction. A second
        process recording another key at the same moment waits for this one
        to commit, then reads this key and is refused.
        """
        self._require_writable()
        check_text("embedder key", key, max_bytes=MAX_FACT_TEXT_BYTES)
        value: object = replace
        if not isinstance(value, bool):
            raise TypeError("replace must be True or False")
        async with self._transaction():
            recorded = await self._recorded_embedder_key(self._conn)
            if not _rules.check_embedder_key(recorded, key, replace=replace):
                return
            await self._conn.execute(
                """INSERT INTO engram_meta (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (_rules.EMBEDDER_META_KEY, key),
            )
        if recorded is not None:
            logger.warning(
                "embedder changed from %r to %r. Vectors made by the old one do not "
                "compare with the new; run engram-admin reembed.",
                recorded,
                key,
            )

    # ── Row -> Pydantic ──────────────────────────────────────────────────
    #
    # A row's scope is read as stored (`stored_scope`): every read names its
    # scope, so the ids already match one the caller named.

    @staticmethod
    def _row_to_fact(row: aiosqlite.Row) -> Fact:
        scope = stored_scope(row["org_id"], row["user_id"])
        sp_s = row["source_span_start"]
        sp_e = row["source_span_end"]
        source_span: tuple[int, int] | None = (sp_s, sp_e) if sp_s is not None else None
        return Fact(
            id=UUID(row["id"]),
            text=row["text"],
            scope=scope,
            valid_from=_parse_required_dt(row["valid_from"]),
            invalid_at=_parse_dt(row["invalid_at"]),
            confidence=row["confidence"],
            category=row["category"],
            polarity=Polarity(row["polarity"]),
            tier=MemoryTier(row["tier"]),
            event_date=_parse_dt(row["event_date"]),
            mention_date=_parse_dt(row["mention_date"]),
            source_event_id=row["source_event_id"],
            source_message_id=UUID(row["source_message_id"]) if row["source_message_id"] else None,
            source_span=source_span,
            supersedes=UUID(row["supersedes"]) if row["supersedes"] else None,
            superseded_by=UUID(row["superseded_by"]) if row["superseded_by"] else None,
            access_count=row["access_count"],
            last_accessed=_parse_dt(row["last_accessed"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            session_id=row["session_id"],
        )

    @staticmethod
    def _row_to_message(row: aiosqlite.Row) -> ChatMessage:
        return ChatMessage(
            id=UUID(row["id"]),
            scope=stored_scope(row["org_id"], row["user_id"]),
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            timestamp=_parse_required_dt(row["timestamp"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
