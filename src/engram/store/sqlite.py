"""SQLite-backed implementation of `EngramStore`."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from types import TracebackType
from uuid import UUID

import aiosqlite

from engram.errors import StoreError
from engram.models import ChatMessage, Fact, MemoryTier, Polarity
from engram.scope import Scope


def _load_schema() -> str:
    return (files("engram.store") / "schema.sql").read_text(encoding="utf-8")


def _dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class SqliteStore:
    """Async SQLite + FTS5 store. Open with `await SqliteStore.open(path)`."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: str | Path) -> SqliteStore:
        conn = await aiosqlite.connect(str(path))
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_load_schema())
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
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

    async def upsert_fact(self, fact: Fact) -> None:
        try:
            await self._conn.execute(
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
                    None,
                ),
            )
            await self._conn.commit()
        except aiosqlite.Error as e:
            raise StoreError(f"upsert_fact: {e}") from e

    async def get_fact(self, fact_id: UUID, scope: Scope) -> Fact | None:
        async with self._conn.execute(
            "SELECT * FROM facts WHERE id=? AND org_id=? AND user_id=?",
            (str(fact_id), scope.org_id, scope.user_id),
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_fact(row) if row else None

    async def list_facts_by_session(
        self, session_id: str, scope: Scope, limit: int = 100
    ) -> list[Fact]:
        async with self._conn.execute(
            """SELECT * FROM facts
               WHERE org_id=? AND user_id=? AND session_id=?
               ORDER BY valid_from DESC LIMIT ?""",
            (scope.org_id, scope.user_id, session_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def keyword_search(self, query: str, scope: Scope, limit: int = 30) -> list[Fact]:
        sanitized = self._sanitize_fts(query)
        if not sanitized:
            return []
        try:
            async with self._conn.execute(
                """SELECT facts.*, bm25(facts_fts) AS rank
                   FROM facts_fts
                   JOIN facts ON facts.rowid = facts_fts.rowid
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

    async def record_access(self, fact_id: UUID) -> None:
        await self._conn.execute(
            "UPDATE facts SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (_dt(datetime.now().astimezone()), str(fact_id)),
        )
        await self._conn.commit()

    # ── Messages ─────────────────────────────────────────────────────────

    async def upsert_message(self, message: ChatMessage) -> None:
        await self._conn.execute(
            """INSERT INTO messages (
                id, org_id, user_id, session_id, role, content, timestamp, metadata
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                session_id=excluded.session_id,
                role=excluded.role,
                content=excluded.content,
                timestamp=excluded.timestamp,
                metadata=excluded.metadata""",
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
        await self._conn.commit()

    async def list_messages(
        self, session_id: str, scope: Scope, limit: int = 1000
    ) -> list[ChatMessage]:
        async with self._conn.execute(
            """SELECT * FROM messages
               WHERE org_id=? AND user_id=? AND session_id=?
               ORDER BY timestamp ASC LIMIT ?""",
            (scope.org_id, scope.user_id, session_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_message(r) for r in rows]

    # ── Row -> Pydantic ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_fact(row: aiosqlite.Row) -> Fact:
        scope = Scope(org_id=row["org_id"], user_id=row["user_id"])
        sp_s = row["source_span_start"]
        sp_e = row["source_span_end"]
        source_span: tuple[int, int] | None = (sp_s, sp_e) if sp_s is not None else None
        return Fact(
            id=UUID(row["id"]),
            text=row["text"],
            scope=scope,
            valid_from=datetime.fromisoformat(row["valid_from"]),
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
        )

    @staticmethod
    def _row_to_message(row: aiosqlite.Row) -> ChatMessage:
        return ChatMessage(
            id=UUID(row["id"]),
            scope=Scope(org_id=row["org_id"], user_id=row["user_id"]),
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
