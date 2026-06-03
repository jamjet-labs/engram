"""SQLite-backed implementation of `EngramStore`."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import aiosqlite

from engram.errors import StoreError
from engram.models import (
    ChatMessage,
    Event,
    Fact,
    LifecycleState,
    MemorySystem,
    MemoryTier,
    Polarity,
    PromotionState,
)
from engram.scope import Scope

_TYPED_MEMORY_METADATA_KEY = "_engram_typed_memory"


def _load_schema() -> str:
    return (files("engram.store") / "schema.sql").read_text(encoding="utf-8")


_MIGRATE_TO_V2 = """
-- v2: promote typed memory fields from metadata JSON to dedicated columns
ALTER TABLE facts ADD COLUMN memory_system TEXT;
ALTER TABLE facts ADD COLUMN memory_subtype TEXT;
ALTER TABLE facts ADD COLUMN secondary_systems TEXT;
ALTER TABLE facts ADD COLUMN tags TEXT;
ALTER TABLE facts ADD COLUMN lifecycle_state TEXT;
ALTER TABLE facts ADD COLUMN promotion_state TEXT;
ALTER TABLE facts ADD COLUMN retrieval_policy TEXT;
ALTER TABLE facts ADD COLUMN valid_until TEXT;
CREATE INDEX IF NOT EXISTS idx_facts_memory_system ON facts(org_id, user_id, memory_system);
CREATE INDEX IF NOT EXISTS idx_facts_lifecycle ON facts(org_id, user_id, lifecycle_state);
"""


def _dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


async def _populate_typed_columns_from_metadata(conn: aiosqlite.Connection) -> None:
    rows = await conn.execute_fetchall(
        "SELECT rowid, id, metadata FROM facts WHERE memory_system IS NULL"
    )
    for rowid, _fid, md_json in rows:
        md = json.loads(md_json) if md_json else {}
        typed = md.get(_TYPED_MEMORY_METADATA_KEY, md)
        ms = typed.get("memory_system", "working")
        ms_sub = typed.get("memory_subtype")
        secondary = json.dumps(typed.get("secondary_systems", []))
        tags = json.dumps(typed.get("tags", []))
        lc = typed.get("lifecycle_state", "durable")
        ps = typed.get("promotion_state", "raw")
        rp = typed.get("retrieval_policy")
        vu = typed.get("valid_until")
        await conn.execute(
            """UPDATE facts SET
                memory_system=?, memory_subtype=?, secondary_systems=?,
                tags=?, lifecycle_state=?, promotion_state=?,
                retrieval_policy=?, valid_until=?
            WHERE rowid=?""",
            (ms, ms_sub, secondary, tags, lc, ps, rp, vu, rowid),
        )


def _metadata_with_typed_fields(fact: Fact) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(fact.metadata)
    metadata.pop(_TYPED_MEMORY_METADATA_KEY, None)
    typed_metadata: dict[str, Any] = {
        "memory_system": fact.memory_system.value,
        "secondary_systems": [system.value for system in fact.secondary_systems],
        "tags": fact.tags,
        "lifecycle_state": fact.lifecycle_state.value,
        "promotion_state": fact.promotion_state.value,
    }
    if fact.memory_subtype is not None:
        typed_metadata["memory_subtype"] = fact.memory_subtype
    if fact.retrieval_policy is not None:
        typed_metadata["retrieval_policy"] = fact.retrieval_policy
    if fact.valid_until is not None:
        typed_metadata["valid_until"] = fact.valid_until.isoformat()
    metadata[_TYPED_MEMORY_METADATA_KEY] = typed_metadata
    return metadata


def _typed_memory_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get(_TYPED_MEMORY_METADATA_KEY)
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return metadata


def _memory_system_from_metadata(metadata: dict[str, Any]) -> MemorySystem:
    raw = metadata.get("memory_system", MemorySystem.WORKING.value)
    try:
        return MemorySystem(str(raw))
    except ValueError:
        return MemorySystem.WORKING


def _secondary_systems_from_metadata(metadata: dict[str, Any]) -> list[MemorySystem]:
    raw = metadata.get("secondary_systems", [])
    if not isinstance(raw, list):
        return []
    systems = []
    for item in raw:
        try:
            systems.append(MemorySystem(str(item)))
        except ValueError:
            continue
    return systems


def _tags_from_metadata(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("tags", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _lifecycle_state_from_metadata(metadata: dict[str, Any]) -> LifecycleState:
    raw = metadata.get("lifecycle_state", LifecycleState.DURABLE.value)
    try:
        return LifecycleState(str(raw))
    except ValueError:
        return LifecycleState.DURABLE


def _promotion_state_from_metadata(metadata: dict[str, Any]) -> PromotionState:
    raw = metadata.get("promotion_state", PromotionState.RAW.value)
    try:
        return PromotionState(str(raw))
    except ValueError:
        return PromotionState.RAW


def _memory_subtype_from_metadata(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("memory_subtype")
    return str(raw) if raw is not None else None


def _retrieval_policy_from_metadata(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("retrieval_policy")
    return str(raw) if raw is not None else None


def _valid_until_from_metadata(metadata: dict[str, Any]) -> datetime | None:
    raw = metadata.get("valid_until")
    return _parse_dt(str(raw)) if raw is not None else None


def _json_path(key: str) -> str:
    return f"$._engram_typed_memory.{key}"


def _legacy_json_path(key: str) -> str:
    return f"$.{key}"


def _append_metadata_filter(
    clauses: list[str],
    params: list[object],
    *,
    key: str,
    values: tuple[str, ...] | None,
    default_value: str | None,
    negate: bool = False,
) -> None:
    if values is None:
        return
    nested_expr = f"json_extract(facts.metadata, '{_json_path(key)}')"
    legacy_expr = f"json_extract(facts.metadata, '{_legacy_json_path(key)}')"
    placeholders = ",".join("?" for _ in values)
    parts = [
        f"{nested_expr} IN ({placeholders})",
        (
            f"(json_type(facts.metadata, '$.{_TYPED_MEMORY_METADATA_KEY}') IS NULL "
            f"AND {legacy_expr} IN ({placeholders}))"
        ),
    ]
    params.extend(values)
    params.extend(values)
    if default_value is not None and default_value in values:
        parts.append(
            f"({nested_expr} IS NULL AND "
            f"(json_type(facts.metadata, '$.{_TYPED_MEMORY_METADATA_KEY}') IS NOT NULL "
            f"OR {legacy_expr} IS NULL))"
        )
    joined = " OR ".join(parts)
    clauses.append(f"NOT ({joined})" if negate else f"({joined})")


def _append_metadata_tags_filter(
    clauses: list[str], params: list[object], tags: tuple[str, ...] | None
) -> None:
    if tags is None:
        return
    for tag in tags:
        clauses.append(
            "(EXISTS ("
            "SELECT 1 FROM json_each(facts.metadata, '$._engram_typed_memory.tags') "
            "WHERE value = ?"
            ") OR (json_type(facts.metadata, '$._engram_typed_memory') IS NULL "
            "AND EXISTS ("
            "SELECT 1 FROM json_each(facts.metadata, '$.tags') WHERE value = ?"
            ")))"
        )
        params.extend((tag, tag))


class SqliteStore:
    """Async SQLite + FTS5 store. Open with `await SqliteStore.open(path)`."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: str | Path) -> SqliteStore:
        conn = await aiosqlite.connect(str(path))
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_load_schema())
        cursor = await conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        current_version = row[0] if row else 0
        if current_version < 2:
            await conn.executescript(_MIGRATE_TO_V2)
            await conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            await _populate_typed_columns_from_metadata(conn)
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
            tags_json = json.dumps(fact.tags)
            secondary_json = json.dumps([s.value for s in fact.secondary_systems])
            await self._conn.execute(
                """
                INSERT INTO facts (
                    id, org_id, user_id, text, valid_from, invalid_at,
                    confidence, category, polarity, tier,
                    event_date, mention_date, source_event_id,
                    source_message_id, source_span_start, source_span_end,
                    supersedes, superseded_by,
                    access_count, last_accessed, metadata, session_id,
                    memory_system, memory_subtype, secondary_systems,
                    tags, lifecycle_state, promotion_state,
                    retrieval_policy, valid_until
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                          ?,?,?,?,?,?,?,?)
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
                    session_id=excluded.session_id,
                    memory_system=excluded.memory_system,
                    memory_subtype=excluded.memory_subtype,
                    secondary_systems=excluded.secondary_systems,
                    tags=excluded.tags,
                    lifecycle_state=excluded.lifecycle_state,
                    promotion_state=excluded.promotion_state,
                    retrieval_policy=excluded.retrieval_policy,
                    valid_until=excluded.valid_until
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
                    json.dumps(_metadata_with_typed_fields(fact)),
                    fact.session_id,
                    fact.memory_system.value,
                    fact.memory_subtype,
                    secondary_json,
                    tags_json,
                    fact.lifecycle_state.value,
                    fact.promotion_state.value,
                    fact.retrieval_policy,
                    _dt(fact.valid_until),
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

    async def keyword_search(
        self,
        query: str,
        scope: Scope,
        limit: int = 30,
        memory_systems: tuple[MemorySystem, ...] | None = None,
        memory_subtypes: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        include_lifecycle_states: tuple[LifecycleState, ...] | None = None,
        exclude_lifecycle_states: tuple[LifecycleState, ...] = (),
    ) -> list[Fact]:
        sanitized = self._sanitize_fts(query)
        if not sanitized:
            return []
        clauses = [
            "facts_fts MATCH ?",
            "facts_fts.org_id = ?",
            "facts_fts.user_id = ?",
        ]
        params: list[object] = [sanitized, scope.org_id, scope.user_id]
        _append_metadata_filter(
            clauses,
            params,
            key="memory_system",
            values=tuple(system.value for system in memory_systems)
            if memory_systems is not None
            else None,
            default_value=MemorySystem.WORKING.value,
        )
        _append_metadata_filter(
            clauses,
            params,
            key="memory_subtype",
            values=memory_subtypes,
            default_value=None,
        )
        _append_metadata_tags_filter(clauses, params, tags)
        _append_metadata_filter(
            clauses,
            params,
            key="lifecycle_state",
            values=tuple(state.value for state in include_lifecycle_states)
            if include_lifecycle_states is not None
            else None,
            default_value=LifecycleState.DURABLE.value,
        )
        _append_metadata_filter(
            clauses,
            params,
            key="lifecycle_state",
            values=tuple(state.value for state in exclude_lifecycle_states),
            default_value=LifecycleState.DURABLE.value,
            negate=True,
        )
        params.append(limit)
        try:
            async with self._conn.execute(
                f"""SELECT facts.*, bm25(facts_fts) AS rank
                   FROM facts_fts
                   JOIN facts ON facts.rowid = facts_fts.rowid
                   WHERE {' AND '.join(clauses)}
                   ORDER BY rank LIMIT ?""",
                tuple(params),
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
        sanitized = self._sanitize_fts(query)
        if not sanitized:
            return []
        try:
            async with self._conn.execute(
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
            async with self._conn.execute(
                f"SELECT rowid, session_id FROM facts WHERE rowid IN ({placeholders})",
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

    # ── Events (Phase 11) ────────────────────────────────────────────────

    async def upsert_event(self, event: Event) -> None:
        try:
            await self._conn.execute(
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
                    metadata=excluded.metadata""",
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
            await self._conn.commit()
        except aiosqlite.Error as e:
            raise StoreError(f"upsert_event: {e}") from e

    async def get_event(self, event_id: UUID, scope: Scope) -> Event | None:
        async with self._conn.execute(
            "SELECT * FROM events WHERE id=? AND org_id=? AND user_id=?",
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
        sanitized = self._sanitize_fts(query) if query else ""
        try:
            if sanitized:
                # Step 1: FTS5 (bm25 only valid against the FTS5 table directly)
                async with self._conn.execute(
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
                clauses = ["org_id=?", "user_id=?", f"rowid IN ({placeholders})"]
                if time_start is not None:
                    clauses.append("time_start >= ?")
                    params.append(_dt(time_start))
                if time_end is not None:
                    clauses.append("(time_end IS NULL OR time_end <= ?)")
                    params.append(_dt(time_end))
                sql = (
                    f"SELECT events.*, events.rowid AS _rowid "
                    f"FROM events WHERE {' AND '.join(clauses)} LIMIT ?"
                )
                params.append(limit * 4)
                async with self._conn.execute(sql, params) as cur:
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
                f"SELECT * FROM events WHERE {' AND '.join(clauses)} "
                f"ORDER BY time_start DESC LIMIT ?"
            )
            params2.append(limit)
            async with self._conn.execute(sql, params2) as cur:
                rows = await cur.fetchall()
            return [self._row_to_event(r) for r in rows]
        except aiosqlite.Error as e:
            raise StoreError(f"search_events: {e}") from e

    @staticmethod
    def _row_to_event(row: aiosqlite.Row) -> Event:
        return Event(
            id=UUID(row["id"]),
            scope=Scope(org_id=row["org_id"], user_id=row["user_id"]),
            subject_canonical=row["subject_canonical"],
            verb=row["verb"],
            object_canonical=row["object_canonical"],
            time_start=datetime.fromisoformat(row["time_start"]),
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
        metadata = cast(dict[str, Any], json.loads(row["metadata"]) if row["metadata"] else {})
        typed_metadata = _typed_memory_metadata(metadata)
        user_metadata = dict(metadata)
        user_metadata.pop(_TYPED_MEMORY_METADATA_KEY, None)

        col_val = lambda name: row[name]  # noqa: E731
        ms_raw = col_val("memory_system")
        ms = str(ms_raw) if ms_raw else _memory_system_from_metadata(typed_metadata).value
        ms_sub_raw = col_val("memory_subtype")
        ms_sub = str(ms_sub_raw) if ms_sub_raw else _memory_subtype_from_metadata(typed_metadata)
        secondary_raw = col_val("secondary_systems")
        secondary_list: list[MemorySystem] | list[str]
        if secondary_raw and isinstance(secondary_raw, str):
            secondary_list = json.loads(secondary_raw)
        elif secondary_raw:
            secondary_list = list(secondary_raw)
        else:
            secondary_list = _secondary_systems_from_metadata(typed_metadata)
        tags_raw = col_val("tags")
        tags_list: list[str]
        if tags_raw and isinstance(tags_raw, str):
            tags_list = json.loads(tags_raw)
        elif tags_raw:
            tags_list = list(tags_raw)
        else:
            tags_list = _tags_from_metadata(typed_metadata)
        lc_raw = col_val("lifecycle_state")
        lc = str(lc_raw) if lc_raw else _lifecycle_state_from_metadata(typed_metadata).value
        ps_raw = col_val("promotion_state")
        ps = str(ps_raw) if ps_raw else _promotion_state_from_metadata(typed_metadata).value
        rp_raw = col_val("retrieval_policy")
        rp = str(rp_raw) if rp_raw else _retrieval_policy_from_metadata(typed_metadata)
        vu_raw = col_val("valid_until")
        vu = str(vu_raw) if vu_raw else _valid_until_from_metadata(typed_metadata)

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
            memory_system=MemorySystem(str(ms)),
            memory_subtype=str(ms_sub) if ms_sub else None,
            secondary_systems=(
                [MemorySystem(s) for s in secondary_list if isinstance(s, str)]
                if secondary_list else []
            ),
            tags=[str(t) for t in tags_list] if tags_list else [],
            lifecycle_state=LifecycleState(str(lc)),
            promotion_state=PromotionState(str(ps)),
            retrieval_policy=str(rp) if rp else None,
            valid_until=_parse_dt(str(vu)) if vu else None,
            event_date=_parse_dt(row["event_date"]),
            mention_date=_parse_dt(row["mention_date"]),
            source_event_id=row["source_event_id"],
            source_message_id=UUID(row["source_message_id"]) if row["source_message_id"] else None,
            source_span=source_span,
            supersedes=UUID(row["supersedes"]) if row["supersedes"] else None,
            superseded_by=UUID(row["superseded_by"]) if row["superseded_by"] else None,
            access_count=row["access_count"],
            last_accessed=_parse_dt(row["last_accessed"]),
            metadata=user_metadata,
            session_id=row["session_id"],
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
