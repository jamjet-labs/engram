"""Reads during concurrent writes must only ever see committed data.

Readers used to share the write connection, so a recall() running while a
write was half-done saw rows that were about to be hidden: a new fact quoting
revoked text before its match row, a fact derived from a revoked fact before
its derivation link, a hidden fact mid-rewrite, an event before its match
row, and erased text between the scrub and the delete. Each test runs readers
in other tasks while one task writes, and counts any revoked content that
comes back. The count must be zero.

Adapted from the reviewer's probe_read_window3, probe_read_window_events and
probe_erase_window4.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from engram import Engram
from engram.models import Event
from engram.scope import Scope

OPS = Scope(org_id="default", user_id="ops")
A = "The vault code for building seven is four four one two."
B = "The backup server root password is purple elephant ninety."
WRITES = 40


def _target(kind: str, tmp_path: Path) -> str:
    return ":memory:" if kind == "memory" else str(tmp_path / "e.db")


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_recall_never_returns_writes_that_are_about_to_be_hidden(
    kind: str, tmp_path: Path
) -> None:
    async with await Engram.open(_target(kind, tmp_path)) as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        old = await memory.record(f"Old handover: {B}", user_id="ops")
        await memory.revoke(a, reason="r", actor="s")
        await memory.revoke(b, reason="r", actor="s")

        leaks: list[str] = []
        stop = False

        async def reader() -> None:
            while not stop:
                for query in ("purple elephant", "zebrafish", "old handover"):
                    for hit in await memory.recall(query, user_id="ops", top_k=50):
                        text = hit.fact.text
                        if "purple elephant" in text or text.startswith("Zebrafish"):
                            leaks.append(text[:40])
                await asyncio.sleep(0)

        readers = [asyncio.create_task(reader()) for _ in range(3)]
        try:
            for i in range(WRITES):
                await memory.record(f"New handover {i}: {B}", user_id="ops")
                await memory.record(
                    f"Zebrafish summary {i} of the building code",
                    user_id="ops",
                    derived_from=[a.id],
                )
                await memory._store.upsert_fact(
                    old.model_copy(update={"confidence": 0.5 + (i % 5) / 10}), source="update"
                )
        finally:
            stop = True
            await asyncio.gather(*readers)
        assert leaks == [], f"{len(leaks)} leaked reads, e.g. {leaks[:3]}"


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_search_events_never_returns_an_event_before_its_match_row(
    kind: str, tmp_path: Path
) -> None:
    async with await Engram.open(_target(kind, tmp_path)) as memory:
        b = await memory.record(B, user_id="ops")
        await memory.revoke(b, reason="r", actor="s")
        leaks = 0
        stop = False

        async def reader() -> None:
            nonlocal leaks
            while not stop:
                events = await memory.search_events("purple elephant", user_id="ops", limit=50)
                leaks += sum(1 for e in events if "purple elephant" in " ".join(e.aliases))
                await asyncio.sleep(0)

        readers = [asyncio.create_task(reader()) for _ in range(2)]
        try:
            for i in range(WRITES):
                await memory._store.upsert_event(
                    Event(
                        scope=OPS,
                        subject_canonical="ops",
                        verb="noted",
                        object_canonical=f"handover {i}",
                        aliases=[B],
                        time_start=datetime(2026, 3, 1, tzinfo=UTC),
                    )
                )
        finally:
            stop = True
            await asyncio.gather(*readers)
        assert leaks == 0


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_recall_during_erase_never_returns_the_erased_text(kind: str, tmp_path: Path) -> None:
    # Deterministic: after every DELETE or revocation scrub inside erase(), a
    # recall() from another task gets a chance to run before erase goes on.
    async with await Engram.open(_target(kind, tmp_path)) as memory:
        text = "Patient 7 diagnosis is acute myeloid leukemia stage two"
        fact = await memory.record(text, user_id="ops")
        for j in range(3):
            await memory.record(f"Copy {j}: {text}", user_id="ops")
        await memory.revoke(fact, reason="gdpr", actor="dpo")

        conn = memory._store._conn
        real_execute = conn.execute
        probes: list[asyncio.Task[list[object]]] = []

        def execute(sql: str, *args: object, **kwargs: object) -> object:
            head = sql.lstrip().upper()
            if not (head.startswith("DELETE") or head.startswith("UPDATE REVOCATIONS")):
                return real_execute(sql, *args, **kwargs)

            async def then_read() -> object:
                result = await real_execute(sql, *args, **kwargs)
                probe = asyncio.create_task(
                    memory.recall("myeloid leukemia", user_id="ops", top_k=20)  # type: ignore[arg-type]
                )
                probes.append(probe)  # type: ignore[arg-type]
                await asyncio.wait({probe}, timeout=0.2)
                return result

            return then_read()

        conn.execute = execute  # type: ignore[method-assign]
        try:
            await memory.erase(fact, reason="ticket 7", actor="dpo")
        finally:
            conn.execute = real_execute  # type: ignore[method-assign]
        results = await asyncio.gather(*probes)
        assert probes, "the hook never ran"
        leaked = [h.fact.text for hits in results for h in hits if "myeloid" in h.fact.text]  # type: ignore[attr-defined]
        assert leaked == []


async def test_recall_does_not_wait_for_a_long_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a file database reads use their own connection, and recording access
    # counts is queued, so recall() returns while a write transaction is open.
    async with await Engram.open(str(tmp_path / "e.db")) as memory:
        await memory.record("alice prefers espresso", user_id="alice")
        b = await memory.record(B, user_id="ops")
        reached, release = asyncio.Event(), asyncio.Event()
        real = memory._store._match_scope

        async def slow(*args: object, **kwargs: object) -> None:
            reached.set()
            await release.wait()
            await real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(memory._store, "_match_scope", slow)
        revoking = asyncio.create_task(memory.revoke(b, reason="r", actor="s"))
        await reached.wait()
        try:
            hits = await asyncio.wait_for(
                memory.recall("espresso", user_id="alice", top_k=5), timeout=5
            )
            assert [h.fact.text for h in hits] == ["alice prefers espresso"]
        finally:
            release.set()
            await revoking
        await memory._store.flush_access()
        [hit] = await memory.recall("espresso", user_id="alice", top_k=1)
        await memory._store.flush_access()
        fact = await memory._store.get_fact(hit.fact.id, Scope(org_id="default", user_id="alice"))
        assert fact is not None and fact.access_count == 2


# ── orderings, as defence in depth ─────────────────────────────────────
# With separate read connections no other task sees a transaction in
# progress. The writes are still ordered so that even a read inside the same
# transaction never sees revoked or erased text: side rows first, base rows
# last on insert; base rows first on delete.


class _SameTxReader:
    """After every write statement, read the live views on the write connection."""

    def __init__(self, memory: Engram, needle: str) -> None:
        self.conn = memory._store._conn
        self.real = self.conn.execute
        self.real_many = self.conn.executemany
        self.needle = needle
        self.seen: list[str] = []

    async def _probe(self) -> None:
        for view, col in (("live_facts", "text"), ("live_events", "aliases")):
            async with self.real(
                f"SELECT {col} FROM {view} WHERE {col} LIKE ?", (f"%{self.needle}%",)
            ) as cur:
                self.seen += [r[0][:40] for r in await cur.fetchall()]

    def __enter__(self) -> _SameTxReader:
        def is_write(sql: str) -> bool:
            return sql.lstrip().split(None, 1)[0].upper() in ("INSERT", "UPDATE", "DELETE")

        def execute(sql: str, *args: object, **kwargs: object) -> object:
            if not is_write(sql):  # reads are used as `async with`; leave them alone
                return self.real(sql, *args, **kwargs)

            async def run() -> object:
                result = await self.real(sql, *args, **kwargs)
                await self._probe()
                return result

            return run()

        async def executemany(sql: str, *args: object, **kwargs: object) -> object:
            result = await self.real_many(sql, *args, **kwargs)
            await self._probe()
            return result

        self.conn.execute = execute  # type: ignore[method-assign]
        self.conn.executemany = executemany  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        self.conn.execute = self.real  # type: ignore[method-assign]
        self.conn.executemany = self.real_many  # type: ignore[method-assign]


async def test_upserts_never_expose_revoked_text_inside_their_transaction() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        b = await memory.record(B, user_id="ops")
        old = await memory.record(f"Old handover: {B}", user_id="ops")
        await memory.revoke(a, reason="r", actor="s")
        await memory.revoke(b, reason="r", actor="s")
        with _SameTxReader(memory, "purple elephant") as quoting:
            await memory._store.upsert_fact(
                old.model_copy(update={"confidence": 0.7}), source="update"
            )
            await memory._store.upsert_fact(
                old.model_copy(update={"id": uuid4(), "text": f"New handover: {B}"})
            )
            await memory._store.upsert_event(
                Event(
                    scope=OPS,
                    subject_canonical="ops",
                    verb="noted",
                    object_canonical="handover",
                    aliases=[B],
                    time_start=datetime(2026, 3, 1, tzinfo=UTC),
                )
            )
        with _SameTxReader(memory, "Zebrafish") as derived:
            await memory.record("Zebrafish summary of the code", user_id="ops", derived_from=[a.id])
        assert quoting.seen == []
        assert derived.seen == []


async def test_erase_never_exposes_erased_text_inside_its_transaction() -> None:
    async with await Engram.open(":memory:") as memory:
        text = "Patient 7 diagnosis is acute myeloid leukemia stage two"
        fact = await memory.record(text, user_id="ops")
        for j in range(3):
            await memory.record(f"Copy {j}: {text}", user_id="ops")
        await memory.revoke(fact, reason="gdpr", actor="dpo")
        with _SameTxReader(memory, "myeloid") as reader:
            await memory._store.erase(fact.id, OPS, reason="ticket 7", actor="dpo")
        assert reader.seen == []


async def test_the_read_connection_refuses_writes(tmp_path: Path) -> None:
    import aiosqlite

    async with await Engram.open(str(tmp_path / "e.db")) as memory:
        assert memory._store._rconn is not None
        with pytest.raises(aiosqlite.OperationalError, match="readonly"):
            await memory._store._rconn.execute("DELETE FROM revocations")


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_reads_never_see_a_write_that_is_later_rolled_back(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The base property behind all of the above: uncommitted rows are invisible
    # to other tasks, so a write that fails or is cancelled was never seen.
    async with await Engram.open(_target(kind, tmp_path)) as memory:
        await memory.record("zebra crossing notes for monday", user_id="ops")
        reached, release = asyncio.Event(), asyncio.Event()
        real = memory._store._drop_stale_matches

        async def paused(*args: object, **kwargs: object) -> None:
            reached.set()
            await release.wait()
            await real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(memory._store, "_drop_stale_matches", paused)
        writing = asyncio.create_task(
            memory.record("zebra stripes count is forty two", user_id="ops")
        )
        await reached.wait()  # the fact row is written, not committed
        reading = asyncio.create_task(memory.recall("zebra", user_id="ops", top_k=10))
        await asyncio.wait({reading}, timeout=0.3)
        writing.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await writing
        texts = [h.fact.text for h in await reading]
        assert texts == ["zebra crossing notes for monday"]
