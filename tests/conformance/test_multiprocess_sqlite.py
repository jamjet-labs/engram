"""Several processes on one SQLite file.

Each test starts real worker processes (tests/conformance/worker.py) on a file
in the test's own directory. SQLite only: another file holds the Postgres
version of these. Skipped where a backend keeps nothing between opens.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Any

import anyio
import pytest

from engram import Engram
from engram.store import sqlite_migrations
from tests.conformance.backends import Backend
from tests.conformance.worker import CRASH_EXIT_CODE, injected_migration

pytestmark = [pytest.mark.multiprocess, pytest.mark.persistent, pytest.mark.sqlite_only]

REPO_ROOT = Path(__file__).resolve().parents[2]
ACME = ["acme", "alice"]
TIMEOUT_S = 120.0


class WorkerProcess:
    """One worker subprocess. Commands go in as JSON lines, replies come back."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc

    async def send(self, **cmd: Any) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(cmd) + "\n").encode())
        await self.proc.stdin.drain()

    async def reply(self) -> dict[str, Any]:
        """The next reply line. {"closed": True} when the process ended with none."""
        assert self.proc.stdout is not None
        line = await asyncio.wait_for(self.proc.stdout.readline(), TIMEOUT_S)
        if not line:
            return {"closed": True}
        reply: dict[str, Any] = json.loads(line)
        return reply

    async def replies(self, n: int) -> list[dict[str, Any]]:
        return [await self.reply() for _ in range(n)]

    async def call(self, **cmd: Any) -> dict[str, Any]:
        await self.send(**cmd)
        return await self.reply()


@asynccontextmanager
async def start_workers(
    target: str,
    *crash_at: str | None,
    barrier: bool = False,
    extra_args: Sequence[Sequence[str]] = (),
) -> AsyncIterator[list[WorkerProcess]]:
    """One worker per argument (None for no crash label), each past its imports.

    A worker holds its commands until it reads a "go" line. With `barrier`
    the test sends that itself, to start several workers together; otherwise
    it is sent here. `extra_args[i]`, when given, is added to worker i's
    command line (for example ["--embedder", "synthetic:256"]).
    """
    workers: list[WorkerProcess] = []
    try:
        for i, label in enumerate(crash_at):
            args = [sys.executable, "-m", "tests.conformance.worker", "--target", target]
            if label:
                args += ["--crash-at", label]
            if i < len(extra_args):
                args += list(extra_args[i])
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=REPO_ROOT,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": str(REPO_ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            w = WorkerProcess(proc)
            workers.append(w)
            assert (await w.reply()) == {"ready": True}
        if not barrier:
            for w in workers:
                await w.send(op="go")
        yield workers
    finally:
        for w in workers:
            if w.proc.returncode is None:
                assert w.proc.stdin is not None
                w.proc.stdin.close()
                try:
                    await asyncio.wait_for(w.proc.wait(), 30)
                except TimeoutError:
                    w.proc.kill()
                    await w.proc.wait()


async def new_file(backend: Backend, tmp_path: Path) -> str:
    return await backend.new_target(tmp_path)


async def init_file(target: str) -> None:
    """Create the schema in `target`, as a first run would."""
    e = await Engram.open(target)
    await e.close()


def rows(target: str, sql: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(f"file:{target}?mode=ro", uri=True)) as db:
        return db.execute(sql).fetchall()


def versions(target: str) -> list[int]:
    return [r[0] for r in rows(target, "SELECT version FROM schema_version ORDER BY rowid")]


def all_versions() -> list[int]:
    return [1, *range(sqlite_migrations.BASELINE_VERSION, sqlite_migrations.SCHEMA_VERSION + 1)]


def fact_count(target: str) -> int:
    return int(rows(target, "SELECT count(*) FROM facts")[0][0])


def db_digest(target: str) -> str:
    """The data the file holds, independent of WAL checkpoints."""
    parts = []
    tables = (
        "schema_version",
        "facts",
        "fact_meta",
        "messages",
        "revocations",
        "erasures",
        "scope_state",
        "engram_meta",
    )
    for table in tables:
        parts.append(repr(rows(target, f"SELECT * FROM {table} ORDER BY 1")))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def epochs(target: str) -> list[tuple[Any, ...]]:
    return rows(target, "SELECT org_id, user_id, erase_epoch FROM scope_state ORDER BY 1, 2")


async def two_workers_record(
    target: str, per_worker: int, labels: tuple[str, str] = ("a", "b")
) -> list[list[dict[str, Any]]]:
    async with start_workers(target, None, None, barrier=True) as (w1, w2):
        for w, label in ((w1, labels[0]), (w2, labels[1])):
            for i in range(per_worker):
                await w.send(
                    op="record",
                    scope=ACME,
                    text=f"Entry {label}{i} was filed under tag zq{label}n{i}.",
                )
        await w1.send(op="go")
        await w2.send(op="go")
        return [await w1.replies(per_worker), await w2.replies(per_worker)]


async def test_two_processes_record_200_each_into_one_scope(
    backend: Backend, tmp_path: Path
) -> None:
    target = await new_file(backend, tmp_path)
    await init_file(target)
    first, second = await two_workers_record(target, 200)
    assert [r for r in first + second if not r["ok"]] == []
    assert fact_count(target) == 400

    third = await Engram.open(target)
    try:
        for label in ("a", "b"):
            for i in range(200):
                tag = f"zq{label}n{i}"
                hits = await third.recall(tag, user_id="alice", org_id="acme", top_k=5)
                assert f"Entry {label}{i} was filed under tag {tag}." in [h.fact.text for h in hits]
    finally:
        await third.close()


async def test_two_processes_open_a_fresh_file_at_once(backend: Backend, tmp_path: Path) -> None:
    for _round in range(8):
        target = await backend.new_target(tmp_path)
        assert not await anyio.Path(target).exists()
        async with start_workers(target, None, None, barrier=True) as (w1, w2):
            await w1.send(op="open_only")
            await w2.send(op="open_only")
            at = time.time() + 0.2
            await w1.send(op="go", at=at)
            await w2.send(op="go", at=at)
            replies = [await w1.reply(), await w2.reply()]
        assert [r["ok"] for r in replies] == [True, True], replies
        assert versions(target) == all_versions()


async def test_crash_mid_migration_then_reopen_finishes_it(
    backend: Backend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = await new_file(backend, tmp_path)
    async with start_workers(target, None) as (w,):
        kept = await w.call(op="record", scope=ACME, text="Alice keeps her bike in the blue shed.")
    assert kept["ok"]
    before = versions(target)
    assert before == all_versions()

    async with start_workers(target, "migration-step") as (crasher,):
        await crasher.send(op="open_only")
        assert await crasher.reply() == {"closed": True}
        assert await asyncio.wait_for(crasher.proc.wait(), 30) == CRASH_EXIT_CODE

    # The crashed open committed nothing: no version row, no table, same facts.
    assert versions(target) == before
    assert rows(target, "SELECT name FROM sqlite_master WHERE name = 'crash_probe'") == []
    assert fact_count(target) == 1

    # The next open runs the migration, once, whatever came before.
    extra = injected_migration()
    monkeypatch.setattr(sqlite_migrations, "MIGRATIONS", (*sqlite_migrations.MIGRATIONS, extra))
    for _ in range(2):
        e = await Engram.open(target)
        hits = await e.recall("bike", user_id="alice", org_id="acme")
        await e.close()
        assert [h.fact.text for h in hits] == ["Alice keeps her bike in the blue shed."]
    assert versions(target) == [*before, extra.version]
    assert rows(target, "SELECT x FROM crash_probe") == [(1,)]


async def test_newer_schema_is_refused_by_every_process_and_nothing_is_written(
    backend: Backend, tmp_path: Path
) -> None:
    target = await new_file(backend, tmp_path)
    async with start_workers(target, None, None, None) as (early, late_a, late_b):
        # One process already has the file open when the newer library arrives.
        assert (await early.call(op="record", scope=ACME, text="Written before the upgrade."))["ok"]
        with closing(sqlite3.connect(target)) as db:
            newer = sqlite_migrations.SCHEMA_VERSION + 1
            db.execute("INSERT INTO schema_version (version) VALUES (?)", (newer,))
            db.commit()
        before = db_digest(target)

        refused = [
            await early.call(op="record", scope=ACME, text="Written after the upgrade."),
            await late_a.call(op="record", scope=ACME, text="Written by a late process."),
            await late_b.call(op="open_only"),
            await late_b.call(op="record", scope=ACME, text="Written by another late process."),
        ]
    assert [(r["ok"], r.get("error")) for r in refused] == [(False, "SchemaVersionError")] * 4
    assert db_digest(target) == before
    assert fact_count(target) == 1
    assert versions(target)[-1] == newer


async def test_crash_after_the_fact_row_before_commit_leaves_no_row(
    backend: Backend, tmp_path: Path
) -> None:
    target = await new_file(backend, tmp_path)
    async with start_workers(target, None) as (w,):
        assert (await w.call(op="record", scope=ACME, text="Kept: the vet is on Elm Street."))["ok"]
    assert fact_count(target) == 1

    async with start_workers(target, "fact-row-written") as (crasher,):
        await crasher.send(op="record", scope=ACME, text="Lost: the vet moved to Oak Street.")
        assert await crasher.reply() == {"closed": True}
        assert await asyncio.wait_for(crasher.proc.wait(), 30) == CRASH_EXIT_CODE

    assert fact_count(target) == 1
    assert rows(target, "SELECT count(*) FROM fact_meta") == [(1,)]
    e = await Engram.open(target)
    try:
        hits = await e.recall("vet", user_id="alice", org_id="acme")
        assert [h.fact.text for h in hits] == ["Kept: the vet is on Elm Street."]
    finally:
        await e.close()


async def test_crash_before_the_erase_commits_keeps_the_fact(
    backend: Backend, tmp_path: Path
) -> None:
    target = await new_file(backend, tmp_path)
    async with start_workers(target, None) as (w,):
        rec = await w.call(op="record", scope=ACME, text="Alice keeps her bike in the blue shed.")
    fact_id = rec["result"]["id"]
    before = db_digest(target)

    async with start_workers(target, "erase-before-commit") as (crasher,):
        await crasher.send(op="erase", scope=ACME, id=fact_id, reason="asked to", actor="admin")
        assert await crasher.reply() == {"closed": True}
        assert await asyncio.wait_for(crasher.proc.wait(), 30) == CRASH_EXIT_CODE

    assert db_digest(target) == before
    assert fact_count(target) == 1


async def test_crash_before_the_scope_erase_commits_keeps_the_turns_and_the_epoch(
    backend: Backend, tmp_path: Path
) -> None:
    """The scope-wide erase and its epoch move in one commit, or neither moves."""
    target = await new_file(backend, tmp_path)
    seed = await Engram.open(target)
    try:
        await seed.record_message(
            "Alice's locker code is 4417.", session_id="s1", user_id="alice", org_id="acme"
        )
    finally:
        await seed.close()
    before = db_digest(target)
    assert epochs(target) == []

    async with start_workers(target, "erase-before-commit") as (crasher,):
        await crasher.send(op="erase_all", scope=ACME, reason="asked to", actor="admin")
        assert await crasher.reply() == {"closed": True}
        assert await asyncio.wait_for(crasher.proc.wait(), 30) == CRASH_EXIT_CODE

    assert db_digest(target) == before
    assert epochs(target) == []

    # Without the crash, the turn goes and the epoch moves.
    async with start_workers(target, None) as (w,):
        done = await w.call(op="erase_all", scope=ACME, reason="asked to", actor="admin")
    assert done["ok"] and done["result"] == {"erased": 0}, done
    assert rows(target, "SELECT count(*) FROM messages") == [(0,)]
    assert epochs(target) == [("acme", "alice", 1)]


async def test_read_only_process_sees_a_writers_commits(backend: Backend, tmp_path: Path) -> None:
    target = await new_file(backend, tmp_path)
    await init_file(target)

    async def texts(w: WorkerProcess, query: str) -> list[str]:
        r = await w.call(op="recall", scope=ACME, query=query)
        assert r["ok"], r
        return [h["text"] for h in r["result"]]

    async with start_workers(target, None, None) as (writer, reader):
        assert (await reader.call(op="open_only", read_only=True))["ok"]
        assert await texts(reader, "bike") == []

        first = await writer.call(
            op="record", scope=ACME, text="Alice keeps her bike in the blue shed."
        )
        assert await texts(reader, "bike") == ["Alice keeps her bike in the blue shed."]

        await writer.call(op="record", scope=ACME, text="Alice keeps her skis in the red garage.")
        assert await texts(reader, "skis") == ["Alice keeps her skis in the red garage."]

        # The reader cannot write, and the writer can still be revoked under it.
        refused = await reader.call(op="record", scope=ACME, text="Written by a reader.")
        assert (refused["ok"], refused["error"]) == (False, "ReadOnlyStoreError")
        revoked = await writer.call(
            op="revoke", scope=ACME, id=first["result"]["id"], reason="moved", actor="admin"
        )
        assert revoked["ok"], revoked
        assert await texts(reader, "bike") == []
    assert fact_count(target) == 2
