"""One Engram process for the multi-process tests.

    python -m tests.conformance.worker --target <path> [--embedder synthetic:<dim>]
                                       [--crash-at <label>]

Run from the repository root. --embedder picks the embedder the Engram opens
with (default: Engram's own, synthetic at 384 dimensions). The worker prints
{"ready": true} once its imports are done, then reads one JSON command per
line on stdin and answers each with one JSON line:

    {"ok": true, "t0": <time.monotonic()>, "t1": <time.monotonic()>, "result": ...}
    {"ok": false, "error": "<exception class name>", "message": "<text>"}

Commands received before a {"op": "go"} line wait for it, so a test can queue
the same work on two workers and start them together. Commands:

    {"op": "open_only", "read_only": false}
    {"op": "record", "scope": [org, user], "text": "..."}   -> {"id": "<uuid>"}
    {"op": "recall", "scope": [org, user], "query": "...", "top_k": 10}
    {"op": "revoke", "scope": [...], "id": "<uuid>", "reason": "...", "actor": "..."}
    {"op": "erase", "scope": [...], "id": "<uuid>", "reason": "...", "actor": "..."}
    {"op": "erase_all", "scope": [...], "reason": "...", "actor": "..."}  -> {"erased": n}
    {"op": "wait", "seconds": 0.5}
    {"op": "go", "at": <time.time() seconds, optional>}

`record` takes no id: the Engram makes one and returns it. Every command but
`wait` and `go` opens the Engram on first use.

--crash-at kills the process with exit code 137 at a named internal point, by
patching a private function of the store in this process only. The labels live
here, in test code; production code carries no hooks:

    migration-step       after a test-only migration ran, before the open commits
                         (the worker adds that migration; see injected_migration)
    fact-row-written     after a fact row was inserted, before its commit
    erase-before-commit  after the erase deleted its rows, before its commit: a
                         fact's erase, or the scope-wide erase once it has also
                         moved the erase epoch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any
from uuid import UUID

from engram import Engram
from engram.embedding import SyntheticEmbedding
from engram.embedding.base import EmbeddingProvider
from engram.store import sqlite as sqlite_mod
from engram.store import sqlite_migrations
from engram.store.sqlite import SqliteStore
from engram.store.sqlite_migrations import Migration

CRASH_EXIT_CODE = 137
CRASH_LABELS = ("migration-step", "fact-row-written", "erase-before-commit")


def injected_migration() -> Migration:
    """A migration only tests know: one past the real list, two statements."""
    return Migration(
        version=sqlite_migrations.SCHEMA_VERSION + 1,
        statements=(
            "CREATE TABLE crash_probe (x INTEGER NOT NULL)",
            "INSERT INTO crash_probe (x) VALUES (1)",
        ),
    )


def _die() -> None:
    os._exit(CRASH_EXIT_CODE)


def install_crash(label: str) -> None:
    """Patch the named seam so the process dies there."""
    if label == "migration-step":
        extra = injected_migration()
        sqlite_migrations.MIGRATIONS = (*sqlite_migrations.MIGRATIONS, extra)
        real_apply = sqlite_mod._apply_migration

        async def apply_then_die(conn: Any, migration: Migration) -> None:
            await real_apply(conn, migration)
            if migration.version == extra.version:
                _die()

        sqlite_mod._apply_migration = apply_then_die
    elif label == "fact-row-written":
        real_drop = SqliteStore._drop_stale_matches

        async def drop_then_die(self: SqliteStore, kind: str, *args: Any, **kw: Any) -> None:
            await real_drop(self, kind, *args, **kw)
            if kind == "fact":
                _die()

        SqliteStore._drop_stale_matches = drop_then_die  # type: ignore[method-assign]
    elif label == "erase-before-commit":
        real_erase = SqliteStore._erase_in_tx
        real_bump = SqliteStore._bump_erase_epoch

        async def erase_then_die(self: SqliteStore, *args: Any, **kw: Any) -> Any:
            await real_erase(self, *args, **kw)
            _die()

        async def bump_then_die(self: SqliteStore, *args: Any, **kw: Any) -> None:
            await real_bump(self, *args, **kw)
            _die()

        SqliteStore._erase_in_tx = erase_then_die  # type: ignore[method-assign]
        SqliteStore._bump_erase_epoch = bump_then_die  # type: ignore[method-assign]
    else:
        raise SystemExit(f"unknown crash label {label!r}; known: {', '.join(CRASH_LABELS)}")


def make_embedder(spec: str) -> EmbeddingProvider:
    """The embedder an --embedder value names. Only ``synthetic:<dim>`` today."""
    scheme, _, dim = spec.partition(":")
    if scheme != "synthetic" or not dim.isdigit():
        raise SystemExit(f"unknown embedder {spec!r}; known: synthetic:<dim>")
    return SyntheticEmbedding(dim=int(dim))


class Worker:
    def __init__(self, target: str, embedder: EmbeddingProvider | None = None) -> None:
        self.target = target
        self.embedder = embedder
        self.engram: Engram | None = None

    async def _engram(self, read_only: bool = False) -> Engram:
        if self.engram is None:
            self.engram = await Engram.open(
                self.target, embedder=self.embedder, read_only=read_only
            )
        return self.engram

    async def run(self, cmd: dict[str, Any]) -> Any:
        op = cmd["op"]
        if op == "wait":
            await asyncio.sleep(float(cmd["seconds"]))
            return None
        if op == "open_only":
            await self._engram(bool(cmd.get("read_only", False)))
            return None
        e = await self._engram()
        org, user = cmd["scope"]
        if op == "record":
            fact = await e.record(cmd["text"], user_id=user, org_id=org)
            return {"id": str(fact.id)}
        if op == "recall":
            hits = await e.recall(
                cmd["query"], user_id=user, org_id=org, top_k=cmd.get("top_k", 10)
            )
            return [{"id": str(h.fact.id), "text": h.fact.text} for h in hits]
        if op in ("revoke", "erase"):
            call = e.revoke if op == "revoke" else e.erase
            await call(
                UUID(cmd["id"]), reason=cmd["reason"], actor=cmd["actor"], user_id=user, org_id=org
            )
            return None
        if op == "erase_all":
            n = await e.erase_user(
                user_id=user, org_id=org, reason=cmd["reason"], actor=cmd["actor"]
            )
            return {"erased": n}
        raise ValueError(f"unknown op {op!r}")

    async def answer(self, cmd: dict[str, Any]) -> None:
        t0 = time.monotonic()
        try:
            result = await self.run(cmd)
            reply: dict[str, Any] = {"ok": True, "t0": t0, "t1": time.monotonic(), "result": result}
        except Exception as exc:
            reply = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:300]}
        print(json.dumps(reply), flush=True)

    async def close(self) -> None:
        if self.engram is not None:
            await self.engram.close()


async def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--embedder")
    parser.add_argument("--crash-at", choices=CRASH_LABELS)
    args = parser.parse_args(argv)
    if args.crash_at:
        install_crash(args.crash_at)
    embedder = make_embedder(args.embedder) if args.embedder else None
    worker = Worker(args.target, embedder)
    print(json.dumps({"ready": True}), flush=True)
    queued: list[dict[str, Any]] = []
    started = False
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            cmd = json.loads(line)
            if cmd["op"] == "go":
                started = True
                # An optional "at" (time.time() seconds) lines several workers
                # up to the same instant: spin, as sleep wakes too coarsely.
                while time.time() < float(cmd.get("at", 0)):
                    pass
                for q in queued:
                    await worker.answer(q)
                queued.clear()
            elif started:
                await worker.answer(cmd)
            else:
                queued.append(cmd)
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
