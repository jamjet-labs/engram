"""Write transactions on one Engram share one aiosqlite connection.

Two governance calls from different tasks used to interleave inside one
SQLite transaction: a second BEGIN failed, and its rollback threw away the
first call's uncommitted writes while the first call still reported success.
A cancelled call skipped its rollback and left the transaction open, so the
next unrelated write committed half of it. Each test here pauses one call in
the middle of its transaction and runs another against it.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from engram import Engram
from engram.scope import Scope

A = "The vault code for building seven is four four one two."
B = "The backup server root password is purple elephant ninety."
X = "The cafeteria serves lentil soup every second friday."
OPS = Scope(org_id="default", user_id="ops")
# Long enough for the old code to reach its BEGIN / rollback; the fixed code
# just waits on the lock for this long.
SETTLE = 0.3


class Gate:
    """Pause the first call that reaches a patched method until released."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self._armed = True

    def wrap(self, fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def paused(*args: Any, **kwargs: Any) -> Any:
            if self._armed:
                self._armed = False
                self.reached.set()
                await self.release.wait()
            return await fn(*args, **kwargs)

        return paused


async def _settle(task: asyncio.Task[Any]) -> None:
    await asyncio.wait({task}, timeout=SETTLE)


def _rev_rows(db: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(db) as conn:
        return list(conn.execute("SELECT fact_id, reinstated_at FROM revocations"))


async def test_revoke_and_reinstate_at_once_both_land(monkeypatch: pytest.MonkeyPatch) -> None:
    async with await Engram.open(":memory:") as memory:
        b = await memory.record(B, user_id="ops")
        x = await memory.record(X, user_id="ops")
        await memory.revoke(x, reason="r", actor="sec")

        gate = Gate()
        monkeypatch.setattr(memory._store, "_match_scope", gate.wrap(memory._store._match_scope))
        t1 = asyncio.create_task(memory.revoke(b, reason="leaked", actor="sec"))
        await gate.reached.wait()  # revoke(B) is inside its transaction
        t2 = asyncio.create_task(memory.reinstate(x, reason="ok", actor="sec"))
        await _settle(t2)
        gate.release.set()
        results = await asyncio.gather(t1, t2, return_exceptions=True)

        assert not [r for r in results if isinstance(r, BaseException)], results
        assert [r.fact_id for r in await memory.revocations(user_id="ops")] == [b.id]
        hidden = await memory._store.hidden_fact_ids(OPS)
        assert b.id in hidden
        assert x.id not in hidden


async def test_two_revokes_at_once_both_land(monkeypatch: pytest.MonkeyPatch) -> None:
    async with await Engram.open(":memory:") as memory:
        b = await memory.record(B, user_id="ops")
        x = await memory.record(X, user_id="ops")

        gate = Gate()
        monkeypatch.setattr(memory._store, "_match_scope", gate.wrap(memory._store._match_scope))
        t1 = asyncio.create_task(memory.revoke(b, reason="leaked", actor="sec"))
        await gate.reached.wait()
        t2 = asyncio.create_task(memory.revoke(x, reason="stale", actor="sec"))
        await _settle(t2)
        gate.release.set()
        results = await asyncio.gather(t1, t2, return_exceptions=True)

        assert not [r for r in results if isinstance(r, BaseException)], results
        revoked = {r.fact_id for r in await memory.revocations(user_id="ops")}
        assert revoked == {b.id, x.id}


async def test_record_does_not_commit_half_a_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A plain write in another task must not commit revoke's unfinished work.
    db = tmp_path / "e.db"
    async with await Engram.open(str(db)) as memory:
        b = await memory.record(B, user_id="ops")
        gate = Gate()
        monkeypatch.setattr(memory._store, "_match_scope", gate.wrap(memory._store._match_scope))
        t1 = asyncio.create_task(memory.revoke(b, reason="leaked", actor="sec"))
        await gate.reached.wait()
        t2 = asyncio.create_task(memory.record(X, user_id="ops"))
        await _settle(t2)
        # Another process sees only committed data.
        assert _rev_rows(db) == []
        gate.release.set()
        results = await asyncio.gather(t1, t2, return_exceptions=True)
        assert not [r for r in results if isinstance(r, BaseException)], results
        assert [fid for fid, _ in _rev_rows(db)] == [str(b.id)]


async def test_cancelled_revoke_leaves_nothing_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    async with await Engram.open(":memory:") as memory:
        b = await memory.record(B, user_id="ops")
        c = await memory.record(f"Ticket 12 says: {B}", user_id="ops")
        gate = Gate()
        monkeypatch.setattr(memory._store, "_match_scope", gate.wrap(memory._store._match_scope))
        task = asyncio.create_task(memory.revoke(b, reason="leaked", actor="sec"))
        await gate.reached.wait()  # revocation row written, matches not yet
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not memory._store._conn.in_transaction
        await memory.record("unrelated later write", user_id="ops")
        # Either the whole revocation or none of it: here, none.
        assert await memory.revocations(user_id="ops") == []
        hidden = await memory._store.hidden_fact_ids(OPS)
        assert b.id not in hidden
        assert c.id not in hidden


async def test_cancelled_reinstate_keeps_the_tombstone(monkeypatch: pytest.MonkeyPatch) -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        await memory.revoke(a, reason="r", actor="sec")
        gate = Gate()
        monkeypatch.setattr(memory._store, "_match_scope", gate.wrap(memory._store._match_scope))
        task = asyncio.create_task(memory.reinstate(a, reason="ok", actor="sec"))
        await gate.reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not memory._store._conn.in_transaction
        await memory.record("unrelated later write", user_id="ops")
        assert [r.fact_id for r in await memory.revocations(user_id="ops")] == [a.id]
        assert a.id in await memory._store.hidden_fact_ids(OPS)


async def test_cancelled_erase_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record(A, user_id="ops")
        gate = Gate()
        monkeypatch.setattr(memory._store, "_erase_in_tx", gate.wrap(memory._store._erase_in_tx))
        task = asyncio.create_task(memory.erase(a, reason="ticket 9", actor="dpo"))
        await gate.reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not memory._store._conn.in_transaction
        await memory.record("unrelated later write", user_id="ops")
        assert await memory.explain(a.id, user_id="ops") is not None


async def test_cancelled_record_does_not_leak_revoked_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cancelled between the fact row and its match row: if the half-written
    # fact were committed later, it would quote revoked text and be visible.
    async with await Engram.open(":memory:") as memory:
        b = await memory.record(B, user_id="ops")
        await memory.revoke(b, reason="leaked", actor="sec")
        gate = Gate()
        monkeypatch.setattr(memory._store, "_match_item", gate.wrap(memory._store._match_item))
        task = asyncio.create_task(memory.record(f"Ticket 12 says: {B}", user_id="ops"))
        await gate.reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not memory._store._conn.in_transaction
        await memory.record("unrelated later write", user_id="ops")
        hits = await memory.recall("backup server root password", user_id="ops", top_k=50)
        assert not any("purple elephant" in h.fact.text for h in hits)


async def test_begin_immediate_serialises_writers_across_connections(tmp_path: Path) -> None:
    # Two processes revoke the same fact. Without BEGIN IMMEDIATE the first
    # one's "is it already revoked?" read is not protected, both insert, and
    # the fact ends up with two live tombstones.
    db = tmp_path / "e.db"
    one = await Engram.open(str(db))
    two = await Engram.open(str(db))
    try:
        a = await one.record(A, user_id="ops")
        reached, release = asyncio.Event(), asyncio.Event()
        conn = one._store._conn
        real_execute = conn.execute

        def execute(sql: str, *args: Any, **kwargs: Any) -> Any:
            if sql.lstrip().startswith("INSERT INTO revocations"):

                async def paused() -> Any:
                    reached.set()
                    await release.wait()
                    return await real_execute(sql, *args, **kwargs)

                return paused()
            return real_execute(sql, *args, **kwargs)

        conn.execute = execute  # type: ignore[method-assign]
        t1 = asyncio.create_task(one.revoke(a.id, reason="r", actor="p1", user_id="ops"))
        await reached.wait()
        t2 = asyncio.create_task(two.revoke(a.id, reason="r", actor="p2", user_id="ops"))
        await _settle(t2)
        release.set()
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1.id == r2.id
        assert len(await two.revocations(user_id="ops")) == 1
    finally:
        await one.close()
        await two.close()
