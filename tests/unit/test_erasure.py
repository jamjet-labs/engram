"""Hard erasure: content and everything derived from it, leaving a content-free record."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from engram import Engram, NotFoundError
from engram.models import Event
from engram.scope import Scope

SECRET = "Alice was diagnosed with zebracornitis in March."
TOKEN = b"zebracorn"


def _disk_bytes(db: Path) -> bytes:
    out = b""
    for p in db.parent.iterdir():
        if p.name.startswith(db.name):
            out += p.read_bytes()
    return out


async def _seed(memory: Engram) -> dict[str, object]:
    scope = Scope(org_id="default", user_id="alice")
    turn = await memory.record_message(SECRET, session_id="s1", user_id="alice")
    await memory.record_message("thanks for telling me", session_id="s1", user_id="alice")
    # A transcript copy in another session with no link to the fact.
    await memory.record_message("fyi: " + SECRET.lower(), session_id="s2", user_id="alice")
    fact = await memory.record(
        SECRET, user_id="alice", session_id="s1", source_message_id=turn.id, actor="ingest"
    )
    copy = await memory.record(SECRET.upper(), user_id="alice", session_id="s3")
    derived = await memory.record(
        "Alice should avoid long flights while ill.", user_id="alice", derived_from=[fact.id]
    )
    keeper = await memory.record("Alice prefers espresso.", user_id="alice", session_id="s1")
    event = Event(
        scope=scope,
        subject_canonical="alice",
        verb="diagnosed",
        object_canonical="zebracornitis",
        time_start=datetime(2026, 3, 1, tzinfo=UTC),
        source_fact_ids=[fact.id],
    )
    await memory._store.upsert_event(event)
    return {"fact": fact, "copy": copy, "derived": derived, "keeper": keeper, "event": event}


async def test_erase_removes_content_copies_derivations_events_and_transcripts() -> None:
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        s = await _seed(memory)
        fact = s["fact"]
        record = await memory.erase(
            fact, reason="GDPR art. 17 request #42", actor="dpo", user_id="alice"
        )

        erased = set(record.fact_ids)
        assert {fact.id, s["copy"].id, s["derived"].id} == erased  # type: ignore[attr-defined]
        assert record.artifacts["events"] == 1
        assert record.artifacts["messages"] == 2  # linked turn + unlinked transcript copy
        assert record.artifacts["vectors"] == 3

        store = memory._store
        for fid in erased:
            assert await store.get_fact(fid, scope) is None
            assert await memory.explain(fid, user_id="alice") is None
        assert await store.get_event(s["event"].id, scope) is None  # type: ignore[attr-defined]
        assert [m.content for m in await store.list_messages("s1", scope)] == [
            "thanks for telling me"
        ]
        assert await store.list_messages("s2", scope) == []
        # Unrelated memory survives.
        got = await memory.recall("espresso", user_id="alice")
        assert s["keeper"].id in {r.fact.id for r in got}  # type: ignore[attr-defined]


async def test_erasure_record_holds_no_content() -> None:
    async with await Engram.open(":memory:") as memory:
        s = await _seed(memory)
        record = await memory.erase(s["fact"], reason="request 42", actor="dpo", user_id="alice")
        dumped = record.model_dump_json().lower()
        assert "zebracorn" not in dumped
        assert "flights" not in dumped
        [listed] = await memory.erasures(user_id="alice")
        assert listed.id == record.id
        assert "zebracorn" not in listed.model_dump_json().lower()


async def test_erase_rejects_a_reason_that_quotes_the_content() -> None:
    async with await Engram.open(":memory:") as memory:
        s = await _seed(memory)
        with pytest.raises(ValueError):
            await memory.erase(s["fact"], reason="erase: " + SECRET, actor="dpo", user_id="alice")


async def test_erase_requires_actor_and_reason_and_scope() -> None:
    async with await Engram.open(":memory:") as memory:
        s = await _seed(memory)
        with pytest.raises(ValueError):
            await memory.erase(s["fact"], reason="", actor="dpo", user_id="alice")
        with pytest.raises(ValueError):
            await memory.erase(s["fact"], reason="r", actor="", user_id="alice")
        with pytest.raises(NotFoundError):
            await memory.erase(s["fact"].id, reason="r", actor="dpo", user_id="mallory")  # type: ignore[attr-defined]
        with pytest.raises(ValueError):  # a Fact carries its own scope
            await memory.erase(s["fact"], reason="r", actor="dpo", user_id="mallory")


async def test_erase_scrubs_revocation_tombstones_and_can_free_the_content() -> None:
    """With block_reinsertion=False the store forgets the claim entirely."""
    async with await Engram.open(":memory:") as memory:
        s = await _seed(memory)
        rev = await memory.revoke(
            s["fact"], reason="wrong: zebracornitis was ruled out", actor="dr", user_id="alice"
        )
        await memory.erase(
            s["fact"], reason="request 42", actor="dpo", user_id="alice", block_reinsertion=False
        )
        [listed] = await memory.revocations(user_id="alice", include_reinstated=True)
        assert listed.id == rev.id
        assert "zebracorn" not in listed.model_dump_json().lower()
        again = await memory.record(SECRET, user_id="alice")
        explained = await memory.explain(again.id, user_id="alice")
        assert explained is not None
        assert explained.revocation_status == "active"


async def test_vector_index_entries_are_physically_dropped() -> None:
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        s = await _seed(memory)
        si = memory._vec._scopes[scope]
        labels = [si.uuid_to_int[s[k].id] for k in ("fact", "copy", "derived")]  # type: ignore[attr-defined]
        erased_vecs = np.array(si.index.get_items(labels))

        def stored_vectors() -> np.ndarray:
            idx = memory._vec._scopes[scope].index
            return np.array(idx.get_items(idx.get_ids_list()))

        def holds_any_erased(stored: np.ndarray) -> bool:
            return bool((np.abs(stored[:, None, :] - erased_vecs[None]).max(-1) < 1e-6).any())

        assert holds_any_erased(stored_vectors())  # control
        record = await memory.erase(s["fact"], reason="r", actor="dpo", user_id="alice")
        assert record.artifacts["vectors"] == 3
        assert not holds_any_erased(stored_vectors())
        si = memory._vec._scopes[scope]
        assert si.index.element_count == await memory._vec.count(scope) == 1
        got = await memory.recall("espresso", user_id="alice")
        assert s["keeper"].id in {r.fact.id for r in got}  # type: ignore[attr-defined]


async def test_erased_text_is_gone_from_the_database_files(tmp_path: Path) -> None:
    db = tmp_path / "erase.db"
    memory = await Engram.open(db)
    try:
        for i in range(40):
            await memory.record(f"filler fact number {i} about coffee", user_id="alice")
        s = await _seed(memory)
        await memory._store.checkpoint()
        assert TOKEN in _disk_bytes(db)  # control: the content is on disk before erasure
        record = await memory.erase(s["fact"], reason="request 42", actor="dpo", user_id="alice")
        assert record.guarantees["sqlite_secure_delete"] is True
        assert record.guarantees["wal_checkpointed"] is True
    finally:
        await memory.close()
    assert TOKEN not in _disk_bytes(db)


async def test_erasure_reports_a_blocked_wal_checkpoint_honestly(tmp_path: Path) -> None:
    """A reader holding a snapshot stops the checkpoint; the record must say so."""
    import sqlite3

    db = tmp_path / "busy.db"
    memory = await Engram.open(db)
    reader = sqlite3.connect(str(db))
    try:
        s = await _seed(memory)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM facts").fetchone()
        record = await memory.erase(s["fact"], reason="request 42", actor="dpo", user_id="alice")
        assert record.guarantees["wal_checkpointed"] is False
    finally:
        reader.close()
        await memory.close()
