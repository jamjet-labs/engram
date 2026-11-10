"""A write in flight when erase_user runs must not land after the erase.

Each test holds a write at its slow step (the LLM call, the embedding, or
between two stored turns) with an asyncio.Event, runs a complete erase_user,
then lets the write finish. Nothing is mocked out of the check itself.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from engram import Engram
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import ScopeErasedError, StoreError
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Fact
from engram.scope import Scope
from engram.server.http import build_http_app

TOKEN = "r" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FRANK = Scope(org_id="default", user_id="frank")
SECRET = "Frank's diagnosis is type 2 diabetes"


class Gate:
    """Holds the first call that reaches it until released."""

    def __init__(self) -> None:
        self.armed = True
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def hold(self) -> None:
        if self.armed:
            self.armed = False
            self.entered.set()
            await self.release.wait()

    async def wait_entered(self) -> None:
        await asyncio.wait_for(self.entered.wait(), timeout=10)


class GatedLLM:
    """Returns one fact per call, after the gate opens."""

    def __init__(self, gate: Gate | None = None) -> None:
        self.gate = gate

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        if self.gate is not None:
            await self.gate.hold()
        fact = {
            "text": SECRET,
            "category": "health",
            "polarity": "affirmative",
            "confidence": 0.9,
            "entities": [],
            "event_date": None,
        }
        return LLMResponse(content=json.dumps({"facts": [fact]}), finish_reason="stop")


class GatedEmbedder(SyntheticEmbedding):
    """Holds the first embed call (made after the fact is stored) until released."""

    def __init__(self, gate: Gate) -> None:
        super().__init__(dim=384)
        self.gate = gate

    async def embed(self, texts: list[str]) -> list[list[float]]:
        await self.gate.hold()
        return await super().embed(texts)


async def _open(
    tmp_path: Path, llm: GatedLLM | None = None, embedder: SyntheticEmbedding | None = None
) -> Engram:
    # A file database, as served: reads go through the second connection.
    return await Engram.open(tmp_path / "e.db", llm=llm, embedder=embedder)


@pytest_asyncio.fixture
async def gate() -> AsyncIterator[Gate]:
    g = Gate()
    yield g
    g.release.set()


def _client(memory: Engram) -> httpx.AsyncClient:
    app = build_http_app(memory, auth_token=TOKEN)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", headers=AUTH
    )


def _turn() -> ChatMessage:
    return ChatMessage(
        scope=FRANK,
        session_id="s",
        role="user",
        content="I have type 2 diabetes",
        timestamp=datetime.now().astimezone(),
    )


async def _erase(memory: Engram) -> int:
    return await memory.erase_user(user_id="frank", reason="gdpr request", actor="dpo")


async def _assert_nothing_left(memory: Engram, c: httpx.AsyncClient | None = None) -> None:
    assert await memory._store.list_fact_ids(FRANK) == []
    assert await memory.recall("diagnosis diabetes", user_id="frank") == []
    assert await memory._vec.count(FRANK) == 0
    if c is not None:
        r = await c.get("/v1/memory/search", params={"q": "diagnosis diabetes", "user_id": "frank"})
        assert r.json()["results"] == []


# ── The LLM call: save_messages and add ─────────────────────────────────


async def test_save_messages_extraction_does_not_land_after_delete_user(
    tmp_path: Path, gate: Gate
) -> None:
    memory = await _open(tmp_path, llm=GatedLLM(gate))
    try:
        async with _client(memory) as c:
            save = asyncio.create_task(
                c.post(
                    "/v1/memory/messages",
                    json={
                        "conversation_id": "clinic",
                        "user_id": "frank",
                        "messages": [{"role": "user", "content": "I have type 2 diabetes"}],
                    },
                )
            )
            await gate.wait_entered()
            r = await c.request("DELETE", "/v1/memory/users/frank")
            assert r.status_code == 200 and r.json()["success"] is True
            gate.release.set()
            r = await save
            assert r.status_code == 201, r.text
            assert r.json()["fact_ids"] == []
            await _assert_nothing_left(memory, c)
            got = await c.get("/v1/memory/messages/clinic", params={"user_id": "frank"})
            assert got.json()["total"] == 0
    finally:
        await memory.close()


async def test_add_extraction_does_not_land_after_delete_user(tmp_path: Path, gate: Gate) -> None:
    memory = await _open(tmp_path, llm=GatedLLM(gate))
    try:
        async with _client(memory) as c:
            add = asyncio.create_task(
                c.post(
                    "/v1/memory",
                    json={
                        "user_id": "frank",
                        "messages": [{"role": "user", "content": "I have type 2 diabetes"}],
                    },
                )
            )
            await gate.wait_entered()
            r = await c.request("DELETE", "/v1/memory/users/frank")
            assert r.status_code == 200 and r.json()["success"] is True
            gate.release.set()
            r = await add
            assert r.status_code == 201, r.text
            assert r.json() == {"success": True, "fact_count": 0, "fact_ids": []}
            await _assert_nothing_left(memory, c)
    finally:
        await memory.close()


async def test_extract_logs_that_the_scope_was_erased(
    tmp_path: Path, gate: Gate, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="engram.engram")
    memory = await _open(tmp_path, llm=GatedLLM(gate))
    try:
        msg = _turn()
        task = asyncio.create_task(memory.extract([msg]))
        await gate.wait_entered()
        await _erase(memory)
        gate.release.set()
        assert await task == []
        [rec] = [r for r in caplog.records if "erased while extraction ran" in r.getMessage()]
        assert rec.levelname == "INFO"
        assert SECRET not in rec.getMessage()
    finally:
        await memory.close()


async def test_save_messages_turns_erased_before_extraction_starts(
    tmp_path: Path, gate: Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The erase lands between two stored turns, before extraction starts."""
    memory = await _open(tmp_path, llm=GatedLLM())
    stored = memory._store.upsert_message

    async def store_then_wait(message: ChatMessage) -> None:
        await stored(message)
        await gate.hold()

    monkeypatch.setattr(memory._store, "upsert_message", store_then_wait)
    try:
        async with _client(memory) as c:
            save = asyncio.create_task(
                c.post(
                    "/v1/memory/messages",
                    json={
                        "conversation_id": "clinic",
                        "user_id": "frank",
                        "messages": [
                            {"role": "user", "content": "I have type 2 diabetes"},
                            {"role": "assistant", "content": "Noted"},
                        ],
                    },
                )
            )
            await gate.wait_entered()
            assert (await c.request("DELETE", "/v1/memory/users/frank")).status_code == 200
            gate.release.set()
            r = await save
            assert r.status_code == 201, r.text
            assert r.json()["fact_ids"] == []
            await _assert_nothing_left(memory, c)
    finally:
        await memory.close()


# ── The embedding: record, and the window between upsert and vec.add ────


async def test_record_erased_before_its_vector_is_added_leaves_no_vector(
    tmp_path: Path, gate: Gate
) -> None:
    memory = await _open(tmp_path, embedder=GatedEmbedder(gate))
    try:
        task = asyncio.create_task(memory.record(SECRET, user_id="frank"))
        await gate.wait_entered()  # the fact is stored; its embedding is in flight
        assert await _erase(memory) == 1
        gate.release.set()
        fact = await task
        await _assert_nothing_left(memory)
        assert await memory._store.fact_exists(fact.id, FRANK) is False
    finally:
        await memory.close()


async def test_extract_erased_before_its_vector_is_added_leaves_no_vector(
    tmp_path: Path, gate: Gate
) -> None:
    memory = await _open(tmp_path, llm=GatedLLM(), embedder=GatedEmbedder(gate))
    try:
        msg = _turn()
        task = asyncio.create_task(memory.extract([msg]))
        await gate.wait_entered()
        assert await _erase(memory) == 1
        gate.release.set()
        assert await task == []
        await _assert_nothing_left(memory)
    finally:
        await memory.close()


async def test_record_refused_when_the_erase_lands_before_its_upsert(
    tmp_path: Path, gate: Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _open(tmp_path)
    checked = memory._store.is_content_revoked

    async def slow_check(text: str, scope: Scope) -> bool:
        await gate.hold()
        return await checked(text, scope)

    monkeypatch.setattr(memory._store, "is_content_revoked", slow_check)
    try:
        task = asyncio.create_task(memory.record(SECRET, user_id="frank", reject_revoked=True))
        await gate.wait_entered()
        await _erase(memory)
        gate.release.set()
        with pytest.raises(ScopeErasedError):
            await task
        await _assert_nothing_left(memory)
    finally:
        await memory.close()


async def test_record_route_reports_the_refusal_as_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = await _open(tmp_path)

    async def refused(**kw: Any) -> Fact:
        raise ScopeErasedError("the scope was erased while this write ran")

    monkeypatch.setattr(memory, "record", refused)
    try:
        async with _client(memory) as c:
            r = await c.post("/v1/memory/record", json={"text": "x", "user_id": "frank"})
        assert r.status_code == 409
        assert "erased" in r.json()["detail"]
    finally:
        await memory.close()


# ── Controls: writes that start after the erase are new data ────────────


async def test_writes_started_after_delete_user_are_stored(tmp_path: Path) -> None:
    memory = await _open(tmp_path, llm=GatedLLM())
    try:
        await memory.record("Frank likes tea", user_id="frank")
        assert await _erase(memory) == 1
        recorded = await memory.record("Frank likes coffee", user_id="frank")
        async with _client(memory) as c:
            r = await c.post(
                "/v1/memory/messages",
                json={
                    "conversation_id": "clinic",
                    "user_id": "frank",
                    "messages": [{"role": "user", "content": "I have type 2 diabetes"}],
                },
            )
            [extracted] = r.json()["fact_ids"]
            r = await c.post(
                "/v1/memory",
                json={"user_id": "frank", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.json()["fact_count"] == 1
        ids = {str(i) for i in await memory._store.list_fact_ids(FRANK)}
        assert {str(recorded.id), extracted} <= ids and len(ids) == 3
        assert await memory._vec.count(FRANK) == 3
    finally:
        await memory.close()


# ── The store check ──────────────────────────────────────────────────────


def test_scope_erased_error_is_not_a_store_error() -> None:
    assert not issubclass(ScopeErasedError, StoreError)


async def _residue_time(memory: Engram) -> datetime:
    [record] = [e for e in await memory.erasures(user_id="frank") if not e.fact_ids]
    assert record.artifacts["scope_erasure"] == 1
    return record.erased_at


@pytest.mark.parametrize(
    ("offset", "refused"),
    [
        (timedelta(0), True),  # the same instant: refused
        (-timedelta(microseconds=1), True),
        (timedelta(microseconds=1), False),
    ],
)
async def test_upsert_fact_compares_times_not_strings(
    tmp_path: Path, offset: timedelta, refused: bool
) -> None:
    memory = await _open(tmp_path)
    try:
        await _erase(memory)
        at = await _residue_time(memory)
        # The same instant written with another UTC offset, so the ISO
        # strings do not sort the way the times do.
        since = (at + offset).astimezone(timezone(timedelta(hours=-11)))
        fact = Fact(text="Frank likes tea", scope=FRANK, valid_from=at)
        if refused:
            with pytest.raises(ScopeErasedError):
                await memory._store.upsert_fact(fact, not_erased_since=since)
            assert await memory._store.list_fact_ids(FRANK) == []
        else:
            await memory._store.upsert_fact(fact, not_erased_since=since)
            assert await memory._store.list_fact_ids(FRANK) == [fact.id]
    finally:
        await memory.close()


async def test_only_a_scope_wide_erasure_in_the_same_scope_refuses(tmp_path: Path) -> None:
    memory = await _open(tmp_path)
    try:
        since = datetime.now().astimezone()
        # A per-fact erase in frank's scope, and a whole-scope erase of bob.
        fact = await memory.record("Frank likes tea", user_id="frank")
        await memory.erase(fact.id, reason="asked", actor="dpo", user_id="frank")
        await memory.erase_user(user_id="bob", reason="gdpr request", actor="dpo")
        late = Fact(text="Frank likes coffee", scope=FRANK, valid_from=since)
        await memory._store.upsert_fact(late, not_erased_since=since)
        assert await memory._store.list_fact_ids(FRANK) == [late.id]
    finally:
        await memory.close()
