"""The v1 REST routes the Java SDK and Spring Boot starter call (Rust v0.5.x shapes)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import quote, quote_plus
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from engram import Engram
from engram.errors import EmbeddingError, ExtractionError, StoreError
from engram.governance import normalize_text
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Fact
from engram.scope import Scope
from engram.server.http import build_http_app
from engram.server.legacy_v1 import register_legacy_routes

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
NO_SUCH_FACT = "00000000-0000-0000-0000-000000000000"
LOGGER = "engram.server.legacy_v1"
_TICK = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


class _FrozenClock(datetime):
    """A clock that never advances, like a coarse one read twice in one tick."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return _TICK if tz is None else _TICK.astimezone(tz)


class OneFactLLM:
    """Each extraction call returns one fact naming its call number."""

    def __init__(self) -> None:
        self.n = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.n += 1
        fact = {
            "text": f"fact {self.n}: the user likes pizza",
            "category": "user_preference",
            "polarity": "affirmative",
            "confidence": 0.9,
            "entities": [],
            "event_date": None,
        }
        return LLMResponse(content=json.dumps({"facts": [fact]}), finish_reason="stop")


@pytest_asyncio.fixture
async def served() -> AsyncIterator[tuple[TestClient, Engram]]:
    memory = await Engram.open(":memory:", llm=OneFactLLM(), allow_default_scope=True)
    client = TestClient(build_http_app(memory, auth_token=TOKEN), base_url="http://127.0.0.1")
    try:
        yield client, memory
    finally:
        await memory.close()


@pytest.fixture
def client(served: tuple[TestClient, Engram]) -> TestClient:
    return served[0]


@pytest_asyncio.fixture
async def no_llm_client() -> AsyncIterator[TestClient]:
    memory = await Engram.open(":memory:", allow_default_scope=True)
    try:
        yield TestClient(build_http_app(memory, auth_token=TOKEN), base_url="http://127.0.0.1")
    finally:
        await memory.close()


def _add(c: TestClient, user: str = "alice") -> str:
    r = c.post(
        "/v1/memory",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": "I like pizza"}], "user_id": user},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["success"] is True and body["fact_count"] == 1
    return str(body["fact_ids"][0])


def _record(c: TestClient, text: str, user: str = "alice") -> str:
    r = c.post("/v1/memory/record", headers=AUTH, json={"text": text, "user_id": user})
    assert r.status_code == 200, r.text
    return str(r.json()["fact_id"])


def _save(c: TestClient, messages: list[dict[str, Any]], conv: str = "c1") -> dict[str, Any]:
    r = c.post(
        "/v1/memory/messages",
        headers=AUTH,
        json={"conversation_id": conv, "user_id": "alice", "messages": messages},
    )
    assert r.status_code == 201, r.text
    body: dict[str, Any] = r.json()
    return body


# ── Health and auth ─────────────────────────────────────────────────────


def test_health_is_open(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "engram"}
    assert client.head("/health").status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://evil.example"},
        {"Host": "evil.example"},
    ],
)
async def test_health_skips_the_origin_and_host_guards(headers: dict[str, str]) -> None:
    # Without a token both guards are on; /health is exempt from them like /healthz.
    memory = await Engram.open(":memory:", allow_default_scope=True)
    try:
        app = build_http_app(memory, allow_unauthenticated=True)
        c = TestClient(app, base_url="http://127.0.0.1")
        assert c.get("/health", headers=headers).status_code == 200
        assert c.get("/v1/memory/stats", headers=headers).status_code in (403, 421)
    finally:
        await memory.close()


@pytest.mark.parametrize("path", ["/health/", "/healthy", "/health.json"])
def test_health_exemption_is_exact(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v1/memory"),
        ("get", "/v1/memory/recall?q=x"),
        ("post", "/v1/memory/context"),
        ("delete", f"/v1/memory/facts/{NO_SUCH_FACT}"),
        ("get", "/v1/memory/search?q=x"),
        ("get", "/v1/memory/stats"),
        ("post", "/v1/memory/consolidate"),
        ("delete", "/v1/memory/users/alice"),
        ("post", "/v1/memory/messages"),
        ("get", "/v1/memory/messages/c1"),
        ("get", "/v1/memory/messages"),
        ("delete", "/v1/memory/messages/c1"),
    ],
)
def test_every_legacy_route_needs_the_token(client: TestClient, method: str, path: str) -> None:
    r = getattr(client, method)(path)
    assert r.status_code == 401
    assert r.json() == {"error": "invalid bearer token"}
    wrong = getattr(client, method)(path, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    # The middleware refuses unknown paths too, so also check the route exists.
    served = getattr(client, method)(path, headers=AUTH)
    assert served.status_code not in (401, 405)
    assert served.json() != {"detail": "Not Found"}


# ── Add, recall, search, context ────────────────────────────────────────


def test_add_needs_messages(client: TestClient) -> None:
    r = client.post("/v1/memory", headers=AUTH, json={"messages": [], "user_id": "alice"})
    assert r.status_code == 400
    assert "messages" in r.json()["error"]


def test_add_without_an_llm_is_400(no_llm_client: TestClient) -> None:
    r = no_llm_client.post(
        "/v1/memory", headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 400
    assert "LLM" in r.json()["error"]


def test_add_refuses_an_unknown_role(client: TestClient) -> None:
    r = client.post(
        "/v1/memory", headers=AUTH, json={"messages": [{"role": "robot", "content": "hi"}]}
    )
    assert r.status_code == 422


def test_add_without_a_session_leaves_the_default_conversation_alone(
    client: TestClient,
) -> None:
    # "default" is Spring AI's default conversation id. add stores no turns,
    # so its facts must not share a session with stored turns: forgetting one
    # would withhold that conversation (live_messages rule (c)).
    _save(client, [{"role": "user", "content": "hello there"}], conv="default")
    fid = _add(client)
    assert client.delete(f"/v1/memory/facts/{fid}", headers=AUTH).status_code == 200
    convs = client.get("/v1/memory/messages", params={"user_id": "alice"}, headers=AUTH)
    assert convs.json()["conversation_ids"] == ["default"]
    got = client.get("/v1/memory/messages/default", params={"user_id": "alice"}, headers=AUTH)
    assert got.json()["total"] == 1


@pytest.mark.asyncio
async def test_add_gives_each_call_its_own_session_unless_one_is_sent(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served
    sessions: list[str] = []
    extract = memory.extract

    async def spy(messages: list[ChatMessage], **kw: Any) -> list[Fact]:
        sessions.append(messages[0].session_id)
        return await extract(messages, **kw)

    monkeypatch.setattr(memory, "extract", spy)
    _add(client)
    _add(client)
    msgs = [{"role": "user", "content": "I like pizza"}]
    r = client.post("/v1/memory", headers=AUTH, json={"messages": msgs, "session_id": "s1"})
    assert r.status_code == 201
    assert sessions[2] == "s1"
    assert all(s.startswith("v1add") for s in sessions[:2]) and sessions[0] != sessions[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("2026-03-01T12:30:00+05:30", datetime(2026, 3, 1, 7, 0, tzinfo=UTC)),
        ("2026-03-01T07:00:00Z", datetime(2026, 3, 1, 7, 0, tzinfo=UTC)),
        ("2026-03-01T07:00:00", None),  # no offset, so not RFC 3339
        ("2026-03-01", None),
        ("yesterday", None),
        (None, None),
    ],
)
async def test_add_passes_an_rfc3339_timestamp_as_the_session_date(
    served: tuple[TestClient, Engram],
    monkeypatch: pytest.MonkeyPatch,
    sent: str | None,
    expected: datetime | None,
) -> None:
    client, memory = served
    dates: list[datetime | None] = []
    extract = memory.extract

    async def spy(messages: list[ChatMessage], **kw: Any) -> list[Fact]:
        dates.append(kw.get("session_date"))
        return await extract(messages, **kw)

    monkeypatch.setattr(memory, "extract", spy)
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "I like pizza"}]}
    if sent is not None:
        body["timestamp"] = sent
    r = client.post("/v1/memory", headers=AUTH, json=body)
    assert r.status_code == 201, r.text
    assert dates == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ExtractionError, EmbeddingError])
async def test_add_extraction_failure_is_500_with_an_error_key(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    client, memory = served

    async def failing(*args: Any, **kw: Any) -> list[Fact]:
        raise error("provider said: Incorrect API key provided: sk-abc***xyz")

    monkeypatch.setattr(memory, "extract", failing)
    r = client.post(
        "/v1/memory", headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 500
    assert set(r.json()) == {"error"} and error.__name__ in r.json()["error"]
    assert "sk-" not in r.json()["error"]


def test_missing_user_id_is_the_default_user(client: TestClient) -> None:
    r = client.post(
        "/v1/memory", headers=AUTH, json={"messages": [{"role": "user", "content": "pizza"}]}
    )
    fid = r.json()["fact_ids"][0]
    mine = client.get("/v1/memory/recall", params={"q": "pizza"}, headers=AUTH).json()
    assert [h["fact_id"] for h in mine["results"]] == [fid]
    other = client.get(
        "/v1/memory/recall", params={"q": "pizza", "user_id": "alice"}, headers=AUTH
    ).json()
    assert other["results"] == []


def test_recall_rust_and_native_shapes(client: TestClient) -> None:
    fid = _add(client)
    rust = client.get(
        "/v1/memory/recall",
        params={"q": "pizza", "user_id": "alice", "max_results": 5},
        headers=AUTH,
    ).json()
    assert rust["total"] == len(rust["results"]) >= 1
    hit = rust["results"][0]
    assert hit["fact_id"] == fid
    assert set(hit) == {"fact_id", "text", "tier", "category", "confidence"}
    assert hit["tier"] == "working" and hit["category"] == "user_preference"
    native = client.get(
        "/v1/memory/recall",
        params={"query": "pizza", "user_id": "alice", "top_k": 5},
        headers=AUTH,
    ).json()
    assert native["hits"][0]["fact_id"] == fid
    assert set(native["hits"][0]) == {
        "fact_id",
        "text",
        "score",
        "vector_score",
        "keyword_score",
        "rerank_score",
        "category",
        "polarity",
        "event_date",
    }


def test_recall_hits_serialize_as_the_native_search_does(client: TestClient) -> None:
    # POST /search still goes through response_model; GET /recall now builds
    # its hits by hand. The JSON must be the same, datetimes and enums included.
    client.post(
        "/v1/memory/record",
        headers=AUTH,
        json={
            "text": "alice ate pizza in Rome",
            "user_id": "alice",
            "event_date": "2026-03-01T12:30:00+05:30",
            "category": "event",
        },
    )
    _record(client, "alice likes pizza")
    params = {"query": "pizza", "user_id": "alice", "top_k": 5}
    native = client.post("/v1/memory/search", headers=AUTH, json=params).json()["hits"]
    got = client.get("/v1/memory/recall", params=params, headers=AUTH).json()["hits"]
    assert len(native) == 2 and any(h["event_date"] for h in native)
    assert got == native


def test_recall_max_results_limits_results(client: TestClient) -> None:
    for i in range(3):
        _record(client, f"pizza fact number {i}")
    r = client.get(
        "/v1/memory/recall",
        params={"q": "pizza", "user_id": "alice", "max_results": 2},
        headers=AUTH,
    ).json()
    assert r["total"] == len(r["results"]) == 2


def test_recall_needs_q_or_query(client: TestClient) -> None:
    r = client.get("/v1/memory/recall", params={"user_id": "alice"}, headers=AUTH)
    assert r.status_code == 422


def test_search_get_and_context_post(client: TestClient) -> None:
    fid = _add(client)
    s = client.get(
        "/v1/memory/search", params={"q": "pizza", "user_id": "alice", "top_k": 3}, headers=AUTH
    ).json()
    assert [h["fact_id"] for h in s["results"]] == [fid] and s["total"] == 1
    ctx = client.post(
        "/v1/memory/context",
        headers=AUTH,
        json={
            "query": "pizza",
            "user_id": "alice",
            "token_budget": 500,
            "format": "system_prompt",
        },
    ).json()
    assert "pizza" in ctx["text"]
    assert ctx["facts_included"] == 1 and ctx["facts_omitted"] == 0
    assert 0 < ctx["token_count"] <= 500


def test_context_counts_a_multiline_fact_once(client: TestClient) -> None:
    _record(client, "the user likes pizza\nand also pasta")
    ctx = client.post(
        "/v1/memory/context", headers=AUTH, json={"query": "pizza", "user_id": "alice"}
    ).json()
    assert "pasta" in ctx["text"]
    assert ctx["facts_included"] == 1 and ctx["facts_omitted"] == 0


def test_context_reports_facts_the_budget_left_out(client: TestClient) -> None:
    _record(client, "the user likes pizza with olives")
    _record(client, "the user likes pizza with onions")
    # 10 tokens is 40 characters: room for one "- ..." line of 34, not two.
    ctx = client.post(
        "/v1/memory/context",
        headers=AUTH,
        json={"query": "pizza", "user_id": "alice", "token_budget": 10},
    ).json()
    assert ctx["facts_included"] == 1 and ctx["facts_omitted"] == 1
    assert ctx["token_count"] <= 10


# ── Forget, stats, consolidate ──────────────────────────────────────────


def test_forget_then_stats(client: TestClient) -> None:
    fid = _add(client)
    r = client.request("DELETE", f"/v1/memory/facts/{fid}", headers=AUTH, json={"reason": "x"})
    assert r.status_code == 200
    assert r.json() == {"success": True, "deleted_fact_id": fid}
    gone = client.get("/v1/memory/recall", params={"q": "pizza", "user_id": "alice"}, headers=AUTH)
    assert gone.json()["results"] == []
    stats = client.get("/v1/memory/stats", headers=AUTH).json()
    assert stats == {
        "org_id": "default",
        "user_id": None,
        "total_facts": 1,
        "valid_facts": 0,
        "invalidated_facts": 1,
    }


@pytest.mark.asyncio
async def test_forget_records_the_reason_and_the_rest_actor(
    served: tuple[TestClient, Engram],
) -> None:
    client, memory = served
    fid = _add(client)
    r = client.request(
        "DELETE", f"/v1/memory/facts/{fid}", headers=AUTH, json={"reason": "user asked"}
    )
    assert r.status_code == 200
    [rev] = await memory.revocations(user_id="alice")
    assert (str(rev.fact_id), rev.reason, rev.actor) == (fid, "user asked", "restv1")


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [None, {}, {"reason": None}, {"reason": "   "}])
async def test_forget_without_a_reason_uses_a_default(
    served: tuple[TestClient, Engram], body: dict[str, Any] | None
) -> None:
    client, memory = served
    fid = _add(client)
    r = client.request("DELETE", f"/v1/memory/facts/{fid}", headers=AUTH, json=body)
    assert r.status_code == 200, r.text
    [rev] = await memory.revocations(user_id="alice")
    assert rev.reason and rev.reason.strip() == rev.reason
    assert rev.actor == "restv1"


def test_forget_unknown_fact_is_404(client: TestClient) -> None:
    r = client.delete(f"/v1/memory/facts/{NO_SUCH_FACT}", headers=AUTH)
    assert r.status_code == 404 and set(r.json()) == {"error"}


def test_forget_twice_is_idempotent(client: TestClient) -> None:
    fid = _add(client)
    for _ in range(2):
        r = client.delete(f"/v1/memory/facts/{fid}", headers=AUTH)
        assert r.json() == {"success": True, "deleted_fact_id": fid}


def test_forget_bad_id_is_400(client: TestClient) -> None:
    r = client.delete("/v1/memory/facts/not-a-uuid", headers=AUTH)
    assert r.status_code == 400
    assert "fact_id" in r.json()["error"]


def test_consolidate_is_501(client: TestClient) -> None:
    r = client.post("/v1/memory/consolidate", headers=AUTH, json={"user_id": "alice"})
    assert r.status_code == 501
    assert "consolidat" in r.json()["error"]


# ── Messages and delete-user ────────────────────────────────────────────


def test_messages_round_trip_and_delete_user(client: TestClient) -> None:
    saved = _save(
        client,
        [{"role": "user", "content": "my dog is Rex"}, {"role": "assistant", "content": "nice"}],
    )
    assert saved["success"] is True and len(saved["message_ids"]) == 2
    assert len(saved["fact_ids"]) == 1
    got = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH).json()
    assert [m["content"] for m in got["messages"]] == ["my dog is Rex", "nice"]
    assert [m["id"] for m in got["messages"]] == saved["message_ids"]
    assert got["total"] == 2
    first = got["messages"][0]
    keys = {"id", "conversation_id", "role", "content", "seq", "created_at", "metadata"}
    assert set(first) == keys
    assert (first["conversation_id"], first["role"], first["metadata"]) == ("c1", "user", {})
    assert first["created_at"].endswith("+00:00")
    convs = client.get("/v1/memory/messages", params={"user_id": "alice"}, headers=AUTH).json()
    assert convs == {"conversation_ids": ["c1"], "total": 1}
    _add(client, user="bob")
    r = client.delete("/v1/memory/users/alice", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"success": True, "deleted_facts": 1}
    left = client.get("/v1/memory/messages", params={"user_id": "alice"}, headers=AUTH).json()
    assert left["conversation_ids"] == []
    bob = client.get("/v1/memory/recall", params={"q": "pizza", "user_id": "bob"}, headers=AUTH)
    assert bob.json()["total"] == 1


def test_turns_of_one_save_keep_their_order_on_a_coarse_clock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every now() returns the same instant. Turns that tie on timestamp are
    # ordered by random id, so the route must give each turn its own time.
    monkeypatch.setattr("engram.scoped.datetime", _FrozenClock)
    monkeypatch.setattr("engram.server.legacy_v1.datetime", _FrozenClock)
    turns = [(role, f"turn {i}") for i, role in enumerate(["user", "assistant"] * 6)]
    _save(client, [{"role": r, "content": c} for r, c in turns])
    got = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH)
    assert [(m["role"], m["content"]) for m in got.json()["messages"]] == turns


def test_seq_is_the_turn_position_in_the_conversation(client: TestClient) -> None:
    _save(
        client,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    )
    every = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH)
    assert [m["seq"] for m in every.json()["messages"]] == [0, 1, 2]
    last = client.get(
        "/v1/memory/messages/c1", params={"user_id": "alice", "last_n": 2}, headers=AUTH
    ).json()
    assert [(m["content"], m["seq"]) for m in last["messages"]] == [("two", 1), ("three", 2)]
    assert last["total"] == 2


@pytest.mark.asyncio
async def test_forgetting_an_extracted_fact_withholds_its_conversation(
    served: tuple[TestClient, Engram],
) -> None:
    # Extracted facts carry the conversation id but no link to one turn, so
    # live_messages rule (c) withholds that whole conversation, turns saved
    # later included, until the fact is reinstated. Fails closed.
    client, memory = served
    saved = _save(client, [{"role": "user", "content": "my dog is Rex"}], conv="c1")
    _save(client, [{"role": "user", "content": "the weather is fine"}], conv="c2")
    [fid] = saved["fact_ids"]
    assert client.delete(f"/v1/memory/facts/{fid}", headers=AUTH).status_code == 200
    c1 = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH)
    assert c1.json() == {"messages": [], "total": 0}
    c2 = client.get("/v1/memory/messages/c2", params={"user_id": "alice"}, headers=AUTH)
    assert c2.json()["total"] == 1
    convs = client.get("/v1/memory/messages", params={"user_id": "alice"}, headers=AUTH)
    assert convs.json()["conversation_ids"] == ["c2"]
    _save(client, [{"role": "user", "content": "and a cat called Tom"}], conv="c1")
    later = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH)
    assert later.json()["total"] == 0
    await memory.reinstate(UUID(fid), reason="restored", actor="test", user_id="alice")
    back = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH)
    assert [m["content"] for m in back.json()["messages"]] == [
        "my dog is Rex",
        "and a cat called Tom",
    ]


@pytest.mark.parametrize("last_n", [0, -1])
def test_last_n_below_one_is_422(client: TestClient, last_n: int) -> None:
    r = client.get(
        "/v1/memory/messages/c1", params={"user_id": "alice", "last_n": last_n}, headers=AUTH
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_save_messages_extracts_from_the_stored_turns(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served
    seen: list[ChatMessage] = []
    extract = memory.extract

    async def spy(messages: list[ChatMessage], **kw: Any) -> list[Fact]:
        seen.extend(messages)
        return await extract(messages, **kw)

    monkeypatch.setattr(memory, "extract", spy)
    saved = _save(
        client, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    assert [str(m.id) for m in seen] == saved["message_ids"]
    stored = await memory.list_messages("c1", user_id="alice")
    assert [(m.id, m.timestamp) for m in seen] == [(m.id, m.timestamp) for m in stored]


def test_save_messages_without_an_llm_keeps_the_turns(no_llm_client: TestClient) -> None:
    # Rust saved the turns and reported fact_ids null when extraction was off
    # or failed. A 4xx here would make the client retry and store them twice.
    saved = _save(no_llm_client, [{"role": "user", "content": "hello"}])
    assert saved["success"] is True and len(saved["message_ids"]) == 1
    assert saved["fact_ids"] is None
    got = no_llm_client.get(
        "/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH
    ).json()
    assert got["total"] == 1


def test_save_without_an_llm_logs_at_debug_without_a_traceback(
    no_llm_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    _save(no_llm_client, [{"role": "user", "content": "hello"}])
    records = [r for r in caplog.records if r.name == LOGGER]
    assert [(r.levelno, r.exc_info) for r in records] == [(logging.DEBUG, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ExtractionError, EmbeddingError])
async def test_save_extraction_failure_keeps_the_turns_and_warns(
    served: tuple[TestClient, Engram],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: type[Exception],
) -> None:
    client, memory = served

    async def failing(*args: Any, **kw: Any) -> list[Fact]:
        raise error("provider down")

    monkeypatch.setattr(memory, "extract", failing)
    saved = _save(client, [{"role": "user", "content": "hello"}])
    assert saved["fact_ids"] is None and len(saved["message_ids"]) == 1
    warned = [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]
    assert len(warned) == 1 and warned[0].exc_info is not None


def test_save_messages_needs_messages(client: TestClient) -> None:
    r = client.post(
        "/v1/memory/messages", headers=AUTH, json={"conversation_id": "c1", "messages": []}
    )
    assert r.status_code == 400
    assert "messages" in r.json()["error"]


def test_save_messages_accepts_metadata_but_does_not_return_it(client: TestClient) -> None:
    # The revocation matcher reads turn content only, so metadata is not stored.
    _save(client, [{"role": "user", "content": "hi", "metadata": {"k": "v"}}])
    got = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH).json()
    assert got["messages"][0]["metadata"] == {}


def test_dropped_metadata_is_logged_once_per_request(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    _save(
        client,
        [
            {"role": "user", "content": "a", "metadata": {"k": "secret-value"}},
            {"role": "assistant", "content": "b", "metadata": {"k": "other"}},
        ],
    )
    _save(
        client, [{"role": "user", "content": "c", "metadata": {}}, {"role": "user", "content": "d"}]
    )
    dropped = [r for r in caplog.records if r.name == LOGGER and "metadata" in r.getMessage()]
    assert [r.levelno for r in dropped] == [logging.DEBUG]
    assert "secret-value" not in dropped[0].getMessage()


def test_delete_messages(client: TestClient) -> None:
    _save(client, [{"role": "user", "content": "hello"}])
    r = client.delete("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH)
    assert r.json() == {"success": True, "deleted_count": 1}
    got = client.get("/v1/memory/messages/c1", params={"user_id": "alice"}, headers=AUTH).json()
    assert got == {"messages": [], "total": 0}


# The old fixed actor, its spelling variants, and the reason's fixed prefix.
_PLANTS = ["restv1", "RESTv1.", "deleteuser", "restv1x", "deleteuser restv1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("forget_plant", [False, True])
async def test_delete_user_cannot_be_blocked_by_a_planted_fact(
    served: tuple[TestClient, Engram], forget_plant: bool
) -> None:
    client, memory = served
    _record(client, "Carol lives at 12 Main Street", user="carol")
    plants = [_record(client, text, user="carol") for text in _PLANTS]
    if forget_plant:  # withheld from every read path, still in the scope
        for fid in plants:
            assert client.delete(f"/v1/memory/facts/{fid}", headers=AUTH).status_code == 200
    r = client.delete("/v1/memory/users/carol", headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json() == {"success": True, "deleted_facts": len(_PLANTS) + 1}
    assert await memory._store.list_fact_ids(Scope(org_id="default", user_id="carol")) == []
    for q in ["restv1 deleteuser", "Main Street"]:
        got = client.get("/v1/memory/recall", params={"q": q, "user_id": "carol"}, headers=AUTH)
        assert got.json()["results"] == []


@pytest.mark.asyncio
async def test_delete_user_refused_audit_fields_is_409(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served

    async def refused(**kw: Any) -> int:
        raise ValueError("erasure actor must not quote the content being erased")

    monkeypatch.setattr(memory, "erase_user", refused)
    r = client.delete("/v1/memory/users/carol", headers=AUTH)
    assert r.status_code == 409
    assert set(r.json()) == {"error"}
    assert "calling again is safe" in r.json()["error"].lower()


@pytest.mark.asyncio
async def test_delete_user_residue_refusal_is_503(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served
    listed = memory._store.list_fact_ids

    async def list_then_write(scope: Scope) -> list[UUID]:
        ids = await listed(scope)
        await memory.record("alice likes jazz", user_id="alice")
        return ids

    monkeypatch.setattr(memory._store, "list_fact_ids", list_then_write)
    r = client.delete("/v1/memory/users/alice", headers=AUTH)
    assert r.status_code == 503
    assert set(r.json()) == {"error"} and "retrying is safe" in r.json()["error"].lower()


@pytest.mark.asyncio
async def test_delete_user_other_store_error_is_500_without_a_retry_promise(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served

    async def broken(**kw: Any) -> int:
        raise StoreError("erase: disk I/O error")

    monkeypatch.setattr(memory, "erase_user", broken)
    r = client.delete("/v1/memory/users/alice", headers=AUTH)
    assert r.status_code == 500
    assert set(r.json()) == {"error"}
    assert "retry" not in r.json()["error"].lower()
    assert "disk" not in r.json()["error"].lower()  # the cause goes to the log


@pytest.mark.asyncio
async def test_delete_user_audit_fields_are_single_tokens(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served
    calls: list[dict[str, Any]] = []

    async def capture(**kw: Any) -> int:
        calls.append(kw)
        return 0

    monkeypatch.setattr(memory, "erase_user", capture)
    for _ in range(2):
        assert client.delete("/v1/memory/users/alice", headers=AUTH).status_code == 200
    assert [(c["user_id"], c["org_id"]) for c in calls] == [("alice", "default")] * 2
    reasons = [c["reason"] for c in calls]
    actors = [c["actor"] for c in calls]
    for value in [*reasons, *actors]:
        assert len(normalize_text(value).split()) == 1, value
    assert all(r.startswith("deleteuser") for r in reasons)
    assert all(a.startswith("restv1x") and len(a) > len("restv1x") for a in actors)
    # Fresh on every call: no fixed string is left to plant.
    assert reasons[0] != reasons[1] and actors[0] != actors[1]


# ── Path ids as the published Java client sends them ────────────────────


def _save_as(c: TestClient, conv: str, user: str = "ivy") -> None:
    r = c.post(
        "/v1/memory/messages",
        headers=AUTH,
        json={
            "conversation_id": conv,
            "user_id": user,
            "messages": [{"role": "user", "content": f"note in {conv}"}],
        },
    )
    assert r.status_code == 201, r.text


@pytest.mark.parametrize("conv", ["support chat", "tenant/42", "a+b", "100% sure", "café/ünï"])
def test_message_routes_take_ids_form_encoded_like_urlencoder(
    client: TestClient, conv: str
) -> None:
    _save_as(client, conv)
    seg = quote_plus(conv)  # java.net.URLEncoder.encode(id, UTF_8)
    got = client.get(f"/v1/memory/messages/{seg}", params={"user_id": "ivy"}, headers=AUTH)
    assert got.status_code == 200, got.text
    assert [(m["conversation_id"], m["content"]) for m in got.json()["messages"]] == [
        (conv, f"note in {conv}")
    ]
    r = client.delete(f"/v1/memory/messages/{seg}", params={"user_id": "ivy"}, headers=AUTH)
    assert r.json() == {"success": True, "deleted_count": 1}
    assert (
        client.get("/v1/memory/messages", params={"user_id": "ivy"}, headers=AUTH).json()[
            "conversation_ids"
        ]
        == []
    )


def test_message_routes_read_plus_as_a_space_and_2b_as_a_plus(client: TestClient) -> None:
    _save_as(client, "a+b")
    _save_as(client, "a b")
    for path, conv in [("a+b", "a b"), ("a%2Bb", "a+b"), ("a%20b", "a b")]:
        got = client.get(f"/v1/memory/messages/{path}", params={"user_id": "ivy"}, headers=AUTH)
        assert [m["conversation_id"] for m in got.json()["messages"]] == [conv], path


def test_message_routes_refuse_an_empty_conversation_id(client: TestClient) -> None:
    for method in ("GET", "DELETE"):
        r = client.request(method, "/v1/memory/messages/", headers=AUTH)
        assert r.status_code == 400 and set(r.json()) == {"error"}, method


@pytest.mark.asyncio
async def test_delete_user_refuses_any_query_string(served: tuple[TestClient, Engram]) -> None:
    client, memory = served
    _record(client, "Bob's PIN hint is his cat", user="bob")
    for path in ("/v1/memory/users/bob?x=1", "/v1/memory/users/bob?=", "/v1/memory/users/bob?x#1"):
        r = client.delete(path, headers=AUTH)
        assert r.status_code == 400 and set(r.json()) == {"error"}, path
        assert "nothing was erased" in r.json()["error"].lower()
    r = client.delete("/v1/memory/users/bob", params={"org_id": "acme"}, headers=AUTH)
    assert r.status_code == 400
    assert len(await memory._store.list_fact_ids(Scope(org_id="default", user_id="bob"))) == 1
    assert await memory.erasures(user_id="bob") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", ["bob?", "bob#1234", "bob?#1"])
async def test_delete_user_cannot_see_an_empty_query_or_a_fragment(
    served: tuple[TestClient, Engram], sent: str
) -> None:
    """Pins the limit the legacy_v1 docstring and README state: these arrive as "bob"."""
    client, memory = served
    _record(client, "Bob's PIN hint is his cat", user="bob")
    r = client.delete("/v1/memory/users/" + sent, headers=AUTH)
    assert r.json() == {"success": True, "deleted_facts": 1}
    assert await memory._store.list_fact_ids(Scope(org_id="default", user_id="bob")) == []


@pytest.mark.asyncio
async def test_delete_user_takes_a_slash_in_the_id(served: tuple[TestClient, Engram]) -> None:
    client, memory = served
    _record(client, "tenant fact", user="tenant")
    _record(client, "tenant 42 fact", user="tenant/42")
    r = client.delete("/v1/memory/users/tenant/42", headers=AUTH)
    assert r.json() == {"success": True, "deleted_facts": 1}
    assert await memory._store.list_fact_ids(Scope(org_id="default", user_id="tenant/42")) == []
    assert len(await memory._store.list_fact_ids(Scope(org_id="default", user_id="tenant"))) == 1


# ── Low-risk input limits ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_refuses_a_reason_over_1000_characters(
    served: tuple[TestClient, Engram],
) -> None:
    client, memory = served
    path = f"/v1/memory/facts/{_add(client)}"
    r = client.request("DELETE", path, headers=AUTH, json={"reason": "a" * 1001})
    assert r.status_code == 422
    assert await memory.revocations(user_id="alice") == []
    r = client.request("DELETE", path, headers=AUTH, json={"reason": "a" * 1000})
    assert r.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"user_id": ""}, {"org_id": ""}, {"user_id": "", "org_id": ""}])
async def test_delete_messages_refuses_an_empty_scope_id(
    served: tuple[TestClient, Engram], params: dict[str, str]
) -> None:
    client, memory = served
    r = client.post(
        "/v1/memory/messages",
        headers=AUTH,
        json={"conversation_id": "default", "messages": [{"role": "user", "content": "mine"}]},
    )
    assert r.status_code == 201
    r = client.delete("/v1/memory/messages/default", params=params, headers=AUTH)
    assert r.status_code == 400 and set(r.json()) == {"error"}
    assert [m.content for m in await memory.list_messages("default")] == ["mine"]


def test_delete_user_refuses_an_empty_id(client: TestClient) -> None:
    r = client.delete("/v1/memory/users/", headers=AUTH)
    assert r.status_code == 400 and set(r.json()) == {"error"}


# ── Forget, stats and delete user name their scope ──────────────────────
#
# As in Rust (jamjet PR 193): `?org_id=` defaults to "default", and without
# `?user_id=` forget and stats act on the whole org.


def _record_in(c: TestClient, text: str, *, user: str, org: str = "default") -> str:
    r = c.post(
        "/v1/memory/record", headers=AUTH, json={"text": text, "user_id": user, "org_id": org}
    )
    assert r.status_code == 200, r.text
    return str(r.json()["fact_id"])


def _hits(c: TestClient, q: str, *, user: str, org: str = "default") -> list[str]:
    r = c.get("/v1/memory/recall", params={"q": q, "user_id": user, "org_id": org}, headers=AUTH)
    assert r.status_code == 200, r.text
    return [h["fact_id"] for h in r.json()["results"]]


@pytest.mark.asyncio
async def test_forget_cannot_reach_another_org(served: tuple[TestClient, Engram]) -> None:
    client, memory = served
    fid = _record_in(client, "alice keeps a red bike", user="alice", org="acme")
    unknown = client.delete(f"/v1/memory/facts/{NO_SUCH_FACT}", headers=AUTH).json()
    for params in ({}, {"org_id": "globex"}, {"org_id": "globex", "user_id": "alice"}):
        r = client.delete(f"/v1/memory/facts/{fid}", params=params, headers=AUTH)
        assert r.status_code == 404, params
        # Reads exactly like an unknown id.
        assert r.json() == {"error": unknown["error"].replace(NO_SUCH_FACT, fid)}, params
    assert await memory.revocations(user_id="alice", org_id="acme") == []
    assert _hits(client, "red bike", user="alice", org="acme") == [fid]

    r = client.delete(f"/v1/memory/facts/{fid}", params={"org_id": "acme"}, headers=AUTH)
    assert r.json() == {"success": True, "deleted_fact_id": fid}
    assert [
        str(rev.fact_id) for rev in await memory.revocations(user_id="alice", org_id="acme")
    ] == [fid]
    assert _hits(client, "red bike", user="alice", org="acme") == []


@pytest.mark.asyncio
async def test_forget_with_user_id_cannot_reach_another_user(
    served: tuple[TestClient, Engram],
) -> None:
    client, memory = served
    fid = _record_in(client, "bob keeps a red bike", user="bob")
    for params in ({"user_id": "alice"}, {"user_id": "alice", "org_id": "default"}):
        r = client.delete(f"/v1/memory/facts/{fid}", params=params, headers=AUTH)
        assert r.status_code == 404 and r.json() == {"error": f"fact {fid} not found"}, params
    assert await memory.revocations(user_id="bob") == []
    assert _hits(client, "red bike", user="bob") == [fid]

    r = client.delete(f"/v1/memory/facts/{fid}", params={"user_id": "bob"}, headers=AUTH)
    assert r.json() == {"success": True, "deleted_fact_id": fid}
    assert [str(rev.fact_id) for rev in await memory.revocations(user_id="bob")] == [fid]


@pytest.mark.asyncio
async def test_forget_with_both_ids_cannot_reach_the_same_user_in_another_org(
    served: tuple[TestClient, Engram],
) -> None:
    # Bob's fact is in the default org, so a user path that drops the org
    # would find it from any org.
    client, memory = served
    fid = _record_in(client, "bob keeps a red bike", user="bob")
    for params in ({"org_id": "globex", "user_id": "bob"}, {"org_id": "acme", "user_id": "bob"}):
        r = client.delete(f"/v1/memory/facts/{fid}", params=params, headers=AUTH)
        assert r.status_code == 404 and r.json() == {"error": f"fact {fid} not found"}, params
    assert await memory.revocations(user_id="bob") == []
    assert _hits(client, "red bike", user="bob") == [fid]


@pytest.mark.asyncio
async def test_forget_with_both_ids_revokes_in_the_named_org(
    served: tuple[TestClient, Engram],
) -> None:
    # Alice's fact is in acme only, so a user path that drops the org would
    # miss it.
    client, memory = served
    fid = _record_in(client, "alice keeps a blue kite", user="alice", org="acme")
    r = client.delete(
        f"/v1/memory/facts/{fid}", params={"org_id": "acme", "user_id": "alice"}, headers=AUTH
    )
    assert r.json() == {"success": True, "deleted_fact_id": fid}
    assert [
        str(rev.fact_id) for rev in await memory.revocations(user_id="alice", org_id="acme")
    ] == [fid]
    assert _hits(client, "blue kite", user="alice", org="acme") == []


async def _seed_counts(client: TestClient, memory: Engram) -> None:
    """default/alice: 2 facts, 1 forgotten. default/bob: 1. acme/alice: 1."""
    _record_in(client, "alice likes tea", user="alice")
    gone = _record_in(client, "alice likes jazz", user="alice")
    await memory.revoke(UUID(gone), reason="wrong", actor="test", user_id="alice")
    _record_in(client, "bob likes chess", user="bob")
    _record_in(client, "alice likes rowing", user="alice", org="acme")


def _stats(c: TestClient, **params: str) -> dict[str, Any]:
    r = c.get("/v1/memory/stats", params=params, headers=AUTH)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


@pytest.mark.asyncio
async def test_stats_count_only_the_named_scope(served: tuple[TestClient, Engram]) -> None:
    client, memory = served
    await _seed_counts(client, memory)
    assert _stats(client, user_id="alice") == {
        "org_id": "default",
        "user_id": "alice",
        "total_facts": 2,
        "valid_facts": 1,
        "invalidated_facts": 1,
    }
    assert _stats(client, user_id="alice", org_id="acme") == {
        "org_id": "acme",
        "user_id": "alice",
        "total_facts": 1,
        "valid_facts": 1,
        "invalidated_facts": 0,
    }
    assert _stats(client, user_id="carol") == {
        "org_id": "default",
        "user_id": "carol",
        "total_facts": 0,
        "valid_facts": 0,
        "invalidated_facts": 0,
    }


@pytest.mark.asyncio
async def test_stats_without_user_counts_the_org(served: tuple[TestClient, Engram]) -> None:
    client, memory = served
    await _seed_counts(client, memory)
    assert _stats(client) == {
        "org_id": "default",
        "user_id": None,
        "total_facts": 3,
        "valid_facts": 2,
        "invalidated_facts": 1,
    }
    assert _stats(client, org_id="acme") == {
        "org_id": "acme",
        "user_id": None,
        "total_facts": 1,
        "valid_facts": 1,
        "invalidated_facts": 0,
    }
    assert _stats(client, org_id="globex")["total_facts"] == 0


async def _fact_ids(memory: Engram, org: str, user: str) -> list[UUID]:
    return await memory._store.list_fact_ids(Scope(org_id=org, user_id=user))


@pytest.mark.asyncio
async def test_delete_user_erases_only_the_org_in_the_body(
    served: tuple[TestClient, Engram],
) -> None:
    client, memory = served
    _record_in(client, "bob keeps a red bike", user="bob")
    _record_in(client, "bob keeps a blue bike", user="bob", org="acme")
    r = client.request("DELETE", "/v1/memory/users/bob", headers=AUTH, json={"org_id": "acme"})
    assert r.json() == {"success": True, "deleted_facts": 1}
    assert await _fact_ids(memory, "acme", "bob") == []
    assert len(await _fact_ids(memory, "default", "bob")) == 1
    assert await memory.erasures(user_id="bob") == []
    assert await memory.erasures(user_id="bob", org_id="acme") != []

    # No body, or a null org, is the default org, as before.
    for kw in ({"json": {"org_id": None}}, {}):
        _record_in(client, "bob keeps a green bike", user="bob")
        r = client.request("DELETE", "/v1/memory/users/bob", headers=AUTH, **kw)
        assert r.json()["success"] is True, kw
        assert await _fact_ids(memory, "default", "bob") == [], kw


@pytest.mark.asyncio
async def test_delete_user_with_a_query_string_is_still_400(
    served: tuple[TestClient, Engram],
) -> None:
    # The Java client sends the user id unencoded, so `?org_id=` in the URL
    # could be part of the id. The org goes in the body.
    client, memory = served
    _record_in(client, "bob keeps a red bike", user="bob")
    _record_in(client, "bob keeps a blue bike", user="bob", org="acme")
    for path in ("/v1/memory/users/bob?org_id=acme", "/v1/memory/users/bob?org_id=default"):
        r = client.request("DELETE", path, headers=AUTH, json={"org_id": "acme"})
        assert r.status_code == 400 and set(r.json()) == {"error"}, path
    assert len(await _fact_ids(memory, "default", "bob")) == 1
    assert len(await _fact_ids(memory, "acme", "bob")) == 1
    assert await memory.erasures(user_id="bob") == []
    assert await memory.erasures(user_id="bob", org_id="acme") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [{"org": "acme"}, {"org_id": "acme", "user_id": "carol"}, ["acme"], "acme"],
    ids=["misspelt-key", "extra-key", "list", "string"],
)
async def test_delete_user_refuses_a_body_it_does_not_read(
    served: tuple[TestClient, Engram], body: Any
) -> None:
    # A misspelt org key must not fall back to the default org and erase there.
    client, memory = served
    _record_in(client, "bob keeps a red bike", user="bob")
    _record_in(client, "bob keeps a blue bike", user="bob", org="acme")
    r = client.request("DELETE", "/v1/memory/users/bob", headers=AUTH, json=body)
    assert r.status_code == 422
    assert len(await _fact_ids(memory, "default", "bob")) == 1
    assert len(await _fact_ids(memory, "acme", "bob")) == 1
    assert await memory.erasures(user_id="bob") == []


# ── Empty and blank ids are 400, and nothing is done ────────────────────
#
# Each route table below is compared with the app's own routes, so a new
# route fails these tests until its ids are listed.

IDS = ("user_id", "org_id")
BLANKS = ["", " ", "\t", "　"]
TURN = [{"role": "user", "content": "I like tea"}]

# (blank id, a fact id the seed stored in the default scope) -> (path, request kwargs)
Slot = Callable[[str, str], tuple[str, dict[str, Any]]]


def _in_body(path: str, field: str, **body: Any) -> Slot:
    return lambda blank, fid: (path.format(fid=fid), {"json": {**body, field: blank}})


def _in_query(path: str, field: str, **params: Any) -> Slot:
    return lambda blank, fid: (path.format(fid=fid), {"params": {**params, field: blank}})


V1_ID_SLOTS: dict[str, dict[str, Slot]] = {
    "GET /health": {},
    "HEAD /health": {},
    "POST /v1/memory/consolidate": {},  # reads no id; always 501
    "POST /v1/memory": {f: _in_body("/v1/memory", f, messages=TURN) for f in IDS},
    "GET /v1/memory/search": {f: _in_query("/v1/memory/search", f, q="tea") for f in IDS},
    "POST /v1/memory/context": {f: _in_body("/v1/memory/context", f, query="tea") for f in IDS},
    "DELETE /v1/memory/facts/{fact_id}": {f: _in_query("/v1/memory/facts/{fid}", f) for f in IDS},
    "GET /v1/memory/stats": {f: _in_query("/v1/memory/stats", f) for f in IDS},
    "DELETE /v1/memory/users/{user_id:path}": {
        "path user_id": lambda blank, fid: ("/v1/memory/users/" + quote(blank, safe=""), {}),
        "body org_id": _in_body("/v1/memory/users/default", "org_id"),
    },
    "POST /v1/memory/messages": {
        f: _in_body("/v1/memory/messages", f, conversation_id="c1", messages=TURN) for f in IDS
    },
    "GET /v1/memory/messages/{conversation_id:path}": {
        f: _in_query("/v1/memory/messages/c1", f) for f in IDS
    },
    "GET /v1/memory/messages": {f: _in_query("/v1/memory/messages", f) for f in IDS},
    "DELETE /v1/memory/messages/{conversation_id:path}": {
        f: _in_query("/v1/memory/messages/c1", f) for f in IDS
    },
}

_EXTRACT_TURN = [{"content": "I like tea", "role": "user", "session_id": "s1"}]

NATIVE_ID_SLOTS: dict[str, dict[str, Slot]] = {
    "GET /healthz": {},
    "HEAD /healthz": {},
    "POST /v1/memory/record": {f: _in_body("/v1/memory/record", f, text="I like tea") for f in IDS},
    "POST /v1/memory/search": {f: _in_body("/v1/memory/search", f, query="tea") for f in IDS},
    "POST /v1/memory/recall_with_provenance": {
        f: _in_body("/v1/memory/recall_with_provenance", f, query="tea") for f in IDS
    },
    "GET /v1/memory/recall": {f: _in_query("/v1/memory/recall", f, q="tea") for f in IDS},
    "GET /v1/memory/raw_facts": {f: _in_query("/v1/memory/raw_facts", f, query="tea") for f in IDS},
    "GET /v1/memory/context": {f: _in_query("/v1/memory/context", f, query="tea") for f in IDS},
    "POST /v1/memory/extract": {
        f: _in_body("/v1/memory/extract", f, messages=_EXTRACT_TURN) for f in IDS
    },
    "POST /v1/sessions/{session_id}/messages": {
        f: _in_body("/v1/sessions/s1/messages", f, content="hi", role="user", session_id="s1")
        for f in IDS
    },
}

MCP_TOOL_ARGS: dict[str, dict[str, Any]] = {
    "memory_record": {"text": "I like tea"},
    "memory_recall": {"query": "tea"},
    "memory_context": {"query": "tea"},
}


def _stub_engram() -> Engram:
    # Building the routes reads only this flag. No request reaches the stub.
    return cast(Engram, SimpleNamespace(allow_default_scope=True))


def _route_keys(app: FastAPI) -> list[str]:
    return sorted(
        f"{method} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    )


def _legacy_routes() -> list[str]:
    app = FastAPI()
    register_legacy_routes(app, _stub_engram())
    return _route_keys(app)


def _native_routes() -> list[str]:
    legacy = set(_legacy_routes())
    app = build_http_app(_stub_engram(), allow_unauthenticated=True)
    return [key for key in _route_keys(app) if key not in legacy]


def test_the_id_tables_list_every_route_the_app_serves() -> None:
    assert set(_legacy_routes()) == set(V1_ID_SLOTS)
    assert set(_native_routes()) == set(NATIVE_ID_SLOTS)


@dataclass
class _Seeded:
    memory: Engram
    llm: OneFactLLM
    client: TestClient
    fact_id: str  # a fact in the default scope, where a blank id used to land


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[_Seeded]:
    llm = OneFactLLM()
    memory = await Engram.open(":memory:", llm=llm, allow_default_scope=True)
    app = build_http_app(memory, auth_token=TOKEN)
    # A 500 shows as a status, so a red run says what the route answered.
    client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    try:
        mine = await memory.record("the default user likes tea")
        await memory.record("alice likes tea", user_id="alice")
        await memory.record_message("I like tea", session_id="c1")
        await memory.record_message("I like tea", session_id="c1", user_id="alice")
        yield _Seeded(memory, llm, client, str(mine.id))
    finally:
        await memory.close()


async def _everything(s: _Seeded) -> tuple[dict[str, list[str]], int]:
    """Every row of every table, and the LLM call count: what "nothing was done" means."""
    store = s.memory._store
    await store.flush_access()
    conn = store._conn
    async with conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'") as cur:
        tables = sorted(r[0] for r in await cur.fetchall())
    rows: dict[str, list[str]] = {}
    for table in tables:  # names from sqlite_master, not from a request
        async with conn.execute(f'SELECT * FROM "{table}"') as cur:
            rows[table] = sorted(repr(tuple(r)) for r in await cur.fetchall())
    return rows, s.llm.n


async def _refused_with_nothing_done(
    s: _Seeded, route: str, slots: dict[str, Slot], key: str, values: list[str] = BLANKS
) -> None:
    method = route.split(" ", 1)[0]
    for slot, build in slots.items():
        for value in values:
            before = await _everything(s)
            path, kw = build(value, s.fact_id)
            r = s.client.request(method, path, headers=AUTH, **kw)
            where = f"{route} {slot}={value!r}"
            assert r.status_code == 400, f"{where}: {r.status_code} {r.text}"
            assert set(r.json()) == {key}, where
            assert await _everything(s) == before, where


# Not blank, but no scope id: Scope refuses each, and the route must say so with 400.
NO_SCOPE_IDS = [" alice", "alice ", "a\x1fb", "a\x7fb"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "table", "key"),
    [
        *(pytest.param(r, V1_ID_SLOTS, "error", id=r) for r in _legacy_routes()),
        *(pytest.param(r, NATIVE_ID_SLOTS, "detail", id=r) for r in _native_routes()),
    ],
)
async def test_ids_a_scope_refuses_are_400_on_every_route(
    seeded: _Seeded, route: str, table: dict[str, dict[str, Slot]], key: str
) -> None:
    await _refused_with_nothing_done(seeded, route, table[route], key, NO_SCOPE_IDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _legacy_routes())
async def test_blank_ids_are_400_on_every_v1_route(seeded: _Seeded, route: str) -> None:
    assert route in V1_ID_SLOTS, f"{route}: list its ids in V1_ID_SLOTS"
    await _refused_with_nothing_done(seeded, route, V1_ID_SLOTS[route], "error")


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [*_native_routes(), *(f"MCP {tool}" for tool in MCP_TOOL_ARGS)])
async def test_native_routes_and_mcp_tools_refuse_blank_ids(seeded: _Seeded, target: str) -> None:
    if not target.startswith("MCP "):
        assert target in NATIVE_ID_SLOTS, f"{target}: list its ids in NATIVE_ID_SLOTS"
        await _refused_with_nothing_done(seeded, target, NATIVE_ID_SLOTS[target], "detail")
        return
    pytest.importorskip("mcp")
    from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

    from engram.server.mcp import build_mcp_server

    server = build_mcp_server(seeded.memory)
    listed = await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    assert {t.name for t in listed.root.tools} == set(MCP_TOOL_ARGS)
    tool = target.removeprefix("MCP ")
    for field in IDS:
        for blank in BLANKS:
            before = await _everything(seeded)
            params = CallToolRequestParams(
                name=tool, arguments={**MCP_TOOL_ARGS[tool], field: blank}
            )
            result = await server.request_handlers[CallToolRequest](
                CallToolRequest(method="tools/call", params=params)
            )
            where = f"{tool} {field}={blank!r}"
            assert result.root.isError is True, where
            assert "empty or blank" in result.root.content[0].text, where
            assert await _everything(seeded) == before, where
