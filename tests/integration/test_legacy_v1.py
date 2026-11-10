"""The v1 REST routes the Java SDK and Spring Boot starter call (Rust v0.5.x shapes)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, tzinfo
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from engram import Engram
from engram.errors import EmbeddingError, ExtractionError, StoreError
from engram.governance import normalize_text
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Fact
from engram.server.http import build_http_app

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
    memory = await Engram.open(":memory:", llm=OneFactLLM())
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
    memory = await Engram.open(":memory:")
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
    memory = await Engram.open(":memory:")
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
    assert stats == {"total_facts": 1, "valid_facts": 0, "invalidated_facts": 1}


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
    monkeypatch.setattr("engram.engram.datetime", _FrozenClock)
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


def test_delete_user_refuses_when_the_audit_fields_quote_a_fact(client: TestClient) -> None:
    # The actor is the one token "restv1"; a fact whose whole text is that
    # token is quoted by it, so erase_user refuses before deleting anything.
    fid = _record(client, "restv1", user="carol")
    r = client.delete("/v1/memory/users/carol", headers=AUTH)
    assert r.status_code == 409
    assert set(r.json()) == {"error"}
    assert "calling again is safe" in r.json()["error"].lower()
    still = client.get(
        "/v1/memory/recall", params={"q": "restv1", "user_id": "carol"}, headers=AUTH
    )
    assert [h["fact_id"] for h in still.json()["results"]] == [fid]


@pytest.mark.asyncio
async def test_delete_user_store_error_is_503(
    served: tuple[TestClient, Engram], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, memory = served

    async def racing(**kw: Any) -> int:
        raise StoreError("a fact was written while the scope was erased")

    monkeypatch.setattr(memory, "erase_user", racing)
    r = client.delete("/v1/memory/users/alice", headers=AUTH)
    assert r.status_code == 503
    assert set(r.json()) == {"error"} and "retrying is safe" in r.json()["error"].lower()


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
    assert [(c["user_id"], c["org_id"], c["actor"]) for c in calls] == [
        ("alice", "default", "restv1")
    ] * 2
    reasons = [c["reason"] for c in calls]
    for value in [*reasons, calls[0]["actor"]]:
        assert len(normalize_text(value).split()) == 1, value
    assert all(r.startswith("deleteuser") for r in reasons)
    assert reasons[0] != reasons[1]
