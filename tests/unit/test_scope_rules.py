"""Scope rules: what a scope id may hold, and when the default scope is allowed.

A scope id is 1 to 128 characters. It is not blank, has no leading or
trailing whitespace, and holds no control character or lone surrogate. Root
`Engram` methods refuse a missing scope, and the literal "default", unless
the Engram was opened with ``allow_default_scope=True``. An empty or blank
id is refused always, governance calls included. The server keeps its
contract: a missing id there means "default".

The refusal tests open a strict Engram on a store that records every call,
so they also show that nothing reaches the store or the embedder first. The
tests that need stored data take `open_memory`, which opens with
``allow_default_scope=True``, so they run on every backend.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from engram import Engram
from engram.classify.base import QuestionType
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import ScopeRequiredError
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Fact
from engram.scope import Scope
from engram.store.sqlite import SqliteStore
from tests.conformance.backends import OpenMemory

FIELDS = ("org_id", "user_id")
BIKE = "Alice keeps her bike in the blue shed."
KEY = "Alice hides the spare key under the doormat."


def _scope_with(field: str, value: str) -> Scope:
    ids = {"org_id": "acme", "user_id": "alice"}
    ids[field] = value
    return Scope(**ids)


# ── What a scope id may hold ────────────────────────────────────────────


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [" ", "\t", "\u3000"], ids=["space", "tab", "ideographic"])
def test_blank_scope_ids_are_refused(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        _scope_with(field, value)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize(
    "char", ["\x00", "\x1f", "\x7f", "\x85"], ids=["nul", "unit-separator", "del", "next-line"]
)
def test_control_characters_are_refused(field: str, char: str) -> None:
    with pytest.raises(ValueError, match="control character"):
        _scope_with(field, f"ali{char}ce")


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [" alice", "alice ", "\u3000alice", "alice\u00a0"])
def test_leading_or_trailing_whitespace_is_refused(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="whitespace"):
        _scope_with(field, value)


ALLOWED = [
    "o'r",
    'say "hi"',
    "a\\b",
    "100%",
    "a_b",
    "acme/eng",
    "alice smith",
    "\u00fc",
    "u\u0308",
    "\U0001f98a",
    "x' OR '1'='1",
]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", ALLOWED)
def test_quote_backslash_percent_underscore_slash_are_allowed(field: str, value: str) -> None:
    assert getattr(_scope_with(field, value), field) == value


def test_scope_ids_are_not_normalized() -> None:
    # The same letter written two ways is two scopes.
    assert Scope(org_id="acme", user_id="\u00fc") != Scope(org_id="acme", user_id="u\u0308")


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("char", ["x", "\U0001f98a"], ids=["ascii", "emoji"])
def test_128_characters_allowed_129_refused(field: str, char: str) -> None:
    assert getattr(_scope_with(field, char * 128), field) == char * 128
    with pytest.raises(ValueError):
        _scope_with(field, char * 129)


@pytest.mark.parametrize("field", FIELDS)
def test_lone_surrogate_refused(field: str) -> None:
    with pytest.raises(ValueError):
        _scope_with(field, "ali\ud800ce")


# ── Root methods and the default scope ─────────────────────────────────


class SpyEmbedder(SyntheticEmbedding):
    """Synthetic vectors, and a record of every batch it was asked to embed."""

    def __init__(self) -> None:
        super().__init__(dim=384)
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await super().embed(texts)


class SpyLLM:
    """Answers every prompt with no facts and no events, and counts the calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content='{"facts": [], "events": []}', finish_reason="stop")


class RecordingStore:
    """Passes every call to a real store and records the method's name."""

    def __init__(self, inner: SqliteStore) -> None:
        self._inner = inner
        self.calls: list[str] = []

    @property
    def read_only(self) -> bool:
        return self._inner.read_only

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._inner, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return await method(*args, **kwargs)

        return call


async def _strict(
    embedder: SpyEmbedder, llm: SpyLLM | None = None
) -> tuple[Engram, RecordingStore]:
    store = RecordingStore(await SqliteStore.open(":memory:"))
    memory = await Engram.open(store=store, embedder=embedder, llm=llm)  # type: ignore[arg-type]
    return memory, store


Ids = dict[str, str]
RootCall = Callable[[Engram, Ids], Awaitable[object]]

# Every public Engram method that takes user_id or org_id, with the other
# arguments it needs. The test below finds the methods with inspect, so a new
# one fails until it is listed here.
ROOT_CALLS: dict[str, RootCall] = {
    "record": lambda m, ids: m.record(BIKE, **ids),
    "record_message": lambda m, ids: m.record_message("Thanks!", session_id="s1", **ids),
    "recall": lambda m, ids: m.recall("bike", **ids),
    "context": lambda m, ids: m.context("where is the bike", **ids),
    "context_entries": lambda m, ids: m.context_entries("where is the bike", **ids),
    "search_events": lambda m, ids: m.search_events("bike", **ids),
    "supersede": lambda m, ids: m.supersede(uuid4(), uuid4(), **ids),
    "revoke": lambda m, ids: m.revoke(uuid4(), reason="wrong", actor="test", **ids),
    "reinstate": lambda m, ids: m.reinstate(uuid4(), reason="right", actor="test", **ids),
    "erase": lambda m, ids: m.erase(uuid4(), reason="asked to", actor="test", **ids),
    "erase_user": lambda m, ids: m.erase_user(reason="asked to", actor="test", **ids),
    "explain": lambda m, ids: m.explain(uuid4(), **ids),
    "revocations": lambda m, ids: m.revocations(**ids),
    "erasures": lambda m, ids: m.erasures(**ids),
    "list_conversations": lambda m, ids: m.list_conversations(**ids),
    "list_messages": lambda m, ids: m.list_messages("s1", **ids),
    "delete_conversation": lambda m, ids: m.delete_conversation("s1", **ids),
}


def _methods_that_take_a_scope() -> list[str]:
    return sorted(
        name
        for name, member in inspect.getmembers(Engram, inspect.isfunction)
        if not name.startswith("_")
        and {"user_id", "org_id"} & set(inspect.signature(member).parameters)
    )


def test_root_call_table_lists_exactly_the_methods_that_take_a_scope() -> None:
    assert set(ROOT_CALLS) == set(_methods_that_take_a_scope())


@pytest.mark.parametrize("method", _methods_that_take_a_scope())
@pytest.mark.parametrize(
    "given",
    [{}, {"user_id": "alice"}, {"org_id": "acme"}],
    ids=["no-ids", "user-only", "org-only"],
)
async def test_every_root_method_refuses_a_missing_scope(method: str, given: Ids) -> None:
    assert method in ROOT_CALLS, f"{method} takes a scope: add it to ROOT_CALLS"
    embedder = SpyEmbedder()
    memory, store = await _strict(embedder)
    try:
        with pytest.raises(ScopeRequiredError, match="allow_default_scope"):
            await ROOT_CALLS[method](memory, given)
        assert store.calls == []
        assert embedder.calls == []
    finally:
        await memory.close()


class SpyDecomposer:
    def __init__(self) -> None:
        self.calls = 0

    async def decompose(self, question: str) -> list[str]:
        self.calls += 1
        return ["Where is the bike?", "Where is the key?"]


class SpyClassifier:
    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, query: str) -> QuestionType:
        self.calls += 1
        return QuestionType.SINGLE_SESSION_USER


async def test_context_refuses_the_scope_before_the_classifier_or_decomposer() -> None:
    embedder = SpyEmbedder()
    memory, store = await _strict(embedder)
    decomposer, classifier = SpyDecomposer(), SpyClassifier()
    memory._decomposer = decomposer  # type: ignore[assignment]
    question = "Where is the bike? Where is the key?"
    try:
        with pytest.raises(ScopeRequiredError):
            await memory.context(question, user_id="alice", classifier=classifier, decompose=True)
        with pytest.raises(ScopeRequiredError):
            await memory.context_entries(question, user_id="alice", decompose=True)
        assert (decomposer.calls, classifier.calls) == (0, 0)
        assert store.calls == []
        assert embedder.calls == []
    finally:
        await memory.close()


DEFAULT_IDS = [
    pytest.param({"user_id": "default", "org_id": "acme"}, id="default-user"),
    pytest.param({"user_id": "alice", "org_id": "default"}, id="default-org"),
    pytest.param({"user_id": "default", "org_id": "default"}, id="both-default"),
]


@pytest.mark.parametrize("given", DEFAULT_IDS)
async def test_explicit_default_needs_allow_default_scope(given: Ids) -> None:
    embedder, llm = SpyEmbedder(), SpyLLM()
    strict, store = await _strict(embedder, llm)
    try:
        assert strict.allow_default_scope is False  # strict unless asked
        refused: list[str] = []
        for name, call in ROOT_CALLS.items():
            try:
                await call(strict, given)
            except ScopeRequiredError:
                refused.append(name)
        assert refused == list(ROOT_CALLS)

        # A Fact or a message in the default scope needs the switch too.
        fact = Fact(text=BIKE, scope=Scope(**given), valid_from=datetime.now(UTC))
        message = ChatMessage(
            scope=Scope(**given),
            session_id="s1",
            role="user",
            content="I keep my bike in the shed.",
            timestamp=datetime.now(UTC),
        )
        governed: dict[str, Callable[[], Awaitable[object]]] = {
            "revoke": lambda: strict.revoke(fact, reason="wrong", actor="test"),
            "reinstate": lambda: strict.reinstate(fact, reason="right", actor="test"),
            "erase": lambda: strict.erase(fact, reason="asked to", actor="test"),
            "extract": lambda: strict.extract([message]),
            "extract persist=False": lambda: strict.extract([message], persist=False),
            "extract_events": lambda: strict.extract_events([message]),
            "extract_events persist=False": lambda: strict.extract_events([message], persist=False),
        }
        refused = []
        for name, gcall in governed.items():
            try:
                await gcall()
            except ScopeRequiredError:
                refused.append(name)
        assert refused == list(governed)
        assert store.calls == []
        assert embedder.calls == []
        assert llm.calls == 0
    finally:
        await strict.close()

    async with await Engram.open(":memory:", allow_default_scope=True) as allowed:
        stored = await allowed.record(BIKE, **given)
        assert stored.scope == Scope(**given)
        assert [h.fact.id for h in await allowed.recall("bike", **given)] == [stored.id]


async def test_allow_default_scope_keeps_0_2_behaviour() -> None:
    async with await Engram.open(":memory:", allow_default_scope=True) as memory:
        assert memory.allow_default_scope is True
        fact = await memory.record(BIKE)
        assert fact.scope == Scope(org_id="default", user_id="default")
        bob = await memory.record("Bob keeps his bike at work.", user_id="bob")
        assert bob.scope == Scope(org_id="default", user_id="bob")

        # Not given and "default" name the same scope.
        both: Ids = {"user_id": "default", "org_id": "default"}
        for ids in ({}, {"user_id": "default"}, {"org_id": "default"}, both):
            assert [h.fact.id for h in await memory.recall("bike", **ids)] == [fact.id]
        assert [h.fact.id for h in await memory.recall("bike", user_id="bob")] == [bob.id]

        turn = await memory.record_message("Where is my bike?", session_id="s1")
        assert turn.scope == Scope()
        assert await memory.list_conversations() == ["s1"]
        assert [m.id for m in await memory.list_messages("s1")] == [turn.id]
        record = await memory.erase(fact.id, reason="asked to", actor="test")
        assert record.fact_ids == [fact.id]

        # The switch never lets an empty or blank id through.
        for ids in ({"user_id": ""}, {"org_id": " "}):
            with pytest.raises(ScopeRequiredError, match="empty or blank"):
                await memory.recall("bike", **ids)


@pytest.mark.parametrize("value", ["yes", 1, None])
async def test_allow_default_scope_must_be_a_bool(value: object) -> None:
    # A truthy string such as "false" from a config file must not widen anything.
    with pytest.raises(TypeError, match="allow_default_scope must be True or False"):
        await Engram.open(":memory:", allow_default_scope=value)  # type: ignore[arg-type]


# ── Empty and blank ids, governance included ───────────────────────────


async def test_erase_with_empty_user_id_never_touches_default(open_memory: OpenMemory) -> None:
    memory = await open_memory()
    fact = await memory.record(BIKE, user_id="default")
    with pytest.raises(ScopeRequiredError):
        await memory.erase(fact.id, reason="asked to", actor="test", user_id="")
    assert await memory.explain(fact.id, user_id="default") is not None
    assert [h.fact.id for h in await memory.recall("bike", user_id="default")] == [fact.id]
    assert await memory.erasures(user_id="default") == []


EMPTY_IDS = [
    pytest.param({"user_id": ""}, id="empty-user"),
    pytest.param({"org_id": ""}, id="empty-org"),
    pytest.param({"user_id": "   "}, id="blank-user"),
    pytest.param({"org_id": "\t"}, id="blank-org"),
    pytest.param({"user_id": "", "org_id": ""}, id="both-empty"),
]


@pytest.mark.parametrize("given", EMPTY_IDS)
async def test_revoke_reinstate_explain_and_erase_user_refuse_an_empty_scope(
    open_memory: OpenMemory, given: Ids
) -> None:
    memory = await open_memory()
    kept = await memory.record(BIKE)
    revoked = await memory.record(KEY)
    await memory.revoke(revoked, reason="no longer true", actor="test")

    calls: dict[str, Callable[[], Awaitable[object]]] = {
        "revoke by id": lambda: memory.revoke(kept.id, reason="wrong", actor="t", **given),
        "revoke a Fact": lambda: memory.revoke(kept, reason="wrong", actor="t", **given),
        "reinstate by id": lambda: memory.reinstate(revoked.id, reason="ok", actor="t", **given),
        "reinstate a Fact": lambda: memory.reinstate(revoked, reason="ok", actor="t", **given),
        "erase by id": lambda: memory.erase(kept.id, reason="asked", actor="t", **given),
        "erase a Fact": lambda: memory.erase(kept, reason="asked", actor="t", **given),
        "supersede": lambda: memory.supersede(kept.id, revoked.id, **given),
        "explain": lambda: memory.explain(kept.id, **given),
        "erase_user": lambda: memory.erase_user(reason="asked", actor="t", **given),
    }
    refused: list[str] = []
    for name, call in calls.items():
        try:
            await call()
        except ScopeRequiredError as e:
            assert "empty or blank" in str(e)
            refused.append(name)
    assert refused == list(calls)

    # Nothing in the default scope changed.
    assert [h.fact.id for h in await memory.recall("bike")] == [kept.id]
    assert [r.fact_id for r in await memory.revocations()] == [revoked.id]
    assert await memory.erasures() == []


# ── Ids that are not plain strings ─────────────────────────────────────


NOT_STRING_IDS = [
    pytest.param({"user_id": b"default", "org_id": b"default"}, id="bytes-default"),
    pytest.param({"user_id": "alice", "org_id": bytearray(b"default")}, id="bytearray-default-org"),
    pytest.param({"user_id": bytearray(b"default"), "org_id": "acme"}, id="bytearray-default-user"),
    pytest.param({"user_id": b" ", "org_id": "acme"}, id="bytes-blank-user"),
]


@pytest.mark.parametrize("given", NOT_STRING_IDS)
async def test_an_id_that_is_not_a_string_is_refused(given: dict[str, Any]) -> None:
    # Scope decodes bytes to str, so b"default" would name the default scope
    # after the checks on the raw value had passed.
    embedder, llm = SpyEmbedder(), SpyLLM()
    strict, store = await _strict(embedder, llm)
    try:
        refused: list[str] = []
        for name, call in ROOT_CALLS.items():
            try:
                await call(strict, given)
            except ScopeRequiredError as e:
                assert "must be a string" in str(e)
                refused.append(name)
        assert refused == list(ROOT_CALLS)

        # The same for ids given alongside a Fact.
        acme_alice = Scope(org_id="acme", user_id="alice")
        fact = Fact(text=BIKE, scope=acme_alice, valid_from=datetime.now(UTC))
        governed: dict[str, Callable[[], Awaitable[object]]] = {
            "revoke": lambda: strict.revoke(fact, reason="wrong", actor="test", **given),
            "reinstate": lambda: strict.reinstate(fact, reason="right", actor="test", **given),
            "erase": lambda: strict.erase(fact, reason="asked to", actor="test", **given),
        }
        refused = []
        for name, gcall in governed.items():
            try:
                await gcall()
            except ScopeRequiredError as e:
                assert "must be a string" in str(e)
                refused.append(name)
        assert refused == list(governed)
        assert store.calls == []
        assert embedder.calls == []
        assert llm.calls == 0
    finally:
        await strict.close()

    # The switch does not change the rule.
    async with await Engram.open(":memory:", allow_default_scope=True) as allowed:
        with pytest.raises(ScopeRequiredError, match="must be a string"):
            await allowed.record(BIKE, **given)


class _NeverEqualStr(str):
    """Compares unequal to everything. Scope still stores it as a plain str."""

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return str.__hash__(self)


@pytest.mark.parametrize("field", FIELDS)
async def test_default_is_judged_on_the_id_the_scope_stores(field: str) -> None:
    embedder = SpyEmbedder()
    strict, store = await _strict(embedder)
    given: dict[str, Any] = {"org_id": "acme", "user_id": "alice"}
    given[field] = _NeverEqualStr("default")
    try:
        refused: list[str] = []
        for name, call in ROOT_CALLS.items():
            try:
                await call(strict, given)
            except ScopeRequiredError as e:
                assert '"default" names the default scope' in str(e)
                refused.append(name)
        assert refused == list(ROOT_CALLS)
        assert store.calls == []
        assert embedder.calls == []
    finally:
        await strict.close()


# ── The server keeps its contract ──────────────────────────────────────


async def test_server_still_defaults_missing_ids_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The server as engram-server starts it: the Java client sends no org and
    # often no user, and those mean "default".
    import uvicorn
    from httpx import ASGITransport, AsyncClient

    import engram.cli as cli

    seen: dict[str, Any] = {}

    async def fake_serve(self: uvicorn.Server, sockets: Any = None) -> None:
        transport = ASGITransport(app=self.config.app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1:9999") as c:
            r = await c.post("/v1/memory/record", json={"text": BIKE})
            seen["record"] = r.status_code
            fact_id = r.json()["fact_id"]
            r = await c.post(
                "/v1/memory/search",
                json={"query": "bike", "user_id": "default", "org_id": "default"},
            )
            seen["native search, default ids"] = [h["fact_id"] for h in r.json()["hits"]]
            r = await c.get("/v1/memory/search", params={"q": "bike"})
            seen["v1 search, no ids"] = [h["fact_id"] for h in r.json()["results"]]
            r = await c.get("/v1/memory/search", params={"q": "bike", "user_id": "bob"})
            seen["v1 search, another user"] = r.json()["results"]
            r = await c.post(
                "/v1/memory/messages",
                json={"conversation_id": "c1", "messages": [{"role": "user", "content": "hi"}]},
            )
            seen["save messages"] = r.status_code
            r = await c.get("/v1/memory/messages/c1")
            seen["turns"] = [m["content"] for m in r.json()["messages"]]
            r = await c.get("/v1/memory/messages")
            seen["conversations"] = r.json()["conversation_ids"]
            seen["fact_id"] = fact_id

    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    await cli.serve_engram(
        host="127.0.0.1",
        port=9999,
        db_path=":memory:",
        auth_token=None,
        allow_unauthenticated=True,
        log_level="INFO",
    )
    fact_id = seen.pop("fact_id")
    assert seen == {
        "record": 200,
        "native search, default ids": [fact_id],
        "v1 search, no ids": [fact_id],
        "v1 search, another user": [],
        "save messages": 201,
        "turns": ["hi"],
        "conversations": ["c1"],
    }


async def test_server_needs_an_engram_that_allows_the_default_scope() -> None:
    from fastapi import FastAPI

    from engram.server.http import build_http_app
    from engram.server.legacy_v1 import register_legacy_routes
    from engram.server.mcp import build_mcp_server

    async with await Engram.open(":memory:") as strict:
        with pytest.raises(ValueError, match="allow_default_scope=True"):
            build_http_app(strict, allow_unauthenticated=True)
        with pytest.raises(ValueError, match="allow_default_scope=True"):
            register_legacy_routes(FastAPI(), strict)
        with pytest.raises(ValueError, match="allow_default_scope=True"):
            build_mcp_server(strict)
    async with await Engram.open(":memory:", allow_default_scope=True) as allowed:
        build_http_app(allowed, allow_unauthenticated=True)
        build_mcp_server(allowed)
