"""Embedder identity, query and document embedding, and embedding before the write.

Every embedder names itself: provider, model and vector size. The retriever
embeds a query with the embedder's `embed_query` when it has one; record,
extract and reinstate embed with `embed_documents`. Every vector is checked
before it is used. record and extract embed before they write, so an embedder
that fails, or answers with a bad vector, leaves nothing stored.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from engram.embedding import (
    EmbedderIdentity,
    OllamaEmbedding,
    SyntheticEmbedding,
    check_vector,
    embed_documents,
    embed_query,
    embedder_identity,
)
from engram.embedding.openai import OpenAIEmbedding
from engram.engram import Engram
from engram.errors import EmbeddingError, EmbeddingResponseError
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage
from engram.retrieve.hybrid import HybridRetriever
from engram.scope import Scope
from engram.vector.hnsw import HnswVectorStore
from tests.conformance.backends import OpenMemory, OpenStore

WHO = {"org_id": "acme", "user_id": "alice"}
ACME_ALICE = Scope(org_id="acme", user_id="alice")
DIM = 8
BIKE = "Alice keeps her bike in the shed."
HELMET = "Alice bought a new helmet."
GOOD = [0.5] * DIM


class PlainEmbedder:
    """Only `dim` and `embed`, all 0.2 asked of an embedder. Records each call."""

    def __init__(self, dim: int = DIM) -> None:
        self._inner = SyntheticEmbedding(dim=dim)
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed", list(texts)))
        return await self._inner.embed(texts)


class TaskTypeEmbedder(PlainEmbedder):
    """Embeds queries and documents apart, as a provider with task types does."""

    @property
    def identity(self) -> EmbedderIdentity:
        return EmbedderIdentity("test", "task-type", self.dim)

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", [text]))
        [v] = await self._inner.embed([text])
        return v

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("documents", list(texts)))
        return await self._inner.embed(texts)


class AnswerEmbedder:
    """Answers each call with what `answer` makes for that many texts."""

    identity = EmbedderIdentity("test", "fixed-answer", DIM)

    def __init__(self, answer: Callable[[int], list[list[float]]]) -> None:
        self._answer = answer

    @property
    def dim(self) -> int:
        return DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return self._answer(len(texts))


class FailingEmbedder:
    """Raises on every call, as a provider that is down does."""

    identity = EmbedderIdentity("test", "failing", DIM)

    @property
    def dim(self) -> int:
        return DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("the provider answered 503")


class TwoFactsLLM:
    """Extracts the same two facts from any conversation."""

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        facts = [{"text": text, "confidence": 0.9} for text in (BIKE, HELMET)]
        return LLMResponse(content=json.dumps({"facts": facts}), finish_reason="stop")


def _turn() -> ChatMessage:
    return ChatMessage(
        scope=ACME_ALICE,
        session_id="s1",
        role="user",
        content="My bike is in the shed, and I bought a new helmet.",
        timestamp=datetime.now(UTC),
    )


async def _assert_nothing_stored(memory: Engram) -> None:
    assert await memory._store.list_fact_ids(ACME_ALICE) == []
    assert await memory._store.keyword_search("bike helmet", ACME_ALICE) == []
    assert await memory._vec.count(ACME_ALICE) == 0


# ── Identity ────────────────────────────────────────────────────────────


def test_builtin_identities() -> None:
    builtins = [
        (SyntheticEmbedding(dim=384), EmbedderIdentity("synthetic", "blake2b-v1", 384)),
        (SyntheticEmbedding(dim=64), EmbedderIdentity("synthetic", "blake2b-v1", 64)),
        (OllamaEmbedding(), EmbedderIdentity("ollama", "nomic-embed-text:latest", 768)),
        (
            OllamaEmbedding(model="mxbai-embed-large", dim=1024),
            EmbedderIdentity("ollama", "mxbai-embed-large", 1024),
        ),
        (
            OpenAIEmbedding(api_key="sk-test"),
            EmbedderIdentity("openai", "text-embedding-3-small", 1536),
        ),
        (
            OpenAIEmbedding(model="text-embedding-3-large", api_key="sk-test"),
            EmbedderIdentity("openai", "text-embedding-3-large", 3072),
        ),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a built-in never warns
        for embedder, want in builtins:
            assert embedder.identity == want
            assert embedder_identity(embedder) == want
    assert SyntheticEmbedding(dim=384).identity.key == "synthetic/blake2b-v1/384"
    assert OllamaEmbedding().identity.key == "ollama/nomic-embed-text:latest/768"
    with pytest.raises(FrozenInstanceError):
        SyntheticEmbedding().identity.dim = 768  # type: ignore[misc]


def test_unknown_embedder_gets_a_class_identity_and_a_warning() -> None:
    embedder = PlainEmbedder(dim=12)
    with pytest.warns(UserWarning) as caught:
        identity = embedder_identity(embedder)
    assert identity == EmbedderIdentity(f"{__name__}.PlainEmbedder", "", 12)
    assert identity.key == f"{__name__}.PlainEmbedder//12"
    assert len(caught) == 1
    message = str(caught[0].message)
    assert f"{__name__}.PlainEmbedder" in message
    assert "identity" in message


def test_an_identity_must_be_one_and_agree_with_dim() -> None:
    class WrongSize(PlainEmbedder):
        identity = EmbedderIdentity("test", "wrong-size", 768)

    class NotAnIdentity(PlainEmbedder):
        identity = "test/not-an-identity/8"

    with pytest.raises(ValueError, match="768"):
        embedder_identity(WrongSize())
    with pytest.raises(TypeError, match="EmbedderIdentity"):
        embedder_identity(NotAnIdentity())


# ── Queries and documents ───────────────────────────────────────────────


async def test_retriever_uses_embed_query_when_present(open_store: OpenStore) -> None:
    store = await open_store()
    embedder = TaskTypeEmbedder()
    retriever = HybridRetriever(store, HnswVectorStore(dim=embedder.dim), embedder)
    await retriever.search("where is the bike", ACME_ALICE, top_k=3)
    assert embedder.calls == [("query", ["where is the bike"])]


async def test_retriever_without_embed_query_embeds_the_query_alone(
    open_store: OpenStore,
) -> None:
    store = await open_store()
    embedder = PlainEmbedder()
    retriever = HybridRetriever(store, HnswVectorStore(dim=embedder.dim), embedder)
    await retriever.search("where is the bike", ACME_ALICE, top_k=3)
    assert embedder.calls == [("embed", ["where is the bike"])]


async def test_record_uses_embed_documents(open_memory: OpenMemory) -> None:
    embedder = TaskTypeEmbedder()
    memory = await open_memory(embedder=embedder)
    fact = await memory.record(BIKE, **WHO)
    assert embedder.calls == [("documents", [BIKE])]
    [found] = await memory.recall("where is the bike", **WHO)
    assert found.fact.id == fact.id and found.vector_score > 0
    assert embedder.calls[-1] == ("query", ["where is the bike"])


async def test_extract_embeds_its_facts_in_one_call(open_memory: OpenMemory) -> None:
    embedder = TaskTypeEmbedder()
    memory = await open_memory(embedder=embedder, llm=TwoFactsLLM())
    facts = await memory.extract([_turn()])
    assert [f.text for f in facts] == [BIKE, HELMET]
    assert embedder.calls == [("documents", [BIKE, HELMET])]
    assert await memory._vec.count(ACME_ALICE) == 2


async def test_reinstate_uses_embed_documents(open_memory: OpenMemory) -> None:
    embedder = TaskTypeEmbedder()
    memory = await open_memory(embedder=embedder)
    fact = await memory.record(BIKE, **WHO)
    await memory.revoke(fact, reason="wrong", actor="test")
    assert await memory._vec.count(ACME_ALICE) == 0
    embedder.calls.clear()
    await memory.reinstate(fact, reason="it was right", actor="test")
    assert embedder.calls == [("documents", [BIKE])]
    assert await memory._vec.count(ACME_ALICE) == 1


async def test_no_texts_make_no_embedder_call() -> None:
    embedder = TaskTypeEmbedder()
    assert await embed_documents(embedder, []) == []
    assert embedder.calls == []


# ── Embed before the write ──────────────────────────────────────────────


async def test_record_stores_nothing_when_the_embedder_fails(open_memory: OpenMemory) -> None:
    memory = await open_memory(embedder=FailingEmbedder())
    with pytest.raises(EmbeddingError, match="503"):
        await memory.record(BIKE, **WHO)
    await _assert_nothing_stored(memory)


async def test_extract_stores_nothing_when_the_embedder_fails(open_memory: OpenMemory) -> None:
    memory = await open_memory(embedder=FailingEmbedder(), llm=TwoFactsLLM())
    with pytest.raises(EmbeddingError, match="503"):
        await memory.extract([_turn()])
    await _assert_nothing_stored(memory)


# ── Every vector is checked ─────────────────────────────────────────────

BAD_VECTORS: dict[str, list[float]] = {
    "one value short": [0.5] * (DIM - 1),
    "one value long": [0.5] * (DIM + 1),
    "empty": [],
    "NaN": [math.nan, *[0.5] * (DIM - 1)],
    "infinite": [*[0.5] * (DIM - 1), math.inf],
    "zero": [0.0] * DIM,
}


def _last_is(bad: list[float]) -> Callable[[int], list[list[float]]]:
    # Good vectors and one bad one, last, so nothing before it may be stored either.
    return lambda n: [list(GOOD) for _ in range(n - 1)] + [list(bad)]


BAD_ANSWERS: dict[str, Callable[[int], list[list[float]]]] = {
    **{name: _last_is(bad) for name, bad in BAD_VECTORS.items()},
    "one vector too many": lambda n: [list(GOOD) for _ in range(n + 1)],
    "one vector too few": lambda n: [list(GOOD) for _ in range(n - 1)],
}

WRITES: dict[str, Callable[[Engram], Awaitable[object]]] = {
    "record": lambda m: m.record(BIKE, **WHO),
    "extract": lambda m: m.extract([_turn()]),
}


@pytest.mark.parametrize("call", sorted(WRITES))
@pytest.mark.parametrize("answer", sorted(BAD_ANSWERS))
async def test_wrong_length_nan_or_zero_vector_is_refused_before_any_write(
    open_memory: OpenMemory, call: str, answer: str
) -> None:
    memory = await open_memory(embedder=AnswerEmbedder(BAD_ANSWERS[answer]), llm=TwoFactsLLM())
    with pytest.raises(EmbeddingResponseError):
        await WRITES[call](memory)
    await _assert_nothing_stored(memory)


@pytest.mark.parametrize("answer", sorted(BAD_ANSWERS))
async def test_a_bad_query_vector_is_refused(open_memory: OpenMemory, answer: str) -> None:
    memory = await open_memory(embedder=AnswerEmbedder(BAD_ANSWERS[answer]))
    with pytest.raises(EmbeddingResponseError):
        await memory.recall("where is the bike", **WHO)
    with pytest.raises(EmbeddingResponseError):
        await embed_query(memory._embed, "where is the bike")


async def test_a_bad_vector_is_named_by_its_index_never_its_text() -> None:
    embedder = AnswerEmbedder(lambda n: [list(GOOD), [math.nan] * DIM])
    with pytest.raises(EmbeddingResponseError, match="vector 1 of 2") as caught:
        await embed_documents(embedder, ["the first private note", "the second private note"])
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("bad", sorted(BAD_VECTORS))
def test_check_vector_refuses_a_bad_vector(bad: str) -> None:
    with pytest.raises(EmbeddingResponseError):
        check_vector(BAD_VECTORS[bad], DIM)


def test_check_vector_gives_back_floats() -> None:
    out = check_vector([1, 0, 0.5], 3)
    assert out == (1.0, 0.0, 0.5)
    assert all(type(x) is float for x in out)
    assert check_vector(GOOD, DIM) == tuple(GOOD)
