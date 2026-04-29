from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from engram.embedding.synthetic import SyntheticEmbedding
from engram.models import Fact
from engram.retrieve.hybrid import HybridRetriever
from engram.retrieve.temporal import TemporalIntent, detect_temporal_intent
from engram.scope import Scope
from engram.store.sqlite import SqliteStore
from engram.vector.hnsw import HnswVectorStore


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


# ── Temporal intent ────────────────────────────────────────────────


def test_temporal_intent_duration() -> None:
    assert detect_temporal_intent("how many days did I spend?") == TemporalIntent.DURATION


def test_temporal_intent_recency() -> None:
    # "ago" is the strongest recency signal; "last week" / "yesterday" too.
    assert detect_temporal_intent("how many days ago?") == TemporalIntent.RECENCY
    assert detect_temporal_intent("when did I last visit?") == TemporalIntent.RECENCY
    assert detect_temporal_intent("yesterday what did I do?") == TemporalIntent.RECENCY


def test_temporal_intent_ordering() -> None:
    assert detect_temporal_intent("did A happen before B?") == TemporalIntent.ORDERING


def test_temporal_intent_point_in_time() -> None:
    assert detect_temporal_intent("on Monday what did I do?") == TemporalIntent.POINT_IN_TIME


def test_temporal_intent_none_on_atemporal() -> None:
    assert detect_temporal_intent("what color do I prefer?") is None


# ── HybridRetriever ────────────────────────────────────────────────


@pytest_asyncio.fixture
async def setup_retriever() -> tuple[HybridRetriever, Scope, list[Fact]]:
    """Set up an in-memory retriever with 5 hand-curated facts."""
    fact_store = await SqliteStore.open(":memory:")
    embedder = SyntheticEmbedding(dim=64)
    vec_store = HnswVectorStore(dim=64)
    scope = Scope(org_id="acme", user_id="alice")

    fact_texts = [
        "alice prefers espresso over drip coffee",
        "alice works at acme corp",
        "alice's brother lives in tokyo",
        "the weather in paris is pleasant in spring",
        "alice has a pet cat named whiskers",
    ]
    facts: list[Fact] = []
    for text in fact_texts:
        f = Fact(
            text=text,
            scope=scope,
            valid_from=_now(),
            id=uuid4(),
        )
        await fact_store.upsert_fact(f)
        [v] = await embedder.embed([text])
        await vec_store.add(f.id, v, scope)
        facts.append(f)

    retriever = HybridRetriever(fact_store=fact_store, vector_store=vec_store, embedder=embedder)
    return retriever, scope, facts


async def test_search_empty_query_returns_empty(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, scope, _ = setup_retriever
    assert await retriever.search("", scope) == []


async def test_search_returns_relevant_first(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, scope, _facts = setup_retriever
    results = await retriever.search("coffee preference", scope, top_k=3)
    assert len(results) >= 1
    # The espresso fact should be first
    assert "espresso" in results[0].fact.text.lower()


async def test_search_keyword_only_match(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    """A query whose embedding is unrelated but whose tokens match exactly."""
    retriever, scope, _ = setup_retriever
    results = await retriever.search("whiskers", scope, top_k=3)
    assert len(results) >= 1
    assert any("whiskers" in r.fact.text.lower() for r in results)


async def test_search_respects_scope(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, _, _ = setup_retriever
    bob = Scope(org_id="acme", user_id="bob")
    results = await retriever.search("espresso", bob, top_k=3)
    assert results == []


async def test_search_top_k_bound(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, scope, _ = setup_retriever
    results = await retriever.search("alice", scope, top_k=2)
    assert len(results) <= 2


async def test_search_records_access(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, scope, _ = setup_retriever
    [hit] = await retriever.search("espresso", scope, top_k=1)
    fetched = await retriever._facts.get_fact(hit.fact.id, scope)
    assert fetched is not None
    assert fetched.access_count >= 1


# ── Stub reranker integration ──────────────────────────────────────


class _ReverseReranker:
    """Test reranker that just reverses input order. Proves wiring works."""

    async def rerank(self, query: str, scored: list, top_k: int = 10) -> list:
        out = list(reversed(scored))
        for s in out:
            s.rerank_score = -s.score  # arbitrary
        return out[:top_k]


async def test_temporal_score_zero_on_atemporal_query(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, scope, _ = setup_retriever
    results = await retriever.search("alice prefers", scope, top_k=3)
    # No temporal intent in query -> all temporal scores zero
    assert all(r.temporal_score == 0.0 for r in results)


async def test_temporal_score_nonzero_on_temporal_query() -> None:
    """Facts with event_date close to the temporal anchor get higher temporal_score."""
    from datetime import timedelta

    fact_store = await SqliteStore.open(":memory:")
    embedder = SyntheticEmbedding(dim=64)
    vec_store = HnswVectorStore(dim=64)
    scope = Scope(org_id="acme", user_id="alice")
    anchor = datetime(2026, 4, 30, tzinfo=UTC)

    recent = Fact(
        text="user did something yesterday",
        scope=scope,
        valid_from=anchor,
        event_date=anchor - timedelta(days=1),
        id=uuid4(),
    )
    distant = Fact(
        text="user did something long ago",
        scope=scope,
        valid_from=anchor,
        event_date=anchor - timedelta(days=365),
        id=uuid4(),
    )
    for f in (recent, distant):
        await fact_store.upsert_fact(f)
        [v] = await embedder.embed([f.text])
        await vec_store.add(f.id, v, scope)

    retriever = HybridRetriever(fact_store=fact_store, vector_store=vec_store, embedder=embedder)
    # "yesterday" triggers RECENCY intent
    results = await retriever.search(
        "what happened yesterday?", scope, top_k=2, temporal_anchor=anchor
    )
    # Recent fact should rank above distant
    assert "yesterday" in results[0].fact.text
    assert results[0].temporal_score > results[1].temporal_score
    await fact_store.close()
    await vec_store.close()


@pytest.mark.asyncio
async def test_reranker_is_invoked(
    setup_retriever: tuple[HybridRetriever, Scope, list[Fact]],
) -> None:
    retriever, scope, _ = setup_retriever
    retriever_with_rerank = HybridRetriever(
        fact_store=retriever._facts,
        vector_store=retriever._vec,
        embedder=retriever._embed,
        reranker=_ReverseReranker(),
    )
    reranked = await retriever_with_rerank.search("alice", scope, top_k=3)
    # The stub reranker writes rerank_score = -score; non-zero proves it ran.
    assert all(r.rerank_score != 0.0 for r in reranked)
    # And the order it produces is the reversed candidate pool, truncated.
    # Adjacent rerank_scores should be non-decreasing in absolute value
    # (we reversed ascending input → output is descending, so rerank_score
    # which is the negation is ascending → all negative).
    assert all(r.rerank_score <= 0.0 for r in reranked)
