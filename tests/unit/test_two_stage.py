"""Phase 9: two-stage (session-first) retrieval tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest_asyncio

from engram.embedding.synthetic import SyntheticEmbedding
from engram.models import Fact
from engram.retrieve.base import RetrievalConfig
from engram.retrieve.hybrid import HybridRetriever
from engram.scope import Scope
from engram.store.sqlite import SqliteStore
from engram.vector.hnsw import HnswVectorStore


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def setup() -> tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope]:
    """3 sessions x 3 facts each, with Session A heavily about coffee."""
    store = await SqliteStore.open(":memory:")
    embedder = SyntheticEmbedding(dim=64)
    vec = HnswVectorStore(dim=64)
    scope = Scope(org_id="acme", user_id="alice")

    sessions: dict[str, list[str]] = {
        "session-A": [
            "alice prefers espresso over drip coffee",
            "alice's favorite coffee shop is in tokyo",
            "alice drinks coffee every morning",
        ],
        "session-B": [
            "alice has a pet cat named whiskers",
            "alice's cat is gray and white",
            "alice feeds the cat at 7am",
        ],
        "session-C": [
            "alice works at acme corp",
            "alice's office is in shibuya",
            "alice's manager is named bob",
        ],
    }
    for sid, texts in sessions.items():
        for text in texts:
            f = Fact(id=uuid4(), text=text, scope=scope, valid_from=_now(), session_id=sid)
            await store.upsert_fact(f)
            [v] = await embedder.embed([text])
            await vec.add(f.id, v, scope)
    return store, vec, embedder, scope


# ── aggregate_sessions ──────────────────────────────────────────────


async def test_aggregate_sessions_ranks_topical_session_first(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    store, _, _, scope = setup
    ranked = await store.aggregate_sessions("coffee", scope, top_sessions=3)
    assert len(ranked) >= 1
    # Session A is dense in coffee references; should win.
    assert ranked[0][0] == "session-A"


async def test_aggregate_sessions_excludes_null_session_id(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    store, _, _embedder, scope = setup
    f = Fact(
        id=uuid4(), text="orphan fact about coffee", scope=scope, valid_from=_now()
    )  # no session_id
    await store.upsert_fact(f)
    ranked = await store.aggregate_sessions("coffee", scope, top_sessions=10)
    assert all(sid is not None for sid, _ in ranked)


async def test_aggregate_sessions_empty_query(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    store, _, _, scope = setup
    assert await store.aggregate_sessions("", scope) == []


# ── Two-stage retrieval ─────────────────────────────────────────────


async def test_two_stage_filters_to_top_session(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    store, vec, embedder, scope = setup
    cfg = RetrievalConfig(enable_two_stage=True, two_stage_top_sessions=1)
    retriever = HybridRetriever(fact_store=store, vector_store=vec, embedder=embedder, config=cfg)
    results = await retriever.search("coffee", scope, top_k=10)
    # All hits should be from session-A only
    sessions = {r.fact.session_id for r in results}
    assert sessions == {"session-A"}


async def test_two_stage_returns_facts_from_top_K_sessions(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    store, vec, embedder, scope = setup
    cfg = RetrievalConfig(enable_two_stage=True, two_stage_top_sessions=2)
    retriever = HybridRetriever(fact_store=store, vector_store=vec, embedder=embedder, config=cfg)
    results = await retriever.search("coffee", scope, top_k=10)
    sessions = {r.fact.session_id for r in results if r.fact.session_id}
    # No more than 2 distinct sessions
    assert len(sessions) <= 2


async def test_two_stage_off_returns_global_pool(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    store, vec, embedder, scope = setup
    cfg = RetrievalConfig(enable_two_stage=False)
    retriever = HybridRetriever(fact_store=store, vector_store=vec, embedder=embedder, config=cfg)
    # Query that hits multiple sessions
    results = await retriever.search("alice", scope, top_k=10)
    sessions = {r.fact.session_id for r in results if r.fact.session_id}
    # Without two-stage, we can hit all 3
    assert len(sessions) >= 2


async def test_two_stage_falls_through_when_session_id_missing(
    setup: tuple[SqliteStore, HnswVectorStore, SyntheticEmbedding, Scope],
) -> None:
    """If facts don't have session_id, two-stage should not block retrieval."""
    store = await SqliteStore.open(":memory:")
    embedder = SyntheticEmbedding(dim=64)
    vec = HnswVectorStore(dim=64)
    scope = Scope(org_id="acme", user_id="alice")
    f = Fact(id=uuid4(), text="alice prefers espresso", scope=scope, valid_from=_now())
    await store.upsert_fact(f)
    [v] = await embedder.embed([f.text])
    await vec.add(f.id, v, scope)

    cfg = RetrievalConfig(enable_two_stage=True)
    retriever = HybridRetriever(fact_store=store, vector_store=vec, embedder=embedder, config=cfg)
    results = await retriever.search("coffee preference", scope, top_k=5)
    # Stage-1 returns nothing (no facts have session_id) -> we should still
    # get the fact via global retrieval fall-through.
    assert len(results) == 1
    assert "espresso" in results[0].fact.text
    await store.close()
    await vec.close()
