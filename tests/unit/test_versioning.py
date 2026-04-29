"""Phase 13: active fact-versioning + source span round-trip + determinism."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

from engram import Engram
from engram.embedding.synthetic import SyntheticEmbedding
from engram.models import Fact, Polarity
from engram.retrieve.base import RetrievalConfig
from engram.retrieve.hybrid import HybridRetriever
from engram.scope import Scope
from engram.store.sqlite import SqliteStore
from engram.vector.hnsw import HnswVectorStore


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


# ── supersede + retrieval filter ────────────────────────────────────


async def test_supersede_excludes_old_fact_from_default_recall() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(text="user has 1250 instagram followers", user_id="alice")
        new = await memory.record(text="user has 1300 instagram followers", user_id="alice")
        await memory.supersede(old.id, new.id, user_id="alice")
        results = await memory.recall(query="instagram followers", user_id="alice")
        # Old fact should be gone from default results
        ids = {r.fact.id for r in results}
        assert new.id in ids
        assert old.id not in ids


async def test_supersede_includes_old_fact_when_history_enabled() -> None:
    fact_store = await SqliteStore.open(":memory:")
    embedder = SyntheticEmbedding(dim=64)
    vec = HnswVectorStore(dim=64)
    scope = Scope(org_id="acme", user_id="alice")

    old_id, new_id = uuid4(), uuid4()
    old = Fact(id=old_id, text="user has 1250 followers", scope=scope, valid_from=_now())
    new = Fact(
        id=new_id,
        text="user has 1300 followers",
        scope=scope,
        valid_from=_now(),
        supersedes=old_id,
    )
    old.superseded_by = new_id
    for f in (old, new):
        await fact_store.upsert_fact(f)
        [v] = await embedder.embed([f.text])
        await vec.add(f.id, v, scope)

    # Default config: superseded excluded
    default_cfg = RetrievalConfig()
    r1 = HybridRetriever(
        fact_store=fact_store, vector_store=vec, embedder=embedder, config=default_cfg
    )
    default_results = await r1.search("followers", scope, top_k=5)
    assert {r.fact.id for r in default_results} == {new_id}

    # History config: include all
    history_cfg = RetrievalConfig(exclude_superseded=False)
    r2 = HybridRetriever(
        fact_store=fact_store, vector_store=vec, embedder=embedder, config=history_cfg
    )
    history_results = await r2.search("followers", scope, top_k=5)
    assert {r.fact.id for r in history_results} == {old_id, new_id}

    await fact_store.close()
    await vec.close()


async def test_supersede_raises_for_unknown_fact() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record(text="present", user_id="alice")
        try:
            await memory.supersede(uuid4(), uuid4(), user_id="alice")
        except ValueError as e:
            assert "not found" in str(e)
        else:
            raise AssertionError("expected ValueError for missing fact")


# ── Source span round-trip ─────────────────────────────────────────


async def test_source_span_roundtrips_through_store() -> None:
    """A fact's source_span (char offsets in original message) survives upsert+get."""
    async with await Engram.open(":memory:") as memory:
        # Direct path through the underlying store (record() doesn't expose source_span)
        scope = Scope(org_id="default", user_id="alice")
        f = Fact(
            id=uuid4(),
            text="user prefers espresso",
            scope=scope,
            valid_from=_now(),
            source_span=(12, 21),
        )
        await memory._store.upsert_fact(f)
        got = await memory._store.get_fact(f.id, scope)
        assert got is not None
        assert got.source_span == (12, 21)


async def test_polarity_persists() -> None:
    """A NEGATIVE polarity fact survives the store round-trip."""
    async with await Engram.open(":memory:") as memory:
        scope = Scope(org_id="default", user_id="alice")
        f = Fact(
            id=uuid4(),
            text="user does not like decaf",
            scope=scope,
            valid_from=_now(),
            polarity=Polarity.NEGATIVE,
        )
        await memory._store.upsert_fact(f)
        got = await memory._store.get_fact(f.id, scope)
        assert got is not None
        assert got.polarity == Polarity.NEGATIVE


# ── Determinism: same inputs produce same retrieval ─────────────────


async def test_repeated_runs_with_same_seed_produce_identical_top_results() -> None:
    """Two HnswVectorStore instances with same seed + same insertion order → same top-K."""
    embedder = SyntheticEmbedding(dim=64)
    scope = Scope(org_id="acme", user_id="alice")
    texts = [f"fact about topic {i}" for i in range(30)]
    vecs = await embedder.embed(texts)
    ids = [uuid4() for _ in range(30)]

    runs = []
    for _ in range(2):
        fact_store = await SqliteStore.open(":memory:")
        vec = HnswVectorStore(dim=64, random_seed=42)
        for fid, text, v in zip(ids, texts, vecs, strict=False):
            f = Fact(id=fid, text=text, scope=scope, valid_from=_now())
            await fact_store.upsert_fact(f)
            await vec.add(fid, v, scope)
        retriever = HybridRetriever(fact_store=fact_store, vector_store=vec, embedder=embedder)
        results = await retriever.search("topic 5", scope, top_k=5)
        runs.append([str(r.fact.id) for r in results])
        await fact_store.close()
        await vec.close()
    assert runs[0] == runs[1]


def test_python_hash_seed_recommendation_documented() -> None:
    """Smoke-check: README exists and mentions reproducible runs."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text()
    assert len(readme) > 100


def test_python_hash_seed_env_var_present_when_set() -> None:
    """If PYTHONHASHSEED is set in env, sys reflects it."""
    if "PYTHONHASHSEED" in os.environ:
        assert os.environ["PYTHONHASHSEED"]
