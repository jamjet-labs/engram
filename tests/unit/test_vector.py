from __future__ import annotations

import math
from uuid import uuid4

import pytest

from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import StoreError
from engram.scope import Scope
from engram.vector.hnsw import HnswVectorStore


@pytest.fixture
def vec_store() -> HnswVectorStore:
    return HnswVectorStore(dim=64)


@pytest.fixture
def acme_alice() -> Scope:
    return Scope(org_id="acme", user_id="alice")


@pytest.fixture
def acme_bob() -> Scope:
    return Scope(org_id="acme", user_id="bob")


# ── Basics ──────────────────────────────────────────────────────────


async def test_add_then_search_returns_self(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    e = SyntheticEmbedding(dim=64)
    [v] = await e.embed(["alice prefers espresso"])
    fid = uuid4()
    await vec_store.add(fid, v, acme_alice)
    results = await vec_store.search(v, acme_alice, k=1)
    assert len(results) == 1
    assert results[0].fact_id == fid
    assert results[0].score >= 0.99  # near-identical, after normalization


async def test_search_ranks_relevant_above_irrelevant(
    vec_store: HnswVectorStore, acme_alice: Scope
) -> None:
    e = SyntheticEmbedding(dim=64)
    related_id = uuid4()
    unrelated_id = uuid4()
    [v_related] = await e.embed(["alice prefers espresso coffee"])
    [v_unrelated] = await e.embed(["the moon is made of cheese"])
    await vec_store.add(related_id, v_related, acme_alice)
    await vec_store.add(unrelated_id, v_unrelated, acme_alice)
    [q] = await e.embed(["espresso"])
    results = await vec_store.search(q, acme_alice, k=2)
    assert len(results) == 2
    assert results[0].fact_id == related_id


async def test_search_returns_empty_for_unknown_scope(
    vec_store: HnswVectorStore, acme_bob: Scope
) -> None:
    [q] = await SyntheticEmbedding(dim=64).embed(["anything"])
    results = await vec_store.search(q, acme_bob, k=10)
    assert results == []


# ── Scope isolation ─────────────────────────────────────────────────


async def test_scope_isolation(
    vec_store: HnswVectorStore, acme_alice: Scope, acme_bob: Scope
) -> None:
    e = SyntheticEmbedding(dim=64)
    [v_alice] = await e.embed(["alice fact"])
    [v_bob] = await e.embed(["bob fact"])
    a_id, b_id = uuid4(), uuid4()
    await vec_store.add(a_id, v_alice, acme_alice)
    await vec_store.add(b_id, v_bob, acme_bob)

    alice_results = await vec_store.search(v_alice, acme_alice, k=10)
    assert len(alice_results) == 1
    assert alice_results[0].fact_id == a_id

    bob_results = await vec_store.search(v_bob, acme_bob, k=10)
    assert len(bob_results) == 1
    assert bob_results[0].fact_id == b_id


# ── Counts + delete ─────────────────────────────────────────────────


async def test_count_starts_at_zero(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    assert await vec_store.count(acme_alice) == 0


async def test_count_increments_with_add(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    e = SyntheticEmbedding(dim=64)
    [v1] = await e.embed(["one"])
    [v2] = await e.embed(["two"])
    await vec_store.add(uuid4(), v1, acme_alice)
    await vec_store.add(uuid4(), v2, acme_alice)
    assert await vec_store.count(acme_alice) == 2


async def test_delete_removes_from_search(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    e = SyntheticEmbedding(dim=64)
    [v] = await e.embed(["delete me"])
    fid = uuid4()
    await vec_store.add(fid, v, acme_alice)
    assert await vec_store.count(acme_alice) == 1
    await vec_store.delete(fid, acme_alice)
    assert await vec_store.count(acme_alice) == 0
    results = await vec_store.search(v, acme_alice, k=10)
    assert results == []


async def test_delete_unknown_id_is_noop(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    await vec_store.delete(uuid4(), acme_alice)  # nothing to delete in empty scope
    assert await vec_store.count(acme_alice) == 0


# ── Validation ──────────────────────────────────────────────────────


async def test_add_rejects_wrong_dim(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    with pytest.raises(StoreError):
        await vec_store.add(uuid4(), [0.0] * 128, acme_alice)


async def test_search_rejects_wrong_dim(vec_store: HnswVectorStore, acme_alice: Scope) -> None:
    with pytest.raises(StoreError):
        await vec_store.search([0.0] * 128, acme_alice, k=1)


# ── Determinism ─────────────────────────────────────────────────────


async def test_same_inserts_same_results(acme_alice: Scope) -> None:
    """Repeated runs with same seed + same inputs produce same neighbor order."""
    e = SyntheticEmbedding(dim=64)
    texts = [f"fact {i}" for i in range(20)]
    vecs = await e.embed(texts)
    ids = [uuid4() for _ in range(20)]

    runs: list[list[str]] = []
    for _ in range(2):
        store = HnswVectorStore(dim=64, random_seed=42)
        for fid, v in zip(ids, vecs, strict=False):
            await store.add(fid, v, acme_alice)
        [q] = await e.embed(["fact 5"])
        results = await store.search(q, acme_alice, k=5)
        runs.append([str(r.fact_id) for r in results])
        await store.close()
    assert runs[0] == runs[1]


async def test_close_clears_state(acme_alice: Scope) -> None:
    e = SyntheticEmbedding(dim=64)
    [v] = await e.embed(["x"])
    store = HnswVectorStore(dim=64)
    await store.add(uuid4(), v, acme_alice)
    assert await store.count(acme_alice) == 1
    await store.close()
    assert await store.count(acme_alice) == 0


# ── Smoke check on cosine math ──────────────────────────────────────


def test_cosine_distance_to_similarity_math() -> None:
    # Identical vectors -> distance 0 -> similarity 1.0
    assert max(0.0, 1.0 - 0.0 / 2.0) == 1.0
    # Orthogonal vectors -> distance 1 -> similarity 0.5
    assert math.isclose(max(0.0, 1.0 - 1.0 / 2.0), 0.5)
    # Opposite vectors -> distance 2 -> similarity 0.0
    assert max(0.0, 1.0 - 2.0 / 2.0) == 0.0
