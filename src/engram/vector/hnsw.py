"""hnswlib-backed in-memory vector store, scope-isolated.

Implementation notes
--------------------
- One HNSW index per `Scope` (org_id, user_id). Cheap to maintain and avoids
  the post-hoc filtering cost; trades RAM at extreme scale for simplicity at
  Phase 2 scale (≤1M facts/scope).
- Cosine similarity (HNSW `space="cosine"`) — vectors are L2-normalized
  internally, so search returns angular distance d∈[0,2]; we convert to
  similarity ∈[0,1] via `1 - d/2`.
- Deterministic insertion: items are stored with monotonic int ids
  derived from a per-scope counter. The fact UUID -> int id mapping is
  preserved in `_uuid_to_int` so deletes work.
- `random_seed=42` pins HNSW's randomized graph construction. Combined with
  insertion order this gives bit-identical indexes across runs (important for
  reproducible benchmarks; full SHA-256 level assignment lands in Phase 13).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import hnswlib  # type: ignore[import-untyped]
import numpy as np

from engram.errors import StoreError
from engram.scope import Scope
from engram.vector.base import VectorMatch


@dataclass
class _ScopeIndex:
    """Per-scope HNSW + bookkeeping."""

    index: Any  # hnswlib.Index
    uuid_to_int: dict[UUID, int] = field(default_factory=dict)
    int_to_uuid: dict[int, UUID] = field(default_factory=dict)
    deleted: set[UUID] = field(default_factory=set)
    next_id: int = 0


class HnswVectorStore:
    """Per-scope hnswlib indexes, all in process memory."""

    def __init__(
        self,
        dim: int,
        max_elements_per_scope: int = 1_000_000,
        ef_construction: int = 200,
        m: int = 16,
        random_seed: int = 42,
    ) -> None:
        self._dim = dim
        self._max_elements = max_elements_per_scope
        self._ef_construction = ef_construction
        self._m = m
        self._random_seed = random_seed
        self._scopes: dict[Scope, _ScopeIndex] = {}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def _get_or_create(self, scope: Scope) -> _ScopeIndex:
        with self._lock:
            existing = self._scopes.get(scope)
            if existing is not None:
                return existing
            idx = hnswlib.Index(space="cosine", dim=self._dim)
            idx.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._m,
                random_seed=self._random_seed,
            )
            idx.set_ef(64)
            si = _ScopeIndex(index=idx)
            self._scopes[scope] = si
            return si

    async def add(self, fact_id: UUID, vector: list[float], scope: Scope) -> None:
        if len(vector) != self._dim:
            raise StoreError(f"vector dim {len(vector)} does not match store dim {self._dim}")
        si = self._get_or_create(scope)
        with self._lock:
            si.deleted.discard(fact_id)
            existing = si.uuid_to_int.get(fact_id)
            if existing is not None:
                # Replace by deleting + re-adding under same int id
                si.index.mark_deleted(existing)
            int_id = si.next_id
            si.next_id += 1
            si.uuid_to_int[fact_id] = int_id
            si.int_to_uuid[int_id] = fact_id
            si.index.add_items(np.array([vector], dtype=np.float32), [int_id])

    async def search(self, query: list[float], scope: Scope, k: int = 10) -> list[VectorMatch]:
        if len(query) != self._dim:
            raise StoreError(f"query dim {len(query)} does not match store dim {self._dim}")
        si = self._scopes.get(scope)
        if si is None:
            return []
        # Live count = num_elements - deleted
        live = max(0, si.index.element_count - len(si.deleted))
        if live == 0:
            return []
        k = min(k, live)
        try:
            labels, distances = si.index.knn_query(np.array([query], dtype=np.float32), k=k)
        except RuntimeError:
            # hnswlib raises if you query an index with too few non-deleted points
            return []
        out: list[VectorMatch] = []
        for label, dist in zip(labels[0], distances[0], strict=False):
            uid = si.int_to_uuid.get(int(label))
            if uid is None or uid in si.deleted:
                continue
            # cosine-distance d in [0, 2] -> similarity in [0, 1]
            # Clamp tightly: hnswlib can return slightly negative distances
            # for identical vectors (e.g. -5e-8), pushing similarity over 1.0.
            sim = min(1.0, max(0.0, 1.0 - float(dist) / 2.0))
            out.append(VectorMatch(fact_id=uid, score=sim))
        return out

    async def delete(self, fact_id: UUID, scope: Scope) -> None:
        si = self._scopes.get(scope)
        if si is None:
            return
        with self._lock:
            int_id = si.uuid_to_int.get(fact_id)
            if int_id is None:
                return
            try:
                si.index.mark_deleted(int_id)
            except RuntimeError:
                # Already deleted — idempotent
                pass
            si.deleted.add(fact_id)

    async def count(self, scope: Scope) -> int:
        si = self._scopes.get(scope)
        if si is None:
            return 0
        return int(max(0, si.index.element_count - len(si.deleted)))

    async def close(self) -> None:
        # In-memory only; nothing to release.
        self._scopes.clear()
