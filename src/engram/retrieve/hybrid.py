"""HybridRetriever — merges vector + keyword + temporal candidates, optionally reranks."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import UUID

from engram.embedding.base import EmbeddingProvider
from engram.models import Fact
from engram.retrieve.base import Reranker, RetrievalConfig, ScoredFact
from engram.retrieve.temporal import TemporalIntent, detect_temporal_intent
from engram.scope import Scope
from engram.store.base import EngramStore
from engram.vector.base import VectorStore


class HybridRetriever:
    """6-signal hybrid retrieval (vector + keyword for now; graph + temporal in Phase 5+).

    Pipeline:
      1. Embed query
      2. Vector search top-(k * pool_multiplier)
      3. Keyword (FTS5) search top-(k * pool_multiplier)
      4. Merge by fact_id: score = vector_weight*vec + keyword_weight*kw
      5. Fetch full Fact rows from FactStore
      6. Optional rerank via Reranker; truncate to top_k
    """

    def __init__(
        self,
        fact_store: EngramStore,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._facts = fact_store
        self._vec = vector_store
        self._embed = embedder
        self._config = config or RetrievalConfig()
        self._reranker = reranker

    async def search(
        self,
        query: str,
        scope: Scope,
        top_k: int = 10,
        temporal_anchor: datetime | None = None,
    ) -> list[ScoredFact]:
        if not query.strip():
            return []
        cfg = self._config
        candidate_k = top_k * cfg.candidate_pool_multiplier
        intent = detect_temporal_intent(query)
        anchor = temporal_anchor or datetime.now(UTC)

        # 1. Vector search
        [q_vec] = await self._embed.embed([query])
        vec_matches = await self._vec.search(q_vec, scope, k=candidate_k)
        vec_scores: dict[UUID, float] = {m.fact_id: m.score for m in vec_matches}

        # 2. Keyword search
        kw_facts = await self._facts.keyword_search(query, scope, limit=candidate_k)
        # Normalize by inverse rank (top result -> 1.0, last -> 1/candidate_k)
        kw_scores: dict[UUID, float] = {}
        for i, f in enumerate(kw_facts):
            kw_scores[f.id] = 1.0 - (i / max(1, candidate_k))

        # 3. Merge candidate IDs
        all_ids: set[UUID] = set(vec_scores.keys()) | set(kw_scores.keys())
        if not all_ids:
            return []

        # 4. Resolve full Fact rows (some IDs may live only in vec store, some only in kw)
        # First, the keyword side already has the full Fact rows
        facts_by_id: dict[UUID, Fact] = {f.id: f for f in kw_facts}
        # Then fetch any missing IDs from fact_store
        missing = [fid for fid in all_ids if fid not in facts_by_id]
        for fid in missing:
            fetched = await self._facts.get_fact(fid, scope)
            if fetched is not None:
                facts_by_id[fid] = fetched

        # 5. Compute merged score
        scored: list[ScoredFact] = []
        for fid, fact in facts_by_id.items():
            vs = vec_scores.get(fid, 0.0)
            ks = kw_scores.get(fid, 0.0)
            ts = (
                _temporal_score(fact, anchor, intent, cfg.temporal_sigma_days)
                if intent is not None
                else 0.0
            )
            final = cfg.vector_weight * vs + cfg.keyword_weight * ks + cfg.temporal_weight * ts
            scored.append(
                ScoredFact(
                    fact=fact,
                    score=final,
                    vector_score=vs,
                    keyword_score=ks,
                    temporal_score=ts,
                )
            )

        scored.sort(key=lambda s: s.score, reverse=True)

        # 6. Optional rerank — over-fetch then truncate
        if self._reranker is not None:
            scored = await self._reranker.rerank(query, scored, top_k=top_k)
        else:
            scored = scored[:top_k]

        # 7. Record access for the surviving results
        for sf in scored:
            await self._facts.record_access(sf.fact.id)

        return scored


def _temporal_score(
    fact: Fact, anchor: datetime, intent: TemporalIntent, sigma_days: float
) -> float:
    """Return a [0, 1] temporal-relevance score for a fact given query intent.

    - If the fact has no event_date AND no mention_date, return 0.
    - For RECENCY/POINT_IN_TIME: Gaussian decay from anchor.
    - For DURATION: prefer facts with explicit event_date over those without.
    - For ORDERING: same as RECENCY (most recent gets highest score).
    """
    ref = fact.event_date or fact.mention_date
    if ref is None:
        return 0.5 if intent == TemporalIntent.DURATION else 0.0
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    days = abs((anchor - ref).total_seconds()) / 86400.0
    return float(math.exp(-(days * days) / (2.0 * sigma_days * sigma_days)))
