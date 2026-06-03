"""HybridRetriever — merges vector + keyword + temporal candidates, optionally reranks."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from engram.embedding.base import EmbeddingProvider
from engram.models import Fact, LifecycleState, MemorySystem
from engram.retrieve.base import Reranker, RetrievalConfig, ScoredFact
from engram.retrieve.temporal import TemporalIntent, detect_temporal_intent
from engram.scope import Scope
from engram.store.base import EngramStore
from engram.vector.base import VectorStore

IntentConfidence = Literal["low", "medium", "high"]


class HybridRetriever:
    """6-signal hybrid retrieval (vector + keyword for now; graph + temporal in Phase 5+)."""

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
        memory_systems: tuple[MemorySystem, ...] | None = None,
        memory_subtypes: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        include_lifecycle_states: tuple[LifecycleState, ...] | None = None,
        exclude_lifecycle_states: tuple[LifecycleState, ...] | None = None,
        intent_confidence: IntentConfidence | None = None,
        allow_sensitive_recall: bool = False,
    ) -> list[ScoredFact]:
        if not query.strip():
            return []
        cfg = self._config
        active_memory_systems = memory_systems if memory_systems is not None else cfg.memory_systems
        active_memory_subtypes = (
            memory_subtypes if memory_subtypes is not None else cfg.memory_subtypes
        )
        active_tags = tags if tags is not None else cfg.tags
        active_include_lifecycle = (
            include_lifecycle_states
            if include_lifecycle_states is not None
            else cfg.include_lifecycle_states
        )
        active_exclude_lifecycle = (
            exclude_lifecycle_states
            if exclude_lifecycle_states is not None
            else cfg.exclude_lifecycle_states
        )
        if active_include_lifecycle is not None and exclude_lifecycle_states is None:
            active_exclude_lifecycle = tuple(
                state for state in active_exclude_lifecycle if state not in active_include_lifecycle
            )

        strict_filter = intent_confidence == "high" if intent_confidence is not None else True
        boost_weight = cfg.intent_boost_weight if intent_confidence in ("low", "medium") else 0.0

        candidate_k = top_k * cfg.candidate_pool_multiplier
        if any(
            filter_value is not None
            for filter_value in (
                active_memory_systems if strict_filter else None,
                active_memory_subtypes,
                active_tags,
                active_include_lifecycle,
            )
        ):
            candidate_k = max(candidate_k, 1000)
        intent = detect_temporal_intent(query)
        anchor = temporal_anchor or datetime.now(UTC)

        allowed_sessions: set[str] | None = None
        if cfg.enable_two_stage:
            session_scores = await self._facts.aggregate_sessions(
                query, scope, top_sessions=cfg.two_stage_top_sessions
            )
            if session_scores:
                allowed_sessions = {sid for sid, _ in session_scores}

        [q_vec] = await self._embed.embed([query])
        vec_matches = await self._vec.search(q_vec, scope, k=candidate_k)
        vec_scores: dict[UUID, float] = {m.fact_id: m.score for m in vec_matches}

        kw_facts = await self._facts.keyword_search(
            query,
            scope,
            limit=candidate_k,
            memory_systems=active_memory_systems if strict_filter else None,
            memory_subtypes=active_memory_subtypes,
            tags=active_tags,
            include_lifecycle_states=active_include_lifecycle,
            exclude_lifecycle_states=active_exclude_lifecycle,
        )
        kw_scores: dict[UUID, float] = {}
        for i, f in enumerate(kw_facts):
            kw_scores[f.id] = 1.0 - (i / max(1, candidate_k))

        all_ids: set[UUID] = set(vec_scores.keys()) | set(kw_scores.keys())
        if not all_ids:
            return []

        facts_by_id: dict[UUID, Fact] = {f.id: f for f in kw_facts}
        missing = [fid for fid in all_ids if fid not in facts_by_id]
        for fid in missing:
            fetched = await self._facts.get_fact(fid, scope)
            if fetched is not None:
                facts_by_id[fid] = fetched

        scored: list[ScoredFact] = []
        for fid, fact in facts_by_id.items():
            if cfg.exclude_superseded and fact.superseded_by is not None:
                continue
            if (
                strict_filter
                and active_memory_systems is not None
                and fact.memory_system not in active_memory_systems
            ):
                continue
            if (
                active_memory_subtypes is not None
                and fact.memory_subtype not in active_memory_subtypes
            ):
                continue
            if active_tags is not None and not set(active_tags).issubset(set(fact.tags)):
                continue
            if (
                active_include_lifecycle is not None
                and fact.lifecycle_state not in active_include_lifecycle
            ):
                continue
            if fact.lifecycle_state in active_exclude_lifecycle:
                continue
            if (
                allowed_sessions is not None
                and fact.session_id is not None
                and fact.session_id not in allowed_sessions
            ):
                continue
            if (
                not allow_sensitive_recall
                and fact.retrieval_policy == "explicit_only"
                and not active_memory_subtypes
                and not active_tags
            ):
                continue
            vs = vec_scores.get(fid, 0.0)
            ks = kw_scores.get(fid, 0.0)
            ts = (
                _temporal_score(fact, anchor, intent, cfg.temporal_sigma_days)
                if intent is not None
                else 0.0
            )

            boost = 0.0
            if (
                boost_weight > 0
                and active_memory_systems is not None
                and fact.memory_system in active_memory_systems
            ):
                boost = boost_weight

            final = (
                cfg.vector_weight * vs
                + cfg.keyword_weight * ks
                + cfg.temporal_weight * ts
                + boost
            )
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

        if self._reranker is not None:
            scored = await self._reranker.rerank(query, scored, top_k=top_k)
        else:
            scored = scored[:top_k]

        for sf in scored:
            await self._facts.record_access(sf.fact.id)

        return scored


def _temporal_score(
    fact: Fact, anchor: datetime, intent: TemporalIntent, sigma_days: float
) -> float:
    ref = fact.event_date or fact.mention_date
    if ref is None:
        return 0.5 if intent == TemporalIntent.DURATION else 0.0
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    days = abs((anchor - ref).total_seconds()) / 86400.0
    return float(math.exp(-(days * days) / (2.0 * sigma_days * sigma_days)))
