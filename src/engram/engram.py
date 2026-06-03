"""High-level Engram facade — the embedded-mode entry point.

This is what `from engram import Engram` returns. It composes the store +
vector index + embedder + (optional) extractor + reranker into a single
async context-managed object.

For HTTP / MCP service exposure, see `engram.server`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

from engram.classify.base import QuestionClassifier, budget_for
from engram.embedding.base import EmbeddingProvider
from engram.embedding.synthetic import SyntheticEmbedding
from engram.extract.event_extractor import EventExtractor
from engram.extract.pipeline import ExtractionPipeline
from engram.llm.base import LLMClient
from engram.llm.tier import ModelTier
from engram.memory_router import RuleBasedMemoryRouter
from engram.models import (
    ChatMessage,
    Event,
    ExtractedFact,
    Fact,
    LifecycleState,
    MemorySystem,
    MemoryTier,
    Polarity,
    PromotionState,
)
from engram.query_intent import RuleBasedQueryIntentClassifier
from engram.read.decomposer import QueryDecomposer, should_decompose
from engram.retrieve.base import Reranker, RetrievalConfig, ScoredFact
from engram.retrieve.hybrid import HybridRetriever
from engram.retrieve.rrf import reciprocal_rank_fusion
from engram.scope import Scope
from engram.store.sqlite import SqliteStore
from engram.vector.hnsw import HnswVectorStore


class Engram:
    """The main embedded-mode Engram client.

    Open with `Engram.open(path)` and use as an async context manager:

        async with await Engram.open("./engram.db") as memory:
            await memory.record(user_id="alice", text="I prefer espresso.")
            facts = await memory.recall(user_id="alice", query="coffee preference")
    """

    def __init__(
        self,
        store: SqliteStore,
        vector_store: HnswVectorStore,
        embedder: EmbeddingProvider,
        retriever: HybridRetriever,
        extraction: ExtractionPipeline | None = None,
        event_extractor: EventExtractor | None = None,
        tier: ModelTier | None = None,
    ) -> None:
        self._store = store
        self._vec = vector_store
        self._embed = embedder
        self._retrieve = retriever
        self._extract = extraction
        self._events = event_extractor
        self._memory_router = RuleBasedMemoryRouter()
        self._query_intent = RuleBasedQueryIntentClassifier()
        self.tier = tier
        # Decomposer is constructed when a tier is supplied (uses utility LLM).
        # Caller can replace with `engram._decomposer = ...` for tests.
        self._decomposer: QueryDecomposer | None = (
            QueryDecomposer(tier.utility) if tier is not None else None
        )

    @classmethod
    async def open(
        cls,
        path: str | Path = ":memory:",
        embedder: EmbeddingProvider | None = None,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
        retrieval_config: RetrievalConfig | None = None,
        tier: ModelTier | None = None,
    ) -> Engram:
        """Open an Engram instance backed by SQLite + in-memory HNSW.

        Defaults:
        - embedder: SyntheticEmbedding(dim=384) — deterministic, offline-safe.
          Pass an OllamaEmbedding or OpenAIEmbedding for real semantics.
        - llm: None — extraction unavailable until provided.
        - reranker: None — set to a CrossEncoderReranker for higher precision.
        - tier: None — pass a ``ModelTier`` to split reader (answer generation)
          from utility (verifier, decomposer, ReAct brain). When ``tier`` is set
          and ``llm`` is not, ``tier.utility`` is used for extraction/event work.
        """
        store = await SqliteStore.open(path)
        embedder = embedder or SyntheticEmbedding(dim=384)
        vec = HnswVectorStore(dim=embedder.dim)
        retriever = HybridRetriever(
            fact_store=store,
            vector_store=vec,
            embedder=embedder,
            config=retrieval_config,
            reranker=reranker,
        )
        effective_llm = llm or (tier.utility if tier else None)
        extraction = ExtractionPipeline(effective_llm) if effective_llm is not None else None
        events = EventExtractor(effective_llm) if effective_llm is not None else None
        return cls(store, vec, embedder, retriever, extraction, events, tier=tier)

    async def close(self) -> None:
        await self._vec.close()
        await self._store.close()

    async def __aenter__(self) -> Engram:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ── Public API ──────────────────────────────────────────────────────

    async def extract_events(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
    ) -> list[Event]:
        """Phase 11: extract SVO events from a chat segment via LLM.

        Persists the events into the event calendar by default. Requires an
        LLM to have been passed at `Engram.open(llm=...)`.
        """
        if self._events is None:
            raise RuntimeError("event extraction requires an LLM; pass `llm=...` to Engram.open()")
        if not messages:
            return []
        events = await self._events.extract(messages, session_date=session_date)
        if persist:
            for ev in events:
                await self._store.upsert_event(ev)
        return events

    async def search_events(
        self,
        query: str = "",
        user_id: str = "default",
        org_id: str = "default",
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        limit: int = 10,
    ) -> list[Event]:
        """Phase 11: query the event calendar by FTS + optional time window."""
        return await self._store.search_events(
            query,
            Scope(org_id=org_id, user_id=user_id),
            time_start=time_start,
            time_end=time_end,
            limit=limit,
        )

    async def supersede(
        self,
        old_fact_id: UUID,
        new_fact_id: UUID,
        user_id: str = "default",
        org_id: str = "default",
    ) -> None:
        """Mark `old_fact_id` as superseded by `new_fact_id`.

        Phase 13 active versioning: superseded facts are filtered out of
        default retrieval (set `RetrievalConfig.exclude_superseded=False` to
        include the history).

        Both facts must exist in the same scope.
        """
        scope = Scope(org_id=org_id, user_id=user_id)
        old = await self._store.get_fact(old_fact_id, scope)
        new = await self._store.get_fact(new_fact_id, scope)
        if old is None:
            raise ValueError(f"old fact {old_fact_id} not found in scope")
        if new is None:
            raise ValueError(f"new fact {new_fact_id} not found in scope")
        old.superseded_by = new_fact_id
        new.supersedes = old_fact_id
        await self._store.upsert_fact(old)
        await self._store.upsert_fact(new)

    async def record(
        self,
        text: str,
        user_id: str = "default",
        org_id: str = "default",
        session_id: str | None = None,
        category: str | None = None,
        confidence: float = 1.0,
        polarity: Polarity = Polarity.AFFIRMATIVE,
        tier: MemoryTier = MemoryTier.WORKING,
        valid_from: datetime | None = None,
        event_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        role: str | None = None,
        memory_system: MemorySystem | str | None = None,
        memory_subtype: str | None = None,
        secondary_systems: tuple[MemorySystem | str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        lifecycle_state: LifecycleState | str | None = None,
        promotion_state: PromotionState | str | None = None,
        retrieval_policy: str | None = None,
        valid_until: datetime | None = None,
        route_memory: bool = False,
    ) -> Fact:
        """Record a single fact (no LLM extraction). Embeds + stores in one call.

        Pass `session_id` to enable Phase 9 two-stage retrieval — facts in the
        same session can then be retrieved together via session-first ranking.

        Pass `role` (e.g., "user", "assistant") to tag the fact's origin role.
        Stored in ``metadata["role"]``; queryable via ``Engram.context(role_filter=...)``.
        """
        scope = Scope(org_id=org_id, user_id=user_id)
        md = dict(metadata or {})
        if role is not None:
            md["role"] = role
        route = (
            self._memory_router.route(text, category=category, metadata=md)
            if route_memory
            else None
        )
        fact = Fact(
            text=text,
            scope=scope,
            valid_from=valid_from or datetime.now().astimezone(),
            session_id=session_id,
            category=category,
            confidence=confidence,
            polarity=polarity,
            tier=tier,
            memory_system=_coerce_memory_system(memory_system)
            or (route.memory_system if route is not None else MemorySystem.WORKING),
            memory_subtype=(
                memory_subtype
                if memory_subtype is not None
                else (route.memory_subtype if route else None)
            ),
            secondary_systems=_coerce_memory_system_list(secondary_systems)
            or (route.secondary_systems if route is not None else []),
            tags=list(tags) if tags is not None else (route.tags if route is not None else []),
            lifecycle_state=_coerce_lifecycle_state(lifecycle_state)
            or (route.lifecycle_state if route is not None else LifecycleState.DURABLE),
            promotion_state=_coerce_promotion_state(promotion_state)
            or (route.promotion_state if route is not None else PromotionState.RAW),
            retrieval_policy=retrieval_policy,
            valid_until=valid_until,
            event_date=event_date,
            metadata={**md, **({"memory_route_reason": route.reason} if route else {})},
        )
        await self._store.upsert_fact(fact)
        [vec] = await self._embed.embed([text])
        await self._vec.add(fact.id, vec, scope)
        return fact

    async def record_message(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
        user_id: str = "default",
        org_id: str = "default",
        timestamp: datetime | None = None,
    ) -> ChatMessage:
        """Record a raw chat turn (no extraction)."""
        msg = ChatMessage(
            scope=Scope(org_id=org_id, user_id=user_id),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=timestamp or datetime.now().astimezone(),
        )
        await self._store.upsert_message(msg)
        return msg

    async def extract(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
    ) -> list[Fact]:
        """Extract durable facts from a conversation. Optionally persist them.

        Requires an LLM to have been passed at `Engram.open(llm=...)`.
        """
        if self._extract is None:
            raise RuntimeError("extraction requires an LLM; pass `llm=...` to Engram.open()")
        if not messages:
            return []
        scope = messages[0].scope
        # All input messages share a session — pick the first one's session_id
        sid: str | None = messages[0].session_id if messages else None
        extracted: list[ExtractedFact] = await self._extract.extract(
            messages, session_date=session_date
        )
        out: list[Fact] = []
        if persist:
            for ef in extracted:
                route = self._memory_router.route(
                    ef.text, category=ef.category, metadata=ef.metadata
                )
                fact = Fact(
                    id=uuid4(),
                    text=ef.text,
                    scope=scope,
                    valid_from=session_date or datetime.now().astimezone(),
                    session_id=sid,
                    confidence=ef.confidence,
                    category=ef.category,
                    polarity=ef.polarity,
                    memory_system=route.memory_system,
                    memory_subtype=route.memory_subtype,
                    secondary_systems=route.secondary_systems,
                    tags=route.tags,
                    lifecycle_state=route.lifecycle_state,
                    promotion_state=route.promotion_state,
                    event_date=ef.event_date,
                    mention_date=ef.mention_date,
                    metadata={**ef.metadata, "memory_route_reason": route.reason},
                )
                await self._store.upsert_fact(fact)
                [vec] = await self._embed.embed([ef.text])
                await self._vec.add(fact.id, vec, scope)
                out.append(fact)
        else:
            # Non-persisting branch: synthesize Facts in-memory only
            for ef in extracted:
                route = self._memory_router.route(
                    ef.text, category=ef.category, metadata=ef.metadata
                )
                out.append(
                    Fact(
                        id=uuid4(),
                        text=ef.text,
                        scope=scope,
                        valid_from=session_date or datetime.now().astimezone(),
                        session_id=sid,
                        confidence=ef.confidence,
                        category=ef.category,
                        polarity=ef.polarity,
                        memory_system=route.memory_system,
                        memory_subtype=route.memory_subtype,
                        secondary_systems=route.secondary_systems,
                        tags=route.tags,
                        lifecycle_state=route.lifecycle_state,
                        promotion_state=route.promotion_state,
                        event_date=ef.event_date,
                        mention_date=ef.mention_date,
                        metadata={**ef.metadata, "memory_route_reason": route.reason},
                    )
                )
        return out

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        org_id: str = "default",
        top_k: int = 10,
        memory_systems: tuple[MemorySystem | str, ...] | None = None,
        memory_subtypes: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        include_lifecycle_states: tuple[LifecycleState | str, ...] | None = None,
        exclude_lifecycle_states: tuple[LifecycleState | str, ...] | None = None,
        infer_memory_filters: bool = False,
        allow_sensitive_recall: bool = False,
    ) -> list[ScoredFact]:
        """Hybrid (vector + keyword) retrieval, optionally reranked.

        When ``infer_memory_filters=True``, the query intent classifier
        determines both the memory filters and confidence level.

        - ``high`` confidence → strict system/subtype/tag filtering (current behavior).
        - ``low`` / ``medium`` confidence → matching facts get a score boost
          but non-matching facts are still included (soft boost).

        ``allow_sensitive_recall``: when True, facts with
        ``retrieval_policy="explicit_only"`` are included even without
        explicit subtype/tag filters. Default False (sensitive vault facts
        are excluded unless the caller provides specific filters).
        """
        typed_systems = _coerce_memory_systems(memory_systems)
        typed_subtypes = memory_subtypes
        typed_tags = tags
        intent_confidence: str | None = None
        if infer_memory_filters:
            intent = self._query_intent.classify(query)
            typed_systems = typed_systems or intent.memory_systems
            typed_subtypes = typed_subtypes or intent.memory_subtypes
            typed_tags = typed_tags or intent.tags
            intent_confidence = intent.confidence
        return await self._retrieve.search(
            query,
            Scope(org_id=org_id, user_id=user_id),
            top_k=top_k,
            memory_systems=typed_systems,
            memory_subtypes=typed_subtypes,
            tags=typed_tags,
            include_lifecycle_states=_coerce_lifecycle_states(include_lifecycle_states),
            exclude_lifecycle_states=_coerce_lifecycle_states(exclude_lifecycle_states),
            intent_confidence=intent_confidence,  # type: ignore[arg-type]
            allow_sensitive_recall=allow_sensitive_recall,
        )

    async def context(
        self,
        query: str,
        user_id: str = "default",
        org_id: str = "default",
        token_budget: int | None = None,
        chars_per_token: int = 4,
        classifier: QuestionClassifier | None = None,
        decompose: bool = False,
        role_filter: tuple[str, ...] | None = None,
        memory_systems: tuple[MemorySystem | str, ...] | None = None,
        memory_subtypes: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        include_lifecycle_states: tuple[LifecycleState | str, ...] | None = None,
        exclude_lifecycle_states: tuple[LifecycleState | str, ...] | None = None,
        infer_memory_filters: bool = False,
        allow_sensitive_recall: bool = False,
    ) -> str:
        """Assemble a context string from top-N facts that fit `token_budget`.

        Crude char-based budgeting: assumes ~4 chars/token.

        Phase 10: if `classifier` is provided AND `token_budget` is None,
        auto-pick the budget per the LongMemEval category (1.5K-7.5K from
        AgentMemory's calibration).

        Item 2 (decomposer wiring): when ``decompose=True`` AND a decomposer is
        attached AND the question looks compound (heuristic gate), split the
        question into sub-queries, retrieve top-15 per sub-query in parallel,
        and fuse via reciprocal rank fusion.

        ``role_filter``: optional tuple of role names (e.g. ``("user",)``).
        When provided, retrieved candidates are filtered to those whose
        ``fact.metadata["role"]`` is in the tuple. Filter is applied AFTER
        recall but BEFORE the token-budget loop.
        """
        if token_budget is None:
            qt = await classifier.classify(query) if classifier is not None else None
            token_budget = budget_for(qt)
        char_budget = token_budget * chars_per_token

        candidates: list[ScoredFact]
        if decompose and self._decomposer is not None and should_decompose(query):
            subqueries = await self._decomposer.decompose(query)
        else:
            subqueries = [query]

        typed_systems = memory_systems
        typed_subtypes = memory_subtypes
        typed_tags = tags
        if infer_memory_filters:
            intent = self._query_intent.classify(query)
            typed_systems = typed_systems or intent.memory_systems
            typed_subtypes = typed_subtypes or intent.memory_subtypes
            typed_tags = typed_tags or intent.tags
        include_lifecycle = include_lifecycle_states
        exclude_lifecycle = exclude_lifecycle_states

        if len(subqueries) == 1:
            candidates = await self.recall(
                subqueries[0],
                user_id=user_id,
                org_id=org_id,
                top_k=30,
                memory_systems=typed_systems,
                memory_subtypes=typed_subtypes,
                tags=typed_tags,
                include_lifecycle_states=include_lifecycle,
                exclude_lifecycle_states=exclude_lifecycle,
                infer_memory_filters=False,
                allow_sensitive_recall=allow_sensitive_recall,
            )
        else:
            per_q = await asyncio.gather(
                *[
                    self.recall(
                        sq,
                        user_id=user_id,
                        org_id=org_id,
                        top_k=15,
                        memory_systems=typed_systems,
                        memory_subtypes=typed_subtypes,
                        tags=typed_tags,
                        include_lifecycle_states=include_lifecycle,
                        exclude_lifecycle_states=exclude_lifecycle,
                        infer_memory_filters=False,
                        allow_sensitive_recall=allow_sensitive_recall,
                    )
                    for sq in subqueries
                ]
            )

            ranked_lists = [[sf.fact.id for sf in lst] for lst in per_q]
            fused_ids = reciprocal_rank_fusion(ranked_lists, k=60)
            by_id: dict[UUID, ScoredFact] = {sf.fact.id: sf for lst in per_q for sf in lst}
            candidates = [by_id[fid] for fid in fused_ids if fid in by_id]

        # Role filter — applied AFTER recall, BEFORE the token-budget loop.
        if role_filter is not None:
            candidates = [c for c in candidates if c.fact.metadata.get("role") in role_filter]

        lines: list[str] = []
        running = 0
        for sf in candidates:
            line = (
                f"[{sf.fact.event_date.date().isoformat()}] {sf.fact.text}"
                if sf.fact.event_date
                else f"- {sf.fact.text}"
            )
            if running + len(line) + 1 > char_budget:
                break
            lines.append(line)
            running += len(line) + 1
        return "\n".join(lines)


def _coerce_memory_system(value: MemorySystem | str | None) -> MemorySystem | None:
    if value is None:
        return None
    return value if isinstance(value, MemorySystem) else MemorySystem(str(value))


def _coerce_memory_system_list(
    values: tuple[MemorySystem | str, ...] | None,
) -> list[MemorySystem]:
    if values is None:
        return []
    return [_coerce_memory_system(value) or MemorySystem.WORKING for value in values]


def _coerce_memory_systems(
    values: tuple[MemorySystem | str, ...] | None,
) -> tuple[MemorySystem, ...] | None:
    if values is None:
        return None
    return tuple(_coerce_memory_system(value) or MemorySystem.WORKING for value in values)


def _coerce_lifecycle_state(value: LifecycleState | str | None) -> LifecycleState | None:
    if value is None:
        return None
    return value if isinstance(value, LifecycleState) else LifecycleState(str(value))


def _coerce_lifecycle_states(
    values: tuple[LifecycleState | str, ...] | None,
) -> tuple[LifecycleState, ...] | None:
    if values is None:
        return None
    return tuple(_coerce_lifecycle_state(value) or LifecycleState.DURABLE for value in values)


def _coerce_promotion_state(value: PromotionState | str | None) -> PromotionState | None:
    if value is None:
        return None
    return value if isinstance(value, PromotionState) else PromotionState(str(value))
