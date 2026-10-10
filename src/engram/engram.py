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
from engram.governance import (
    ErasureRecord,
    Provenance,
    Revocation,
    RevokedContentError,
    content_hash,
    require_text,
)
from engram.llm.base import LLMClient
from engram.llm.tier import ModelTier
from engram.models import ChatMessage, Event, ExtractedFact, Fact, MemoryTier, Polarity
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
        actor: str | None = None,
        reason: str | None = None,
    ) -> Revocation:
        """Revoke ``old_fact_id`` and link ``new_fact_id`` as its replacement.

        The old fact (and any copy of its text in this scope) is withheld from
        every read path from now on, whatever ``RetrievalConfig`` says. The
        replacement must be visible and must say something different: a
        replacement that normalizes to the same text would be hidden by the
        tombstone it creates.
        """
        scope = Scope(org_id=org_id, user_id=user_id)
        old = await self._store.explain(old_fact_id, scope)
        if old is None:
            raise ValueError(f"old fact {old_fact_id} not found in scope")
        new = await self._store.get_fact(new_fact_id, scope)
        if new is None:
            if await self._store.explain(new_fact_id, scope) is not None:
                raise ValueError(f"new fact {new_fact_id} is revoked and cannot be a replacement")
            raise ValueError(f"new fact {new_fact_id} not found in scope")
        if await self._store.content_hash_of(old_fact_id, scope) == content_hash(new.text):
            raise ValueError("replacement has the same normalized text as the fact it replaces")
        rev = await self._store.revoke(
            old_fact_id,
            scope,
            reason=reason or f"superseded by {new_fact_id}",
            actor=actor or "unspecified",
            replaced_by=new_fact_id,
        )
        await self._vec.delete(old_fact_id, scope)
        return rev

    async def revoke(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        user_id: str = "default",
        org_id: str = "default",
    ) -> Revocation:
        """Tombstone a fact so no read path returns it again.

        Also withholds every fact in this scope with the same normalized text
        and every fact recorded with ``derived_from`` pointing at it. The fact
        stays in the database for audit; use :meth:`erase` to delete it.
        Idempotent: revoking a revoked fact returns the existing tombstone.
        """
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        scope = Scope(org_id=org_id, user_id=user_id)
        fact_id = fact.id if isinstance(fact, Fact) else fact
        rev = await self._store.revoke(fact_id, scope, reason=reason, actor=actor)
        # Not needed for correctness (the store withholds it), but keeps the
        # revoked vector from taking a slot in the candidate pool.
        await self._vec.delete(fact_id, scope)
        return rev

    async def reinstate(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        user_id: str = "default",
        org_id: str = "default",
    ) -> list[Revocation]:
        """Lift the tombstones hiding a fact. The only way to un-revoke.

        The tombstones stay in the audit trail with who lifted them and why.
        """
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        scope = Scope(org_id=org_id, user_id=user_id)
        fact_id = fact.id if isinstance(fact, Fact) else fact
        lifted, visible = await self._store.reinstate(fact_id, scope, reason=reason, actor=actor)
        if visible:
            vecs = await self._embed.embed([f.text for f in visible])
            for f, v in zip(visible, vecs, strict=True):
                await self._vec.add(f.id, v, scope)
        return lifted

    async def erase(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        user_id: str = "default",
        org_id: str = "default",
    ) -> ErasureRecord:
        """Hard-delete a fact, its copies, what was derived from it, and their vectors.

        Returns a content-free :class:`ErasureRecord`. ``reason`` must not
        quote the erased text. See the README section on erasure for what the
        backend can and cannot vouch for.
        """
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        scope = Scope(org_id=org_id, user_id=user_id)
        fact_id = fact.id if isinstance(fact, Fact) else fact
        record = await self._store.erase(fact_id, scope, reason=reason, actor=actor)
        removed = await self._vec.purge(record.fact_ids, scope)
        return await self._store.finalize_erasure(
            record, {"vectors": removed}, {"vector_index_rebuilt": True}
        )

    async def explain(
        self, fact_id: UUID, user_id: str = "default", org_id: str = "default"
    ) -> Provenance | None:
        """Provenance and revocation status for a fact, visible or not. Never its text."""
        return await self._store.explain(fact_id, Scope(org_id=org_id, user_id=user_id))

    async def revocations(
        self, user_id: str = "default", org_id: str = "default", include_reinstated: bool = False
    ) -> list[Revocation]:
        return await self._store.list_revocations(
            Scope(org_id=org_id, user_id=user_id), include_reinstated=include_reinstated
        )

    async def erasures(
        self, user_id: str = "default", org_id: str = "default"
    ) -> list[ErasureRecord]:
        return await self._store.list_erasures(Scope(org_id=org_id, user_id=user_id))

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
        actor: str | None = None,
        derived_from: list[UUID] | None = None,
        source_message_id: UUID | None = None,
        reject_revoked: bool = False,
    ) -> Fact:
        """Record a single fact (no LLM extraction). Embeds + stores in one call.

        Pass `session_id` to enable Phase 9 two-stage retrieval — facts in the
        same session can then be retrieved together via session-first ranking.

        Pass `role` (e.g., "user", "assistant") to tag the fact's origin role.
        Stored in ``metadata["role"]``; queryable via ``Engram.context(role_filter=...)``.

        Governance:

        - ``actor`` is stored as the writer in the fact's provenance.
        - ``derived_from`` names facts this one was concluded from. If any of
          them is revoked (now or later) this fact is withheld too.
        - ``source_message_id`` links the transcript turn the fact came from.
        - Text that matches revoked content is stored but withheld (a write
          cannot un-revoke). ``reject_revoked=True`` raises instead.
        """
        scope = Scope(org_id=org_id, user_id=user_id)
        if reject_revoked and await self._store.is_content_revoked(text, scope):
            raise RevokedContentError("text matches revoked content in this scope")
        md = dict(metadata or {})
        if role is not None:
            md["role"] = role
        fact = Fact(
            text=text,
            scope=scope,
            valid_from=valid_from or datetime.now().astimezone(),
            session_id=session_id,
            category=category,
            confidence=confidence,
            polarity=polarity,
            tier=tier,
            event_date=event_date,
            metadata=md,
            source_message_id=source_message_id,
        )
        await self._store.upsert_fact(
            fact, writer=actor, source="record", derived_from=derived_from
        )
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
        actor: str | None = None,
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
                fact = Fact(
                    id=uuid4(),
                    text=ef.text,
                    scope=scope,
                    valid_from=session_date or datetime.now().astimezone(),
                    session_id=sid,
                    confidence=ef.confidence,
                    category=ef.category,
                    polarity=ef.polarity,
                    event_date=ef.event_date,
                    mention_date=ef.mention_date,
                )
                await self._store.upsert_fact(fact, writer=actor, source="extract")
                [vec] = await self._embed.embed([ef.text])
                await self._vec.add(fact.id, vec, scope)
                out.append(fact)
        else:
            # Non-persisting branch: synthesize Facts in-memory only
            for ef in extracted:
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
                        event_date=ef.event_date,
                        mention_date=ef.mention_date,
                    )
                )
        return out

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        org_id: str = "default",
        top_k: int = 10,
    ) -> list[ScoredFact]:
        """Hybrid (vector + keyword) retrieval, optionally reranked."""
        return await self._retrieve.search(
            query, Scope(org_id=org_id, user_id=user_id), top_k=top_k
        )

    async def context_entries(
        self,
        query: str,
        user_id: str = "default",
        org_id: str = "default",
        decompose: bool = False,
        role_filter: tuple[str, ...] | None = None,
    ) -> list[ScoredFact]:
        """The ranked memories :meth:`context` renders, each with its provenance.

        Use this when the caller needs to show why a memory went into a prompt.
        """
        candidates: list[ScoredFact]
        if decompose and self._decomposer is not None and should_decompose(query):
            subqueries = await self._decomposer.decompose(query)
        else:
            subqueries = [query]

        if len(subqueries) == 1:
            candidates = await self.recall(subqueries[0], user_id=user_id, org_id=org_id, top_k=30)
        else:
            per_q = await asyncio.gather(
                *[self.recall(sq, user_id=user_id, org_id=org_id, top_k=15) for sq in subqueries]
            )
            ranked_lists = [[sf.fact.id for sf in lst] for lst in per_q]
            fused_ids = reciprocal_rank_fusion(ranked_lists, k=60)
            by_id: dict[UUID, ScoredFact] = {sf.fact.id: sf for lst in per_q for sf in lst}
            candidates = [by_id[fid] for fid in fused_ids if fid in by_id]

        # Role filter — applied AFTER recall, BEFORE the token-budget loop.
        if role_filter is not None:
            candidates = [c for c in candidates if c.fact.metadata.get("role") in role_filter]
        return candidates

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

        candidates = await self.context_entries(
            query, user_id=user_id, org_id=org_id, decompose=decompose, role_filter=role_filter
        )

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
