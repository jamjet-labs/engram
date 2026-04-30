"""High-level Engram facade — the embedded-mode entry point.

This is what `from engram import Engram` returns. It composes the store +
vector index + embedder + (optional) extractor + reranker into a single
async context-managed object.

For HTTP / MCP service exposure, see `engram.server`.
"""

from __future__ import annotations

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
from engram.models import ChatMessage, Event, ExtractedFact, Fact, MemoryTier, Polarity
from engram.retrieve.base import Reranker, RetrievalConfig, ScoredFact
from engram.retrieve.hybrid import HybridRetriever
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
    ) -> Fact:
        """Record a single fact (no LLM extraction). Embeds + stores in one call.

        Pass `session_id` to enable Phase 9 two-stage retrieval — facts in the
        same session can then be retrieved together via session-first ranking.
        """
        scope = Scope(org_id=org_id, user_id=user_id)
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
            metadata=metadata or {},
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
                await self._store.upsert_fact(fact)
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

    async def context(
        self,
        query: str,
        user_id: str = "default",
        org_id: str = "default",
        token_budget: int | None = None,
        chars_per_token: int = 4,
        classifier: QuestionClassifier | None = None,
    ) -> str:
        """Assemble a context string from top-N facts that fit `token_budget`.

        Crude char-based budgeting: assumes ~4 chars/token. Phase 6 will replace
        with a proper tokenizer.

        Phase 10: if `classifier` is provided AND `token_budget` is None,
        auto-pick the budget per the LongMemEval category (1.5K-7.5K from
        AgentMemory's calibration).
        """
        if token_budget is None:
            qt = await classifier.classify(query) if classifier is not None else None
            token_budget = budget_for(qt)
        char_budget = token_budget * chars_per_token
        candidates = await self.recall(query, user_id=user_id, org_id=org_id, top_k=30)
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
