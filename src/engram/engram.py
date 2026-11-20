"""High-level Engram facade — the embedded-mode entry point.

This is what `from engram import Engram` returns. It composes the store +
vector index + embedder + (optional) extractor + reranker into a single
async context-managed object.

Each method that names a scope runs on a `ScopedEngram` for that scope
(`engram.scoped`), which holds the one implementation. For HTTP / MCP
service exposure, see `engram.server`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID

from engram.classify.base import QuestionClassifier
from engram.embedding.base import EmbeddingProvider
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import (
    ReadOnlyStoreError,
    ScopeRequiredError,
)
from engram.extract.event_extractor import EventExtractor
from engram.extract.pipeline import ExtractionPipeline
from engram.governance import (
    ErasureRecord,
    Provenance,
    Revocation,
    require_text,
)
from engram.llm.base import LLMClient
from engram.llm.tier import ModelTier
from engram.models import ChatMessage, Event, Fact, MemoryTier, Polarity
from engram.read.decomposer import QueryDecomposer
from engram.retrieve.base import Reranker, RetrievalConfig, ScoredFact
from engram.retrieve.hybrid import HybridRetriever
from engram.scope import Scope, ScopeWriteGuard
from engram.scoped import ScopedEngram, _event_extraction, _extraction
from engram.scoped import context_lines as context_lines
from engram.store.base import EngramStore
from engram.store.open import open_store
from engram.vector.hnsw import HnswVectorStore

_DEFAULT_ID = "default"


def _check_allow_default_scope(value: object) -> None:
    # A truthy string such as "false" from a config file must not widen anything.
    if not isinstance(value, bool):
        raise TypeError("allow_default_scope must be True or False")


def _check_given_ids(user_id: str | None, org_id: str | None) -> None:
    # Only a str or None. Scope decodes bytes to str, so a bytes id would pass
    # these checks and then name a scope they never saw.
    for name, value in (("org_id", org_id), ("user_id", user_id)):
        if value is None:
            continue
        if not isinstance(value, str):
            raise ScopeRequiredError(f"{name} must be a string")
        if not value.strip():
            raise ScopeRequiredError(f"{name} must not be empty or blank")


def _require_both_named(org_id: object, user_id: object) -> None:
    # A handle always names its scope, with or without allow_default_scope.
    for name, value in (("org_id", org_id), ("user_id", user_id)):
        if value is None:
            raise ScopeRequiredError(
                f'{name} is required: a handle names its scope, and "default" names the '
                "default one on an Engram opened with allow_default_scope=True"
            )


class Engram:
    """The main embedded-mode Engram client.

    Open with `Engram.open(path)` and use as an async context manager:

        async with await Engram.open("./engram.db") as memory:
            await memory.record("I prefer espresso.", org_id="acme", user_id="alice")
            facts = await memory.recall("coffee preference", org_id="acme", user_id="alice")

    Every method that reads or writes memory names its scope with ``org_id``
    and ``user_id``. A missing id, or the id ``"default"``, raises
    ScopeRequiredError unless the Engram was opened with
    ``allow_default_scope=True``. An id that is not a string, or is empty or
    blank, always raises.

    To hand code one scope and nothing else, give it ``memory.scoped(org_id=...,
    user_id=...)``: a `ScopedEngram` whose methods take no scope.

    It works with any store that meets `engram.store.EngramStore`; pass one
    with `Engram.open(store=...)`.
    """

    def __init__(
        self,
        store: EngramStore,
        vector_store: HnswVectorStore,
        embedder: EmbeddingProvider,
        retriever: HybridRetriever,
        extraction: ExtractionPipeline | None = None,
        event_extractor: EventExtractor | None = None,
        tier: ModelTier | None = None,
        *,
        allow_default_scope: bool = False,
    ) -> None:
        _check_allow_default_scope(allow_default_scope)
        self._allow_default_scope = allow_default_scope
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
        *,
        store: EngramStore | None = None,
        read_only: bool = False,
        allow_default_scope: bool = False,
    ) -> Engram:
        """Open an Engram instance on a store, with an in-memory HNSW index.

        Storage:
        - path: a SQLite file, or ":memory:" (the default).
        - store: an open store to use instead of ``path``: any object that
          meets `engram.store.EngramStore`. The Engram owns it from then on,
          and `close()` closes it. Passing a ``path`` as well is a ValueError.
        - read_only: open the store for reads only. Every write method then
          raises `ReadOnlyStoreError` before any embedder or LLM call. A
          ``store`` passed with ``read_only=True`` must itself be read-only.

        Scopes:
        - allow_default_scope: False (the default) makes every call name its
          scope: a missing ``user_id`` or ``org_id``, or the id ``"default"``,
          raises ScopeRequiredError before anything is read or written. True
          restores 0.2 behaviour: a missing id means ``"default"``. Either
          way an empty or blank id raises. Must be True or False (TypeError).

        Defaults:
        - embedder: SyntheticEmbedding(dim=384) — deterministic, offline-safe.
          Pass an OllamaEmbedding or OpenAIEmbedding for real semantics.
        - llm: None — extraction unavailable until provided.
        - reranker: None — set to a CrossEncoderReranker for higher precision.
        - tier: None — pass a ``ModelTier`` to split reader (answer generation)
          from utility (verifier, decomposer, ReAct brain). When ``tier`` is set
          and ``llm`` is not, ``tier.utility`` is used for extraction/event work.
        """
        _check_allow_default_scope(allow_default_scope)
        if store is not None:
            if str(path) != ":memory:":
                raise ValueError("pass a path or a store, not both")
            if read_only and not store.read_only:
                raise ValueError(
                    "read_only=True needs a store opened read-only; open the store "
                    "with read_only=True, or leave read_only out"
                )
        else:
            store = await open_store(path, read_only=read_only)
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
        return cls(
            store,
            vec,
            embedder,
            retriever,
            extraction,
            events,
            tier=tier,
            allow_default_scope=allow_default_scope,
        )

    @property
    def read_only(self) -> bool:
        """True when the store is read-only: every write method raises ReadOnlyStoreError."""
        return self._store.read_only

    @property
    def allow_default_scope(self) -> bool:
        """True when a missing id, or the id "default", names the default scope."""
        return self._allow_default_scope

    def _resolve_scope(self, user_id: str | None, org_id: str | None) -> Scope:
        """The scope a call names. Every handle, so every root method, resolves it here.

        ``None`` means not given. Not given, or the literal ``"default"``, is
        refused with ScopeRequiredError unless this Engram was opened with
        ``allow_default_scope=True``, and then not given means ``"default"``.
        An id that is not a string, or is empty or blank, is refused always.
        Any other id that Scope refuses raises ValueError.
        """
        _check_given_ids(user_id, org_id)
        if not self._allow_default_scope:
            for name, value in (("org_id", org_id), ("user_id", user_id)):
                if value is None:
                    raise ScopeRequiredError(
                        f"{name} is required: name the scope, or open the Engram with "
                        "allow_default_scope=True to use the default one"
                    )
        scope = Scope(
            org_id=_DEFAULT_ID if org_id is None else org_id,
            user_id=_DEFAULT_ID if user_id is None else user_id,
        )
        if not self._allow_default_scope:
            # Read from the Scope, not the arguments: a str subclass can
            # compare unequal to "default" and still be stored as it.
            for name, value in (("org_id", scope.org_id), ("user_id", scope.user_id)):
                if value == _DEFAULT_ID:
                    raise ScopeRequiredError(
                        f'{name} "default" names the default scope, which needs the Engram '
                        "opened with allow_default_scope=True"
                    )
        return scope

    def _require_writable(self) -> None:
        # First thing in every write method: before validation, any store
        # call, and any embedder or LLM call.
        if self._store.read_only:
            raise ReadOnlyStoreError("this Engram was opened read-only; writes are refused")

    async def close(self) -> None:
        """Close the vector index and the store, a store passed to `open` included."""
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

    # ── Scoped handles ──────────────────────────────────────────────────

    def scoped(self, *, org_id: str, user_id: str) -> ScopedEngram:
        """A handle on one scope. Its methods take no scope and cannot reach another.

        The ids are checked once, here, by the rules every root method
        follows: an id that is not a string, or is empty or blank, raises
        ScopeRequiredError, and so does ``"default"`` unless this Engram was
        opened with ``allow_default_scope=True``. Any other id that Scope
        refuses raises ValueError. Both ids must be given: ``None`` raises
        ScopeRequiredError even when the default scope is allowed.
        """
        _require_both_named(org_id, user_id)
        return ScopedEngram(self, org_id=org_id, user_id=user_id)

    def _handle(self, user_id: str | None, org_id: str | None) -> ScopedEngram:
        """The handle a root method runs on. ``None`` means not given, as in `_resolve_scope`."""
        return ScopedEngram(self, org_id=org_id, user_id=user_id)

    def _governed(self, fact: Fact | UUID, user_id: str | None, org_id: str | None) -> ScopedEngram:
        """Handle for a governance call: on the fact's own scope, or the one passed with an id.

        The rules of `_resolve_scope` hold either way: an id that is not a
        string, or is empty or blank, is refused, and a fact in the default
        scope needs allow_default_scope. Ids given with a Fact must match it.
        """
        if isinstance(fact, Fact):
            _check_given_ids(user_id, org_id)
            if (user_id is not None and user_id != fact.scope.user_id) or (
                org_id is not None and org_id != fact.scope.org_id
            ):
                raise ValueError("user_id/org_id do not match the fact's scope")
            return self._handle(fact.scope.user_id, fact.scope.org_id)
        return self._handle(user_id, org_id)

    def _message_handle(self, messages: list[ChatMessage]) -> ScopedEngram:
        """Handle on the first message's scope; it refuses messages from any other."""
        first = messages[0].scope
        return self._handle(first.user_id, first.org_id)

    # ── Public API ──────────────────────────────────────────────────────
    #
    # Write methods check read-only first, so a read-only Engram says so
    # before the scope is checked. Each then runs the same method on a handle.

    async def extract_events(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
        *,
        guard: ScopeWriteGuard | None = None,
    ) -> list[Event]:
        """Phase 11: extract SVO events from a chat segment via LLM.

        Runs `ScopedEngram.extract_events` in the first message's scope. A
        message from any other scope, or a ``guard`` for another scope, raises
        ScopeError before the LLM call. Persists the events by default.
        Requires an LLM to have been passed at `Engram.open(llm=...)`. With
        ``persist=False`` nothing is stored, so it also works read-only.
        """
        if persist:
            self._require_writable()
        if not messages:
            _event_extraction(self, persist)
            return []
        return await self._message_handle(messages).extract_events(
            messages, session_date=session_date, persist=persist, guard=guard
        )

    async def search_events(
        self,
        query: str = "",
        user_id: str | None = None,
        org_id: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        limit: int = 10,
    ) -> list[Event]:
        """Phase 11: query the event calendar by FTS + optional time window."""
        return await self._handle(user_id, org_id).search_events(
            query, time_start=time_start, time_end=time_end, limit=limit
        )

    async def supersede(
        self,
        old_fact_id: UUID,
        new_fact_id: UUID,
        user_id: str | None = None,
        org_id: str | None = None,
        actor: str | None = None,
        reason: str | None = None,
    ) -> Revocation:
        """Revoke ``old_fact_id`` and link ``new_fact_id`` as its replacement.

        See `ScopedEngram.supersede`.
        """
        self._require_writable()
        return await self._handle(user_id, org_id).supersede(
            old_fact_id, new_fact_id, actor=actor, reason=reason
        )

    async def revoke(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> Revocation:
        """Tombstone a fact so no read path returns it again.

        Pass a ``Fact`` to use its scope, or an id with ``user_id``/``org_id``.
        See `ScopedEngram.revoke`.
        """
        self._require_writable()
        return await self._governed(fact, user_id, org_id).revoke(fact, reason=reason, actor=actor)

    async def reinstate(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> list[Revocation]:
        """Lift the tombstones placed on this fact. The only way to un-revoke.

        Pass a ``Fact`` to use its scope, or an id with ``user_id``/``org_id``.
        See `ScopedEngram.reinstate`.
        """
        self._require_writable()
        return await self._governed(fact, user_id, org_id).reinstate(
            fact, reason=reason, actor=actor
        )

    async def erase(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        user_id: str | None = None,
        org_id: str | None = None,
        block_reinsertion: bool = True,
    ) -> ErasureRecord:
        """Hard-delete a fact, its copies and quotes, what was derived from it, and their vectors.

        Pass a ``Fact`` to use its scope, or an id with ``user_id``/``org_id``.
        See `ScopedEngram.erase`.
        """
        self._require_writable()
        return await self._governed(fact, user_id, org_id).erase(
            fact, reason=reason, actor=actor, block_reinsertion=block_reinsertion
        )

    async def forget(self, fact_id: UUID, *, reason: str, actor: str) -> bool:
        """Revoke a fact named by id alone, for the v1 REST contract.

        Soft and reversible, like the Rust server's invalidate. Returns False
        if no fact has this id, or if it was erased before the revoke landed.
        Use `erase` to hard-delete.

        Not an authorization check: it resolves the owner of any fact id across
        every scope and revokes it there. Use it only where the caller is
        trusted for every scope, as on a one-token server.
        """
        self._require_writable()
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        scope = await self._store.find_fact_scope(fact_id)
        if scope is None:
            return False
        return await self._handle(scope.user_id, scope.org_id).forget(
            fact_id, reason=reason, actor=actor
        )

    async def erase_user(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        reason: str,
        actor: str,
    ) -> int:
        """Hard-delete everything held for one (org_id, user_id) scope.

        The right-to-erasure path. Returns the number of facts erased. See
        `ScopedEngram.erase_all` for what goes, what stays as the audit trail,
        and what to do when it raises.
        """
        self._require_writable()
        return await self._handle(user_id, org_id).erase_all(reason=reason, actor=actor)

    async def list_conversations(
        self, *, user_id: str | None = None, org_id: str | None = None
    ) -> list[str]:
        """Session ids with a visible transcript turn, by first turn."""
        return await self._handle(user_id, org_id).list_conversations()

    async def delete_conversation(
        self, session_id: str, *, user_id: str | None = None, org_id: str | None = None
    ) -> int:
        """Hard-delete one session's transcript. Facts extracted from it stay.

        See `ScopedEngram.delete_conversation`.
        """
        self._require_writable()
        return await self._handle(user_id, org_id).delete_conversation(session_id)

    async def fact_counts(self) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) across every tenant.

        Not scoped to a caller: like `forget`, it assumes the caller is trusted
        for every scope, as on a one-token server. `ScopedEngram.fact_counts`
        counts one scope.
        """
        return await self._store.fact_counts(None)

    async def list_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        last_n: int | None = None,
    ) -> list[ChatMessage]:
        """A session's visible transcript turns, oldest first. See `ScopedEngram.list_messages`."""
        return await self._handle(user_id, org_id).list_messages(session_id, last_n=last_n)

    async def explain(
        self, fact_id: UUID, user_id: str | None = None, org_id: str | None = None
    ) -> Provenance | None:
        """Provenance and revocation status for a fact, visible or not. Never its text."""
        return await self._handle(user_id, org_id).explain(fact_id)

    async def revocations(
        self,
        user_id: str | None = None,
        org_id: str | None = None,
        include_reinstated: bool = False,
    ) -> list[Revocation]:
        return await self._handle(user_id, org_id).revocations(
            include_reinstated=include_reinstated
        )

    async def erasures(
        self, user_id: str | None = None, org_id: str | None = None
    ) -> list[ErasureRecord]:
        return await self._handle(user_id, org_id).erasures()

    async def record(
        self,
        text: str,
        user_id: str | None = None,
        org_id: str | None = None,
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

        See `ScopedEngram.record` for session ids, roles and the governance
        arguments (``actor``, ``derived_from``, ``source_message_id``,
        ``reject_revoked``).
        """
        self._require_writable()
        return await self._handle(user_id, org_id).record(
            text,
            session_id=session_id,
            category=category,
            confidence=confidence,
            polarity=polarity,
            tier=tier,
            valid_from=valid_from,
            event_date=event_date,
            metadata=metadata,
            role=role,
            actor=actor,
            derived_from=derived_from,
            source_message_id=source_message_id,
            reject_revoked=reject_revoked,
        )

    async def record_message(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
        user_id: str | None = None,
        org_id: str | None = None,
        timestamp: datetime | None = None,
        guard: ScopeWriteGuard | None = None,
    ) -> ChatMessage:
        """Record a raw chat turn (no extraction). See `ScopedEngram.record_message`.

        ``guard`` comes from ``scoped(org_id=..., user_id=...).guard()`` for the
        same scope; with one, a turn written after that scope was erased is
        refused with ScopeErasedError.
        """
        self._require_writable()
        return await self._handle(user_id, org_id).record_message(
            content,
            role,
            session_id,
            timestamp=timestamp,
            guard=guard,
        )

    async def extract(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
        actor: str | None = None,
        guard: ScopeWriteGuard | None = None,
    ) -> list[Fact]:
        """Extract durable facts from a conversation. Optionally persist them.

        Runs `ScopedEngram.extract` in the first message's scope. A message
        from any other scope, or a ``guard`` for another scope, raises
        ScopeError before the LLM call.
        """
        if persist:
            self._require_writable()
        if not messages:
            _extraction(self, persist)
            return []
        return await self._message_handle(messages).extract(
            messages,
            session_date=session_date,
            persist=persist,
            actor=actor,
            guard=guard,
        )

    async def recall(
        self,
        query: str,
        user_id: str | None = None,
        org_id: str | None = None,
        top_k: int = 10,
    ) -> list[ScoredFact]:
        """Hybrid (vector + keyword) retrieval, optionally reranked."""
        return await self._handle(user_id, org_id).recall(query, top_k=top_k)

    async def context_entries(
        self,
        query: str,
        user_id: str | None = None,
        org_id: str | None = None,
        decompose: bool = False,
        role_filter: tuple[str, ...] | None = None,
    ) -> list[ScoredFact]:
        """The ranked memories :meth:`context` renders, each with its provenance.

        See `ScopedEngram.context_entries`.
        """
        return await self._handle(user_id, org_id).context_entries(
            query, decompose=decompose, role_filter=role_filter
        )

    async def context(
        self,
        query: str,
        user_id: str | None = None,
        org_id: str | None = None,
        token_budget: int | None = None,
        chars_per_token: int = 4,
        classifier: QuestionClassifier | None = None,
        decompose: bool = False,
        role_filter: tuple[str, ...] | None = None,
    ) -> str:
        """Assemble a context string from top-N facts that fit `token_budget`.

        See `ScopedEngram.context` for budgets, the decomposer and ``role_filter``.
        """
        return await self._handle(user_id, org_id).context(
            query,
            token_budget=token_budget,
            chars_per_token=chars_per_token,
            classifier=classifier,
            decompose=decompose,
            role_filter=role_filter,
        )
