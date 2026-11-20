"""High-level Engram facade — the embedded-mode entry point.

This is what `from engram import Engram` returns. It composes the store +
vector index + embedder + (optional) extractor + reranker into a single
async context-managed object.

For HTTP / MCP service exposure, see `engram.server`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

from engram.classify.base import QuestionClassifier, budget_for
from engram.embedding.base import EmbeddingProvider
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import (
    NotFoundError,
    ReadOnlyStoreError,
    ScopeErasedError,
    ScopeRequiredError,
)
from engram.extract.event_extractor import EventExtractor
from engram.extract.pipeline import ExtractionPipeline
from engram.governance import (
    ErasureRecord,
    Provenance,
    Revocation,
    RevokedContentError,
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
from engram.store.base import EngramStore
from engram.store.open import open_store
from engram.validate import (
    MAX_FACT_TEXT_BYTES,
    MAX_MESSAGE_BYTES,
    check_json,
    check_text,
)
from engram.vector.hnsw import HnswVectorStore

logger = logging.getLogger(__name__)

_DEFAULT_ID = "default"


def _check_allow_default_scope(value: object) -> None:
    # A truthy string such as "false" from a config file must not widen anything.
    if not isinstance(value, bool):
        raise TypeError("allow_default_scope must be True or False")


def _refuse_blank_ids(user_id: str | None, org_id: str | None) -> None:
    for name, value in (("org_id", org_id), ("user_id", user_id)):
        if isinstance(value, str) and not value.strip():
            raise ScopeRequiredError(f"{name} must not be empty or blank")


class Engram:
    """The main embedded-mode Engram client.

    Open with `Engram.open(path)` and use as an async context manager:

        async with await Engram.open("./engram.db") as memory:
            await memory.record("I prefer espresso.", org_id="acme", user_id="alice")
            facts = await memory.recall("coffee preference", org_id="acme", user_id="alice")

    Every method that reads or writes memory names its scope with ``org_id``
    and ``user_id``. A missing id, or the id ``"default"``, raises
    ScopeRequiredError unless the Engram was opened with
    ``allow_default_scope=True``. An empty or blank id always raises.

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
        """The scope a call names. Every root method resolves its scope here first.

        ``None`` means not given. Not given, or the literal ``"default"``, is
        refused with ScopeRequiredError unless this Engram was opened with
        ``allow_default_scope=True``, and then not given means ``"default"``.
        An empty or blank id is refused always. Any other id that Scope
        refuses raises ValueError.
        """
        _refuse_blank_ids(user_id, org_id)
        if not self._allow_default_scope:
            for name, value in (("org_id", org_id), ("user_id", user_id)):
                if value is None:
                    raise ScopeRequiredError(
                        f"{name} is required: name the scope, or open the Engram with "
                        "allow_default_scope=True to use the default one"
                    )
                if value == _DEFAULT_ID:
                    raise ScopeRequiredError(
                        f'{name} "default" names the default scope, which needs the Engram '
                        "opened with allow_default_scope=True"
                    )
        return Scope(
            org_id=_DEFAULT_ID if org_id is None else org_id,
            user_id=_DEFAULT_ID if user_id is None else user_id,
        )

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

    # ── Public API ──────────────────────────────────────────────────────

    async def extract_events(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
    ) -> list[Event]:
        """Phase 11: extract SVO events from a chat segment via LLM.

        Persists the events into the event calendar by default. Requires an
        LLM to have been passed at `Engram.open(llm=...)`. With
        ``persist=False`` nothing is stored, so it also works read-only.
        """
        if persist:
            self._require_writable()
        if self._events is None:
            raise RuntimeError("event extraction requires an LLM; pass `llm=...` to Engram.open()")
        if not messages:
            return []
        self._message_scope(messages)
        events = await self._events.extract(messages, session_date=session_date)
        if persist:
            for ev in events:
                await self._store.upsert_event(ev)
        return events

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
        return await self._store.search_events(
            query,
            self._resolve_scope(user_id, org_id),
            time_start=time_start,
            time_end=time_end,
            limit=limit,
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

        The old fact (and any copy or quote of its text in this scope) is
        withheld from every read path from now on, whatever
        ``RetrievalConfig`` says. The replacement must be visible and must not
        repeat the old text, or it would be hidden by the tombstone it creates.
        If the old fact is already revoked, the replacement is attached to the
        existing tombstone; a different replacement than the recorded one is
        refused.
        """
        self._require_writable()
        scope = self._resolve_scope(user_id, org_id)
        try:
            rev = await self._store.revoke(
                old_fact_id,
                scope,
                reason=reason or f"superseded by {new_fact_id}",
                actor=actor or "unspecified",
                replaced_by=new_fact_id,
            )
        except NotFoundError as e:
            raise ValueError(str(e)) from e
        await self._drop_hidden_vectors(scope)
        return rev

    async def _drop_hidden_vectors(self, scope: Scope) -> None:
        # Not needed for correctness (the store withholds hidden facts), but
        # keeps hidden vectors from filling the fixed candidate pool.
        for fid in await self._store.hidden_fact_ids(scope):
            await self._vec.delete(fid, scope)

    def _governed_scope(self, fact: Fact | UUID, user_id: str | None, org_id: str | None) -> Scope:
        """Scope for a governance call: the fact's own, or the one passed with an id.

        The rules of `_resolve_scope` hold either way: an empty or blank id is
        refused, and a fact in the default scope needs allow_default_scope.
        """
        if isinstance(fact, Fact):
            _refuse_blank_ids(user_id, org_id)
            if (user_id is not None and user_id != fact.scope.user_id) or (
                org_id is not None and org_id != fact.scope.org_id
            ):
                raise ValueError("user_id/org_id do not match the fact's scope")
            return self._resolve_scope(fact.scope.user_id, fact.scope.org_id)
        return self._resolve_scope(user_id, org_id)

    def _message_scope(self, messages: list[ChatMessage]) -> Scope:
        """The scope extraction writes to: the first message's, by the same rules."""
        first = messages[0].scope
        return self._resolve_scope(first.user_id, first.org_id)

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

        Also withholds every fact, transcript turn and event in this scope
        whose text equals or quotes it, and every fact recorded with
        ``derived_from`` pointing at it. The fact stays in the database for
        audit; use :meth:`erase` to delete it. Idempotent. Pass a ``Fact`` to
        use its scope, or an id with ``user_id``/``org_id``.
        """
        self._require_writable()
        scope = self._governed_scope(fact, user_id, org_id)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        fact_id = fact.id if isinstance(fact, Fact) else fact
        rev = await self._store.revoke(fact_id, scope, reason=reason, actor=actor)
        await self._drop_hidden_vectors(scope)
        return rev

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

        Works on the revoked fact itself, not on a copy hidden by its
        tombstone. The tombstones stay in the audit trail with who lifted
        them and why. The fact can stay hidden if another tombstone also
        covers it; check :meth:`explain`.
        """
        self._require_writable()
        scope = self._governed_scope(fact, user_id, org_id)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
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
        user_id: str | None = None,
        org_id: str | None = None,
        block_reinsertion: bool = True,
    ) -> ErasureRecord:
        """Hard-delete a fact, its copies and quotes, what was derived from it, and their vectors.

        Returns a content-free :class:`ErasureRecord`. ``reason`` and ``actor``
        must not quote the erased text. If the fact was revoked, its tombstone
        keeps the content hash so the text stays blocked; pass
        ``block_reinsertion=False`` to drop it too. See the README section on
        erasure for what the backend can and cannot vouch for.
        """
        self._require_writable()
        scope = self._governed_scope(fact, user_id, org_id)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        fact_id = fact.id if isinstance(fact, Fact) else fact
        record = await self._store.erase(
            fact_id, scope, reason=reason, actor=actor, block_reinsertion=block_reinsertion
        )
        removed = await self._vec.purge(record.fact_ids, scope)
        return await self._store.finalize_erasure(
            record, {"vectors": removed}, {"vector_index_rebuilt": True}
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
        try:
            await self.revoke(
                fact_id, reason=reason, actor=actor, user_id=scope.user_id, org_id=scope.org_id
            )
        except NotFoundError:
            return False  # erased between the lookup and the revoke
        return True

    async def erase_user(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        reason: str,
        actor: str,
    ) -> int:
        """Hard-delete everything held for one (org_id, user_id) scope.

        Every fact goes through `erase` with `block_reinsertion=False`, so its
        copies, quotes, derived facts, events, cited transcript turns and
        vectors go with it, and each leaves a content-free erasure record.
        Then `erase_scope_residue` deletes the scope's remaining turns, events,
        entities and relationships, clears the content hash and normalized
        text from every tombstone in the scope (a hash kept by an earlier
        `erase(block_reinsertion=True)` included), and writes one more erasure
        record with the counts and no fact ids. This is the right-to-erasure
        path, and a short text's hash can be confirmed by guessing.
        Tombstones and erasure records stay as the content-free audit trail.

        ``reason`` and ``actor`` are refused with ValueError, before anything
        is deleted, when blank or when they quote any fact text in the scope,
        hidden facts included (the rule `erase` applies). Only a fact written
        after that check can make a later `erase` refuse part-way.

        Not one transaction. ScopeResidueRefusedError (a StoreError) from the
        last step means a fact was written to the scope while this ran; the
        facts erased so far stay erased, and calling again is safe and
        finishes. Any other StoreError is a storage fault: what was erased
        stays erased, but calling again may fail the same way. Cost grows
        faster than linearly with the number of facts in the scope, so one call
        on a large scope may run long; each call erases more, so repeating it
        finishes the job. Returns the number of facts erased.
        """
        self._require_writable()
        scope = self._resolve_scope(user_id, org_id)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        for label, value in (("reason", reason), ("actor", actor)):
            if await self._store.fact_text_quoted_by(value, scope):
                raise ValueError(f"erasure {label} must not quote the content being erased")
        erased = 0
        for fact_id in await self._store.list_fact_ids(scope):
            try:
                record = await self.erase(
                    fact_id,
                    reason=reason,
                    actor=actor,
                    user_id=scope.user_id,
                    org_id=scope.org_id,
                    block_reinsertion=False,
                )
            except NotFoundError:
                continue  # already removed as a copy or derivation of an earlier one
            erased += len(record.fact_ids)
        residue = await self._store.erase_scope_residue(scope, reason=reason, actor=actor)
        await self._store.finalize_erasure(residue, {}, {})
        return erased

    async def list_conversations(
        self, *, user_id: str | None = None, org_id: str | None = None
    ) -> list[str]:
        """Session ids with a visible transcript turn, by first turn."""
        return await self._store.list_session_ids(self._resolve_scope(user_id, org_id))

    async def delete_conversation(
        self, session_id: str, *, user_id: str | None = None, org_id: str | None = None
    ) -> int:
        """Hard-delete one session's transcript. Facts extracted from it stay.

        Returns the number of turns removed. The facts lose their
        `source_message_id` link. If one of those facts was revoked, rule (c)
        of `live_messages` withholds every later turn in a session reusing
        that id. This is accepted fail-closed behavior.
        """
        self._require_writable()
        return await self._store.delete_session_messages(
            session_id, self._resolve_scope(user_id, org_id)
        )

    async def fact_counts(self) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) across every tenant.

        Not scoped to a caller: like `forget`, it assumes the caller is trusted
        for every scope, as on a one-token server.
        """
        return await self._store.fact_counts()

    async def list_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        last_n: int | None = None,
    ) -> list[ChatMessage]:
        """A session's visible transcript turns, oldest first.

        ``last_n=None`` returns every visible turn, with no cap. ``last_n=N``
        returns the newest N, oldest first. ``last_n`` below 1 raises
        ValueError.
        """
        return await self._store.list_session_messages(
            session_id, self._resolve_scope(user_id, org_id), last_n=last_n
        )

    async def explain(
        self, fact_id: UUID, user_id: str | None = None, org_id: str | None = None
    ) -> Provenance | None:
        """Provenance and revocation status for a fact, visible or not. Never its text."""
        return await self._store.explain(fact_id, self._resolve_scope(user_id, org_id))

    async def revocations(
        self,
        user_id: str | None = None,
        org_id: str | None = None,
        include_reinstated: bool = False,
    ) -> list[Revocation]:
        return await self._store.list_revocations(
            self._resolve_scope(user_id, org_id), include_reinstated=include_reinstated
        )

    async def erasures(
        self, user_id: str | None = None, org_id: str | None = None
    ) -> list[ErasureRecord]:
        return await self._store.list_erasures(self._resolve_scope(user_id, org_id))

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
        - If :meth:`erase_user` erases this scope while the call runs, the
          fact is not stored and ScopeErasedError is raised; if the erase
          lands after the fact is stored, it erases the fact and its vector.
        """
        self._require_writable()
        # Before any await: an erase_user that finishes after this refuses the write.
        started = datetime.now().astimezone()
        scope = self._resolve_scope(user_id, org_id)
        check_text("text", text, max_bytes=MAX_FACT_TEXT_BYTES)
        if session_id is not None:
            check_text("session_id", session_id, max_bytes=MAX_FACT_TEXT_BYTES)
        md = dict(metadata or {})
        if role is not None:
            md["role"] = role
        check_json("metadata", md)
        if reject_revoked and await self._store.is_content_revoked(text, scope):
            raise RevokedContentError("text matches revoked content in this scope")
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
        try:
            await self._store.upsert_fact(
                fact,
                writer=actor,
                source="record",
                derived_from=derived_from,
                not_erased_since=started,
            )
        except ScopeErasedError:
            logger.info("record: the scope was erased while the write ran; nothing stored")
            raise
        if await self._store.provenance([fact.id], scope):
            [vec] = await self._embed.embed([text])
            await self._add_vector(fact.id, vec, scope)
        return fact

    async def _add_vector(self, fact_id: UUID, vec: list[float], scope: Scope) -> bool:
        """Index a stored fact's vector. False if an erase removed the fact meanwhile.

        An erase that runs between the upsert and the add purges a vector that
        is not there yet, so the add would leave one for a fact that is gone.
        Checking after the add catches that order; an erase that commits after
        the check purges the vector itself.
        """
        await self._vec.add(fact_id, vec, scope)
        if await self._store.fact_row_committed(fact_id, scope):
            return True
        await self._vec.purge([fact_id], scope)
        return False

    async def record_message(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
        user_id: str | None = None,
        org_id: str | None = None,
        timestamp: datetime | None = None,
        not_erased_since: datetime | None = None,
    ) -> ChatMessage:
        """Record a raw chat turn (no extraction).

        With ``not_erased_since`` (timezone-aware), raises ScopeErasedError and
        stores nothing if :meth:`erase_user` erased the scope at or after it.
        """
        self._require_writable()
        scope = self._resolve_scope(user_id, org_id)
        check_text("content", content, max_bytes=MAX_MESSAGE_BYTES, allow_blank=True)
        check_text("session_id", session_id, max_bytes=MAX_FACT_TEXT_BYTES)
        msg = ChatMessage(
            scope=scope,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=timestamp or datetime.now().astimezone(),
        )
        await self._store.upsert_message(msg, not_erased_since=not_erased_since)
        return msg

    async def extract(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
        actor: str | None = None,
        not_erased_since: datetime | None = None,
    ) -> list[Fact]:
        """Extract durable facts from a conversation. Optionally persist them.

        Requires an LLM to have been passed at `Engram.open(llm=...)`. With
        ``persist=False`` nothing is stored, so it also works read-only.

        If :meth:`erase_user` erases the scope while this runs, the facts not
        yet stored are dropped, and the result lists only those stored before
        the erase. ``not_erased_since`` is when the write started; it defaults
        to the start of this call. Pass an earlier time when the messages were
        stored before this call, so an erase that removed them in between also
        refuses the facts drawn from them. It must be timezone-aware (ValueError).
        """
        if persist:
            self._require_writable()
        if not_erased_since is not None and not_erased_since.tzinfo is None:
            raise ValueError("not_erased_since must be timezone-aware")
        started = datetime.now().astimezone() if not_erased_since is None else not_erased_since
        if self._extract is None:
            raise RuntimeError("extraction requires an LLM; pass `llm=...` to Engram.open()")
        if not messages:
            return []
        scope = self._message_scope(messages)
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
                try:
                    await self._store.upsert_fact(
                        fact, writer=actor, source="extract", not_erased_since=started
                    )
                except ScopeErasedError:
                    logger.info(
                        "extract: the scope was erased while extraction ran; "
                        "returning %d of %d facts",
                        len(out),
                        len(extracted),
                    )
                    break
                if not await self._store.provenance([fact.id], scope):
                    continue  # stored, but born withheld: matches revoked content
                [vec] = await self._embed.embed([ef.text])
                if not await self._add_vector(fact.id, vec, scope):
                    continue  # erased between the upsert and the vector add
                out.append(fact)
        else:
            # Non-persisting branch: synthesize Facts in-memory only, minus
            # anything a tombstone would withhold if it were stored.
            for ef in extracted:
                if await self._store.is_content_revoked(ef.text, scope):
                    continue
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
        user_id: str | None = None,
        org_id: str | None = None,
        top_k: int = 10,
    ) -> list[ScoredFact]:
        """Hybrid (vector + keyword) retrieval, optionally reranked."""
        return await self._retrieve.search(query, self._resolve_scope(user_id, org_id), top_k=top_k)

    async def context_entries(
        self,
        query: str,
        user_id: str | None = None,
        org_id: str | None = None,
        decompose: bool = False,
        role_filter: tuple[str, ...] | None = None,
    ) -> list[ScoredFact]:
        """The ranked memories :meth:`context` renders, each with its provenance.

        Use this when the caller needs to show why a memory went into a prompt.
        """
        self._resolve_scope(user_id, org_id)  # refuse before the decomposer's LLM call
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
        user_id: str | None = None,
        org_id: str | None = None,
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
        scope = self._resolve_scope(user_id, org_id)  # before the classifier runs
        if token_budget is None:
            qt = await classifier.classify(query) if classifier is not None else None
            token_budget = budget_for(qt)
        candidates = await self.context_entries(
            query,
            user_id=scope.user_id,
            org_id=scope.org_id,
            decompose=decompose,
            role_filter=role_filter,
        )
        return "\n".join(context_lines(candidates, token_budget, chars_per_token))


def context_lines(
    entries: list[ScoredFact], token_budget: int, chars_per_token: int = 4
) -> list[str]:
    """The lines :meth:`Engram.context` joins: one per entry, in order, while they fit.

    Stops at the first entry that does not fit the budget. One item per fact,
    even when the fact's text holds newlines, so ``len(result)`` is the number
    of facts included.
    """
    char_budget = token_budget * chars_per_token
    lines: list[str] = []
    running = 0
    for sf in entries:
        line = (
            f"[{sf.fact.event_date.date().isoformat()}] {sf.fact.text}"
            if sf.fact.event_date
            else f"- {sf.fact.text}"
        )
        if running + len(line) + 1 > char_budget:
            break
        lines.append(line)
        running += len(line) + 1
    return lines
