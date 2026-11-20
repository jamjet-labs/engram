"""A handle on one scope of an Engram.

`Engram.scoped(org_id=..., user_id=...)` checks the scope once and returns a
`ScopedEngram`. Its methods take no scope: every read and write goes to the
scope the handle was made for.

- A fact id from another scope reads exactly like an unknown id.
- A `Fact`, or a message passed to `extract`, from another scope is refused
  with ScopeError before anything is read or written, and before any
  embedder or LLM call.
- Every fact, scored fact, event, message, revocation and erasure record the
  store hands back is checked against the scope (`_only_in`) before the
  handle returns it or acts on it. A store that returned another scope's
  data raises ScopeError instead of leaking it.
- A write that started before `erase_all` never lands after it: it carries a
  `ScopeWriteGuard`, the scope's erase epoch when it started, and the store
  refuses it if the epoch has moved.

The root `Engram` methods run on a handle, so this module holds the one
implementation of each.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from engram.classify.base import budget_for
from engram.embedding.base import embed_documents
from engram.errors import NotFoundError, ScopeErasedError, ScopeError
from engram.governance import (
    ErasureRecord,
    Provenance,
    Revocation,
    RevokedContentError,
    require_text,
)
from engram.models import ChatMessage, Event, ExtractedFact, Fact, MemoryTier, Polarity
from engram.read.decomposer import should_decompose
from engram.retrieve.base import ScoredFact
from engram.retrieve.rrf import reciprocal_rank_fusion
from engram.scope import Scope, ScopeWriteGuard
from engram.validate import MAX_FACT_TEXT_BYTES, MAX_MESSAGE_BYTES, check_json, check_text

if TYPE_CHECKING:
    from engram.classify.base import QuestionClassifier
    from engram.engram import Engram
    from engram.extract.event_extractor import EventExtractor
    from engram.extract.pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)

_Owned = Fact | ScoredFact | Event | ChatMessage | Revocation | ErasureRecord


def _names(scope: Scope) -> tuple[str, str]:
    return scope.org_id, scope.user_id


def _owner(item: _Owned) -> tuple[str, str]:
    """The (org_id, user_id) an item says it belongs to."""
    if isinstance(item, ScoredFact):
        if item.provenance is not None and item.provenance.fact_id != item.fact.id:
            raise ScopeError("the store returned provenance for another fact")
        return _names(item.fact.scope)
    if isinstance(item, Fact | Event | ChatMessage):
        return _names(item.scope)
    return item.org_id, item.user_id


def _only_in[T: _Owned](scope: Scope, items: Iterable[T]) -> list[T]:
    """``items`` as a list, or ScopeError if any of them names another scope.

    The store is asked for one scope only, so another scope's item here is a
    store bug. Raising fails closed. The message names no scope and no text.
    """
    out = list(items)
    want = _names(scope)
    for item in out:
        if _owner(item) != want:
            raise ScopeError(f"the store returned a {type(item).__name__} from another scope")
    return out


def _extraction(engram: Engram, persist: bool) -> ExtractionPipeline:
    """The checks `extract` makes before it looks at the messages."""
    if persist:
        engram._require_writable()
    if engram._extract is None:
        raise RuntimeError("extraction requires an LLM; pass `llm=...` to Engram.open()")
    return engram._extract


def _check_guard_type(guard: ScopeWriteGuard | None) -> None:
    value: object = guard
    if value is not None and not isinstance(value, ScopeWriteGuard):
        raise TypeError("guard must be a ScopeWriteGuard, from ScopedEngram.guard()")


def _event_extraction(engram: Engram, persist: bool) -> EventExtractor:
    """The checks `extract_events` makes before it looks at the messages."""
    if persist:
        engram._require_writable()
    if engram._events is None:
        raise RuntimeError("event extraction requires an LLM; pass `llm=...` to Engram.open()")
    return engram._events


class ScopedEngram:
    """Reads and writes one (org_id, user_id) scope of an Engram. No method takes a scope.

    Get one with `Engram.scoped`. The handle shares its Engram's store, index
    and models, and works while the Engram is open. Code given only the
    handle has no call that names another scope; the calls that reach more
    than one scope stay on the root `Engram`.

    Parameters that came after ``user_id`` and ``org_id`` in a root method
    are keyword-only here, so a call ported from the root cannot shift an
    argument into another slot.
    """

    __slots__ = ("_engram", "_scope")

    def __init__(self, engram: Engram, *, org_id: str | None, user_id: str | None) -> None:
        # The ids go through the root's rules, so a handle built here obeys
        # them as one from Engram.scoped does. None means not given.
        self._engram = engram
        self._scope = engram._resolve_scope(user_id, org_id)

    @classmethod
    def _on_stored_scope(cls, engram: Engram, scope: Scope) -> ScopedEngram:
        """A handle on a scope as a store holds it, its ids not checked again.

        For `Engram.erase_refused_scope` only, which checks the ids itself and
        never hands the handle out.
        """
        handle = cls.__new__(cls)
        handle._engram = engram
        handle._scope = scope
        return handle

    @property
    def scope(self) -> Scope:
        """The scope every call on this handle reads and writes."""
        return self._scope

    # ── Checks ──────────────────────────────────────────────────────────

    def _fact_id(self, fact: Fact | UUID) -> UUID:
        # A Fact says which scope it is in, so another scope's is refused
        # outright. A bare id is looked up in this scope only.
        if isinstance(fact, Fact):
            if _names(fact.scope) != _names(self._scope):
                raise ScopeError("the fact belongs to another scope than this handle's")
            return fact.id
        return fact

    def _own_guard(self, guard: ScopeWriteGuard | None) -> None:
        # Before anything is read or written, and before any LLM call.
        _check_guard_type(guard)
        if guard is not None and _names(guard.scope) != _names(self._scope):
            raise ScopeError("the guard is for another scope than this handle's")

    def _own_messages(self, messages: Sequence[ChatMessage]) -> None:
        want = _names(self._scope)
        for i, message in enumerate(messages):
            if _names(message.scope) != want:
                raise ScopeError(
                    f"message {i} belongs to another scope than this handle's; "
                    "extract from one scope at a time"
                )

    async def _drop_hidden_vectors(self) -> None:
        # Not needed for correctness (the store withholds hidden facts), but
        # keeps hidden vectors from filling the fixed candidate pool.
        engram = self._engram
        for fid in await engram._store.hidden_fact_ids(self._scope):
            await engram._vec.delete(fid, self._scope)

    async def _add_vector(self, fact_id: UUID, vec: list[float]) -> bool:
        """Index a stored fact's vector. False if an erase removed the fact meanwhile.

        An erase that runs between the upsert and the add purges a vector that
        is not there yet, so the add would leave one for a fact that is gone.
        Checking after the add catches that order; an erase that commits after
        the check purges the vector itself.
        """
        engram = self._engram
        await engram._vec.add(fact_id, vec, self._scope)
        if await engram._store.fact_row_committed(fact_id, self._scope):
            return True
        await engram._vec.purge([fact_id], self._scope)
        return False

    # ── Writes ──────────────────────────────────────────────────────────

    async def guard(self) -> ScopeWriteGuard:
        """The scope's erase epoch now, for a write that starts now.

        Take it when a write starts and pass it to each `record_message`,
        `extract` and `extract_events` the write makes. If :meth:`erase_all`
        erases the scope after this, those calls store nothing and raise
        ScopeErasedError (`extract` and `extract_events` return what they
        stored before). No clock is read. A read: it works read-only too.
        `record`, `extract` and `extract_events` take their own when given none.
        """
        guard = await self._engram._store.scope_guard(self._scope)
        if not isinstance(guard, ScopeWriteGuard) or _names(guard.scope) != _names(self._scope):
            raise ScopeError("the store returned a guard for another scope")
        return guard

    async def record(
        self,
        text: str,
        *,
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
        """Record a single fact (no LLM extraction). Embeds, then stores, in one call.

        The text is embedded (`embed_documents`) before anything is written,
        so an embedder error, or a bad vector (EmbeddingResponseError), raises
        and stores nothing.

        Pass `session_id` to enable Phase 9 two-stage retrieval: facts in the
        same session can then be retrieved together via session-first ranking.

        Pass `role` (e.g., "user", "assistant") to tag the fact's origin role.
        Stored in ``metadata["role"]``; queryable via ``context(role_filter=...)``.

        Governance:

        - ``actor`` is stored as the writer in the fact's provenance.
        - ``derived_from`` names facts this one was concluded from. If any of
          them is revoked (now or later) this fact is withheld too. An id
          that is not a fact in this scope raises NotFoundError.
        - ``source_message_id`` links the transcript turn the fact came from.
          It must be a turn in this scope (NotFoundError).
        - Text that matches revoked content is stored but withheld (a write
          cannot un-revoke). ``reject_revoked=True`` raises instead.
        - If :meth:`erase_all` erases this scope while the call runs, the
          fact is not stored and ScopeErasedError is raised; if the erase
          lands after the fact is stored, it erases the fact and its vector.
        """
        engram = self._engram
        engram._require_writable()
        scope = self._scope
        check_text("text", text, max_bytes=MAX_FACT_TEXT_BYTES)
        if session_id is not None:
            check_text("session_id", session_id, max_bytes=MAX_FACT_TEXT_BYTES)
        md = dict(metadata or {})
        if role is not None:
            md["role"] = role
        check_json("metadata", md)
        # The first await: an erase_all that commits after this refuses the write.
        guard = await self.guard()
        store = engram._store
        if reject_revoked and await store.is_content_revoked(text, scope):
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
        # Before the write: a failed embedding stores nothing. After the
        # guard: an erase that commits while the embedder runs refuses it.
        [vec] = await embed_documents(engram._embed, [text])
        try:
            await store.upsert_fact(
                fact,
                writer=actor,
                source="record",
                derived_from=derived_from,
                guard=guard,
            )
        except ScopeErasedError:
            logger.info("record: the scope was erased while the write ran; nothing stored")
            raise
        if await store.provenance([fact.id], scope):
            await self._add_vector(fact.id, vec)
        return fact

    async def record_message(
        self,
        content: str,
        role: str = "user",
        session_id: str = "default",
        *,
        timestamp: datetime | None = None,
        guard: ScopeWriteGuard | None = None,
    ) -> ChatMessage:
        """Record a raw chat turn (no extraction).

        With ``guard`` (from :meth:`guard`), raises ScopeErasedError and
        stores nothing if :meth:`erase_all` erased the scope after the guard
        was taken. A guard for another scope raises ScopeError.
        """
        engram = self._engram
        engram._require_writable()
        self._own_guard(guard)
        check_text("content", content, max_bytes=MAX_MESSAGE_BYTES, allow_blank=True)
        check_text("session_id", session_id, max_bytes=MAX_FACT_TEXT_BYTES)
        msg = ChatMessage(
            scope=self._scope,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=timestamp or datetime.now().astimezone(),
        )
        await engram._store.upsert_message(msg, guard=guard)
        return msg

    async def extract(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
        actor: str | None = None,
        guard: ScopeWriteGuard | None = None,
    ) -> list[Fact]:
        """Extract durable facts from a conversation. Optionally persist them.

        Every message must be in this handle's scope; otherwise ScopeError,
        before the LLM sees any of them. Requires an LLM to have been passed at
        `Engram.open(llm=...)`. With ``persist=False`` nothing is stored, so it
        also works read-only.

        The facts are embedded together (`embed_documents`) before the first
        is written, so an embedder error, or a bad vector
        (EmbeddingResponseError), raises and stores none of them.

        If :meth:`erase_all` erases the scope while this runs, the facts not
        yet stored are dropped, and the result lists only those stored before
        the erase. ``guard`` says when the write started; without one, this
        takes one before the LLM call. Pass the one taken before the messages
        were stored, so an erase that removed them in between also refuses the
        facts drawn from them. A guard for another scope raises ScopeError,
        before the LLM call.
        """
        engram = self._engram
        pipeline = _extraction(engram, persist)
        self._own_guard(guard)
        if not messages:
            return []
        self._own_messages(messages)
        scope = self._scope
        store = engram._store
        if persist and guard is None:
            guard = await self.guard()
        # All input messages share a session: take the first one's session_id
        sid: str | None = messages[0].session_id
        extracted: list[ExtractedFact] = await pipeline.extract(messages, session_date=session_date)
        out: list[Fact] = []
        if persist:
            facts = [
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
                for ef in extracted
            ]
            # One call, before the first write: a failed embedding stores none.
            vectors = await embed_documents(engram._embed, [f.text for f in facts])
            for fact, vec in zip(facts, vectors, strict=True):
                try:
                    await store.upsert_fact(fact, writer=actor, source="extract", guard=guard)
                except ScopeErasedError:
                    logger.info(
                        "extract: the scope was erased while extraction ran; "
                        "returning %d of %d facts",
                        len(out),
                        len(extracted),
                    )
                    break
                if not await store.provenance([fact.id], scope):
                    continue  # stored, but born withheld: matches revoked content
                if not await self._add_vector(fact.id, vec):
                    continue  # erased between the upsert and the vector add
                out.append(fact)
        else:
            # Non-persisting branch: synthesize Facts in-memory only, minus
            # anything a tombstone would withhold if it were stored.
            for ef in extracted:
                if await store.is_content_revoked(ef.text, scope):
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

    async def extract_events(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
        persist: bool = True,
        *,
        guard: ScopeWriteGuard | None = None,
    ) -> list[Event]:
        """Phase 11: extract SVO events from a chat segment via LLM.

        Every message must be in this handle's scope; otherwise ScopeError,
        before the LLM sees any of them. Persists the events into the event
        calendar by default. Requires an LLM to have been passed at
        `Engram.open(llm=...)`. With ``persist=False`` nothing is stored, so
        it also works read-only.

        ``guard`` works as in :meth:`extract`: if :meth:`erase_all` erases the
        scope while this runs, the events not yet stored are dropped, and the
        result lists only those stored before the erase.
        """
        engram = self._engram
        extractor = _event_extraction(engram, persist)
        self._own_guard(guard)
        if not messages:
            return []
        self._own_messages(messages)
        if persist and guard is None:
            guard = await self.guard()
        events = _only_in(self._scope, await extractor.extract(messages, session_date=session_date))
        if not persist:
            return events
        stored: list[Event] = []
        for ev in events:
            try:
                await engram._store.upsert_event(ev, guard=guard)
            except ScopeErasedError:
                logger.info(
                    "extract_events: the scope was erased while event extraction ran; "
                    "returning %d of %d events",
                    len(stored),
                    len(events),
                )
                break
            stored.append(ev)
        return stored

    async def supersede(
        self,
        old_fact_id: UUID,
        new_fact_id: UUID,
        *,
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
        refused. Either id outside this scope raises ValueError, as an unknown
        id does.
        """
        engram = self._engram
        engram._require_writable()
        try:
            rev = await engram._store.revoke(
                old_fact_id,
                self._scope,
                reason=reason or f"superseded by {new_fact_id}",
                actor=actor or "unspecified",
                replaced_by=new_fact_id,
            )
        except NotFoundError as e:
            raise ValueError(str(e)) from e
        _only_in(self._scope, [rev])
        await self._drop_hidden_vectors()
        return rev

    async def revoke(self, fact: Fact | UUID, *, reason: str, actor: str) -> Revocation:
        """Tombstone a fact so no read path returns it again.

        Also withholds every fact, transcript turn and event in this scope
        whose text equals or quotes it, and every fact recorded with
        ``derived_from`` pointing at it. The fact stays in the database for
        audit; use :meth:`erase` to delete it. Idempotent. A ``Fact`` from
        another scope raises ScopeError; an id outside this scope raises
        NotFoundError, as an unknown id does.
        """
        engram = self._engram
        engram._require_writable()
        fact_id = self._fact_id(fact)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        rev = await engram._store.revoke(fact_id, self._scope, reason=reason, actor=actor)
        _only_in(self._scope, [rev])
        await self._drop_hidden_vectors()
        return rev

    async def reinstate(self, fact: Fact | UUID, *, reason: str, actor: str) -> list[Revocation]:
        """Lift the tombstones placed on this fact. The only way to un-revoke.

        Works on the revoked fact itself, not on a copy hidden by its
        tombstone. The tombstones stay in the audit trail with who lifted
        them and why. The fact can stay hidden if another tombstone also
        covers it; check :meth:`explain`. Another scope's fact is refused as
        in :meth:`revoke`.
        """
        engram = self._engram
        engram._require_writable()
        fact_id = self._fact_id(fact)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        lifted, visible = await engram._store.reinstate(
            fact_id, self._scope, reason=reason, actor=actor
        )
        # Before the embedder sees any text or the index takes any vector.
        _only_in(self._scope, lifted)
        _only_in(self._scope, visible)
        if visible:
            # Checked before any reaches the index. The tombstones are lifted
            # already: on an embedder error the facts are visible, found by
            # keyword search, with no vector.
            vecs = await embed_documents(engram._embed, [f.text for f in visible])
            for f, v in zip(visible, vecs, strict=True):
                await engram._vec.add(f.id, v, self._scope)
        return lifted

    async def erase(
        self,
        fact: Fact | UUID,
        *,
        reason: str,
        actor: str,
        block_reinsertion: bool = True,
    ) -> ErasureRecord:
        """Hard-delete a fact, its copies and quotes, what was derived from it, and their vectors.

        Returns a content-free :class:`ErasureRecord`. ``reason`` and ``actor``
        must not quote the erased text. If the fact was revoked, its tombstone
        keeps the content hash so the text stays blocked; pass
        ``block_reinsertion=False`` to drop it too. See the README section on
        erasure for what the backend can and cannot vouch for. Another
        scope's fact is refused as in :meth:`revoke`.
        """
        engram = self._engram
        engram._require_writable()
        fact_id = self._fact_id(fact)
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        store = engram._store
        record = await store.erase(
            fact_id, self._scope, reason=reason, actor=actor, block_reinsertion=block_reinsertion
        )
        # The purge touches this scope's index only. The record is checked
        # before finalize writes to it.
        removed = await engram._vec.purge(record.fact_ids, self._scope)
        _only_in(self._scope, [record])
        final = await store.finalize_erasure(
            record, {"vectors": removed}, {"vector_index_rebuilt": True}
        )
        return _only_in(self._scope, [final])[0]

    async def forget(self, fact_id: UUID, *, reason: str, actor: str) -> bool:
        """Revoke a fact of this scope by id. Soft and reversible, like :meth:`revoke`.

        Returns False if this scope holds no fact with this id (another
        scope's id included), or if it was erased before the revoke landed.
        Use :meth:`erase` to hard-delete.
        """
        try:
            await self.revoke(fact_id, reason=reason, actor=actor)
        except NotFoundError:
            return False
        return True

    async def erase_all(self, *, reason: str, actor: str) -> int:
        """Hard-delete everything held for this scope.

        Every fact goes through :meth:`erase` with `block_reinsertion=False`, so
        its copies, quotes, derived facts, events, cited transcript turns and
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
        engram = self._engram
        engram._require_writable()
        scope = self._scope
        store = engram._store
        reason = require_text("reason", reason)
        actor = require_text("actor", actor)
        for label, value in (("reason", reason), ("actor", actor)):
            if await store.fact_text_quoted_by(value, scope):
                raise ValueError(f"erasure {label} must not quote the content being erased")
        erased = 0
        for fact_id in await store.list_fact_ids(scope):
            try:
                record = await self.erase(
                    fact_id, reason=reason, actor=actor, block_reinsertion=False
                )
            except NotFoundError:
                continue  # already removed as a copy or derivation of an earlier one
            erased += len(record.fact_ids)
        residue = await store.erase_scope_residue(scope, reason=reason, actor=actor)
        _only_in(scope, [residue])  # before finalize writes to it
        await store.finalize_erasure(residue, {}, {})
        return erased

    async def delete_conversation(self, session_id: str) -> int:
        """Hard-delete one session's transcript. Facts extracted from it stay.

        Returns the number of turns removed. The facts lose their
        `source_message_id` link. If one of those facts was revoked, rule (c)
        of `live_messages` withholds every later turn in a session reusing
        that id. This is accepted fail-closed behavior.
        """
        engram = self._engram
        engram._require_writable()
        return await engram._store.delete_session_messages(session_id, self._scope)

    # ── Reads ───────────────────────────────────────────────────────────

    async def recall(self, query: str, *, top_k: int = 10) -> list[ScoredFact]:
        """Hybrid (vector + keyword) retrieval, optionally reranked."""
        found = await self._engram._retrieve.search(query, self._scope, top_k=top_k)
        return _only_in(self._scope, found)

    async def context_entries(
        self,
        query: str,
        *,
        decompose: bool = False,
        role_filter: tuple[str, ...] | None = None,
    ) -> list[ScoredFact]:
        """The ranked memories :meth:`context` renders, each with its provenance.

        Use this when the caller needs to show why a memory went into a prompt.
        """
        decomposer = self._engram._decomposer
        candidates: list[ScoredFact]
        if decompose and decomposer is not None and should_decompose(query):
            subqueries = await decomposer.decompose(query)
        else:
            subqueries = [query]

        if len(subqueries) == 1:
            candidates = await self.recall(subqueries[0], top_k=30)
        else:
            per_q = await asyncio.gather(*[self.recall(sq, top_k=15) for sq in subqueries])
            ranked_lists = [[sf.fact.id for sf in lst] for lst in per_q]
            fused_ids = reciprocal_rank_fusion(ranked_lists, k=60)
            by_id: dict[UUID, ScoredFact] = {sf.fact.id: sf for lst in per_q for sf in lst}
            candidates = [by_id[fid] for fid in fused_ids if fid in by_id]

        # Role filter: applied AFTER recall, BEFORE the token-budget loop.
        if role_filter is not None:
            candidates = [c for c in candidates if c.fact.metadata.get("role") in role_filter]
        return candidates

    async def context(
        self,
        query: str,
        *,
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
        candidates = await self.context_entries(query, decompose=decompose, role_filter=role_filter)
        return "\n".join(context_lines(candidates, token_budget, chars_per_token))

    async def search_events(
        self,
        query: str = "",
        *,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        limit: int = 10,
    ) -> list[Event]:
        """Phase 11: query the event calendar by FTS + optional time window."""
        found = await self._engram._store.search_events(
            query, self._scope, time_start=time_start, time_end=time_end, limit=limit
        )
        return _only_in(self._scope, found)

    async def explain(self, fact_id: UUID) -> Provenance | None:
        """Provenance and revocation status for a fact, visible or not. Never its text.

        None when this scope holds no fact with this id, another scope's id
        included.
        """
        store = self._engram._store
        # A Provenance names no scope, so it cannot be checked after the
        # fact. Ask only about an id this scope holds; a fact never moves
        # to another scope, so the answer is this scope's.
        if fact_id not in await store.derivation_targets_in_scope([fact_id], self._scope):
            return None
        prov = await store.explain(fact_id, self._scope)
        if prov is not None and prov.fact_id != fact_id:
            raise ScopeError("the store returned provenance for another fact")
        return prov

    async def revocations(self, *, include_reinstated: bool = False) -> list[Revocation]:
        """This scope's tombstones; lifted ones too with ``include_reinstated``."""
        found = await self._engram._store.list_revocations(
            self._scope, include_reinstated=include_reinstated
        )
        return _only_in(self._scope, found)

    async def erasures(self) -> list[ErasureRecord]:
        """This scope's erasure records."""
        return _only_in(self._scope, await self._engram._store.list_erasures(self._scope))

    async def list_conversations(self) -> list[str]:
        """Session ids with a visible transcript turn, by first turn."""
        return await self._engram._store.list_session_ids(self._scope)

    async def list_messages(
        self, session_id: str, *, last_n: int | None = None
    ) -> list[ChatMessage]:
        """A session's visible transcript turns, oldest first.

        ``last_n=None`` returns every visible turn, with no cap. ``last_n=N``
        returns the newest N, oldest first. ``last_n`` below 1 raises
        ValueError.
        """
        found = await self._engram._store.list_session_messages(
            session_id, self._scope, last_n=last_n
        )
        return _only_in(self._scope, found)

    async def fact_counts(self) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) in this scope."""
        return await self._engram._store.fact_counts(self._scope)


def context_lines(
    entries: list[ScoredFact], token_budget: int, chars_per_token: int = 4
) -> list[str]:
    """The lines :meth:`ScopedEngram.context` joins: one per entry, in order, while they fit.

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
