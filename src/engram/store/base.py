"""Storage protocol. Concrete backends implement it; SQLite is the one today."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from engram.governance import ErasureRecord, Provenance, Revocation
from engram.models import ChatMessage, Event, Fact
from engram.scope import Scope, ScopeWriteGuard


@runtime_checkable
class EngramStore(Protocol):
    """Async storage for facts, chat messages, events and their governance records.

    Concrete implementations: `engram.store.sqlite.SqliteStore`. `Engram` and
    `HybridRetriever` use only the members below, so any store that has them
    can sit behind the facade (`Engram.open(store=...)`).

    Revocation contract: every method that returns facts, events or messages
    must withhold revoked content (see `engram.governance`). `provenance()` is
    the last check retrieval runs before handing results back, so it must
    omit ids that are not visible at the time of the call.

    Read-only contract: when `read_only` is true, every write method raises
    `ReadOnlyStoreError` before it reads or writes anything, `record_access`
    does nothing, and every read works as usual.
    """

    @property
    @abstractmethod
    def read_only(self) -> bool:
        """True when the store was opened for reads only."""
        ...

    @abstractmethod
    async def close(self) -> None: ...

    # ── Facts ────────────────────────────────────────────────────────────

    @abstractmethod
    async def upsert_fact(
        self,
        fact: Fact,
        *,
        writer: str | None = None,
        source: str = "upsert",
        derived_from: Iterable[UUID] | None = None,
        guard: ScopeWriteGuard | None = None,
    ) -> None:
        """Insert or update a fact, its provenance row and its derivation links.

        With ``guard`` (from `scope_guard`), compares the scope's erase epoch
        inside the write's transaction: raises ScopeErasedError and writes
        nothing if it has moved since the guard was taken. A guard for another
        scope raises ScopeError.
        """
        ...

    @abstractmethod
    async def get_fact(self, fact_id: UUID, scope: Scope) -> Fact | None: ...

    @abstractmethod
    async def get_facts(self, fact_ids: Iterable[UUID], scope: Scope) -> list[Fact]: ...

    @abstractmethod
    async def list_facts_by_session(
        self, session_id: str, scope: Scope, limit: int = 100
    ) -> list[Fact]: ...

    @abstractmethod
    async def keyword_search(self, query: str, scope: Scope, limit: int = 30) -> list[Fact]: ...

    @abstractmethod
    async def aggregate_sessions(
        self, query: str, scope: Scope, top_sessions: int = 5
    ) -> list[tuple[str, float]]:
        """Phase 9: rank sessions by aggregate fact relevance to a query."""
        ...

    @abstractmethod
    async def record_access(self, fact_id: UUID) -> None:
        """Count a read of this fact. May be applied later. Does nothing when read-only."""
        ...

    @abstractmethod
    async def flush_access(self) -> None:
        """Wait until every counted read is written."""
        ...

    # ── Messages and events ──────────────────────────────────────────────

    @abstractmethod
    async def upsert_message(
        self, message: ChatMessage, *, guard: ScopeWriteGuard | None = None
    ) -> None:
        """Write a transcript turn. ``guard`` works as in `upsert_fact`."""
        ...

    @abstractmethod
    async def list_messages(
        self, session_id: str, scope: Scope, limit: int = 1000
    ) -> list[ChatMessage]: ...

    @abstractmethod
    async def upsert_event(self, event: Event, *, guard: ScopeWriteGuard | None = None) -> None:
        """Write an event. ``guard`` works as in `upsert_fact`."""
        ...

    @abstractmethod
    async def get_event(self, event_id: UUID, scope: Scope) -> Event | None: ...

    @abstractmethod
    async def search_events(
        self,
        query: str,
        scope: Scope,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        limit: int = 10,
    ) -> list[Event]:
        """Return events matching the query string + optional time-window."""
        ...

    # ── Governance: provenance, revocation, erasure ──────────────────────

    @abstractmethod
    async def is_content_revoked(self, text: str, scope: Scope) -> bool:
        """True if an active tombstone in scope would withhold this text."""
        ...

    @abstractmethod
    async def provenance(
        self, fact_ids: Iterable[UUID], scope: Scope
    ) -> dict[UUID, Provenance]: ...

    @abstractmethod
    async def explain(self, fact_id: UUID, scope: Scope) -> Provenance | None:
        """Provenance and revocation status for one fact, visible or not. Never its text."""
        ...

    @abstractmethod
    async def derivation_targets_in_scope(self, ids: Iterable[UUID], scope: Scope) -> set[UUID]:
        """Which of ``ids`` exist in ``scope``, visible or not. Ids only."""
        ...

    @abstractmethod
    async def hidden_fact_ids(self, scope: Scope) -> set[UUID]:
        """Ids of the facts in scope that are withheld now. Ids only.

        The facade uses it to drop withheld facts' vectors from its index. It
        leaves the protocol when stores keep their own vectors.
        """
        ...

    @abstractmethod
    async def revoke(
        self,
        fact_id: UUID,
        scope: Scope,
        *,
        reason: str,
        actor: str,
        replaced_by: UUID | None = None,
    ) -> Revocation: ...

    @abstractmethod
    async def reinstate(
        self, fact_id: UUID, scope: Scope, *, reason: str, actor: str
    ) -> tuple[list[Revocation], list[Fact]]:
        """Lift this fact's tombstones. Returns them, and the facts visible again."""
        ...

    @abstractmethod
    async def list_revocations(
        self, scope: Scope, include_reinstated: bool = False
    ) -> list[Revocation]: ...

    @abstractmethod
    async def erase(
        self,
        fact_id: UUID,
        scope: Scope,
        *,
        reason: str,
        actor: str,
        block_reinsertion: bool = True,
    ) -> ErasureRecord: ...

    @abstractmethod
    async def finalize_erasure(
        self, record: ErasureRecord, extra_artifacts: dict[str, int], extra: dict[str, Any]
    ) -> ErasureRecord:
        """Add what the caller removed outside the store to the erasure record."""
        ...

    @abstractmethod
    async def list_erasures(self, scope: Scope) -> list[ErasureRecord]: ...

    # ── v1 REST support (ids and counts only, no text) ───────────────────
    #
    # `find_fact_owner_in_org` and `fact_counts_in_org` read every user of
    # the one org they are given, for the root calls named for that
    # (`Engram.forget_in_org`, `Engram.fact_counts_in_org`). Nothing else
    # here reads more than one org; `refused_scopes` lists ids of scopes no
    # other call can name. `fact_row_committed` leaves when stores keep
    # their own vectors.

    @abstractmethod
    async def fact_row_committed(self, fact_id: UUID, scope: Scope) -> bool:
        """Is the fact's row there as of the latest commit? A bool, never text."""
        ...

    @abstractmethod
    async def find_fact_owner_in_org(self, fact_id: UUID, org_id: str) -> Scope | None:
        """The scope of the fact with this id in ``org_id``, hidden facts included.

        None when no fact in that org has the id, so another org's id reads
        like an unknown one.
        """
        ...

    @abstractmethod
    async def list_fact_ids(self, scope: Scope) -> list[UUID]:
        """Every fact id in scope, hidden ones included. For erasure only."""
        ...

    @abstractmethod
    async def fact_text_quoted_by(self, value: str, scope: Scope) -> bool:
        """Does ``value`` quote any fact text in scope, hidden facts included? A bool."""
        ...

    @abstractmethod
    async def fact_counts(self, scope: Scope) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) in ``scope``."""
        ...

    @abstractmethod
    async def fact_counts_in_org(self, org_id: str) -> tuple[int, int]:
        """(all facts, visible and not invalidated facts) of every user in ``org_id``."""
        ...

    @abstractmethod
    async def refused_scopes(self) -> list[tuple[str, str]]:
        """(org_id, user_id) of each scope holding memory under ids Scope refuses, sorted.

        Rows written before ids were checked. Ids only. Memory is what
        `Engram.erase_refused_scope` removes, so a scope left with only its
        audit rows is not listed. A store reads such rows back with
        `engram.scope.stored_scope`.
        """
        ...

    @abstractmethod
    async def list_session_ids(self, scope: Scope) -> list[str]:
        """Sessions with a visible transcript turn in scope, by first turn."""
        ...

    @abstractmethod
    async def list_session_messages(
        self, session_id: str, scope: Scope, *, last_n: int | None = None
    ) -> list[ChatMessage]:
        """A session's visible turns, oldest first; the newest ``last_n`` when given."""
        ...

    @abstractmethod
    async def delete_session_messages(self, session_id: str, scope: Scope) -> int:
        """Hard-delete one session's transcript in scope. Returns turns removed."""
        ...

    @abstractmethod
    async def erase_scope_residue(self, scope: Scope, *, reason: str, actor: str) -> ErasureRecord:
        """Hard-delete what per-fact erasure leaves in a scope; refuse while facts remain.

        Moves the scope's erase epoch on by one in the same transaction.
        """
        ...

    @abstractmethod
    async def scope_guard(self, scope: Scope) -> ScopeWriteGuard:
        """The scope's erase epoch as of the latest commit; 0 if it was never erased. A read."""
        ...

    # ── Embedder identity ────────────────────────────────────────────────
    #
    # One key per store: the `EmbedderIdentity.key` of the embedder its
    # vectors come from. `Engram.open` records it on a writable open and
    # refuses an open with another, so vectors from two embedders never mix.

    @abstractmethod
    async def embedder_key(self) -> str | None:
        """The embedder key the store records, as of the latest commit; None if none. A read."""
        ...

    @abstractmethod
    async def set_embedder_key(self, key: str, *, replace: bool) -> None:
        """Record ``key`` as the store's embedder. Check and write are one transaction.

        No key recorded: records it. The same key: does nothing. Another key:
        with ``replace=False`` raises EmbedderMismatchError naming both keys
        and writes nothing; with ``replace=True`` replaces it. So of two
        processes that record different keys at once, one wins and the other
        is refused. ``key`` is checked like other text (blank or NUL is a
        ValueError); ``replace`` must be a bool (TypeError).
        """
        ...
