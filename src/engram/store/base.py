"""Storage protocol. Concrete backends implement it; SQLite is the one today."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from engram.governance import ErasureRecord, Provenance, Revocation
from engram.models import ChatMessage, Event, Fact
from engram.scope import Scope


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
    ) -> None: ...

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
    async def upsert_message(self, message: ChatMessage) -> None: ...

    @abstractmethod
    async def list_messages(
        self, session_id: str, scope: Scope, limit: int = 1000
    ) -> list[ChatMessage]: ...

    @abstractmethod
    async def upsert_event(self, event: Event) -> None: ...

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
