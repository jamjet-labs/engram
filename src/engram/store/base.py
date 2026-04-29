"""Storage protocol — concrete backends (SQLite, Postgres) implement this."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from engram.models import ChatMessage, Event, Fact
from engram.scope import Scope


@runtime_checkable
class EngramStore(Protocol):
    """Async storage interface for facts + raw chat messages.

    Concrete implementations: `engram.store.sqlite.SqliteStore`.
    """

    @abstractmethod
    async def upsert_fact(self, fact: Fact) -> None: ...

    @abstractmethod
    async def get_fact(self, fact_id: UUID, scope: Scope) -> Fact | None: ...

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
    async def upsert_message(self, message: ChatMessage) -> None: ...

    @abstractmethod
    async def list_messages(
        self, session_id: str, scope: Scope, limit: int = 1000
    ) -> list[ChatMessage]: ...

    @abstractmethod
    async def record_access(self, fact_id: UUID) -> None: ...

    # Phase 11: SVO event calendar ──────────────────────────────────────

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

    @abstractmethod
    async def close(self) -> None: ...
