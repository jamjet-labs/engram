"""VectorStore protocol + match type."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, Field

from engram.scope import Scope


class VectorMatch(BaseModel):
    """A scored vector search hit."""

    fact_id: UUID
    score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class VectorStore(Protocol):
    """Async vector index keyed by `fact_id` and scoped by `(org_id, user_id)`."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def add(self, fact_id: UUID, vector: list[float], scope: Scope) -> None: ...

    @abstractmethod
    async def search(self, query: list[float], scope: Scope, k: int = 10) -> list[VectorMatch]: ...

    @abstractmethod
    async def delete(self, fact_id: UUID, scope: Scope) -> None: ...

    @abstractmethod
    async def count(self, scope: Scope) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...
