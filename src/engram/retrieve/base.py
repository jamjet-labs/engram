"""Retrieval primitives — ScoredFact, RetrievalConfig, Reranker protocol."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from engram.models import Fact


class ScoredFact(BaseModel):
    """A fact with merged retrieval scores from multiple signals."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    fact: Fact
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    temporal_score: float = 0.0


class RetrievalConfig(BaseModel):
    """Default weights from the AgentMemory analysis (2026-04-29).

    Slightly tuned from the v0.6 Rust defaults: rerank gets the highest weight
    when present; otherwise vector+keyword dominate. Temporal kicks in only
    when the query expresses temporal intent (RECENCY / DURATION / etc).
    """

    vector_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    temporal_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    candidate_pool_multiplier: int = Field(default=3, ge=1, le=10)
    temporal_sigma_days: float = Field(default=30.0, gt=0.0)


@runtime_checkable
class Reranker(Protocol):
    """Reorder a candidate list by direct (query, fact.text) relevance.

    Implementations live in `engram.retrieve.rerank`.
    """

    @abstractmethod
    async def rerank(
        self, query: str, scored: list[ScoredFact], top_k: int = 10
    ) -> list[ScoredFact]: ...
