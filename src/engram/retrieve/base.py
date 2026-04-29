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

    # Phase 9: two-stage retrieval. When True, retrieve top-K sessions first
    # (by aggregate fact relevance) then restrict the candidate pool to facts
    # from those sessions. Helps when haystacks are pre-segmented.
    enable_two_stage: bool = False
    two_stage_top_sessions: int = Field(default=3, ge=1, le=20)

    # Phase 13: active fact-versioning. When True (default), facts whose
    # `superseded_by` is set are excluded from results — only the canonical
    # latest version surfaces. Set False to include the full history.
    exclude_superseded: bool = True


@runtime_checkable
class Reranker(Protocol):
    """Reorder a candidate list by direct (query, fact.text) relevance.

    Implementations live in `engram.retrieve.rerank`.
    """

    @abstractmethod
    async def rerank(
        self, query: str, scored: list[ScoredFact], top_k: int = 10
    ) -> list[ScoredFact]: ...
