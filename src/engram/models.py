"""Core Pydantic schemas for Engram v2.

Wire-protocol commitment: these JSON shapes match Rust v0.5.x exactly so existing
clients (Spring Boot starter, langchain4j, Python SDK, Java SDK) keep working.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from engram.scope import Scope


class MemoryTier(StrEnum):
    """Storage tier for a fact. Influences eviction + retrieval boosting."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVAL = "archival"


class Polarity(StrEnum):
    """Whether a fact asserts, negates, or hypothesizes."""

    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    HYPOTHETICAL = "hypothetical"


class Entity(BaseModel):
    """A named entity referenced by one or more facts."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=512)
    aliases: list[str] = Field(default_factory=list)
    scope: Scope = Field(default_factory=Scope)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A subject-predicate-object triple linking entities."""

    id: UUID = Field(default_factory=uuid4)
    source: str
    relation: str
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractedFact(BaseModel):
    """A fact emitted by the extraction pipeline before being persisted as a `Fact`.

    The shape mirrors the `extract.rs` ExtractedFact in Rust v0.5.x.
    """

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    category: str | None = None
    polarity: Polarity = Polarity.AFFIRMATIVE
    entities: list[str] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    event_date: datetime | None = None
    mention_date: datetime | None = None
    source_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    """A persisted memory fact."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=1)
    scope: Scope
    valid_from: datetime
    invalid_at: datetime | None = None

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    category: str | None = None
    polarity: Polarity = Polarity.AFFIRMATIVE
    tier: MemoryTier = MemoryTier.WORKING

    event_date: datetime | None = None
    mention_date: datetime | None = None
    source_event_id: str | None = None
    source_message_id: UUID | None = None
    source_span: tuple[int, int] | None = None
    session_id: str | None = None  # Phase 9: enables two-stage session-first retrieval

    supersedes: UUID | None = None
    superseded_by: UUID | None = None

    entity_refs: list[UUID] = Field(default_factory=list)

    access_count: int = 0
    last_accessed: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_valid_at(self, when: datetime) -> bool:
        """Return True if the fact is in its validity window at the given time."""
        if when < self.valid_from:
            return False
        if self.invalid_at is not None and when >= self.invalid_at:
            return False
        return True


class Event(BaseModel):
    """Phase 11: an SVO event tuple with absolute time range.

    Inspired by the Chronos benchmark architecture. Each event is a structured
    fact about something that happened, with canonical subject/verb/object
    plus a `[time_start, time_end]` window. Aliases let the same event be
    matched by surface variants ("dinner with Mom" / "family dinner").
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    scope: Scope
    subject_canonical: str = Field(min_length=1, max_length=512)
    verb: str = Field(min_length=1, max_length=128)
    object_canonical: str = Field(min_length=1, max_length=512)
    time_start: datetime
    time_end: datetime | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    source_fact_ids: list[UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A raw conversation turn — stored alongside extracted facts so we can
    re-extract or drill back to source utterances.

    Per-turn storage from day 1 — the SVO event calendar (Phase 11) builds on this.
    """

    id: UUID = Field(default_factory=uuid4)
    scope: Scope
    session_id: str = Field(min_length=1, max_length=256)
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
