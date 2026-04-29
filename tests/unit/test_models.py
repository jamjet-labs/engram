from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from engram.models import (
    ChatMessage,
    Entity,
    ExtractedFact,
    Fact,
    MemoryTier,
    Polarity,
    Relationship,
)
from engram.scope import Scope


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


# ── Fact ────────────────────────────────────────────────────────────


def test_fact_minimal_construction() -> None:
    f = Fact(
        text="user prefers espresso",
        scope=Scope(org_id="acme", user_id="alice"),
        valid_from=_now(),
    )
    assert isinstance(f.id, UUID)
    assert f.confidence == 1.0
    assert f.tier == MemoryTier.WORKING
    assert f.polarity == Polarity.AFFIRMATIVE
    assert f.access_count == 0


def test_fact_id_is_unique_per_instance() -> None:
    a = Fact(text="x", scope=Scope(), valid_from=_now())
    b = Fact(text="x", scope=Scope(), valid_from=_now())
    assert a.id != b.id


def test_fact_confidence_range_validation() -> None:
    with pytest.raises(ValidationError):
        Fact(text="x", scope=Scope(), valid_from=_now(), confidence=1.5)
    with pytest.raises(ValidationError):
        Fact(text="x", scope=Scope(), valid_from=_now(), confidence=-0.1)


def test_fact_is_valid_when_no_invalid_at() -> None:
    f = Fact(text="x", scope=Scope(), valid_from=_now())
    assert f.is_valid_at(_now()) is True


def test_fact_is_invalid_after_invalid_at() -> None:
    later = datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)
    f = Fact(text="x", scope=Scope(), valid_from=_now(), invalid_at=later)
    assert f.is_valid_at(_now()) is True
    assert f.is_valid_at(datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)) is False


def test_fact_serialization_roundtrip() -> None:
    f = Fact(
        text="user prefers espresso",
        scope=Scope(org_id="acme", user_id="alice"),
        valid_from=_now(),
        category="preference",
        event_date=_now(),
    )
    data = f.model_dump()
    f2 = Fact.model_validate(data)
    assert f == f2


# ── ExtractedFact ────────────────────────────────────────────────────


def test_extracted_fact_default_polarity() -> None:
    ef = ExtractedFact(text="user likes coffee", confidence=0.9)
    assert ef.polarity == Polarity.AFFIRMATIVE
    assert ef.event_date is None
    assert ef.entities == []


def test_extracted_fact_negative_polarity() -> None:
    ef = ExtractedFact(
        text="user does not like decaf",
        confidence=0.95,
        polarity=Polarity.NEGATIVE,
    )
    assert ef.polarity == Polarity.NEGATIVE


# ── ChatMessage ──────────────────────────────────────────────────────


def test_chat_message_minimal() -> None:
    m = ChatMessage(
        scope=Scope(org_id="acme", user_id="alice"),
        session_id="sess-1",
        role="user",
        content="hello",
        timestamp=_now(),
    )
    assert m.role == "user"
    assert m.session_id == "sess-1"


def test_chat_message_role_validation() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(
            scope=Scope(),
            session_id="s",
            role="invalid",  # type: ignore[arg-type]
            content="x",
            timestamp=_now(),
        )


# ── Entity / Relationship ────────────────────────────────────────────


def test_entity_minimal() -> None:
    e = Entity(name="Alice", scope=Scope())
    assert e.name == "Alice"
    assert e.aliases == []


def test_relationship_minimal() -> None:
    r = Relationship(source="alice", relation="prefers", target="espresso")
    assert r.source == "alice"
    assert r.relation == "prefers"
