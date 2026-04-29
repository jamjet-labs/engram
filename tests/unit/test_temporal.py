from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.models import ExtractedFact
from engram.temporal.resolver import RelativeDateResolver, resolve_relative_dates

# A Tuesday — picked deliberately so weekday math is interesting.
ANCHOR = datetime(2024, 3, 12, 12, 0, 0, tzinfo=UTC)


def test_resolver_requires_tz_aware_anchor() -> None:
    with pytest.raises(ValueError):
        RelativeDateResolver(datetime(2024, 3, 12))  # naive


def test_yesterday() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert (
        r.resolve("user went to the gym yesterday").date()
        == datetime(2024, 3, 11, tzinfo=UTC).date()
    )


def test_today() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("user is at the office today").date() == ANCHOR.date()


def test_tomorrow() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("flight is tomorrow").date() == datetime(2024, 3, 13, tzinfo=UTC).date()


def test_last_tuesday_resolves_to_one_week_back() -> None:
    # Anchor is Tuesday (2024-03-12). "last Tuesday" -> 2024-03-05.
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("they met last Tuesday").date() == datetime(2024, 3, 5, tzinfo=UTC).date()


def test_last_friday_resolves_back_in_same_week() -> None:
    # Anchor Tuesday; "last Friday" -> previous Friday = 2024-03-08
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("dinner was last Friday").date() == datetime(2024, 3, 8, tzinfo=UTC).date()


def test_next_thursday() -> None:
    # Anchor Tuesday (Mar 12); next Thursday = Mar 14
    r = RelativeDateResolver(ANCHOR)
    assert (
        r.resolve("appointment is next Thursday").date() == datetime(2024, 3, 14, tzinfo=UTC).date()
    )


def test_n_days_ago() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert (
        r.resolve("user got the keys 5 days ago").date() == datetime(2024, 3, 7, tzinfo=UTC).date()
    )


def test_n_weeks_ago() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert (
        r.resolve("started new job 3 weeks ago").date() == datetime(2024, 2, 20, tzinfo=UTC).date()
    )


def test_n_months_ago() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("moved 2 months ago").date() == datetime(2024, 1, 12, tzinfo=UTC).date()


def test_in_n_days() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("event in 4 days").date() == datetime(2024, 3, 16, tzinfo=UTC).date()


def test_no_match_returns_none() -> None:
    r = RelativeDateResolver(ANCHOR)
    assert r.resolve("the user prefers espresso") is None


def test_first_match_wins() -> None:
    """When multiple expressions appear, the earliest in text is used."""
    r = RelativeDateResolver(ANCHOR)
    out = r.resolve("yesterday I went, but tomorrow I'll go again")
    assert out is not None
    assert out.date() == datetime(2024, 3, 11, tzinfo=UTC).date()


# ── resolve_relative_dates batch helper ─────────────────────────────


def _ef(text: str, event_date: datetime | None = None) -> ExtractedFact:
    return ExtractedFact(text=text, confidence=0.9, event_date=event_date)


def test_resolve_batch_fills_only_missing_event_date() -> None:
    explicit = datetime(2020, 1, 1, tzinfo=UTC)
    facts = [
        _ef("user went to the gym yesterday"),
        _ef("explicit date already set", event_date=explicit),
        _ef("no temporal expression at all"),
    ]
    out = resolve_relative_dates(facts, anchor=ANCHOR)
    assert out[0].event_date is not None
    assert out[0].event_date.date() == datetime(2024, 3, 11, tzinfo=UTC).date()
    assert out[1].event_date == explicit  # unchanged
    assert out[2].event_date is None  # nothing matched


def test_resolve_batch_returns_new_list() -> None:
    facts = [_ef("user went yesterday")]
    out = resolve_relative_dates(facts, anchor=ANCHOR)
    assert out is not facts
    assert facts[0].event_date is None  # input unchanged
    assert out[0].event_date is not None
