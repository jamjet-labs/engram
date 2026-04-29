"""Resolve relative date expressions in fact text to absolute ISO-8601 datetimes.

The resolver is the *fallback* path — if extraction (Phase 3) successfully fills
`event_date`, we trust it. If `event_date` is null, we try to recover an
absolute date by scanning fact text for relative expressions ("yesterday",
"3 weeks ago", "last Tuesday") and resolving against an anchor.

This addresses the Phase 8 gap: temporal-reasoning failures on LongMemEval
where the extraction model leaves `event_date=null` for resolvable phrases.

Examples
--------
    >>> r = RelativeDateResolver(anchor=datetime(2024, 3, 12, tzinfo=UTC))
    >>> r.resolve("user went to the gym yesterday")
    datetime(2024, 3, 11, 0, 0, tzinfo=UTC)
    >>> r.resolve("they had dinner last Tuesday")
    datetime(2024, 3, 5, 0, 0, tzinfo=UTC)
    >>> r.resolve("user's birthday is in 3 weeks")
    datetime(2024, 4, 2, 0, 0, tzinfo=UTC)
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import NamedTuple

from dateutil.relativedelta import relativedelta

from engram.models import ExtractedFact

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Relative-offset patterns. Order matters: more specific first.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), "tomorrow"),
    (
        re.compile(
            r"\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.IGNORECASE,
        ),
        "last_weekday",
    ),
    (
        re.compile(
            r"\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.IGNORECASE,
        ),
        "next_weekday",
    ),
    (re.compile(r"\blast week\b", re.IGNORECASE), "last_week"),
    (re.compile(r"\blast month\b", re.IGNORECASE), "last_month"),
    (re.compile(r"\blast year\b", re.IGNORECASE), "last_year"),
    (re.compile(r"\bnext week\b", re.IGNORECASE), "next_week"),
    (re.compile(r"\bnext month\b", re.IGNORECASE), "next_month"),
    (re.compile(r"\bnext year\b", re.IGNORECASE), "next_year"),
    (
        re.compile(
            r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\s+ago\b",
            re.IGNORECASE,
        ),
        "n_units_ago",
    ),
    (
        re.compile(
            r"\bin\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)\b",
            re.IGNORECASE,
        ),
        "in_n_units",
    ),
]


class _Match(NamedTuple):
    raw: str
    kind: str
    groups: tuple[str, ...]


class RelativeDateResolver:
    """Resolve relative date expressions against an anchor datetime.

    Anchor is typically the conversation's session date (haystack_dates entry
    in LongMemEval terms). All returned datetimes are tz-aware (UTC) at midnight.
    """

    def __init__(self, anchor: datetime) -> None:
        if anchor.tzinfo is None:
            raise ValueError("anchor must be timezone-aware")
        # Preserve the caller's timezone so equality comparisons are predictable.
        self._anchor = anchor

    @property
    def anchor(self) -> datetime:
        return self._anchor

    def resolve(self, text: str) -> datetime | None:
        """Return the absolute datetime implied by the first relative expression
        in `text`, or None if none matched."""
        match = self._first_match(text)
        if match is None:
            return None
        return self._compute(match)

    @staticmethod
    def _first_match(text: str) -> _Match | None:
        # Look for the earliest match across all patterns
        best: tuple[int, _Match] | None = None
        for pat, kind in _PATTERNS:
            m = pat.search(text)
            if m is None:
                continue
            start = m.start()
            if best is None or start < best[0]:
                best = (start, _Match(raw=m.group(0), kind=kind, groups=m.groups()))
        return best[1] if best else None

    def _compute(self, m: _Match) -> datetime:
        anchor = self._anchor
        kind = m.kind
        if kind == "yesterday":
            return anchor - timedelta(days=1)
        if kind == "today":
            return anchor
        if kind == "tomorrow":
            return anchor + timedelta(days=1)
        if kind in ("last_weekday", "next_weekday"):
            wd = _WEEKDAYS[m.groups[0].lower()]
            cur = anchor.weekday()
            if kind == "last_weekday":
                # Walk back: distance from cur to wd modulo 7, never zero (use 7)
                delta = (cur - wd) % 7 or 7
                return anchor - timedelta(days=delta)
            else:
                delta = (wd - cur) % 7 or 7
                return anchor + timedelta(days=delta)
        if kind == "last_week":
            return anchor - timedelta(days=7)
        if kind == "last_month":
            return anchor - relativedelta(months=1)
        if kind == "last_year":
            return anchor - relativedelta(years=1)
        if kind == "next_week":
            return anchor + timedelta(days=7)
        if kind == "next_month":
            return anchor + relativedelta(months=1)
        if kind == "next_year":
            return anchor + relativedelta(years=1)
        if kind == "n_units_ago":
            n = int(m.groups[0])
            unit = m.groups[1].lower().rstrip("s")
            return _shift(anchor, -n, unit)
        if kind == "in_n_units":
            n = int(m.groups[0])
            unit = m.groups[1].lower().rstrip("s")
            return _shift(anchor, n, unit)
        return anchor


def _shift(d: datetime, n: int, unit: str) -> datetime:
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    if unit == "month":
        return d + relativedelta(months=n)
    if unit == "year":
        return d + relativedelta(years=n)
    return d


def resolve_relative_dates(facts: list[ExtractedFact], anchor: datetime) -> list[ExtractedFact]:
    """Fill `event_date` on facts that don't have one but reference relative dates.

    Returns a new list (does not mutate input). Facts with an existing event_date
    are passed through unchanged.
    """
    resolver = RelativeDateResolver(anchor)
    out: list[ExtractedFact] = []
    for f in facts:
        if f.event_date is not None:
            out.append(f)
            continue
        resolved = resolver.resolve(f.text)
        if resolved is not None:
            out.append(f.model_copy(update={"event_date": resolved}))
        else:
            out.append(f)
    return out
