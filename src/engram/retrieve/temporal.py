"""Lightweight temporal intent detector (regex-based, no LLM).

Ported from Engram Rust v0.6 `temporal_parser.rs`. Phase 8 will expand this
into ingestion-time relative-date resolution; this module just classifies
the *type* of temporal question.
"""

from __future__ import annotations

import re
from enum import StrEnum

_DURATION_RE = re.compile(
    r"\b(how many|how long|duration|number of days|"
    r"days between|weeks between|months between)\b",
    re.IGNORECASE,
)
_RECENCY_RE = re.compile(
    r"\b(recent|recently|last (week|month|year|day|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|monday)|yesterday|today|"
    r"this (week|month|year|morning|afternoon|evening)|ago|"
    r"when did .* last|did .* last)\b",
    re.IGNORECASE,
)
_ORDERING_RE = re.compile(
    r"\b(before|after|first|last|earliest|latest|then|later)\b",
    re.IGNORECASE,
)
_POINT_IN_TIME_RE = re.compile(
    r"\b(when|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in (january|february|march|april|may|june|july|august|"
    r"september|october|november|december))\b",
    re.IGNORECASE,
)


class TemporalIntent(StrEnum):
    DURATION = "duration"  # how many days/weeks
    RECENCY = "recency"  # recent / last X / yesterday
    ORDERING = "ordering"  # before / after
    POINT_IN_TIME = "point_in_time"  # when / on Monday


def detect_temporal_intent(query: str) -> TemporalIntent | None:
    """Return the strongest matching temporal intent, or None.

    RECENCY is checked before DURATION because phrases like "how many days ago"
    contain both signals; the recency interpretation is correct
    (asking about elapsed time from now, not duration between two events).
    """
    if _RECENCY_RE.search(query):
        return TemporalIntent.RECENCY
    if _DURATION_RE.search(query):
        return TemporalIntent.DURATION
    if _ORDERING_RE.search(query):
        return TemporalIntent.ORDERING
    if _POINT_IN_TIME_RE.search(query):
        return TemporalIntent.POINT_IN_TIME
    return None
