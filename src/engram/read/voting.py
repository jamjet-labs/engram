"""Self-consistency voting helpers (item 6 / #8).

Adaptive self-consistency: when the verifier returns PARTIAL on the initial
reader answer AND the question category is in the eligible set (where reader
nondeterminism most often hurts), sample N reader responses and majority-vote.
"""

from __future__ import annotations

import re
from collections import Counter

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_answer(s: str) -> str:
    """Lowercase + strip surrounding whitespace + drop punctuation.

    Used to make 'forty-two', 'forty two!', 'Forty Two' vote as the same answer.
    """
    return _PUNCT_RE.sub("", s.strip().lower())


def majority_vote(answers: list[str]) -> str:
    """Return the most common normalised answer; first-seen wins ties."""
    if not answers:
        raise ValueError("answers must be non-empty")
    norm = [normalize_answer(a) for a in answers]
    counts = Counter(norm)
    most_common, _ = counts.most_common(1)[0]
    return most_common
