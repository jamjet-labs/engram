"""Reciprocal Rank Fusion — combine multiple ranked lists into one.

Standard RRF: score(d) = sum over each list of 1 / (k + rank(d, list)).
k=60 is the canonical paper constant.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(ranked_lists: list[list[UUID]], k: int = 60) -> list[UUID]:
    if not ranked_lists:
        return []
    scores: dict[UUID, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] += 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda p: p[1], reverse=True)]
