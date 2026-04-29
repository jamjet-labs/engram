"""Cross-encoder reranker tests.

Skipped if `sentence-transformers` is not installed (it's an optional dep).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from engram.models import Fact
from engram.retrieve.base import ScoredFact
from engram.scope import Scope


def _have_st() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


def _mk(text: str, score: float = 0.5) -> ScoredFact:
    return ScoredFact(
        fact=Fact(
            id=uuid4(),
            text=text,
            scope=Scope(),
            valid_from=_now(),
        ),
        score=score,
    )


@pytest.mark.skipif(not _have_st(), reason="sentence-transformers not installed")
async def test_cross_encoder_rerank_promotes_relevant_above_topical_neighbor() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scored = [
        _mk("the moon is made of cheese", score=0.9),  # high prior, low relevance
        _mk("paris is the capital of france", score=0.5),  # low prior, high relevance
    ]
    out = await reranker.rerank("what is the capital of france?", scored, top_k=2)
    assert "paris" in out[0].fact.text.lower()


@pytest.mark.skipif(not _have_st(), reason="sentence-transformers not installed")
async def test_cross_encoder_empty_input() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    out = await reranker.rerank("query", [], top_k=10)
    assert out == []


@pytest.mark.skipif(not _have_st(), reason="sentence-transformers not installed")
async def test_cross_encoder_top_k_truncates() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scored = [_mk(f"fact number {i}") for i in range(5)]
    out = await reranker.rerank("query", scored, top_k=2)
    assert len(out) == 2


@pytest.mark.skipif(not _have_st(), reason="sentence-transformers not installed")
async def test_cross_encoder_writes_rerank_score() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scored = [_mk("fact a"), _mk("fact b")]
    out = await reranker.rerank("query", scored, top_k=2)
    for sf in out:
        # CE scores are unbounded floats; just check it was set
        assert sf.rerank_score != 0.0
