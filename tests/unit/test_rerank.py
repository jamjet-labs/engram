"""Cross-encoder reranker tests.

Skipped if `sentence-transformers` is not installed (it's an optional dep), or
if the model is not in the local Hugging Face cache: tests never download it
(the network guard in tests/network_guard.py would refuse), and the hub runs offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from engram.models import Fact
from engram.retrieve.base import ScoredFact
from engram.scope import Scope

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _have_st() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _have_model() -> bool:
    if not _have_st():
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)


needs_model = pytest.mark.skipif(
    not _have_model(),
    reason=f"sentence-transformers or {MODEL} in the local Hugging Face cache is missing",
)


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the model from the local cache, with no request to the hub."""
    if _have_st():
        import huggingface_hub.constants

        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)


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


@needs_model
async def test_cross_encoder_rerank_promotes_relevant_above_topical_neighbor() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scored = [
        _mk("the moon is made of cheese", score=0.9),  # high prior, low relevance
        _mk("paris is the capital of france", score=0.5),  # low prior, high relevance
    ]
    out = await reranker.rerank("what is the capital of france?", scored, top_k=2)
    assert "paris" in out[0].fact.text.lower()


@needs_model
async def test_cross_encoder_empty_input() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    out = await reranker.rerank("query", [], top_k=10)
    assert out == []


@needs_model
async def test_cross_encoder_top_k_truncates() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scored = [_mk(f"fact number {i}") for i in range(5)]
    out = await reranker.rerank("query", scored, top_k=2)
    assert len(out) == 2


@needs_model
async def test_cross_encoder_writes_rerank_score() -> None:
    from engram.retrieve.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    scored = [_mk("fact a"), _mk("fact b")]
    out = await reranker.rerank("query", scored, top_k=2)
    for sf in out:
        # CE scores are unbounded floats; just check it was set
        assert sf.rerank_score != 0.0
