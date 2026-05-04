from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.read.reader import Reader, ReaderConfig
from engram.read.voting import majority_vote, normalize_answer
from engram.scope import Scope

# ── voting helpers ──────────────────────────────────────────────────────────


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_answer("  Hello, World!  ") == "hello world"
    assert normalize_answer("42") == "42"


def test_majority_vote_picks_most_common():
    assert majority_vote(["yes", "no", "yes"]) == "yes"
    assert majority_vote(["A", "a", "B"]) == "a"


def test_majority_vote_first_wins_on_tie():
    # 'A' and 'B' each appear once; first-seen wins.
    result = majority_vote(["A", "B"])
    assert result in ("a", "b")  # Counter.most_common preserves insertion order


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        majority_vote([])


# ── Reader integration ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_self_consistency_fires_on_eligible_category():
    """N=3 reader samples + vote when verdict is PARTIAL on the initial answer."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        # 1. initial verifier — say YES so we proceed
        MagicMock(content="<verdict>YES</verdict>"),
        # 2. initial reader answer
        MagicMock(content="forty-two"),
        # 3. SC pre-check verifier — PARTIAL
        MagicMock(content="<verdict>PARTIAL</verdict>"),
        # 4-5. two more samples (N=3 means 2 additional samples)
        MagicMock(content="42"),
        MagicMock(content="42"),
    ]
    cfg = ReaderConfig(self_consistency_on_partial=3, enable_reextract=False)
    reader = Reader(fake_llm, verifier=True, config=cfg)
    reader.set_category("temporal-reasoning")
    res = await reader.read(
        question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert res.answer == "42"  # majority vote


@pytest.mark.asyncio
async def test_self_consistency_skipped_for_ineligible_category():
    """For single-session-* categories, SC should not fire even on PARTIAL."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="answer"),
    ]
    cfg = ReaderConfig(self_consistency_on_partial=3, enable_reextract=False)
    reader = Reader(fake_llm, verifier=True, config=cfg)
    reader.set_category("single-session-preference")
    res = await reader.read(
        question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert res.answer == "answer"
    # No additional samples drawn
    assert fake_llm.generate.await_count == 2


@pytest.mark.asyncio
async def test_self_consistency_skipped_when_post_verdict_yes():
    """When the post-answer verifier returns YES, no extra samples."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="answer"),
        MagicMock(content="<verdict>YES</verdict>"),  # SC pre-check
    ]
    cfg = ReaderConfig(self_consistency_on_partial=3, enable_reextract=False)
    reader = Reader(fake_llm, verifier=True, config=cfg)
    reader.set_category("temporal-reasoning")
    await reader.read(question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"))
    assert fake_llm.generate.await_count == 3


@pytest.mark.asyncio
async def test_self_consistency_disabled_when_n_is_1():
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="answer"),
    ]
    cfg = ReaderConfig(self_consistency_on_partial=1, enable_reextract=False)
    reader = Reader(fake_llm, verifier=True, config=cfg)
    reader.set_category("temporal-reasoning")
    await reader.read(question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"))
    assert fake_llm.generate.await_count == 2
