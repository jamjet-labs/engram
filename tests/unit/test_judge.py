from unittest.mock import AsyncMock

import pytest

from benchmarks.judge import JudgeResult, judge_one


@pytest.mark.asyncio
async def test_judge_returns_correct_on_match():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "yes"
    res = await judge_one(
        question="Q?",
        expected="42",
        predicted="forty-two (42)",
        category="single-session-user",
        llm=fake_llm,
    )
    assert isinstance(res, JudgeResult)
    assert res.correct is True


@pytest.mark.asyncio
async def test_judge_returns_incorrect_on_no():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "no"
    res = await judge_one(
        question="Q?",
        expected="42",
        predicted="something else",
        category="single-session-user",
        llm=fake_llm,
    )
    assert res.correct is False


@pytest.mark.asyncio
async def test_judge_handles_messy_response():
    """Judge should still extract yes/no from leading whitespace, casing, trailing punctuation."""
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "  Yes.  "
    res = await judge_one(
        question="Q?", expected="42", predicted="42", category="x", llm=fake_llm,
    )
    assert res.correct is True
