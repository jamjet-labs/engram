from unittest.mock import AsyncMock

import pytest

from engram import Engram
from engram.classify.rules import RuleBasedClassifier
from engram.llm.tier import ModelTier


@pytest.mark.asyncio
async def test_context_with_decompose_off_does_one_recall(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    tier = ModelTier.default()
    async with await Engram.open(":memory:", tier=tier) as memory:
        await memory.record(text="I live in Berlin", user_id="alice")
        await memory.record(text="My brother's name is Max", user_id="alice")
        ctx = await memory.context(
            query="Where do I live and what's my brother's name?",
            user_id="alice",
            decompose=False,
            classifier=RuleBasedClassifier(),
        )
        assert "Berlin" in ctx or "Max" in ctx


@pytest.mark.asyncio
async def test_context_with_decompose_on_calls_decomposer(monkeypatch):
    """When decompose=True and the question is compound, decomposer is invoked
    and we get a fused context that surfaces facts for both sub-questions."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    tier = ModelTier.default()
    async with await Engram.open(":memory:", tier=tier) as memory:
        await memory.record(text="I live in Berlin", user_id="alice")
        await memory.record(text="My brother's name is Max", user_id="alice")
        # Inject mock decomposer to bypass real LLM call.
        mock_dec = AsyncMock()
        mock_dec.decompose = AsyncMock(
            return_value=["Where do I live?", "What's my brother's name?"]
        )
        memory._decomposer = mock_dec
        ctx = await memory.context(
            query="Where do I live and what is my brother's name today?",
            user_id="alice",
            decompose=True,
            classifier=RuleBasedClassifier(),
        )
        # Both facts should appear (RRF surfaces per-subquery top hits).
        assert "Berlin" in ctx
        assert "Max" in ctx
        mock_dec.decompose.assert_called_once()


@pytest.mark.asyncio
async def test_context_decompose_skipped_for_atomic_question(monkeypatch):
    """should_decompose gate means atomic questions skip the LLM call."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    tier = ModelTier.default()
    async with await Engram.open(":memory:", tier=tier) as memory:
        await memory.record(text="I love espresso", user_id="alice")
        mock_dec = AsyncMock()
        mock_dec.decompose = AsyncMock(return_value=["Q?"])
        memory._decomposer = mock_dec
        # Short atomic question — gate should skip decomposer
        ctx = await memory.context(
            query="What do I drink?",
            user_id="alice",
            decompose=True,
            classifier=RuleBasedClassifier(),
        )
        assert "espresso" in ctx
        mock_dec.decompose.assert_not_called()
