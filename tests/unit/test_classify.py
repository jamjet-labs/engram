"""Phase 10: question classifier + per-category budgets."""

from __future__ import annotations

import pytest

from engram.classify.base import (
    CATEGORY_BUDGETS,
    DEFAULT_BUDGET,
    QuestionType,
    budget_for,
)
from engram.classify.llm import LLMClassifier
from engram.classify.rules import RuleBasedClassifier
from engram.llm.base import LLMMessage, LLMResponse

# ── Budgets ─────────────────────────────────────────────────────────


def test_budget_for_known_categories() -> None:
    for qt, expected in CATEGORY_BUDGETS.items():
        assert budget_for(qt) == expected


def test_budget_for_none_uses_default() -> None:
    assert budget_for(None) == DEFAULT_BUDGET


def test_all_categories_have_budgets() -> None:
    for qt in QuestionType:
        assert qt in CATEGORY_BUDGETS


# ── Rule-based classifier ───────────────────────────────────────────


@pytest.fixture
def rules() -> RuleBasedClassifier:
    return RuleBasedClassifier()


@pytest.mark.parametrize(
    "query, expected",
    [
        ("how many days ago did I visit?", QuestionType.TEMPORAL_REASONING),
        ("yesterday what did I eat?", QuestionType.TEMPORAL_REASONING),
        ("did A happen before B?", QuestionType.TEMPORAL_REASONING),
        ("how many sessions did we discuss this across?", QuestionType.MULTI_SESSION),
        ("sum of all my expenses combined", QuestionType.MULTI_SESSION),
        ("what is my Instagram follower count now?", QuestionType.KNOWLEDGE_UPDATE),
        ("what is the latest version of my software?", QuestionType.KNOWLEDGE_UPDATE),
        ("what did you recommend for dinner?", QuestionType.SINGLE_SESSION_ASSISTANT),
        ("you said something about that earlier", QuestionType.SINGLE_SESSION_ASSISTANT),
        ("what is my favorite color?", QuestionType.SINGLE_SESSION_PREFERENCE),
        ("do I prefer espresso or drip?", QuestionType.SINGLE_SESSION_PREFERENCE),
        ("what is my brother's name?", QuestionType.SINGLE_SESSION_USER),
    ],
)
async def test_rule_classifier_basic(
    rules: RuleBasedClassifier, query: str, expected: QuestionType
) -> None:
    result = await rules.classify(query)
    assert result == expected


# ── LLM classifier (mocked) ─────────────────────────────────────────


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(content=self._response, finish_reason="stop")


async def test_llm_classifier_parses_valid_json() -> None:
    classifier = LLMClassifier(_FakeLLM('{"category": "temporal-reasoning"}'))
    result = await classifier.classify("how many days ago?")
    assert result == QuestionType.TEMPORAL_REASONING


async def test_llm_classifier_falls_back_on_invalid_category() -> None:
    classifier = LLMClassifier(_FakeLLM('{"category": "not-a-real-category"}'))
    # Should fall back to rule-based; "how many days ago?" -> TEMPORAL_REASONING
    result = await classifier.classify("how many days ago did this happen?")
    assert result == QuestionType.TEMPORAL_REASONING


async def test_llm_classifier_falls_back_on_invalid_json() -> None:
    classifier = LLMClassifier(_FakeLLM("not json at all"))
    result = await classifier.classify("what's my favorite color?")
    assert result == QuestionType.SINGLE_SESSION_PREFERENCE


# ── Engram.context budget integration ───────────────────────────────


async def test_engram_context_uses_classifier_budget() -> None:
    """Engram.context should consult classifier when token_budget is None."""
    from engram import Engram

    async with await Engram.open(":memory:") as memory:
        # Pile in many short facts so budget actually bounds the result
        for i in range(50):
            await memory.record(text=f"fact {i} about coffee preference", user_id="alice")
        rules = RuleBasedClassifier()
        # "favorite" -> SINGLE_SESSION_PREFERENCE -> budget 3500 tokens
        ctx = await memory.context(
            query="what is my favorite drink?", user_id="alice", classifier=rules
        )
        # Without budget kwarg + with classifier, budget defaults to 3500 -> ~14000 chars max.
        # Should fit ALL 50 short facts (each ~30 chars).
        assert len(ctx.split("\n")) >= 10


async def test_engram_context_explicit_budget_wins_over_classifier() -> None:
    """If caller supplies token_budget, classifier is bypassed."""
    from engram import Engram

    async with await Engram.open(":memory:") as memory:
        for i in range(50):
            await memory.record(text=f"fact {i} about coffee", user_id="alice")
        rules = RuleBasedClassifier()
        ctx = await memory.context(
            query="what is my favorite drink?",
            user_id="alice",
            token_budget=50,  # tiny — overrides classifier's 3500
            classifier=rules,
        )
        # 50 tokens * 4 chars = 200-char budget, much smaller than 50*30 char facts.
        assert len(ctx) < 250
