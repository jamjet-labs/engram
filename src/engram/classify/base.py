"""Question classifier protocol + the canonical category set + per-category budgets.

The categories mirror LongMemEval's question-type taxonomy. Token budgets are
calibrated from the AgentMemory analysis (2026-04-29):

- single-session-user      1,500 tokens
- single-session-assistant 3,500
- single-session-preference 3,500
- temporal-reasoning       5,000
- multi-session            7,500
- knowledge-update         2,500
"""

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum
from typing import Protocol, runtime_checkable


class QuestionType(StrEnum):
    SINGLE_SESSION_USER = "single-session-user"
    SINGLE_SESSION_ASSISTANT = "single-session-assistant"
    SINGLE_SESSION_PREFERENCE = "single-session-preference"
    TEMPORAL_REASONING = "temporal-reasoning"
    MULTI_SESSION = "multi-session"
    KNOWLEDGE_UPDATE = "knowledge-update"


# AgentMemory's per-category token budgets (2026-04-29 leaderboard analysis).
CATEGORY_BUDGETS: dict[QuestionType, int] = {
    QuestionType.SINGLE_SESSION_USER: 1500,
    QuestionType.SINGLE_SESSION_ASSISTANT: 3500,
    QuestionType.SINGLE_SESSION_PREFERENCE: 3500,
    QuestionType.TEMPORAL_REASONING: 5000,
    QuestionType.MULTI_SESSION: 7500,
    QuestionType.KNOWLEDGE_UPDATE: 2500,
}

# Default fallback when classification confidence is low or category unknown.
DEFAULT_BUDGET = 2500


@runtime_checkable
class QuestionClassifier(Protocol):
    """Predict which LongMemEval-style category a query belongs to."""

    @abstractmethod
    async def classify(self, query: str) -> QuestionType: ...


def budget_for(qt: QuestionType | None) -> int:
    """Token budget for a category. Falls back to `DEFAULT_BUDGET` for None."""
    if qt is None:
        return DEFAULT_BUDGET
    return CATEGORY_BUDGETS.get(qt, DEFAULT_BUDGET)
