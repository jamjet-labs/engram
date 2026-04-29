"""LLM-based question classifier — calls a small model (gpt-4o-mini etc.)."""

from __future__ import annotations

import json
import logging

from engram.classify.base import QuestionType
from engram.classify.rules import RuleBasedClassifier
from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage

logger = logging.getLogger(__name__)


_CLASSIFY_SYSTEM_PROMPT = """\
You classify a single user question into exactly one category from this list:
- single-session-user
- single-session-assistant
- single-session-preference
- temporal-reasoning
- multi-session
- knowledge-update

Definitions:
- single-session-user: question about a personal fact / event / detail of the user mentioned once
- single-session-assistant: question about something the assistant said/recommended/explained
- single-session-preference: question about a user preference (likes, dislikes, favorites)
- temporal-reasoning: requires reasoning about time (durations, ordering, recency)
- multi-session: requires synthesizing facts across multiple conversations
- knowledge-update: question whose answer changed over time; the user wants the *current* state

Output STRICT JSON: {"category": "<one of the values above>"}
No explanation. No markdown.
"""


class LLMClassifier:
    """Classify a query via an LLM. Falls back to rule-based on parse failure."""

    def __init__(self, llm: LLMClient, fallback: RuleBasedClassifier | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or RuleBasedClassifier()

    async def classify(self, query: str) -> QuestionType:
        try:
            resp = await self._llm.generate(
                [
                    LLMMessage(role="system", content=_CLASSIFY_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=query),
                ],
                temperature=0.0,
                json_mode=True,
                max_tokens=40,
            )
            data = json.loads(resp.content.strip())
            cat = data.get("category", "")
            return QuestionType(cat)
        except (ExtractionError, json.JSONDecodeError, ValueError) as e:
            logger.warning("LLM classifier failed (%s); falling back to rules", e)
            return await self._fallback.classify(query)
