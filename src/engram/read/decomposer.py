"""Query decomposer: split multi-part questions into atomic sub-questions."""

from __future__ import annotations

import json
import logging
import re

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.read.prompts import DECOMPOSER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_CONJUNCTION_RE = re.compile(
    r"\b(and|or|both|either|while|whereas|along with|as well as)\b",
    re.IGNORECASE,
)


def should_decompose(question: str) -> bool:
    """Cheap heuristic — return True if question is plausibly compound.

    Avoids unnecessary LLM calls when a question is clearly atomic.
    Compound when: more than one '?', OR ≥10 words AND a coordinating conjunction.
    The 10-word floor is calibrated against LongMemEval, where most compound
    questions are 10-15 words; 12+ would miss too many real cases.
    """
    if question.count("?") > 1:
        return True
    if len(question.split()) >= 10 and _CONJUNCTION_RE.search(question):
        return True
    return False


class QueryDecomposer:
    """Split a complex question into atomic sub-questions via an LLM."""

    def __init__(self, llm: LLMClient, max_subqueries: int = 4) -> None:
        self._llm = llm
        self._max = max_subqueries

    async def decompose(self, question: str) -> list[str]:
        """Return [question] for atomic queries, [sub1, sub2, ...] for compound."""
        try:
            resp = await self._llm.generate(
                [
                    LLMMessage(role="system", content=DECOMPOSER_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=question),
                ],
                temperature=0.0,
                json_mode=True,
                max_tokens=300,
            )
            data = json.loads(resp.content.strip())
            subs = data.get("sub_questions", [])
            if not isinstance(subs, list):
                return [question]
            cleaned = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
            if not cleaned:
                return [question]
            return cleaned[: self._max]
        except (ExtractionError, json.JSONDecodeError, ValueError) as e:
            logger.warning("decomposer parse failed (%s); using original query", e)
            return [question]
