"""Query decomposer: split multi-part questions into atomic sub-questions."""

from __future__ import annotations

import json
import logging

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.read.prompts import DECOMPOSER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
