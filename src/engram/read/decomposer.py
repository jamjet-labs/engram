"""Query decomposer: split multi-part questions into atomic sub-questions."""

from __future__ import annotations

import json
import logging
import re

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.read.prompts import DECOMPOSER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# 'and' is the only conjunction we trust as a real compound signal in
# LongMemEval. 'or' usually marks a comparison ("X or Y?") that should NOT
# be split — splitting loses the comparison itself.
_AND_RE = re.compile(r"\band\b", re.IGNORECASE)

# Temporal / ordering / comparative markers — questions containing these
# should NOT be decomposed because:
#   - "first/last/before/after/between" mark ordering/comparison logic that
#     atomic subqueries can't reproduce
#   - "how long/many days/months" expect a single numeric/durational answer,
#     not a fan-out
#   - "when did" is a single-fact temporal query
# Empirically (Phase B diagnostic), removing decomposition on these recovers
# the -6pp temporal-reasoning regression seen in the unconstrained version.
_TEMPORAL_SKIP_RE = re.compile(
    r"\b("
    r"first|last|earliest|latest|"
    r"before|after|between|since|ago|prior\s+to|"
    r"when\s+did|when\s+was|what\s+was\s+the\s+date|"
    r"how\s+long|how\s+many\s+(days|weeks|months|years|hours|minutes|times)|"
    r"how\s+often"
    r")\b",
    re.IGNORECASE,
)


def should_decompose(question: str) -> bool:
    """Cheap heuristic — return True if question is plausibly compound.

    Avoids unnecessary LLM calls when a question is clearly atomic, and
    avoids breaking temporal/ordering questions that decompose poorly.

    Compound when: more than one '?', OR ≥10 words AND contains 'and' AND
    contains no temporal/ordering markers.
    """
    if _TEMPORAL_SKIP_RE.search(question):
        return False
    if question.count("?") > 1:
        return True
    if len(question.split()) >= 10 and _AND_RE.search(question):
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
