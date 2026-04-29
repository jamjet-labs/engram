"""Reader: answer a question from retrieved context with verifier-backed abstention."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.read.prompts import READER_SYSTEM_PROMPT, TODAY_CLAUSE, VERIFIER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

Verdict = Literal["YES", "NO", "PARTIAL"]


class ReadResult(BaseModel):
    """Reader output — answer + verifier metadata."""

    answer: str
    verdict: Verdict | None = None
    missing: str | None = None
    abstained: bool = False
    decomposed_subqueries: list[str] = Field(default_factory=list)


class Reader:
    """Reading layer with optional pre-verification.

    Pipeline:
      1. (Optional) verifier checks if facts can answer the question
      2. If verifier says NO, return abstention
      3. Otherwise the reader generates an answer

    Set `verifier=False` to skip the verifier (saves one LLM call).
    """

    def __init__(self, llm: LLMClient, verifier: bool = True) -> None:
        self._llm = llm
        self._verifier_enabled = verifier

    async def read(
        self,
        question: str,
        context: str,
        today: datetime | None = None,
    ) -> ReadResult:
        """Answer `question` from `context`. Pass `today` to anchor temporal queries."""
        today_clause = TODAY_CLAUSE.format(today=today.date().isoformat()) if today else ""

        verdict: Verdict | None = None
        missing: str | None = None
        if self._verifier_enabled:
            verdict, missing = await self._verify(question, context)
            if verdict == "NO":
                return ReadResult(
                    answer="I don't know",
                    verdict=verdict,
                    missing=missing,
                    abstained=True,
                )

        # Generate answer
        try:
            resp = await self._llm.generate(
                [
                    LLMMessage(
                        role="system",
                        content=READER_SYSTEM_PROMPT.format(
                            today_clause=today_clause,
                            context=context,
                            question=question,
                        ),
                    ),
                    LLMMessage(role="user", content=question),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        except ExtractionError as e:
            raise ExtractionError(f"reader generate failed: {e}") from e
        answer = resp.content.strip()
        abstained = answer.lower().startswith("i don't know")
        return ReadResult(answer=answer, verdict=verdict, missing=missing, abstained=abstained)

    async def _verify(self, question: str, context: str) -> tuple[Verdict, str | None]:
        """Run the verifier.

        Returns (verdict, missing). On parse failure, returns ('PARTIAL', None).
        """
        try:
            resp = await self._llm.generate(
                [
                    LLMMessage(
                        role="system",
                        content=VERIFIER_SYSTEM_PROMPT.format(context=context, question=question),
                    ),
                    LLMMessage(role="user", content=question),
                ],
                temperature=0.0,
                json_mode=True,
                max_tokens=120,
            )
            data = json.loads(resp.content.strip())
            v = data.get("verdict", "PARTIAL").upper()
            if v not in ("YES", "NO", "PARTIAL"):
                v = "PARTIAL"
            missing = data.get("missing")
            if missing is not None and not isinstance(missing, str):
                missing = None
            # `v` is one of "YES"/"NO"/"PARTIAL" by virtue of the check above.
            return (v, missing)
        except (ExtractionError, json.JSONDecodeError, ValueError) as e:
            logger.warning("verifier parse failed (%s); defaulting to PARTIAL", e)
            return ("PARTIAL", None)


def format_context_with_confidence(
    facts_with_scores: list[tuple[str, float]],
    event_dates: list[str | None] | None = None,
) -> str:
    """Format a context string with [confidence] tags inline.

    facts_with_scores: list of (fact_text, confidence) tuples.
    event_dates: optional parallel list of ISO date strings; included as [YYYY-MM-DD] prefix.

    The output is what the reader prompt expects.
    """
    lines = []
    n = len(facts_with_scores)
    dates = event_dates or [None] * n
    for (text, conf), date in zip(facts_with_scores, dates, strict=False):
        prefix = f"[{date}] " if date else ""
        lines.append(f"- {prefix}{text} [confidence: {conf:.2f}]")
    return "\n".join(lines)
