"""ExtractionPipeline — orchestrates LLM extraction over chat messages."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from engram.errors import ExtractionError
from engram.extract.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
)
from engram.llm.base import LLMClient, LLMMessage
from engram.models import ChatMessage, ExtractedFact
from engram.temporal.resolver import resolve_relative_dates

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Run an LLM over chat messages to produce `ExtractedFact`s.

    The pipeline:
      1. Formats messages with a system + user prompt
      2. Calls the LLM in JSON mode
      3. Parses the JSON response
      4. Validates each fact against `ExtractedFact` (drops malformed entries)
      5. Stamps `mention_date` on each fact from `session_date`
    """

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries

    async def extract(
        self,
        messages: list[ChatMessage],
        session_date: datetime | None = None,
    ) -> list[ExtractedFact]:
        if not messages:
            return []

        turns = [{"role": m.role, "content": m.content} for m in messages]
        user_prompt = build_extraction_user_prompt(
            turns,
            session_date.date().isoformat() if session_date else None,
        )

        prompt_messages = [
            LLMMessage(role="system", content=EXTRACTION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._llm.generate(prompt_messages, temperature=0.0, json_mode=True)
                parsed = _parse_json_response(resp.content)
                facts = _coerce_facts(parsed, mention_date=session_date)
                # Phase 8: fallback resolver for relative dates the LLM left as null.
                if session_date is not None:
                    facts = resolve_relative_dates(facts, anchor=session_date)
                return facts
            except (ExtractionError, json.JSONDecodeError, ValidationError, ValueError) as e:
                last_err = e
                logger.warning(
                    "extraction attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
        # All retries exhausted; raise structured error
        raise ExtractionError(f"extraction failed after retries: {last_err}")


def _parse_json_response(content: str) -> dict[str, Any]:
    """Tolerant JSON parse — strip markdown fences if a model couldn't help itself."""
    text = content.strip()
    # Strip common ```json fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object, got {type(obj).__name__}")
    return obj


def _coerce_facts(parsed: dict[str, Any], mention_date: datetime | None) -> list[ExtractedFact]:
    """Turn the LLM JSON dict into a validated list of ExtractedFact.

    Drops malformed entries instead of failing the whole batch — the cost of
    retrying for one bad fact in twenty is worse than dropping the bad one.
    """
    raw_facts = parsed.get("facts", [])
    if not isinstance(raw_facts, list):
        return []
    out: list[ExtractedFact] = []
    for entry in raw_facts:
        if not isinstance(entry, dict):
            continue
        # Normalize: accept "polarity" as missing/None -> default
        if entry.get("polarity") is None:
            entry.pop("polarity", None)
        # Normalize: ensure "entities" is a list
        ent = entry.get("entities", [])
        if not isinstance(ent, list):
            entry["entities"] = []
        try:
            fact = ExtractedFact.model_validate(entry)
        except ValidationError as e:
            logger.debug("dropping malformed fact: %s | error: %s", entry, e)
            continue
        if mention_date is not None and fact.mention_date is None:
            fact = fact.model_copy(update={"mention_date": mention_date})
        out.append(fact)
    return out
