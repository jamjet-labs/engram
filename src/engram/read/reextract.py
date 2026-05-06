"""Query-time conditioned re-extraction (item 5 / N2).

When the verifier returns PARTIAL/NO, re-extract the top candidate sessions
on-the-fly conditioned on the actual question. Output facts are ephemeral —
never persisted to the store. The re-extracted facts get fed back into the
context for one re-read attempt.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.models import Fact, MemoryTier, Polarity
from engram.scope import Scope
from engram.store.base import EngramStore

logger = logging.getLogger(__name__)


REEXTRACT_PROMPT = """You are extracting facts that answer a SPECIFIC question.

Question: {question}

Conversation:
{conversation}

Extract ONLY facts directly relevant to the question. Output strict JSON:
{{"facts": [{{"text": "...", "confidence": 0.9}}, ...]}}

If no relevant facts in this conversation: {{"facts": []}}. No markdown."""


class QueryConditionedReextractor:
    """Re-extract from candidate sessions conditioned on a specific question."""

    def __init__(self, llm: LLMClient, max_sessions: int = 3) -> None:
        self._llm = llm
        self._max_sessions = max_sessions

    async def reextract(
        self,
        question: str,
        candidate_session_ids: list[str],
        store: EngramStore,
        scope: Scope,
    ) -> list[Fact]:
        out: list[Fact] = []
        for sid in candidate_session_ids[: self._max_sessions]:
            msgs = await store.list_messages(sid, scope, limit=200)
            if not msgs:
                continue
            convo = "\n".join(f"[{m.role}] {m.content}" for m in msgs)
            prompt = REEXTRACT_PROMPT.format(question=question, conversation=convo)
            try:
                resp = await self._llm.generate(
                    [LLMMessage(role="user", content=prompt)],
                    temperature=0.0,
                    json_mode=True,
                    max_tokens=400,
                )
            except ExtractionError as e:
                logger.warning("reextract LLM failed for session %s: %s", sid, e)
                continue
            try:
                data = json.loads(resp.content.strip())
            except (json.JSONDecodeError, ValueError):
                continue
            for entry in data.get("facts", []):
                if not isinstance(entry, dict) or "text" not in entry:
                    continue
                out.append(
                    Fact(
                        id=uuid4(),
                        text=entry["text"],
                        scope=scope,
                        valid_from=datetime.now().astimezone(),
                        session_id=sid,
                        confidence=float(entry.get("confidence", 0.5)),
                        polarity=Polarity.AFFIRMATIVE,
                        tier=MemoryTier.WORKING,
                    )
                )
        return out
