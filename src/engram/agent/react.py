"""ReAct retrieval agent over the SVO event calendar (item 7 / Phase 11b).

Brain: utility-tier LLM (gpt-4o-mini by default — same provider as the tools
text-protocol). Reuses the shared ``ToolRegistry`` from item 4 — same tool
implementations work in both modes.

Termination rules (per spec §6.3):
  - Agent emits ``final_answer`` tool call
  - n_hops ≥ max_hops (default 4)
  - Wall-clock elapsed ≥ total_budget_s (default 60)
  - Same tool called with identical args twice in a row (loop detection)
  - Free-text answer emitted alongside or instead of tool calls (accept it)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from engram.llm.tier import ModelTier
from engram.scope import Scope
from engram.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class AgentStep(BaseModel):
    """One hop in the ReAct trace — a tool call or a free-text emission."""

    hop: int
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: str | None = None
    text_emitted: str | None = None
    elapsed_s: float = 0.0


class AgentResult(BaseModel):
    """Outcome of a ReAct run — answer + trace + termination metadata."""

    answer: str
    abstained: bool
    trace: list[AgentStep] = Field(default_factory=list)
    n_hops: int = 0


class ReActAgent:
    """LLM brain + tools loop. Same TOOL_USE/TOOL_RESULT text protocol as the
    tool-augmented Reader.
    """

    def __init__(
        self,
        tier: ModelTier,
        tools: ToolRegistry,
        max_hops: int = 4,
        hop_timeout_s: float = 15.0,
        total_budget_s: float = 60.0,
    ) -> None:
        self._tier = tier
        self._tools = tools
        self._max_hops = max_hops
        self._hop_timeout_s = hop_timeout_s
        self._total_budget_s = total_budget_s

    async def answer(
        self, question: str, scope: Scope, today: datetime | None = None
    ) -> AgentResult:
        # Implemented in 7.2
        raise NotImplementedError("implemented in 7.2")
