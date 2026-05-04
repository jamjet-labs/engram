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

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from engram.llm.base import LLMMessage
from engram.llm.tier import ModelTier
from engram.scope import Scope
from engram.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

_TOOL_USE_RE = re.compile(r"\[TOOL_USE\](\{.*?\})\[/TOOL_USE\]", re.DOTALL)

REACT_SYSTEM = """\
You are an agent that answers questions by calling tools to query the user's memory.

Available tools:
{tool_list}

To call a tool, output exactly:
[TOOL_USE]{{"name": "<tool name>", "input": {{...}}}}[/TOOL_USE]

When you have the final answer, call:
[TOOL_USE]{{"name": "final_answer", "input": {{"answer": "<answer>"}}}}[/TOOL_USE]

Be efficient. Each tool call is expensive. Do not repeat the same call with the same input.
Today's date: {today}

Question: {question}"""


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
        today_str = today.date().isoformat() if today else "unknown"
        tool_list = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self._tools.for_anthropic()
        )
        sys = REACT_SYSTEM.format(
            tool_list=tool_list, today=today_str, question=question
        )
        history: list[LLMMessage] = [
            LLMMessage(role="system", content=sys),
            LLMMessage(role="user", content=question),
        ]
        trace: list[AgentStep] = []
        last_call_signature: str | None = None
        started = time.time()

        for hop in range(1, self._max_hops + 1):
            if time.time() - started > self._total_budget_s:
                break
            t0 = time.time()
            try:
                resp = await asyncio.wait_for(
                    self._tier.utility.generate(
                        history, temperature=0.0, max_tokens=400
                    ),
                    timeout=self._hop_timeout_s,
                )
            except TimeoutError:  # asyncio.TimeoutError is now a builtin alias
                trace.append(
                    AgentStep(
                        hop=hop,
                        elapsed_s=self._hop_timeout_s,
                        text_emitted="(hop timeout)",
                    )
                )
                break

            text = resp.content.strip()
            m = _TOOL_USE_RE.search(text)
            if not m:
                # Free-text answer — accept it as the final answer.
                trace.append(
                    AgentStep(hop=hop, text_emitted=text, elapsed_s=time.time() - t0)
                )
                return AgentResult(
                    answer=text, abstained=False, trace=trace, n_hops=hop
                )

            try:
                call = json.loads(m.group(1))
                tool_name = call["name"]
                tool_input = call.get("input", {})
            except (KeyError, json.JSONDecodeError):
                trace.append(
                    AgentStep(hop=hop, text_emitted=text, elapsed_s=time.time() - t0)
                )
                return AgentResult(
                    answer="(malformed tool call)",
                    abstained=True,
                    trace=trace,
                    n_hops=hop,
                )

            sig = f"{tool_name}|{json.dumps(tool_input, sort_keys=True)}"
            if sig == last_call_signature:
                trace.append(
                    AgentStep(
                        hop=hop,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        text_emitted="(loop detected — same call)",
                        elapsed_s=time.time() - t0,
                    )
                )
                break
            last_call_signature = sig

            if tool_name == "final_answer":
                ans = tool_input.get("answer", "")
                trace.append(
                    AgentStep(
                        hop=hop,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_result=ans,
                        elapsed_s=time.time() - t0,
                    )
                )
                return AgentResult(
                    answer=ans, abstained=False, trace=trace, n_hops=hop
                )

            try:
                result = await self._tools.dispatch(tool_name, tool_input)
                tool_out = result.content
            except KeyError:
                tool_out = f"(unknown tool: {tool_name})"
            except Exception as e:
                # Broad catch is intentional — any tool exception must surface
                # as text the agent can react to, never crash the run.
                tool_out = f"(tool error: {e})"

            trace.append(
                AgentStep(
                    hop=hop,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_result=tool_out,
                    elapsed_s=time.time() - t0,
                )
            )
            history.append(LLMMessage(role="assistant", content=text))
            history.append(
                LLMMessage(
                    role="user",
                    content=f"[TOOL_RESULT]{tool_out}[/TOOL_RESULT]",
                )
            )

        return AgentResult(
            answer="I don't know", abstained=True, trace=trace, n_hops=len(trace)
        )
