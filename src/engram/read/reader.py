"""Reader: answer a question from retrieved context with verifier-backed abstention."""

from __future__ import annotations

import json as _json
import logging
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from engram.errors import ExtractionError
from engram.llm.base import LLMClient, LLMMessage
from engram.read.prompts import (
    READER_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
    TODAY_CLAUSE,
    VERIFIER_SYSTEM_PROMPT,
)
from engram.scope import Scope
from engram.tools.base import ToolRegistry, ToolResult

_VERDICT_RE = re.compile(r"<verdict>\s*(YES|NO|PARTIAL)\s*</verdict>", re.IGNORECASE)
_MISSING_RE = re.compile(r"<missing>\s*(.*?)\s*</missing>", re.IGNORECASE | re.DOTALL)
_TOOL_USE_RE = re.compile(r"\[TOOL_USE\](\{.*?\})\[/TOOL_USE\]", re.DOTALL)

_MAX_TOOL_CALLS = 5

logger = logging.getLogger(__name__)

Verdict = Literal["YES", "NO", "PARTIAL"]


class ReadResult(BaseModel):
    """Reader output — answer + verifier metadata."""

    answer: str
    verdict: Verdict | None = None
    missing: str | None = None
    abstained: bool = False
    decomposed_subqueries: list[str] = Field(default_factory=list)
    solved_by: Literal["solver", "reader", "synthesis"] | None = None


class ReaderConfig(BaseModel):
    """Optional pluggable behaviours for the Reader."""

    # Use Any to dodge a circular import (engram.solve.temporal imports nothing
    # from engram.read, but solve depends on store/llm which are heavy).
    solver: Any = None
    # Tool-augmented reading (item 4 / N5): when set, the reader can emit
    # [TOOL_USE]{name, input}[/TOOL_USE] blocks; the registry dispatches them
    # and feeds the result back as [TOOL_RESULT]…[/TOOL_RESULT] for the next turn.
    tools: ToolRegistry | None = None
    # Escalation rung (a) — query-time re-extraction (item 5 / N2)
    enable_reextract: bool = True
    # Escalation rung (b) — adaptive self-consistency (item 6 / #8). When > 1
    # AND verdict is PARTIAL AND category is eligible, sample N reader responses
    # and majority-vote. N=1 disables.
    self_consistency_on_partial: int = 1

    model_config = {"arbitrary_types_allowed": True}


# Categories where reader nondeterminism most often hurts and self-consistency
# pays off (per spec §3.2). Restricting to these cuts the cost ~50%.
ELIGIBLE_SC_CATEGORIES = frozenset({"temporal-reasoning", "multi-session", "knowledge-update"})


class Reader:
    """Reading layer with optional pre-verification + programmatic pre-pass.

    Pipeline:
      1. (Optional) Programmatic solver pre-pass — if config.solver is set AND a
         scope is provided AND the solver returns a deterministic answer, return
         it immediately (verdict="YES", solved_by="solver"). No LLM call.
      2. (Optional) verifier checks if facts can answer the question
      3. If verifier says NO, return abstention
      4. Otherwise the reader generates an answer

    Set ``verifier=False`` to skip the verifier (saves one LLM call).
    """

    def __init__(
        self,
        llm: LLMClient,
        verifier: bool = True,
        config: ReaderConfig | None = None,
        mode: Literal["recall", "synthesis"] = "recall",
    ) -> None:
        self._llm = llm
        self._verifier_enabled = verifier
        self._config = config or ReaderConfig()
        self._mode = mode
        # Escalation rung (a) — wired by attach_reextractor()
        self._reextractor: Any | None = None
        self._reextract_store: Any | None = None
        self._candidate_sessions_provider: Any | None = None
        # Escalation rung (c) — wired by attach_react()
        self._react: Any | None = None
        # Per-question category for self-consistency gating (set by caller)
        self._category: str | None = None

    def set_category(self, category: str | None) -> None:
        """Inform the reader of the question category before ``read()``.

        Used by escalation rung (b) to gate self-consistency to the categories
        where it pays off (temporal-reasoning, multi-session, knowledge-update).
        """
        self._category = category

    def attach_react(self, react_agent: Any) -> None:
        """Wire the ReAct fallback (escalation rung c).

        Fires when post-self-consistency verdict is still PARTIAL/NO. The agent
        gets a tool registry (typically the same one passed via ReaderConfig.tools)
        and tries multi-hop retrieval to find an answer.
        """
        self._react = react_agent

    def attach_reextractor(
        self,
        reextractor: Any,
        store: Any,
        candidate_sessions_provider: Any,
    ) -> None:
        """Wire the query-time re-extraction rung.

        ``candidate_sessions_provider`` is a sync callable ``str -> list[str]``
        that maps a question to the top-K candidate session IDs (typically derived
        from the recall result for that question). Passing a cached map lets the
        caller pre-compute candidates per question and inject them.
        """
        self._reextractor = reextractor
        self._reextract_store = store
        self._candidate_sessions_provider = candidate_sessions_provider

    async def read(
        self,
        question: str,
        context: str,
        today: datetime | None = None,
        scope: Scope | None = None,
    ) -> ReadResult:
        """Answer `question` from `context`. Pass `today` to anchor temporal queries.

        ``scope`` is required for the programmatic solver pre-pass; if omitted,
        the solver is skipped even when configured.

        When ``mode="synthesis"`` was passed to the constructor, this short-circuits
        to a synthesis-only branch (SYNTHESIS_PROMPT + no verifier + no tool loop +
        no escalation). Use for preference/recommendation questions where the
        judge rewards grounded synthesis over literal fact-recall.
        """
        if self._mode == "synthesis":
            return await self._read_synthesis(question, context)

        # 1. Solver pre-pass (item 3 / N1)
        if self._config.solver is not None and scope is not None:
            try:
                tq = await self._config.solver.parse(question, today=today)
                if tq is not None:
                    sr = await self._config.solver.solve(tq, scope)
                    if sr is not None:
                        return ReadResult(
                            answer=str(sr.answer),
                            verdict="YES",
                            abstained=False,
                            solved_by="solver",
                        )
            except Exception as e:  # never fail the read because the solver erred
                logger.warning("solver pre-pass failed (%s); falling through to LLM", e)

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
                    solved_by="reader",
                )

        # Generate answer — with optional tool loop
        sys_prompt = READER_SYSTEM_PROMPT.format(
            today_clause=today_clause,
            context=context,
            question=question,
        )
        if self._config.tools is not None:
            sys_prompt += (
                "\n\nYou have access to tools. To call one, output exactly:\n"
                '[TOOL_USE]{"name": "...", "input": {...}}[/TOOL_USE]\n'
                "Available tools:\n"
                + "\n".join(
                    f"- {t['name']}: {t['description']}" for t in self._config.tools.for_anthropic()
                )
            )

        history: list[LLMMessage] = [
            LLMMessage(role="system", content=sys_prompt),
            LLMMessage(role="user", content=question),
        ]
        answer = ""
        tool_calls = 0
        for _ in range(_MAX_TOOL_CALLS + 1):
            try:
                resp = await self._llm.generate(history, temperature=0.0, max_tokens=500)
            except ExtractionError as e:
                raise ExtractionError(f"reader generate failed: {e}") from e

            text = resp.content.strip()
            m = _TOOL_USE_RE.search(text) if self._config.tools is not None else None
            if not m:
                answer = text
                break

            # If the model produced text content BEFORE the tool call, treat
            # that as the final answer ("first answer wins" rule).
            before_tool = text[: m.start()].strip()
            if before_tool:
                answer = before_tool
                break

            try:
                call = _json.loads(m.group(1))
                tool_name = call["name"]
                tool_input = call.get("input", {})
            except (KeyError, _json.JSONDecodeError):
                answer = text  # malformed tool call — accept as final
                break

            # Narrow Optional[ToolRegistry] for mypy: we only reach here after
            # `m` matched, which requires self._config.tools is not None
            # (per the guard at the top of the loop body).
            assert self._config.tools is not None
            try:
                result = await self._config.tools.dispatch(tool_name, tool_input)
            except Exception as e:
                result = ToolResult(content=f"(tool {tool_name} error: {e})")

            history.append(LLMMessage(role="assistant", content=text))
            history.append(
                LLMMessage(
                    role="user",
                    content=f"[TOOL_RESULT]{result.content}[/TOOL_RESULT]",
                )
            )
            tool_calls += 1
            if tool_calls >= _MAX_TOOL_CALLS:
                # One final call to let the model produce a real answer.
                resp = await self._llm.generate(history, temperature=0.0, max_tokens=300)
                answer = resp.content.strip()
                break

        # Escalation rung (a) — query-time conditioned re-extraction (item 5 / N2)
        final_verdict: Verdict | None = verdict
        if (
            self._verifier_enabled
            and self._config.enable_reextract
            and self._reextractor is not None
            and self._candidate_sessions_provider is not None
            and self._reextract_store is not None
            and scope is not None
        ):
            post_verdict, _ = await self._verify(question, context + f"\n\nAnswer: {answer}")
            final_verdict = post_verdict
            if post_verdict in ("PARTIAL", "NO"):
                sids = self._candidate_sessions_provider(question)
                if sids:
                    try:
                        ephemeral = await self._reextractor.reextract(
                            question=question,
                            candidate_session_ids=sids,
                            store=self._reextract_store,
                            scope=scope,
                        )
                    except Exception as e:
                        logger.warning("reextract failed: %s", e)
                        ephemeral = []
                    if ephemeral:
                        extra_ctx = "\n".join(
                            f"- {f.text} [confidence: {f.confidence:.2f}]" for f in ephemeral
                        )
                        new_context = context + "\n\n[Re-extracted facts]\n" + extra_ctx
                        sys_prompt2 = READER_SYSTEM_PROMPT.format(
                            today_clause=today_clause,
                            context=new_context,
                            question=question,
                        )
                        try:
                            resp = await self._llm.generate(
                                [
                                    LLMMessage(role="system", content=sys_prompt2),
                                    LLMMessage(role="user", content=question),
                                ],
                                temperature=0.0,
                                max_tokens=300,
                            )
                            answer = resp.content.strip()
                            final_verdict = (
                                await self._verify(
                                    question,
                                    new_context + f"\n\nAnswer: {answer}",
                                )
                            )[0]
                        except ExtractionError as e:
                            logger.warning("reextract re-read failed: %s", e)

        # Escalation rung (b) — adaptive self-consistency (item 6 / #8)
        if (
            self._verifier_enabled
            and self._config.self_consistency_on_partial > 1
            and self._category in ELIGIBLE_SC_CATEGORIES
        ):
            sc_verdict, _ = await self._verify(question, context + f"\n\nAnswer: {answer}")
            if sc_verdict in ("PARTIAL", "NO"):
                from engram.read.voting import majority_vote

                n = self._config.self_consistency_on_partial
                samples: list[str] = [answer]
                sys_prompt_sc = READER_SYSTEM_PROMPT.format(
                    today_clause=today_clause,
                    context=context,
                    question=question,
                )
                for _ in range(n - 1):
                    try:
                        resp = await self._llm.generate(
                            [
                                LLMMessage(role="system", content=sys_prompt_sc),
                                LLMMessage(role="user", content=question),
                            ],
                            temperature=0.4,  # diversity
                            max_tokens=300,
                        )
                        samples.append(resp.content.strip())
                    except ExtractionError as e:
                        logger.warning("self-consistency sample failed: %s", e)
                if len(samples) > 1:
                    answer = majority_vote(samples)
                    final_verdict = sc_verdict

        # Escalation rung (c) — ReAct fallback (item 7 / Phase 11b)
        if self._verifier_enabled and self._react is not None and scope is not None:
            post_verdict, _ = await self._verify(question, context + f"\n\nAnswer: {answer}")
            if post_verdict in ("PARTIAL", "NO"):
                try:
                    agent_res = await self._react.answer(
                        question=question, scope=scope, today=today
                    )
                    if not agent_res.abstained:
                        answer = agent_res.answer
                        final_verdict = "YES"
                except Exception as e:
                    logger.warning("ReAct fallback failed: %s", e)

        abstained = answer.lower().startswith("i don't know")
        return ReadResult(
            answer=answer,
            verdict=final_verdict,
            missing=missing,
            abstained=abstained,
            solved_by="reader",
        )

    async def _read_synthesis(self, question: str, context: str) -> ReadResult:
        """Synthesis-mode branch — direct LLM call, skips verifier/tool/escalation.

        Bypasses the entire fact-recall pipeline. Calibrated for preference/
        recommendation questions where the answer is a synthesis grounded in
        stored user statements, not a literal fact retrieval.
        """
        sys_prompt = SYNTHESIS_PROMPT.format(context=context, question=question)
        try:
            resp = await self._llm.generate(
                [
                    LLMMessage(role="system", content=sys_prompt),
                    LLMMessage(role="user", content=question),
                ],
                temperature=0.0,
                max_tokens=400,
            )
        except ExtractionError:
            return ReadResult(
                answer="I don't know",
                abstained=True,
                solved_by="synthesis",
            )
        answer = resp.content.strip()
        return ReadResult(
            answer=answer,
            abstained=answer.lower().startswith("i don't know"),
            solved_by="synthesis",
        )

    async def _verify(self, question: str, context: str) -> tuple[Verdict, str | None]:
        """Run the verifier.

        Returns (verdict, missing). On parse failure, returns ('PARTIAL', None).

        Uses XML tags (verdict, missing) rather than JSON because Anthropic
        models honor structured-text more reliably than JSON-mode for short
        outputs like this. OpenAI/Ollama also handle XML fine.
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
                max_tokens=120,
            )
        except ExtractionError as e:
            logger.warning("verifier LLM call failed (%s); defaulting to PARTIAL", e)
            return ("PARTIAL", None)

        text = resp.content.strip()
        m = _VERDICT_RE.search(text)
        if m is None:
            logger.warning("verifier missing <verdict> tag; defaulting to PARTIAL: %r", text[:120])
            return ("PARTIAL", None)
        v = m.group(1).upper()
        verdict: Verdict = v if v in ("YES", "NO", "PARTIAL") else "PARTIAL"  # type: ignore[assignment]

        missing: str | None = None
        m2 = _MISSING_RE.search(text)
        if m2 is not None:
            raw = m2.group(1).strip()
            if raw and raw.lower() != "none":
                missing = raw[:200]
        return (verdict, missing)


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
