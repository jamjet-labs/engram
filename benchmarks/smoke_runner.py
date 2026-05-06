"""Smoke runner — ablation harness for 100-q LongMemEval subset.

Produces JSONL traces for ablation analysis. Same selection algorithm
as longmemeval_v2.py for comparable numbers across runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict


def stratified_sample(oracle_path: str, n: int = 100) -> list[dict[str, Any]]:
    data = json.loads(Path(oracle_path).read_text())
    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in data:
        by_type.setdefault(q["question_type"], []).append(q)
    for t in by_type:
        by_type[t].sort(key=lambda q: sum(len(s) for s in q["haystack_sessions"]))
    selected: list[dict[str, Any]] = []
    types = sorted(by_type.keys())
    idx = 0
    while len(selected) < n and any(by_type.values()):
        t = types[idx % len(types)]
        if by_type[t]:
            selected.append(by_type[t].pop(0))
        idx += 1
    return selected[:n]


@dataclass
class SmokeFlags:
    reader: str = "gpt-4o-mini"  # gpt-4o-mini | claude-haiku | claude-sonnet-4-6
    decompose: bool = False
    solver: bool = False
    tools: bool = False
    reextract: bool = False
    self_consistency: bool = False
    react: bool = False
    ft_cross_encoder: bool = False
    n: int = 100
    out_dir: Path = Path("./benchmarks/reports")


def parse_args() -> SmokeFlags:
    p = argparse.ArgumentParser()
    p.add_argument("--reader", default="gpt-4o-mini")
    p.add_argument("--decompose", action="store_true")
    p.add_argument("--solver", action="store_true")
    p.add_argument("--tools", action="store_true")
    p.add_argument("--reextract", action="store_true")
    p.add_argument("--self-consistency", action="store_true")
    p.add_argument("--react", action="store_true")
    p.add_argument("--ft-cross-encoder", action="store_true")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("./benchmarks/reports"))
    a = p.parse_args()
    return SmokeFlags(
        reader=a.reader, decompose=a.decompose, solver=a.solver, tools=a.tools,
        reextract=a.reextract, self_consistency=a.self_consistency,
        react=a.react, ft_cross_encoder=a.ft_cross_encoder, n=a.n, out_dir=a.out_dir,
    )


class TraceRecord(TypedDict, total=False):
    qid: str
    category: str
    predicted_category: str
    is_preference: bool
    synthesis_used: bool
    solver_fired: bool
    decomposer: dict[str, Any]
    recall: list[dict[str, Any]]
    reader: dict[str, Any]
    verifier: dict[str, Any]
    escalation: list[dict[str, Any]]
    react: dict[str, Any] | None
    final: dict[str, Any]


def write_trace_line(path: Path, rec: TraceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ── Runnable harness ────────────────────────────────────────────────────────

ORACLE = Path(
    os.environ.get(
        "LONGMEMEVAL_ORACLE",
        "../jamjet-research/paper/experiments/longmemeval_repo/data/longmemeval_oracle.json",
    )
).expanduser().resolve()


SYNTHESIS_PROMPT = """You are answering a USER's preference/recommendation question.

CONTEXT (the user's prior statements, in their own words):
{context}

QUESTION: {question}

Your job is to give a HELPFUL, GROUNDED recommendation:
1. Identify what the user has stated about their interests, tastes, ownership, or current situation.
2. Use those stated preferences as the basis for your recommendation.
3. Tailor your answer SPECIFICALLY to what the user has said. If the user mentioned specific items (e.g., "Fender Stratocaster"), incorporate them.
4. If the context contains absolutely nothing relevant to the topic, say "I don't know" — but try first.
5. Concise: 2-4 sentences or a short bulleted list.

Answer:"""  # noqa: E501


def _build_tier(reader: str) -> Any:
    """Construct a ModelTier for the named reader.

    Imported lazily so unit tests of stratified_sample / writer don't pay
    the import cost or the OPENAI_API_KEY assertion.
    """
    from engram.llm.tier import ModelTier

    if reader == "gpt-4o-mini":
        return ModelTier.default()
    if reader == "claude-sonnet-4-6":
        return ModelTier.sonnet_reader()
    if reader == "claude-haiku-4-5-20251001":
        return ModelTier.haiku_reader()
    raise SystemExit(f"unknown --reader: {reader}")


async def _synthesis_read(tier: Any, context: str, question: str) -> str:
    """Synthesis-mode read for preference questions.

    Bypasses the Reader pipeline (no verifier, no tool loop, no escalation
    rungs). Calls tier.reader.generate directly with SYNTHESIS_PROMPT.

    Returns the model's answer string (trimmed). On any ExtractionError,
    returns "I don't know" so a single bad LLM call does not crash a 100-q
    smoke.
    """
    from engram.errors import ExtractionError
    from engram.llm.base import LLMMessage

    sys_prompt = SYNTHESIS_PROMPT.format(context=context, question=question)
    try:
        resp = await tier.reader.generate(
            [
                LLMMessage(role="system", content=sys_prompt),
                LLMMessage(role="user", content=question),
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except ExtractionError:
        return "I don't know"
    return resp.content.strip()


def _parse_haystack_date(raw: str) -> datetime:
    return datetime.strptime(raw.split(" (")[0], "%Y/%m/%d").replace(tzinfo=UTC)


async def _ingest_chunks(
    memory: Any,
    q: dict[str, Any],
    user_id: str,
    role_filter: tuple[str, ...] | None = None,
) -> int:
    """Ingest haystack chunks into memory.

    When ``role_filter`` is provided, only turns whose role is in the filter
    are ingested. ``None`` (default) ingests everything — backward-compatible
    with the previous signature.
    """
    n = 0
    for sid, sdate_raw, session in zip(
        q["haystack_session_ids"],
        q["haystack_dates"],
        q["haystack_sessions"],
        strict=False,
    ):
        sdate = _parse_haystack_date(sdate_raw)
        for turn in session:
            if role_filter is not None and turn.get("role") not in role_filter:
                continue
            await memory.record(
                text=turn["content"],
                user_id=user_id,
                session_id=sid,
                event_date=sdate,
            )
            n += 1
    return n


async def _ingest_events(memory: Any, q: dict[str, Any], user_id: str) -> int:
    """Populate the SVO event calendar by calling extract_events per session.

    Required when --solver is enabled — the temporal solver needs SVO events
    to query against. One LLM call per session (utility tier).

    Returns the total number of events extracted across all sessions.
    """
    from engram.models import ChatMessage
    from engram.scope import Scope

    scope = Scope(org_id="default", user_id=user_id)
    n_events = 0
    for sid, sdate_raw, session in zip(
        q["haystack_session_ids"],
        q["haystack_dates"],
        q["haystack_sessions"],
        strict=False,
    ):
        sdate = _parse_haystack_date(sdate_raw)
        msgs = [
            ChatMessage(
                scope=scope,
                session_id=sid,
                role=turn["role"],
                content=turn["content"],
                timestamp=sdate,
            )
            for turn in session
        ]
        try:
            events = await memory.extract_events(msgs, session_date=sdate, persist=True)
            n_events += len(events)
        except Exception as e:
            print(f"  [event extraction failed for session {sid}: {e}]")
    return n_events


def _register_final_answer_tool(reg: Any) -> None:
    """Register a final_answer tool used by the ReAct agent to terminate."""
    from typing import ClassVar

    from engram.tools.base import ToolResult

    class _FinalAnswerTool:
        name: ClassVar[str] = "final_answer"
        description: ClassVar[str] = (
            "Emit the final answer to the user. Always call this when you have the answer."
        )
        input_schema: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        async def __call__(self, answer: str) -> ToolResult:
            return ToolResult(content=answer, raw=answer)

    reg.register(_FinalAnswerTool())


def _build_tools(memory: Any, solver: Any, flags: SmokeFlags) -> Any:
    """Construct a ToolRegistry with all 6 built-in tools wired up."""
    from engram.scope import Scope
    from engram.tools.base import ToolRegistry
    from engram.tools.count_between import CountBetweenTool
    from engram.tools.dates import AddDaysTool, DaysBetweenTool
    from engram.tools.search_events import SearchEventsTool
    from engram.tools.search_facts import SearchFactsTool
    from engram.tools.solve_temporal import SolveTemporalTool

    scope = Scope(org_id="default", user_id="alice")
    reg = ToolRegistry()
    reg.register(SearchFactsTool(engram=memory, scope=scope))
    reg.register(SearchEventsTool(engram=memory, scope=scope))
    if solver is not None:
        reg.register(SolveTemporalTool(solver=solver, scope=scope))
    reg.register(CountBetweenTool(engram=memory, scope=scope))
    reg.register(AddDaysTool())
    reg.register(DaysBetweenTool())
    return reg


async def main() -> None:
    flags = parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY not set (judge requires it)")
    if "claude" in flags.reader and "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY not set for Anthropic reader")
    if not ORACLE.exists():
        raise SystemExit(f"oracle not found: {ORACLE}")

    # Lazy imports — keep the unit-test surface light.
    from benchmarks.judge import judge_one
    from engram import Engram
    from engram.classify.rules import RuleBasedClassifier
    from engram.embedding.ollama import OllamaEmbedding
    from engram.llm.openai import OpenAILLM
    from engram.read.decomposer import should_decompose
    from engram.read.preference_gate import is_preference_question
    from engram.read.reader import Reader, ReaderConfig
    from engram.retrieve.base import RetrievalConfig
    from engram.retrieve.rerank import CrossEncoderReranker
    from engram.scope import Scope
    from engram.solve.temporal import TemporalSolver

    flags.out_dir.mkdir(parents=True, exist_ok=True)
    # Include flag fingerprint in run_id so concurrent smokes with different
    # configs don't collide on the same timestamp.
    flag_tag = "".join(
        c for c, v in [
            ("d", flags.decompose), ("s", flags.solver), ("t", flags.tools),
            ("x", flags.reextract), ("c", flags.self_consistency),
            ("r", flags.react), ("f", flags.ft_cross_encoder),
        ] if v
    ) or "base"
    run_id = f"smoke_{int(time.time())}_{flags.reader.replace('/', '_')}_{flag_tag}_n{flags.n}"
    trace_path = flags.out_dir / f"{run_id}.jsonl"
    report_path = flags.out_dir / f"{run_id}.md"

    print(f"Loading dataset + reranker (reader={flags.reader}, n={flags.n})...")
    selected = stratified_sample(str(ORACLE), n=flags.n)
    tier = _build_tier(flags.reader)
    embedder = OllamaEmbedding(model="nomic-embed-text", dim=768)
    if flags.ft_cross_encoder:
        ft_path = os.environ.get(
            "ENGRAM_FT_CE_PATH", "training/cross_encoder/checkpoints/v1"
        )
        if not Path(ft_path).exists():  # noqa: ASYNC240 — sync stat is fine here
            raise SystemExit(
                f"--ft-cross-encoder set but model not found at {ft_path}. "
                "Train via: uv run python -m training.cross_encoder.train ..."
            )
        print(f"Using fine-tuned cross-encoder: {ft_path}")
        reranker = CrossEncoderReranker(model_name=ft_path)
    else:
        reranker = CrossEncoderReranker()
    classifier = RuleBasedClassifier()
    judge_llm = OpenAILLM(model="gpt-4o-mini")
    cfg = RetrievalConfig(enable_two_stage=True, two_stage_top_sessions=3)

    n_correct = 0
    by_cat: dict[str, list[bool]] = {}
    t0 = time.time()
    for i, q in enumerate(selected, 1):
        print(f"\n=== [{i}/{len(selected)}] {q['question_type']} ===")
        print(f"Q: {q['question'][:120]}")
        # Pre-classify so we can branch the synthesis path.
        predicted_qt = await classifier.classify(q["question"])
        true_qt = q["question_type"]
        is_pref = is_preference_question(q["question"], predicted_qt)

        async with await Engram.open(
            ":memory:",
            embedder=embedder,
            tier=tier,
            reranker=reranker,
            retrieval_config=cfg,
        ) as memory:
            if is_pref:
                # ── Synthesis path ─────────────────────────────────────────
                # Filter to user turns only — embedding similarity ranks
                # info-dense assistant explanations above terse user
                # preferences if both are ingested.
                n_chunks = await _ingest_chunks(
                    memory, q, user_id="alice", role_filter=("user",)
                )
                n_events = 0
                ctx = await memory.context(
                    query=q["question"], user_id="alice", token_budget=3500
                )
                synthesis_used = True
                synth_answer = await _synthesis_read(tier, ctx, q["question"])
                synth_abstained = synth_answer.lower().startswith("i don't know")
                # Normalize to ReadResult-shaped locals for trace consistency
                res_answer = synth_answer
                res_verdict = None
                res_missing = None
                res_abstained = synth_abstained
                res_solved_by = "synthesis"
                today = (
                    _parse_haystack_date(q["question_date"])
                    if q.get("question_date")
                    else None
                )
            else:
                # ── Existing path (unchanged) ──────────────────────────────
                synthesis_used = False
                n_chunks = await _ingest_chunks(memory, q, user_id="alice")
                # When --solver or --react is enabled, also populate the SVO event
                # calendar so the temporal solver / ReAct agent's search_events
                # tool has data to query against.
                if flags.solver or flags.react:
                    n_events = await _ingest_events(memory, q, user_id="alice")
                else:
                    n_events = 0
                ctx = await memory.context(
                    query=q["question"],
                    user_id="alice",
                    classifier=classifier,
                    decompose=flags.decompose,
                )
                today = (
                    _parse_haystack_date(q["question_date"])
                    if q.get("question_date")
                    else None
                )
                solver = (
                    TemporalSolver(store=memory._store, llm=tier.utility)
                    if flags.solver
                    else None
                )
                # When --react is set, we always need a tool registry (ReAct uses
                # it for its tool calls). If the user didn't pass --tools, build one
                # for the agent privately. The reader itself only sees the registry
                # when --tools is explicitly set.
                tools_reg = (
                    _build_tools(memory, solver, flags) if flags.tools else None
                )
                react_tools_reg = (
                    tools_reg
                    if tools_reg is not None
                    else (_build_tools(memory, solver, flags) if flags.react else None)
                )
                # The ReAct agent needs a final_answer tool too, regardless of mode.
                if react_tools_reg is not None and "final_answer" not in (
                    react_tools_reg.names() if hasattr(react_tools_reg, "names") else []
                ):
                    _register_final_answer_tool(react_tools_reg)

                reader = Reader(
                    tier.reader,
                    config=ReaderConfig(
                        solver=solver,
                        tools=tools_reg,
                        enable_reextract=flags.reextract,
                        self_consistency_on_partial=3 if flags.self_consistency else 1,
                    ),
                )
                reader.set_category(q["question_type"])
                if flags.react and react_tools_reg is not None:
                    from engram.agent.react import ReActAgent

                    react = ReActAgent(tier=tier, tools=react_tools_reg, max_hops=4)
                    reader.attach_react(react)
                # Escalation rung (a) — re-extract on PARTIAL/NO. Pre-compute the
                # candidate-session list so the sync provider in attach_reextractor
                # can return it without an extra recall round trip.
                if flags.reextract:
                    from engram.read.reextract import QueryConditionedReextractor

                    hits = await memory.recall(
                        q["question"], user_id="alice", top_k=30
                    )
                    cand_sids: list[str] = []
                    seen: set[str] = set()
                    for sf in hits:
                        sid = sf.fact.session_id
                        if sid and sid not in seen:
                            cand_sids.append(sid)
                            seen.add(sid)
                    rx = QueryConditionedReextractor(llm=tier.utility)
                    reader.attach_reextractor(
                        reextractor=rx,
                        store=memory._store,
                        candidate_sessions_provider=lambda _q, _s=cand_sids: _s,
                    )
                res = await reader.read(
                    question=q["question"],
                    context=ctx,
                    today=today,
                    scope=Scope(org_id="default", user_id="alice"),
                )
                res_answer = res.answer
                res_verdict = res.verdict
                res_missing = res.missing
                res_abstained = res.abstained
                res_solved_by = res.solved_by

            judged = await judge_one(
                question=q["question"],
                expected=str(q["answer"]),
                predicted=res_answer,
                category=true_qt,
                llm=judge_llm,
            )

        n_correct += int(judged.correct)
        by_cat.setdefault(true_qt, []).append(judged.correct)
        decomposer_fired = (
            (not is_pref)            # synthesis path bypasses the decomposer entirely
            and flags.decompose
            and should_decompose(q["question"])
        )
        rec: TraceRecord = {
            "qid": q["question_id"],
            "category": true_qt,
            "predicted_category": predicted_qt.value,
            "is_preference": is_pref,
            "synthesis_used": synthesis_used,
            "solver_fired": res_solved_by == "solver",
            "decomposer": {"fired": decomposer_fired, "subqueries": [q["question"]]},
            "recall": [
                {
                    "sq": q["question"],
                    "n_candidates": n_chunks,
                    "n_events": n_events,
                    "top_k": 0,
                    "fact_ids": [],
                }
            ],
            "reader": {
                "model": flags.reader,
                "tool_calls": [],
                "answer": res_answer,
                "tokens": {"in": 0, "out": 0},
            },
            "verifier": {"verdict": res_verdict, "missing": res_missing},
            "escalation": [],
            "react": None,
            "final": {
                "answer": res_answer,
                "abstained": res_abstained,
                "judge_verdict": "correct" if judged.correct else "incorrect",
                "cost_usd": 0.0,
            },
        }
        write_trace_line(trace_path, rec)
        print(f"Expected: {q['answer']}")
        print(f"Predicted: {res_answer}")
        print(f"judged={judged.correct}")

    pct = 100 * n_correct / max(1, len(selected))
    elapsed = time.time() - t0
    cat_lines = []
    for cat, lst in sorted(by_cat.items()):
        c = sum(lst)
        cat_lines.append(f"- `{cat}`: {c}/{len(lst)} = {100 * c / len(lst):.0f}%")
    report_path.write_text(
        f"# Smoke {run_id}\n\n"
        f"## Configuration\n\n"
        f"- reader: `{flags.reader}`\n"
        f"- decompose: {flags.decompose}\n"
        f"- solver: {flags.solver}\n"
        f"- tools: {flags.tools}\n"
        f"- reextract: {flags.reextract}\n"
        f"- self_consistency: {flags.self_consistency}\n"
        f"- react: {flags.react}\n"
        f"- ft_cross_encoder: {flags.ft_cross_encoder}\n"
        f"- n: {flags.n}\n\n"
        f"## Result\n\n"
        f"**{n_correct}/{len(selected)} = {pct:.1f}%** "
        f"(judge: gpt-4o-mini) in {elapsed:.0f}s\n\n"
        f"### Per-category\n\n"
        + "\n".join(cat_lines)
        + f"\n\nTrace: `{trace_path.name}`\n"
    )
    print(f"\n=== {n_correct}/{len(selected)} = {pct:.1f}% in {elapsed:.0f}s ===")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
