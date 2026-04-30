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


def _parse_haystack_date(raw: str) -> datetime:
    return datetime.strptime(raw.split(" (")[0], "%Y/%m/%d").replace(tzinfo=UTC)


async def _ingest_chunks(memory: Any, q: dict[str, Any], user_id: str) -> int:
    n = 0
    for sid, sdate_raw, session in zip(
        q["haystack_session_ids"],
        q["haystack_dates"],
        q["haystack_sessions"],
        strict=False,
    ):
        sdate = _parse_haystack_date(sdate_raw)
        for turn in session:
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
    from engram.read.reader import Reader, ReaderConfig
    from engram.retrieve.base import RetrievalConfig
    from engram.retrieve.rerank import CrossEncoderReranker
    from engram.scope import Scope
    from engram.solve.temporal import TemporalSolver

    flags.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"smoke_{int(time.time())}_{flags.reader.replace('/', '_')}"
    trace_path = flags.out_dir / f"{run_id}.jsonl"
    report_path = flags.out_dir / f"{run_id}.md"

    print(f"Loading dataset + reranker (reader={flags.reader}, n={flags.n})...")
    selected = stratified_sample(str(ORACLE), n=flags.n)
    tier = _build_tier(flags.reader)
    embedder = OllamaEmbedding(model="nomic-embed-text", dim=768)
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
        async with await Engram.open(
            ":memory:",
            embedder=embedder,
            tier=tier,
            reranker=reranker,
            retrieval_config=cfg,
        ) as memory:
            n_chunks = await _ingest_chunks(memory, q, user_id="alice")
            # When --solver is enabled, also populate the SVO event calendar
            # so the temporal solver has data to query against.
            if flags.solver:
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
                _parse_haystack_date(q["question_date"]) if q.get("question_date") else None
            )
            solver = (
                TemporalSolver(store=memory._store, llm=tier.utility)
                if flags.solver
                else None
            )
            reader = Reader(tier.reader, config=ReaderConfig(solver=solver))
            res = await reader.read(
                question=q["question"],
                context=ctx,
                today=today,
                scope=Scope(org_id="default", user_id="alice"),
            )
            judged = await judge_one(
                question=q["question"],
                expected=str(q["answer"]),
                predicted=res.answer,
                category=q["question_type"],
                llm=judge_llm,
            )
        n_correct += int(judged.correct)
        by_cat.setdefault(q["question_type"], []).append(judged.correct)
        decomposer_fired = flags.decompose and should_decompose(q["question"])
        rec: TraceRecord = {
            "qid": q["question_id"],
            "category": q["question_type"],
            "solver_fired": res.solved_by == "solver",
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
                "answer": res.answer,
                "tokens": {"in": 0, "out": 0},
            },
            "verifier": {"verdict": res.verdict, "missing": res.missing},
            "escalation": [],
            "react": None,
            "final": {
                "answer": res.answer,
                "abstained": res.abstained,
                "judge_verdict": "correct" if judged.correct else "incorrect",
                "cost_usd": 0.0,
            },
        }
        write_trace_line(trace_path, rec)
        print(f"Expected: {q['answer']}")
        print(f"Predicted: {res.answer}")
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
