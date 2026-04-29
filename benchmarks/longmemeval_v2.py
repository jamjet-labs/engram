# ruff: noqa: ASYNC240
"""LongMemEval Phase 6 parity smoke — Engram v2 vs Rust v0.6 (66.4%).

Pipeline per question:
  1. Open a fresh Engram instance (in-memory)
  2. For each haystack session, ingest each turn as a Fact with that session's
     session_id and session-date as event_date anchor
  3. recall(query, top_k=10) with two-stage + temporal scoring + cross-encoder rerank
  4. Reader (gpt-4o-mini, temperature=0) generates the final answer
  5. Substring proxy match against expected answer

This is NOT the official LongMemEval judge — it's a proxy for quick parity signals.
For a real submission, use the LongMemEval evaluate_qa.py with gpt-4o-mini judge.

Usage:
    set -a && source /path/to/.env && set +a
    # Optional: point at the LongMemEval oracle JSON if not in the default location.
    export LONGMEMEVAL_ORACLE=/path/to/longmemeval_oracle.json
    # Optional: where to write per-question results (default: ./benchmarks/phase_6_results.json).
    export ENGRAM_BENCH_OUT=./benchmarks/run_$(date +%s).json
    python -m benchmarks.longmemeval_v2 --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engram import Engram
from engram.classify.rules import RuleBasedClassifier
from engram.embedding.ollama import OllamaEmbedding
from engram.llm.anthropic import AnthropicLLM
from engram.read.reader import Reader
from engram.retrieve.base import RetrievalConfig
from engram.retrieve.rerank import CrossEncoderReranker

_DEFAULT_ORACLE = (
    "../jamjet-research/paper/experiments/longmemeval_repo/data/longmemeval_oracle.json"
)
ORACLE_PATH = Path(os.environ.get("LONGMEMEVAL_ORACLE", _DEFAULT_ORACLE)).expanduser().resolve()


def _parse_haystack_date(raw: str) -> datetime:
    """LongMemEval haystack_dates are like '2023/04/10 (Mon) 23:07'."""
    head = raw.split(" (")[0]
    return datetime.strptime(head, "%Y/%m/%d").replace(tzinfo=UTC)


async def _ingest(memory: Engram, q: dict[str, Any], user_id: str) -> int:
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


async def _answer(
    memory: Engram, reader: Reader, classifier: RuleBasedClassifier, q: dict[str, Any], user_id: str
) -> str:
    ctx = await memory.context(query=q["question"], user_id=user_id, classifier=classifier)
    # The question_date is "today" for temporal reasoning purposes.
    today = _parse_haystack_date(q["question_date"]) if q.get("question_date") else None
    result = await reader.read(question=q["question"], context=ctx, today=today)
    return result.answer


def _correct(expected: Any, predicted: str) -> bool:
    e = str(expected).lower().strip()
    p = predicted.lower()
    if e in p:
        return True
    # any non-trivial token from expected appearing in predicted
    return any(tok in p for tok in e.split() if len(tok) > 3)


async def main(limit: int) -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY not set; export before running")
    if not ORACLE_PATH.exists():
        raise SystemExit(f"oracle dataset not found: {ORACLE_PATH}")

    print("Loading dataset + reranker...")
    data = json.loads(ORACLE_PATH.read_text())
    # Pick `limit` questions. For limit >= len(data), use the whole dataset
    # in original order (so 500 means literally all 500 LongMemEval-S
    # questions — same set, same order across runs).
    if limit >= len(data):
        selected: list[dict[str, Any]] = list(data)
    else:
        # Sample evenly across categories, smallest haystack first.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for q in data:
            by_type.setdefault(q["question_type"], []).append(q)
        for t in by_type:
            by_type[t].sort(key=lambda q: sum(len(s) for s in q["haystack_sessions"]))
        selected = []
        types = sorted(by_type.keys())
        idx = 0
        while len(selected) < limit and any(by_type.values()):
            t = types[idx % len(types)]
            if by_type[t]:
                selected.append(by_type[t].pop(0))
            idx += 1
        selected = selected[:limit]

    # Stack: Ollama for embeddings (local, free), Claude Haiku for reading (paid),
    # ms-marco cross-encoder for reranking (local, free). OpenAI key in env was
    # invalid; this combination demonstrates the full v2 pipeline without it.
    embedder = OllamaEmbedding(model="nomic-embed-text", dim=768)
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    reranker = CrossEncoderReranker()
    classifier = RuleBasedClassifier()

    cfg = RetrievalConfig(enable_two_stage=True, two_stage_top_sessions=3)

    results: list[dict[str, Any]] = []
    t_total = time.time()
    for i, q in enumerate(selected, 1):
        print(f"\n=== [{i}/{len(selected)}] {q['question_type']} ===")
        print(f"Q: {q['question'][:120]}")
        async with await Engram.open(
            ":memory:",
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            retrieval_config=cfg,
        ) as memory:
            t0 = time.time()
            n_chunks = await _ingest(memory, q, user_id="alice")
            t_ingest = time.time() - t0
            t1 = time.time()
            # Verifier disabled by default for Anthropic — its JSON mode is
            # less reliable than OpenAI's, so verdict was always defaulting to
            # PARTIAL anyway. Set ENGRAM_BENCH_VERIFIER=1 to re-enable.
            verifier_on = os.environ.get("ENGRAM_BENCH_VERIFIER") == "1"
            answer = await _answer(
                memory,
                reader=Reader(llm, verifier=verifier_on),
                classifier=classifier,
                q=q,
                user_id="alice",
            )
            t_answer = time.time() - t1
        ok = _correct(q["answer"], answer)
        results.append(
            {
                "id": q["question_id"],
                "type": q["question_type"],
                "expected": str(q["answer"]),
                "predicted": answer,
                "n_chunks": n_chunks,
                "t_ingest": round(t_ingest, 2),
                "t_answer": round(t_answer, 2),
                "correct": ok,
            }
        )
        print(f"Expected: {q['answer']}")
        print(f"Predicted: {answer}")
        print(f"correct={ok}  n_chunks={n_chunks}  ingest={t_ingest:.1f}s  answer={t_answer:.1f}s")

    t_total_s = time.time() - t_total
    correct = sum(1 for r in results if r["correct"])
    print("\n" + "=" * 60)
    print("PHASE 6 PARITY SMOKE — RESULTS")
    print("=" * 60)
    pct = 100 * correct / max(1, len(results))
    print(f"Questions: {len(results)}")
    print(f"Correct (substring proxy): {correct}/{len(results)} = {pct:.1f}%")
    print(f"Total wall-clock: {t_total_s:.1f}s")
    print()
    by_type_count: dict[str, list[bool]] = {}
    for r in results:
        by_type_count.setdefault(r["type"], []).append(r["correct"])
    for t, lst in sorted(by_type_count.items()):
        c = sum(lst)
        print(f"  [{t}] {c}/{len(lst)} = {100 * c / len(lst):.0f}%")
    out_path = Path(os.environ.get("ENGRAM_BENCH_OUT", "./benchmarks/phase_6_results.json"))
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()
    asyncio.run(main(args.limit))
