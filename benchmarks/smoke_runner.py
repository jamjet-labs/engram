"""Smoke runner — ablation harness for 100-q LongMemEval subset.

Produces JSONL traces for ablation analysis. Same selection algorithm
as longmemeval_v2.py for comparable numbers across runs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
