#!/usr/bin/env python3
"""Typed memory benchmark runner — baseline vs typed vs inferred.

Usage:
    uv run python -m benchmarks.typed_memory.runner
    uv run python -m benchmarks.typed_memory.runner --fixtures fixtures.json --top-k 5 10 20
    uv run python -m benchmarks.typed_memory.runner \\
        --output-json /tmp/results.json --output-md /tmp/results.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram import Engram, MemorySystem

# ── fixture model ──────────────────────────────────────────────────────────


@dataclass
class SetupEvent:
    content: str
    expected_memory_system: str | None = None
    expected_relation: str | None = None
    lifecycle_state: str | None = None
    project: str | None = None


@dataclass
class Task:
    id: str
    category: str
    setup_events: list[SetupEvent]
    query: str
    expected_retrieval_contains: list[str]
    expected_behavior: str
    negative_assertions: list[str]
    forbidden_retrieval_contains: list[str] = field(default_factory=list)
    expected_memory_systems: list[str] | None = None
    project: str | None = None


# ── results ────────────────────────────────────────────────────────────────


@dataclass
class TaskResult:
    task_id: str
    category: str
    arm: str
    passed: bool
    retrieval_texts: list[str]
    found_expected: list[str]
    found_forbidden: list[str]
    matched_system: bool | None
    latency_ms: float
    details: str = ""


@dataclass
class BenchmarkSummary:
    arms: list[str]
    total: int
    per_arm: dict[str, list[TaskResult]]
    per_category: dict[str, dict[str, float]]


# ── fixture loading ────────────────────────────────────────────────────────


def load_tasks(path: str | Path) -> list[Task]:
    with open(path) as f:
        raw = json.load(f)
    tasks: list[Task] = []
    for item in raw:
        events = [
            SetupEvent(
                content=e["content"],
                expected_memory_system=e.get("expected_memory_system"),
                expected_relation=e.get("expected_relation"),
                lifecycle_state=e.get("lifecycle_state"),
                project=e.get("project"),
            )
            for e in item.get("setup_events", [])
        ]
        tasks.append(
            Task(
                id=item["id"],
                category=item.get("category", "general"),
                setup_events=events,
                query=item["query"],
                expected_retrieval_contains=item.get("expected_retrieval_contains", []),
                expected_behavior=item.get("expected_behavior", ""),
                negative_assertions=item.get("negative_assertions", []),
                forbidden_retrieval_contains=item.get("forbidden_retrieval_contains", []),
                expected_memory_systems=item.get("expected_memory_systems"),
                project=item.get("project"),
            )
        )
    return tasks


# ── single-task runner ─────────────────────────────────────────────────────


def _pick_memory_sys(name: str | None) -> MemorySystem | None:
    if name is None:
        return None
    try:
        return MemorySystem(name)
    except ValueError:
        return None


async def _run_task(
    task: Task, arm: str, top_k: int
) -> TaskResult:
    async with await Engram.open(":memory:") as memory:
        t0 = time.perf_counter()

        for ev in task.setup_events:
            ms = _pick_memory_sys(ev.expected_memory_system)
            await memory.record(
                text=ev.content,
                user_id=task.project or "default",
                route_memory=True,
                memory_system=ms,
            )

        if arm == "baseline":
            results = await memory.recall(
                query=task.query,
                user_id=task.project or "default",
                top_k=top_k,
            )
        elif arm == "typed":
            sys_filter = _resolve_typed_systems(task)
            results = await memory.recall(
                query=task.query,
                user_id=task.project or "default",
                top_k=top_k,
                memory_systems=sys_filter,
            )
        elif arm == "inferred":
            results = await memory.recall(
                query=task.query,
                user_id=task.project or "default",
                top_k=top_k,
                infer_memory_filters=True,
            )
        else:
            raise ValueError(f"unknown arm: {arm}")

        latency = (time.perf_counter() - t0) * 1000
        retrieval_texts = [sf.fact.text for sf in results]

    found_expected = [
        t for t in task.expected_retrieval_contains
        if any(t in r for r in retrieval_texts)
    ]
    found_forbidden = [
        t for t in task.forbidden_retrieval_contains
        if any(t in r for r in retrieval_texts)
    ]
    matched_system = None

    if results and task.expected_memory_systems:
        expected_sys = _resolve_typed_systems(task)
        matched_system = any(
            sf.fact.memory_system in expected_sys for sf in results
        )

    expected_all_found = len(found_expected) == len(task.expected_retrieval_contains)
    no_forbidden = len(found_forbidden) == 0
    passed = expected_all_found and no_forbidden

    details_parts = []
    if not expected_all_found:
        missing = set(task.expected_retrieval_contains) - set(found_expected)
        details_parts.append(f"missing: {missing}")
    if found_forbidden:
        details_parts.append(f"forbidden: {found_forbidden}")

    return TaskResult(
        task_id=task.id,
        category=task.category,
        arm=arm,
        passed=passed,
        retrieval_texts=retrieval_texts,
        found_expected=found_expected,
        found_forbidden=found_forbidden,
        matched_system=matched_system,
        latency_ms=round(latency, 1),
        details="; ".join(details_parts),
    )


def _resolve_typed_systems(task: Task) -> tuple[MemorySystem, ...] | None:
    if task.expected_memory_systems:
        return tuple(
            ms for ms in (MemorySystem(s) for s in task.expected_memory_systems) if ms
        )
    systems = set()
    for ev in task.setup_events:
        if ev.expected_memory_system:
            try:
                systems.add(MemorySystem(ev.expected_memory_system))
            except ValueError:
                pass
    return tuple(systems) if systems else None


# ── full benchmark ─────────────────────────────────────────────────────────


async def run_benchmark(
    tasks: list[Task],
    arms: list[str] | None = None,
    top_k: int = 10,
) -> BenchmarkSummary:
    if arms is None:
        arms = ["baseline", "typed", "inferred"]

    per_arm: dict[str, list[TaskResult]] = {a: [] for a in arms}
    total = len(tasks)

    for task in tasks:
        for arm in arms:
            result = await _run_task(task, arm, top_k)
            per_arm[arm].append(result)

    per_category: dict[str, dict[str, float]] = {}
    for arm in arms:
        for r in per_arm[arm]:
            if r.category not in per_category:
                per_category[r.category] = {}
            if arm not in per_category[r.category]:
                per_category[r.category][arm] = 0.0

    for arm in arms:
        for r in per_arm[arm]:
            if r.passed:
                per_category[r.category][arm] = (
                    per_category[r.category].get(arm, 0) + 1
                )
    for cat in per_category:
        for arm in per_category[cat]:
            total_cat = sum(1 for r in per_arm[arm] if r.category == cat)
            if total_cat:
                per_category[cat][arm] = round(
                    per_category[cat][arm] / total_cat, 3
                )

    return BenchmarkSummary(
        arms=arms,
        total=total,
        per_arm=per_arm,
        per_category=per_category,
    )


# ── reporting ──────────────────────────────────────────────────────────────


def summarize(summary: BenchmarkSummary) -> dict[str, Any]:
    output: dict[str, Any] = {
        "total_tasks": summary.total,
        "arms": {},
        "per_category": {},
    }
    for arm in summary.arms:
        results = summary.per_arm[arm]
        passed = sum(1 for r in results if r.passed)
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
        output["arms"][arm] = {
            "passed": passed,
            "total": len(results),
            "pass_rate": round(passed / len(results), 3) if results else 0,
            "avg_latency_ms": round(avg_latency, 1),
            "failures": [
                {"task_id": r.task_id, "details": r.details}
                for r in results
                if not r.passed
            ],
        }
        output["arms"][arm]["failures"] = [
            {"task_id": r.task_id, "details": r.details}
            for r in results
            if not r.passed
        ]
    for arm in summary.arms:
        results = summary.per_arm[arm]
        total_forbidden = sum(len(r.found_forbidden) for r in results)
        output["arms"][arm]["total_forbidden_terms"] = total_forbidden
        output["arms"][arm]["avg_forbidden_per_task"] = (
            round(total_forbidden / len(results), 2) if results else 0
        )
        output["arms"][arm]["total_missing"] = sum(
            len(r.found_expected) for r in results
        )
    output["per_category"] = {
        cat: {arm: rate for arm, rate in arms.items()}
        for cat, arms in sorted(summary.per_category.items())
    }
    return output


def format_markdown(summary: BenchmarkSummary) -> str:
    lines: list[str] = [
        "# Typed Memory Benchmark Report",
        "",
        f"**Tasks:** {summary.total}  ",
        f"**Arms:** {', '.join(summary.arms)}  ",
        "",
        "## Pass rates by arm",
        "",
        "| Arm | Passed | Total | Rate | Avg Latency (ms) | Avg Forbidden |",
        "|-----|--------|-------|------|-------------------|---------------|",
    ]
    for arm in summary.arms:
        results = summary.per_arm[arm]
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        rate = round(passed / total, 3) if total else 0
        avg_lat = (
            round(sum(r.latency_ms for r in results) / total, 1) if total else 0
        )
        total_forbidden = sum(len(r.found_forbidden) for r in results)
        avg_forbidden = round(total_forbidden / total, 2) if total else 0
        lines.append(f"| {arm} | {passed} | {total} | {rate} | {avg_lat} | {avg_forbidden} |")

    lines.extend(
         [
             "",
             "## Pass rates by category",
             "",
             "| Category | " + " | ".join(summary.arms) + " |",
             "|---|" + "---|" * len(summary.arms),
         ]
    )
    for cat in sorted(summary.per_category.keys()):
        rates = [str(summary.per_category[cat].get(arm, "-")) for arm in summary.arms]
        lines.append(f"| {cat} | " + " | ".join(rates) + " |")

    lines.extend(["", "## Failures", ""])
    for arm in summary.arms:
        failures = [r for r in summary.per_arm[arm] if not r.passed]
        if failures:
            lines.append(f"### {arm}")
            lines.append("")
            lines.append("| Task | Details |")
            lines.append("|------|---------|")
            for r in failures:
                lines.append(f"| {r.task_id} | {r.details} |")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Typed memory benchmark runner")
    p.add_argument(
        "--fixtures",
        default=Path(__file__).parent / "fixtures.json",
        type=Path,
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--arms", nargs="*", default=["baseline", "typed", "inferred"])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    p.add_argument("--shuffle", action="store_true", help="randomize task order")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.fixtures.exists():
        print(f"fixtures not found: {args.fixtures}", file=sys.stderr)
        return 1

    tasks = load_tasks(args.fixtures)
    if not tasks:
        print("no tasks loaded", file=sys.stderr)
        return 1

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)

    summary = await run_benchmark(tasks, arms=args.arms, top_k=args.top_k)
    report = summarize(summary)

    print(format_markdown(summary))
    print(f"\n{'='*60}")
    print(f"Total: {summary.total} tasks | Arms: {', '.join(summary.arms)}")
    for arm in args.arms:
        r = report["arms"][arm]
        print(f"  {arm}: {r['passed']}/{r['total']} passed ({r['pass_rate']})")

    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2))
        print(f"\njson → {args.output_json}")
    if args.output_md:
        args.output_md.write_text(format_markdown(summary))
        print(f"md  → {args.output_md}")

    return 0


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
