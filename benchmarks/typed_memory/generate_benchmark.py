#!/usr/bin/env python3
"""Generate larger randomized typed-memory benchmark fixtures.

Usage:
    uv run python -m benchmarks.typed_memory.generate_benchmark \\
        --seed-tasks fixtures.json --output generated-fixtures.json --count 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ── distractor pools ───────────────────────────────────────────────────────


_SEMANTIC_DISTRACTORS = [
    "The database connection string is stored in /etc/config/db.env.",
    "Maximum upload file size is 25 MB.",
    "Session timeout is set to 30 minutes of inactivity.",
    "The application uses JWT tokens for authentication.",
    "Logging level defaults to INFO in production.",
    "Rate limiting is 100 requests per minute per user.",
    "The primary datacenter is in us-east-1.",
    "Backups run daily at 02:00 UTC.",
    "The API gateway supports WebSocket connections.",
    "Cache TTL is 5 minutes for product listings.",
    "The build artifact is a single fat JAR.",
    "Health check endpoint is at /healthz.",
    "The deployment uses blue-green strategy.",
    "Feature flags are managed through LaunchDarkly.",
    "Error tracking is sent to Sentry.",
    "Content is served through a CDN with 1-hour cache.",
    "Database migrations use Flyway.",
    "The monorepo is structured with pnpm workspaces.",
    "Environment variables are loaded from .env file.",
    "The CI pipeline runs on GitHub Actions.",
]

_PROCEDURAL_DISTRACTORS = [
    "Incident response: acknowledge, triage, mitigate, postmortem.",
    "Code review checklist: correctness, testing, docs, performance.",
    "Feature branch workflow: create branch, commit, push, open PR.",
    "Database migration process: backup, test locally, run migration, verify.",
    "Monitoring setup: install agent, configure alerts, test notification.",
    "Dependency update process: check changelog, update, run tests, commit.",
    "Onboarding: create account, set up dev environment, first PR.",
    "Security audit: review dependencies, check secrets, scan endpoints.",
    "Performance tuning: profile, identify bottleneck, optimize, re-profile.",
    "Documentation update: identify changes, update docs, review, publish.",
    "Rollback procedure: identify breaking change, revert, verify fix.",
    "Certificate renewal: check expiry, generate new cert, deploy, verify.",
    "Load test process: define scenarios, run test, analyze results.",
    "A/B test setup: define metric, split traffic, run experiment, analyze.",
    "Log analysis: collect logs, identify patterns, correlate, report.",
]

_EPISODIC_DISTRACTORS = [
    "Session event: investigated a memory leak in the event loop.",
    "Session event: user reported slow query performance on dashboard.",
    "Session event: fixed a race condition in the payment webhook.",
    "Session event: upgraded PostgreSQL from 14 to 16.",
    "Session event: migrated CI from Jenkins to GitHub Actions.",
    "Session event: debugged an intermittent 503 error in production.",
    "Session event: refactored the authentication middleware.",
    "Session event: added end-to-end tests for the checkout flow.",
    "Session event: optimized the image processing pipeline.",
    "Session event: implemented WebSocket reconnection logic.",
    "Session event: fixed cross-origin request issues in the API.",
    "Session event: updated SSL certificates for the staging environment.",
    "Session event: reduced Docker image size from 1.2GB to 400MB.",
    "Session event: added structured logging to all services.",
    "Session event: patched a critical security vulnerability in auth.",
]

_WORKING_DISTRACTORS = [
    "Active session: currently debugging the CI pipeline failure.",
    "Active session: investigating the performance regression in v3.2.",
    "Active session: drafting the migration plan for the new API.",
    "Active session: setting up the development environment on new machine.",
    "Active session: reviewing the pull request for the auth module.",
    "Active session: writing unit tests for the payment processor.",
    "Active session: updating the deployment configuration.",
    "Active session: troubleshooting the WebSocket connection drop.",
    "Active session: experimenting with a new caching strategy.",
    "Active session: analyzing the latest error logs from production.",
]

_PROSPECTIVE_DISTRACTORS = [
    "After the database migration completes, update the connection strings.",
    "When the new API is deployed, deprecate the old v1 endpoints.",
    "After the security audit finishes, rotate all API keys.",
    "When Q2 planning starts, propose the performance improvement initiative.",
    "After the team expands, split the monolith into microservices.",
    "When the certification expires, schedule renewal training.",
    "After the test suite runs, check for flaky tests.",
    "When the deployment finishes, verify the health check endpoint.",
    "After the contract is signed, set up the integration environment.",
    "When the budget is approved, hire additional SRE team members.",
]

_PREFERENCE_DISTRACTORS = [
    "For this team, always write tests before implementation.",
    "Prefer functional programming patterns for data transformations.",
    "Use conventional commits for all commit messages.",
    "Prefer async/await over raw callbacks in new code.",
    "Use TypeScript strict mode for all frontend projects.",
    "Prefer PostgreSQL over MongoDB for new services.",
    "Always include API documentation in every pull request.",
    "Use feature flags for all new functionality.",
    "Prefer composition over inheritance in class design.",
    "Write documentation in Markdown, not Confluence.",
]


def _random_distractors(
    pool: list[str],
    exclude_systems: set[str] | None = None,
    n: int = 3,
) -> list[dict[str, Any]]:
    selected = random.sample(pool, min(n, len(pool)))
    return [
        {
            "content": text,
            "expected_memory_system": "",
        }
        for text in selected
    ]


def _system_pool(system: str) -> list[str]:
    pools = {
        "semantic": _SEMANTIC_DISTRACTORS,
        "procedural": _PROCEDURAL_DISTRACTORS,
        "episodic": _EPISODIC_DISTRACTORS,
        "working": _WORKING_DISTRACTORS,
        "prospective": _PROSPECTIVE_DISTRACTORS,
        "preference": _PREFERENCE_DISTRACTORS,
    }
    return pools.get(system, [])


# ── fixture generation ─────────────────────────────────────────────────────


def _original_systems_from_task(task: dict[str, Any]) -> list[str]:
    systems = set()
    for ev in task.get("setup_events", []):
        sys_name = ev.get("expected_memory_system", "")
        if sys_name:
            systems.add(sys_name)
    return list(systems) if systems else ["semantic"]


def _make_distractor(distractor_text: str) -> dict[str, Any]:
    return {
        "content": distractor_text,
    }


def generate(
    seed_tasks: list[dict[str, Any]],
    count: int = 100,
    distractors_per_project: int = 5,
    same_system_n: int = 3,
    cross_system_n: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []

    while len(output) < count:
        task = rng.choice(seed_tasks)
        base_task = json.loads(json.dumps(task))
        original_systems = _original_systems_from_task(base_task)
        primary_system = original_systems[0]

        variant_id = f"{base_task['id']}_v{len(output) + 1}"
        base_task["id"] = variant_id
        base_task["expected_memory_systems"] = original_systems
        base_task["forbidden_retrieval_contains"] = []
        original_events = list(base_task["setup_events"])

        same_pool = _system_pool(primary_system)
        all_systems = ["semantic", "procedural", "episodic", "working", "prospective", "preference"]
        cross_pools = [s for s in all_systems if s != primary_system]

        same_chosen = rng.sample(same_pool, min(same_system_n, len(same_pool))) if same_pool else []
        for text in same_chosen:
            base_task["setup_events"].append(_make_distractor(text))

        cross_chosen = rng.sample(cross_pools, min(cross_system_n, len(cross_pools)))
        for cross_sys in cross_chosen:
            pool = _system_pool(cross_sys)
            if pool:
                for text in rng.sample(pool, 1):
                    base_task["setup_events"].append(_make_distractor(text))

        generated_forbidden = []
        for ev in base_task["setup_events"][len(original_events):]:
            content = ev["content"]
            key = content.split(":")[0] if ":" in content else content[:40]
            generated_forbidden.append(key)
        base_task["forbidden_retrieval_contains"] = generated_forbidden

        rng.shuffle(base_task["setup_events"])
        output.append(base_task)

    return output


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate typed memory benchmark fixtures")
    p.add_argument(
        "--seed-tasks",
        default=Path(__file__).parent / "fixtures.json",
        type=Path,
    )
    p.add_argument("--output", type=Path, default=Path("generated-fixtures.json"))
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--distractors-per-project", type=int, default=5)
    p.add_argument("--same-system-n", type=int, default=3)
    p.add_argument("--cross-system-n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.seed_tasks.exists():
        print(f"seed tasks not found: {args.seed_tasks}", file=sys.stderr)
        return 1

    with open(args.seed_tasks) as f:
        seed = json.load(f)

    fixtures = generate(
        seed_tasks=seed,
        count=args.count,
        distractors_per_project=args.distractors_per_project,
        same_system_n=args.same_system_n,
        cross_system_n=args.cross_system_n,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixtures, indent=2))
    print(f"Generated {len(fixtures)} fixtures → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
