"""Example: per-category routing with synthesis-mode reader.

Demonstrates the v0.1.0 API for preference/recommendation questions.
The user's chat history is ingested with role tags; preference questions
take a synthesis path (user-only retrieval + recommendation-grounded
prompt + verifier off), while other categories take the existing
recall pipeline.

Run from the engram repo root:

    set -a && source /path/to/.env && set +a    # OPENAI_API_KEY
    uv run python examples/05_preference_synthesis.py
"""

from __future__ import annotations

import asyncio
import os

from engram import (
    Engram,
    Reader,
    RuleBasedClassifier,
    Scope,
    is_preference_question,
)
from engram.embedding.synthetic import SyntheticEmbedding
from engram.llm.tier import ModelTier

CHAT_HISTORY = [
    {
        "role": "user",
        "content": "I'm considering upgrading from a Fender Stratocaster to a Gibson Les Paul.",
    },
    {
        "role": "assistant",
        "content": (
            "Both are iconic. Strats are bright and snappy, "
            "Les Pauls are thicker and warmer."
        ),
    },
    {
        "role": "user",
        "content": "I mostly play blues and rock at home — quiet practice setup.",
    },
    {
        "role": "assistant",
        "content": (
            "For low-volume blues/rock, a Les Paul into a small tube combo or "
            "modeler shines."
        ),
    },
]

QUESTIONS = [
    "Any tips on what to look for in a new guitar?",  # preference / recommendation
    "What guitar do I currently own?",  # plain recall
]


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment")

    embedder = SyntheticEmbedding(dim=128)  # use Ollama or OpenAI in production
    tier = ModelTier.default()  # gpt-4o-mini reader + utility
    classifier = RuleBasedClassifier()

    async with await Engram.open(":memory:", embedder=embedder, tier=tier) as memory:
        # Ingest with role tags. Always tag — retrieval can opt-filter later.
        for turn in CHAT_HISTORY:
            await memory.record(
                text=turn["content"],
                role=turn["role"],
                user_id="alice",
                session_id="s1",
            )

        for q in QUESTIONS:
            qt = await classifier.classify(q)
            is_pref = is_preference_question(q, qt)
            label = "synthesis" if is_pref else "recall"
            print(f"\n[{label}] Q: {q}")

            if is_pref:
                ctx = await memory.context(
                    query=q,
                    user_id="alice",
                    role_filter=("user",),
                    token_budget=2000,
                )
                reader = Reader(tier.reader, mode="synthesis")
                res = await reader.read(question=q, context=ctx)
            else:
                ctx = await memory.context(
                    query=q,
                    user_id="alice",
                    classifier=classifier,
                )
                reader = Reader(tier.reader)  # default mode="recall"
                res = await reader.read(
                    question=q,
                    context=ctx,
                    scope=Scope(org_id="default", user_id="alice"),
                )

            print(f"A: {res.answer}")


if __name__ == "__main__":
    asyncio.run(main())
