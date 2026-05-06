"""Example: Engram (memory) + JamJet (durable execution) composed.

The two libraries play complementary roles:

  • Engram     answers "what does this user/agent remember?"
  • JamJet     answers "what happens when the process restarts mid-execution?"

This example wires Engram into a JamJet @durable workflow as the memory
layer. The workflow records each user turn into Engram, retrieves
preference-aware context for the next response, and the @durable
decorator ensures the whole flow can survive a crash + replay without
duplicating side-effects.

Prereqs:
    uv add jamjet-engram jamjet
    export OPENAI_API_KEY=...

Run:
    uv run python examples/06_with_jamjet.py
"""

from __future__ import annotations

import asyncio
import os

from jamjet import durable, tool

from engram import Engram, Reader, RuleBasedClassifier, is_preference_question
from engram.embedding.synthetic import SyntheticEmbedding
from engram.llm.tier import ModelTier

# ── Engram-backed tools that the JamJet workflow can call ──────────────────


@tool
async def remember(text: str, role: str, user_id: str) -> str:
    """Record a chat turn into Engram. Tagged with role for retrieval-time filtering."""
    async with await Engram.open(":memory:", embedder=SyntheticEmbedding(dim=128)) as memory:
        fact = await memory.record(text=text, role=role, user_id=user_id)
        return f"Recorded fact {fact.id} for {user_id}"


@tool
async def answer_with_memory(question: str, user_id: str) -> str:
    """Compose Engram's per-category routing with a JamJet-driven agent step.

    For preference/recommendation questions, Engram routes to synthesis-mode
    reading (recommendation grounded in stored user preferences). For other
    categories, recall-mode (verifier-backed fact recall) runs.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return "(skipped: OPENAI_API_KEY not set)"

    embedder = SyntheticEmbedding(dim=128)
    tier = ModelTier.default()
    classifier = RuleBasedClassifier()

    async with await Engram.open(":memory:", embedder=embedder, tier=tier) as memory:
        # Pretend the user has a stored preference. In a real workflow this
        # would already be in Engram's store from prior @durable steps.
        await memory.record(
            text="user prefers Netflix stand-up specials, especially storytelling-heavy ones",
            role="user",
            user_id=user_id,
        )

        qt = await classifier.classify(question)

        if is_preference_question(question, qt):
            ctx = await memory.context(
                query=question, user_id=user_id, role_filter=("user",), token_budget=1500
            )
            reader = Reader(tier.reader, mode="synthesis")
        else:
            ctx = await memory.context(query=question, user_id=user_id, classifier=classifier)
            reader = Reader(tier.reader)

        result = await reader.read(question=question, context=ctx)
        return result.answer


# ── JamJet @durable workflow that composes the tools above ────────────────


@durable
async def conversation_turn(user_id: str, question: str) -> str:
    """A single conversational turn, durably executed.

    If this process crashes mid-flight, the @durable decorator's replay
    mechanism re-runs from the last completed step. Tool calls (@tool)
    are checkpointed by JamJet, so `remember` and `answer_with_memory`
    don't duplicate side-effects.
    """
    answer = await answer_with_memory(question=question, user_id=user_id)
    await remember(text=question, role="user", user_id=user_id)
    await remember(text=answer, role="assistant", user_id=user_id)
    return answer


async def main() -> None:
    user_id = "alice"

    # Preference question — Engram routes to synthesis mode.
    answer = await conversation_turn(
        user_id=user_id,
        question="What show should I watch tonight?",
    )
    print("\n>>> alice: What show should I watch tonight?")
    print(f"<<< {answer}")


if __name__ == "__main__":
    asyncio.run(main())
