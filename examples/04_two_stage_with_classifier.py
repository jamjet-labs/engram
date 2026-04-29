"""Showcase Phase 9 + 10: session-first retrieval + per-category budgeted context."""

import asyncio

from engram import Engram
from engram.classify.rules import RuleBasedClassifier
from engram.retrieve.base import RetrievalConfig


async def main() -> None:
    cfg = RetrievalConfig(enable_two_stage=True, two_stage_top_sessions=2)
    async with await Engram.open(":memory:", retrieval_config=cfg) as memory:
        # Session A: about coffee
        for text in [
            "alice prefers espresso",
            "alice's favorite coffee shop is in tokyo",
            "alice drinks coffee every morning",
        ]:
            await memory.record(text=text, user_id="alice", session_id="session-A")
        # Session B: about her cat
        for text in [
            "alice has a cat named whiskers",
            "the cat is gray and white",
            "the cat eats at 7am",
        ]:
            await memory.record(text=text, user_id="alice", session_id="session-B")

        rules = RuleBasedClassifier()
        ctx = await memory.context(
            query="what is my favorite drink?", user_id="alice", classifier=rules
        )
        print("Context (auto-budgeted via classifier):")
        print(ctx)


if __name__ == "__main__":
    asyncio.run(main())
