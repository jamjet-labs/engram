"""Showcase Phase 9 + 10: session-first retrieval + per-category budgeted context."""

import asyncio

from engram import Engram
from engram.classify.rules import RuleBasedClassifier
from engram.retrieve.base import RetrievalConfig


async def main() -> None:
    cfg = RetrievalConfig(enable_two_stage=True, two_stage_top_sessions=2)
    async with await Engram.open(":memory:", retrieval_config=cfg) as memory:
        alice = memory.scoped(org_id="acme", user_id="alice")
        # Session A: about coffee
        for text in [
            "alice prefers espresso",
            "alice's favorite coffee shop is in tokyo",
            "alice drinks coffee every morning",
        ]:
            await alice.record(text, session_id="session-A")
        # Session B: about her cat
        for text in [
            "alice has a cat named whiskers",
            "the cat is gray and white",
            "the cat eats at 7am",
        ]:
            await alice.record(text, session_id="session-B")

        rules = RuleBasedClassifier()
        ctx = await alice.context("what is my favorite drink?", classifier=rules)
        print("Context (auto-budgeted via classifier):")
        print(ctx)


if __name__ == "__main__":
    asyncio.run(main())
