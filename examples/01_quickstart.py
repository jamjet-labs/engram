"""Quickstart — record + recall in 10 lines.

Run: python examples/01_quickstart.py
"""

import asyncio

from engram import Engram


async def main() -> None:
    """Note: this uses the default SyntheticEmbedding, which is deterministic
    but lexical-only. For real semantic recall, swap in OllamaEmbedding or
    OpenAIEmbedding (see examples/02_extraction_with_ollama.py).
    """
    async with await Engram.open(":memory:") as memory:
        # A scoped handle names its org and user once. Its methods take no scope,
        # so they cannot reach another one.
        alice = memory.scoped(org_id="acme", user_id="alice")
        await alice.record("Alice prefers espresso over drip.")
        await alice.record("Alice's brother lives in Tokyo.")
        await alice.record("Alice has a pet cat named Whiskers.")

        # Keyword-only query — FTS5 picks the espresso fact directly.
        results = await alice.recall("espresso", top_k=3)
        print("Top results for 'espresso':")
        for r in results:
            print(f"  [{r.score:.3f}] {r.fact.text}")


if __name__ == "__main__":
    asyncio.run(main())
