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
        await memory.record(text="Alice prefers espresso over drip.", user_id="alice")
        await memory.record(text="Alice's brother lives in Tokyo.", user_id="alice")
        await memory.record(text="Alice has a pet cat named Whiskers.", user_id="alice")

        # Keyword-only query — FTS5 picks the espresso fact directly.
        results = await memory.recall(query="espresso", user_id="alice", top_k=3)
        print("Top results for 'espresso':")
        for r in results:
            print(f"  [{r.score:.3f}] {r.fact.text}")


if __name__ == "__main__":
    asyncio.run(main())
