"""Extract facts from a conversation using local Ollama.

Prereq: `ollama pull llama3.2:3b && ollama pull nomic-embed-text`
Run: python examples/02_extraction_with_ollama.py
"""

import asyncio
from datetime import UTC, datetime

from engram import Engram
from engram.embedding.ollama import OllamaEmbedding
from engram.llm.ollama import OllamaLLM
from engram.models import ChatMessage
from engram.scope import Scope


async def main() -> None:
    async with await Engram.open(
        ":memory:",
        embedder=OllamaEmbedding(model="nomic-embed-text", dim=768),
        llm=OllamaLLM(model="llama3.2:3b"),
    ) as memory:
        scope = Scope(org_id="acme", user_id="alice")
        session_date = datetime(2024, 3, 12, tzinfo=UTC)
        msgs = [
            ChatMessage(
                scope=scope,
                session_id="s1",
                role="user",
                content="My name is Alice and I prefer espresso. I went to Tokyo last Tuesday.",
                timestamp=session_date,
            ),
            ChatMessage(
                scope=scope,
                session_id="s1",
                role="assistant",
                content="Got it! Tokyo is great this time of year.",
                timestamp=session_date,
            ),
        ]
        facts = await memory.extract(msgs, session_date=session_date)
        print(f"Extracted {len(facts)} facts:")
        for f in facts:
            ed = f.event_date.date().isoformat() if f.event_date else "no-date"
            print(f"  [{f.confidence:.2f}] [{ed}] {f.text}")

        results = await memory.recall(query="when was Tokyo trip?", user_id="alice", top_k=3)
        print("\nRecall on 'when was Tokyo trip?':")
        for r in results:
            print(f"  [{r.score:.3f}] {r.fact.text}")


if __name__ == "__main__":
    asyncio.run(main())
