# Engram

> Durable memory layer for AI agents. Temporal knowledge graph, semantic search, and MCP-native tools.

**Engram v2.0 is in active development.** This is the Python rewrite. The Rust v0.5.x implementation is in maintenance mode at [jamjet-labs/jamjet](https://github.com/jamjet-labs/jamjet/tree/main/runtime/engram).

## Status

🚧 v0.1.0a0 in progress (Phase 1 of 13).
- Functional parity with Rust v0.5.x: target Week 7.
- LongMemEval ≥96%: target Week 13.
- Repo currently private; goes public at v0.1.0 GA.

## Install (when 0.1.0 ships)

```bash
pip install jamjet-engram
```

## Quickstart (preview)

```python
from engram import Engram

async with Engram.open("./engram.db") as memory:
    await memory.record(user_id="alice", text="I prefer espresso over drip.")
    facts = await memory.recall(user_id="alice", query="coffee preference")
```

## Why v2.0 in Python?

Short version:
- Researchers and contributors live in Python
- ML ecosystem (rerankers, embedders, agents) is Python-first
- ~50% faster iteration loop on benchmark climbing
- Wire protocol unchanged — Spring Boot starter, langchain4j, Java SDK keep working

## License

MIT
