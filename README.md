# Engram

> Durable memory layer for AI agents. Temporal knowledge graph, semantic search, and MCP-native tools.

🚧 **v0.1.0a — Phase 1 of the Python rewrite is shipping.** Repo currently private; goes public at v0.1.0 GA. See `CHANGELOG.md` for what landed when.

The Rust v0.5.x implementation is in maintenance mode at [jamjet-labs/jamjet](https://github.com/jamjet-labs/jamjet/tree/main/runtime/engram).

---

## What's in here today

- ✅ Phase 1 — Pydantic schemas (`Fact`, `ChatMessage`, `Scope`, ...) + async SQLite + FTS5
- ✅ Phase 2 — Embedding providers (Ollama, OpenAI, Synthetic) + hnswlib vector index
- ✅ Phase 3 — Extraction pipeline + LLM backends (Ollama, OpenAI, Anthropic)
- ✅ Phase 4 — 6-signal hybrid retrieval + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- ✅ Phase 5 — `Engram` facade + FastAPI HTTP API + MCP server
- ✅ Phase 8 — Ingestion-time temporal grounding (`yesterday`, `last Tuesday`, `3 weeks ago` → ISO-8601)
- ✅ Phase 9 — Two-stage retrieval (top-K sessions, then facts within)
- ✅ Phase 10 — Question classifier (rule-based + LLM) + per-category token budgets
- ✅ Phase 12 — Reading layer (verifier-backed abstention, query decomposer, confidence-aware context)
- ✅ Phase 13 — Active fact-versioning (`supersede` API + retrieval filter) + determinism

164 tests, mypy `--strict`, ruff clean.

## Status

Functional parity with Rust v0.5.x: target **Week 7**.
LongMemEval ≥96%: target **Week 13**.

## Install (when 0.1.0 ships)

```bash
pip install jamjet-engram
```

## Quickstart

### Embedded mode (in-process)

```python
import asyncio
from engram import Engram

async def main():
    async with await Engram.open(":memory:") as memory:
        await memory.record(text="Alice prefers espresso.", user_id="alice")
        await memory.record(text="Alice's brother lives in Tokyo.", user_id="alice")
        results = await memory.recall(query="coffee preference", user_id="alice")
        for r in results:
            print(f"  [{r.score:.3f}] {r.fact.text}")

asyncio.run(main())
```

### With Ollama (local LLMs, no cloud keys)

```python
from engram import Engram
from engram.embedding.ollama import OllamaEmbedding
from engram.llm.ollama import OllamaLLM

memory = await Engram.open(
    "./engram.db",
    embedder=OllamaEmbedding(model="nomic-embed-text"),
    llm=OllamaLLM(model="llama3.2:3b"),
)
```

### With cross-encoder reranking

```python
from engram import Engram
from engram.retrieve.rerank import CrossEncoderReranker

memory = await Engram.open(
    "./engram.db",
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
```

### HTTP server

```python
import uvicorn
from engram import Engram
from engram.server.http import build_http_app

memory = await Engram.open("./engram.db")
app = build_http_app(memory)
uvicorn.run(app, host="127.0.0.1", port=19090)
```

Then:

```bash
curl -X POST http://127.0.0.1:19090/v1/memory/record \
     -H 'Content-Type: application/json' \
     -d '{"text": "alice prefers espresso", "user_id": "alice"}'

curl -X POST http://127.0.0.1:19090/v1/memory/search \
     -H 'Content-Type: application/json' \
     -d '{"query": "coffee", "user_id": "alice", "top_k": 5}'
```

### MCP server

```python
from engram.server.mcp import build_mcp_server
import mcp.server.stdio

server = build_mcp_server(memory)
async with mcp.server.stdio.stdio_server() as streams:
    await server.run(streams[0], streams[1], server.create_initialization_options())
```

Tools: `memory_record`, `memory_recall`, `memory_context` (names match Rust v0.5.x).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ engram.Engram (facade)                                   │
│                                                          │
│  record / record_message / extract / recall / context    │
│  supersede                                               │
└──────┬─────────────┬─────────────┬─────────────┬────────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Sqlite   │  │ Hnswlib  │  │ LLM +    │  │ Reader / │
│ Store    │  │ Vector   │  │ Extract  │  │ Verifier │
│ + FTS5   │  │ + Cosine │  │ Pipeline │  │ /Decomp. │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

   ▲                                            ▲
   │                                            │
   └─── HybridRetriever (vec + kw + temporal ───┘
        + rerank, two-stage session filter,
        + supersede filter)
```

## Reproducible runs

For benchmarks / comparison runs, set:

```bash
export PYTHONHASHSEED=42
```

The default `HnswVectorStore` uses `random_seed=42`. With identical
insertion order, retrieval is deterministic across runs.

## Why v2.0 in Python?

Short version:
- Researchers and contributors live in Python
- ML ecosystem (rerankers, embedders, agents) is Python-first
- ~50% faster iteration loop on benchmark climbing
- Wire protocol unchanged — Spring Boot starter, langchain4j, Java SDK keep working

Long version: see `docs/architecture.md` (forthcoming).

## License

MIT
