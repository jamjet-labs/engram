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

## Benchmarks — LongMemEval-S

Engram's reference benchmark is [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024) — 500 questions about a long synthetic chat history. We use the official `gpt-4o-mini` judge for scoring.

### Latest result

**68.0%** on a 100-question stratified subset, configuration:

- Reader: `gpt-4o-mini` (default)
- Embedder: Ollama `nomic-embed-text` (768-dim, local)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (base, no fine-tune — see ablation notes)
- Pipeline flags: `--decompose --tools`

| category | score | n |
|---|---|---|
| `single-session-assistant` | 88% | 17 |
| `knowledge-update` | 76% | 17 |
| `temporal-reasoning` | 75% | 16 |
| `single-session-user` | 69% | 16 |
| `multi-session` | 65% | 17 |
| `single-session-preference` | 35% | 17 |
| **overall** | **68%** | **100** |

This is **+4pp over the bare-pipeline baseline** (no `--decompose`, no `--tools`). For comparison, [AgentMemory](https://arxiv.org/abs/2501.00309) reports 96.2% on this benchmark — Engram is currently well behind that frontier; documented follow-up work below.

### Reproduce

```bash
set -a && source /path/to/.env && set +a    # OPENAI_API_KEY required
export LONGMEMEVAL_ORACLE=/path/to/longmemeval_oracle.json
uv run python -m benchmarks.smoke_runner --n 100 --decompose --tools
```

The runner writes a per-question JSONL trace + a markdown report under `benchmarks/reports/`.

### What we tried, what worked, what didn't

Full programme write-up: [`docs/superpowers/specs/2026-04-30-engram-v2-tier-2-3-design.md`](docs/superpowers/specs/2026-04-30-engram-v2-tier-2-3-design.md). 8 items implemented and ablated independently:

| item | what it does | shipped | net |
|---|---|---|---|
| Model tier scaffolding | split reader / utility LLMs | ✓ default | gpt-4o-mini stays default; Sonnet alternate available |
| Decomposer wiring (`--decompose`) | split compound questions, RRF fuse | ✓ ON | +1pp overall, +6pp multi-session |
| Tool-augmented reader (`--tools`) | text-protocol tool calls (search_facts, search_events, solve_temporal, count_between, add_days, days_between) | ✓ ON | +3pp overall |
| Programmatic temporal solver (`--solver`) | parse to DSL, deterministic SVO calendar lookup | flag, default OFF | parser too eager; misfires on knowledge-update |
| Query-time re-extraction (`--reextract`) | re-extract candidate sessions on PARTIAL verdict | flag, default OFF | no detectable lift at n=50 |
| Adaptive self-consistency (`--self-consistency`) | N=3 reader samples + vote on PARTIAL | flag, default OFF | gpt-4o-mini already too deterministic |
| ReAct retrieval agent (`--react`) | multi-hop tool-using agent fallback | flag, default OFF | overwrites borderline-correct answers with worse ones |
| Fine-tuned cross-encoder (`--ft-cross-encoder`) | LongMemEval-trained MiniLM | flag, default OFF | +3 nDCG@10 but -7pp downstream — labels misaligned with multi-session task structure |

The negative results are as informative as the positive ones — the spec doc explains the diagnosis for each.

### Documented follow-up work

- Native Anthropic tool-use API path (would unlock Sonnet at +5-10pp on tool-augmented runs)
- Question-type-aware cross-encoder training labels (different positive criteria per LongMemEval category)
- Stricter solver pre-gate (require explicit anchor words like "before"/"after")
- Verifier calibration to reduce ReAct false-fires

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
