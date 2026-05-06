# Engram

> Durable memory layer for AI agents. Temporal knowledge graph, semantic search, and MCP-native tools.

[![CI](https://github.com/jamjet-labs/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/jamjet-labs/engram/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**v0.1.0 GA** — Engram is the Python rewrite of jamjet's memory layer. See `CHANGELOG.md` for the release history.

The Rust v0.5.x implementation is in maintenance mode at [jamjet-labs/jamjet](https://github.com/jamjet-labs/jamjet/tree/main/runtime/engram).

---

## Features

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
- ✅ v0.1.0 — Per-category routing API: `Engram.record(role=...)`, `Engram.context(role_filter=...)`, `Reader(mode="recall" | "synthesis")`. Public re-exports for the full library surface. Lifted LongMemEval-S `single-session-preference` from 29% to 71%.

309 tests, mypy `--strict`, ruff clean. CI gates on coverage (`--cov-fail-under=79`) and downstream `py.typed` compatibility.

**Getting started:** see [`docs/quickstart.md`](docs/quickstart.md).

## Running in production

Engram is async-first SQLite + hnswlib by default. For multi-tenant deployments, a few practical notes:

**Scope isolation.** Every API call takes `user_id` + `org_id` (combined into a `Scope`). Facts, vectors, and chat-message storage are partitioned by scope at the SQL and HNSW levels — no cross-tenant leakage by construction. There is no "admin" scope that can read all data.

**Embedding provider tradeoffs.** `OllamaEmbedding` runs entirely local (private, free, slower) — recommended for sensitive data or offline usage. `OpenAIEmbedding` is faster and more accurate but sends every recorded text to OpenAI. `SyntheticEmbedding` is for tests only.

**LLM API keys.** Engram never stores API keys. They're read from the environment by the `OpenAILLM` / `AnthropicLLM` / `OllamaLLM` clients. Use a secrets manager in production.

**Rate limits.** The default extraction pipeline calls one LLM per session ingested. Bulk imports of large chat histories should chunk + rate-limit to fit your provider's tier. The `OpenAILLM` and `AnthropicLLM` clients use the official SDKs which handle retries with exponential backoff.

**Determinism.** `HnswVectorStore` defaults to `random_seed=42` for reproducible benchmark runs. Set `PYTHONHASHSEED=42` and use the same insertion order for byte-for-byte reproducibility.

**Per-category routing (v0.1.0).** For preference/recommendation questions, route to `Reader(mode="synthesis")` with `Engram.context(role_filter=("user",))`. Other categories take the default fact-recall reader. See [docs/quickstart.md](docs/quickstart.md#per-category-routing-preferences) for the canonical pattern.

## Roadmap

- 🎯 **LongMemEval-S ≥96%** (currently 71%) — frontier comparison is AgentMemory at 96.2% (full set). Documented gap-closing work in the benchmark section below.
- 🎯 Postgres backend (currently SQLite-only)
- 🎯 Performance benchmarks: recall latency + throughput
- 🎯 Migration guide from Rust v0.5.x

## Install

```bash
uv add jamjet-engram                # or: pip install jamjet-engram
uv add 'jamjet-engram[rerank]'      # optional cross-encoder reranker
```

Requires Python 3.12+.

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

**71.0%** on a 100-question stratified subset, configuration:

- Reader: `gpt-4o-mini` (default)
- Embedder: Ollama `nomic-embed-text` (768-dim, local)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (base, no fine-tune — see ablation notes)
- Pipeline flags: `--decompose --tools`
- Preference-aware routing: live in the benchmark harness (smoke runner branches preference-classified questions onto a recommendation-grounded synthesis path; library API follow-up tracked below)

| category | score | n |
|---|---|---|
| `single-session-assistant` | 88% | 17 |
| `temporal-reasoning` | 75% | 16 |
| `knowledge-update` | 71% | 17 |
| `single-session-user` | 69% | 16 |
| `single-session-preference` | 65% | 17 |
| `multi-session` | 59% | 17 |
| **overall** | **71%** | **100** |

This is **+3pp over the previous best (68%)** that shipped with the Tier 2-3 ablation programme — the lift comes from `single-session-preference` jumping 35% → 65% via the preference-aware routing described below. For comparison, [AgentMemory](https://arxiv.org/abs/2501.00309) reports 96.2% on this benchmark — Engram is currently well behind that frontier; documented follow-up work below.

### Reproduce

```bash
set -a && source /path/to/.env && set +a    # OPENAI_API_KEY required
export LONGMEMEVAL_ORACLE=/path/to/longmemeval_oracle.json
uv run python -m benchmarks.smoke_runner --n 100 --decompose --tools
```

The runner writes a per-question JSONL trace + a markdown report under `benchmarks/reports/`.

### What we tried, what worked, what didn't

Two ablation programmes have shipped to date.

**Tier 2-3 (April 2026):** 8 items implemented and ablated independently. Headline result: 64% → 68% (+4pp over the bare pipeline baseline).

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

**Preference uplift v2 (May 2026):** category-aware reading, after attempt-1 (a `PreferenceExpander` + tool + preamble approach) was rolled back as a null result. Headline: **single-session-preference 29% → 65% (+36pp); overall 64% → 71% (+7pp)**, zero per-category regressions.

| item | what it does | shipped | net |
|---|---|---|---|
| `is_preference_question` predicate | catches conversational advice queries the rule classifier mis-tags ("any tips on guitars?", "recommend a movie") | ✓ live | 76% recall, 87% precision on 100q stratified |
| User-only ingest for preference questions | filters out assistant turns at ingest so user's terse preference statements aren't buried under info-dense assistant explanations | ✓ live (smoke runner) | rank-1 retrieval of user preferences |
| Synthesis-mode reader prompt | recommendation-grounded prompt; bypasses verifier + tool loop + escalation rungs | ✓ live (smoke runner) | 77% solve rate on activated questions (10/13) |

Diagnosis from attempt-1's failure: the verifier short-circuits the new path before tools fire (NO verdict on sparse context), AND topic-shaped subquery expansion ranks assistant explanations above terse user preferences (the user's relevant chunk dropped from rank 2 → outside top 5). The verifier-gated `Reader` is calibrated for fact-recall; preference questions are recommendation tasks — wrong tool for the job. Synthesis mode replaces the read path entirely for that one category, gated on the predicate.

The negative results across both programmes are as informative as the positive ones.

### Documented follow-up work

**Promote v2 from benchmark to library:** the synthesis path currently lives only inside `benchmarks/smoke_runner.py`. Library users importing `engram` get the existing verifier-gated reader. Next plan: add `Engram.context(role_filter=...)` parameter, `Reader(mode="recall" | "synthesis")`, and a `SYNTHESIS_PROMPT` constant in `src/engram/read/prompts.py` so callers can opt in to per-category routing.

**Tighten the preference predicate:** current 76% recall caps preference category at ~13/17 = 76% even with perfect synthesis solving. The 4 missed shapes are decision-help questions ("what do you think?", "would you recommend X?") that don't match the current regex. Lifting recall to ~95% with 77% solve rate would push the category to ~76%.

**Cross-model validation:** repeat the v2 smoke with Sonnet-4.6 reader to confirm the synthesis prompt generalizes.

**From Tier 2-3 (still open):**
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
