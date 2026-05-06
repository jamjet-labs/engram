# Engram

> **Durable memory for AI agents.** Multi-tenant by default. Local-first or cloud. Benchmarked, not vapor.

[![CI](https://github.com/jamjet-labs/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/jamjet-labs/engram/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-jamjet--engram-orange)](https://pypi.org/project/jamjet-engram)

```python
from engram import Engram

async with await Engram.open(":memory:") as memory:
    await memory.record("Alice prefers espresso over drip.", user_id="alice", role="user")
    await memory.record("Alice's brother lives in Tokyo.",   user_id="alice", role="user")

    results = await memory.recall(query="what does alice drink?", user_id="alice", top_k=1)
    print(results[0].fact.text)
    #  Alice prefers espresso over drip.
```

That's the whole shape. Record what users (or agents) say; ask questions later; get back the relevant facts. With per-user isolation, temporal grounding, and a recommendation mode that goes beyond simple recall.

---

## Why Engram?

There are several memory libraries for agents already ([Mem0](https://github.com/mem0ai/mem0), [Letta](https://github.com/letta-ai/letta), [Zep](https://github.com/getzep/zep), and the published [AgentMemory](https://arxiv.org/abs/2501.00309)). Engram makes a few specific bets:

**1. Mode-aware reading.** Most memory layers do one thing: retrieve facts that match your query. Engram ships two distinct read paths: `Reader(mode="recall")` for "what did the user say about X?" and `Reader(mode="synthesis")` for "recommend something based on the user's preferences." On LongMemEval-S, this lifts `single-session-preference` accuracy from 29% to 71%.

**2. Multi-tenant by default.** Every API takes `user_id` + `org_id` (a `Scope`). Facts, vectors, and chat messages are partitioned at the SQL and HNSW levels — no cross-tenant leakage by construction. There is no "admin" scope that can read across tenants.

**3. Local-first OR cloud.** Pluggable providers for both LLM (Ollama / OpenAI / Anthropic) and embeddings (Ollama / OpenAI / synthetic). Run completely offline with `nomic-embed-text` + `llama3.2:3b`, or swap in cloud providers when you need throughput.

**4. Active fact versioning.** `Engram.supersede(old_id, new_id)` marks a fact as replaced. Recall filters superseded facts by default. So when a user changes a preference, the old one doesn't poison future answers.

**5. Benchmarked, not vapor.** Reproducible LongMemEval-S runs in [`benchmarks/`](benchmarks/). We publish overall and per-category numbers (currently **71% overall**, see below). The 25-point gap to AgentMemory's frontier 96.2% is a public roadmap item — not hidden.

**6. MCP-native.** Ships an [MCP server](src/engram/server/mcp.py) so Claude Code, Cursor, and other agentic tools can use Engram memory directly. Three tools: `memory_record`, `memory_recall`, `memory_context`.

---

## Install

```bash
uv add jamjet-engram                  # or: pip install jamjet-engram
uv add 'jamjet-engram[rerank]'        # cross-encoder reranker (recommended)
```

Requires Python 3.12+. SQLite (stdlib) and `hnswlib` (auto-installed) are the only persistence dependencies.

---

## 60-second tour

### Record + recall

```python
import asyncio
from engram import Engram

async def main():
    async with await Engram.open(":memory:") as memory:
        await memory.record("user prefers oat milk", user_id="alice")
        await memory.record("user drinks coffee twice a day", user_id="alice")
        await memory.record("user's dog is named Whiskey", user_id="alice")

        results = await memory.recall(query="coffee", user_id="alice", top_k=3)
        for r in results:
            print(f"  [{r.score:.2f}]  {r.fact.text}")

asyncio.run(main())
```

### Build a context window for an LLM

```python
ctx = await memory.context(
    query="what should I order at the cafe?",
    user_id="alice",
    token_budget=2000,
)
# ctx is a newline-joined string ready to drop into a system prompt
```

### Per-category routing (the differentiator)

For preference/recommendation questions, route to synthesis mode — the reader generates a recommendation grounded in stored user preferences instead of just listing facts.

```python
from engram import Engram, Reader, RuleBasedClassifier, is_preference_question
from engram.llm.tier import ModelTier

tier = ModelTier.default()  # gpt-4o-mini
classifier = RuleBasedClassifier()

async with await Engram.open(":memory:", tier=tier) as memory:
    for turn in conversation_history:
        await memory.record(text=turn["content"], role=turn["role"], user_id="alice")

    question = "Recommend a movie for tonight"
    qt = await classifier.classify(question)

    if is_preference_question(question, qt):
        ctx = await memory.context(query=question, user_id="alice", role_filter=("user",))
        reader = Reader(tier.reader, mode="synthesis")
    else:
        ctx = await memory.context(query=question, user_id="alice", classifier=classifier)
        reader = Reader(tier.reader)

    res = await reader.read(question=question, context=ctx)
    print(res.answer)
```

Full walk-through: [`docs/quickstart.md`](docs/quickstart.md). Working example: [`examples/05_preference_synthesis.py`](examples/05_preference_synthesis.py).

### Run as an HTTP server

```python
import uvicorn
from engram import Engram
from engram.server.http import build_http_app

memory = await Engram.open("./engram.db")
uvicorn.run(build_http_app(memory), host="127.0.0.1", port=19090)
```

```bash
curl -sX POST localhost:19090/v1/memory/record \
  -H 'Content-Type: application/json' \
  -d '{"text": "alice prefers espresso", "user_id": "alice"}'
```

Endpoints: `record`, `extract`, `recall`, `search`, `context`, `raw_facts`, sessions, messages, healthz.

### Run as an MCP server (for Claude Code, Cursor, etc.)

```python
from engram.server.mcp import build_mcp_server
import mcp.server.stdio

server = build_mcp_server(memory)
async with mcp.server.stdio.stdio_server() as streams:
    await server.run(streams[0], streams[1], server.create_initialization_options())
```

---

## Core concepts

| Concept | What it is | Why you care |
|---|---|---|
| **`Scope`** | `(org_id, user_id)` pair attached to every fact | Multi-tenancy. No cross-tenant leakage at the SQL or vector index layer. |
| **`Fact`** | Pydantic model: text, scope, validity window, confidence, optional event_date / session_id / role / metadata | The unit of memory. Versioned via `supersede`, scored at retrieval, governed by per-tenant scope. |
| **`Engram`** | Async facade over store + embedder + retriever + (optional) extractor + (optional) tier | The thing you actually call. `record()`, `recall()`, `context()`, `extract()`. |
| **`Reader`** | LLM-driven answer generator with two modes: `"recall"` (verifier-gated fact recall) and `"synthesis"` (recommendation grounded in user preferences) | The mode you pick changes the whole pipeline — verifier on or off, tool loop or not, recall prompt or synthesis prompt. |
| **`RuleBasedClassifier`** | Maps a question to one of the 6 LongMemEval question types | Used for per-category token budgets and to decide between recall vs synthesis. Override with your own `QuestionClassifier` if you need different categories. |
| **`MemoryTier`** | `WORKING` / `EPISODIC` / `SEMANTIC` enum on each fact | Optional structuring of memory by time horizon. The store doesn't enforce it; your code does. |
| **`Tool` + `ToolRegistry`** | The reader's tool-use protocol (text-marker `[TOOL_USE]{...}[/TOOL_USE]`) | When `mode="recall"`, the reader can call tools mid-generation. Six built-in tools (search_facts, search_events, solve_temporal, count_between, add_days, days_between). Bring your own. |

All importable from the top-level package: `from engram import Engram, Scope, Fact, Reader, ...`.

---

## Architecture (one paragraph + a diagram)

Three layers under the `Engram` facade. **Storage** = `SqliteStore` (FTS5 for keyword) + `HnswVectorStore` (cosine similarity). **Retrieval** = `HybridRetriever` blends vector / keyword / temporal scoring with optional cross-encoder rerank. **Reading** = `Reader` (verifier-gated recall) or synthesis-mode (verifier off, recommendation prompt). Optional extras: `ExtractionPipeline` (LLM extracts facts from chat turns), `EventExtractor` (SVO event calendar for temporal solvers), `QueryDecomposer` (splits compound questions), and a tool registry the reader can call mid-generation. Everything is async; everything is scope-isolated.

```
                   ┌──────────────────────────────────────────────┐
                   │  Engram (async facade)                       │
                   │  record  •  recall  •  context  •  extract  │
                   │  supersede  •  record_message               │
                   └───────────┬───────────┬──────────┬──────────┘
              ┌────────────────┘           │          └──────────────┐
              ▼                            ▼                         ▼
   ┌──────────────────┐          ┌──────────────────┐      ┌──────────────────┐
   │  SqliteStore     │          │  HnswVectorStore │      │  Reader          │
   │  facts + msgs    │          │  per-scope HNSW  │      │  recall mode     │
   │  FTS5 keyword    │          │  cosine          │      │   ↳ verifier     │
   │  scope-isolated  │          │  deterministic   │      │   ↳ tool loop    │
   └──────────────────┘          └──────────────────┘      │   ↳ escalation   │
              │                            │                │  synthesis mode  │
              └─────────────┬──────────────┘                │   ↳ direct LLM   │
                            ▼                                └──────────────────┘
                  ┌──────────────────┐
                  │  HybridRetriever │  vector + keyword + temporal scoring
                  │  + reranker      │  cross-encoder rerank (optional)
                  └──────────────────┘
```

Files to start exploring: [`src/engram/engram.py`](src/engram/engram.py) is the facade. [`src/engram/read/reader.py`](src/engram/read/reader.py) is where the two modes live. [`benchmarks/smoke_runner.py`](benchmarks/smoke_runner.py) is the LongMemEval harness.

---

## Benchmarks — LongMemEval-S

Engram is benchmarked against [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024) — 500 questions about a long synthetic chat history, judged by `gpt-4o-mini`.

### Latest result (v0.1.0)

**71.0%** on a 100-question stratified subset (`gpt-4o-mini` reader, `--decompose --tools`, preference-aware routing on).

| category | score | n |
|---|---|---|
| `single-session-assistant` | 88% | 17 |
| `temporal-reasoning` | 75% | 16 |
| `single-session-preference` | 71% | 17 |
| `knowledge-update` | 71% | 17 |
| `single-session-user` | 69% | 16 |
| `multi-session` | 59% | 17 |
| **overall** | **71%** | **100** |

Frontier comparison: AgentMemory reports **96.2%** on the full 500. Engram is currently 25pp behind. Documented gap-closing work in [Roadmap](#roadmap).

### Reproduce

```bash
set -a && source /path/to/.env && set +a    # OPENAI_API_KEY required
export LONGMEMEVAL_ORACLE=/path/to/longmemeval_oracle.json
uv run python -m benchmarks.smoke_runner --n 100 --decompose --tools
```

The runner writes a per-question JSONL trace + a markdown report to `benchmarks/reports/`. Set `PYTHONHASHSEED=42` for fully deterministic insertion order.

### What we tried, what worked, what didn't

The benchmark history (visible in `CHANGELOG.md`) documents two ablation programmes — a **Tier 2-3 batch** (April 2026, 64% → 68%, 8 techniques ablated independently) and a **preference uplift** (May 2026, 29% → 71% on `single-session-preference` after a failed attempt-1 led to a redesign). We publish the negative results too: `--ft-cross-encoder` looked promising on IR metrics but lost 7pp downstream because the training labels misaligned with multi-session task structure; `--reextract` and `--self-consistency` didn't help because the verifier short-circuited them. Honest history makes the benchmark numbers credible.

---

## Running in production

**Scope isolation.** Every API call takes `user_id` + `org_id` (combined into a `Scope`). Facts, vectors, and chat-message storage are partitioned by scope at the SQL and HNSW levels — no cross-tenant leakage by construction. There is no "admin" scope that can read all data.

**Embedding provider tradeoffs.** `OllamaEmbedding` runs entirely local (private, free, slower) — recommended for sensitive data or offline usage. `OpenAIEmbedding` is faster and more accurate but sends every recorded text to OpenAI. `SyntheticEmbedding` is for tests only.

**LLM API keys.** Engram never stores API keys. They're read from the environment by the `OpenAILLM` / `AnthropicLLM` / `OllamaLLM` clients. Use a secrets manager in production.

**Rate limits.** The default extraction pipeline calls one LLM per session ingested. Bulk imports of large chat histories should chunk + rate-limit to fit your provider's tier. The `OpenAILLM` and `AnthropicLLM` clients use the official SDKs which handle retries with exponential backoff.

**Determinism.** `HnswVectorStore` defaults to `random_seed=42` for reproducible benchmark runs. Set `PYTHONHASHSEED=42` and use the same insertion order for byte-for-byte reproducibility.

**Per-category routing.** For preference/recommendation questions, route to `Reader(mode="synthesis")` with `Engram.context(role_filter=("user",))`. Other categories take the default fact-recall reader. See [`docs/quickstart.md`](docs/quickstart.md) for the canonical pattern.

---

## Contributing

We genuinely welcome contributions. The repo is set up so a new contributor can ship a meaningful change in a couple of hours.

### Quick start

```bash
git clone https://github.com/jamjet-labs/engram.git
cd engram
uv sync --all-extras
uv run pytest                # 309 tests, ~50s
uv run ruff check && uv run ruff format --check && uv run mypy src/engram
```

CI gates on all three. Coverage floor is 79% (`--cov-fail-under=79`) and a downstream `py.typed` job builds the wheel and verifies `mypy --strict` passes for library users.

Full developer guide: [CONTRIBUTING.md](CONTRIBUTING.md). Security policy: [SECURITY.md](SECURITY.md).

### Where contributions are most welcome

| Area | What you'd do | Difficulty |
|---|---|---|
| **New embedding provider** | Implement `EmbeddingProvider` for Cohere / Voyage / local-sbert / etc. Pattern in [`src/engram/embedding/`](src/engram/embedding/). | Easy — single file + tests |
| **New LLM provider** | Implement `LLMClient` for Vertex / Together / etc. Pattern in [`src/engram/llm/`](src/engram/llm/). | Easy |
| **New tool for the reader** | Implement `Tool` for domain-specific work (e.g., `search_calendar`, `query_database`). Pattern in [`src/engram/tools/`](src/engram/tools/). | Easy |
| **Tighten preference predicate recall** | Currently 76% on LongMemEval-S — misses "what do you think?" decision-help shapes. ~5 questions worth of headroom. See [`src/engram/read/preference_gate.py`](src/engram/read/preference_gate.py). | Medium — requires benchmark validation |
| **Postgres backend** | Implement `EngramStore` against Postgres. Schema in [`src/engram/store/`](src/engram/store/). | Medium — schema + tests + CI matrix entry |
| **Performance benchmarks** | Recall latency / throughput at scale. There's no baseline yet — open territory. | Medium |
| **Migration guide from Rust v0.5.x** | Document the API mapping between the Rust runtime and the Python rewrite. | Easy if you've used both |
| **Move LongMemEval ≥80%** | The 25pp gap to AgentMemory's 96.2% is the headline gap. Pick a category (multi-session at 59% is the lowest) and find a +5pp improvement. | Hard — but high-impact |

We tag issues with `good first issue` for the first three categories. Open an issue before starting a large change so we can sanity-check the approach.

### What makes a good PR here

- A failing test first (TDD is enforced by the CI checks)
- For benchmark-affecting changes, an n=100 smoke result vs the published baseline (within ±2pp on non-target categories, real lift on the target)
- A `CHANGELOG.md` entry under `[Unreleased]`
- Conventional commit message: `feat(scope): description` or `fix(scope): description`

### How decisions get made

Architectural changes go through a brainstorm → spec → plan → implementation cycle. The specs and plans are local-only (gitignored under `docs/superpowers/`); the *outcomes* land in CHANGELOG entries with PR references. We document negative results too — see the "Preference uplift v1 (rolled back)" entry for a worked example of a strategy that didn't move the metric and what we learned.

---

## Roadmap

In rough priority order:

- **LongMemEval-S ≥85%** — the top headline. Currently 71%; AgentMemory frontier is 96.2%. Multi-session at 59% is the lowest-hanging fruit.
- **Postgres backend** — currently SQLite-only. Schema is straightforward; needs a `EngramStore` Postgres implementation + CI matrix entry.
- **Sonnet / Haiku reader compatibility** — the v0.1.0 synthesis prompt is calibrated against `gpt-4o-mini`. Sonnet/Haiku regress on tool-protocol parsing. Native Anthropic tool-use API path would unlock +5–10pp on tool-augmented runs.
- **Performance benchmarks** — recall latency, throughput, memory footprint at scale. No published baseline yet.
- **Per-category synthesis modes for non-preference categories** — only `single-session-preference` currently uses synthesis mode. Multi-session might benefit from a different tailored prompt.
- **Migration guide from Rust v0.5.x** — the Rust runtime stays in maintenance mode at [jamjet-labs/jamjet](https://github.com/jamjet-labs/jamjet/tree/main/runtime/engram); a side-by-side API map would help users migrate.
- **Cross-language client SDKs** — Engram is Python today. The HTTP API is stable enough that a TypeScript / Go client is mostly mechanical.

If any of these are interesting, open an issue and let's talk.

---

## Project meta

- **License:** Apache 2.0 — see [LICENSE](LICENSE).
- **Security:** Report privately to security@jamjet.dev. See [SECURITY.md](SECURITY.md).
- **Changelog:** [CHANGELOG.md](CHANGELOG.md).
- **Rust v0.5.x runtime:** in maintenance mode at [jamjet-labs/jamjet/runtime/engram](https://github.com/jamjet-labs/jamjet/tree/main/runtime/engram). Wire protocol stays compatible.
- **Examples:** [`examples/`](examples/) — start with `01_quickstart.py`.
- **Architecture deep-dive:** `docs/architecture.md` (forthcoming).

---

*Engram is built by [JamJet Labs](https://jamjet.dev). The benchmark targets and design decisions are public; the failures (and there have been a few) are documented in CHANGELOG. If you'd like to help us push past 71% — please do.*
