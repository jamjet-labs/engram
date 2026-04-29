# Changelog

All notable changes to Engram v2.0 are documented here.

## [Unreleased] (v0.1.0a, 2026-04-30)

### Added

#### Phase 1 — repo bootstrap + storage schema
- Pydantic v2 schemas: `Fact`, `ExtractedFact`, `ChatMessage`, `Entity`, `Relationship`, `Scope`, `MemoryTier`, `Polarity`
- `SqliteStore` (async, FTS5, scope-isolated multi-tenancy, idempotent upserts, async context manager)
- `EngramStore` protocol
- Error hierarchy (`EngramError`, `StoreError`, `NotFoundError`, ...)
- CI: ruff + ruff format + mypy --strict + pytest on Python 3.12 and 3.13

#### Phase 2 — embedding + vector storage
- `EmbeddingProvider` protocol; `SyntheticEmbedding`, `OllamaEmbedding`, `OpenAIEmbedding`
- `VectorStore` protocol with `VectorMatch` Pydantic model
- `HnswVectorStore` — per-scope hnswlib indexes, cosine similarity, deterministic `random_seed=42`

#### Phase 3 — extraction + LLM clients
- `LLMClient` protocol with `LLMMessage` / `LLMResponse`
- `OllamaLLM`, `OpenAILLM`, `AnthropicLLM` backends
- `ExtractionPipeline` with strict JSON parsing + tolerant fact validation

#### Phase 4 — hybrid retrieval + reranking
- `HybridRetriever`: 6-signal scoring (vector + keyword + temporal, optional rerank)
- `RetrievalConfig` with AgentMemory-style weights (vector=0.55, keyword=0.30, temporal=0.15)
- `Reranker` protocol; `CrossEncoderReranker` via `sentence-transformers`
- `detect_temporal_intent` (RECENCY / DURATION / ORDERING / POINT_IN_TIME)

#### Phase 5 — server surfaces + facade
- `Engram` facade — embedded-mode entry point, async context manager
- FastAPI HTTP server (`/v1/memory/record|search|recall|context|extract|raw_facts`, sessions/messages, healthz)
- MCP server (`memory_record`, `memory_recall`, `memory_context` tools)

#### Phase 8 — temporal grounding
- `RelativeDateResolver` — regex-based parser for "yesterday", "last Tuesday", "N units ago", etc.
- `resolve_relative_dates` batch helper, anchored to session date
- `ExtractionPipeline` auto-resolves missing `event_date` via the resolver
- HybridRetriever computes `temporal_score` (Gaussian decay) when query has temporal intent

#### Phase 9 — two-stage (session-first) retrieval
- `Fact.session_id` field; persisted by SqliteStore
- `EngramStore.aggregate_sessions(query, scope, top_sessions)` — Stage 1 ranking via FTS5 BM25
- `RetrievalConfig.enable_two_stage` + `two_stage_top_sessions` opt-in
- Falls through to global retrieval when no facts have session_id

#### Phase 10 — question classifier + per-category budgets
- `QuestionType` StrEnum mirroring LongMemEval taxonomy
- `RuleBasedClassifier` — regex precedence (multi-session > knowledge-update > temporal > assistant > preference > user)
- `LLMClassifier` — single LLM call with rule-based fallback
- `CATEGORY_BUDGETS` from AgentMemory (1500-7500 tokens)
- `Engram.context(classifier=...)` auto-picks budget per category

#### Phase 12 — reading layer hardening
- `Reader` — verifier-then-reader pipeline; abstains on `verdict=NO`
- `ReadResult` Pydantic model
- `QueryDecomposer` — LLM-driven compound-question splitter
- `format_context_with_confidence` helper for `[confidence: 0.92]` annotations

#### Phase 13 — active fact-versioning + determinism
- `Engram.supersede(old_id, new_id)` — explicit version transition API
- `RetrievalConfig.exclude_superseded` (default True) — filter superseded facts from default recall
- `source_span` round-trip through SQLite verified
- Determinism smoke test: same seed + same insertion order → same top-K

### Tests
164 tests. mypy `--strict` clean. ruff + ruff format clean. CI green on Python 3.12 + 3.13.
