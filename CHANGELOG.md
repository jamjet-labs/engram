# Changelog

All notable changes to Engram v2.0 (the Python rewrite) are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Engram adheres to [Semantic Versioning](https://semver.org/) from v0.1.0 onward.

## [Unreleased]

### Changed
- **License: MIT → Apache 2.0.** Aligns with `jamjet-labs/jamjet` (Rust runtime) and the rest of the agent-memory ecosystem (Mem0, Letta, Zep all Apache 2.0). Apache 2.0 also includes an explicit patent grant from contributors, valued by enterprise adopters. Clean switch — repo was private during the MIT period, no MIT-licensed copies are in the wild.

### Added
- `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, `.github/pull_request_template.md` — inbound contribution scaffolding for the public-launch transition.

### Fixed
- `benchmarks/smoke_runner.py` and `benchmarks/longmemeval_v2.py`: replaced developer-machine-specific oracle-path fallbacks (`../jamjet-research/...`) with a generic `./longmemeval_oracle.json` default. The `LONGMEMEVAL_ORACLE` env var still takes priority; the new default is a sensible cwd-relative path that fails fast with a clear message if the file isn't there.

## [0.1.0] - 2026-05-06

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

#### Tier 2-3 ablation programme (April 2026)
- 8 retrieval/reading techniques implemented and ablated independently on LongMemEval-S 100q
- 5 shipped behind flags default OFF (`--solver`, `--reextract`, `--self-consistency`, `--react`, `--ft-cross-encoder`)
- 3 shipped default ON (model tier, `--decompose`, `--tools`)
- Net: 64% → 68% on LongMemEval-S 100q
- Detailed write-up in commit history (PR #1)

#### Preference uplift v1 (May 2026, rolled back)
- `feat/preference-uplift` (tagged `preference-uplift-attempt-1`) shipped a `PreferenceExpander` + `SearchPreferencesTool` + reader-prompt preamble
- n=100 result: 29% → 29% (no change). Rolled back, kept for reference
- Diagnostic: verifier short-circuited the new path before tools fired
- Lesson: verifier-gated readers are hostile to "retrieve-more" rungs (saved to project memory)

#### Preference uplift v2 (May 2026, shipped — PR #2)
- `is_preference_question` permissive predicate at `engram.read.preference_gate`
- Smoke-runner-only synthesis path (recommendation-grounded reader prompt, user-only ingest, verifier off)
- n=100 result: 29% → 65% on `single-session-preference`; 64% → 71% overall; zero per-category regressions
- The architectural insight: LongMemEval preference questions are recommendation tasks, not fact-recall tasks — different judge criteria need a different read path

#### CI green (PR #3)
- mypy `--strict` now passes (was 2 errors in `read/reader.py` and `read/reextract.py`)
- ruff check + ruff format both clean across `src/engram` and `tests/`
- Latent runtime bug fixed: `read/reextract.py` was calling `list_messages_by_session` which didn't exist on the store; renamed to `list_messages` (the actual method)

#### Per-category routing API (PR #4 — v0.1.0 GA)
- `Engram.record(role=...)` propagates role to `metadata["role"]`
- `Engram.context(role_filter=...)` filters recall by stored role
- `Reader(mode="recall" | "synthesis")` coupled package: synthesis mode uses `SYNTHESIS_PROMPT`, skips verifier, skips tool loop, skips escalation
- `SYNTHESIS_PROMPT` constant in `src/engram/read/prompts.py`
- Public re-exports in `engram/__init__.py` (22 symbols): `Engram`, `Scope`, `Fact`, `ChatMessage`, `Reader`, `ReaderConfig`, `RetrievalConfig`, `QuestionType`, `RuleBasedClassifier`, `Polarity`, `MemoryTier`, `is_preference_question`, `Tool`, `ToolRegistry`, `ToolResult`, `LLMClient`, `LLMMessage`, `LLMResponse`, `EngramError`, `ExtractionError`, `NotFoundError`, `StoreError`
- Smoke runner refactored to use the new public APIs
- Quickstart docs at `docs/quickstart.md`
- New example `examples/05_preference_synthesis.py`
- `CONTRIBUTING.md`, `SECURITY.md`

### CI
- Coverage gate (`--cov-fail-under=79`)
- Downstream `py.typed` smoke-check job — builds wheel, installs in fresh venv, runs `mypy --strict` on a stub importing all 22 public re-exports

### Tests
~280 tests (units + integration). mypy `--strict` clean. ruff + ruff format clean. CI green on Python 3.12 + 3.13.

[unreleased]: https://github.com/jamjet-labs/engram/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jamjet-labs/engram/releases/tag/v0.1.0
