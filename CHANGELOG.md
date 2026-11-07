# Changelog

All notable changes to Engram v2.0 (the Python rewrite) are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Engram adheres to [Semantic Versioning](https://semver.org/) from v0.1.0 onward.

## [Unreleased]

Target version: 0.3.0.

### Added
- **Revocation enforced at retrieval.** `Engram.revoke()` writes a tombstone; `Engram.supersede()` now revokes the old fact and links the replacement. Revoked facts are withheld from every read path through one SQL view (`hidden_facts`), including vector hits from a stale index, keyword search, session ranking, events linked by `source_fact_ids`, transcript reads used by query-time re-extraction, the reader tools, HTTP and MCP. Motivated by arXiv 2609.08258.
- Content tombstones: re-recording revoked text, a trivial variant of it, or a fact, transcript turn or event that quotes it, is stored but withheld. `record(reject_revoked=True)` raises `RevokedContentError` instead. `Engram.reinstate()` on the revoked fact is the only way back and is audited. Query-time re-extraction and `extract()` drop withheld text.
- `record(derived_from=[...])`: a fact concluded from a revoked fact is withheld too, transitively.
- `Engram.erase()`: hard delete of a fact, its copies and quotes, derived facts, linked or quoting events and transcript turns, with a content-free `ErasureRecord`. Revoked content stays blocked after erasure unless `block_reinsertion=False`. Enables SQLite `secure_delete` and FTS5 `secure-delete`, checkpoints the WAL, and rebuilds the vector index without the erased vectors.
- Provenance: `ScoredFact.provenance`, `Engram.explain()`, `Engram.context_entries()`, `POST /v1/memory/recall_with_provenance`, and `include_provenance` on the MCP `memory_recall` tool. `record()` takes `actor` and `source_message_id`; the HTTP and MCP record calls take optional `actor` and `derived_from`.
- `Engram.revocations()` and `Engram.erasures()` list the audit records.
- `engram-server --allow-unauthenticated` (env `ENGRAM_ALLOW_UNAUTHENTICATED=1`): start without a token on a non-loopback host, with a startup warning. For deployments where a proxy in front does the auth.
- `engram-server --allowed-origin <origin>` (repeatable; env `ENGRAM_ALLOWED_ORIGINS`, comma-separated) and `build_http_app(allowed_origins=...)`: browser origins allowed to call the server. No CORS headers are added.
- `engram-server --allowed-host <name>` (repeatable; env `ENGRAM_ALLOWED_HOSTS`) and `build_http_app(allowed_hosts=...)`: hostnames, besides the loopback names, a server without a token will answer.
- `engram-server` allows its own loopback origins (`http://127.0.0.1:<port>`, `http://localhost:<port>`, `http://[::1]:<port>`), so "Try it out" on `/docs` works. `HEAD /healthz` returns 200.
- `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, `.github/pull_request_template.md` — inbound contribution scaffolding for the public-launch transition.
- `.github/workflows/release.yml` — automated release on `v*.*.*` tag push: builds wheel + sdist with `uv build`, publishes to PyPI via OIDC trusted publisher (no long-lived secrets), creates a GitHub Release with auto-generated notes. One-time PyPI side setup needed to register the trusted publisher (instructions in the workflow header).

### Changed
- **Breaking: the bearer token now gates every HTTP route, not only `/mcp`.** With `ENGRAM_AUTH_TOKEN` set, `/v1/memory/*`, `/v1/sessions/*`, `/docs` and `/openapi.json` return 401 without `Authorization: Bearer <token>`. Only `/healthz` stays open. REST clients that talk to a server or Docker image started with a token must now send it (the Java SDK reads `ENGRAM_TOKEN`). The image already refused to start without a token in 0.2.0; that is unchanged, and the startup error now says what to set.
- **Breaking: requests with a browser `Origin` header not on the allowlist get 403**, on every route except `/healthz`, including `build_http_app()` used as a library. The allowlist is empty by default. Requests with no `Origin` header are unaffected.
- **Breaking: a server without a token only answers requests whose `Host` is `127.0.0.1`, `localhost` or `::1`** (any port); others get 421. This includes `build_http_app()` used as a library, so a Starlette `TestClient` (which sends `Host: testserver`) needs a loopback `base_url` or `allowed_hosts=["testserver"]`.
- **Breaking: `build_http_app()` needs `auth_token="..."` or `allow_unauthenticated=True`.** Calling it with neither raises `ValueError` instead of returning an app anyone can use. Its Host, Origin and Fetch Metadata checks stop browsers being used against the app; they are not access control.
- Without a token, a request whose `Sec-Fetch-Site` is `cross-site` or `same-site` gets 403 unless it carries an allowed `Origin`, which covers `<img>` and `<script>` requests that send no `Origin`.
- `engram-server` refuses a token shorter than 16 characters on a host other machines can reach. On loopback it still only warns.
- `build_http_app(auth_token="")` raises `ValueError` instead of meaning "no auth". A token that is not printable ASCII without spaces is refused at startup (for example one ending in a newline from a secrets file), since no request could match it.
- `Engram.supersede()` returns the `Revocation` it wrote and accepts optional `actor` and `reason`. Superseded facts are now withheld even with `RetrievalConfig(exclude_superseded=False)`; that flag now only applies to `superseded_by` values set by hand.
- Version 0.3.0 in `pyproject.toml` (not yet published).
- New side tables (`fact_meta`, `fact_derivations`, `revocations`, `erasures`) and views are created on open; rows from 0.2 databases are backfilled with content hashes. No change to the existing HTTP response shapes.
- **License: MIT → Apache 2.0.** Aligns with `jamjet-labs/jamjet` (Rust runtime) and the rest of the agent-memory ecosystem (Mem0, Letta, Zep all Apache 2.0). Apache 2.0 also includes an explicit patent grant from contributors, valued by enterprise adopters. Clean switch — repo was private during the MIT period, no MIT-licensed copies are in the wild.

### Fixed
- `SqliteStore.upsert_fact`, `upsert_message` and `upsert_event` no longer overwrite a row whose id belongs to another scope (`ScopeError`), and `source_message_id` must name a message in the fact's scope.
- HNSW search returned no vector hits after a fact was deleted and added again (the live count included superseded labels, so `k` exceeded the live set and hnswlib raised).
- A bearer token with non-ASCII bytes made the auth check raise and return 500; it now gets 401.
- `benchmarks/smoke_runner.py` and `benchmarks/longmemeval_v2.py`: replaced developer-machine-specific oracle-path fallbacks (`../jamjet-research/...`) with a generic `./longmemeval_oracle.json` default. The `LONGMEMEVAL_ORACLE` env var still takes priority; the new default is a sensible cwd-relative path that fails fast with a clear message if the file isn't there.

### Security
- The `/v1/memory/*` and `/v1/sessions/*` REST routes were never auth-gated, even with `ENGRAM_AUTH_TOKEN` set and the server bound to `0.0.0.0` (the Docker default). Anyone who could reach the port could read and write any user's memory without the token. Fixed by the app-wide token check above.
- The MCP Streamable HTTP endpoint did not validate `Origin`, so a web page could reach a server on loopback through DNS rebinding. With `--no-auth` that meant full read and write access. Fixed by the Origin allowlist above, plus the `Host` check for servers without a token, since a rebound page's same-origin GETs carry no `Origin`.
- README: removed the claim of "no cross-tenant leakage by construction". `user_id` and `org_id` come from the request, so over HTTP and MCP one token is one trust domain: any holder can read and write every scope in that database. The README now says so and shows how to run one server per tenant.

## [0.2.0] — 2026-05-09

### Added
- `engram-server` script entry point (typer-based CLI). Run `engram-server --help` for flags.
- **MCP Streamable HTTP transport** at `POST /mcp` (and `GET /mcp`) on the FastAPI app, mounted alongside the existing `/v1/memory/*` REST routes.
- Bearer-token auth for `/mcp` via `$ENGRAM_AUTH_TOKEN` (fail-closed; server refuses to start if unset). `--no-auth` opt-out is rejected unless `--host` is loopback (`127.0.0.1` or `::1`).
- `Dockerfile` shipping `ghcr.io/jamjet-labs/engram-py-server:0.2.0` — `python:3.13-slim` base, non-root user, healthcheck, `/data` volume.
- `src/engram/server/auth.py` (bearer-token ASGI middleware) and `src/engram/server/transport.py` (Streamable HTTP adapter) modules.
- CI: tagged release also builds and pushes `ghcr.io/jamjet-labs/engram-py-server:<version>` (and `:latest` for non-prerelease) to GHCR.

### Changed
- `build_http_app(engram)` now accepts optional `mcp_server` and `auth_token` keyword arguments. Existing single-arg callers keep working.
- Bumped version in `pyproject.toml` to `0.2.0`. Added `typer>=0.12` to runtime dependencies.

### Deferred
- Stdio transport in the Python binary (CLI flag `--transport=stdio` reserved; rejects with a clear message in v0.2).
- OAuth2 dynamic client registration for `/mcp`.
- `/v1/memory/*` auth gating.
- Replacing the Rust `ghcr.io/jamjet-labs/engram-server` image. Stdio Claude Desktop users keep using v0.5.0.
- Tool surface expansion to match Rust's 11 tools (Engram v2 tool-port roadmap, separate workstream).

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

[unreleased]: https://github.com/jamjet-labs/engram/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jamjet-labs/engram/releases/tag/v0.2.0
[0.1.0]: https://github.com/jamjet-labs/engram/releases/tag/v0.1.0
