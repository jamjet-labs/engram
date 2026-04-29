# Changelog

All notable changes to Engram v2.0 are documented here.

## [Unreleased]

### Added (Phase 1, 2026-04-30)
- Pydantic v2 schemas: `Fact`, `ExtractedFact`, `ChatMessage`, `Entity`, `Relationship`, `Scope`, `MemoryTier`, `Polarity`
- `SqliteStore` async backend with FTS5 keyword search, per-turn message storage, scope-isolated multi-tenancy, async context manager, idempotent upserts, access counting
- `EngramStore` protocol for backend pluggability
- Error hierarchy mirroring Rust v0.5.x `MemoryError`
- 34 tests (17 model, 15 store, 2 hypothesis property tests), 95% coverage
- CI: ruff + ruff format + mypy --strict + pytest on Python 3.12 and 3.13
- Wire-protocol commitment doc + per-turn schema design (Phase 11 SVO calendar foundation in place from day 1)
