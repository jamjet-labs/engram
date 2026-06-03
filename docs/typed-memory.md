# Typed Memory

Engram's typed-memory layer adds functional metadata on top of existing facts. It improves retrieval precision and context packing without changing default `record`, `recall`, `context`, or MCP behavior.

## Memory hierarchy

- `memory_system`: strict broad enum (`working`, `episodic`, `semantic`, `procedural`, `prospective`, `preference`)
- `memory_subtype`: flexible string below the system (`tool_preference`, `release_process`, `trigger`, etc.)
- `tags`: lightweight flexible labels for further filtering

## Record with explicit types

```python
await memory.record(
    "For this repo, always use pnpm instead of npm.",
    memory_system="preference",
    memory_subtype="tool_preference",
    tags=("pnpm",),
)
```

## Record with deterministic routing

```python
await memory.record(
    "Release workflow: run tests, update changelog, bump version.",
    route_memory=True,
)
```

`RuleBasedMemoryRouter` assigns memory system, subtype, lifecycle state, promotion state, and tags. It is offline-safe and deterministic.

## Typed recall

```python
results = await memory.recall(
    "How should I install dependencies?",
    memory_systems=("preference",),
    memory_subtypes=("tool_preference",),
    tags=("pnpm",),
)
```

Expired and superseded lifecycle states are excluded by default.

## Query-intent inference

Set `infer_memory_filters=True` to conservatively infer typed filters from strong query cues. Explicit filters always win.

```python
results = await memory.recall(
    "In this repo, what package manager should I use?",
    user_id="repo-a",
    infer_memory_filters=True,
)
```

## MCP compatibility

Existing MCP tool names remain unchanged. Optional typed fields are available on `memory_record`, `memory_recall`, and `memory_context`.

## Storage compatibility

Typed fields roundtrip through existing `Fact.metadata` JSON under the internal `_engram_typed_memory` namespace. This avoids a schema migration while preserving user metadata keys like `memory_system`, `memory_subtype`, and `tags`.
