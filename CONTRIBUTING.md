# Contributing to Engram

Thanks for your interest in contributing. Engram is a Python rewrite of the Rust v0.5.x memory layer; we're shipping toward v0.1.0 GA.

## Development setup

```bash
git clone https://github.com/jamjet-labs/engram.git
cd engram
uv sync --all-extras
```

`uv` ([install](https://docs.astral.sh/uv/)) handles the venv, dependencies, and Python version.

## Running tests

```bash
uv run pytest                   # all unit + integration tests
uv run pytest tests/unit/ -q    # unit only, quiet
uv run pytest -k preference -v  # filter by name
```

The test suite is ~280 tests; expect ~50s wall-clock locally.

## Running lint + type checks (everything CI runs)

```bash
uv run ruff check src/engram tests/
uv run ruff format --check src/engram tests/
uv run mypy src/engram
```

All three must pass for CI to go green. Use `uv run ruff format src/engram tests/` to auto-fix format issues.

## Branch naming

- `feat/<short-desc>` — new features
- `fix/<short-desc>` — bug fixes
- `chore/<short-desc>` — non-functional changes (CI, docs, dependencies)
- `docs/<short-desc>` — documentation-only changes

## Commit messages

Use conventional commits: `feat(scope): description`, `fix(scope): description`, etc. Common scopes: `engram`, `read`, `retrieve`, `tools`, `smoke_runner`, `ci`, `docs`.

## PR conventions

- Squash-merge; the squashed commit becomes the entry in `CHANGELOG.md`
- PR description includes a brief summary, the test plan, and any benchmark deltas

## Running the LongMemEval-S benchmark

See `README.md` "Reproduce" section. Requires `OPENAI_API_KEY` and the LongMemEval oracle JSON.

## Reporting bugs

Open an issue at https://github.com/jamjet-labs/engram/issues.
