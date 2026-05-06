## Summary

What this PR does, in 1-3 sentences.

## Why

The user-visible problem this solves, or the engineering motivation. Link to an issue if there is one.

## What changed

- Bullet point per logical change
- Helpful for the reviewer to scan

## Test plan

- [ ] All unit tests pass (`uv run pytest tests/unit/`)
- [ ] Lint + format clean (`uv run ruff check && uv run ruff format --check`)
- [ ] mypy clean (`uv run mypy src/engram`)
- [ ] (If benchmark-affecting) n=100 smoke result vs published baseline (within ±2pp on non-target categories, real lift on the target)

## Notes for the reviewer

Anything tricky, anything you're unsure about, anything you'd like a second opinion on.
