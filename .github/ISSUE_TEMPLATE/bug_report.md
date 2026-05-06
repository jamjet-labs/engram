---
name: Bug report
about: Something doesn't work the way the docs say it should
title: "[bug] "
labels: bug
---

## What happened

A clear description of the unexpected behaviour.

## What you expected

What you thought would happen, and where in the docs / API you got that expectation.

## Reproducer

Smallest possible code that triggers the issue. Ideally something we can paste and run.

```python
import asyncio
from engram import Engram

async def main():
    async with await Engram.open(":memory:") as memory:
        # ...
        pass

asyncio.run(main())
```

## Environment

- engram version: `python -c "import engram; print(engram.__version__)"` (e.g. `0.1.0`)
- Python version: `python --version` (e.g. `3.13.1`)
- OS: macOS / Linux / Windows + version
- LLM/embedding provider: OpenAI / Anthropic / Ollama / synthetic + model name
- Storage path: `:memory:` or a file path

## Logs / traceback

```
Paste any error output here.
```

## Anything else

Workarounds you've tried, related issues, etc.
