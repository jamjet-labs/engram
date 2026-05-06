# Engram quickstart

This guide walks you from `pip install` to a running per-category preference pipeline in ~150 lines of Python.

## Install

```bash
uv add jamjet-engram                # or: pip install jamjet-engram
uv add 'jamjet-engram[rerank]'      # optional cross-encoder rerank
```

You'll need an LLM provider — `OpenAI`, `Anthropic`, or local `Ollama`. Set the appropriate API key in your environment.

## 30-second example: record + recall

```python
import asyncio
from engram import Engram, Scope
from engram.embedding.synthetic import SyntheticEmbedding


async def main():
    embedder = SyntheticEmbedding(dim=128)  # toy embedder; use Ollama/OpenAI for real use
    async with await Engram.open(":memory:", embedder=embedder) as memory:
        await memory.record(text="alice has a cat named Luna", user_id="alice")
        await memory.record(text="alice loves Italian food", user_id="alice")

        ctx = await memory.context(query="what does alice like to eat?", user_id="alice")
        print(ctx)

asyncio.run(main())
```

You'll see a context string with the relevant fact retrieved. That's the core pattern: `record()` to ingest, `context()` to retrieve.

## Add a classifier + budget

`Engram.context` integrates with a question classifier to pick a per-category token budget (calibrated on LongMemEval).

```python
import asyncio
from engram import Engram, RuleBasedClassifier
from engram.embedding.synthetic import SyntheticEmbedding


async def main():
    embedder = SyntheticEmbedding(dim=128)
    classifier = RuleBasedClassifier()

    async with await Engram.open(":memory:", embedder=embedder) as memory:
        await memory.record(text="alice met Bob in Paris in 2019", user_id="alice")

        ctx = await memory.context(
            query="when did alice meet Bob?",
            user_id="alice",
            classifier=classifier,    # auto-picks budget for temporal-reasoning
        )
        print(ctx)

asyncio.run(main())
```

## Per-category routing (preferences)

For preference/recommendation questions ("any tips on a new guitar?", "recommend a movie"), Engram has a synthesis-mode reader that produces a recommendation grounded in the user's stated preferences. Opt in by gating on `is_preference_question`:

```python
import asyncio
from engram import (
    Engram,
    Reader,
    RuleBasedClassifier,
    Scope,
    is_preference_question,
)
from engram.embedding.synthetic import SyntheticEmbedding
from engram.llm.tier import ModelTier


async def main():
    embedder = SyntheticEmbedding(dim=128)
    tier = ModelTier.default()  # gpt-4o-mini reader + utility
    classifier = RuleBasedClassifier()

    async with await Engram.open(":memory:", embedder=embedder, tier=tier) as memory:
        # Ingest with role tags — every turn tagged at ingest time
        chat = [
            {"role": "user", "content": "I love stand-up comedy on Netflix"},
            {"role": "assistant", "content": "What kind of comedy do you enjoy?"},
            {"role": "user", "content": "Storytelling-heavy specials, like Mulaney."},
        ]
        for turn in chat:
            await memory.record(
                text=turn["content"],
                role=turn["role"],
                user_id="alice",
                session_id="s1",
            )

        # Per question, classify and route
        question = "Recommend a show for tonight"
        qt = await classifier.classify(question)
        is_pref = is_preference_question(question, qt)

        if is_pref:
            ctx = await memory.context(
                query=question,
                user_id="alice",
                role_filter=("user",),    # only USER turns surface
                token_budget=2000,
            )
            reader = Reader(tier.reader, mode="synthesis")
            res = await reader.read(question=question, context=ctx)
        else:
            ctx = await memory.context(query=question, user_id="alice", classifier=classifier)
            reader = Reader(tier.reader)
            res = await reader.read(
                question=question, context=ctx,
                scope=Scope(org_id="default", user_id="alice"),
            )

        print(res.answer)

asyncio.run(main())
```

The synthesis path lifts `single-session-preference` accuracy on LongMemEval-S from 29% to 65% in our own benchmarks. See [README.md](../README.md#benchmarks--longmemeval-s) for full numbers.

## What next

- **Reader modes:** `Reader(mode="recall")` (default) for fact-recall questions; `Reader(mode="synthesis")` for preference/recommendation.
- **Filters:** `Engram.context(role_filter=("user",))` — restrict retrieval to certain roles.
- **Tools:** the recall-mode reader can call tools mid-generation. See `examples/04_two_stage_with_classifier.py`.
- **Production deployment:** see "Running in production" in [README.md](../README.md).

For deeper patterns:

- `examples/01_quickstart.py` — minimal record + recall
- `examples/02_extraction_with_ollama.py` — LLM-driven extraction from chat
- `examples/03_http_server.py` — FastAPI deployment
- `examples/04_two_stage_with_classifier.py` — two-stage retrieval
- `examples/05_preference_synthesis.py` — per-category routing (this guide's full example)
