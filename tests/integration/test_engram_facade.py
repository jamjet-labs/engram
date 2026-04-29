"""End-to-end tests against the public `Engram` facade."""

from __future__ import annotations

from datetime import UTC, datetime

from engram import Engram


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


async def test_record_then_recall() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record(text="alice prefers espresso", user_id="alice")
        results = await memory.recall(query="coffee preference", user_id="alice")
        assert len(results) >= 1
        assert "espresso" in results[0].fact.text.lower()


async def test_record_respects_scope() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record(text="alice fact", user_id="alice")
        await memory.record(text="bob fact", user_id="bob")
        alice_results = await memory.recall(query="fact", user_id="alice")
        assert len(alice_results) == 1
        assert "alice" in alice_results[0].fact.text


async def test_record_message_then_extract_without_llm_raises() -> None:
    async with await Engram.open(":memory:") as memory:
        msg = await memory.record_message(content="hello", session_id="s1")
        try:
            await memory.extract([msg])
        except RuntimeError as e:
            assert "llm" in str(e).lower()
        else:
            raise AssertionError("expected RuntimeError when no LLM configured")


async def test_context_returns_budgeted_string() -> None:
    async with await Engram.open(":memory:") as memory:
        for i in range(20):
            await memory.record(text=f"fact number {i} about coffee", user_id="alice")
        ctx = await memory.context(query="coffee", user_id="alice", token_budget=100)
        assert isinstance(ctx, str)
        # 100 tokens * 4 chars = 400 char budget, generous so we should get >=1 line
        assert ctx
        assert len(ctx) <= 100 * 4 + 200  # allow some slack for line boundaries


async def test_record_returns_fact_with_event_date() -> None:
    async with await Engram.open(":memory:") as memory:
        ed = _now()
        fact = await memory.record(text="alice went to paris", user_id="alice", event_date=ed)
        assert fact.event_date == ed
