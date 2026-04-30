from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from engram.solve.temporal import SolverResult, TemporalQuery, TemporalSolver


def test_temporal_query_minimal():
    q = TemporalQuery(op="count")
    assert q.op == "count"
    assert q.subject is None


def test_temporal_query_with_window():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    q = TemporalQuery(op="count", verb="run", window=(start, end))
    assert q.window == (start, end)


def test_solver_result_round_trip():
    r = SolverResult(answer=4, confidence=0.95, evidence_event_ids=[uuid4()])
    assert r.answer == 4
    assert r.confidence == 0.95
    assert len(r.evidence_event_ids) == 1


def test_solver_result_string_answer():
    r = SolverResult(answer="2024-05-01", confidence=0.8)
    assert r.answer == "2024-05-01"


# 3.2: parse() tests
@pytest.mark.asyncio
async def test_parse_returns_none_for_non_temporal():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = '{"op": null}'
    s = TemporalSolver(store=AsyncMock(), llm=fake_llm)
    assert await s.parse("What is my favourite colour?") is None


@pytest.mark.asyncio
async def test_parse_count_op():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = (
        '{"op": "count", "verb": "run", "object": "marathon", '
        '"bound": "before", "anchor_event": "Run for the Cure"}'
    )
    s = TemporalSolver(store=AsyncMock(), llm=fake_llm)
    q = await s.parse("How many marathons did I run before Run for the Cure?")
    assert q is not None
    assert q.op == "count"
    assert q.verb == "run"
    assert q.bound == "before"
    assert q.anchor_event == "Run for the Cure"


@pytest.mark.asyncio
async def test_parse_handles_malformed_json_gracefully():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "not json"
    s = TemporalSolver(store=AsyncMock(), llm=fake_llm)
    assert await s.parse("How many?") is None


@pytest.mark.asyncio
async def test_parse_rejects_unknown_op():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = '{"op": "frobnicate"}'
    s = TemporalSolver(store=AsyncMock(), llm=fake_llm)
    assert await s.parse("?") is None


@pytest.mark.asyncio
async def test_parse_passes_today_into_prompt():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = '{"op": null}'
    s = TemporalSolver(store=AsyncMock(), llm=fake_llm)
    await s.parse("?", today=datetime(2024, 5, 1, tzinfo=UTC))
    call_args = fake_llm.generate.await_args
    user_msg = call_args[0][0][0].content
    assert "2024-05-01" in user_msg
