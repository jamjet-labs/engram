from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from engram import Engram
from engram.models import Event
from engram.read.reader import Reader, ReaderConfig
from engram.scope import Scope
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


# 3.3: solve() tests
async def _seed_events(memory, scope, *, verb, object_canonical, dates):
    for d in dates:
        ev = Event(
            id=uuid4(),
            scope=scope,
            subject_canonical="user",
            verb=verb,
            object_canonical=object_canonical,
            time_start=d,
            time_end=None,
            confidence=1.0,
            aliases=[],
        )
        await memory._store.upsert_event(ev)


@pytest.mark.asyncio
async def test_solve_count_before_anchor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        await _seed_events(
            memory, scope, verb="run", object_canonical="marathon",
            dates=[datetime(2023, m, 1, tzinfo=UTC) for m in (1, 3, 5, 7)],
        )
        anchor_dt = datetime(2023, 8, 1, tzinfo=UTC)
        await _seed_events(
            memory, scope, verb="participate", object_canonical="charity gala",
            dates=[anchor_dt],
        )
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        q = TemporalQuery(
            op="count", verb="run", object="marathon",
            anchor_event="charity gala", bound="before",
        )
        result = await s.solve(q, scope)
        assert result is not None
        assert result.answer == 4


@pytest.mark.asyncio
async def test_solve_rejects_anchor_with_no_word_overlap(monkeypatch):
    """Sanity check: store may fall back to listing all events for FTS-unfriendly
    anchor names (e.g. all-hyphen). Solver must reject those false matches."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        await _seed_events(
            memory, scope, verb="run", object_canonical="marathon",
            dates=[datetime(2023, 1, 1, tzinfo=UTC)],
        )
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        # Anchor name has nothing in common with seeded events; sanitiser would
        # drop the dashes entirely, so the store would otherwise return marathons.
        q = TemporalQuery(
            op="count", verb="run", anchor_event="--", bound="before",
        )
        assert await s.solve(q, scope) is None


@pytest.mark.asyncio
async def test_solve_returns_none_for_unknown_anchor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        q = TemporalQuery(
            op="count", verb="run", anchor_event="nonexistent-event", bound="before",
        )
        assert await s.solve(q, scope) is None


@pytest.mark.asyncio
async def test_solve_count_after_anchor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        await _seed_events(
            memory, scope, verb="run", object_canonical="marathon",
            dates=[datetime(2023, m, 1, tzinfo=UTC) for m in (1, 3, 9, 11)],
        )
        anchor_dt = datetime(2023, 5, 1, tzinfo=UTC)
        await _seed_events(
            memory, scope, verb="run", object_canonical="anchor-event",
            dates=[anchor_dt],
        )
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        q = TemporalQuery(
            op="count", verb="run", object="marathon",
            anchor_event="anchor-event", bound="after",
        )
        result = await s.solve(q, scope)
        assert result is not None
        assert result.answer == 2  # months 9 and 11


@pytest.mark.asyncio
async def test_solve_returns_none_when_op_unsupported(monkeypatch):
    """duration / ordering / before_after / elapsed all fall through for now."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        for op in ("duration", "ordering", "before_after", "elapsed"):
            q = TemporalQuery(op=op, verb="run")
            assert await s.solve(q, scope) is None


@pytest.mark.asyncio
async def test_solve_count_returns_none_without_verb_or_object(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        # Need at least one event in the store so the empty-store guard passes.
        await _seed_events(
            memory, scope, verb="x", object_canonical="y",
            dates=[datetime(2023, 1, 1, tzinfo=UTC)],
        )
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        q = TemporalQuery(op="count")
        assert await s.solve(q, scope) is None


@pytest.mark.asyncio
async def test_solve_returns_none_when_event_store_empty(monkeypatch):
    """Fail-closed: empty SVO calendar means we can't answer; don't return 0."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        q = TemporalQuery(op="count", verb="run", object="marathon")
        # No events seeded → solver must return None, not SolverResult(answer=0).
        assert await s.solve(q, scope) is None


@pytest.mark.asyncio
async def test_solve_returns_none_when_count_is_zero(monkeypatch):
    """Defer to LLM when count=0 — in LongMemEval the user has usually done
    the thing, so 0 means our SVO extraction missed the relevant events."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        # Seed an unrelated event so the empty-store guard passes.
        await _seed_events(
            memory, scope, verb="cook", object_canonical="dinner",
            dates=[datetime(2023, 1, 1, tzinfo=UTC)],
        )
        s = TemporalSolver(store=memory._store, llm=AsyncMock())
        # Count for "run marathon" — no matches exist.
        q = TemporalQuery(op="count", verb="run", object="marathon")
        assert await s.solve(q, scope) is None


# 3.4: Reader pre-pass integration
@pytest.mark.asyncio
async def test_reader_uses_solver_when_it_returns_answer():
    """When the solver answers, the LLM is never called."""
    fake_llm = AsyncMock()
    fake_solver = AsyncMock()
    fake_solver.parse = AsyncMock(return_value=TemporalQuery(op="count", verb="run"))
    fake_solver.solve = AsyncMock(return_value=SolverResult(answer=4, confidence=0.95))

    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(solver=fake_solver))
    res = await reader.read(
        question="How many marathons before Cure?",
        context="some context",
        today=datetime(2023, 8, 1, tzinfo=UTC),
        scope=Scope(org_id="default", user_id="alice"),
    )
    assert res.answer == "4"
    assert res.solved_by == "solver"
    fake_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_reader_falls_through_when_solver_returns_none():
    """When the solver can't handle the question, the LLM is invoked."""
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "fallback answer"
    fake_solver = AsyncMock()
    fake_solver.parse = AsyncMock(return_value=None)
    fake_solver.solve = AsyncMock(return_value=None)

    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(solver=fake_solver))
    res = await reader.read(
        question="What's my favourite colour?",
        context="context",
        scope=Scope(org_id="default", user_id="alice"),
    )
    assert res.answer == "fallback answer"
    assert res.solved_by == "reader"
    fake_llm.generate.assert_called()


@pytest.mark.asyncio
async def test_reader_skips_solver_without_scope():
    """Without a scope, the solver pre-pass is skipped — back-compat."""
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "answer"
    fake_solver = AsyncMock()
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(solver=fake_solver))
    res = await reader.read(question="?", context="ctx")
    assert res.answer == "answer"
    fake_solver.parse.assert_not_called()


@pytest.mark.asyncio
async def test_reader_solver_failure_does_not_break_read():
    """If the solver raises, the reader still succeeds via the LLM path."""
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "fallback"
    fake_solver = AsyncMock()
    fake_solver.parse = AsyncMock(side_effect=RuntimeError("boom"))
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(solver=fake_solver))
    res = await reader.read(
        question="?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert res.answer == "fallback"
