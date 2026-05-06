from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from engram import Engram
from engram.models import Event
from engram.scope import Scope
from engram.solve.temporal import (
    SolverResult,
    TemporalQuery,
    TemporalSolver,
)
from engram.tools.base import Tool, ToolRegistry, ToolResult
from engram.tools.count_between import CountBetweenTool
from engram.tools.dates import AddDaysTool, DaysBetweenTool
from engram.tools.search_events import SearchEventsTool
from engram.tools.search_facts import SearchFactsTool
from engram.tools.solve_temporal import SolveTemporalTool

# ── Tool protocol + ToolRegistry ────────────────────────────────────────────


class _Echo:
    name = "echo"
    description = "Echo input"
    input_schema = {  # noqa: RUF012 — test stub
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def __call__(self, text: str) -> ToolResult:
        return ToolResult(content=f"echo: {text}")


@pytest.mark.asyncio
async def test_tool_protocol_satisfied():
    e = _Echo()
    assert isinstance(e, Tool)
    res = await e(text="hi")
    assert res.content == "echo: hi"


def test_tool_registry_register_and_dispatch_schema():
    reg = ToolRegistry()
    reg.register(_Echo())
    assert reg.names() == ["echo"]
    schemas_a = reg.for_anthropic()
    assert any(s["name"] == "echo" for s in schemas_a)
    schemas_o = reg.for_openai()
    assert any(s["function"]["name"] == "echo" for s in schemas_o)


@pytest.mark.asyncio
async def test_tool_registry_dispatch_invokes_tool():
    reg = ToolRegistry()
    reg.register(_Echo())
    res = await reg.dispatch("echo", {"text": "world"})
    assert res.content == "echo: world"


@pytest.mark.asyncio
async def test_tool_registry_dispatch_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        await reg.dispatch("missing", {})


def test_tool_registry_signature_stable_across_orderings():
    a = ToolRegistry()
    b = ToolRegistry()

    class _A:
        name = "a"
        description = ""
        input_schema = {"type": "object", "properties": {}}  # noqa: RUF012

        async def __call__(self) -> ToolResult:
            return ToolResult(content="a")

    class _B:
        name = "b"
        description = ""
        input_schema = {"type": "object", "properties": {}}  # noqa: RUF012

        async def __call__(self) -> ToolResult:
            return ToolResult(content="b")

    a.register(_A())
    a.register(_B())
    b.register(_B())
    b.register(_A())
    assert a.signature() == b.signature()


# ── Date tools ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_days_tool():
    res = await AddDaysTool()(date="2024-03-01", n=10)
    assert res.content == "2024-03-11"


@pytest.mark.asyncio
async def test_add_days_negative():
    res = await AddDaysTool()(date="2024-03-15", n=-5)
    assert res.content == "2024-03-10"


@pytest.mark.asyncio
async def test_days_between_tool():
    res = await DaysBetweenTool()(start="2024-03-01", end="2024-03-15")
    assert res.content == "14"


@pytest.mark.asyncio
async def test_days_between_negative_when_reversed():
    res = await DaysBetweenTool()(start="2024-03-15", end="2024-03-01")
    assert res.content == "-14"


# ── search_facts ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_facts_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    async with await Engram.open(":memory:") as memory:
        await memory.record(text="I love espresso", user_id="alice")
        scope = Scope(org_id="default", user_id="alice")
        res = await SearchFactsTool(engram=memory, scope=scope)(query="coffee preference", top_k=3)
        assert "espresso" in res.content.lower()


@pytest.mark.asyncio
async def test_search_facts_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    async with await Engram.open(":memory:") as memory:
        scope = Scope(org_id="default", user_id="alice")
        res = await SearchFactsTool(engram=memory, scope=scope)(query="anything")
        assert "(no facts found)" in res.content


# ── search_events ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_events_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        await memory._store.upsert_event(
            Event(
                id=uuid4(),
                scope=scope,
                subject_canonical="user",
                verb="run",
                object_canonical="marathon",
                time_start=datetime(2024, 5, 1, tzinfo=UTC),
                time_end=None,
                confidence=1.0,
                aliases=["ran a marathon"],
            )
        )
        res = await SearchEventsTool(engram=memory, scope=scope)(query="marathon")
        assert "marathon" in res.content.lower()


# ── solve_temporal ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_solve_temporal_tool_dispatches_to_solver():
    fake_solver = AsyncMock(spec=TemporalSolver)
    fake_solver.parse = AsyncMock(return_value=TemporalQuery(op="count"))
    fake_solver.solve = AsyncMock(return_value=SolverResult(answer=4, confidence=0.95))
    res = await SolveTemporalTool(solver=fake_solver, scope=Scope(org_id="d", user_id="a"))(
        query="How many marathons before Cure?"
    )
    assert "4" in res.content


@pytest.mark.asyncio
async def test_solve_temporal_tool_unparseable_returns_message():
    fake_solver = AsyncMock(spec=TemporalSolver)
    fake_solver.parse = AsyncMock(return_value=None)
    res = await SolveTemporalTool(solver=fake_solver, scope=Scope(org_id="d", user_id="a"))(
        query="What's my favourite colour?"
    )
    assert "could not solve" in res.content.lower()


# ── count_between ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count_between_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scope = Scope(org_id="default", user_id="alice")
    async with await Engram.open(":memory:") as memory:
        for m in (3, 5, 7):
            await memory._store.upsert_event(
                Event(
                    id=uuid4(),
                    scope=scope,
                    subject_canonical="user",
                    verb="run",
                    object_canonical="marathon",
                    time_start=datetime(2024, m, 1, tzinfo=UTC),
                    time_end=None,
                    confidence=1.0,
                    aliases=[],
                )
            )
        res = await CountBetweenTool(engram=memory, scope=scope)(
            start="2024-01-01", end="2024-12-31", verb="run"
        )
        assert res.content == "3"
