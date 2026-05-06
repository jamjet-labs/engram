from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.agent.react import AgentResult, AgentStep, ReActAgent
from engram.llm.tier import ModelTier
from engram.scope import Scope
from engram.tools.base import ToolRegistry, ToolResult


def test_agent_step_round_trip():
    step = AgentStep(
        hop=1,
        tool_name="search_facts",
        tool_input={"query": "x"},
        tool_result="result text",
        elapsed_s=0.5,
    )
    assert step.hop == 1
    assert step.tool_name == "search_facts"
    assert step.tool_input == {"query": "x"}


def test_agent_result_round_trip():
    res = AgentResult(answer="42", abstained=False, n_hops=2)
    assert res.answer == "42"
    assert res.n_hops == 2
    assert res.trace == []


def test_agent_step_defaults():
    s = AgentStep(hop=3)
    assert s.tool_name is None
    assert s.tool_input == {}
    assert s.text_emitted is None


# 7.2: ReActAgent loop tests


class _DummyTool:
    name = "dummy"
    description = "dummy tool"
    input_schema = {  # noqa: RUF012
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }

    async def __call__(self, x: str = "") -> ToolResult:
        return ToolResult(content=f"dummy:{x}")


class _FinalAnswerTool:
    name = "final_answer"
    description = "Emit the final answer."
    input_schema = {  # noqa: RUF012
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    async def __call__(self, answer: str) -> ToolResult:
        return ToolResult(content=answer, raw=answer)


def _build_tier_with_mock(mock_llm):
    tier = ModelTier.__new__(ModelTier)
    tier.reader = mock_llm
    tier.utility = mock_llm
    return tier


def _build_registry():
    reg = ToolRegistry()
    reg.register(_DummyTool())
    reg.register(_FinalAnswerTool())
    return reg


@pytest.mark.asyncio
async def test_react_terminates_on_final_answer():
    fake = AsyncMock()
    fake.generate.return_value = MagicMock(
        content='[TOOL_USE]{"name": "final_answer", "input": {"answer": "42"}}[/TOOL_USE]'
    )
    agent = ReActAgent(tier=_build_tier_with_mock(fake), tools=_build_registry(), max_hops=4)
    res = await agent.answer("Q?", scope=Scope(org_id="d", user_id="a"))
    assert res.answer == "42"
    assert res.n_hops == 1
    assert not res.abstained


@pytest.mark.asyncio
async def test_react_terminates_at_max_hops():
    fake = AsyncMock()
    # Always emit a non-final tool call → never terminates via final_answer
    fake.generate.return_value = MagicMock(
        content='[TOOL_USE]{"name": "dummy", "input": {"x": "loop"}}[/TOOL_USE]'
    )
    agent = ReActAgent(tier=_build_tier_with_mock(fake), tools=_build_registry(), max_hops=2)
    res = await agent.answer("Q?", scope=Scope(org_id="d", user_id="a"))
    assert res.abstained is True


@pytest.mark.asyncio
async def test_react_loop_detection_same_args_twice():
    fake = AsyncMock()
    fake.generate.return_value = MagicMock(
        content='[TOOL_USE]{"name": "dummy", "input": {"x": "same"}}[/TOOL_USE]'
    )
    agent = ReActAgent(tier=_build_tier_with_mock(fake), tools=_build_registry(), max_hops=10)
    res = await agent.answer("Q?", scope=Scope(org_id="d", user_id="a"))
    # First call goes through; second call with identical args triggers loop detection
    assert res.n_hops <= 2
    assert res.abstained is True


@pytest.mark.asyncio
async def test_react_accepts_free_text_as_answer():
    """If the LLM emits free text without TOOL_USE, accept it as the final answer."""
    fake = AsyncMock()
    fake.generate.return_value = MagicMock(content="The answer is direct text.")
    agent = ReActAgent(tier=_build_tier_with_mock(fake), tools=_build_registry(), max_hops=4)
    res = await agent.answer("Q?", scope=Scope(org_id="d", user_id="a"))
    assert "direct text" in res.answer
    assert res.n_hops == 1


@pytest.mark.asyncio
async def test_react_handles_unknown_tool_gracefully():
    fake = AsyncMock()
    final_call = '[TOOL_USE]{"name": "final_answer", "input": {"answer": "recovered"}}[/TOOL_USE]'
    fake.generate.side_effect = [
        MagicMock(content='[TOOL_USE]{"name": "nonexistent", "input": {}}[/TOOL_USE]'),
        MagicMock(content=final_call),
    ]
    agent = ReActAgent(tier=_build_tier_with_mock(fake), tools=_build_registry(), max_hops=4)
    res = await agent.answer("Q?", scope=Scope(org_id="d", user_id="a"))
    assert "recovered" in res.answer


@pytest.mark.asyncio
async def test_react_handles_malformed_tool_call():
    fake = AsyncMock()
    fake.generate.return_value = MagicMock(content='[TOOL_USE]{"bad json}[/TOOL_USE]')
    agent = ReActAgent(tier=_build_tier_with_mock(fake), tools=_build_registry(), max_hops=4)
    res = await agent.answer("Q?", scope=Scope(org_id="d", user_id="a"))
    assert res.abstained is True
