from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.read.reader import Reader, ReaderConfig
from engram.scope import Scope
from engram.tools.base import ToolRegistry, ToolResult


class _StaticTool:
    name = "static"
    description = "Returns 42 always"
    input_schema = {"type": "object", "properties": {}}  # noqa: RUF012

    async def __call__(self) -> ToolResult:
        return ToolResult(content="42")


@pytest.mark.asyncio
async def test_reader_dispatches_tool_call_then_finalises():
    """Reader sees a tool_use response on the first turn, dispatches it,
    then on the second turn the model emits a final answer using the tool result."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        # Turn 1: tool call
        MagicMock(content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'),
        # Turn 2: final answer using tool output
        MagicMock(content="The answer is 42"),
    ]
    reg = ToolRegistry()
    reg.register(_StaticTool())
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(tools=reg))
    res = await reader.read(
        question="What is the answer?",
        context="ignore",
        scope=Scope(org_id="default", user_id="alice"),
    )
    assert "42" in res.answer
    assert fake_llm.generate.await_count == 2


@pytest.mark.asyncio
async def test_reader_no_tools_passes_through():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value = MagicMock(content="direct answer")
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(tools=None))
    res = await reader.read(
        question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert res.answer == "direct answer"
    assert fake_llm.generate.await_count == 1


@pytest.mark.asyncio
async def test_reader_first_answer_wins_when_text_before_tool():
    """Provider quirk: if the model emits text AND a tool call in the same turn,
    treat the text as the final answer and ignore the tool call."""
    fake_llm = AsyncMock()
    fake_llm.generate.return_value = MagicMock(
        content='Here is my answer: 7\n[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'
    )
    reg = ToolRegistry()
    reg.register(_StaticTool())
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(tools=reg))
    res = await reader.read(
        question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert "7" in res.answer
    assert fake_llm.generate.await_count == 1


@pytest.mark.asyncio
async def test_reader_handles_unknown_tool_gracefully():
    """Tool dispatch failure shouldn't crash the read; error is fed back as result."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(
            content='[TOOL_USE]{"name": "missing", "input": {}}[/TOOL_USE]'
        ),
        MagicMock(content="Recovered answer"),
    ]
    reg = ToolRegistry()
    reg.register(_StaticTool())
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(tools=reg))
    res = await reader.read(
        question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert res.answer == "Recovered answer"


@pytest.mark.asyncio
async def test_reader_caps_tool_calls_at_max():
    """If the model loops tool calls, cap at _MAX_TOOL_CALLS=5 and force a final answer."""
    fake_llm = AsyncMock()
    # Always emit a tool call, never a final answer
    fake_llm.generate.return_value = MagicMock(
        content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'
    )
    reg = ToolRegistry()
    reg.register(_StaticTool())
    reader = Reader(fake_llm, verifier=False, config=ReaderConfig(tools=reg))
    # Override the final-call to return a real answer
    fake_llm.generate.side_effect = [
        MagicMock(content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'),
        MagicMock(content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'),
        MagicMock(content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'),
        MagicMock(content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'),
        MagicMock(content='[TOOL_USE]{"name": "static", "input": {}}[/TOOL_USE]'),
        MagicMock(content="Forced final answer"),  # the post-cap call
    ]
    res = await reader.read(
        question="Q?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    assert res.answer == "Forced final answer"
    assert fake_llm.generate.await_count == 6  # 5 tool calls + 1 forced final
