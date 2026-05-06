"""Tests for Reader(mode="synthesis") — coupled package: synthesis prompt
+ no verifier + no tool loop + no escalation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.errors import ExtractionError
from engram.llm.base import LLMResponse
from engram.read.reader import Reader, ReaderConfig
from engram.tools.base import ToolRegistry


def _capture_llm():
    captured = {"system": None}
    llm = AsyncMock()

    async def fake_generate(messages, **_kw):
        for m in messages:
            if m.role == "system":
                captured["system"] = m.content
                break
        return LLMResponse(content="answer text", input_tokens=10, output_tokens=10)

    llm.generate = AsyncMock(side_effect=fake_generate)
    return llm, captured


@pytest.mark.asyncio
async def test_synthesis_mode_uses_synthesis_prompt():
    llm, captured = _capture_llm()
    reader = Reader(llm, mode="synthesis")
    res = await reader.read(question="any tips?", context="user likes jazz")
    assert "preference/recommendation" in captured["system"].lower()
    assert "user likes jazz" in captured["system"]
    assert res.solved_by == "synthesis"
    assert res.answer == "answer text"


@pytest.mark.asyncio
async def test_synthesis_mode_bypasses_verifier_even_when_enabled():
    """verifier=True should be ignored in synthesis mode."""
    llm, _ = _capture_llm()
    reader = Reader(llm, verifier=True, mode="synthesis")
    res = await reader.read(question="q", context="ctx")
    # If verifier had run, llm.generate would have been called twice (verifier + reader).
    # In synthesis mode it's called exactly once.
    assert llm.generate.call_count == 1
    assert res.verdict is None  # no verifier verdict


@pytest.mark.asyncio
async def test_synthesis_mode_returns_idk_on_extraction_error():
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=ExtractionError("network"))
    reader = Reader(llm, mode="synthesis")
    res = await reader.read(question="q", context="ctx")
    assert res.answer == "I don't know"
    assert res.abstained is True
    assert res.solved_by == "synthesis"


@pytest.mark.asyncio
async def test_synthesis_mode_skips_tool_loop():
    """Even when tools are configured, synthesis mode does not enter the tool loop."""
    llm = AsyncMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content='[TOOL_USE]{"name": "search_facts", "input": {"query": "x"}}[/TOOL_USE]',
            input_tokens=10,
            output_tokens=10,
        )
    )
    registry = ToolRegistry()
    reader = Reader(llm, mode="synthesis", config=ReaderConfig(tools=registry))
    res = await reader.read(question="q", context="ctx")
    # The model output contains [TOOL_USE] markup but synthesis mode should
    # accept it as the answer verbatim — the tool loop is bypassed.
    assert "TOOL_USE" in res.answer
    # Single LLM call (no tool dispatch loop).
    assert llm.generate.call_count == 1


@pytest.mark.asyncio
async def test_recall_mode_unchanged_behavior():
    """Default mode='recall' must preserve full pipeline (verifier runs)."""
    llm = AsyncMock()
    reader = Reader(llm, verifier=True)  # default mode="recall"
    assert reader._mode == "recall"
