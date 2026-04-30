"""Phase 12: reading layer hardening tests."""

from __future__ import annotations

import json

from engram.llm.base import LLMMessage, LLMResponse
from engram.read.decomposer import QueryDecomposer
from engram.read.reader import Reader, format_context_with_confidence


class _FakeLLM:
    """Mock LLM that returns scripted responses based on prompt content."""

    def __init__(self, responses: dict[str, str], default: str = "") -> None:
        self._responses = responses
        self._default = default
        self.calls: list[list[LLMMessage]] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(messages)
        # Match by substring of the system prompt
        sys_content = next((m.content for m in messages if m.role == "system"), "")
        for key, response in self._responses.items():
            if key in sys_content:
                return LLMResponse(content=response, finish_reason="stop")
        return LLMResponse(content=self._default, finish_reason="stop")


# ── Reader: answer + verifier ──────────────────────────────────────


async def test_reader_returns_answer_when_verifier_yes() -> None:
    llm = _FakeLLM(
        {
            "verify": "<verdict>YES</verdict><missing>none</missing>",
            "answer a question": "espresso",
        }
    )
    reader = Reader(llm, verifier=True)
    result = await reader.read("what does alice prefer?", "- alice prefers espresso")
    assert result.answer == "espresso"
    assert result.verdict == "YES"
    assert result.abstained is False


async def test_reader_abstains_when_verifier_no() -> None:
    llm = _FakeLLM({"verify": "<verdict>NO</verdict><missing>no preference fact found</missing>"})
    reader = Reader(llm, verifier=True)
    result = await reader.read("what does alice prefer?", "- alice has a cat")
    assert result.answer == "I don't know"
    assert result.verdict == "NO"
    assert result.abstained is True
    assert result.missing == "no preference fact found"


async def test_reader_skips_verifier_when_disabled() -> None:
    llm = _FakeLLM({"answer a question": "espresso"})
    reader = Reader(llm, verifier=False)
    result = await reader.read("what?", "- some context")
    assert result.answer == "espresso"
    assert result.verdict is None
    # Only one call (the reader, not the verifier)
    assert len(llm.calls) == 1


async def test_reader_detects_idk_in_answer() -> None:
    """Reader should mark abstained when the model itself outputs 'I don't know'."""
    llm = _FakeLLM(
        {
            "verify": "<verdict>PARTIAL</verdict><missing>weak</missing>",
            "answer a question": "I don't know",
        }
    )
    reader = Reader(llm, verifier=True)
    result = await reader.read("?", "- something")
    assert result.abstained is True


async def test_reader_handles_missing_verdict_tag_as_partial() -> None:
    llm = _FakeLLM(
        {
            "verify": "this has no XML tags at all",
            "answer a question": "some answer",
        }
    )
    reader = Reader(llm, verifier=True)
    result = await reader.read("?", "- ctx")
    # Verifier defaults to PARTIAL, so we still call the reader
    assert result.verdict == "PARTIAL"
    assert result.answer == "some answer"


async def test_reader_tolerates_chatty_verifier_output() -> None:
    """Some models add prose around the tags. Regex extraction should still work."""
    llm = _FakeLLM(
        {
            "verify": "Sure, here's my analysis: <verdict>YES</verdict><missing>none</missing>",
            "answer a question": "the answer",
        }
    )
    reader = Reader(llm, verifier=True)
    result = await reader.read("?", "- ctx")
    assert result.verdict == "YES"
    assert result.answer == "the answer"


# ── format_context_with_confidence ─────────────────────────────────


def test_format_context_with_confidence_basic() -> None:
    out = format_context_with_confidence(
        [("alice prefers espresso", 0.92), ("alice has a cat", 0.75)]
    )
    assert "alice prefers espresso" in out
    assert "0.92" in out
    assert "0.75" in out


def test_format_context_with_confidence_includes_dates() -> None:
    out = format_context_with_confidence([("user moved", 1.0)], event_dates=["2024-03-12"])
    assert "[2024-03-12]" in out


# ── QueryDecomposer ────────────────────────────────────────────────


async def test_decomposer_atomic_question_returns_self() -> None:
    llm = _FakeLLM({"complex multi-part question": json.dumps({"sub_questions": ["what is X?"]})})
    d = QueryDecomposer(llm)
    out = await d.decompose("what is X?")
    assert out == ["what is X?"]


async def test_decomposer_compound_question_splits() -> None:
    llm = _FakeLLM(
        {
            "complex multi-part question": json.dumps(
                {"sub_questions": ["what is A?", "what is B?", "did A happen before B?"]}
            )
        }
    )
    d = QueryDecomposer(llm)
    out = await d.decompose("what is A and B and did A come before B?")
    assert len(out) == 3
    assert "did A happen before B?" in out


async def test_decomposer_caps_at_max_subqueries() -> None:
    llm = _FakeLLM(
        {"complex multi-part question": json.dumps({"sub_questions": [f"q{i}" for i in range(10)]})}
    )
    d = QueryDecomposer(llm, max_subqueries=3)
    out = await d.decompose("compound query")
    assert len(out) == 3


async def test_decomposer_falls_back_on_invalid_json() -> None:
    llm = _FakeLLM({"complex multi-part question": "not json"})
    d = QueryDecomposer(llm)
    out = await d.decompose("original question")
    assert out == ["original question"]


async def test_decomposer_falls_back_on_empty_subqueries() -> None:
    llm = _FakeLLM({"complex multi-part question": json.dumps({"sub_questions": []})})
    d = QueryDecomposer(llm)
    out = await d.decompose("original")
    assert out == ["original"]
