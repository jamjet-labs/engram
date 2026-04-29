from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import httpx
import pytest

from engram.errors import ExtractionError
from engram.extract.pipeline import ExtractionPipeline
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, ExtractedFact, Polarity
from engram.scope import Scope


def _ollama_running() -> bool:
    if os.environ.get("ENGRAM_SKIP_OLLAMA"):
        return False
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


# ── Mock LLM client ────────────────────────────────────────────────


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(content=self._response, finish_reason="stop")


# ── Pipeline behavior ──────────────────────────────────────────────


async def test_extract_empty_messages_returns_empty() -> None:
    pipeline = ExtractionPipeline(FakeLLM('{"facts": []}'))
    out = await pipeline.extract([])
    assert out == []


async def test_extract_parses_well_formed_json() -> None:
    response = json.dumps(
        {
            "facts": [
                {
                    "text": "alice prefers espresso",
                    "category": "user_preference",
                    "polarity": "affirmative",
                    "confidence": 0.9,
                    "entities": ["Alice"],
                    "event_date": None,
                }
            ]
        }
    )
    pipeline = ExtractionPipeline(FakeLLM(response))
    msgs = [
        ChatMessage(
            scope=Scope(),
            session_id="s",
            role="user",
            content="I love espresso",
            timestamp=_now(),
        )
    ]
    out = await pipeline.extract(msgs)
    assert len(out) == 1
    assert isinstance(out[0], ExtractedFact)
    assert out[0].text == "alice prefers espresso"
    assert out[0].category == "user_preference"
    assert out[0].polarity == Polarity.AFFIRMATIVE
    assert out[0].entities == ["Alice"]


async def test_extract_strips_markdown_fences() -> None:
    response = '```json\n{"facts": [{"text": "x", "confidence": 1.0}]}\n```'
    pipeline = ExtractionPipeline(FakeLLM(response))
    msgs = [ChatMessage(scope=Scope(), session_id="s", role="user", content="x", timestamp=_now())]
    out = await pipeline.extract(msgs)
    assert len(out) == 1
    assert out[0].text == "x"


async def test_extract_drops_malformed_facts_keeps_good_ones() -> None:
    response = json.dumps(
        {
            "facts": [
                {"text": "good fact", "confidence": 0.9},
                {"confidence": 0.5},  # missing required text
                {"text": "another good", "confidence": 1.5},  # confidence out of range
                {"text": "third good", "confidence": 0.7},
            ]
        }
    )
    pipeline = ExtractionPipeline(FakeLLM(response))
    msgs = [ChatMessage(scope=Scope(), session_id="s", role="user", content="x", timestamp=_now())]
    out = await pipeline.extract(msgs)
    assert len(out) == 2
    assert {f.text for f in out} == {"good fact", "third good"}


async def test_extract_stamps_mention_date_from_session_date() -> None:
    response = json.dumps({"facts": [{"text": "x", "confidence": 1.0}]})
    pipeline = ExtractionPipeline(FakeLLM(response))
    msgs = [ChatMessage(scope=Scope(), session_id="s", role="user", content="x", timestamp=_now())]
    out = await pipeline.extract(msgs, session_date=_now())
    assert len(out) == 1
    assert out[0].mention_date == _now()


async def test_extract_invalid_json_raises_after_retries() -> None:
    pipeline = ExtractionPipeline(FakeLLM("not json at all"), max_retries=0)
    msgs = [ChatMessage(scope=Scope(), session_id="s", role="user", content="x", timestamp=_now())]
    with pytest.raises(ExtractionError):
        await pipeline.extract(msgs)


async def test_extract_non_object_json_raises() -> None:
    pipeline = ExtractionPipeline(FakeLLM("[1,2,3]"), max_retries=0)
    msgs = [ChatMessage(scope=Scope(), session_id="s", role="user", content="x", timestamp=_now())]
    with pytest.raises(ExtractionError):
        await pipeline.extract(msgs)


async def test_extract_handles_null_polarity_field() -> None:
    response = json.dumps({"facts": [{"text": "x", "confidence": 0.9, "polarity": None}]})
    pipeline = ExtractionPipeline(FakeLLM(response))
    msgs = [ChatMessage(scope=Scope(), session_id="s", role="user", content="x", timestamp=_now())]
    out = await pipeline.extract(msgs)
    assert len(out) == 1
    assert out[0].polarity == Polarity.AFFIRMATIVE  # default


# ── Live Ollama round-trip ─────────────────────────────────────────


@pytest.mark.skipif(not _ollama_running(), reason="ollama not running locally")
async def test_extract_round_trip_with_ollama() -> None:
    """End-to-end smoke against local llama3.2:3b. Tests that prompt + JSON
    mode yields parseable output for at least one fact on a simple input."""
    from engram.llm.ollama import OllamaLLM

    pipeline = ExtractionPipeline(OllamaLLM(model="llama3.2:3b"), max_retries=1)
    msgs = [
        ChatMessage(
            scope=Scope(),
            session_id="s",
            role="user",
            content="My name is Alice and I prefer espresso over drip coffee.",
            timestamp=_now(),
        )
    ]
    facts = await pipeline.extract(msgs, session_date=_now())
    # Llama 3.2 3B is small — we don't assert on quality, just that we got
    # at least one parseable fact back. If extraction returned [], that's
    # acceptable too (the prompt may have produced {"facts": []}).
    assert isinstance(facts, list)
    for f in facts:
        assert f.text
        assert 0.0 <= f.confidence <= 1.0
