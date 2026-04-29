from __future__ import annotations

import os

import httpx
import pytest

from engram.errors import ExtractionError
from engram.llm.base import LLMMessage
from engram.llm.ollama import OllamaLLM


def _ollama_running() -> bool:
    if os.environ.get("ENGRAM_SKIP_OLLAMA"):
        return False
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


# ── LLMMessage / LLMResponse ────────────────────────────────────────


def test_llm_message_role_validation() -> None:
    from pydantic import ValidationError

    LLMMessage(role="user", content="hi")  # ok
    with pytest.raises(ValidationError):
        LLMMessage(role="invalid", content="x")  # type: ignore[arg-type]


# ── OllamaLLM ────────────────────────────────────────────────────────


@pytest.mark.skipif(not _ollama_running(), reason="ollama not running locally")
async def test_ollama_generate_returns_content() -> None:
    llm = OllamaLLM(model="llama3.2:3b")
    resp = await llm.generate(
        [
            LLMMessage(role="system", content="You answer in one short sentence."),
            LLMMessage(role="user", content="What color is the sky on a clear day?"),
        ],
        temperature=0.0,
        max_tokens=50,
    )
    assert resp.content
    assert resp.model == "llama3.2:3b"
    assert resp.finish_reason == "stop"


@pytest.mark.skipif(not _ollama_running(), reason="ollama not running locally")
async def test_ollama_json_mode_returns_valid_json() -> None:
    import json as _json

    llm = OllamaLLM(model="llama3.2:3b")
    resp = await llm.generate(
        [
            LLMMessage(
                role="system",
                content='Output JSON of shape {"answer": "<short string>"} only.',
            ),
            LLMMessage(role="user", content="What is 2+2?"),
        ],
        temperature=0.0,
        json_mode=True,
        max_tokens=50,
    )
    parsed = _json.loads(resp.content)
    assert "answer" in parsed


async def test_ollama_unreachable_raises() -> None:
    llm = OllamaLLM(base_url="http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(ExtractionError):
        await llm.generate([LLMMessage(role="user", content="hi")])
