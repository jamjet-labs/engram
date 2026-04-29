"""LLM client protocol shared by Ollama / OpenAI / Anthropic / Google backends."""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


def normalize_finish_reason(raw: str | None) -> FinishReason | None:
    """Map provider-specific finish-reason strings to our canonical set.

    OpenAI: stop, length, tool_calls, content_filter, function_call (legacy)
    Anthropic: end_turn, max_tokens, stop_sequence, tool_use, refusal
    Ollama: depends; we already pass 'stop' or 'length'
    """
    if raw is None:
        return None
    mapping: dict[str, FinishReason] = {
        "stop": "stop",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "length": "length",
        "max_tokens": "length",
        "tool_calls": "tool_calls",
        "tool_use": "tool_calls",
        "function_call": "tool_calls",
        "content_filter": "content_filter",
        "refusal": "content_filter",
    }
    return mapping.get(raw, "stop")


class LLMResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: FinishReason | None = None
    model: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Async chat-completion API. Implementations live in `engram.llm.{ollama,openai,...}`."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse: ...
