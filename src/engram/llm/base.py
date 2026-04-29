"""LLM client protocol shared by Ollama / OpenAI / Anthropic / Google backends."""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
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
