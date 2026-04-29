"""Anthropic chat backend.

Lazy-imported; install with `pip install jamjet-engram[llm-anthropic]`.
"""

from __future__ import annotations

import os
from typing import Any

from engram.errors import ExtractionError
from engram.llm.base import LLMMessage, LLMResponse


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ExtractionError(
                "anthropic package not installed; pip install 'jamjet-engram[llm-anthropic]'"
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ExtractionError("ANTHROPIC_API_KEY not set")
        self._client: Any = AsyncAnthropic(api_key=key)
        self._model = model

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        # Anthropic API splits system prompt from user/assistant turns
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        chat = [m.model_dump() for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as e:
            raise ExtractionError(f"anthropic error: {e}") from e
        # Concatenate text blocks (anthropic returns content as a list)
        content_parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text is not None:
                content_parts.append(text)
        return LLMResponse(
            content="".join(content_parts),
            input_tokens=resp.usage.input_tokens if resp.usage else 0,
            output_tokens=resp.usage.output_tokens if resp.usage else 0,
            finish_reason=resp.stop_reason or "stop",
            model=self._model,
        )
