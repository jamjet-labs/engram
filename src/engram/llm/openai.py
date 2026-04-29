"""OpenAI chat backend (lazy-imported; install with `pip install jamjet-engram[llm-openai]`)."""

from __future__ import annotations

import os
from typing import Any

from engram.errors import ExtractionError
from engram.llm.base import LLMMessage, LLMResponse


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ExtractionError(
                "openai package not installed; pip install 'jamjet-engram[llm-openai]'"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ExtractionError("OPENAI_API_KEY not set")
        self._client: Any = AsyncOpenAI(api_key=key)
        self._model = model

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ExtractionError(f"openai chat error: {e}") from e
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            model=self._model,
        )
