"""Ollama chat backend (local; default `http://localhost:11434`)."""

from __future__ import annotations

import httpx

from engram.errors import ExtractionError
from engram.llm.base import LLMMessage, LLMResponse


class OllamaLLM:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        if max_tokens is not None:
            options = payload["options"]
            assert isinstance(options, dict)
            options["num_predict"] = max_tokens
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(f"{self._base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            raise ExtractionError(f"ollama HTTP error: {e}") from e

        msg = data.get("message", {})
        content = str(msg.get("content", ""))
        # ollama doesn't always return token counts; fall back to 0
        return LLMResponse(
            content=content,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            finish_reason="stop" if data.get("done") else "length",
            model=self._model,
        )
