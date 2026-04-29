"""OpenAI embedding backend.

Lazy-imports the `openai` SDK so it remains an optional install.
Install with `pip install jamjet-engram[embed-openai]`.
"""

from __future__ import annotations

import os
from typing import Any

from engram.errors import EmbeddingError

# Map model name -> default vector dim
_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding:
    """Embed via OpenAI's API. Requires `openai` package and `OPENAI_API_KEY`."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise EmbeddingError(
                "openai package not installed; pip install 'jamjet-engram[embed-openai]'"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EmbeddingError("OPENAI_API_KEY not set")
        self._client: Any = AsyncOpenAI(api_key=key)
        self._model = model
        self._dim = dim if dim is not None else _MODEL_DIMS.get(model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.embeddings.create(model=self._model, input=texts)
        except Exception as e:
            raise EmbeddingError(f"openai embeddings error: {e}") from e
        return [list(d.embedding) for d in resp.data]
