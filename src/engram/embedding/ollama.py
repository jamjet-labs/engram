"""Ollama embedding backend."""

from __future__ import annotations

import asyncio

import httpx

from engram.errors import EmbeddingError


class OllamaEmbedding:
    """Embed via a local Ollama server (default `http://localhost:11434`).

    Pull the model first: `ollama pull nomic-embed-text`.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        dim: int = 768,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            results = await asyncio.gather(
                *(self._embed_one(client, t) for t in texts), return_exceptions=False
            )
        return list(results)

    async def _embed_one(self, client: httpx.AsyncClient, text: str) -> list[float]:
        try:
            r = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            r.raise_for_status()
            payload = r.json()
            emb = payload.get("embedding")
            if not isinstance(emb, list):
                raise EmbeddingError(f"ollama returned no embedding: {payload}")
            return [float(x) for x in emb]
        except httpx.HTTPError as e:
            raise EmbeddingError(f"ollama HTTP error: {e}") from e
