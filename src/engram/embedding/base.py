"""EmbeddingProvider protocol."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Convert text to dense vector embeddings.

    Implementations live in `engram.embedding.{ollama,openai,synthetic,...}`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality. All vectors returned by `embed` are of this length."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, each of length `self.dim`."""
        ...
