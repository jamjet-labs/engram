"""OpenAIEmbedding passes a smaller size to the API, or refuses one it cannot give."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from engram.embedding.openai import OpenAIEmbedding
from engram.errors import EmbeddingError


class _Embeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.5])])


def _embedder(model: str, dim: int | None) -> tuple[OpenAIEmbedding, _Embeddings]:
    e = OpenAIEmbedding(model=model, api_key="sk-test", dim=dim)
    fake = _Embeddings()

    class Client:
        embeddings = fake

    e._client = Client()
    return e, fake


async def test_a_smaller_size_is_sent_as_dimensions() -> None:
    e, fake = _embedder("text-embedding-3-small", 256)
    await e.embed(["a"])
    assert fake.calls[0]["dimensions"] == 256


async def test_the_default_size_sends_no_dimensions() -> None:
    e, fake = _embedder("text-embedding-3-small", None)
    await e.embed(["a"])
    assert "dimensions" not in fake.calls[0]


def test_a_model_that_cannot_shorten_refuses_another_size() -> None:
    with pytest.raises(EmbeddingError, match="1536"):
        OpenAIEmbedding(model="text-embedding-ada-002", api_key="sk-test", dim=256)
    OpenAIEmbedding(model="text-embedding-ada-002", api_key="sk-test", dim=1536)
