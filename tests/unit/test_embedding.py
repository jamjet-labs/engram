from __future__ import annotations

import math
import os

import httpx
import pytest

from engram.embedding.ollama import OllamaEmbedding
from engram.embedding.synthetic import SyntheticEmbedding
from engram.errors import EmbeddingError

# ── SyntheticEmbedding ──────────────────────────────────────────────


async def test_synthetic_dim_matches_property() -> None:
    e = SyntheticEmbedding(dim=128)
    [v] = await e.embed(["hello"])
    assert e.dim == 128
    assert len(v) == 128


async def test_synthetic_is_deterministic() -> None:
    e = SyntheticEmbedding(dim=64)
    a = await e.embed(["alice prefers espresso"])
    b = await e.embed(["alice prefers espresso"])
    assert a == b


async def test_synthetic_different_text_different_vector() -> None:
    e = SyntheticEmbedding(dim=64)
    [a] = await e.embed(["alice"])
    [b] = await e.embed(["bob"])
    assert a != b


async def test_synthetic_empty_string_produces_unit_vector() -> None:
    e = SyntheticEmbedding(dim=8)
    [v] = await e.embed([""])
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-6)


async def test_synthetic_empty_batch_returns_empty() -> None:
    e = SyntheticEmbedding()
    out = await e.embed([])
    assert out == []


async def test_synthetic_normalized_to_unit_length() -> None:
    e = SyntheticEmbedding(dim=64)
    vs = await e.embed(["a", "b c", "the quick brown fox"])
    for v in vs:
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-5)


# ── OllamaEmbedding ─────────────────────────────────────────────────


def _ollama_running() -> bool:
    if os.environ.get("ENGRAM_SKIP_OLLAMA"):
        return False
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


@pytest.mark.skipif(not _ollama_running(), reason="ollama not running locally")
async def test_ollama_embed_returns_vectors() -> None:
    e = OllamaEmbedding(dim=768)
    vs = await e.embed(["hello world", "the quick brown fox"])
    assert len(vs) == 2
    assert all(len(v) == 768 for v in vs)
    # Vectors should be different
    assert vs[0] != vs[1]


@pytest.mark.skipif(not _ollama_running(), reason="ollama not running locally")
async def test_ollama_embed_empty_batch() -> None:
    e = OllamaEmbedding()
    assert await e.embed([]) == []


async def test_ollama_unreachable_raises() -> None:
    e = OllamaEmbedding(base_url="http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(EmbeddingError):
        await e.embed(["x"])
