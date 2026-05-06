"""Tests for Engram.record() role convenience parameter."""

from __future__ import annotations

import pytest

from engram import Engram
from engram.embedding.synthetic import SyntheticEmbedding


@pytest.mark.asyncio
async def test_record_with_role_stores_role_in_metadata():
    embedder = SyntheticEmbedding(dim=128)
    async with await Engram.open(":memory:", embedder=embedder) as memory:
        fact = await memory.record(text="user prefers dark roast", role="user", user_id="alice")
        assert fact.metadata.get("role") == "user"


@pytest.mark.asyncio
async def test_record_without_role_leaves_metadata_empty():
    embedder = SyntheticEmbedding(dim=128)
    async with await Engram.open(":memory:", embedder=embedder) as memory:
        fact = await memory.record(text="hello", user_id="alice")
        assert "role" not in fact.metadata


@pytest.mark.asyncio
async def test_record_role_does_not_clobber_other_metadata():
    embedder = SyntheticEmbedding(dim=128)
    async with await Engram.open(":memory:", embedder=embedder) as memory:
        fact = await memory.record(
            text="hello",
            role="user",
            metadata={"source": "import"},
            user_id="alice",
        )
        assert fact.metadata == {"source": "import", "role": "user"}
