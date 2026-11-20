"""Tests for Engram.record() role convenience parameter."""

from __future__ import annotations

import pytest

from engram.embedding.synthetic import SyntheticEmbedding
from tests.conformance.backends import OpenMemory


@pytest.mark.asyncio
async def test_record_with_role_stores_role_in_metadata(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    fact = await memory.record(text="user prefers dark roast", role="user", user_id="alice")
    assert fact.metadata.get("role") == "user"


@pytest.mark.asyncio
async def test_record_without_role_leaves_metadata_empty(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    fact = await memory.record(text="hello", user_id="alice")
    assert "role" not in fact.metadata


@pytest.mark.asyncio
async def test_record_role_does_not_clobber_other_metadata(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    fact = await memory.record(
        text="hello",
        role="user",
        metadata={"source": "import"},
        user_id="alice",
    )
    assert fact.metadata == {"source": "import", "role": "user"}
