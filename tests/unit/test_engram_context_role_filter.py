"""Tests for Engram.context() role_filter parameter."""

from __future__ import annotations

import pytest

from engram.embedding.synthetic import SyntheticEmbedding
from tests.conformance.backends import OpenMemory


@pytest.mark.asyncio
async def test_context_role_filter_user_only_returns_user_facts(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    await memory.record(text="user fact about coffee", role="user", user_id="alice")
    await memory.record(text="assistant fact about coffee", role="assistant", user_id="alice")

    ctx = await memory.context(
        query="coffee",
        user_id="alice",
        role_filter=("user",),
        token_budget=1500,
    )
    assert "user fact about coffee" in ctx
    assert "assistant fact about coffee" not in ctx


@pytest.mark.asyncio
async def test_context_no_role_filter_returns_all_facts(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    await memory.record(text="user fact about coffee", role="user", user_id="alice")
    await memory.record(text="assistant fact about coffee", role="assistant", user_id="alice")

    ctx = await memory.context(query="coffee", user_id="alice", token_budget=1500)
    assert "user fact about coffee" in ctx
    assert "assistant fact about coffee" in ctx


@pytest.mark.asyncio
async def test_context_role_filter_empty_when_no_facts_match(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    await memory.record(text="assistant only", role="assistant", user_id="alice")

    ctx = await memory.context(
        query="anything",
        user_id="alice",
        role_filter=("user",),
        token_budget=1500,
    )
    assert ctx == ""


@pytest.mark.asyncio
async def test_context_role_filter_multiple_roles(open_memory: OpenMemory):
    embedder = SyntheticEmbedding(dim=128)
    memory = await open_memory(embedder=embedder)
    await memory.record(text="user thing", role="user", user_id="alice")
    await memory.record(text="assistant thing", role="assistant", user_id="alice")
    await memory.record(text="system thing", role="system", user_id="alice")

    ctx = await memory.context(
        query="thing",
        user_id="alice",
        role_filter=("user", "assistant"),
        token_budget=1500,
    )
    assert "user thing" in ctx
    assert "assistant thing" in ctx
    assert "system thing" not in ctx
