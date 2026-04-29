"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from engram.scope import Scope
from engram.store.sqlite import SqliteStore


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SqliteStore]:
    s = await SqliteStore.open(":memory:")
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
def acme_alice() -> Scope:
    return Scope(org_id="acme", user_id="alice")
