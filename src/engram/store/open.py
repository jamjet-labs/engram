"""Open a store from a target."""

from __future__ import annotations

from pathlib import Path

from engram.store.base import EngramStore
from engram.store.sqlite import SqliteStore


async def open_store(target: str | Path, *, read_only: bool = False) -> EngramStore:
    """Open the store a target names: a SQLite file path, or ":memory:".

    ``read_only=True`` opens an existing store for reads only. Every write
    method then raises ``ReadOnlyStoreError``. See ``SqliteStore.open``.
    """
    return await SqliteStore.open(target, read_only=read_only)
