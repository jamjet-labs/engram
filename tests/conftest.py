"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from engram.scope import Scope
from engram.store.sqlite import SqliteStore
from tests.conformance.backends import (
    BACKENDS,
    Backend,
    OpenMemory,
    OpenStore,
    is_sqlite,
    selected_backend_names,
)


def pytest_configure(config: pytest.Config) -> None:
    # Fail the run at once, with the known names, on a typo in ENGRAM_TEST_BACKENDS.
    try:
        selected_backend_names()
    except ValueError as e:
        raise pytest.UsageError(str(e)) from None


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run every test that takes `backend` once per selected backend.

    `sqlite_only` keeps a test off backends that are not SQLite. `persistent`
    skips it on a backend that does not keep data between opens.
    """
    if "backend" not in metafunc.fixturenames:
        return
    sqlite_only = metafunc.definition.get_closest_marker("sqlite_only") is not None
    persistent = metafunc.definition.get_closest_marker("persistent") is not None
    params = []
    for name in selected_backend_names():
        b = BACKENDS[name]
        if sqlite_only and not is_sqlite(b):
            continue
        marks = []
        if persistent and not b.persistent:
            marks.append(pytest.mark.skip(reason=f"{name} does not keep data between opens"))
        params.append(pytest.param(name, id=name, marks=marks))
    metafunc.parametrize("backend", params, indirect=True)


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> Backend:
    return BACKENDS[request.param]


@pytest_asyncio.fixture
async def open_memory(backend: Backend, tmp_path: Path) -> AsyncIterator[OpenMemory]:
    """`await open_memory(**Engram.open kwargs)` opens an Engram on a fresh target.

    `await open_memory.reopen(**kw)` opens the same target again.
    """
    opener = OpenMemory(backend, tmp_path)
    try:
        yield opener
    finally:
        await opener.aclose()


@pytest_asyncio.fixture
async def open_store(backend: Backend, tmp_path: Path) -> AsyncIterator[OpenStore]:
    """`await open_store(**kw)` opens a store on a fresh target; `.reopen()` as above."""
    opener = OpenStore(backend, tmp_path)
    try:
        yield opener
    finally:
        await opener.aclose()


@pytest_asyncio.fixture
async def store(open_store: OpenStore) -> SqliteStore:
    """A store on a fresh target of each backend; `open_store` closes it."""
    return await open_store()


@pytest.fixture
def acme_alice() -> Scope:
    return Scope(org_id="acme", user_id="alice")
