"""Storage backends the conformance tests run on.

A backend hands out targets: a fresh place to open a store, given as a path,
":memory:" or a database URL. A test that takes the `backend` fixture, or a
fixture built on it (`open_memory`, `open_store`), runs once per selected
backend. ENGRAM_TEST_BACKENDS selects them by name, comma-separated. Unset, it
selects every registered backend whose `default` is true.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import anyio

from engram.engram import Engram
from engram.store.sqlite import SqliteStore

BACKENDS_ENV = "ENGRAM_TEST_BACKENDS"


class Backend(Protocol):
    name: str  # "sqlite-memory", "sqlite-file", ...
    persistent: bool  # a reopen sees what an earlier open wrote
    default: bool  # runs when ENGRAM_TEST_BACKENDS does not name it

    async def new_target(self, tmp_path: Path) -> str:
        """A fresh, empty target that no other test uses."""
        ...

    async def drop(self, target: str) -> None:
        """Remove a target made by `new_target`."""
        ...


class SqliteMemoryBackend:
    """SQLite in memory. Every open is a new, empty database."""

    name = "sqlite-memory"
    persistent = False
    default = True

    async def new_target(self, tmp_path: Path) -> str:
        return ":memory:"

    async def drop(self, target: str) -> None:
        return None


class SqliteFileBackend:
    """A SQLite file in the test's own temporary directory."""

    name = "sqlite-file"
    persistent = True
    default = True

    async def new_target(self, tmp_path: Path) -> str:
        return str(tmp_path / f"engram-{uuid4().hex[:12]}.db")

    async def drop(self, target: str) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            await anyio.Path(target + suffix).unlink(missing_ok=True)


BACKENDS: dict[str, Backend] = {b.name: b for b in (SqliteMemoryBackend(), SqliteFileBackend())}


def is_sqlite(backend: Backend) -> bool:
    return backend.name.startswith("sqlite-")


def selected_backend_names(environ: Mapping[str, str] | None = None) -> list[str]:
    """The backends this run uses, in the order named or registered.

    Raises ValueError when ENGRAM_TEST_BACKENDS names a backend that is not
    registered.
    """
    env = os.environ if environ is None else environ
    named = [n.strip() for n in env.get(BACKENDS_ENV, "").split(",") if n.strip()]
    if not named:
        return [name for name, b in BACKENDS.items() if b.default]
    unknown = [n for n in named if n not in BACKENDS]
    if unknown:
        raise ValueError(
            f"{BACKENDS_ENV} names unknown backends: {', '.join(unknown)}. "
            f"Known backends: {', '.join(BACKENDS)}."
        )
    return list(dict.fromkeys(named))


class _Closeable(Protocol):
    async def close(self) -> None: ...


class _Opener[T: _Closeable]:
    """Opens handles on fresh targets of one backend and cleans up after.

    `await opener(**kw)` makes a new target and opens it, the way a test
    used to open ":memory:". `await opener.reopen(**kw)` opens the latest
    target again, as a restart or a second instance would. At teardown every
    handle is closed, then every target is dropped.
    """

    def __init__(self, backend: Backend, tmp_path: Path) -> None:
        self.backend = backend
        self._tmp_path = tmp_path
        self._targets: list[str] = []
        self._opened: list[T] = []

    @property
    def target(self) -> str:
        """The latest target, for code that opens it another way (a subprocess)."""
        if not self._targets:
            raise RuntimeError("nothing opened yet; call the opener first")
        return self._targets[-1]

    async def __call__(self, **kwargs: Any) -> T:
        target = await self.backend.new_target(self._tmp_path)
        self._targets.append(target)
        return await self._track(target, kwargs)

    async def reopen(self, **kwargs: Any) -> T:
        return await self._track(self.target, kwargs)

    async def _track(self, target: str, kwargs: dict[str, Any]) -> T:
        handle = await self._open_at(target, **kwargs)
        self._opened.append(handle)
        return handle

    async def _open_at(self, target: str, **kwargs: Any) -> T:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Close every handle, then drop every target. Raises the first error after."""
        errors: list[Exception] = []
        for handle in reversed(self._opened):
            try:
                await handle.close()
            except Exception as e:
                errors.append(e)
        for target in reversed(self._targets):
            try:
                await self.backend.drop(target)
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]


class OpenMemory(_Opener[Engram]):
    """The `open_memory` fixture: Engram instances on this test's targets.

    They open with ``allow_default_scope=True`` unless a test passes False,
    so tests written for 0.2 keep their meaning (most name a user and leave
    the org to "default").
    """

    async def _open_at(self, target: str, **kwargs: Any) -> Engram:
        kwargs.setdefault("allow_default_scope", True)
        return await Engram.open(target, **kwargs)


class OpenStore(_Opener[SqliteStore]):
    """The `open_store` fixture: stores on this test's targets."""

    async def _open_at(self, target: str, **kwargs: Any) -> SqliteStore:
        return await SqliteStore.open(target, **kwargs)
