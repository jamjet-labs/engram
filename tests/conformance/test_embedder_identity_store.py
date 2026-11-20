"""The store records which embedder made its vectors, and Engram.open holds to it.

Vectors from two embedders do not compare. The first writable open records
the embedder's key (provider/model/dim) in the store. A later open with
another key is refused unless it asks to switch, and a read-only open with
another key is refused always. Two processes that open one store at once with
different embedders cannot both win.
"""

from __future__ import annotations

import sqlite3
import time
import warnings
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from engram import Engram
from engram.embedding import EmbedderIdentity, SyntheticEmbedding
from engram.errors import EmbedderMismatchError, ReadOnlyStoreError, StoreError
from engram.models import Fact
from engram.scope import Scope
from engram.store import sqlite_migrations
from engram.store.sqlite import SqliteStore
from tests.conformance.backends import Backend, OpenMemory, OpenStore
from tests.conformance.test_multiprocess_sqlite import start_workers

WHO = {"org_id": "acme", "user_id": "alice"}
DIM = 8
BIKE = "Alice keeps her bike in the blue shed."
FIX = "open with on_embedder_change='switch', then run engram-admin reembed"
# The migration that adds engram_meta. A file below it is from before the
# embedder was recorded.
META_MIGRATION = 4


class NamedEmbedder(SyntheticEmbedding):
    """Synthetic vectors under a model name of the test's choosing."""

    def __init__(self, model: str, dim: int = DIM) -> None:
        super().__init__(dim=dim)
        self._model = model

    @property
    def identity(self) -> EmbedderIdentity:
        return EmbedderIdentity("test", self._model, self.dim)


class AnonymousEmbedder:
    """Only `dim` and `embed`, as an embedder written for 0.2 has."""

    def __init__(self) -> None:
        self._inner = SyntheticEmbedding(dim=DIM)

    @property
    def dim(self) -> int:
        return DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed(texts)


async def _key(memory: Engram) -> str | None:
    return await memory._store.embedder_key()


def _ro(target: str) -> closing[sqlite3.Connection]:
    return closing(sqlite3.connect(Path(target).absolute().as_uri() + "?mode=ro", uri=True))


def _meta(target: str) -> list[tuple[Any, ...]]:
    with _ro(target) as db:
        return db.execute("SELECT key, value FROM engram_meta ORDER BY key").fetchall()


def _versions(target: str) -> list[int]:
    with _ro(target) as db:
        return [r[0] for r in db.execute("SELECT version FROM schema_version ORDER BY 1")]


def _tables(target: str) -> set[str]:
    with _ro(target) as db:
        return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


# ── One process ─────────────────────────────────────────────────────────


async def test_first_writable_open_records_the_embedder(open_memory: OpenMemory) -> None:
    memory = await open_memory(embedder=NamedEmbedder("model-a"))
    assert await _key(memory) == "test/model-a/8"
    # With no embedder given, Engram's default is the one recorded.
    default = await open_memory()
    assert await _key(default) == "synthetic/blake2b-v1/384"


@pytest.mark.persistent
async def test_same_embedder_reopens(open_memory: OpenMemory) -> None:
    first = await open_memory(embedder=NamedEmbedder("model-a"))
    fact = await first.record(BIKE, **WHO)
    await first.close()
    for _ in range(2):
        again = await open_memory.reopen(embedder=NamedEmbedder("model-a"))
        assert await _key(again) == "test/model-a/8"
        assert [h.fact.id for h in await again.recall("blue shed", **WHO)] == [fact.id]
        await again.close()


@pytest.mark.persistent
async def test_other_model_is_refused(open_memory: OpenMemory) -> None:
    first = await open_memory(embedder=NamedEmbedder("model-a"))
    fact = await first.record(BIKE, **WHO)
    await first.close()

    with pytest.raises(EmbedderMismatchError) as caught:
        await open_memory.reopen(embedder=NamedEmbedder("model-b"))
    message = str(caught.value)
    assert "'test/model-a/8'" in message
    assert "'test/model-b/8'" in message
    assert FIX in message
    assert BIKE not in message
    assert isinstance(caught.value, StoreError)

    # Nothing changed: the recorded embedder still opens and finds the fact.
    again = await open_memory.reopen(embedder=NamedEmbedder("model-a"))
    assert await _key(again) == "test/model-a/8"
    assert [h.fact.id for h in await again.recall("blue shed", **WHO)] == [fact.id]


@pytest.mark.persistent
async def test_same_model_other_dimension_is_refused(open_memory: OpenMemory) -> None:
    await (await open_memory(embedder=SyntheticEmbedding(dim=384))).close()

    with pytest.raises(EmbedderMismatchError) as caught:
        await open_memory.reopen(embedder=SyntheticEmbedding(dim=256))
    message = str(caught.value)
    assert "'synthetic/blake2b-v1/384'" in message
    assert "'synthetic/blake2b-v1/256'" in message

    again = await open_memory.reopen(embedder=SyntheticEmbedding(dim=384))
    assert await _key(again) == "synthetic/blake2b-v1/384"


@pytest.mark.persistent
async def test_switch_records_the_new_key(
    open_memory: OpenMemory, caplog: pytest.LogCaptureFixture
) -> None:
    first = await open_memory(embedder=NamedEmbedder("model-a"))
    fact = await first.record(BIKE, **WHO)
    await first.close()

    caplog.clear()
    switched = await open_memory.reopen(
        embedder=NamedEmbedder("model-b"), on_embedder_change="switch"
    )
    assert await _key(switched) == "test/model-b/8"
    # The memory stays; only the recorded embedder changed, and the log says so.
    assert [h.fact.id for h in await switched.recall("blue shed", **WHO)] == [fact.id]
    [logged] = [r.getMessage() for r in caplog.records if r.levelno >= 30]
    assert "test/model-a/8" in logged and "test/model-b/8" in logged
    await switched.close()

    # From now on the new embedder opens and the old one is refused.
    assert await _key(await open_memory.reopen(embedder=NamedEmbedder("model-b"))) == (
        "test/model-b/8"
    )
    with pytest.raises(EmbedderMismatchError):
        await open_memory.reopen(embedder=NamedEmbedder("model-a"))
    # Switching to the key already recorded changes nothing.
    same = await open_memory.reopen(embedder=NamedEmbedder("model-b"), on_embedder_change="switch")
    assert await _key(same) == "test/model-b/8"


@pytest.mark.sqlite_only
@pytest.mark.persistent
async def test_legacy_file_records_on_first_writable_open(
    backend: Backend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = await backend.new_target(tmp_path)
    # A file from before the embedder was recorded: no engram_meta, one fact.
    with monkeypatch.context() as m:
        older = tuple(x for x in sqlite_migrations.MIGRATIONS if x.version < META_MIGRATION)
        m.setattr(sqlite_migrations, "MIGRATIONS", older)
        store = await SqliteStore.open(target)
        fact = Fact(text=BIKE, scope=Scope(**WHO), valid_from=datetime.now(UTC))
        await store.upsert_fact(fact)
        await store.close()
    assert max(_versions(target)) == META_MIGRATION - 1
    assert "engram_meta" not in _tables(target)

    memory = await Engram.open(target, embedder=NamedEmbedder("model-a"))
    try:
        assert await memory._store.embedder_key() == "test/model-a/8"
        assert [h.fact.id for h in await memory.recall("blue shed", **WHO)] == [fact.id]
    finally:
        await memory.close()
    assert META_MIGRATION in _versions(target)
    assert _meta(target) == [("embedder", "test/model-a/8")]


@pytest.mark.persistent
async def test_read_only_open_with_a_mismatch_is_refused(open_memory: OpenMemory) -> None:
    first = await open_memory(embedder=NamedEmbedder("model-a"))
    fact = await first.record(BIKE, **WHO)
    await first.close()

    with pytest.raises(EmbedderMismatchError) as caught:
        await open_memory.reopen(read_only=True, embedder=NamedEmbedder("model-b"))
    assert "'test/model-a/8'" in str(caught.value)
    assert "'test/model-b/8'" in str(caught.value)

    # A read-only open cannot switch, whatever the key.
    for model in ("model-a", "model-b"):
        with pytest.raises(ReadOnlyStoreError, match="switch"):
            await open_memory.reopen(
                read_only=True, embedder=NamedEmbedder(model), on_embedder_change="switch"
            )

    # The recorded embedder still opens read-only, and nothing changed.
    reader = await open_memory.reopen(read_only=True, embedder=NamedEmbedder("model-a"))
    assert await _key(reader) == "test/model-a/8"
    assert [h.fact.id for h in await reader.recall("blue shed", **WHO)] == [fact.id]


@pytest.mark.sqlite_only
@pytest.mark.persistent
async def test_read_only_open_records_nothing(backend: Backend, tmp_path: Path) -> None:
    target = await backend.new_target(tmp_path)
    await (await SqliteStore.open(target)).close()  # the schema, and no embedder yet

    reader = await Engram.open(target, read_only=True, embedder=NamedEmbedder("model-a"))
    try:
        assert await reader._store.embedder_key() is None
    finally:
        await reader.close()
    assert _meta(target) == []

    # The first writable open records its own embedder, not the reader's.
    writer = await Engram.open(target, embedder=NamedEmbedder("model-b"))
    await writer.close()
    assert _meta(target) == [("embedder", "test/model-b/8")]


@pytest.mark.sqlite_only
@pytest.mark.persistent
async def test_a_refused_open_closes_the_store_it_opened_and_not_one_it_was_given(
    open_memory: OpenMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    await (await open_memory(embedder=NamedEmbedder("model-a"))).close()
    closed: list[SqliteStore] = []
    real_close = SqliteStore.close

    async def counting_close(self: SqliteStore) -> None:
        closed.append(self)
        await real_close(self)

    monkeypatch.setattr(SqliteStore, "close", counting_close)
    for read_only in (False, True):
        with pytest.raises(EmbedderMismatchError):
            await open_memory.reopen(read_only=read_only, embedder=NamedEmbedder("model-b"))
    assert len(closed) == 2

    # A store the caller passed in stays open for the caller.
    given = await SqliteStore.open(open_memory.target)
    with pytest.raises(EmbedderMismatchError):
        await Engram.open(store=given, embedder=NamedEmbedder("model-b"))
    assert closed[2:] == []
    assert await given.embedder_key() == "test/model-a/8"
    await given.close()


async def test_an_embedder_without_an_identity_warns_once_per_open(
    open_memory: OpenMemory,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        memory = await open_memory(embedder=AnonymousEmbedder())
        await memory.record(BIKE, **WHO)
        await memory.recall("blue shed", **WHO)
    assert [w.category for w in caught] == [UserWarning]
    assert await _key(memory) == f"{__name__}.AnonymousEmbedder//{DIM}"


async def test_on_embedder_change_must_be_refuse_or_switch(open_memory: OpenMemory) -> None:
    for value in ("replace", "Switch", "", None, True):
        with pytest.raises(ValueError, match="on_embedder_change"):
            await open_memory(on_embedder_change=value)


# ── The store ───────────────────────────────────────────────────────────


async def test_store_records_one_key_and_refuses_another_unless_replace(
    open_store: OpenStore,
) -> None:
    store = await open_store()
    assert await store.embedder_key() is None
    await store.set_embedder_key("test/model-a/8", replace=False)
    await store.set_embedder_key("test/model-a/8", replace=False)  # the same key: nothing to do
    assert await store.embedder_key() == "test/model-a/8"

    with pytest.raises(EmbedderMismatchError) as caught:
        await store.set_embedder_key("test/model-b/8", replace=False)
    assert "'test/model-a/8'" in str(caught.value)
    assert "'test/model-b/8'" in str(caught.value)
    assert await store.embedder_key() == "test/model-a/8"

    await store.set_embedder_key("test/model-b/8", replace=True)
    assert await store.embedder_key() == "test/model-b/8"

    for bad in ("", "   ", "test/a\x00b/8"):
        with pytest.raises(ValueError):
            await store.set_embedder_key(bad, replace=True)
    with pytest.raises(TypeError):
        await store.set_embedder_key("test/model-c/8", replace="yes")  # type: ignore[arg-type]
    assert await store.embedder_key() == "test/model-b/8"


# ── Two processes ───────────────────────────────────────────────────────

# A check made outside the write's transaction lets both processes win in
# about half of all rounds here, so twelve rounds leave it almost no chance.
ROUNDS = 12
EMBEDDERS = {
    "synthetic:384": "synthetic/blake2b-v1/384",
    "synthetic:256": "synthetic/blake2b-v1/256",
}


@pytest.mark.multiprocess
@pytest.mark.persistent
@pytest.mark.sqlite_only
async def test_two_processes_with_different_embedders_open_at_once_one_wins(
    backend: Backend, tmp_path: Path
) -> None:
    specs = list(EMBEDDERS)
    for n in range(ROUNDS):
        target = await backend.new_target(tmp_path)
        if n % 2:
            # A file that has the schema and no embedder yet, as a store-level
            # open leaves it. Otherwise the file does not exist yet.
            await (await SqliteStore.open(target)).close()
        async with start_workers(
            target, None, None, barrier=True, extra_args=[["--embedder", s] for s in specs]
        ) as workers:
            for w in workers:
                await w.send(op="open_only")
            at = time.time() + 0.2
            for w in workers:
                await w.send(op="go", at=at)
            replies = [await w.reply() for w in workers]

        oks = [r["ok"] for r in replies]
        assert sorted(oks) == [False, True], (n, replies)
        won, lost = oks.index(True), oks.index(False)
        assert replies[lost]["error"] == "EmbedderMismatchError", (n, replies)
        assert EMBEDDERS[specs[won]] in replies[lost]["message"]
        assert EMBEDDERS[specs[lost]] in replies[lost]["message"]
        assert _meta(target) == [("embedder", EMBEDDERS[specs[won]])]
