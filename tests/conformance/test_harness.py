"""Checks on the conformance harness itself.

The harness runs a test once per selected storage backend. These tests check
that it does, that markers choose the right backends, that each test gets its
own target, that a reopen sees the same data, and that conformance files open
memory only through the fixtures.
"""

from __future__ import annotations

import ast
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engram.models import Fact
from engram.scope import Scope
from tests.conformance.backends import (
    BACKENDS,
    BACKENDS_ENV,
    OpenMemory,
    OpenStore,
    selected_backend_names,
)

pytest_plugins = ["pytester"]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FILES = HERE / "FILES"

# ── Backend selection ───────────────────────────────────────────────────

_PROBE = """
import pytest

def test_plain(backend):
    pass

@pytest.mark.persistent
def test_persistent(backend):
    pass

@pytest.mark.sqlite_only
def test_sqlite_only(backend):
    pass

def test_without_backend():
    pass
"""


def _write_probe(pytester: pytest.Pytester, pytestconfig: pytest.Config) -> None:
    """A probe project that loads our conftest and registers our markers."""
    markers = json.dumps(pytestconfig.getini("markers"))
    pytester.makepyprojecttoml(
        "[tool.pytest.ini_options]\n"
        'addopts = "--strict-markers"\n'
        'asyncio_default_fixture_loop_scope = "function"\n'
        f"markers = {markers}\n"
    )
    pytester.makeconftest("pytest_plugins = ['tests.conftest']\n")
    pytester.makepyfile(test_probe=_PROBE)


def _run_probe(pytester: pytest.Pytester, pytestconfig: pytest.Config) -> dict[str, str]:
    """Run the probe module in a fresh pytest session.

    Returns each test id (without the file name) mapped to "passed" or
    "skipped". Any failure fails the caller.
    """
    _write_probe(pytester, pytestconfig)
    reprec = pytester.inline_run("-p", "no:cacheprovider")
    passed, skipped, failed = reprec.listoutcomes()
    assert failed == []
    assert reprec.ret == pytest.ExitCode.OK
    outcomes = {r.nodeid.split("::", 1)[1]: "passed" for r in passed}
    outcomes.update({r.nodeid.split("::", 1)[1]: "skipped" for r in skipped})
    return outcomes


def _ran(outcomes: dict[str, str], test: str) -> list[str]:
    """The backend ids one probe test ran with, in sorted order."""
    prefix = f"{test}["
    return sorted(tid[len(prefix) : -1] for tid in outcomes if tid.startswith(prefix))


def test_every_registered_backend_runs(
    pytester: pytest.Pytester, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under this run's selection, a test taking `backend` runs once per
    # selected backend, and a test without it runs once.
    selected = selected_backend_names()
    outcomes = _run_probe(pytester, pytestconfig)
    assert _ran(outcomes, "test_plain") == sorted(selected)
    assert {outcomes[f"test_plain[{name}]"] for name in selected} == {"passed"}
    assert outcomes["test_without_backend"] == "passed"

    # With nothing named, the selection is every registered default backend.
    monkeypatch.delenv(BACKENDS_ENV, raising=False)
    defaults = [name for name, b in BACKENDS.items() if b.default]
    assert {"sqlite-memory", "sqlite-file"} <= set(defaults)
    assert selected_backend_names() == defaults
    outcomes = _run_probe(pytester, pytestconfig)
    assert _ran(outcomes, "test_plain") == sorted(defaults)

    # A name narrows the run to that backend.
    monkeypatch.setenv(BACKENDS_ENV, " sqlite-file ")
    outcomes = _run_probe(pytester, pytestconfig)
    assert _ran(outcomes, "test_plain") == ["sqlite-file"]


class _RemoteFake:
    """A non-SQLite backend that is never opened; it only receives tests."""

    name = "remote-fake"
    persistent = True
    default = False

    async def new_target(self, tmp_path: Path) -> str:
        raise AssertionError("the probe never opens a target")

    async def drop(self, target: str) -> None:
        raise AssertionError("the probe never opens a target")


def test_markers_choose_the_backends_a_test_runs_on(
    pytester: pytest.Pytester, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(BACKENDS, "remote-fake", _RemoteFake())
    monkeypatch.setenv(BACKENDS_ENV, "sqlite-memory,sqlite-file,remote-fake")
    assert _run_probe(pytester, pytestconfig) == {
        "test_plain[sqlite-memory]": "passed",
        "test_plain[sqlite-file]": "passed",
        "test_plain[remote-fake]": "passed",
        "test_persistent[sqlite-memory]": "skipped",
        "test_persistent[sqlite-file]": "passed",
        "test_persistent[remote-fake]": "passed",
        "test_sqlite_only[sqlite-memory]": "passed",
        "test_sqlite_only[sqlite-file]": "passed",
        "test_without_backend": "passed",
    }


def test_an_unknown_backend_name_stops_the_run(
    pytester: pytest.Pytester, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BACKENDS_ENV, "sqlite-file,nosuch")
    _write_probe(pytester, pytestconfig)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*ENGRAM_TEST_BACKENDS*nosuch*sqlite-memory, sqlite-file*"])


# ── Targets ─────────────────────────────────────────────────────────────


@pytest.mark.persistent
async def test_reopen_sees_the_first_engrams_writes(open_memory: OpenMemory) -> None:
    first = await open_memory()
    fact = await first.record(
        "Alice keeps her bike in the blue shed.", user_id="alice", org_id="acme"
    )
    await first.close()

    second = await open_memory.reopen()
    hits = await second.recall("blue shed", user_id="alice", org_id="acme")
    # The second instance has an empty vector index, so the keyword arm found it.
    assert [h.fact.id for h in hits] == [fact.id]
    assert hits[0].fact.text == "Alice keeps her bike in the blue shed."
    assert hits[0].keyword_score > 0


@pytest.mark.parametrize("run", ["first", "second"])
async def test_targets_are_isolated_between_tests(
    open_store: OpenStore, acme_alice: Scope, run: str
) -> None:
    store = await open_store()
    # Both runs write the word "isolation" into the same scope. Neither may
    # see the other's row, whichever runs first.
    assert await store.keyword_search("isolation", acme_alice) == []
    fact = Fact(
        text=f"isolation check from the {run} run",
        scope=acme_alice,
        valid_from=datetime.now(UTC),
    )
    await store.upsert_fact(fact)
    found = await store.keyword_search("isolation", acme_alice)
    assert [f.id for f in found] == [fact.id]


# ── The FILES guard ─────────────────────────────────────────────────────

_OPENERS = frozenset({"Engram", "SqliteStore"})


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return None


def _is_sqlite_only_mark(node: ast.expr) -> bool:
    name = _dotted(node)
    return name is not None and (name == "mark.sqlite_only" or name.endswith(".mark.sqlite_only"))


def _module_is_sqlite_only(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            continue
        marks = stmt.value.elts if isinstance(stmt.value, ast.List | ast.Tuple) else [stmt.value]
        if any(_is_sqlite_only_mark(m) for m in marks):
            return True
    return False


def direct_opens(source: str) -> list[tuple[int, str]]:
    """Every `Engram.open(` or `SqliteStore.open(` call a conformance file may not make.

    A call is allowed only inside a test function marked `sqlite_only`, or
    anywhere in a module whose `pytestmark` carries `sqlite_only`. Helpers and
    fixtures are not tests, so a call there is reported even if the helper
    carries the mark. Returns (line, owner) pairs.
    """
    tree = ast.parse(source)
    if _module_is_sqlite_only(tree):
        return []
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST, allowed: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_allowed = allowed
            if (
                isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                and child.name.startswith("test")
                and any(_is_sqlite_only_mark(d) for d in child.decorator_list)
            ):
                child_allowed = True
            if (
                not child_allowed
                and isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "open"
            ):
                owner = _dotted(child.func.value)
                if owner is not None and owner.rsplit(".", 1)[-1] in _OPENERS:
                    found.append((child.lineno, owner))
            visit(child, child_allowed)

    visit(tree, False)
    return found


def conformance_files() -> list[str]:
    """Paths listed in FILES, relative to the repository root."""
    lines = FILES.read_text(encoding="utf-8").splitlines()
    return [s for line in lines if (s := line.strip()) and not s.startswith("#")]


def test_conformance_files_never_open_engram_directly() -> None:
    problems: list[str] = []
    for rel in conformance_files():
        path = REPO_ROOT / rel
        if not path.is_file():
            problems.append(f"{rel}: listed in {FILES.name} but not found")
            continue
        for line, owner in direct_opens(path.read_text(encoding="utf-8")):
            problems.append(f"{rel}:{line}: {owner}.open(")
    assert problems == [], (
        "Conformance files open memory through the open_memory or open_store "
        "fixture, or mark the test sqlite_only:\n" + "\n".join(problems)
    )


_FINDER_CASES = {
    "call in a test": (
        """
        async def test_x():
            memory = await Engram.open(":memory:")
        """,
        [(3, "Engram")],
    ),
    "call in a helper": (
        """
        async def setup():
            return await SqliteStore.open(":memory:")
        """,
        [(3, "SqliteStore")],
    ),
    "call in a fixture": (
        """
        @pytest_asyncio.fixture
        async def store():
            yield await SqliteStore.open(":memory:")
        """,
        [(4, "SqliteStore")],
    ),
    "dotted owner": (
        """
        async def test_x():
            async with await engram.Engram.open() as memory:
                pass
        """,
        [(3, "engram.Engram")],
    ),
    "test marked sqlite_only": (
        """
        @pytest.mark.sqlite_only
        async def test_x():
            memory = await Engram.open(":memory:")
            store = await SqliteStore.open(":memory:")
        """,
        [],
    ),
    "module marked sqlite_only": (
        """
        pytestmark = [pytest.mark.asyncio, pytest.mark.sqlite_only]

        async def setup():
            return await SqliteStore.open(":memory:")
        """,
        [],
    ),
    "helper carrying the mark": (
        """
        @pytest.mark.sqlite_only
        async def setup():
            return await SqliteStore.open(":memory:")
        """,
        [(4, "SqliteStore")],
    ),
    "other marks do not exempt": (
        """
        @pytest.mark.persistent
        async def test_x():
            memory = await Engram.open(":memory:")
        """,
        [(4, "Engram")],
    ),
    "fixtures and other opens": (
        """
        async def test_x(open_memory, open_store, tmp_path):
            memory = await open_memory()
            store = await open_store()
            with open(tmp_path / "f") as fh, (tmp_path / "g").open() as gh:
                pass
        """,
        [],
    ),
}


@pytest.mark.parametrize(
    ("source", "expected"), list(_FINDER_CASES.values()), ids=list(_FINDER_CASES)
)
def test_direct_open_finder(source: str, expected: list[tuple[int, str]]) -> None:
    assert direct_opens(textwrap.dedent(source)) == expected
