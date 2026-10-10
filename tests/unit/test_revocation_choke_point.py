"""Structural guard: revocation is enforced in one place.

Every SqliteStore read that can hand content to a caller must select from the
``live_*`` views, which are the only place visibility is decided. Direct reads
of the base tables are allowed only in the governance methods listed here,
which return no fact text to callers (or delete it).
"""

from __future__ import annotations

import ast
import inspect
import re

import engram.store.sqlite as sqlite_mod

BASE_TABLE_READ = re.compile(r"\b(?:FROM|JOIN)\s+(facts|events|messages)\b(?!_)", re.IGNORECASE)

GOVERNANCE_METHODS = {
    "_backfill_fact_meta",  # computes content hashes for pre-0.3 rows
    "_governance_fact",  # unfiltered lookup used by revoke/reinstate/erase
    "derivation_targets_in_scope",  # returns ids only
    "erase",  # finds what to delete
    "explain",  # returns status and provenance, never text
}


def _sql_reads_by_method() -> dict[str, list[str]]:
    tree = ast.parse(inspect.getsource(sqlite_mod))
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SqliteStore":
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                hits = [
                    m.group(0)
                    for c in ast.walk(fn)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)
                    for m in BASE_TABLE_READ.finditer(c.value)
                ]
                if hits:
                    out[fn.name] = hits
    return out


def test_only_governance_methods_read_base_tables() -> None:
    offenders = {k: v for k, v in _sql_reads_by_method().items() if k not in GOVERNANCE_METHODS}
    assert offenders == {}


def test_governance_allowlist_has_no_stale_entries() -> None:
    """Keep the allowlist exact so widening it is a visible, reviewed change."""
    assert set(_sql_reads_by_method()) == GOVERNANCE_METHODS


def test_guard_detects_a_base_table_read() -> None:
    """Control for the guard itself: the pattern must match what it is meant to catch."""
    assert BASE_TABLE_READ.search("SELECT * FROM facts WHERE id=?")
    assert BASE_TABLE_READ.search("x JOIN messages ON")
    assert not BASE_TABLE_READ.search("SELECT * FROM live_facts")
    assert not BASE_TABLE_READ.search("FROM facts_fts WHERE")
