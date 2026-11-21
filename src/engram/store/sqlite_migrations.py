"""Numbered schema changes for SQLite files.

``schema.sql`` builds version 2 (``BASELINE_VERSION``): the 0.3 side tables and
views. Every later change is a ``Migration`` with the next number. A writable
``SqliteStore.open`` runs, in its one write transaction, each migration whose
number is above the highest one the file records, and records each number as
it goes. A failed statement rolls the whole open back, so a file is always at
exactly one version.

Statements are literal SQL. Never format a value into one.
"""

from __future__ import annotations

from dataclasses import dataclass

BASELINE_VERSION = 2


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    # One erase epoch per scope. `erase_scope_residue` moves it in the
    # erase's own transaction, and a write that carries a `ScopeWriteGuard`
    # is refused if it has moved. A scope with no row is at epoch 0.
    Migration(
        version=3,
        statements=(
            """CREATE TABLE scope_state (
                org_id      TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                erase_epoch INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (org_id, user_id)
            )""",
        ),
    ),
    # Settings of the database as a whole, one row per key. Key "embedder"
    # holds the EmbedderIdentity key of the embedder its vectors come from;
    # Engram.open records it and refuses another (`set_embedder_key`).
    Migration(
        version=4,
        statements=(
            "CREATE TABLE engram_meta (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)",
        ),
    ),
)

SCHEMA_VERSION = BASELINE_VERSION + len(MIGRATIONS)


def target_version() -> int:
    """The version this library builds. Read from ``MIGRATIONS`` when called."""
    return BASELINE_VERSION + len(MIGRATIONS)


def pending(found: int) -> tuple[Migration, ...]:
    """The migrations a file at version ``found`` still needs, in order."""
    return tuple(m for m in MIGRATIONS if m.version > found)
