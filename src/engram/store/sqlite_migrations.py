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


MIGRATIONS: tuple[Migration, ...] = ()

SCHEMA_VERSION = BASELINE_VERSION + len(MIGRATIONS)


def target_version() -> int:
    """The version this library builds. Read from ``MIGRATIONS`` when called."""
    return BASELINE_VERSION + len(MIGRATIONS)


def pending(found: int) -> tuple[Migration, ...]:
    """The migrations a file at version ``found`` still needs, in order."""
    return tuple(m for m in MIGRATIONS if m.version > found)
