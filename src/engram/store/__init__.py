from engram.store.base import EngramStore
from engram.store.open import open_store
from engram.store.sqlite import SqliteStore

__all__ = ["EngramStore", "SqliteStore", "open_store"]


def _sqlite_is_a_store(s: SqliteStore) -> EngramStore:
    # Never called. mypy fails here if SqliteStore stops meeting the protocol.
    return s
