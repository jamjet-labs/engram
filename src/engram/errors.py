"""Engram v2 error hierarchy.

Mirrors `MemoryError` variants from Rust v0.5.x so HTTP error responses are stable.
"""


class EngramError(Exception):
    """Base for all Engram errors."""


class StoreError(EngramError):
    """Storage backend failure (SQLite, vector index, etc.)."""


class NotFoundError(EngramError):
    """Requested resource (fact, session, entity) does not exist."""


class ScopeError(EngramError):
    """Operation crossed a scope boundary that's not permitted."""


class ValidationError(EngramError):
    """Input failed validation beyond what Pydantic catches (e.g., business rules)."""


class ExtractionError(EngramError):
    """Extraction pipeline failure (LLM error, schema mismatch, etc.)."""


class EmbeddingError(EngramError):
    """Embedding provider failure."""


class ScopeErasedError(EngramError):
    """A write lost a race with ``Engram.erase_user`` on its scope. Nothing was stored.

    Raised by ``SqliteStore.upsert_fact(not_erased_since=...)`` when the
    scope's data was erased at or after the time the write started. Not a
    StoreError: the store is fine, and retrying would store the data the
    erase was asked to remove.
    """


class ScopeResidueRefusedError(StoreError):
    """``erase_scope_residue`` found facts in the scope, so it deleted nothing.

    A fact was written to the scope while ``Engram.erase_user`` ran. Calling
    ``erase_user`` again erases it and finishes. Other StoreErrors from
    ``erase_user`` are storage faults that a retry may not fix.
    """
