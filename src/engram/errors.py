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
