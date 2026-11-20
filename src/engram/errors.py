"""Engram v2 error hierarchy.

Mirrors `MemoryError` variants from Rust v0.5.x so HTTP error responses are stable.
"""


class EngramError(Exception):
    """Base for all Engram errors."""


class StoreError(EngramError):
    """Storage backend failure (SQLite, vector index, etc.)."""


class ReadOnlyStoreError(StoreError):
    """A write was asked of a store opened read-only. Nothing was written."""


class SchemaVersionError(StoreError):
    """The database's schema version is not one this library can open in this mode."""


class EmbedderMismatchError(StoreError):
    """The database records another embedder than the one this open was given.

    Vectors from two embedders do not compare, so the open is refused and
    nothing is written. The message names both embedder keys and the fix:
    open with ``on_embedder_change="switch"``, then re-embed.
    """


class NotFoundError(EngramError):
    """Requested resource (fact, session, entity) does not exist."""


class ScopeError(EngramError):
    """Operation crossed a scope boundary that's not permitted."""


class ScopeRequiredError(ValueError, EngramError):
    """A call did not name its scope, or named it with an id it cannot use.

    A missing ``user_id`` or ``org_id``, or the literal ``"default"``, is
    refused unless the Engram was opened with ``allow_default_scope=True``.
    An id that is not a string, or is empty or blank, is refused always.
    Nothing was read or written.
    """


class ValidationError(EngramError):
    """Input failed validation beyond what Pydantic catches (e.g., business rules)."""


class ExtractionError(EngramError):
    """Extraction pipeline failure (LLM error, schema mismatch, etc.)."""


class EmbeddingError(EngramError):
    """Embedding provider failure."""


class EmbeddingResponseError(EmbeddingError):
    """The embedder answered, but not with one usable vector per text.

    A wrong count, a wrong length, a NaN or infinite value, or a zero vector.
    The whole answer is refused, so nothing from it is stored or indexed. The
    message names the vector's index, never its text.
    """


class EmbeddingAuthError(EmbeddingError):
    """The embedding service refused the credentials, or none could be found or refreshed.

    Trying again with the same credentials will not help. Nothing was embedded.
    """


class EmbeddingQuotaError(EmbeddingError):
    """The embedding service kept answering 429 (quota or rate limit) until the retries ran out.

    Nothing was embedded. Try again later, or ask for more quota.
    """


class EmbeddingTimeoutError(EmbeddingError):
    """No usable answer came before the retries or the deadline ran out.

    Timeouts, lost connections and 5xx answers end here. Nothing was embedded.
    """


class ScopeErasedError(EngramError):
    """A write lost a race with ``Engram.erase_user`` on its scope. Nothing was stored.

    Raised by a write given a ``ScopeWriteGuard`` (``guard=...``) when the
    scope's erase epoch has moved since the guard was taken: an erase of the
    scope committed after the write started. Not a StoreError: the store is
    fine, and retrying would store the data the erase was asked to remove.
    """


class ScopeResidueRefusedError(StoreError):
    """``erase_scope_residue`` found facts in the scope, so it deleted nothing.

    A fact was written to the scope while ``Engram.erase_user`` ran. Calling
    ``erase_user`` again erases it and finishes. Other StoreErrors from
    ``erase_user`` are storage faults that a retry may not fix.
    """
