"""Revocation, erasure and provenance records.

Revocation is a tombstone. A revoked fact stays in the database (so the
history and the audit trail survive) but no read path returns it. Visibility is
decided in one place: the ``hidden_facts`` view in ``store/schema.sql``. Every
store read selects from the ``live_*`` views built on it, so vector search,
keyword search, session ranking, event search, transcript reads and the
HTTP/MCP/tool layers above them all inherit the same rule.

A tombstone hides:

* the revoked fact itself (by id);
* any fact in the same scope whose text normalizes to the same string
  (see :func:`normalize_text`), whether written before or after the revocation;
* any fact recorded with ``derived_from`` pointing at a hidden fact,
  transitively.

Rule for re-insertion: a plain write cannot un-revoke. Writing the same claim
again (exact text or a trivial variant) produces a fact that is born hidden.
Making the claim visible again takes :meth:`Engram.reinstate`, which needs an
actor and a reason and is recorded on the tombstone. Paraphrases are not
matched; a reworded claim needs its own revocation or a ``derived_from`` link.

Erasure is different: it deletes the content and everything derived from it
and leaves an :class:`ErasureRecord` that carries ids, counts and who/why, but
no content.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from engram.errors import EngramError

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


class RevokedContentError(EngramError):
    """A write matched revoked content and the caller asked for a hard refusal."""


def normalize_text(text: str) -> str:
    """Fold the variants a re-insertion could use without changing the claim.

    NFKC (fullwidth and compatibility forms), drop format characters such as
    zero-width spaces and soft hyphens, casefold, treat punctuation and
    underscores as spaces, collapse whitespace. Nothing semantic: "one" and
    "1" stay different, as do homoglyphs from other scripts.
    """
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = t.casefold()
    t = _NON_WORD.sub(" ", t)
    return " ".join(t.split())


def content_hash(text: str) -> str:
    """SHA-256 of :func:`normalize_text`. The key content tombstones match on."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def require_text(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} is required and must not be blank")
    return value.strip()


class Revocation(BaseModel):
    """A tombstone. Carries who revoked what and why; never the fact text."""

    id: UUID = Field(default_factory=uuid4)
    org_id: str
    user_id: str
    fact_id: UUID
    reason: str
    actor: str
    revoked_at: datetime
    replaced_by: UUID | None = None
    reinstated_at: datetime | None = None
    reinstated_by: str | None = None
    reinstate_reason: str | None = None


HiddenReason = Literal["revoked", "content_tombstone", "derived_from_revoked"]


class Provenance(BaseModel):
    """Why a memory is in front of you: where it came from and its status when checked."""

    fact_id: UUID
    writer: str | None = None
    source: str
    recorded_at: datetime | None = None
    session_id: str | None = None
    source_message_id: UUID | None = None
    source_event_id: str | None = None
    derived_from: list[UUID] = Field(default_factory=list)
    supersedes: UUID | None = None
    superseded_by: UUID | None = None
    revocation_status: Literal["active", "revoked"]
    hidden_reason: HiddenReason | None = None
    hidden_by: list[UUID] = Field(default_factory=list)
    checked_at: datetime


class ErasureRecord(BaseModel):
    """What an erasure removed. Ids and counts only; the content is gone."""

    id: UUID = Field(default_factory=uuid4)
    org_id: str
    user_id: str
    fact_ids: list[UUID]
    actor: str
    reason: str
    erased_at: datetime
    artifacts: dict[str, int] = Field(default_factory=dict)
    guarantees: dict[str, Any] = Field(default_factory=dict)
