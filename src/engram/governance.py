"""Revocation, erasure and provenance records.

Revocation is a tombstone. A revoked fact stays in the database (so the
history and the audit trail survive) but no read path returns it. Visibility is
decided in one place: the ``hidden_facts`` view in ``store/schema.sql``. Every
store read selects from the ``live_*`` views built on it, so vector search,
keyword search, session ranking, event search, transcript reads and the
HTTP/MCP/tool layers above them all inherit the same rule.

A tombstone hides:

* the revoked fact itself (by id);
* any fact, transcript turn or event in the same scope whose text equals the
  revoked text after :func:`normalize_text`, or quotes it (see :func:`quotes`),
  whether written before or after the revocation;
* any fact recorded with ``derived_from`` pointing at a hidden fact,
  transitively.

Rule for re-insertion: a plain write cannot un-revoke. Writing the same claim
again (exact text, a trivial variant, or a quote) produces an item that is
born hidden. Making the claim visible again takes :meth:`Engram.reinstate` on
the revoked fact itself, with an actor and a reason, recorded on the
tombstone. Paraphrases are not matched; a reworded claim needs its own
revocation or a ``derived_from`` link.

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

# Unicode Default_Ignorable_Code_Point ranges (DerivedCoreProperties.txt),
# plus the braille blank, which renders as nothing in most fonts. Characters
# here are removed outright so they cannot split a word and change the hash.
_INVISIBLE: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x2800, 0x2800),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
# Punctuation that changes what a claim says; kept as its own token.
_MEANINGFUL_PUNCT = frozenset("%‰‱#@&")
_NEGATIVE_NUMBER = re.compile(r"(?<![\w.])-(?=\d)")

# Containment (one text quoting another) only counts for revoked texts of at
# least this many tokens; shorter ones match only on equality. Keeps a revoked
# or erased "Paris" from swallowing every sentence that mentions Paris.
MIN_CONTAINMENT_TOKENS = 4


def _invisible(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _INVISIBLE)


class RevokedContentError(EngramError):
    """A write matched revoked content and the caller asked for a hard refusal."""


def normalize_text(text: str) -> str:
    """Fold the variants a re-insertion could use without changing the claim.

    NFKC (fullwidth and compatibility forms); remove format and other
    default-ignorable characters (zero-width space, soft hyphen, variation
    selectors, fillers); casefold; treat punctuation, whitespace and
    underscores as separators. Symbols (``<``, ``>``, ``+``, currency, emoji),
    a minus sign before a number, and ``% # @ &`` are kept as tokens, and
    combining marks stay attached to their letters, so claims that differ in
    those do not collide. Nothing semantic: "one" and "1" stay different, as do
    homoglyphs from other scripts.
    """
    t = unicodedata.normalize("NFKC", text)
    t = _NEGATIVE_NUMBER.sub("\u2212", t)
    out: list[str] = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat == "Cf" or _invisible(ord(ch)):
            continue
        if ch in _MEANINGFUL_PUNCT or cat[0] == "S":
            out.append(f" {ch} ")
        elif cat[0] in "PZC":  # punctuation (incl. "_"), separators, controls
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).casefold().split())


def content_hash(text: str) -> str:
    """SHA-256 of :func:`normalize_text`. The key content tombstones match on."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def quotes(item_norm: str, revoked_norm: str) -> bool:
    """Does a (normalized) text carry a (normalized) revoked or erased text?

    Equality always counts. Whole-token containment counts when the revoked
    text has at least :data:`MIN_CONTAINMENT_TOKENS` tokens. Empty never matches.
    """
    if not revoked_norm or not item_norm:
        return False
    if item_norm == revoked_norm:
        return True
    if len(revoked_norm.split()) < MIN_CONTAINMENT_TOKENS:
        return False
    return f" {revoked_norm} " in f" {item_norm} "


def quotes_any_run(item_norm: str, source_norm: str, run: int = 3) -> bool:
    """Does ``item_norm`` contain any ``run``-token window of ``source_norm``?

    Used to keep audit fields (erasure reason, actor) free of the erased text.
    """
    if not source_norm or not item_norm:
        return False
    toks = source_norm.split()
    if len(toks) <= run:
        return f" {source_norm} " in f" {item_norm} "
    hay = f" {item_norm} "
    return any(f" {' '.join(toks[i : i + run])} " in hay for i in range(len(toks) - run + 1))


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
