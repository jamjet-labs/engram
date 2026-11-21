"""The erase, revocation and embedder rules, written once for every store.

Each function here is pure: it takes rows a store has already read and says
what to delete, refuse or scrub. The store does the reading and the writing
in its own transaction. Keeping the rules here means two stores cannot drift
apart on what an erase removes, what a tombstone keeps, or which embedder
key they accept.

Text comparison is :func:`engram.governance.quotes` (equality, or whole-token
containment of a long enough text) and, for audit fields,
:func:`engram.governance.quotes_any_run`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engram.errors import EmbedderMismatchError
from engram.governance import content_hash, normalize_text, quotes, quotes_any_run
from engram.models import Event

FactRow = tuple[str, str, str | None]
"""(text, normalized text, source_message_id) for one fact in scope."""

EMBEDDER_META_KEY = "embedder"
"""The engram_meta key whose value is the recorded embedder's `EmbedderIdentity.key`."""


def check_embedder_key(recorded: str | None, key: str, *, replace: bool) -> bool:
    """Check ``key`` against the one a store records. True when the store must write it.

    No key recorded: True. The same key: False. Another key: True with
    ``replace``, else EmbedderMismatchError naming both keys. Keys compare
    whole, so another model or another vector size is another embedder.
    """
    if recorded is None:
        return True
    if recorded == key:
        return False
    if replace:
        return True
    raise EmbedderMismatchError(
        f"the database records embedder {recorded!r}, not {key!r}. Vectors from two "
        "embedders do not compare, so nothing was written. To change embedders, "
        "open with on_embedder_change='switch', then run engram-admin reembed."
    )


def erase_closure(
    facts: Mapping[str, FactRow], children: Mapping[str, Sequence[str]], start: str
) -> set[str]:
    """Ids of the facts an erase of ``start`` removes.

    ``facts`` is every fact in the scope, by id. ``children`` maps a fact id to
    the ids of facts recorded with ``derived_from`` pointing at it.

    From each erased fact the closure takes its derived facts, and every fact
    whose normalized text equals or quotes the erased text, repeated to a
    fixpoint. The quote is followed one way: a fact that quotes erased text
    goes, a fact whose text the erased text quotes does not. Ids not in
    ``facts`` (another scope's, or gone) are skipped and not followed.
    """
    erased: set[str] = set()
    frontier = [start]
    while frontier:
        fid = frontier.pop()
        if fid in erased or fid not in facts:
            continue
        erased.add(fid)
        norm = facts[fid][1]
        frontier.extend(c for c in children.get(fid, []) if c not in erased)
        if norm:
            frontier.extend(
                other
                for other, (_, other_norm, _) in facts.items()
                if other not in erased and quotes(other_norm, norm)
            )
    return erased


def erased_norms_and_hashes(
    erased: Collection[str], facts: Mapping[str, FactRow]
) -> tuple[set[str], set[str]]:
    """The normalized texts and content hashes of the erased facts.

    A fact whose text normalizes to nothing adds neither: it quotes nothing
    and its tombstone records no hash.
    """
    norms = {facts[f][1] for f in erased} - {""}
    hashes = {content_hash(facts[f][0]) for f in erased if facts[f][1]}
    return norms, hashes


def refuse_quoting_audit_fields(reason: str, actor: str, norms: Collection[str]) -> None:
    """Raise ``ValueError`` if ``reason`` or ``actor`` carries erased text.

    Any three-token run of an erased normalized text counts. The erasure
    record keeps both fields, so they must not keep the content.
    """
    for label, value in (("reason", reason), ("actor", actor)):
        value_norm = normalize_text(value)
        if any(quotes_any_run(value_norm, n) for n in norms):
            raise ValueError(f"erasure {label} must not quote the content being erased")


def messages_to_erase(
    erased: set[str],
    facts: Mapping[str, FactRow],
    messages: Iterable[tuple[str, str]],
    norms: Collection[str],
) -> set[str]:
    """Ids of the transcript turns an erase removes.

    ``messages`` is ``(message id, content)`` for every turn in the scope.
    A turn goes if an erased fact cites it as its source, or its content
    quotes an erased text.
    """
    out: set[str] = set()
    for f in erased:
        source = facts[f][2]
        if source:
            out.add(source)
    for message_id, content in messages:
        content_norm = normalize_text(content)
        if any(quotes(content_norm, n) for n in norms):
            out.add(message_id)
    return out


def events_to_erase(
    erased: set[str], events: Iterable[tuple[str, Sequence[str], str]], norms: Collection[str]
) -> set[str]:
    """Ids of the events an erase removes.

    ``events`` is ``(event id, source fact ids, event text)`` for every event
    in the scope, with the text from :func:`event_text`. An event goes if it
    lists an erased fact as a source, or its text quotes an erased text.
    """
    out: set[str] = set()
    for event_id, sources, text in events:
        linked = any(s in erased for s in sources)
        text_norm = normalize_text(text)
        if linked or any(quotes(text_norm, n) for n in norms):
            out.add(event_id)
    return out


def event_text(event: Event) -> str:
    """The text of an event that tombstones and erasures match against."""
    return " ".join([event.subject_canonical, event.verb, event.object_canonical, *event.aliases])


@dataclass(frozen=True)
class RevocationScrub:
    """What an erase changes on one tombstone.

    ``scrub_text``: replace the reason, and the reinstatement note if there
    is one, with a scrubbed marker. True for every tombstone the erase
    touches. ``clear_norm``: drop the normalized text, because the tombstone
    is on erased content. ``keep_hash``: keep the content hash so the content
    stays blocked after the erase. The hash is dropped when ``drop_hash``.
    """

    revocation_id: str
    scrub_text: bool
    clear_norm: bool
    keep_hash: bool

    @property
    def drop_hash(self) -> bool:
        """True when the tombstone loses its content hash."""
        return self.clear_norm and not self.keep_hash


def revocations_to_scrub(
    revs: Iterable[Mapping[str, Any]],
    erased: set[str],
    hashes: set[str],
    norms: Collection[str],
    *,
    block_reinsertion: bool,
) -> list[RevocationScrub]:
    """The tombstones in scope an erase must change, in the order given.

    ``revs`` are the scope's revocation rows with the keys ``id``,
    ``fact_id``, ``content_hash``, ``reason`` and ``reinstate_reason``.
    A tombstone is on erased content when it names an erased fact or
    carries the hash of one. Its free text is scrubbed when it is on erased
    content or quotes a three-token run of an erased text. Its hash is kept
    only when it is on erased content, has a hash, and ``block_reinsertion``
    is true; the hash of a short text can be confirmed by guessing.
    """
    out: list[RevocationScrub] = []
    for r in revs:
        names_content = r["fact_id"] in erased or (
            r["content_hash"] is not None and r["content_hash"] in hashes
        )
        reason_quotes = any(
            quotes_any_run(normalize_text(r["reason"]), n)
            or quotes_any_run(normalize_text(r["reinstate_reason"] or ""), n)
            for n in norms
        )
        if not (names_content or reason_quotes):
            continue
        keep_hash = names_content and block_reinsertion and r["content_hash"] is not None
        out.append(
            RevocationScrub(
                revocation_id=r["id"],
                scrub_text=True,
                clear_norm=names_content,
                keep_hash=keep_hash,
            )
        )
    return out


def quote_rows(
    items: Sequence[tuple[str, str, str]], revoked: Sequence[tuple[str, str]]
) -> list[tuple[str, str, str]]:
    """(kind, item_id, revocation_id) for every item whose text quotes a revoked text."""
    rows: list[tuple[str, str, str]] = []
    for kind, item_id, text in items:
        norm = normalize_text(text)
        rows.extend(
            (kind, item_id, rev_id) for rev_id, rev_norm in revoked if quotes(norm, rev_norm)
        )
    return rows
