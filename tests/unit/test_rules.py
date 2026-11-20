"""The erase and revocation rules every store shares, tested without a database.

Expected values are written out by hand. None of them is computed by the
function under test.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engram.models import Event
from engram.scope import Scope
from engram.store._rules import (
    FactRow,
    RevocationScrub,
    erase_closure,
    erased_norms_and_hashes,
    event_text,
    events_to_erase,
    messages_to_erase,
    quote_rows,
    refuse_quoting_audit_fields,
    revocations_to_scrub,
)

MOVED = "alice moved to berlin in march"
# sha256 of the UTF-8 bytes of MOVED, worked out once with hashlib.
MOVED_HASH = "4b508fb74ca07ceeea41ec27f3e65a2c649c2ee0c326429597da7373f253898f"


def _facts() -> dict[str, FactRow]:
    return {
        "a": ("Alice moved to Berlin in March.", MOVED, "m1"),
        "copy": ("ALICE MOVED TO BERLIN IN MARCH", MOVED, None),
        "quoting": (
            "Note: Alice moved to Berlin in March and liked it.",
            "note alice moved to berlin in march and liked it",
            None,
        ),
        "contained": ("Alice moved to Berlin.", "alice moved to berlin", None),
        "derived": (
            "Bob helped with the move last spring.",
            "bob helped with the move last spring",
            "m2",
        ),
        "quotes-derived": (
            "Yes, bob helped with the move last spring!",
            "yes bob helped with the move last spring",
            None,
        ),
        "grandchild": ("Bob owns a van.", "bob owns a van", None),
        "keeper": ("Alice prefers espresso.", "alice prefers espresso", None),
    }


def test_closure_follows_derivations_transitively() -> None:
    facts = {
        "a": ("first", "first", None),
        "b": ("second", "second", None),
        "c": ("third", "third", None),
        "d": ("fourth", "fourth", None),
    }
    children = {"a": ["b"], "b": ["c"]}
    assert erase_closure(facts, children, "a") == {"a", "b", "c"}
    # Derivation runs one way: erasing a child leaves its source alone.
    assert erase_closure(facts, children, "b") == {"b", "c"}
    assert erase_closure(facts, children, "d") == {"d"}


def test_closure_follows_quotes_both_ways_only_from_erased_text() -> None:
    facts = _facts()
    # Equal text and text that quotes the erased text go, whichever fact is
    # erased first. A shorter text the erased one contains stays.
    assert erase_closure(facts, {}, "a") == {"a", "copy", "quoting"}
    assert erase_closure(facts, {}, "copy") == {"a", "copy", "quoting"}
    # The quoting fact's own text is quoted by nothing, so erasing it alone
    # takes nothing else, though it contains the other facts' text.
    assert erase_closure(facts, {}, "quoting") == {"quoting"}
    # The other way round: every text quoting the shorter one goes with it.
    assert erase_closure(facts, {}, "contained") == {"contained", "a", "copy", "quoting"}
    # Quotes are followed from every newly erased fact, derived ones included.
    children = {"a": ["derived"], "derived": ["grandchild"]}
    assert erase_closure(facts, children, "a") == {
        "a",
        "copy",
        "quoting",
        "derived",
        "quotes-derived",
        "grandchild",
    }


def test_closure_quotes_short_text_only_when_equal() -> None:
    facts = {
        "short": ("Paris is nice.", "paris is nice", None),
        "same": ("paris IS nice", "paris is nice", None),
        "longer": ("Paris is nice in May.", "paris is nice in may", None),
    }
    assert erase_closure(facts, {}, "short") == {"short", "same"}


def test_closure_ignores_facts_outside_the_given_rows() -> None:
    facts = {
        "a": ("first", "first", None),
        "b": ("second", "second", None),
    }
    # A derivation may name a fact in another scope; it is not in the rows,
    # and nothing is followed through it.
    children = {"a": ["other-scope"], "other-scope": ["b"]}
    assert erase_closure(facts, children, "a") == {"a"}
    assert erase_closure(facts, children, "other-scope") == set()
    assert erase_closure(facts, {}, "missing") == set()


def test_closure_terminates_on_a_derivation_cycle() -> None:
    facts = {
        "a": ("first", "first", None),
        "b": ("second", "second", None),
        "c": ("third", "third", None),
    }
    children = {"a": ["b"], "b": ["a", "c"], "c": ["c"]}
    assert erase_closure(facts, children, "a") == {"a", "b", "c"}
    assert erase_closure(facts, children, "c") == {"c"}


def test_erased_norms_and_hashes_skip_text_that_normalizes_to_nothing() -> None:
    facts = {
        "a": ("Alice moved to Berlin in March.", MOVED, None),
        "blank": ("...", "", None),
        "kept": ("Bob owns a van.", "bob owns a van", None),
    }
    norms, hashes = erased_norms_and_hashes({"a", "blank"}, facts)
    assert norms == {MOVED}
    assert hashes == {MOVED_HASH}


def test_audit_field_quoting_three_tokens_is_refused() -> None:
    norms = [MOVED]
    with pytest.raises(ValueError) as reason_quotes:
        refuse_quoting_audit_fields("user asked: moved to BERLIN", "admin", norms)
    assert str(reason_quotes.value) == "erasure reason must not quote the content being erased"
    with pytest.raises(ValueError) as actor_quotes:
        refuse_quoting_audit_fields("user request", "Alice moved to", norms)
    assert str(actor_quotes.value) == "erasure actor must not quote the content being erased"
    # Two tokens are not a quote.
    refuse_quoting_audit_fields("moved to", "berlin in", norms)
    # A text of three tokens or fewer is refused only when quoted whole.
    with pytest.raises(ValueError) as short_quoted:
        refuse_quoting_audit_fields("says paris is nice", "admin", ["paris is nice"])
    assert str(short_quoted.value) == "erasure reason must not quote the content being erased"
    refuse_quoting_audit_fields("paris is", "admin", ["paris is nice"])
    refuse_quoting_audit_fields("anything at all", "anyone", [])


def test_messages_cited_or_quoting_are_erased() -> None:
    facts = _facts()
    messages = [
        ("m1", "unrelated turn"),
        ("m2", "unrelated turn"),
        ("m3", "She said: ALICE moved to Berlin, in March!"),
        ("m4", "moved to berlin in march"),
        ("m5", "Alice moved to Berlin in March."),
    ]
    # m1 is cited by the erased fact; m2 only by a fact that stays.
    assert messages_to_erase({"a"}, facts, messages, [MOVED]) == {"m1", "m3", "m5"}
    assert messages_to_erase({"derived"}, facts, messages, []) == {"m2"}
    assert messages_to_erase({"keeper"}, facts, messages, ["alice prefers espresso"]) == set()


def test_events_linked_or_quoting_are_erased() -> None:
    events = [
        ("e1", ["a"], "bob likes tea"),
        ("e2", ["x"], "Alice Moved To Berlin In March!"),
        ("e3", ["x"], "alice moved"),
        ("e4", [], "unrelated"),
        ("e5", ["keeper", "copy"], "bob owns a van"),
    ]
    assert events_to_erase({"a", "copy"}, events, [MOVED]) == {"e1", "e2", "e5"}
    assert events_to_erase(set(), events, [MOVED]) == {"e2"}
    assert events_to_erase({"zzz"}, events, []) == set()


def test_event_text_is_subject_verb_object_then_aliases() -> None:
    event = Event(
        scope=Scope(org_id="acme", user_id="alice"),
        subject_canonical="alice",
        verb="moved_to",
        object_canonical="berlin",
        time_start=datetime(2026, 3, 1, tzinfo=UTC),
        aliases=["Alice B.", "the capital"],
    )
    assert event_text(event) == "alice moved_to berlin Alice B. the capital"


def _rev(
    rev_id: str,
    fact_id: str,
    content_hash: str | None,
    reason: str = "wrong",
    reinstate_reason: str | None = None,
) -> dict[str, str | None]:
    return {
        "id": rev_id,
        "fact_id": fact_id,
        "content_hash": content_hash,
        "reason": reason,
        "reinstate_reason": reinstate_reason,
    }


def _revs() -> list[dict[str, str | None]]:
    return [
        # On the erased fact itself.
        _rev("r1", "a", "h-a"),
        # On a copy in scope: same hash, different fact.
        _rev("r2", "z", "h-a"),
        # Unrelated content, but the reason quotes the erased text.
        _rev("r3", "z2", "h-other", reason="said alice moved to berlin"),
        # Unrelated content, but the reinstatement note quotes it.
        _rev("r4", "z3", None, reinstate_reason="back: moved to berlin in"),
        # On the erased fact, whose text normalized to nothing.
        _rev("r5", "a", None),
        # Nothing to do with the erasure.
        _rev("r6", "z4", "h-z4", reason="alice moved", reinstate_reason="to berlin"),
    ]


def test_scrub_keeps_hash_only_with_block_reinsertion() -> None:
    erased = {"a"}
    hashes = {"h-a"}
    norms = [MOVED]
    assert revocations_to_scrub(_revs(), erased, hashes, norms, block_reinsertion=True) == [
        RevocationScrub(revocation_id="r1", scrub_text=True, clear_norm=True, keep_hash=True),
        RevocationScrub(revocation_id="r2", scrub_text=True, clear_norm=True, keep_hash=True),
        RevocationScrub(revocation_id="r3", scrub_text=True, clear_norm=False, keep_hash=False),
        RevocationScrub(revocation_id="r4", scrub_text=True, clear_norm=False, keep_hash=False),
        RevocationScrub(revocation_id="r5", scrub_text=True, clear_norm=True, keep_hash=False),
    ]
    assert revocations_to_scrub(_revs(), erased, hashes, norms, block_reinsertion=False) == [
        RevocationScrub(revocation_id="r1", scrub_text=True, clear_norm=True, keep_hash=False),
        RevocationScrub(revocation_id="r2", scrub_text=True, clear_norm=True, keep_hash=False),
        RevocationScrub(revocation_id="r3", scrub_text=True, clear_norm=False, keep_hash=False),
        RevocationScrub(revocation_id="r4", scrub_text=True, clear_norm=False, keep_hash=False),
        RevocationScrub(revocation_id="r5", scrub_text=True, clear_norm=True, keep_hash=False),
    ]


def test_scrub_drops_the_hash_only_when_it_clears_the_text_and_keeps_none() -> None:
    kept = RevocationScrub(revocation_id="r1", scrub_text=True, clear_norm=True, keep_hash=True)
    cleared = RevocationScrub(revocation_id="r1", scrub_text=True, clear_norm=True, keep_hash=False)
    reason_only = RevocationScrub(
        revocation_id="r3", scrub_text=True, clear_norm=False, keep_hash=False
    )
    assert kept.drop_hash is False
    assert cleared.drop_hash is True
    assert reason_only.drop_hash is False


def test_quote_rows_pairs_each_item_with_each_text_it_quotes() -> None:
    items = [
        ("fact", "f1", "Alice moved to Berlin in March."),
        ("message", "m1", "so ALICE moved to berlin in march, ok"),
        ("event", "e1", "alice moved to berlin"),
        ("fact", "f2", "Paris is nice"),
    ]
    revoked = [("r1", MOVED), ("r2", "paris is nice"), ("r3", "bob owns a van")]
    assert quote_rows(items, revoked) == [
        ("fact", "f1", "r1"),
        ("message", "m1", "r1"),
        ("fact", "f2", "r2"),
    ]
