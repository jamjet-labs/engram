"""Bypasses found by the independent adversarial review of the revocation PR.

Each test reproduces one reported path on the first revocation commit and
pins the fix.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from engram import Engram, NotFoundError
from engram.errors import ScopeError
from engram.governance import RevokedContentError, normalize_text
from engram.llm.base import LLMMessage, LLMResponse
from engram.models import ChatMessage, Event, Fact
from engram.read.reextract import QueryConditionedReextractor
from engram.scope import Scope

OLD = "Production deploys need approval from one reviewer."
NEW = "Production deploys need approval from two reviewers."
QUERY = "do production deploys need approval from one reviewer"
OPS = Scope(org_id="default", user_id="ops")


class StubLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(content=self.content)


# ── transcripts and re-extraction (review finding 1) ───────────────────


async def test_unlinked_transcript_turn_quoting_revoked_text_is_withheld() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record_message(OLD, session_id="s1", user_id="ops")
        await memory.record_message("As noted: " + OLD.lower(), session_id="s2", user_id="ops")
        await memory.record_message("unrelated", session_id="s2", user_id="ops")
        old = await memory.record(OLD, user_id="ops")  # no session, no message link
        await memory.revoke(old, reason="r", actor="a")
        assert await memory._store.list_messages("s1", OPS) == []
        assert [m.content for m in await memory._store.list_messages("s2", OPS)] == ["unrelated"]


async def test_turn_written_after_revocation_is_withheld() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        await memory.record_message(OLD, session_id="s9", user_id="ops")
        assert await memory._store.list_messages("s9", OPS) == []


async def test_reextraction_drops_facts_that_quote_revoked_text() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record_message("we discussed approvals", session_id="s1", user_id="ops")
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        payload = {"facts": [{"text": "As of March: " + OLD}, {"text": "Ops owns deploys."}]}
        rx = QueryConditionedReextractor(StubLLM(json.dumps(payload)))  # type: ignore[arg-type]
        got = await rx.reextract("approvals?", ["s1"], memory._store, OPS)
        assert [f.text for f in got] == ["Ops owns deploys."]


async def test_extract_does_not_hand_back_revoked_text() -> None:
    llm = StubLLM(json.dumps({"facts": [{"text": OLD, "confidence": 0.9}]}))
    async with await Engram.open(":memory:", llm=llm) as memory:  # type: ignore[arg-type]
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        msg = ChatMessage(
            scope=OPS, session_id="s1", role="user", content="hi", timestamp=datetime.now(UTC)
        )
        assert await memory.extract([msg], persist=False) == []
        assert await memory.extract([msg], persist=True) == []


# ── events from the package's own extractor (finding 2) ────────────────


def _unlinked_event() -> Event:
    return Event(
        scope=OPS,
        subject_canonical="production deploys",
        verb="need",
        object_canonical="approval from one reviewer",
        time_start=datetime(2026, 3, 1, tzinfo=UTC),
    )


async def test_unlinked_event_stating_revoked_text_is_withheld() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory._store.upsert_event(_unlinked_event())
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        assert await memory.search_events("reviewer", user_id="ops") == []
        # and an event written after the revocation
        await memory._store.upsert_event(_unlinked_event())
        assert await memory.search_events("", user_id="ops") == []


async def test_erase_removes_unlinked_events_stating_the_text() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory._store.upsert_event(_unlinked_event())
        old = await memory.record(OLD, user_id="ops")
        record = await memory.erase(old, reason="request 7", actor="dpo")
        assert record.artifacts["events"] == 1
        assert await memory._store.list_erasures(OPS)


# ── normalization (findings 3 and 4) ───────────────────────────────────


@pytest.mark.parametrize("ch", ["️", "͏", "ㅤ", "⠀", "᠎", "\U000e0041"])
async def test_invisible_characters_do_not_escape_the_tombstone(ch: str) -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        variant = OLD.replace("reviewer", f"revi{ch}ewer")
        with pytest.raises(RevokedContentError):
            await memory.record(variant, user_id="ops", reject_revoked=True)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Approve transfers when amount > 10000", "Approve transfers when amount < 10000"),
        ("limit is -5", "limit is 5"),
        ("limit is +5", "limit is -5"),
        ("fee is $5", "fee is €5"),
        ("rate is 5%", "rate is 5"),
        ("ไม่", "ไม้"),  # Thai "not" vs "wood"
        ("👍", "🔥🔥"),
    ],
)
def test_normalization_keeps_different_claims_apart(a: str, b: str) -> None:
    assert normalize_text(a) != normalize_text(b)


async def test_supersede_accepts_a_replacement_that_flips_a_comparison() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record("Approve transfers when amount > 10000", user_id="ops")
        new = await memory.record("Approve transfers when amount < 10000", user_id="ops")
        await memory.supersede(old.id, new.id, user_id="ops", actor="a", reason="r")
        p = await memory.explain(new.id, user_id="ops")
        assert p is not None and p.revocation_status == "active"


async def test_texts_that_normalize_to_nothing_get_no_content_tombstone() -> None:
    async with await Engram.open(":memory:") as memory:
        a = await memory.record("!!!", user_id="ops")
        b = await memory.record("...", user_id="ops")
        await memory.revoke(a, reason="r", actor="a")
        p = await memory.explain(b.id, user_id="ops")
        assert p is not None and p.revocation_status == "active"
        record = await memory.erase(a, reason="request", actor="dpo")
        assert record.fact_ids == [a.id]


# ── erase interactions (findings 5, 7, 8) ──────────────────────────────


async def test_erasing_revoked_content_still_blocks_its_reinsertion_by_default() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        await memory.erase(old, reason="request 7", actor="dpo")
        with pytest.raises(RevokedContentError):
            await memory.record(OLD, user_id="ops", reject_revoked=True)


async def test_erase_can_drop_the_reinsertion_hash_when_asked() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        await memory.erase(old, reason="request 7", actor="dpo", block_reinsertion=False)
        again = await memory.record(OLD, user_id="ops", reject_revoked=True)
        p = await memory.explain(again.id, user_id="ops")
        assert p is not None and p.revocation_status == "active"


SECRET = "my passport number is X1234567"


async def test_erase_removes_facts_that_quote_the_erased_text() -> None:
    async with await Engram.open(":memory:") as memory:
        f = await memory.record(SECRET, user_id="ops")
        quoting = await memory.record(f"Note for later: {SECRET} (from onboarding)", user_id="ops")
        record = await memory.erase(f, reason="request 9", actor="dpo")
        assert quoting.id in record.fact_ids
        assert await memory.explain(quoting.id, user_id="ops") is None


async def test_erase_scrubs_other_audit_text_that_quotes_the_content() -> None:
    async with await Engram.open(":memory:") as memory:
        f = await memory.record(SECRET, user_id="ops")
        other = await memory.record("unrelated claim about deploys", user_id="ops")
        await memory.revoke(other, reason=f"done, {SECRET}", actor="a")
        await memory.erase(f, reason="request 9", actor="dpo")
        [rev] = await memory.revocations(user_id="ops")
        assert "x1234567" not in rev.model_dump_json().lower()


@pytest.mark.parametrize(
    ("reason", "actor"),
    [("request about passport number is X1234567", "dpo"), ("request 9", f"dpo {SECRET}")],
)
async def test_erase_rejects_audit_fields_that_quote_part_of_the_content(
    reason: str, actor: str
) -> None:
    async with await Engram.open(":memory:") as memory:
        f = await memory.record(SECRET, user_id="ops")
        with pytest.raises(ValueError):
            await memory.erase(f, reason=reason, actor=actor)


async def test_erasing_a_short_fact_does_not_delete_turns_that_merely_mention_it() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record_message("I love Paris in spring", session_id="s1", user_id="ops")
        f = await memory.record("Paris", user_id="ops")
        await memory.erase(f, reason="request", actor="dpo")
        got = await memory._store.list_messages("s1", OPS)
        assert [m.content for m in got] == ["I love Paris in spring"]


# ── reinstate and supersede (findings 6 and 10) ────────────────────────


async def test_reinstating_a_born_hidden_copy_does_not_unrevoke_the_original() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        copy = await memory.record(OLD, user_id="ops")
        with pytest.raises(ValueError, match="reinstate"):
            await memory.reinstate(copy, reason="this note is fine", actor="b")
        p = await memory.explain(old.id, user_id="ops")
        assert p is not None and p.revocation_status == "revoked"


async def test_supersede_of_an_already_revoked_fact_still_links_the_replacement() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        new = await memory.record(NEW, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        rev = await memory.supersede(old.id, new.id, user_id="ops")
        assert rev.replaced_by == new.id
        p = await memory.explain(new.id, user_id="ops")
        assert p is not None and p.supersedes == old.id
        other = await memory.record("Deploys need three reviewers.", user_id="ops")
        with pytest.raises(ValueError):
            await memory.supersede(old.id, other.id, user_id="ops")


async def test_governance_calls_take_the_scope_from_a_fact_object() -> None:
    async with await Engram.open(":memory:") as memory:
        f = await memory.record(OLD, user_id="alice", org_id="acme")
        await memory.revoke(f, reason="r", actor="a")
        p = await memory.explain(f.id, user_id="alice", org_id="acme")
        assert p is not None and p.revocation_status == "revoked"
        with pytest.raises(ValueError):
            await memory.reinstate(f, reason="r", actor="a", user_id="bob")


# ── store-level writes (finding 9, Copilot) ────────────────────────────


async def test_upsert_cannot_overwrite_a_fact_in_another_scope() -> None:
    async with await Engram.open(":memory:") as memory:
        f = await memory.record("Alice is allergic to peanuts.", user_id="alice")
        forged = Fact(
            id=f.id,
            text="Alice's allergy: none",
            scope=Scope(user_id="mallory"),
            valid_from=datetime.now(UTC),
        )
        with pytest.raises(ScopeError):
            await memory._store.upsert_fact(forged)
        got = await memory._store.get_fact(f.id, Scope(user_id="alice"))
        assert got is not None and got.text == "Alice is allergic to peanuts."


async def test_source_message_must_belong_to_the_same_scope() -> None:
    async with await Engram.open(":memory:") as memory:
        bobs = await memory.record_message("bob's turn", session_id="s", user_id="bob")
        with pytest.raises(NotFoundError):
            await memory.record("x", user_id="alice", source_message_id=bobs.id)
        with pytest.raises(NotFoundError):
            await memory.record("x", user_id="alice", source_message_id=uuid4())


async def test_unauthenticated_writer_cannot_withhold_a_session_transcript() -> None:
    """A born-hidden copy must not trigger the session-wide transcript rule."""
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        await memory.record_message("deploy notes for 4.2", session_id="s5", user_id="ops")
        await memory.record(OLD.upper(), user_id="ops", session_id="s5")  # attacker write
        got = await memory._store.list_messages("s5", OPS)
        assert [m.content for m in got] == ["deploy notes for 4.2"]


async def test_hidden_copies_do_not_occupy_vector_slots() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        copies = [await memory.record(OLD.upper(), user_id="ops") for _ in range(3)]
        await memory.revoke(old, reason="r", actor="a")
        for c in copies:
            assert c.id in memory._vec._scopes[OPS].deleted
        late = await memory.record(OLD.lower(), user_id="ops")
        assert late.id not in memory._vec._scopes[OPS].uuid_to_int


async def test_decompose_path_withholds_revoked_facts() -> None:
    class Splitter:
        async def decompose(self, q: str) -> list[str]:
            return ["production deploys reviewer", "approval from one reviewer"]

    async with await Engram.open(":memory:") as memory:
        memory._decomposer = Splitter()  # type: ignore[assignment]
        old = await memory.record(OLD, user_id="ops")
        await memory.record(NEW, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        q = "what do production deploys need and who approves them and when do reviewers sign"
        entries = await memory.context_entries(q, user_id="ops", decompose=True)
        assert entries
        assert OLD not in [e.fact.text for e in entries]


async def test_facts_quoting_revoked_text_are_withheld_before_and_after() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        before = await memory.record("Reminder from March: " + OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        after = await memory.record("Per the runbook, " + OLD.lower(), user_id="ops")
        for f in (before, after):
            p = await memory.explain(f.id, user_id="ops")
            assert p is not None and p.revocation_status == "revoked"
            assert p.hidden_reason == "content_tombstone"


def test_symbol_spacing_is_a_trivial_variant() -> None:
    assert normalize_text("fee is $5") == normalize_text("fee is $ 5")
    assert normalize_text("a>b") == normalize_text("a > b")


def test_empty_texts_never_match() -> None:
    from engram.governance import quotes, quotes_any_run

    assert not quotes("", "")
    assert not quotes("anything", "")
    assert not quotes_any_run("", "")


async def test_hash_kept_after_erasure_withholds_a_later_copy_at_read_time() -> None:
    async with await Engram.open(":memory:") as memory:
        old = await memory.record(OLD, user_id="ops")
        await memory.revoke(old, reason="r", actor="a")
        await memory.erase(old, reason="request 7", actor="dpo")
        again = await memory.record(OLD, user_id="ops")
        p = await memory.explain(again.id, user_id="ops")
        assert p is not None and p.revocation_status == "revoked"


async def test_linked_turn_is_withheld_even_when_its_text_differs() -> None:
    async with await Engram.open(":memory:") as memory:
        turn = await memory.record_message("yes, that is the rule", session_id="s1", user_id="ops")
        await memory.record_message("thanks", session_id="s1", user_id="ops")
        f = await memory.record(OLD, user_id="ops", session_id="s1", source_message_id=turn.id)
        await memory.revoke(f, reason="r", actor="a")
        got = await memory._store.list_messages("s1", OPS)
        assert [m.content for m in got] == ["thanks"]


def test_other_format_characters_are_dropped() -> None:
    assert normalize_text("revi￹ewer") == "reviewer"
