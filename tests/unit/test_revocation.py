"""Revocation enforcement.

Reproduces the failure mode reported in "Revoked but Still Authoritative"
(Shen, Toyoda, Leung; arXiv 2609.08258): a store keeps a revoked policy next
to its replacement, the revoked record is still returned at retrieval, ranks
above the replacement, and the agent acts on it.

The agent here is a deterministic stand-in for the paper's LLM agents: it
follows the highest-ranked policy memory it is shown. That is enough to turn
"the revoked fact ranked first" into "the unsafe action was taken".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from engram import Engram, NotFoundError, RetrievalConfig
from engram.governance import RevokedContentError, normalize_text
from engram.models import Event
from engram.retrieve.base import ScoredFact
from engram.scope import Scope
from engram.tools.search_events import SearchEventsTool
from engram.tools.search_facts import SearchFactsTool

OLD = "Production deploys need approval from one reviewer."
NEW = "Production deploys need approval from two reviewers."
QUERY = "do production deploys need approval from one reviewer"

UNSAFE = "deploy_after_one_approval"
SAFE = "wait_for_second_approval"


def _fullwidth(ascii_text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) for c in ascii_text)


def decide_deploy(texts: list[str]) -> str:
    """Follow the top-ranked policy memory, as the paper's agents did."""
    for t in texts:
        low = t.lower()
        if "one reviewer" in low:
            return UNSAFE
        if "two reviewers" in low:
            return SAFE
    return "ask_a_human"


def _texts(results: list[ScoredFact]) -> list[str]:
    return [r.fact.text for r in results]


async def _seed(memory: Engram, user: str = "ops") -> tuple[object, object]:
    old = await memory.record(OLD, user_id=user, session_id="s-2026-03")
    new = await memory.record(NEW, user_id=user, session_id="s-2026-09")
    return old, new


# ── the paper's failure mode ────────────────────────────────────────────


async def test_scenario_control_old_policy_outranks_replacement_before_revocation() -> None:
    """Control: without revocation the old policy ranks first and the agent acts on it.

    If this stops holding, the scenario no longer reproduces the paper and the
    tests below would pass for the wrong reason.
    """
    async with await Engram.open(":memory:") as memory:
        old, _new = await _seed(memory)
        results = await memory.recall(QUERY, user_id="ops", top_k=5)
        assert results[0].fact.id == old.id  # type: ignore[attr-defined]
        assert decide_deploy(_texts(results)) == UNSAFE


async def test_revoked_policy_is_not_returned_and_does_not_drive_the_action() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _new = await _seed(memory)
        await memory.revoke(old, reason="policy changed 2026-09", actor="sec-team", user_id="ops")

        results = await memory.recall(QUERY, user_id="ops", top_k=5)
        assert OLD not in _texts(results)
        assert NEW in _texts(results)
        assert decide_deploy(_texts(results)) == SAFE


async def test_supersede_then_write_back_of_old_text_does_not_resurrect_it() -> None:
    """The paper's write-back path: the old text comes back as a plain write."""
    async with await Engram.open(":memory:") as memory:
        old, new = await _seed(memory)
        await memory.supersede(old.id, new.id, user_id="ops")  # type: ignore[attr-defined]
        # Re-ingest of an old transcript, or an agent recording what it "knows".
        await memory.record(OLD, user_id="ops")

        results = await memory.recall(QUERY, user_id="ops", top_k=5)
        assert OLD not in _texts(results)
        assert decide_deploy(_texts(results)) == SAFE


async def test_supersede_is_enforced_even_when_history_retrieval_is_on() -> None:
    cfg = RetrievalConfig(exclude_superseded=False)
    async with await Engram.open(":memory:", retrieval_config=cfg) as memory:
        old, new = await _seed(memory)
        await memory.supersede(old.id, new.id, user_id="ops")  # type: ignore[attr-defined]
        results = await memory.recall(QUERY, user_id="ops", top_k=5)
        assert OLD not in _texts(results)
        assert decide_deploy(_texts(results)) == SAFE


# ── every read path ─────────────────────────────────────────────────────


async def test_revoked_fact_is_withheld_from_every_facade_and_store_read_path() -> None:
    scope = Scope(org_id="default", user_id="ops")
    async with await Engram.open(":memory:") as memory:
        old, _new = await _seed(memory)
        await memory.revoke(old.id, reason="r", actor="a", user_id="ops")  # type: ignore[attr-defined]
        store = memory._store

        assert OLD not in await memory.context(QUERY, user_id="ops", token_budget=4000)
        assert OLD not in _texts(await memory.context_entries(QUERY, user_id="ops"))
        assert OLD not in await memory.context(
            "production deploys and reviewers and approvals for every service team",
            user_id="ops",
            decompose=True,
        )
        assert await store.get_fact(old.id, scope) is None  # type: ignore[attr-defined]
        assert OLD not in [f.text for f in await store.keyword_search("reviewer", scope)]
        assert await store.list_facts_by_session("s-2026-03", scope) == []
        sessions = [s for s, _ in await store.aggregate_sessions("reviewer", scope)]
        assert "s-2026-03" not in sessions
        assert "s-2026-09" in sessions

        tool_out = await SearchFactsTool(memory, scope)(QUERY, top_k=5)
        assert "one reviewer" not in tool_out.content
        assert str(old.id) not in tool_out.raw  # type: ignore[attr-defined]


async def test_revoked_fact_is_withheld_from_mcp_tools() -> None:
    from mcp.types import CallToolRequest, CallToolRequestParams

    from engram.server.mcp import build_mcp_server

    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old.id, reason="r", actor="a", user_id="ops")  # type: ignore[attr-defined]
        server = build_mcp_server(memory)
        handler = server.request_handlers[CallToolRequest]
        for tool in ("memory_recall", "memory_context"):
            req = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=tool, arguments={"query": QUERY, "user_id": "ops"}
                ),
            )
            out = (await handler(req)).root.content[0].text
            assert "one reviewer" not in out, tool
            assert "two reviewers" in out, tool


async def test_revoked_fact_is_withheld_from_http_routes() -> None:
    from fastapi.testclient import TestClient

    from engram.server.http import build_http_app

    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old.id, reason="r", actor="a", user_id="ops")  # type: ignore[attr-defined]
        client = TestClient(build_http_app(memory))
        params = {"query": QUERY, "user_id": "ops"}
        bodies = [
            client.post("/v1/memory/search", json=params).text,
            client.get("/v1/memory/recall", params=params).text,
            client.get("/v1/memory/raw_facts", params=params).text,
            client.get("/v1/memory/context", params=params).text,
            client.post("/v1/memory/recall_with_provenance", json=params).text,
        ]
        for body in bodies:
            assert "one reviewer" not in body
            assert "two reviewers" in body


async def test_stale_vector_index_in_another_instance_cannot_return_revoked_fact(
    tmp_path: Path,
) -> None:
    """Instance A still holds the vector in its in-memory index; B revokes on disk."""
    db = tmp_path / "shared.db"
    a = await Engram.open(db)
    b = await Engram.open(db)
    try:
        old, _ = await _seed(a)
        await b.revoke(old.id, reason="r", actor="a", user_id="ops")  # type: ignore[attr-defined]
        results = await a.recall(QUERY, user_id="ops", top_k=5)
        assert OLD not in _texts(results)
    finally:
        await a.close()
        await b.close()


async def test_revocation_landing_mid_rerank_is_still_enforced() -> None:
    """A slow reranker gives a window between candidate fetch and return."""

    class RevokingReranker:
        def __init__(self) -> None:
            self.memory: Engram | None = None
            self.target: object = None

        async def rerank(
            self, query: str, scored: list[ScoredFact], top_k: int = 10
        ) -> list[ScoredFact]:
            assert self.memory is not None
            await self.memory.revoke(self.target, reason="r", actor="a", user_id="ops")
            return scored[:top_k]

    rr = RevokingReranker()
    async with await Engram.open(":memory:", reranker=rr) as memory:
        rr.memory = memory
        old, _ = await _seed(memory)
        rr.target = old
        results = await memory.recall(QUERY, user_id="ops", top_k=5)
        assert OLD not in _texts(results)


async def test_events_linked_to_a_revoked_fact_are_withheld() -> None:
    scope = Scope(org_id="default", user_id="ops")
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        linked = Event(
            scope=scope,
            subject_canonical="production deploy",
            verb="requires",
            object_canonical="one reviewer approval",
            time_start=datetime(2026, 3, 1, tzinfo=UTC),
            source_fact_ids=[old.id],  # type: ignore[attr-defined]
        )
        await memory._store.upsert_event(linked)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")

        assert await memory.search_events("reviewer", user_id="ops") == []
        assert await memory.search_events("", user_id="ops") == []
        assert await memory._store.get_event(linked.id, scope) is None
        out = await SearchEventsTool(memory, scope)("reviewer")
        assert "one reviewer" not in out.content


async def test_transcript_of_a_session_holding_a_revoked_fact_is_withheld() -> None:
    """Query-time re-extraction re-reads raw turns; they must not carry the revoked claim."""
    scope = Scope(org_id="default", user_id="ops")
    async with await Engram.open(":memory:") as memory:
        await memory.record_message(OLD, session_id="s-2026-03", user_id="ops")
        await memory.record_message("ok, noted", session_id="s-2026-03", user_id="ops")
        await memory.record_message(NEW, session_id="s-2026-09", user_id="ops")
        old, _ = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")

        assert await memory._store.list_messages("s-2026-03", scope) == []
        kept = await memory._store.list_messages("s-2026-09", scope)
        assert [m.content for m in kept] == [NEW]


async def test_transcript_turn_linked_by_source_message_is_withheld_but_siblings_kept() -> None:
    scope = Scope(org_id="default", user_id="ops")
    async with await Engram.open(":memory:") as memory:
        turn = await memory.record_message(OLD, session_id="s1", user_id="ops")
        await memory.record_message("unrelated turn", session_id="s1", user_id="ops")
        fact = await memory.record(OLD, user_id="ops", session_id="s1", source_message_id=turn.id)
        await memory.revoke(fact, reason="r", actor="a", user_id="ops")
        kept = await memory._store.list_messages("s1", scope)
        assert [m.content for m in kept] == ["unrelated turn"]


# ── write path: a write cannot un-revoke ───────────────────────────────


async def test_reupserting_the_revoked_fact_object_does_not_unrevoke_it() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        old.superseded_by = None  # type: ignore[attr-defined]
        old.invalid_at = None  # type: ignore[attr-defined]
        old.metadata = {"revoked": False}  # type: ignore[attr-defined]
        await memory._store.upsert_fact(old)  # type: ignore[arg-type]
        assert OLD not in _texts(await memory.recall(QUERY, user_id="ops"))


async def test_rewriting_the_revoked_fact_to_new_text_does_not_unrevoke_it() -> None:
    """The tombstone names the id as well as the content."""
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        old.text = "Production deploys can skip review on Fridays."  # type: ignore[attr-defined]
        await memory._store.upsert_fact(old)  # type: ignore[arg-type]
        got = await memory.recall("production deploys skip review fridays", user_id="ops")
        assert old.id not in {r.fact.id for r in got}  # type: ignore[attr-defined]


async def test_rewriting_an_active_fact_into_revoked_text_is_withheld() -> None:
    async with await Engram.open(":memory:") as memory:
        old, new = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        new.text = OLD  # type: ignore[attr-defined]
        await memory._store.upsert_fact(new)  # type: ignore[arg-type]
        assert OLD not in _texts(await memory.recall(QUERY, user_id="ops"))


@pytest.mark.parametrize(
    "variant",
    [
        OLD,
        OLD.upper(),
        "  production   deploys need approval\tfrom one reviewer  ",
        "Production deploys need approval from one reviewer!!!",
        "Production-deploys need approval from one_reviewer",
        _fullwidth("Production") + " deploys need approval from one reviewer.",
        "Produc​tion deploys need approval from one rev­iewer.",  # zero-width, soft hyphen
    ],
)
async def test_reinserting_revoked_content_in_a_trivial_variant_is_withheld(variant: str) -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        again = await memory.record(variant, user_id="ops")
        assert again.id not in {r.fact.id for r in await memory.recall(QUERY, user_id="ops")}
        status = await memory.explain(again.id, user_id="ops")
        assert status is not None
        assert status.revocation_status == "revoked"


async def test_record_strict_mode_rejects_revoked_content() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        with pytest.raises(RevokedContentError):
            await memory.record(OLD.lower(), user_id="ops", reject_revoked=True)


async def test_paraphrase_of_revoked_content_is_not_matched_by_design() -> None:
    """Documented limit: tombstones match normalized text, not meaning.

    A reworded claim needs its own revocation, a supersede link, or a
    derived_from link. This test pins the rule so a change to it is deliberate.
    """
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        para = await memory.record("A single reviewer can approve production deploys.", "ops")
        explained = await memory.explain(para.id, user_id="ops")
        assert explained is not None
        assert explained.revocation_status == "active"


async def test_tombstone_does_not_leak_across_scopes() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory, user="ops")
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        other = await memory.record(OLD, user_id="someone-else")
        got = await memory.recall(QUERY, user_id="someone-else")
        assert other.id in {r.fact.id for r in got}


async def test_fact_derived_from_a_revoked_fact_is_withheld_whichever_comes_first() -> None:
    """Write-back: the agent records a decision it reached from the old policy."""
    decision = "Decided: shipped release 4.2 after a single approval."
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        before = await memory.record(decision, user_id="ops", derived_from=[old.id])  # type: ignore[attr-defined]
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        after = await memory.record(decision + " again", user_id="ops", derived_from=[before.id])
        ids = {r.fact.id for r in await memory.recall("shipped release single approval", "ops")}
        assert before.id not in ids
        assert after.id not in ids


async def test_derived_from_must_name_facts_in_the_same_scope() -> None:
    async with await Engram.open(":memory:") as memory:
        theirs = await memory.record("their fact", user_id="mallory")
        with pytest.raises(NotFoundError):
            await memory.record("mine", user_id="ops", derived_from=[theirs.id])
        with pytest.raises(NotFoundError):
            await memory.record("mine", user_id="ops", derived_from=[uuid4()])


# ── revoke / supersede / reinstate API rules ───────────────────────────


async def test_revoke_records_a_tombstone_with_actor_reason_and_time() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _new = await _seed(memory)
        rev = await memory.revoke(old, reason="policy v2", actor="sec-team", user_id="ops")
        assert rev.fact_id == old.id  # type: ignore[attr-defined]
        assert rev.actor == "sec-team"
        assert rev.reason == "policy v2"
        assert rev.revoked_at is not None
        assert rev.replaced_by is None
        listed = await memory.revocations(user_id="ops")
        assert [r.id for r in listed] == [rev.id]


async def test_revoke_is_idempotent() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        a = await memory.revoke(old, reason="r", actor="a", user_id="ops")
        b = await memory.revoke(old, reason="r", actor="a", user_id="ops")
        assert a.id == b.id
        assert len(await memory.revocations(user_id="ops")) == 1


@pytest.mark.parametrize(("reason", "actor"), [("", "a"), ("r", ""), ("  ", "a"), ("r", " ")])
async def test_revoke_requires_reason_and_actor(reason: str, actor: str) -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        with pytest.raises(ValueError):
            await memory.revoke(old, reason=reason, actor=actor, user_id="ops")


async def test_revoke_cannot_reach_another_scope() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory, user="ops")
        with pytest.raises(NotFoundError):
            await memory.revoke(old.id, reason="r", actor="mallory", user_id="mallory")  # type: ignore[attr-defined]
        assert OLD in _texts(await memory.recall(QUERY, user_id="ops"))


async def test_supersede_records_replacement_and_rejects_bad_replacements() -> None:
    async with await Engram.open(":memory:") as memory:
        old, new = await _seed(memory)
        rev = await memory.supersede(old.id, new.id, user_id="ops", actor="sec", reason="v2")  # type: ignore[attr-defined]
        assert rev.replaced_by == new.id  # type: ignore[attr-defined]
        # The revoked fact cannot be promoted back as a replacement.
        with pytest.raises(ValueError, match="revoked"):
            await memory.supersede(new.id, old.id, user_id="ops")  # type: ignore[attr-defined]
        # A replacement that normalizes to the same text would be hidden by its own tombstone.
        same = await memory.record("Two reviewers approve production deploys.", user_id="ops")
        dup = await memory.record(" two REVIEWERS approve production deploys ", user_id="ops")
        with pytest.raises(ValueError):
            await memory.supersede(same.id, dup.id, user_id="ops")


async def test_reinstate_lifts_the_tombstone_and_keeps_the_audit_trail() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        rev = await memory.revoke(old, reason="r", actor="a", user_id="ops")
        with pytest.raises(ValueError):
            await memory.reinstate(old, reason="", actor="b", user_id="ops")
        await memory.reinstate(old, reason="revocation was a mistake", actor="b", user_id="ops")
        assert OLD in _texts(await memory.recall(QUERY, user_id="ops"))
        [listed] = await memory.revocations(user_id="ops", include_reinstated=True)
        assert listed.id == rev.id
        assert listed.reinstated_by == "b"
        assert listed.reinstated_at is not None
        assert await memory.revocations(user_id="ops") == []


async def test_reinstate_refuses_a_fact_hidden_only_by_derivation() -> None:
    async with await Engram.open(":memory:") as memory:
        old, _ = await _seed(memory)
        child = await memory.record("child", user_id="ops", derived_from=[old.id])  # type: ignore[attr-defined]
        await memory.revoke(old, reason="r", actor="a", user_id="ops")
        with pytest.raises(ValueError):
            await memory.reinstate(child, reason="r", actor="b", user_id="ops")


def test_normalize_text_folds_trivial_variants_only() -> None:
    assert normalize_text("Hello,   WORLD!") == "hello world"
    assert normalize_text(_fullwidth("hello")) == "hello"
    assert normalize_text("he​llo") == "hello"
    assert normalize_text("snake_case") == "snake case"
    assert normalize_text("one reviewer") != normalize_text("1 reviewer")
