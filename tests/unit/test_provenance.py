"""Provenance on every returned memory."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import aiosqlite

from engram import Engram
from engram.scope import Scope


async def test_recall_results_carry_provenance() -> None:
    async with await Engram.open(":memory:") as memory:
        src = await memory.record("source fact about tea", user_id="u", actor="importer")
        f = await memory.record(
            "user prefers green tea",
            user_id="u",
            session_id="s9",
            actor="agent:planner",
            derived_from=[src.id],
        )
        before = datetime.now().astimezone()
        results = await memory.recall("green tea", user_id="u", top_k=5)
        hit = next(r for r in results if r.fact.id == f.id)
        p = hit.provenance
        assert p is not None
        assert p.fact_id == f.id
        assert p.writer == "agent:planner"
        assert p.source == "record"
        assert p.session_id == "s9"
        assert p.derived_from == [src.id]
        assert p.recorded_at is not None
        assert p.revocation_status == "active"
        assert p.checked_at >= before
        assert all(r.provenance is not None for r in results)


async def test_context_entries_carry_provenance() -> None:
    async with await Engram.open(":memory:") as memory:
        await memory.record("user prefers green tea", user_id="u", actor="w")
        entries = await memory.context_entries("green tea", user_id="u")
        assert entries
        assert all(e.provenance is not None and e.provenance.writer == "w" for e in entries)


async def test_explain_reports_revoked_status_and_why() -> None:
    async with await Engram.open(":memory:") as memory:
        f = await memory.record("x is one", user_id="u")
        g = await memory.record("x is two", user_id="u")
        rev = await memory.supersede(f.id, g.id, user_id="u", actor="a", reason="r")
        p = await memory.explain(f.id, user_id="u")
        assert p is not None
        assert p.revocation_status == "revoked"
        assert p.hidden_by == [rev.id]
        assert p.superseded_by == g.id
        assert await memory.explain(f.id, user_id="someone-else") is None


async def test_http_recall_with_provenance_and_legacy_shape_unchanged() -> None:
    from fastapi.testclient import TestClient

    from engram.server.http import build_http_app

    async with await Engram.open(":memory:") as memory:
        client = TestClient(build_http_app(memory))
        client.post(
            "/v1/memory/record",
            json={"text": "alice prefers espresso", "user_id": "alice", "actor": "crm-sync"},
        )
        legacy = client.post("/v1/memory/search", json={"query": "espresso", "user_id": "alice"})
        assert set(legacy.json()["hits"][0]) == {
            "fact_id",
            "text",
            "score",
            "vector_score",
            "keyword_score",
            "rerank_score",
            "category",
            "polarity",
            "event_date",
        }
        r = client.post(
            "/v1/memory/recall_with_provenance", json={"query": "espresso", "user_id": "alice"}
        )
        assert r.status_code == 200
        prov = r.json()["hits"][0]["provenance"]
        assert prov["writer"] == "crm-sync"
        assert prov["revocation_status"] == "active"
        assert prov["checked_at"]


async def test_mcp_recall_can_include_provenance() -> None:
    from mcp.types import CallToolRequest, CallToolRequestParams

    from engram.server.mcp import build_mcp_server

    async with await Engram.open(":memory:") as memory:
        f = await memory.record("alice prefers espresso", user_id="alice", actor="w1")
        handler = build_mcp_server(memory).request_handlers[CallToolRequest]

        async def call(args: dict[str, object]) -> str:
            req = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="memory_recall", arguments=args),
            )
            return str((await handler(req)).root.content[0].text)

        plain = await call({"query": "espresso", "user_id": "alice"})
        assert str(f.id) not in plain
        rich = await call({"query": "espresso", "user_id": "alice", "include_provenance": True})
        assert str(f.id) in rich
        assert "writer=w1" in rich
        assert "status=active" in rich


async def test_pre_0_3_databases_are_backfilled_on_open(tmp_path: Path) -> None:
    """Rows written before fact_meta existed still get content tombstones applied."""
    db = tmp_path / "old.db"
    async with await Engram.open(db) as memory:
        f = await memory.record("legacy claim", user_id="u")
    # Simulate a pre-0.3 database: drop the governance side tables.
    async with aiosqlite.connect(str(db)) as conn:
        await conn.execute("DELETE FROM fact_meta")
        await conn.commit()
    async with await Engram.open(db) as memory:
        p = await memory.explain(f.id, user_id="u")
        assert p is not None
        assert p.source == "legacy"
        assert await memory._store.content_hash_of(f.id, Scope(user_id="u")) is not None
        dup = await memory.record("LEGACY claim!", user_id="u")
        await memory.revoke(dup, reason="r", actor="a", user_id="u")
        assert await memory._store.get_fact(f.id, Scope(user_id="u")) is None
