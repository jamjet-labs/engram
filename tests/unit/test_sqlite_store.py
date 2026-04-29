from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from engram.models import ChatMessage, Fact
from engram.scope import Scope
from engram.store.sqlite import SqliteStore


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


# ── Open / close ─────────────────────────────────────────────────────


async def test_open_in_memory_creates_schema() -> None:
    s = await SqliteStore.open(":memory:")
    assert s is not None
    await s.close()


async def test_open_file_path(tmp_path: Path) -> None:
    db = tmp_path / "engram.db"
    s = await SqliteStore.open(str(db))
    await s.close()
    assert db.exists()


async def test_async_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "engram.db"
    async with await SqliteStore.open(str(db)) as s:
        assert s is not None
    assert db.exists()


# ── Facts CRUD ───────────────────────────────────────────────────────


async def test_upsert_then_get_fact(store: SqliteStore, acme_alice: Scope) -> None:
    f = Fact(text="alice prefers espresso", scope=acme_alice, valid_from=_now())
    await store.upsert_fact(f)
    got = await store.get_fact(f.id, acme_alice)
    assert got is not None
    assert got.id == f.id
    assert got.text == "alice prefers espresso"


async def test_get_fact_returns_none_for_missing(store: SqliteStore, acme_alice: Scope) -> None:
    got = await store.get_fact(uuid4(), acme_alice)
    assert got is None


async def test_get_fact_respects_scope(store: SqliteStore, acme_alice: Scope) -> None:
    f = Fact(text="x", scope=acme_alice, valid_from=_now())
    await store.upsert_fact(f)
    other = Scope(org_id="acme", user_id="bob")
    got = await store.get_fact(f.id, other)
    assert got is None


async def test_upsert_fact_is_idempotent(store: SqliteStore, acme_alice: Scope) -> None:
    f = Fact(text="x", scope=acme_alice, valid_from=_now(), confidence=0.5)
    await store.upsert_fact(f)
    f.confidence = 0.9
    await store.upsert_fact(f)
    got = await store.get_fact(f.id, acme_alice)
    assert got is not None
    assert got.confidence == 0.9


# ── Keyword search (FTS5) ────────────────────────────────────────────


async def test_keyword_search_finds_exact_match(store: SqliteStore, acme_alice: Scope) -> None:
    await store.upsert_fact(
        Fact(text="alice prefers espresso", scope=acme_alice, valid_from=_now())
    )
    await store.upsert_fact(
        Fact(text="weather is sunny today", scope=acme_alice, valid_from=_now())
    )
    results = await store.keyword_search("espresso", acme_alice)
    assert len(results) == 1
    assert "espresso" in results[0].text


async def test_keyword_search_ranks_relevance(store: SqliteStore, acme_alice: Scope) -> None:
    await store.upsert_fact(
        Fact(text="alice loves espresso espresso espresso", scope=acme_alice, valid_from=_now())
    )
    await store.upsert_fact(
        Fact(text="alice tried espresso once", scope=acme_alice, valid_from=_now())
    )
    results = await store.keyword_search("espresso", acme_alice)
    assert len(results) == 2
    assert "loves espresso espresso espresso" in results[0].text


async def test_keyword_search_respects_scope(store: SqliteStore, acme_alice: Scope) -> None:
    bob = Scope(org_id="acme", user_id="bob")
    await store.upsert_fact(
        Fact(text="alice prefers espresso", scope=acme_alice, valid_from=_now())
    )
    await store.upsert_fact(Fact(text="bob prefers latte", scope=bob, valid_from=_now()))
    alice_results = await store.keyword_search("prefers", acme_alice)
    assert len(alice_results) == 1
    assert "alice" in alice_results[0].text.lower()


async def test_keyword_search_empty_query_returns_empty(
    store: SqliteStore, acme_alice: Scope
) -> None:
    await store.upsert_fact(Fact(text="x", scope=acme_alice, valid_from=_now()))
    results = await store.keyword_search("", acme_alice)
    assert results == []


# ── Messages ─────────────────────────────────────────────────────────


async def test_upsert_then_list_messages(store: SqliteStore, acme_alice: Scope) -> None:
    m1 = ChatMessage(scope=acme_alice, session_id="s1", role="user", content="hi", timestamp=_now())
    m2 = ChatMessage(
        scope=acme_alice,
        session_id="s1",
        role="assistant",
        content="hello",
        timestamp=_now() + timedelta(seconds=1),
    )
    await store.upsert_message(m1)
    await store.upsert_message(m2)
    msgs = await store.list_messages("s1", acme_alice)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


async def test_list_messages_orders_by_timestamp(store: SqliteStore, acme_alice: Scope) -> None:
    m_late = ChatMessage(
        scope=acme_alice,
        session_id="s1",
        role="user",
        content="late",
        timestamp=_now() + timedelta(seconds=10),
    )
    m_early = ChatMessage(
        scope=acme_alice, session_id="s1", role="user", content="early", timestamp=_now()
    )
    await store.upsert_message(m_late)
    await store.upsert_message(m_early)
    msgs = await store.list_messages("s1", acme_alice)
    assert msgs[0].content == "early"
    assert msgs[1].content == "late"


async def test_list_messages_respects_scope(store: SqliteStore, acme_alice: Scope) -> None:
    bob = Scope(org_id="acme", user_id="bob")
    await store.upsert_message(
        ChatMessage(
            scope=acme_alice, session_id="s1", role="user", content="alice", timestamp=_now()
        )
    )
    await store.upsert_message(
        ChatMessage(scope=bob, session_id="s1", role="user", content="bob", timestamp=_now())
    )
    alice_msgs = await store.list_messages("s1", acme_alice)
    assert len(alice_msgs) == 1
    assert alice_msgs[0].content == "alice"


# ── record_access ────────────────────────────────────────────────────


async def test_record_access_increments_count(store: SqliteStore, acme_alice: Scope) -> None:
    f = Fact(text="x", scope=acme_alice, valid_from=_now())
    await store.upsert_fact(f)
    await store.record_access(f.id)
    await store.record_access(f.id)
    got = await store.get_fact(f.id, acme_alice)
    assert got is not None
    assert got.access_count == 2
