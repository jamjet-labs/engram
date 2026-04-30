from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from engram.models import ChatMessage
from engram.read.reextract import QueryConditionedReextractor
from engram.scope import Scope


@pytest.mark.asyncio
async def test_reextract_calls_llm_per_session_and_returns_facts():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = (
        '{"facts": [{"text": "alice runs marathons", "confidence": 0.9}]}'
    )
    fake_store = AsyncMock()
    scope = Scope(org_id="default", user_id="alice")
    fake_store.list_messages_by_session = AsyncMock(
        return_value=[
            ChatMessage(
                scope=scope,
                session_id="s1",
                role="user",
                content="I ran a marathon",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        ]
    )
    rx = QueryConditionedReextractor(llm=fake_llm)
    facts = await rx.reextract(
        question="Have I run any marathons?",
        candidate_session_ids=["s1"],
        store=fake_store,
        scope=scope,
    )
    assert len(facts) == 1
    assert "marathons" in facts[0].text


@pytest.mark.asyncio
async def test_reextract_returns_empty_on_malformed_json():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = "not json"
    fake_store = AsyncMock()
    fake_store.list_messages_by_session = AsyncMock(
        return_value=[
            ChatMessage(
                scope=Scope(org_id="d", user_id="a"),
                session_id="s1",
                role="user",
                content="hello",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        ]
    )
    rx = QueryConditionedReextractor(llm=fake_llm)
    facts = await rx.reextract(
        question="?",
        candidate_session_ids=["s1"],
        store=fake_store,
        scope=Scope(org_id="d", user_id="a"),
    )
    assert facts == []


@pytest.mark.asyncio
async def test_reextract_caps_at_max_sessions():
    fake_llm = AsyncMock()
    fake_llm.generate.return_value.content = '{"facts": []}'
    fake_store = AsyncMock()
    fake_store.list_messages_by_session = AsyncMock(return_value=[])
    rx = QueryConditionedReextractor(llm=fake_llm, max_sessions=2)
    await rx.reextract(
        question="?",
        candidate_session_ids=["s1", "s2", "s3", "s4", "s5"],
        store=fake_store,
        scope=Scope(org_id="d", user_id="a"),
    )
    # Only 2 sessions queried (max_sessions=2), even though 5 ids passed.
    assert fake_store.list_messages_by_session.await_count == 2


@pytest.mark.asyncio
async def test_reextract_skips_sessions_with_no_messages():
    fake_llm = AsyncMock()
    fake_store = AsyncMock()
    fake_store.list_messages_by_session = AsyncMock(return_value=[])
    rx = QueryConditionedReextractor(llm=fake_llm)
    facts = await rx.reextract(
        question="?",
        candidate_session_ids=["s1"],
        store=fake_store,
        scope=Scope(org_id="d", user_id="a"),
    )
    assert facts == []
    fake_llm.generate.assert_not_called()
