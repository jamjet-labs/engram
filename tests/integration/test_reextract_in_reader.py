from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.read.reader import Reader, ReaderConfig
from engram.scope import Scope


@pytest.mark.asyncio
async def test_reextract_fires_on_partial_verdict():
    """Reader produces an answer, verifier returns PARTIAL on it,
    so reextractor is invoked, an extra context is built, and re-read happens."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        # 1. initial verifier (verifier_enabled=True, runs first) — say YES so we get past it
        MagicMock(content="<verdict>YES</verdict>"),
        # 2. initial reader answer
        MagicMock(content="weak answer"),
        # 3. post-answer verifier — PARTIAL → triggers reextract rung
        MagicMock(content="<verdict>PARTIAL</verdict>"),
        # 4. re-read after ephemeral facts injected
        MagicMock(content="strong answer 42"),
        # 5. final verifier on re-read
        MagicMock(content="<verdict>YES</verdict>"),
    ]
    fake_reextractor = AsyncMock()
    fake_reextractor.reextract = AsyncMock(
        return_value=[
            MagicMock(text="extra fact about marathons", confidence=0.9),
        ]
    )
    fake_store = AsyncMock()

    reader = Reader(fake_llm, verifier=True, config=ReaderConfig(enable_reextract=True))
    reader.attach_reextractor(
        reextractor=fake_reextractor,
        store=fake_store,
        candidate_sessions_provider=lambda q: ["s1"],
    )
    res = await reader.read(
        question="How many marathons?",
        context="ctx",
        scope=Scope(org_id="default", user_id="alice"),
    )
    fake_reextractor.reextract.assert_called_once()
    assert "42" in res.answer
    assert res.verdict == "YES"


@pytest.mark.asyncio
async def test_reextract_skipped_when_post_verdict_yes():
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),  # initial verifier
        MagicMock(content="answer"),  # reader
        MagicMock(content="<verdict>YES</verdict>"),  # post-answer verifier
    ]
    fake_reextractor = AsyncMock()
    fake_store = AsyncMock()
    reader = Reader(fake_llm, verifier=True, config=ReaderConfig(enable_reextract=True))
    reader.attach_reextractor(fake_reextractor, fake_store, lambda q: ["s1"])
    res = await reader.read(
        question="?",
        context="ctx",
        scope=Scope(org_id="d", user_id="a"),
    )
    fake_reextractor.reextract.assert_not_called()
    assert res.answer == "answer"


@pytest.mark.asyncio
async def test_reextract_skipped_when_no_candidate_sessions():
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),  # initial
        MagicMock(content="answer"),
        MagicMock(content="<verdict>PARTIAL</verdict>"),
    ]
    fake_reextractor = AsyncMock()
    fake_store = AsyncMock()
    reader = Reader(fake_llm, verifier=True, config=ReaderConfig(enable_reextract=True))
    reader.attach_reextractor(fake_reextractor, fake_store, lambda q: [])
    await reader.read(question="?", context="ctx", scope=Scope(org_id="d", user_id="a"))
    fake_reextractor.reextract.assert_not_called()


@pytest.mark.asyncio
async def test_reextract_disabled_via_config():
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="answer"),
    ]
    fake_reextractor = AsyncMock()
    fake_store = AsyncMock()
    reader = Reader(fake_llm, verifier=True, config=ReaderConfig(enable_reextract=False))
    reader.attach_reextractor(fake_reextractor, fake_store, lambda q: ["s1"])
    await reader.read(question="?", context="ctx", scope=Scope(org_id="d", user_id="a"))
    fake_reextractor.reextract.assert_not_called()
