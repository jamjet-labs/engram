from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.read.reader import Reader, ReaderConfig
from engram.scope import Scope


@pytest.mark.asyncio
async def test_react_fires_when_post_verdict_partial():
    """If after the reader (and any other escalation rungs) the verdict is
    still PARTIAL/NO, ReAct kicks in with its tools."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        # 1. initial verifier — YES so we proceed
        MagicMock(content="<verdict>YES</verdict>"),
        # 2. weak initial answer
        MagicMock(content="not sure"),
        # 3. ReAct pre-check verifier — PARTIAL
        MagicMock(content="<verdict>PARTIAL</verdict>"),
    ]
    fake_react = AsyncMock()
    fake_react.answer = AsyncMock(
        return_value=MagicMock(
            answer="ReAct found it: 4", abstained=False, n_hops=2, trace=[]
        )
    )
    reader = Reader(
        fake_llm,
        verifier=True,
        config=ReaderConfig(enable_reextract=False, self_consistency_on_partial=1),
    )
    reader.set_category("temporal-reasoning")
    reader.attach_react(fake_react)
    res = await reader.read(
        question="How many?",
        context="ctx",
        scope=Scope(org_id="d", user_id="a"),
    )
    fake_react.answer.assert_called_once()
    assert "4" in res.answer
    assert res.verdict == "YES"


@pytest.mark.asyncio
async def test_react_skipped_when_post_verdict_yes():
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="answer"),
        MagicMock(content="<verdict>YES</verdict>"),
    ]
    fake_react = AsyncMock()
    reader = Reader(
        fake_llm,
        verifier=True,
        config=ReaderConfig(enable_reextract=False, self_consistency_on_partial=1),
    )
    reader.set_category("multi-session")
    reader.attach_react(fake_react)
    await reader.read(question="?", context="ctx", scope=Scope(org_id="d", user_id="a"))
    fake_react.answer.assert_not_called()


@pytest.mark.asyncio
async def test_react_abstains_does_not_overwrite_answer():
    """If ReAct itself abstains, the original reader answer is kept."""
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="original answer"),
        MagicMock(content="<verdict>PARTIAL</verdict>"),
    ]
    fake_react = AsyncMock()
    fake_react.answer = AsyncMock(
        return_value=MagicMock(
            answer="I don't know", abstained=True, n_hops=4, trace=[]
        )
    )
    reader = Reader(
        fake_llm,
        verifier=True,
        config=ReaderConfig(enable_reextract=False, self_consistency_on_partial=1),
    )
    reader.set_category("temporal-reasoning")
    reader.attach_react(fake_react)
    res = await reader.read(
        question="?", context="ctx", scope=Scope(org_id="d", user_id="a"),
    )
    # ReAct was tried but abstained; original answer is kept
    assert res.answer == "original answer"


@pytest.mark.asyncio
async def test_react_skipped_when_not_attached():
    fake_llm = AsyncMock()
    fake_llm.generate.side_effect = [
        MagicMock(content="<verdict>YES</verdict>"),
        MagicMock(content="answer"),
    ]
    reader = Reader(
        fake_llm,
        verifier=True,
        config=ReaderConfig(enable_reextract=False, self_consistency_on_partial=1),
    )
    res = await reader.read(question="?", context="ctx", scope=Scope(org_id="d", user_id="a"))
    assert res.answer == "answer"
