import pytest

from engram.llm.openai import OpenAILLM
from engram.llm.tier import ModelTier


def test_default_tier_uses_gpt4o_mini_for_both(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    tier = ModelTier.default()
    assert isinstance(tier.reader, OpenAILLM)
    assert isinstance(tier.utility, OpenAILLM)
    # Reader and utility are different instances even when same model
    assert tier.reader is not tier.utility


def test_sonnet_reader_tier(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    tier = ModelTier.sonnet_reader()
    from engram.llm.anthropic import AnthropicLLM

    assert isinstance(tier.reader, AnthropicLLM)
    assert isinstance(tier.utility, OpenAILLM)


def test_haiku_reader_tier(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    tier = ModelTier.haiku_reader()
    from engram.llm.anthropic import AnthropicLLM

    assert isinstance(tier.reader, AnthropicLLM)
    assert isinstance(tier.utility, OpenAILLM)


def test_modeltier_construction_with_explicit_clients(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    a = OpenAILLM(model="gpt-4o-mini")
    b = OpenAILLM(model="gpt-4o-mini")
    tier = ModelTier(reader=a, utility=b)
    assert tier.reader is a
    assert tier.utility is b


@pytest.mark.asyncio
async def test_engram_open_accepts_tier(monkeypatch):
    from engram import Engram

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    tier = ModelTier.default()
    async with await Engram.open(":memory:", tier=tier) as memory:
        assert memory.tier is tier


@pytest.mark.asyncio
async def test_engram_open_without_tier_keeps_back_compat(monkeypatch):
    from engram import Engram

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    async with await Engram.open(":memory:") as memory:
        assert memory.tier is None
