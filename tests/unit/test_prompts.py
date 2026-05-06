"""Tests for prompts module — verify SYNTHESIS_PROMPT is exposed."""

from __future__ import annotations


def test_synthesis_prompt_imports_from_prompts_module():
    from engram.read.prompts import SYNTHESIS_PROMPT

    assert "preference/recommendation" in SYNTHESIS_PROMPT.lower()
    assert "{context}" in SYNTHESIS_PROMPT
    assert "{question}" in SYNTHESIS_PROMPT
    assert "I don't know" in SYNTHESIS_PROMPT
