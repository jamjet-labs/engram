"""Tests for prompts module — verify SYNTHESIS_PROMPT is exposed."""

from __future__ import annotations


def test_synthesis_prompt_imports_from_prompts_module():
    from engram.read.prompts import SYNTHESIS_PROMPT

    assert "preference/recommendation" in SYNTHESIS_PROMPT.lower()
    assert "{context}" in SYNTHESIS_PROMPT
    assert "{question}" in SYNTHESIS_PROMPT
    assert "I don't know" in SYNTHESIS_PROMPT


def test_synthesis_prompt_still_importable_from_smoke_runner():
    """Backward-compat: smoke runner re-exports the constant during transition."""
    from benchmarks.smoke_runner import SYNTHESIS_PROMPT as smoke_prompt
    from engram.read.prompts import SYNTHESIS_PROMPT as core_prompt

    assert smoke_prompt is core_prompt
