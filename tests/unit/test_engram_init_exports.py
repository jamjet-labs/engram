"""Tests for engram top-level public re-exports."""

from __future__ import annotations


def test_top_level_exports_resolve():
    """All documented top-level exports must be importable from `engram`."""
    from engram import (
        ChatMessage,
        Engram,
        EngramError,
        ExtractionError,
        Fact,
        LifecycleState,
        LLMClient,
        LLMMessage,
        LLMResponse,
        MemoryRoute,
        MemorySystem,
        MemoryTier,
        NotFoundError,
        Polarity,
        PromotionState,
        QueryMemoryIntent,
        QuestionType,
        Reader,
        ReaderConfig,
        RetrievalConfig,
        RuleBasedClassifier,
        RuleBasedMemoryRouter,
        RuleBasedQueryIntentClassifier,
        Scope,
        StoreError,
        Tool,
        ToolRegistry,
        ToolResult,
        is_preference_question,
    )

    for name in (
        ChatMessage,
        Engram,
        EngramError,
        ExtractionError,
        Fact,
        is_preference_question,
        LLMClient,
        LLMMessage,
        LLMResponse,
        LifecycleState,
        MemoryRoute,
        MemorySystem,
        MemoryTier,
        NotFoundError,
        Polarity,
        PromotionState,
        QueryMemoryIntent,
        QuestionType,
        Reader,
        ReaderConfig,
        RetrievalConfig,
        RuleBasedClassifier,
        RuleBasedMemoryRouter,
        RuleBasedQueryIntentClassifier,
        Scope,
        StoreError,
        Tool,
        ToolRegistry,
        ToolResult,
    ):
        assert name is not None


def test_top_level_all_lists_exported_names():
    """`engram.__all__` should list every public re-export."""
    import engram

    expected = {
        "ChatMessage",
        "Engram",
        "EngramError",
        "ExtractionError",
        "Fact",
        "is_preference_question",
        "LLMClient",
        "LLMMessage",
        "LLMResponse",
        "LifecycleState",
        "MemoryRoute",
        "MemorySystem",
        "MemoryTier",
        "NotFoundError",
        "Polarity",
        "PromotionState",
        "QueryMemoryIntent",
        "QuestionType",
        "Reader",
        "ReaderConfig",
        "RetrievalConfig",
        "RuleBasedClassifier",
        "RuleBasedMemoryRouter",
        "RuleBasedQueryIntentClassifier",
        "Scope",
        "StoreError",
        "Tool",
        "ToolRegistry",
        "ToolResult",
    }
    actual = set(engram.__all__)
    assert expected == actual, f"diff: {expected ^ actual}"
