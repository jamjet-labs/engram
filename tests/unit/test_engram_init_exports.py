"""Tests for engram top-level public re-exports."""

from __future__ import annotations


def test_top_level_exports_resolve():
    """All documented top-level exports must be importable from `engram`."""
    from engram import (
        ChatMessage,
        EmbedderIdentity,
        EmbedderMismatchError,
        EmbeddingAuthError,
        EmbeddingError,
        EmbeddingQuotaError,
        EmbeddingResponseError,
        EmbeddingTimeoutError,
        Engram,
        EngramError,
        ExtractionError,
        Fact,
        LLMClient,
        LLMMessage,
        LLMResponse,
        MemoryTier,
        NotFoundError,
        Polarity,
        QuestionType,
        Reader,
        ReaderConfig,
        ReadOnlyStoreError,
        RetrievalConfig,
        RuleBasedClassifier,
        SchemaVersionError,
        Scope,
        ScopedEngram,
        ScopeRequiredError,
        StoreError,
        Tool,
        ToolRegistry,
        ToolResult,
        is_preference_question,
    )

    # Smoke check: each is non-None
    for name in (
        ChatMessage,
        EmbedderIdentity,
        EmbedderMismatchError,
        EmbeddingAuthError,
        EmbeddingError,
        EmbeddingQuotaError,
        EmbeddingResponseError,
        EmbeddingTimeoutError,
        Engram,
        EngramError,
        ExtractionError,
        Fact,
        is_preference_question,
        LLMClient,
        LLMMessage,
        LLMResponse,
        MemoryTier,
        NotFoundError,
        Polarity,
        QuestionType,
        Reader,
        ReaderConfig,
        ReadOnlyStoreError,
        RetrievalConfig,
        RuleBasedClassifier,
        SchemaVersionError,
        Scope,
        ScopedEngram,
        ScopeRequiredError,
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
        "EmbedderIdentity",
        "EmbedderMismatchError",
        "EmbeddingAuthError",
        "EmbeddingError",
        "EmbeddingQuotaError",
        "EmbeddingResponseError",
        "EmbeddingTimeoutError",
        "Engram",
        "EngramError",
        "ErasureRecord",
        "ExtractionError",
        "Fact",
        "is_preference_question",
        "LLMClient",
        "LLMMessage",
        "LLMResponse",
        "MemoryTier",
        "NotFoundError",
        "Polarity",
        "Provenance",
        "QuestionType",
        "Reader",
        "ReaderConfig",
        "ReadOnlyStoreError",
        "RetrievalConfig",
        "Revocation",
        "RevokedContentError",
        "RuleBasedClassifier",
        "SchemaVersionError",
        "Scope",
        "ScopeRequiredError",
        "ScopedEngram",
        "StoreError",
        "Tool",
        "ToolRegistry",
        "ToolResult",
    }
    actual = set(engram.__all__)
    assert expected == actual, f"diff: {expected ^ actual}"


def test_vertex_embedding_stays_out_of_the_top_level():
    """VertexEmbedding needs the vertex extra, so it is imported from its module."""
    import engram

    assert "VertexEmbedding" not in engram.__all__
    assert not hasattr(engram, "VertexEmbedding")
