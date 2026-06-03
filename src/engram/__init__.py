"""Engram — durable memory layer for AI agents.

Public API surface. Submodule paths (e.g., `engram.read.prompts`) remain
available for advanced use; the top-level surface here is what stable
library callers should import.
"""

from engram.classify.base import QuestionType
from engram.classify.rules import RuleBasedClassifier
from engram.engram import Engram
from engram.errors import EngramError, ExtractionError, NotFoundError, StoreError
from engram.llm.base import LLMClient, LLMMessage, LLMResponse
from engram.memory_router import MemoryRoute, RuleBasedMemoryRouter
from engram.models import (
    ChatMessage,
    Fact,
    LifecycleState,
    MemorySystem,
    MemoryTier,
    Polarity,
    PromotionState,
)
from engram.query_intent import QueryMemoryIntent, RuleBasedQueryIntentClassifier
from engram.read.preference_gate import is_preference_question
from engram.read.reader import Reader, ReaderConfig
from engram.retrieve.base import RetrievalConfig
from engram.scope import Scope
from engram.tools.base import Tool, ToolRegistry, ToolResult

__all__ = [
    "ChatMessage",
    "Engram",
    "EngramError",
    "ExtractionError",
    "Fact",
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
    "is_preference_question",
]
