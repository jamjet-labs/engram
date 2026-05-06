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
from engram.models import ChatMessage, Fact, MemoryTier, Polarity
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
    "MemoryTier",
    "NotFoundError",
    "Polarity",
    "QuestionType",
    "Reader",
    "ReaderConfig",
    "RetrievalConfig",
    "RuleBasedClassifier",
    "Scope",
    "StoreError",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "is_preference_question",
]
