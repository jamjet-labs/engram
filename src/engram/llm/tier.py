"""ModelTier — split reader (answer generation) from utility (verifier, decomposer, etc.).

Default puts both on gpt-4o-mini for cost optimisation. Promote reader to Sonnet 4.6
via `ModelTier.sonnet_reader()` only if the item-1 ablation justifies the ~30x cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.llm.base import LLMClient


@dataclass
class ModelTier:
    reader: LLMClient
    utility: LLMClient

    @classmethod
    def default(cls) -> ModelTier:
        from engram.llm.openai import OpenAILLM

        return cls(
            reader=OpenAILLM(model="gpt-4o-mini"),
            utility=OpenAILLM(model="gpt-4o-mini"),
        )

    @classmethod
    def sonnet_reader(cls) -> ModelTier:
        from engram.llm.anthropic import AnthropicLLM
        from engram.llm.openai import OpenAILLM

        return cls(
            reader=AnthropicLLM(model="claude-sonnet-4-6"),
            utility=OpenAILLM(model="gpt-4o-mini"),
        )

    @classmethod
    def haiku_reader(cls) -> ModelTier:
        from engram.llm.anthropic import AnthropicLLM
        from engram.llm.openai import OpenAILLM

        return cls(
            reader=AnthropicLLM(model="claude-haiku-4-5-20251001"),
            utility=OpenAILLM(model="gpt-4o-mini"),
        )
