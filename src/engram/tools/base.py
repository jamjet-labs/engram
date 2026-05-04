"""Tool protocol and registry — shared between tool-augmented Reader (N5)
and the ReAct agent (item 7).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ToolResult(BaseModel):
    content: str
    raw: Any = None

    model_config = {"arbitrary_types_allowed": True}


@runtime_checkable
class Tool(Protocol):
    """Async tool with a JSON-schema input description.

    Implementations must expose ``name``, ``description``, ``input_schema`` as
    plain attributes (not properties) so the registry can introspect them
    without instantiating.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    async def __call__(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    """Holds a set of named tools + dispatches by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def for_anthropic(self) -> list[dict[str, Any]]:
        """Tools in Anthropic Messages API ``tools=`` shape."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def for_openai(self) -> list[dict[str, Any]]:
        """Tools in OpenAI Chat Completions ``tools=`` shape."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    async def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return await self._tools[name](**args)

    def signature(self) -> str:
        """Stable signature for caching keys — order-independent."""
        return ",".join(self.names())
