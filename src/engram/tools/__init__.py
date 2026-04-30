"""Tools for the tool-augmented Reader (item 4 / N5) and ReAct agent (item 7).

The same Tool implementations are reused in both modes — registry, schema,
and dispatch logic live here.
"""

from engram.tools.base import Tool, ToolRegistry, ToolResult

__all__ = ["Tool", "ToolRegistry", "ToolResult"]
