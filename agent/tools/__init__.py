"""
agent/tools package.

Registers all available tools into the ToolRegistry.
"""

from agent.tools.web_search_tool import WebSearchTool

__all__ = ["WebSearchTool"]
