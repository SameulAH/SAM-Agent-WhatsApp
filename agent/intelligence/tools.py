"""
Tool Interface & Registry

Abstract interface for tools (Web Search, etc.).
Tool registry for discovery and execution.
Safe stub tool for testing.

Enforces:
- Tools cannot write memory
- Tools cannot modify routing
- Tools must return typed results
- Tools must timeout safely
- No tool side effects
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime

from pydantic import BaseModel


class ToolResult(BaseModel):
    """
    Result from tool execution.
    
    Invariants:
    - success ∈ {true, false}
    - data is JSON-serializable
    - error only present if success=false
    """
    
    success: bool
    data: Dict[str, Any] = {}
    error: Optional[str] = None
    execution_time_ms: int = 0


class ToolInputSchema(BaseModel):
    """Schema for tool inputs."""
    
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []


class ToolInterface(ABC):
    """
    Abstract base for all tools.
    
    Enforces:
    - Deterministic execution
    - No memory mutation
    - No routing modification
    - Typed inputs/outputs
    - Safe failure handling
    """
    
    name: str
    description: str
    input_schema: ToolInputSchema
    
    @abstractmethod
    def execute(self, input_dict: Dict[str, Any]) -> ToolResult:
        """
        Execute tool with given input.
        
        Must:
        - Validate inputs against schema
        - Never mutate memory
        - Never modify routing
        - Return ToolResult
        - Handle exceptions gracefully
        
        Args:
            input_dict: Input parameters
        
        Returns:
            ToolResult with success/error
        """
        pass
    
    def _validate_input(self, input_dict: Dict[str, Any]) -> bool:
        """Validate input against schema."""
        # Check required fields
        for field in self.input_schema.required:
            if field not in input_dict:
                return False
        return True


class WebSearchStubTool(ToolInterface):
    """
    Safe stub tool for Web Search.
    
    Returns deterministic dummy responses.
    For testing and development only.
    """
    
    def __init__(self):
        """Initialize web search stub."""
        self.name = "web_search"
        self.description = "Search the web for information (stub implementation)"
        self.input_schema = ToolInputSchema(
            properties={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results"},
            },
            required=["query"],
        )
    
    def execute(self, input_dict: Dict[str, Any]) -> ToolResult:
        """
        Execute web search (stub).
        
        Returns deterministic dummy results.
        
        Args:
            input_dict: {"query": str, "max_results": int (optional)}
        
        Returns:
            ToolResult with dummy search results
        """
        
        try:
            import time
            start = time.time()
            
            if not self._validate_input(input_dict):
                return ToolResult(
                    success=False,
                    error="Missing required field: query",
                )
            
            query = input_dict.get("query", "").strip()
            max_results = min(input_dict.get("max_results", 3), 5)
            
            if not query:
                return ToolResult(
                    success=False,
                    error="Query cannot be empty",
                )
            
            # Deterministic stub response
            results = [
                {
                    "title": f"Result {i+1} for '{query}'",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a stub result for search term: {query}",
                }
                for i in range(max_results)
            ]
            
            elapsed = int((time.time() - start) * 1000)
            
            return ToolResult(
                success=True,
                data={"results": results, "query": query},
                execution_time_ms=elapsed,
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}",
            )


class ToolRegistry:
    """
    Registry for available tools.
    
    Manages:
    - Tool registration
    - Tool discovery
    - Tool execution (no mutation)
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, ToolInterface] = {}
    
    def register(self, tool: ToolInterface) -> None:
        """
        Register a tool.
        
        Args:
            tool: ToolInterface instance
        """
        if not isinstance(tool, ToolInterface):
            raise TypeError(f"Tool must implement ToolInterface")
        
        self._tools[tool.name] = tool
    
    def get(self, tool_name: str) -> Optional[ToolInterface]:
        """
        Get tool by name.
        
        Args:
            tool_name: Name of tool
        
        Returns:
            ToolInterface or None if not found
        """
        return self._tools.get(tool_name)
    
    def list(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def execute(
        self,
        tool_name: str,
        input_dict: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool safely.
        
        Args:
            tool_name: Name of tool to execute
            input_dict: Input parameters
        
        Returns:
            ToolResult (failure if tool not found)
        """
        
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}",
            )
        
        try:
            return tool.execute(input_dict)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
            )


def get_tool_registry() -> ToolRegistry:
    """Get singleton tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        # Register default tools
        _registry.register(WebSearchStubTool())
    return _registry


_registry: Optional[ToolRegistry] = None
