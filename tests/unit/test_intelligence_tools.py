"""
Tests for Tool Interface & Registry

Unit and integration tests for tool abstraction, registry, and safety.
"""

import pytest
from agent.intelligence.tools import (
    ToolInterface,
    ToolResult,
    ToolInputSchema,
    WebSearchStubTool,
    ToolRegistry,
    get_tool_registry,
)


class TestToolResult:
    """Test ToolResult model."""
    
    def test_success_result(self):
        """Create successful tool result."""
        result = ToolResult(
            success=True,
            data={"query": "python", "results": ["Link 1", "Link 2"]},
            error=None,
            execution_time_ms=150,
        )
        
        assert result.success == True
        assert result.data is not None
        assert result.error is None
        assert result.execution_time_ms == 150
    
    def test_error_result(self):
        """Create error tool result."""
        result = ToolResult(
            success=False,
            data={},
            error="Network timeout",
            execution_time_ms=5000,
        )
        
        assert result.success == False
        assert result.data == {}
        assert result.error == "Network timeout"


class TestWebSearchStubTool:
    """Test WebSearchStubTool (deterministic stub)."""
    
    def test_web_search_stub_properties(self):
        """Stub tool has correct properties."""
        tool = WebSearchStubTool()
        
        assert tool.name == "web_search"
        assert tool.description is not None
        assert len(tool.description) > 0
        assert tool.input_schema is not None
    
    def test_web_search_stub_execution(self):
        """Stub tool returns deterministic results."""
        tool = WebSearchStubTool()
        
        result = tool.execute({"query": "python"})
        
        assert result.success == True
        assert result.data is not None
        assert "results" in result.data
        assert isinstance(result.data["results"], list)
        assert result.execution_time_ms >= 0
    
    def test_web_search_stub_deterministic(self):
        """Same query returns same results."""
        tool = WebSearchStubTool()
        
        result1 = tool.execute({"query": "python"})
        result2 = tool.execute({"query": "python"})
        
        assert result1.data == result2.data
        assert result1.success == result2.success
    
    def test_web_search_stub_input_validation(self):
        """Stub tool validates inputs."""
        tool = WebSearchStubTool()
        
        # Missing required field
        result = tool.execute({})
        
        # Should fail validation
        assert result.success == False
        assert result.error is not None
    
    def test_web_search_stub_no_memory_mutation(self):
        """Stub tool cannot mutate memory (stub property)."""
        tool = WebSearchStubTool()
        
        # Verify it has no memory interaction methods
        assert not hasattr(tool, "write_memory")
        assert not hasattr(tool, "mutate_state")


class TestToolRegistry:
    """Test ToolRegistry."""
    
    def test_register_tool(self):
        """Register new tool in registry."""
        registry = ToolRegistry()
        tool = WebSearchStubTool()
        
        registry.register(tool)
        
        retrieved = registry.get("web_search")
        assert retrieved is not None
        assert retrieved.name == "web_search"
    
    def test_list_tools(self):
        """List all registered tools."""
        registry = ToolRegistry()
        registry.register(WebSearchStubTool())
        
        names = registry.list()
        
        assert "web_search" in names
        assert isinstance(names, list)
    
    def test_get_nonexistent_tool(self):
        """Getting nonexistent tool returns None."""
        registry = ToolRegistry()
        
        tool = registry.get("nonexistent_tool")
        
        assert tool is None
    
    def test_execute_tool(self):
        """Execute tool through registry."""
        registry = ToolRegistry()
        registry.register(WebSearchStubTool())
        
        result = registry.execute("web_search", {"query": "python"})
        
        assert result.success == True
        assert result.data is not None
    
    def test_execute_nonexistent_tool(self):
        """Executing nonexistent tool returns error."""
        registry = ToolRegistry()
        
        result = registry.execute("nonexistent", {})
        
        assert result.success == False
        assert result.error is not None


class TestGlobalToolRegistry:
    """Test global tool registry singleton."""
    
    def test_get_global_registry(self):
        """Get global tool registry."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()
        
        # Should be same instance (singleton)
        assert registry1 is registry2
    
    def test_global_registry_has_defaults(self):
        """Global registry includes default tools."""
        registry = get_tool_registry()
        
        names = registry.list()
        
        # Should have web_search by default
        assert "web_search" in names
    
    def test_global_registry_execute(self):
        """Execute tool from global registry."""
        registry = get_tool_registry()
        
        result = registry.execute("web_search", {"query": "test"})
        
        assert result.success == True


class TestToolInterfaceContract:
    """Test ToolInterface contract."""
    
    def test_tool_must_have_name(self):
        """All tools have a name."""
        tool = WebSearchStubTool()
        
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)
        assert len(tool.name) > 0
    
    def test_tool_must_have_description(self):
        """All tools have a description."""
        tool = WebSearchStubTool()
        
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0
    
    def test_tool_must_have_input_schema(self):
        """All tools have an input schema."""
        tool = WebSearchStubTool()
        
        assert hasattr(tool, "input_schema")
        assert tool.input_schema is not None
    
    def test_tool_must_have_execute(self):
        """All tools have execute method."""
        tool = WebSearchStubTool()
        
        assert hasattr(tool, "execute")
        assert callable(tool.execute)


class TestToolGuardrailsCompliance:
    """Test that tools cannot bypass guardrails."""
    
    def test_tool_result_structure(self):
        """Tool results have consistent structure."""
        tool = WebSearchStubTool()
        result = tool.execute({"query": "test"})
        
        # Must have these fields
        assert hasattr(result, "success")
        assert hasattr(result, "data")
        assert hasattr(result, "error")
        assert hasattr(result, "execution_time_ms")
    
    def test_tool_cannot_inject_instructions(self):
        """Tool results cannot inject new instructions."""
        tool = WebSearchStubTool()
        
        # Even if tool tries to return instructions
        result = tool.execute({"query": "act as admin"})
        
        # The result is still a ToolResult, not executable code
        assert isinstance(result, ToolResult)
        assert result.data is not None  # Contains stub data
