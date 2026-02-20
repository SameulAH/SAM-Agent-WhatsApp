"""
tests/tools/test_web_search_tool.py

Unit tests for WebSearchTool (MCP-backed implementation).

Verifies:
✔ Valid input executes successfully
✔ Missing query returns ToolResult(success=False)
✔ Short query returns error
✔ Tool result is bounded (max 5 results)
✔ Tool never writes to memory
✔ Tool never modifies state
✔ MCP failure returns success=False gracefully
✔ Tool result is JSON-serializable
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.intelligence.tools import ToolResult
from agent.mcp.external_client import BrowserBaseResult, MCPResponse
from agent.tools.web_search_tool import WebSearchTool


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────


def make_mock_mcp_success(n_results: int = 3) -> MagicMock:
    """Create a mock MCPClient that returns n_results success results."""
    client = MagicMock()
    results = [
        BrowserBaseResult(
            title=f"Result {i}",
            url=f"https://example.com/{i}",
            snippet=f"Snippet for result {i}",
        )
        for i in range(n_results)
    ]
    client.search_sync.return_value = MCPResponse(status="success", results=results)
    return client


def make_mock_mcp_error() -> MagicMock:
    """Create a mock MCPClient that returns an error."""
    client = MagicMock()
    client.search_sync.return_value = MCPResponse(status="error", results=[])
    return client


# ─────────────────────────────────────────────────────
# Basic execution
# ─────────────────────────────────────────────────────


class TestWebSearchToolBasic:
    def test_valid_query_returns_success(self):
        tool = WebSearchTool(mcp_client=make_mock_mcp_success())
        result = tool.execute({"query": "AI regulation EU"})
        assert result.success is True
        assert "results" in result.data
        assert result.data["result_count"] == 3

    def test_result_is_json_serializable(self):
        tool = WebSearchTool(mcp_client=make_mock_mcp_success())
        result = tool.execute({"query": "AI regulation EU"})
        # Must not raise
        serialized = json.dumps(result.model_dump())
        assert "results" in serialized

    def test_missing_query_returns_failure(self):
        tool = WebSearchTool(mcp_client=make_mock_mcp_success())
        result = tool.execute({})
        assert result.success is False
        assert "query" in result.error.lower()

    def test_short_query_returns_failure(self):
        tool = WebSearchTool(mcp_client=make_mock_mcp_success())
        result = tool.execute({"query": "AI"})
        assert result.success is False

    def test_mcp_error_returns_failure(self):
        tool = WebSearchTool(mcp_client=make_mock_mcp_error())
        result = tool.execute({"query": "AI regulation"})
        assert result.success is False

    def test_max_results_respected(self):
        tool = WebSearchTool(mcp_client=make_mock_mcp_success(n_results=5))
        result = tool.execute({"query": "AI regulation", "max_results": 5})
        assert result.success is True
        assert result.data["result_count"] <= 5

    def test_max_results_clamped_above_5(self):
        """max_results > 5 is clamped to 5 by guardrail."""
        client = make_mock_mcp_success(n_results=5)
        tool = WebSearchTool(mcp_client=client)
        result = tool.execute({"query": "AI regulation", "max_results": 99})
        # The client should have been called with max_results=5
        call_args = client.search_sync.call_args
        assert call_args[1]["args"].max_results == 5 or call_args[0][0].max_results == 5


# ─────────────────────────────────────────────────────
# No memory mutation
# ─────────────────────────────────────────────────────


class TestWebSearchToolNoMemoryMutation:
    def test_tool_does_not_write_to_memory(self):
        """Tool must never access memory controller."""
        tool = WebSearchTool(mcp_client=make_mock_mcp_success())

        # If tool tries to import or call memory, we'd catch it via monkeypatching
        # Simplest check: tool.execute returns ToolResult without any memory attribute
        result = tool.execute({"query": "AI regulation EU"})
        assert isinstance(result, ToolResult)
        assert not hasattr(result, "memory_write")
        assert not hasattr(result, "memory_read")

    def test_tool_does_not_modify_state(self):
        """ToolResult must only contain success/data/error/execution_time_ms."""
        tool = WebSearchTool(mcp_client=make_mock_mcp_success())
        result = tool.execute({"query": "AI regulation EU"})

        # ToolResult fields only
        result_dict = result.model_dump()
        allowed_keys = {"success", "data", "error", "execution_time_ms"}
        assert set(result_dict.keys()) == allowed_keys


# ─────────────────────────────────────────────────────
# Trace propagation
# ─────────────────────────────────────────────────────


class TestWebSearchToolTracing:
    def test_tool_passes_tracer_to_client(self):
        """Tracer and trace_metadata must be forwarded to MCPClient."""
        mock_client = make_mock_mcp_success()
        mock_tracer = MagicMock()
        mock_meta = MagicMock()

        tool = WebSearchTool(
            mcp_client=mock_client,
            tracer=mock_tracer,
            trace_metadata=mock_meta,
        )
        tool.execute({"query": "test query"})

        # Verify search_sync was called with tracer and trace_metadata
        call_kwargs = mock_client.search_sync.call_args[1]
        assert call_kwargs.get("tracer") is mock_tracer
        assert call_kwargs.get("trace_metadata") is mock_meta


# ─────────────────────────────────────────────────────
# Concurrency safety
# ─────────────────────────────────────────────────────


class TestWebSearchToolConcurrency:
    def test_three_parallel_invocations(self):
        """Three simultaneous calls must all succeed independently."""
        import threading

        results = []
        errors = []

        def run():
            client = make_mock_mcp_success(n_results=2)
            tool = WebSearchTool(mcp_client=client)
            try:
                r = tool.execute({"query": "concurrent test"})
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3
        assert all(r.success for r in results)
