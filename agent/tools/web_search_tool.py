"""
Web Search Tool — MCP-backed implementation.

Implements ToolInterface using MCPClient (BrowserBase MCP server).

Invariants:
- Does NOT write to memory
- Does NOT modify state
- Does NOT log secrets
- Returns structured ToolResult
- Handles all failures gracefully

Usage:
    tool = WebSearchTool()
    result = tool.execute({"query": "AI regulation", "max_results": 3})
"""

import time
from typing import Any, Dict, Optional

from agent.intelligence.tools import ToolInterface, ToolInputSchema, ToolResult
from agent.mcp.external_client import BrowserBaseArgs, MCPClient
from agent.mcp.guardrails import GuardrailViolation, MCPGuardrails


class WebSearchTool(ToolInterface):
    """
    Web search tool backed by a semantic search MCP provider (Exa / Brave / Linkup).

    Provider priority: Exa → Brave → Linkup (first with valid credentials wins).

    Enforces:
    - Schema validation on inputs
    - Guardrail-bounded results (max 5, snippet ≤ 300 chars, budget ≤ 1500 chars)
    - Configurable timeout (default 10 s)
    - No memory mutation
    - No routing modification

    Tool name: "web_search"
    """

    name = "web_search"
    description = (
        "Search the web for real-time information using semantic search (Exa / Brave / Linkup). "
        "Use for current news, recent events, prices, or any info not in the model's training data. "
        "Returns structured results with titles, URLs, and snippets."
    )
    input_schema = ToolInputSchema(
        properties={
            "query": {
                "type": "string",
                "description": "Search query (minimum 3 characters)",
                "minLength": 3,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-5)",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
        },
        required=["query"],
    )

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        tracer: Optional[Any] = None,
        trace_metadata: Optional[Any] = None,
    ):
        """
        Initialize WebSearchTool.

        Args:
            mcp_client: Optional MCPClient instance (creates default if None).
            tracer: Optional Tracer for trace emission.
            trace_metadata: Optional TraceMetadata for trace_id propagation.
        """
        self._client = mcp_client or MCPClient()
        self._tracer = tracer
        self._trace_metadata = trace_metadata

    def execute(self, input_dict: Dict[str, Any]) -> ToolResult:
        """
        Execute web search.

        Does NOT:
        - Write to memory
        - Modify routing state
        - Log API keys or secrets
        - Auto-retry on failure

        Args:
            input_dict: {"query": str, "max_results": int (optional, default 3)}

        Returns:
            ToolResult with results list or error message.
        """
        start_ms = time.time()

        # Input validation
        if not self._validate_input(input_dict):
            return ToolResult(
                success=False,
                error="Missing required field: query",
            )

        query = str(input_dict.get("query", "")).strip()
        max_results = int(input_dict.get("max_results", 3))

        if len(query) < 3:
            return ToolResult(
                success=False,
                error="Query must be at least 3 characters",
            )

        # Clamp max_results to guardrail limit
        max_results = max(1, min(max_results, MCPGuardrails.MAX_RESULTS))

        # Build validated args
        try:
            args = BrowserBaseArgs(query=query, max_results=max_results)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid search arguments: {e}",
            )

        # Execute MCP call (sync wrapper)
        mcp_response = self._client.search_sync(
            args=args,
            tracer=self._tracer,
            trace_metadata=self._trace_metadata,
        )

        elapsed_ms = int((time.time() - start_ms) * 1000)

        if mcp_response.status == "error" or not mcp_response.results:
            return ToolResult(
                success=False,
                error="Web search returned no results or encountered an error.",
                execution_time_ms=elapsed_ms,
            )

        # Convert results to plain dicts (never raw Pydantic in ToolResult.data)
        results_data = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            }
            for r in mcp_response.results
        ]

        return ToolResult(
            success=True,
            data={
                "results": results_data,
                "query": query,
                "result_count": len(results_data),
            },
            execution_time_ms=elapsed_ms,
        )
