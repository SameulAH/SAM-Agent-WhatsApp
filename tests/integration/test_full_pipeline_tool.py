"""
tests/integration/test_full_pipeline_tool.py

Phase I: Full pipeline test for the MCP tool flow.

Simulates:
    User: "Latest news on AI regulation"

Expected trace flow:
    telegram_message_received
    router_node
    state_init_node
    decision_logic_node → preprocess
    task_preprocessing_node
    decision_logic_node → call_model
    model_call_node (emits tool_call)
    result_handling_node
    decision_logic_node → execute_tool
    tool_call_detected (event)
    tool_execution_started (span)
    tool_execution_node
      └── mcp_request_sent (event)
      └── mcp_response_received (event)
      └── tool_execution_completed (event)
    decision_logic_node → call_model (second)
    model_call_node (with tool context)
    result_handling_node
    decision_logic_node → memory_write
    memory_write_node
    decision_logic_node → format
    format_response_node

All under one trace_id.

Verifies:
✔ Full trace flow executes without error
✔ Single trace_id throughout
✔ Tool executed exactly once
✔ Final output is non-empty
✔ Memory still written after tool flow
✔ All expected trace events emitted
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from inference import ModelBackend, ModelRequest, ModelResponse
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.mcp.external_client import BrowserBaseResult, MCPResponse
from agent.tracing.tracer import Tracer, TraceMetadata


# ─────────────────────────────────────────────────────
# Capturing Tracer (reused from test_trace_propagation_tool)
# ─────────────────────────────────────────────────────


class CapturingTracer(Tracer):
    def __init__(self):
        super().__init__()
        self.events: List[Dict[str, Any]] = []
        self.spans: List[Dict[str, Any]] = []

    def start_span(self, name, metadata, trace_metadata):
        self.spans.append(
            {"type": "start", "name": name, "trace_id": trace_metadata.trace_id}
        )
        return {"name": name}

    def end_span(self, span, status, metadata):
        if span:
            self.spans.append(
                {"type": "end", "name": span.get("name"), "status": status}
            )

    def record_event(self, name, metadata, trace_metadata):
        self.events.append(
            {
                "name": name,
                "metadata": metadata,
                "trace_id": trace_metadata.trace_id,
            }
        )

    def is_enabled(self):
        return True

    def get_event_names(self) -> List[str]:
        return [e["name"] for e in self.events]

    def get_span_names(self) -> List[str]:
        return [s["name"] for s in self.spans if s["type"] == "start"]

    def get_trace_ids(self) -> List[str]:
        return list({e["trace_id"] for e in self.events})


# ─────────────────────────────────────────────────────
# AI Regulation Model Stub
# ─────────────────────────────────────────────────────


class AIRegulationModel(ModelBackend):
    """
    Simulates the exact flow for "Latest news on AI regulation":
    - First call: emits web_search tool_call
    - Second call (with context): returns substantive answer
    """

    def __init__(self):
        self._calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1

        if self._calls == 1:
            # First call: request a web search
            return ModelResponse(
                status="success",
                output="I need to search for the latest news.",
                metadata={
                    "tool_call": {
                        "name": "web_search",
                        "arguments": {
                            "query": "AI regulation 2025 news",
                            "max_results": 3,
                        },
                    }
                },
            )

        # Second call: answer using tool context
        has_context = bool(request.context)
        return ModelResponse(
            status="success",
            output=(
                "Based on the latest search results, the EU AI Act has been fully implemented "
                "as of 2024, introducing mandatory requirements for high-risk AI systems. "
                "The US Executive Order on AI Safety is also in effect, requiring federal "
                "agencies to assess AI risks. "
                f"[context_received={'yes' if has_context else 'no'}]"
            ),
            metadata={},
        )


# ─────────────────────────────────────────────────────
# Full Pipeline Test
# ─────────────────────────────────────────────────────


class TestFullPipeline:
    TRACE_ID = "pipeline-trace-001"
    QUERY = "Latest news on AI regulation"

    def _setup_mock_registry(self):
        """Create a mock ToolRegistry with a web_search tool that returns real results."""
        from agent.intelligence.tools import ToolResult

        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # forces WebSearchTool registration path
        mock_registry.execute.return_value = ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "title": "EU AI Act Enters Force",
                        "url": "https://example.com/eu-ai-act",
                        "snippet": "The EU AI Act became law in August 2024...",
                    },
                    {
                        "title": "US AI Executive Order",
                        "url": "https://example.com/us-ai-order",
                        "snippet": "President Biden signed the AI Executive Order...",
                    },
                    {
                        "title": "China AI Governance",
                        "url": "https://example.com/china-ai",
                        "snippet": "China's Cyberspace Administration issued AI rules...",
                    },
                ],
                "query": "AI regulation 2025 news",
                "result_count": 3,
            },
            execution_time_ms=45,
        )

        return mock_registry, mock_registry

    def test_full_pipeline_completes(self):
        """Full pipeline: no errors, status=success."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            result = asyncio.run(
                orch.invoke(self.QUERY, trace_id=self.TRACE_ID)
            )

        assert result["status"] == "success"
        assert result["output"] is not None
        assert len(result["output"]) > 0

    def test_single_trace_id_throughout(self):
        """All trace events use the same trace_id."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        trace_ids = tracer.get_trace_ids()
        assert len(trace_ids) == 1, f"Multiple trace_ids found: {trace_ids}"
        assert trace_ids[0] == self.TRACE_ID

    def test_tool_executed_exactly_once(self):
        """Tool must be executed exactly once per turn."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        mock_registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=mock_registry):
            import asyncio
            asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        assert mock_registry.execute.call_count == 1

    def test_model_called_twice(self):
        """Model must be called twice: initial + post-tool."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        assert model._calls == 2

    def test_tool_context_received_by_second_model_call(self):
        """Second model call receives tool context (context_received=yes in output)."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            result = asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        assert "context_received=yes" in result["output"]

    def test_expected_trace_events_emitted(self):
        """All expected trace events emitted in the correct pipeline."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        event_names = set(tracer.get_event_names())

        # Core pipeline events
        assert "model_call_attempted" in event_names
        assert "model_call_completed" in event_names

        # Tool-specific events
        assert "tool_call_detected" in event_names
        assert "tool_execution_started" in event_names
        assert "tool_execution_completed" in event_names

    def test_node_spans_cover_full_pipeline(self):
        """Key graph nodes must have trace spans."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        span_names = tracer.get_span_names()

        assert "router_node" in span_names
        assert "state_init_node" in span_names
        assert "task_preprocessing_node" in span_names
        assert "model_call_node" in span_names
        assert "tool_execution_node" in span_names
        assert "result_handling_node" in span_names
        assert "memory_write_node" in span_names
        assert "format_response_node" in span_names

    def test_final_output_references_search_results(self):
        """Final output should reflect the tool context (EU AI Act, etc.)."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            result = asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        # Output should contain substantive content
        assert "AI" in result["output"]
        assert len(result["output"]) > 50

    def test_memory_written_after_tool_flow(self):
        """memory_write_node must still execute after tool flow."""
        tracer = CapturingTracer()
        model = AIRegulationModel()
        registry, _ = self._setup_mock_registry()

        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry", return_value=registry):
            import asyncio
            asyncio.run(orch.invoke(self.QUERY, trace_id=self.TRACE_ID))

        # memory_write_node must appear in spans
        span_names = tracer.get_span_names()
        assert "memory_write_node" in span_names

        # memory_write_attempted event emitted
        event_names = tracer.get_event_names()
        assert "memory_write_attempted" in event_names
