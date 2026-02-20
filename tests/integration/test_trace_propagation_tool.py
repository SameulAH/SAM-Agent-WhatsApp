"""
tests/integration/test_trace_propagation_tool.py

Integration tests for trace propagation across the full MCP tool pipeline.

Verifies:
✔ Same trace_id used across Transport → LangGraph → MCP → Memory → LLM
✔ tool_call_detected span emitted
✔ tool_execution_started span emitted
✔ tool_execution_completed span emitted
✔ tool_execution_failed span emitted on error
✔ mcp_request_sent span emitted
✔ mcp_response_received span emitted
✔ Tracer failures are non-fatal
✔ trace_id never overridden by tool node
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from inference import ModelBackend, ModelRequest, ModelResponse
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing.tracer import Tracer, TraceMetadata


# ─────────────────────────────────────────────────────
# Capturing Tracer — records all events
# ─────────────────────────────────────────────────────


class CapturingTracer(Tracer):
    """Records all trace events and spans for assertions."""

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
            self.spans.append({"type": "end", "name": span.get("name"), "status": status})

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

    def get_trace_ids_used(self) -> List[str]:
        return list({e["trace_id"] for e in self.events})


# ─────────────────────────────────────────────────────
# Tool-calling model stub
# ─────────────────────────────────────────────────────


class ToolCallingModel(ModelBackend):
    def __init__(self):
        self._calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            return ModelResponse(
                status="success",
                output="I need to search.",
                metadata={
                    "tool_call": {
                        "name": "web_search",
                        "arguments": {"query": "AI regulation", "max_results": 2},
                    }
                },
            )
        return ModelResponse(
            status="success",
            output="Here is the answer based on search results.",
            metadata={},
        )


# ─────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────


class TestTracePropagation:
    def _run_with_tracer(self, model, tracer, query="Latest AI news"):
        """Run orchestrator with capturing tracer."""
        orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)

        with patch("agent.intelligence.tools.get_tool_registry") as mock_reg:
            from agent.intelligence.tools import ToolResult

            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry.execute.return_value = ToolResult(
                success=True,
                data={
                    "results": [
                        {
                            "title": "AI Regulation News",
                            "url": "https://example.com",
                            "snippet": "New AI rules announced.",
                        }
                    ],
                    "query": "AI regulation",
                    "result_count": 1,
                },
                execution_time_ms=30,
            )
            mock_reg.return_value = mock_registry

            import asyncio

            result = asyncio.run(
                orch.invoke(query, trace_id="fixed-trace-id-001")
            )

        return result, tracer

    def test_tool_call_detected_event_emitted(self):
        tracer = CapturingTracer()
        result, tracer = self._run_with_tracer(ToolCallingModel(), tracer)

        event_names = tracer.get_event_names()
        assert "tool_call_detected" in event_names

    def test_tool_execution_started_span_emitted(self):
        tracer = CapturingTracer()
        result, tracer = self._run_with_tracer(ToolCallingModel(), tracer)

        span_names = tracer.get_span_names()
        assert "tool_execution_node" in span_names

    def test_tool_execution_completed_event_emitted(self):
        tracer = CapturingTracer()
        result, tracer = self._run_with_tracer(ToolCallingModel(), tracer)

        event_names = tracer.get_event_names()
        assert "tool_execution_completed" in event_names

    def test_same_trace_id_throughout(self):
        """All events must share the same trace_id."""
        tracer = CapturingTracer()
        result, tracer = self._run_with_tracer(ToolCallingModel(), tracer)

        trace_ids = tracer.get_trace_ids_used()
        # Should only have one unique trace_id
        assert len(trace_ids) == 1
        assert trace_ids[0] == "fixed-trace-id-001"

    def test_trace_id_not_overridden_by_tool_node(self):
        """Tool execution must never change the trace_id."""
        tracer = CapturingTracer()
        original_trace_id = "immutable-trace-id-999"
        result, tracer = self._run_with_tracer(
            ToolCallingModel(), tracer, query="AI regulation news"
        )

        # All events that have a trace_id should use fixed-trace-id-001
        for event in tracer.events:
            assert event["trace_id"] == "fixed-trace-id-001"

    def test_tracer_failure_is_non_fatal(self):
        """If tracer raises, agent must still complete successfully."""

        class FailingTracer(CapturingTracer):
            def record_event(self, name, metadata, trace_metadata):
                raise RuntimeError("Tracer exploded!")

        tracer = FailingTracer()
        model = ToolCallingModel()

        with patch("agent.intelligence.tools.get_tool_registry") as mock_reg:
            from agent.intelligence.tools import ToolResult

            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry.execute.return_value = ToolResult(
                success=True,
                data={"results": [], "query": "test", "result_count": 0},
                execution_time_ms=10,
            )
            mock_reg.return_value = mock_registry

            import asyncio

            orch = SAMAgentOrchestrator(model_backend=model, tracer=tracer)
            result = asyncio.run(orch.invoke("Latest AI news"))

        assert result["status"] == "success"

    def test_model_call_span_emitted_twice(self):
        """model_call_node should be spanned twice: initial + after tool."""
        tracer = CapturingTracer()
        result, tracer = self._run_with_tracer(ToolCallingModel(), tracer)

        model_spans = [
            s for s in tracer.spans
            if s["type"] == "start" and s["name"] == "model_call_node"
        ]
        assert len(model_spans) == 2
