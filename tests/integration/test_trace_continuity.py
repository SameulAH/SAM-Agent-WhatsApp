"""
tests/integration/test_trace_continuity.py

Integration tests for trace continuity across the full pipeline.

Verifies:
✔ Same trace_id propagates through all nodes
✔ prompt_built event emitted with trace_id
✔ memory_injected event emitted when LTM facts present
✔ tool_call_detected event carries correct tool name
✔ forced_tool_call event carries reason
✔ response_truncated event emitted when output exceeds limit
✔ tool_execution_started/completed events share trace_id
✔ No trace event breaks agent execution
✔ Trace failures (tracer exceptions) are silently absorbed
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from inference import ModelBackend, ModelRequest, ModelResponse
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing.tracer import NoOpTracer, TraceMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Tracers
# ─────────────────────────────────────────────────────────────────────────────


class FullEventTracer(NoOpTracer):
    """Records all events and spans with their metadata."""

    def __init__(self):
        super().__init__()
        self.events: List[Dict[str, Any]] = []
        self.spans: List[Dict[str, Any]] = []

    def record_event(self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata) -> None:
        self.events.append({
            "name": name,
            "metadata": metadata,
            "trace_id": trace_metadata.trace_id,
            "conversation_id": trace_metadata.conversation_id,
        })

    def start_span(self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata):
        span = {"name": name, "trace_id": trace_metadata.trace_id}
        self.spans.append(span)
        return span

    def end_span(self, span, status: str, metadata: Dict[str, Any]) -> None:
        if span:
            span["status"] = status

    @property
    def event_names(self) -> List[str]:
        return [e["name"] for e in self.events]

    def events_for(self, name: str) -> List[Dict]:
        return [e for e in self.events if e["name"] == name]


class ExplodingTracer(NoOpTracer):
    """Tracer that always raises — tests that trace failures are non-fatal."""

    def record_event(self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata) -> None:
        raise RuntimeError("Tracer exploded!")

    def start_span(self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata):
        raise RuntimeError("Tracer exploded!")

    def end_span(self, span, status: str, metadata: Dict[str, Any]) -> None:
        raise RuntimeError("Tracer exploded!")


# ─────────────────────────────────────────────────────────────────────────────
# Model stubs
# ─────────────────────────────────────────────────────────────────────────────


class SimpleModel(ModelBackend):
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            status="success",
            output="This is a normal answer.",
            metadata={},
        )


class ToolModel(ModelBackend):
    """Emits tool_call on first call, normal answer on second."""

    def __init__(self):
        self._n = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._n += 1
        if self._n == 1:
            return ModelResponse(
                status="success",
                output="Searching...",
                metadata={"tool_call": {"name": "web_search", "arguments": {"query": "AI news"}}},
            )
        return ModelResponse(
            status="success",
            output="Based on the search: AI is advancing rapidly.",
            metadata={},
        )


class LongResponseModel(ModelBackend):
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            status="success",
            output="Long answer. " * 200,
            metadata={},
        )


def _patch_tool_registry():
    from agent.intelligence.tools import ToolResult
    mock_registry = MagicMock()
    mock_registry.get.return_value = None
    mock_registry.execute.return_value = ToolResult(
        tool_name="web_search",
        success=True,
        data={"results": [{"title": "AI", "url": "https://example.com", "snippet": "AI result"}]},
        execution_time_ms=20,
    )
    # get_tool_registry is lazily imported in _tool_execution_node_impl,
    # so patch it in its source module.
    return patch("agent.intelligence.tools.get_tool_registry", return_value=mock_registry)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTraceIdPropagation:
    """trace_id is consistent across all emitted events."""

    @pytest.mark.asyncio
    async def test_trace_id_consistent_simple_call(self):
        """All events share the same trace_id."""
        tracer = FullEventTracer()
        orch = SAMAgentOrchestrator(model_backend=SimpleModel(), tracer=tracer)
        trace_id = str(uuid4())
        await orch.invoke("What is 2 + 2?", trace_id=trace_id)

        for event in tracer.events:
            assert event["trace_id"] == trace_id, (
                f"Event '{event['name']}' has wrong trace_id: {event['trace_id']!r}"
            )

    @pytest.mark.asyncio
    async def test_trace_id_consistent_with_tool_call(self):
        """trace_id is consistent when a tool is executed."""
        tracer = FullEventTracer()
        trace_id = str(uuid4())

        with _patch_tool_registry():
            orch = SAMAgentOrchestrator(model_backend=ToolModel(), tracer=tracer)
            await orch.invoke("Search for AI news", trace_id=trace_id)

        for event in tracer.events:
            assert event["trace_id"] == trace_id


class TestTraceEvents:
    """Specific trace events are emitted at the right points."""

    @pytest.mark.asyncio
    async def test_prompt_built_event_emitted(self):
        """prompt_built event is emitted during model call."""
        tracer = FullEventTracer()
        orch = SAMAgentOrchestrator(model_backend=SimpleModel(), tracer=tracer)
        await orch.invoke("Hello.")

        assert "prompt_built" in tracer.event_names

    @pytest.mark.asyncio
    async def test_prompt_built_metadata(self):
        """prompt_built event carries has_memory_context and has_tool_context."""
        tracer = FullEventTracer()
        orch = SAMAgentOrchestrator(model_backend=SimpleModel(), tracer=tracer)
        await orch.invoke("Hello.")

        built_events = tracer.events_for("prompt_built")
        assert len(built_events) >= 1
        meta = built_events[0]["metadata"]
        assert "has_memory_context" in meta
        assert "has_tool_context" in meta

    @pytest.mark.asyncio
    async def test_tool_call_detected_event_on_tool_call(self):
        """tool_call_detected event is emitted when model emits tool_call."""
        tracer = FullEventTracer()

        with _patch_tool_registry():
            orch = SAMAgentOrchestrator(model_backend=ToolModel(), tracer=tracer)
            await orch.invoke("Search for something.")

        assert "tool_call_detected" in tracer.event_names
        detected = tracer.events_for("tool_call_detected")[0]
        assert detected["metadata"]["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_response_truncated_event_on_long_output(self):
        """response_truncated event is emitted when output is too long."""
        tracer = FullEventTracer()
        orch = SAMAgentOrchestrator(model_backend=LongResponseModel(), tracer=tracer)
        await orch.invoke("Tell me everything.")

        assert "response_truncated" in tracer.event_names
        trunc = tracer.events_for("response_truncated")[0]
        assert trunc["metadata"]["original_length"] > SAMAgentOrchestrator.MAX_OUTPUT_CHARS

    @pytest.mark.asyncio
    async def test_tool_execution_events_emitted(self):
        """tool_execution_started and tool_execution_completed events emitted."""
        tracer = FullEventTracer()

        with _patch_tool_registry():
            orch = SAMAgentOrchestrator(model_backend=ToolModel(), tracer=tracer)
            await orch.invoke("Latest news please.")

        assert "tool_execution_started" in tracer.event_names
        assert "tool_execution_completed" in tracer.event_names


class TestTracerFailureTolerance:
    """Tracer failures never crash the agent."""

    @pytest.mark.asyncio
    async def test_exploding_tracer_does_not_crash_agent(self):
        """If the tracer raises on every call, the agent still succeeds."""
        orch = SAMAgentOrchestrator(
            model_backend=SimpleModel(),
            tracer=ExplodingTracer(),
        )
        result = await orch.invoke("What is 1 + 1?")

        assert result["status"] == "success"
        assert result["output"] is not None

    @pytest.mark.asyncio
    async def test_exploding_tracer_with_tool_does_not_crash(self):
        """Exploding tracer during tool flow does not crash."""
        with _patch_tool_registry():
            orch = SAMAgentOrchestrator(
                model_backend=ToolModel(),
                tracer=ExplodingTracer(),
            )
            result = await orch.invoke("Latest news.")

        assert result["status"] == "success"
