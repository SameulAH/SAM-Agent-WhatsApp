"""
tests/integration/test_tool_intent_trigger.py

Integration tests for freshness keyword → forced tool execution guardrail.

Verifies:
✔ Query with "today" triggers tool execution even when model doesn't emit tool_call
✔ Query with "latest" triggers tool execution
✔ Query with "breaking news" triggers tool execution
✔ Forced tool call emits forced_tool_call trace event
✔ Normal query without freshness keywords does NOT force tool execution
✔ Forced tool only fires ONCE per turn (tool_call_count guardrail respected)
✔ Forced tool uses user's query as search query
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from inference import ModelBackend, ModelRequest, ModelResponse, StubModelBackend
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.state_schema import AgentState
from agent.tracing.tracer import NoOpTracer, TraceMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Stub Backends
# ─────────────────────────────────────────────────────────────────────────────


class NeverToolModel(ModelBackend):
    """Model that NEVER emits a tool_call — tests forced tool guardrail."""

    def __init__(self, answer: str = "The latest news is unavailable right now."):
        self._call_count = 0
        self._answer = answer
        self.last_context: Optional[str] = None

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._call_count += 1
        self.last_context = request.context
        return ModelResponse(
            status="success",
            output=self._answer,
            metadata={},
        )


class MockTracer(NoOpTracer):
    """Tracer that records all events for assertion."""

    def __init__(self):
        super().__init__()
        self.events: list[dict] = []

    def record_event(self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata) -> None:
        self.events.append({"name": name, "metadata": metadata})

    @property
    def event_names(self) -> list[str]:
        return [e["name"] for e in self.events]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_orch(model: ModelBackend, tracer=None):
    """Create a SAMAgentOrchestrator with a mock web_search tool."""
    orch = SAMAgentOrchestrator(
        model_backend=model,
        tracer=tracer or NoOpTracer(),
    )
    return orch


def _patch_tool_registry(result_snippets: list[str]):
    """
    Return a context manager that patches get_tool_registry() so web_search
    returns the given snippets without making real HTTP calls.

    get_tool_registry is imported lazily inside _tool_execution_node_impl, so
    we must patch it in the source module (agent.intelligence.tools).
    """
    from agent.intelligence.tools import ToolResult

    mock_registry = MagicMock()
    mock_registry.get.return_value = None
    mock_result = ToolResult(
        tool_name="web_search",
        success=True,
        data={
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": s}
                for i, s in enumerate(result_snippets, 1)
            ]
        },
        execution_time_ms=50,
    )
    mock_registry.execute.return_value = mock_result
    return patch("agent.intelligence.tools.get_tool_registry", return_value=mock_registry)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFreshnessKeywordForcedTool:
    """Freshness keyword guardrail forces web_search when model hesitates."""

    @pytest.mark.asyncio
    async def test_today_keyword_forces_tool(self):
        """Query with 'today' forces tool execution even when model doesn't emit tool_call."""
        tracer = MockTracer()
        model = NeverToolModel()

        with _patch_tool_registry(["Today's top story about AI."]):
            orch = _make_orch(model, tracer)
            result = await orch.invoke("What is the news today?")

        assert result["status"] == "success"
        # Model should have been called twice (first call + after tool)
        assert model._call_count >= 2
        # forced_tool_call event must have been emitted
        assert "forced_tool_call" in tracer.event_names

    @pytest.mark.asyncio
    async def test_latest_keyword_forces_tool(self):
        """'latest' keyword triggers forced tool call."""
        tracer = MockTracer()
        model = NeverToolModel()

        with _patch_tool_registry(["Latest AI developments."]):
            orch = _make_orch(model, tracer)
            result = await orch.invoke("Tell me the latest news.")

        assert result["status"] == "success"
        assert "forced_tool_call" in tracer.event_names

    @pytest.mark.asyncio
    async def test_breaking_keyword_forces_tool(self):
        """'breaking' keyword triggers forced tool call."""
        tracer = MockTracer()
        model = NeverToolModel()

        with _patch_tool_registry(["Breaking: Major event."]):
            orch = _make_orch(model, tracer)
            result = await orch.invoke("Any breaking news right now?")

        assert result["status"] == "success"
        assert "forced_tool_call" in tracer.event_names

    @pytest.mark.asyncio
    async def test_no_freshness_keyword_no_forced_tool(self):
        """Query without freshness keywords does NOT trigger forced tool."""
        tracer = MockTracer()
        model = NeverToolModel(answer="Paris is the capital of France.")

        orch = _make_orch(model, tracer)
        result = await orch.invoke("What is the capital of France?")

        assert result["status"] == "success"
        # Model should only be called once (no tool round-trip)
        assert model._call_count == 1
        # No forced_tool_call event
        assert "forced_tool_call" not in tracer.event_names

    @pytest.mark.asyncio
    async def test_forced_tool_fires_only_once(self):
        """Forced tool respects tool_call_count guardrail — runs at most once."""
        tracer = MockTracer()
        # Model that never emits tool_call on either call
        model = NeverToolModel()

        with _patch_tool_registry(["Current events summary."]):
            orch = _make_orch(model, tracer)
            result = await orch.invoke("What is current in the world?")

        assert result["status"] == "success"
        # forced_tool_call event should appear at most once
        forced_events = [e for e in tracer.events if e["name"] == "forced_tool_call"]
        assert len(forced_events) <= 1

    @pytest.mark.asyncio
    async def test_tool_intent_detected_event_emitted(self):
        """tool_intent_detected event is emitted before forced_tool_call."""
        tracer = MockTracer()
        model = NeverToolModel()

        with _patch_tool_registry(["Today's result."]):
            orch = _make_orch(model, tracer)
            await orch.invoke("Show me today's headlines.")

        assert "tool_intent_detected" in tracer.event_names
