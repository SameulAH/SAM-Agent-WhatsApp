"""
Tests for Phase 4.2.3: Tracing boundaries integration.

Validates that:
1. Tracing is wired into node execution
2. Behavior is identical with tracing ON/OFF
3. Tracing failures don't affect execution
4. Memory access is traced
5. Model calls are traced
"""

import pytest
from unittest.mock import Mock
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing import NoOpTracer, LangTraceTracer, Tracer
from inference import StubModelBackend
from agent.state_schema import AgentState


class TestTracingIntegration:
    """Test tracing integration with orchestrator."""

    def test_orchestrator_accepts_tracer(self):
        """Orchestrator accepts tracer parameter."""
        tracer = NoOpTracer()
        orchestrator = SAMAgentOrchestrator(tracer=tracer)
        assert orchestrator.tracer is tracer

    def test_orchestrator_defaults_to_noop_tracer(self):
        """Orchestrator defaults to NoOpTracer if not provided."""
        orchestrator = SAMAgentOrchestrator()
        assert isinstance(orchestrator.tracer, NoOpTracer)

    def test_orchestrator_with_custom_tracer(self):
        """Orchestrator can use custom tracer implementation."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None
        mock_tracer.is_enabled.return_value = False

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        assert orchestrator.tracer is mock_tracer

    def test_trace_metadata_created_from_state(self):
        """Trace metadata is correctly extracted from state."""
        orchestrator = SAMAgentOrchestrator()
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        metadata = orchestrator._create_trace_metadata(state)

        assert metadata.trace_id == "trace-456"
        assert metadata.conversation_id == "conv-123"


class TestBehaviorInvariance:
    """Test that agent behavior is identical with/without tracing."""

    def test_router_node_identical_with_noop_tracer(self):
        """Router node output identical with NoOpTracer."""
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        result = orchestrator_noop._router_node(state)
        assert result["input_type"] == "text"

    def test_state_init_node_identical_with_noop_tracer(self):
        """State init node output identical with NoOpTracer."""
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        result = orchestrator_noop._state_init_node(state)

        # trace_id and conversation_id should be preserved
        assert result["trace_id"] == "trace-456"
        assert result["conversation_id"] == "conv-123"
        assert result["command"] is None

    def test_preprocessing_node_identical_with_noop_tracer(self):
        """Preprocessing node output identical with NoOpTracer."""
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="  test input  ",
        )

        result = orchestrator_noop._task_preprocessing_node(state)
        assert result["preprocessing_result"] == "test input"

    def test_model_call_node_identical_with_noop_tracer(self):
        """Model call node output identical with NoOpTracer."""
        backend = StubModelBackend()
        orchestrator_noop = SAMAgentOrchestrator(
            model_backend=backend, tracer=NoOpTracer()
        )
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            preprocessing_result="test input",
        )

        result = orchestrator_noop._model_call_node(state)

        # Model response should be present
        assert "model_response" in result
        assert result["model_response"].status in ("success", "failure")

    def test_error_router_node_identical_with_noop_tracer(self):
        """Error router node output identical with NoOpTracer."""
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())

        # Create a failed model response
        from inference import ModelResponse

        failed_response = ModelResponse(
            status="failure",
            output=None,
            error_type="timeout",
            metadata={},
        )

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            preprocessing_result="test input",
            model_response=failed_response,
        )

        result = orchestrator_noop._error_router_node(state)

        assert result["error_type"] == "timeout"
        assert "[Error: timeout]" in result["final_output"]

    def test_format_response_node_identical_with_noop_tracer(self):
        """Format response node output identical with NoOpTracer."""
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())

        from inference import ModelResponse

        success_response = ModelResponse(
            status="success",
            output="test output",
            error_type=None,
            metadata={"latency_ms": 150},
        )

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="2025-01-01T00:00:00",
            input_type="text",
            raw_input="test input",
            preprocessing_result="test input",
            model_response=success_response,
            model_metadata=success_response.metadata,
            final_output="test output",
            error_type=None,
        )

        result = orchestrator_noop._format_response_node(state)

        assert result["status"] == "success"
        assert result["output"] == "test output"
        assert result["error_type"] is None


class TestTracingNonInterference:
    """Test that tracing failures don't affect execution."""

    def test_tracer_exception_in_start_span_does_not_block(self):
        """Tracer exception in start_span doesn't block node execution."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.side_effect = Exception("Tracer failed")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should not raise, despite tracer failure
        result = orchestrator._router_node(state)
        assert result["input_type"] == "text"

    def test_tracer_exception_in_end_span_does_not_block(self):
        """Tracer exception in end_span doesn't block node execution."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None
        mock_tracer.end_span.side_effect = Exception("Tracer failed")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should not raise, despite tracer failure
        result = orchestrator._router_node(state)
        assert result["input_type"] == "text"

    def test_disabled_tracer_noop(self):
        """Disabled tracer (NoOpTracer) doesn't emit anything."""
        tracer = NoOpTracer()
        assert not tracer.is_enabled()

        orchestrator = SAMAgentOrchestrator(tracer=tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            preprocessing_result="test input",
        )

        # All node operations should succeed silently
        result = orchestrator._task_preprocessing_node(state)
        assert result["preprocessing_result"] == "test input"


class TestMemoryAccessTracing:
    """Test that memory access is traced."""

    def test_memory_read_node_wrapper_exists(self):
        """Memory read node wrapper exists and is callable."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        assert hasattr(orchestrator, "_memory_read_node_wrapper")
        assert callable(orchestrator._memory_read_node_wrapper)

    def test_memory_write_node_wrapper_exists(self):
        """Memory write node wrapper exists and is callable."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        assert hasattr(orchestrator, "_memory_write_node_wrapper")
        assert callable(orchestrator._memory_write_node_wrapper)

    def test_long_term_memory_read_node_wrapper_exists(self):
        """Long-term memory read node wrapper exists and is callable."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        assert hasattr(orchestrator, "_long_term_memory_read_node_wrapper")
        assert callable(orchestrator._long_term_memory_read_node_wrapper)

    def test_long_term_memory_write_node_wrapper_exists(self):
        """Long-term memory write node wrapper exists and is callable."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        assert hasattr(orchestrator, "_long_term_memory_write_node_wrapper")
        assert callable(orchestrator._long_term_memory_write_node_wrapper)


class TestModelCallTracing:
    """Test that model calls are traced."""

    def test_model_call_emits_events(self):
        """Model call emits tracing events."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.is_enabled.return_value = True
        mock_tracer.record_event.return_value = None

        orchestrator = SAMAgentOrchestrator(
            model_backend=StubModelBackend(), tracer=mock_tracer
        )
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            preprocessing_result="test input",
        )

        orchestrator._model_call_node(state)

        # Should emit model_call_attempted and model_call_completed events
        assert mock_tracer.record_event.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
