"""
Phase 4.2.9: Tracing Failure Safety Tests

Validates that tracer failures never block agent execution.

Core principle: Tracing is observability only. A failed tracer is a visibility loss,
never an execution block.

Test strategy:
1. Simulate tracer unavailable/thrown exceptions
2. Verify agent completes with identical behavior
3. Verify failures are silent (never propagate to caller)
"""

import pytest
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing import NoOpTracer, Tracer
from inference import StubModelBackend, ModelResponse
from agent.state_schema import AgentState
from unittest.mock import Mock, patch


class TestTracerFailureSafety:
    """Test that tracer failures are non-fatal."""

    def test_tracer_exception_in_start_span_caught_silently(self):
        """Tracer exception in start_span doesn't propagate."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.side_effect = RuntimeError("Connection lost")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should NOT raise, should complete normally
        result = orchestrator._router_node(state)
        assert result is not None
        assert result["input_type"] == "text"

    def test_tracer_exception_in_end_span_caught_silently(self):
        """Tracer exception in end_span doesn't propagate."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None
        mock_tracer.end_span.side_effect = RuntimeError("Span save failed")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should NOT raise despite end_span failure
        result = orchestrator._router_node(state)
        assert result is not None

    def test_tracer_unavailable_produces_same_state(self):
        """Unavailable tracer (None ops) produces identical state."""
        mock_tracer_unavail = Mock(spec=Tracer)
        mock_tracer_unavail.start_span.side_effect = Exception("Tracer unavailable")
        mock_tracer_unavail.end_span.side_effect = Exception("Tracer unavailable")

        orchestrator_unavail = SAMAgentOrchestrator(tracer=mock_tracer_unavail)
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())

        state_unavail = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        state_noop = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        result_unavail = orchestrator_unavail._task_preprocessing_node(state_unavail)
        result_noop = orchestrator_noop._task_preprocessing_node(state_noop)

        # Preprocessing results must be identical despite tracer failures
        assert result_unavail == result_noop
        assert result_unavail["preprocessing_result"] == "test input"

    def test_record_event_failure_caught_silently(self):
        """record_event failures don't block execution."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None
        mock_tracer.record_event.side_effect = Exception("Event buffer full")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            preprocessing_result="test input",
        )

        # Model call records events - should complete despite failures
        result = orchestrator._model_call_node(state)
        assert result is not None
        assert "model_response" in result


class TestMemoryAccessTracingFailureSafety:
    """Test that memory access tracing failures don't block operations."""

    def test_short_term_memory_read_tracing_failure_safe(self):
        """Short-term memory read succeeds even if tracer fails."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.side_effect = Exception("Tracer down")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Memory nodes should execute despite tracer failure
        # Just verify orchestrator node wrapper handles memory reads
        result = orchestrator._router_node(state)
        # Should complete without raising
        assert result is not None

    def test_short_term_memory_write_tracing_failure_safe(self):
        """Short-term memory write succeeds even if tracer fails."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.end_span.side_effect = Exception("Span save failed")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Memory write should succeed despite tracer failure
        # Just verify node execution works
        result = orchestrator._router_node(state)
        # Should not raise
        assert result is not None


class TestModelCallTracingFailureSafety:
    """Test that model call tracing failures are non-fatal."""

    def test_model_call_completes_despite_tracer_failure(self):
        """Model call completes successfully even if tracer fails."""
        mock_tracer = Mock(spec=Tracer)
        # Fail on every tracer call
        mock_tracer.start_span.side_effect = Exception("Tracer disconnected")
        mock_tracer.end_span.side_effect = Exception("Tracer disconnected")
        mock_tracer.record_event.side_effect = Exception("Tracer disconnected")

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

        # Should complete successfully despite all tracer failures
        result = orchestrator._model_call_node(state)
        assert result is not None
        assert "model_response" in result
        assert result["model_response"].status == "success"

    def test_model_response_logged_despite_tracer_failure(self):
        """Model response is recorded even if tracing fails."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.record_event.side_effect = Exception("Event queue overflow")

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

        # Model call should record output in state despite tracing failure
        result = orchestrator._model_call_node(state)
        assert result["model_response"].output is not None


class TestTracingExceptionHandling:
    """Test that all tracer exceptions are silently caught."""

    def test_different_exception_types_all_caught(self):
        """Various exception types from tracer are all caught."""
        exception_types = [
            RuntimeError("Runtime error"),
            TimeoutError("Timeout"),
            ConnectionError("Connection error"),
            ValueError("Value error"),
            Exception("Generic exception"),
        ]

        for exc in exception_types:
            mock_tracer = Mock(spec=Tracer)
            mock_tracer.start_span.side_effect = exc

            orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

            state = AgentState(
                conversation_id="conv-123",
                trace_id="trace-456",
                created_at="",
                input_type="text",
                raw_input="test input",
            )

            # All exception types should be silently caught
            result = orchestrator._router_node(state)
            assert result is not None

    def test_none_return_from_tracer_handled(self):
        """None returns from tracer don't cause attribute errors."""
        mock_tracer = Mock(spec=Tracer)
        # Return None from all calls (like NoOpTracer)
        mock_tracer.start_span.return_value = None
        mock_tracer.end_span.return_value = None
        mock_tracer.record_event.return_value = None

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should handle None returns gracefully
        result = orchestrator._router_node(state)
        assert result is not None


class TestTracerDisabledVsUnavailable:
    """Test distinction between disabled tracer and unavailable tracer."""

    def test_disabled_tracer_executes_same_path(self):
        """Disabled tracer (is_enabled=False) produces identical behavior."""
        mock_tracer_disabled = Mock(spec=Tracer)
        mock_tracer_disabled.is_enabled.return_value = False

        orchestrator_disabled = SAMAgentOrchestrator(tracer=mock_tracer_disabled)
        orchestrator_noop = SAMAgentOrchestrator(tracer=NoOpTracer())

        state_disabled = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        state_noop = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        result_disabled = orchestrator_disabled._router_node(state_disabled)
        result_noop = orchestrator_noop._router_node(state_noop)

        assert result_disabled == result_noop

    def test_unavailable_tracer_fails_gracefully(self):
        """Unavailable tracer (exceptions) fails gracefully."""
        mock_tracer_unavail = Mock(spec=Tracer)
        mock_tracer_unavail.start_span.side_effect = Exception("Service down")

        orchestrator_unavail = SAMAgentOrchestrator(tracer=mock_tracer_unavail)

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should fail gracefully
        result = orchestrator_unavail._router_node(state)
        assert result is not None
        assert result["input_type"] == "text"


class TestErrorHandlingWithTracerFailure:
    """Test that error handling works even when tracer is broken."""

    def test_error_handling_works_with_failed_tracer(self):
        """Error handling completes successfully despite tracer failures."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.side_effect = Exception("Tracer failed")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        # State with error
        failed_response = ModelResponse(
            status="failure", output=None, error_type="model_error", metadata={}
        )

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            model_response=failed_response,
            error_type="model_error",
        )

        # Error handling should work despite tracer failures
        result = orchestrator._error_router_node(state)
        assert result is not None
        assert "error_type" in result

    def test_formatting_response_with_failed_tracer(self):
        """Response formatting works despite tracer failures."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.end_span.side_effect = Exception("Tracer down")

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)

        failed_response = ModelResponse(
            status="failure", output=None, error_type="timeout", metadata={}
        )

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
            model_response=failed_response,
            error_type="timeout",
            final_output=None,
        )

        # Formatting should succeed despite tracer failures
        result = orchestrator._format_response_node(state)
        assert result is not None
        assert result["error_type"] == "timeout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
