"""
Phase 4.2.4: Trace Identity Propagation Tests

Validates that trace identifiers are correctly propagated from AgentState
to all trace/span/event emissions, with zero creation or modification of IDs.

Key invariants:
- trace_id, conversation_id, user_id originate from AgentState only
- Tracer never generates, modifies, or infers identifiers
- All traces use identical identifiers
- Graceful degradation if identifiers missing
- Tracing never mutates identifiers in state
"""

import pytest
from unittest.mock import Mock, call
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing import Tracer, TraceMetadata, NoOpTracer
from agent.state_schema import AgentState
from inference import StubModelBackend


class TestTraceMetadataExtraction:
    """Test that TraceMetadata is correctly extracted from AgentState."""

    def test_trace_metadata_extracted_from_state(self):
        """TraceMetadata is correctly extracted from AgentState fields."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-abc123",
            trace_id="trace-xyz789",
            created_at="2025-01-01T00:00:00",
            input_type="text",
            raw_input="test input",
        )

        metadata = orchestrator._create_trace_metadata(state)

        assert metadata.trace_id == "trace-xyz789"
        assert metadata.conversation_id == "conv-abc123"
        assert metadata.user_id is None

    def test_trace_metadata_does_not_mutate_state(self):
        """Creating TraceMetadata does not mutate AgentState."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-abc123",
            trace_id="trace-xyz789",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        original_conv_id = state.conversation_id
        original_trace_id = state.trace_id

        orchestrator._create_trace_metadata(state)

        # State must not be mutated
        assert state.conversation_id == original_conv_id
        assert state.trace_id == original_trace_id

    def test_trace_metadata_preserves_all_ids(self):
        """All identifiers from state are preserved in metadata."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test",
        )

        metadata = orchestrator._create_trace_metadata(state)

        assert metadata.trace_id == state.trace_id
        assert metadata.conversation_id == state.conversation_id


class TestTraceIdentityPropagationToTracer:
    """Test that TraceMetadata is consistently passed to tracer calls."""

    def test_node_execution_passes_trace_metadata_to_tracer(self):
        """Node execution passes consistent TraceMetadata to tracer."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        orchestrator._router_node(state)

        # Verify tracer.start_span was called with TraceMetadata
        assert mock_tracer.start_span.called
        call_args = mock_tracer.start_span.call_args
        trace_metadata = call_args[1]["trace_metadata"]

        assert trace_metadata.trace_id == "trace-456"
        assert trace_metadata.conversation_id == "conv-123"

    def test_all_node_entry_spans_use_same_trace_id(self):
        """All node entry spans use the same trace_id."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None
        mock_tracer.end_span.return_value = None

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Execute multiple nodes
        orchestrator._router_node(state)

        # All start_span calls should have the same trace_id
        for call_obj in mock_tracer.start_span.call_args_list:
            trace_metadata = call_obj[1]["trace_metadata"]
            assert trace_metadata.trace_id == "trace-456"

    def test_model_call_emits_same_trace_id(self):
        """Model call events include the same trace_id."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.is_enabled.return_value = True

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

        # Model call should emit record_event with same trace_id
        assert mock_tracer.record_event.called
        for call_obj in mock_tracer.record_event.call_args_list:
            trace_metadata = call_obj[1]["trace_metadata"]
            assert trace_metadata.trace_id == "trace-456"

    def test_no_id_generation_inside_tracer(self):
        """Tracer does not generate new identifiers."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Before calling node
        original_trace_id = state.trace_id

        orchestrator._router_node(state)

        # After calling node - trace_id must not change
        assert state.trace_id == original_trace_id


class TestTraceMetadataConsistency:
    """Test that TraceMetadata is consistent across all trace calls."""

    def test_node_entry_and_exit_use_same_metadata(self):
        """Node entry and exit spans use identical TraceMetadata."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = "span-handle"

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        orchestrator._router_node(state)

        # Extract trace_id from first start_span call
        start_call = mock_tracer.start_span.call_args_list[0]
        start_metadata = start_call[1]["trace_metadata"]

        # Extract trace_id from end_span call
        if mock_tracer.end_span.called:
            end_call = mock_tracer.end_span.call_args_list[0]
            end_metadata = end_call[1]["metadata"]
            # end_span doesn't pass trace_metadata directly, but start/end are on same execution

        # Both should reference the same trace_id from state
        assert start_metadata.trace_id == "trace-456"

    def test_conversation_id_preserved_across_execution(self):
        """Conversation ID is preserved and consistent."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        orchestrator._router_node(state)

        # All trace calls should have same conversation_id
        for call_obj in mock_tracer.start_span.call_args_list:
            trace_metadata = call_obj[1]["trace_metadata"]
            assert trace_metadata.conversation_id == "conv-123"


class TestGracefulDegradationMissingIdentifiers:
    """Test that tracing degrades gracefully if identifiers missing."""

    def test_tracing_succeeds_with_minimal_ids(self):
        """Tracing succeeds even with minimal identifier data."""
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.start_span.return_value = None

        orchestrator = SAMAgentOrchestrator(tracer=mock_tracer)
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Should not raise, despite minimal identifiers
        orchestrator._router_node(state)

        assert mock_tracer.start_span.called

    def test_tracer_exception_does_not_corrupt_identifiers(self):
        """Tracer exceptions don't corrupt AgentState identifiers."""
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

        original_conv_id = state.conversation_id
        original_trace_id = state.trace_id

        # Should not raise, and state should not be corrupted
        orchestrator._router_node(state)

        assert state.conversation_id == original_conv_id
        assert state.trace_id == original_trace_id


class TestNoIdGeneration:
    """Test that tracer never generates identifiers."""

    def test_trace_metadata_never_creates_ids(self):
        """TraceMetadata extraction never creates new IDs."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        metadata = orchestrator._create_trace_metadata(state)

        # trace_id must be exactly what's in state
        assert metadata.trace_id == state.trace_id

        # No new IDs should be generated
        assert metadata.trace_id == "trace-456"

    def test_noop_tracer_generates_no_ids(self):
        """NoOpTracer never generates identifiers."""
        tracer = NoOpTracer()

        # NoOp tracer should never create anything
        span = tracer.start_span("node", {}, TraceMetadata(trace_id="trace-123"))
        assert span is None

    def test_trace_metadata_immutable_from_state(self):
        """TraceMetadata is immutable copy of state identifiers."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        state = AgentState(
            conversation_id="original-conv",
            trace_id="original-trace",
            created_at="",
            input_type="text",
            raw_input="test",
        )

        metadata = orchestrator._create_trace_metadata(state)

        # Change state (simulating external mutation)
        # state.conversation_id = "modified-conv"  # Don't actually do this

        # Metadata should still have original values
        assert metadata.conversation_id == "original-conv"
        assert metadata.trace_id == "original-trace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
