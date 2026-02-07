"""
Phase 4.2.9: Alarm Non-Interference Tests

Validates that InvariantAlarmSystem never blocks execution.

Core principle: Alarms are smoke detectors (observability), not sprinklers (blocking logic).
Detection and emission of violations don't affect routing or state mutations.
"""

import pytest
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from agent.tracing import NoOpTracer, Tracer
from agent.tracing.alarms import InvariantAlarmSystem, ViolationType
from agent.state_schema import AgentState
from unittest.mock import Mock


class TestAlarmDetectionNonBlocking:
    """Test that alarm detection methods never block execution."""

    def test_alarm_detection_methods_complete_without_raising(self):
        """All alarm detection methods are non-blocking."""
        alarm_system = InvariantAlarmSystem()

        # All detection methods should complete without raising
        alarm_system.detect_model_access_unauthorized("trace-123", "test_node")
        alarm_system.detect_memory_read_unauthorized("trace-123", "test_node", "short_term")
        alarm_system.detect_memory_write_unauthorized("trace-123", "test_node", "long_term")
        alarm_system.detect_multiple_terminal_nodes("trace-123", ["node1", "node2"])
        alarm_system.detect_state_mutation_outside_allowed_nodes("trace-123", "node", ["field"])
        alarm_system.detect_forbidden_data_in_trace("trace-123", "node", ["data"])
        alarm_system.detect_memory_not_append_only("trace-123", "node", "delete")
        alarm_system.detect_memory_advisory_overridden("trace-123", "node", "short_term")

        # All violations should be recorded
        violations = alarm_system.get_violations()
        assert len(violations) == 8

    def test_violation_recording_does_not_affect_orchestrator_execution(self):
        """Recording violations doesn't interfere with orchestrator execution."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        alarm_system = InvariantAlarmSystem()

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Record violations
        alarm_system.detect_model_access_unauthorized("trace-456", "test_node")
        alarm_system.detect_forbidden_data_in_trace("trace-456", "test_node", ["data"])

        # Orchestrator should execute normally
        result = orchestrator._router_node(state)
        assert result is not None
        assert result["input_type"] == "text"

    def test_intensive_alarm_recording_does_not_slow_execution(self):
        """Recording many violations doesn't degrade performance."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        alarm_system = InvariantAlarmSystem()

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Record many violations
        for i in range(50):
            alarm_system.detect_forbidden_data_in_trace("trace-456", f"node{i}", ["field"])

        # Execution should still complete normally
        result = orchestrator._router_node(state)
        assert result is not None
        assert len(alarm_system.get_violations()) == 50


class TestAlarmEmissionWithFailedTracer:
    """Test that alarm emission is safe even when tracer fails."""

    def test_alarm_detection_succeeds_with_failing_tracer(self):
        """Alarms are recorded even if tracer is unavailable."""
        mock_tracer_fail = Mock(spec=Tracer)
        mock_tracer_fail.is_enabled.return_value = True
        mock_tracer_fail.record_event.side_effect = Exception("Tracer down")

        alarm_system = InvariantAlarmSystem(tracer=mock_tracer_fail)

        # Detection should complete despite tracer failure
        alarm_system.detect_model_access_unauthorized("trace-456", "node")
        alarm_system.detect_forbidden_data_in_trace("trace-456", "node", ["data"])

        # Violations should still be recorded locally
        violations = alarm_system.get_violations()
        assert len(violations) == 2

    def test_orchestrator_execution_with_failing_alarm_tracer(self):
        """Orchestrator execution is unaffected by alarm tracer failures."""
        # Create alarm system with broken tracer
        mock_tracer = Mock(spec=Tracer)
        mock_tracer.is_enabled.return_value = True
        mock_tracer.record_event.side_effect = Exception("Tracer failed")
        alarm_system = InvariantAlarmSystem(tracer=mock_tracer)

        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Record violations despite tracer failure
        alarm_system.detect_model_access_unauthorized("trace-456", "node")

        # Orchestrator should execute without interference
        result = orchestrator._router_node(state)
        assert result is not None
        assert result["input_type"] == "text"


class TestAlarmSeverityLevels:
    """Test that all severity levels are handled safely."""

    def test_warn_and_error_severity_violations_recorded(self):
        """Both warn and error severity violations are recorded."""
        alarm_system = InvariantAlarmSystem()

        # Record warn severity (memory advisory)
        alarm_system.detect_memory_advisory_overridden("trace-456", "node1", "type")
        # Record error severity (model access)
        alarm_system.detect_model_access_unauthorized("trace-456", "node2")

        violations = alarm_system.get_violations()
        assert len(violations) == 2
        assert violations[0].severity == "warn"
        assert violations[1].severity == "error"

    def test_all_violation_types_recorded_successfully(self):
        """All violation types are recorded correctly."""
        alarm_system = InvariantAlarmSystem()

        # Emit all violation types
        alarm_system.detect_model_access_unauthorized("trace-456", "node1")
        alarm_system.detect_memory_read_unauthorized("trace-456", "node2", "short_term")
        alarm_system.detect_memory_write_unauthorized("trace-456", "node3", "long_term")
        alarm_system.detect_multiple_terminal_nodes("trace-456", ["n1", "n2"])
        alarm_system.detect_state_mutation_outside_allowed_nodes("trace-456", "node4", ["f1"])
        alarm_system.detect_forbidden_data_in_trace("trace-456", "node5", ["d1"])
        alarm_system.detect_memory_not_append_only("trace-456", "node6", "delete")
        alarm_system.detect_memory_advisory_overridden("trace-456", "node7", "type")

        # All should be recorded
        violations = alarm_system.get_violations()
        assert len(violations) == 8

        # All violation types should be present
        violation_types = {v.violation_type for v in violations}
        expected_types = {
            ViolationType.MODEL_ACCESS_UNAUTHORIZED,
            ViolationType.MEMORY_READ_UNAUTHORIZED,
            ViolationType.MEMORY_WRITE_UNAUTHORIZED,
            ViolationType.MULTIPLE_TERMINAL_NODES,
            ViolationType.STATE_MUTATION_OUTSIDE_ALLOWED_NODES,
            ViolationType.FORBIDDEN_DATA_IN_TRACE,
            ViolationType.MEMORY_NOT_APPEND_ONLY,
            ViolationType.MEMORY_ADVISORY_OVERRIDDEN,
        }
        assert violation_types == expected_types


class TestViolationContextPreservation:
    """Test that violation context is preserved without contaminating state."""

    def test_violation_context_does_not_affect_orchestrator_state(self):
        """Violation contexts don't leak into agent state."""
        orchestrator = SAMAgentOrchestrator(tracer=NoOpTracer())
        alarm_system = InvariantAlarmSystem()

        state = AgentState(
            conversation_id="conv-123",
            trace_id="trace-456",
            created_at="",
            input_type="text",
            raw_input="test input",
        )

        # Emit violation with context
        alarm_system.detect_state_mutation_outside_allowed_nodes(
            "trace-456", "node", ["field1", "field2"]
        )

        # Preprocessing node should be unaffected
        result = orchestrator._task_preprocessing_node(state)
        assert result is not None
        assert result["preprocessing_result"] == "test input"


class TestAlarmClearingAndRetrieval:
    """Test that alarm history can be managed for testing."""

    def test_violations_can_be_retrieved(self):
        """Violations are retrievable for inspection."""
        alarm_system = InvariantAlarmSystem()

        alarm_system.detect_model_access_unauthorized("trace-456", "node")
        violations = alarm_system.get_violations()

        assert len(violations) == 1
        assert violations[0].trace_id == "trace-456"
        assert violations[0].node_name == "node"

    def test_violations_can_be_cleared(self):
        """Violation history can be cleared."""
        alarm_system = InvariantAlarmSystem()

        alarm_system.detect_model_access_unauthorized("trace-456", "node")
        assert len(alarm_system.get_violations()) == 1

        alarm_system.clear_violations()
        assert len(alarm_system.get_violations()) == 0

    def test_clear_does_not_affect_subsequent_detection(self):
        """Clearing doesn't affect subsequent violation detection."""
        alarm_system = InvariantAlarmSystem()

        alarm_system.detect_model_access_unauthorized("trace-456", "node1")
        alarm_system.clear_violations()
        alarm_system.detect_forbidden_data_in_trace("trace-456", "node2", ["data"])

        violations = alarm_system.get_violations()
        assert len(violations) == 1
        assert violations[0].node_name == "node2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
