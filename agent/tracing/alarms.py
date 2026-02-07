"""
Invariant-violation alarms.

Alarms detect violations of design/observability_invariants.md and design/long_term_memory_invariants.md.

Alarms are smoke detectors, NOT sprinklers:
- Emit events only (no authority)
- Never affect control flow
- Never block execution
- Never make decisions
- Non-fatal if backend is unavailable

Violations detected:
1. Model call without explicit authorization
2. Memory access without authorization
3. Multiple terminal nodes executed (control flow violation)
4. State mutation outside allowed nodes
5. Forbidden data in trace payloads
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ViolationType(Enum):
    """Possible violation types."""

    # Authorization violations
    MODEL_ACCESS_UNAUTHORIZED = "model_access_unauthorized"
    MEMORY_READ_UNAUTHORIZED = "memory_read_unauthorized"
    MEMORY_WRITE_UNAUTHORIZED = "memory_write_unauthorized"

    # Control flow violations
    MULTIPLE_TERMINAL_NODES = "multiple_terminal_nodes"
    UNEXPECTED_ROUTING = "unexpected_routing"

    # State violations
    STATE_MUTATION_OUTSIDE_ALLOWED_NODES = "state_mutation_outside_allowed_node"
    FORBIDDEN_STATE_FIELD_MUTATED = "forbidden_state_field_mutated"

    # Data safety violations
    FORBIDDEN_DATA_IN_TRACE = "forbidden_data_in_trace"
    TRACE_FAILURE_BLOCKING_EXECUTION = "trace_failure_blocking_execution"

    # Memory invariant violations
    MEMORY_NOT_APPEND_ONLY = "memory_not_append_only"
    MEMORY_ADVISORY_OVERRIDDEN = "memory_advisory_overridden"


@dataclass
class InvariantViolationEvent:
    """
    Immutable event representing a detected invariant violation.

    Purpose: Emit-only alarm (never affects control flow or decisions).
    """

    violation_type: ViolationType
    trace_id: str
    node_name: str
    description: str
    context: Dict[str, Any]  # Additional context (filtered for safety)
    severity: str  # "warn", "error" (metadata only, never affects execution)

    def __post_init__(self):
        """Validate fields."""
        if not self.trace_id:
            raise ValueError("trace_id is required for all violations")
        if not self.node_name:
            raise ValueError("node_name is required")
        if self.severity not in ("warn", "error"):
            raise ValueError(f"Invalid severity: {self.severity}")


class InvariantAlarmSystem:
    """
    Detects and emits invariant violations.

    Non-blocking: violations are observed, logged, but never affect execution.
    """

    def __init__(self, tracer: Optional[Any] = None):
        """
        Initialize alarm system.

        Args:
            tracer: Optional tracer to emit alarms to. If None, alarms are collected locally.
        """
        self._tracer = tracer
        self._violations: list[InvariantViolationEvent] = []

    def detect_model_access_unauthorized(
        self, trace_id: str, node_name: str
    ) -> None:
        """Detect unauthorized model access."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.MODEL_ACCESS_UNAUTHORIZED,
            trace_id=trace_id,
            node_name=node_name,
            description="Model was accessed without explicit authorization",
            context={"attempted_at": node_name},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_memory_read_unauthorized(
        self, trace_id: str, node_name: str, memory_type: str
    ) -> None:
        """Detect unauthorized memory read."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.MEMORY_READ_UNAUTHORIZED,
            trace_id=trace_id,
            node_name=node_name,
            description=f"{memory_type} memory was read without authorization",
            context={"memory_type": memory_type, "attempted_at": node_name},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_memory_write_unauthorized(
        self, trace_id: str, node_name: str, memory_type: str
    ) -> None:
        """Detect unauthorized memory write."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.MEMORY_WRITE_UNAUTHORIZED,
            trace_id=trace_id,
            node_name=node_name,
            description=f"{memory_type} memory was written without authorization",
            context={"memory_type": memory_type, "attempted_at": node_name},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_multiple_terminal_nodes(
        self, trace_id: str, executed_nodes: list[str]
    ) -> None:
        """Detect multiple terminal nodes executed (control flow violation)."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.MULTIPLE_TERMINAL_NODES,
            trace_id=trace_id,
            node_name="orchestrator",
            description=f"Multiple terminal nodes executed: {executed_nodes}",
            context={"executed_nodes": executed_nodes},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_state_mutation_outside_allowed_nodes(
        self, trace_id: str, node_name: str, fields_mutated: list[str]
    ) -> None:
        """Detect state mutation outside allowed nodes."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.STATE_MUTATION_OUTSIDE_ALLOWED_NODES,
            trace_id=trace_id,
            node_name=node_name,
            description=f"State mutation detected outside allowed nodes: {fields_mutated}",
            context={"fields_mutated": fields_mutated},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_forbidden_data_in_trace(
        self, trace_id: str, node_name: str, forbidden_fields: list[str]
    ) -> None:
        """Detect forbidden data (prompts, outputs, secrets) in trace."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.FORBIDDEN_DATA_IN_TRACE,
            trace_id=trace_id,
            node_name=node_name,
            description=f"Forbidden data detected in trace: {forbidden_fields}",
            context={"forbidden_fields": forbidden_fields},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_memory_not_append_only(
        self, trace_id: str, node_name: str, operation: str
    ) -> None:
        """Detect memory operation that violates append-only property."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.MEMORY_NOT_APPEND_ONLY,
            trace_id=trace_id,
            node_name=node_name,
            description=f"Memory operation is not append-only: {operation}",
            context={"operation": operation},
            severity="error",
        )
        self._emit_violation(violation)

    def detect_memory_advisory_overridden(
        self, trace_id: str, node_name: str, memory_type: str
    ) -> None:
        """Detect memory advisory being treated as authoritative."""
        violation = InvariantViolationEvent(
            violation_type=ViolationType.MEMORY_ADVISORY_OVERRIDDEN,
            trace_id=trace_id,
            node_name=node_name,
            description=f"Memory advisory was treated as authoritative: {memory_type}",
            context={"memory_type": memory_type},
            severity="warn",
        )
        self._emit_violation(violation)

    def get_violations(self) -> list[InvariantViolationEvent]:
        """Get all recorded violations (for testing)."""
        return list(self._violations)

    def clear_violations(self) -> None:
        """Clear violation history (for testing)."""
        self._violations.clear()

    def _emit_violation(self, violation: InvariantViolationEvent) -> None:
        """
        Emit a violation event.

        Non-blocking: failures are silent.
        """
        try:
            self._violations.append(violation)

            # Attempt to emit via tracer if available
            if self._tracer and self._tracer.is_enabled():
                try:
                    self._tracer.record_event(
                        name=f"invariant_violation.{violation.violation_type.value}",
                        metadata={
                            "description": violation.description,
                            "severity": violation.severity,
                            **violation.context,
                        },
                        trace_metadata=type(
                            "TraceMetadata",
                            (),
                            {
                                "trace_id": violation.trace_id,
                                "conversation_id": None,
                                "user_id": None,
                            },
                        )(),
                    )
                except Exception:
                    # Alarm emission failure is non-fatal
                    pass
        except Exception:
            # Alarm recording failure is non-fatal
            pass
