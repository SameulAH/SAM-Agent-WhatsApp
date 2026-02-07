"""
Trace Schema - Typed representations for Phase 4.2 observability data.

This module defines read-only schemas that represent traced execution data.
No logic, no mutations. Pure data structures.

Immutable by design - all fields are frozen.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class SpanStatus(str, Enum):
    """Span execution status (from Phase 4.2)."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class EventType(str, Enum):
    """Event types emitted during execution."""

    NODE_ENTRY = "node_entry"
    NODE_EXIT = "node_exit"
    MODEL_CALL_ATTEMPTED = "model_call_attempted"
    MODEL_CALL_COMPLETED = "model_call_completed"
    MEMORY_READ_ATTEMPTED = "memory_read_attempted"
    MEMORY_READ_COMPLETED = "memory_read_completed"
    MEMORY_WRITE_ATTEMPTED = "memory_write_attempted"
    MEMORY_WRITE_COMPLETED = "memory_write_completed"
    LONG_TERM_MEMORY_READ_ATTEMPTED = "long_term_memory_read_attempted"
    LONG_TERM_MEMORY_READ_COMPLETED = "long_term_memory_read_completed"
    LONG_TERM_MEMORY_WRITE_ATTEMPTED = "long_term_memory_write_attempted"
    LONG_TERM_MEMORY_WRITE_COMPLETED = "long_term_memory_write_completed"
    ERROR_RAISED = "error_raised"
    CORRECTION_RECEIVED = "correction_received"
    RETRY_REQUESTED = "retry_requested"


class AlarmSeverity(str, Enum):
    """Alarm severity levels."""

    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class TraceMetadata:
    """Trace identity metadata (immutable)."""

    trace_id: str  # Globally unique identifier
    conversation_id: Optional[str] = None  # Session identifier
    user_id: Optional[str] = None  # User identifier (hashed/anonymized)


@dataclass(frozen=True)
class Span:
    """A traced span (node execution, model call, memory access)."""

    span_id: str  # Unique identifier for this span
    node_name: str  # Which node/operation
    status: SpanStatus  # success/failure/skipped
    start_time: datetime  # ISO timestamp
    end_time: datetime  # ISO timestamp
    duration_ms: float  # Execution duration
    metadata: Dict[str, Any]  # Safe metadata (DENY list filtered)
    error_type: Optional[str] = None  # If failed, what type
    error_message: Optional[str] = None  # If failed, why


@dataclass(frozen=True)
class Event:
    """A point-in-time event (model call, memory access, correction)."""

    event_id: str  # Unique identifier
    event_type: EventType  # What happened
    timestamp: datetime  # When
    trace_metadata: TraceMetadata  # Which trace
    context: Dict[str, Any]  # Event-specific context
    turn_number: Optional[int] = None  # Which turn in conversation


@dataclass(frozen=True)
class Alarm:
    """An invariant violation alarm (from Phase 4.2)."""

    alarm_id: str  # Unique identifier
    violation_type: str  # Type of violation
    severity: AlarmSeverity  # warn or error
    trace_id: str  # Which trace
    node_name: str  # Where it happened
    timestamp: datetime  # When
    description: str  # What violated
    context: Dict[str, Any]  # Additional context


@dataclass(frozen=True)
class Trace:
    """Complete trace data for a single execution."""

    trace_id: str  # Execution identifier
    conversation_id: str  # Session identifier
    user_id: Optional[str]  # User identifier
    created_at: datetime  # Execution start time
    completed_at: datetime  # Execution end time
    spans: List[Span]  # All traced spans
    events: List[Event]  # All recorded events
    alarms: List[Alarm]  # All detected violations
    final_output: Optional[str] = None  # Agent's final response
    error_type: Optional[str] = None  # If execution failed


@dataclass(frozen=True)
class MetricsInput:
    """Input to metric computation functions."""

    traces: List[Trace]  # Traces to analyze
    session_id: str  # Session being analyzed (usually conversation_id)
    baseline: Optional[Dict[str, Any]] = None  # Baseline for comparison


# Helper functions to extract common signals

def get_all_events_of_type(trace: Trace, event_type: EventType) -> List[Event]:
    """Extract all events of a given type from a trace."""
    return [e for e in trace.events if e.event_type == event_type]


def get_all_spans_for_node(trace: Trace, node_name: str) -> List[Span]:
    """Extract all spans for a given node from a trace."""
    return [s for s in trace.spans if s.node_name == node_name]


def get_terminal_spans(trace: Trace) -> List[Span]:
    """Extract terminal node spans (response_generation_node, error_handler_node)."""
    terminal_nodes = {"response_generation_node", "error_handler_node"}
    return [s for s in trace.spans if s.node_name in terminal_nodes]


def has_correction_marker(trace: Trace) -> bool:
    """Check if trace contains user correction events."""
    correction_events = get_all_events_of_type(trace, EventType.CORRECTION_RECEIVED)
    return len(correction_events) > 0


def has_retry_marker(trace: Trace) -> bool:
    """Check if trace contains retry requests."""
    retry_events = get_all_events_of_type(trace, EventType.RETRY_REQUESTED)
    return len(retry_events) > 0


def get_total_duration_ms(trace: Trace) -> float:
    """Get total execution duration."""
    if trace.spans:
        return (trace.completed_at - trace.created_at).total_seconds() * 1000
    return 0.0


def get_terminal_node_count(trace: Trace) -> int:
    """Count how many terminal nodes executed."""
    return len(get_terminal_spans(trace))
