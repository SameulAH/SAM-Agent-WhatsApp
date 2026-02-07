"""
Tool-agnostic tracing abstraction.

This module defines the Tracer interface that all observability implementations must follow.
Tracing is strictly passive:
- Never influences execution
- Never mutates state
- Never affects decisions
- Failures are silent and non-fatal

Frozen by design/observability_invariants.md
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TraceMetadata:
    """Metadata associated with a trace span or event."""

    trace_id: str  # Mandatory: globally unique identifier
    conversation_id: Optional[str] = None  # Optional: conversation session
    user_id: Optional[str] = None  # Optional: hashed/anonymized user identifier


class Tracer(ABC):
    """
    Abstract tracing interface.

    All implementations MUST guarantee:
    - No control flow influence
    - No state mutation
    - Non-fatal failures (never raise)
    - Best-effort execution
    """

    @abstractmethod
    def start_span(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> Optional[Any]:
        """
        Start a trace span (e.g., node execution, model call).

        Args:
            name: Span name (e.g., "router_node", "model_call")
            metadata: Structural metadata (node name, duration, status, etc.)
            trace_metadata: Trace identity (trace_id, conversation_id, user_id)

        Returns:
            Span handle (can be used in end_span, or None if tracing disabled)

        MUST NOT:
        - raise exceptions
        - affect control flow
        - modify agent state
        """
        pass

    @abstractmethod
    def end_span(self, span: Any, status: str, metadata: Dict[str, Any]) -> None:
        """
        End a trace span.

        Args:
            span: Span handle from start_span
            status: "success", "failure", or "skipped"
            metadata: Execution results (duration, error_type, etc.)

        MUST NOT:
        - raise exceptions
        - affect control flow
        - modify agent state
        """
        pass

    @abstractmethod
    def record_event(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> None:
        """
        Record a point-in-time event.

        Args:
            name: Event name (e.g., "memory_read_attempted", "invariant_violation")
            metadata: Event data (status, reason, etc.)
            trace_metadata: Trace identity

        MUST NOT:
        - raise exceptions
        - affect control flow
        - modify agent state
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if tracing is enabled.

        Returns:
            True if tracing is active, False otherwise

        Used for: avoiding expensive metadata construction when tracing is disabled
        """
        pass


class NoOpTracer(Tracer):
    """
    No-op tracing implementation (when tracing is disabled).

    Satisfies the Tracer interface but does nothing.
    Used when tracing is disabled or unavailable.
    """

    def start_span(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> Optional[Any]:
        """No-op implementation."""
        return None

    def end_span(self, span: Any, status: str, metadata: Dict[str, Any]) -> None:
        """No-op implementation."""
        pass

    def record_event(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> None:
        """No-op implementation."""
        pass

    def is_enabled(self) -> bool:
        """Tracing is disabled."""
        return False
