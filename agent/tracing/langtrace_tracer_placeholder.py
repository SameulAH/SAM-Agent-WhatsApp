"""
Langtrace implementation (optional future support).

This is a placeholder for potential Langtrace integration in the future.

Currently not activated. When activated, will follow the same constraints as LangSmith:
- No control flow influence
- No state mutation
- Failures are silent and non-fatal
- Metadata-only logging

To activate in future:
1. Import langtrace SDK
2. Implement start_span, end_span, record_event
3. Add TRACER_BACKEND="langtrace" to environment
4. Update agent initialization to instantiate this tracer
"""

from typing import Dict, Any, Optional, Callable

from agent.tracing.tracer import Tracer, TraceMetadata


class LangtraceTracer(Tracer):
    """
    Langtrace implementation (placeholder for future use).

    Not currently activated. When ready to use:
    1. Import langtrace SDK
    2. Implement tracing methods
    3. Add to TRACER_BACKEND options
    4. Set TRACER_BACKEND=langtrace in .env

    Designed with same constraints as LangSmith:
    - Never influences execution
    - Never mutates state
    - Failures are silent
    - Metadata-only data
    """

    def __init__(self, enabled: bool = False, observability_sink: Optional[Callable] = None):
        """
        Initialize Langtrace tracer.

        Args:
            enabled: Currently disabled (set to False to prevent activation)
            observability_sink: Optional callback for recording to local observability store

        Note:
            This tracer is intentionally disabled by default.
            To activate, implement the methods below and set enabled=True.
        """
        super().__init__(observability_sink)
        # Intentionally disabled by design (placeholder)
        self._enabled = False
        # Do NOT import langtrace SDK here
        # Do NOT create client unless explicitly enabled AND implemented

    def start_span(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> Optional[Any]:
        """
        Start a Langtrace span.

        Currently a no-op. Implementation pending.
        """
        return None

    def end_span(self, span: Any, status: str, metadata: Dict[str, Any]) -> None:
        """
        End a Langtrace span.

        Currently a no-op. Implementation pending.
        """
        pass

    def record_event(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> None:
        """
        Record a point-in-time event.

        Currently a no-op. Implementation pending.
        """
        pass

    def is_enabled(self) -> bool:
        """
        Check if Langtrace tracing is enabled.

        Returns:
            False (not yet implemented)
        """
        return False
