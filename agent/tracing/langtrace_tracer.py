"""
LangTrace-backed tracing implementation.

Hooks into LangSmith's tracing service for observability.

Frozen constraints:
- Never influences control flow
- Never mutates state
- Failures are silent and non-fatal
- Only allowed data is traced
"""

from typing import Dict, Any, Optional, Callable
import json

from agent.tracing.tracer import Tracer, TraceMetadata, NoOpTracer


class LangTraceTracer(Tracer):
    """
    LangTrace implementation of the Tracer interface.

    Wraps LangSmith tracing while respecting all observability invariants.

    Safe to fail silently → agent continues unchanged.
    """

    def __init__(self, enabled: bool = True, observability_sink: Optional[Callable] = None):
        """
        Initialize LangTrace tracer.

        Args:
            enabled: Whether to enable tracing. If False, acts as NoOpTracer.
            observability_sink: Optional callback for recording to observability store
        """
        super().__init__(observability_sink)
        self._enabled = enabled
        self._langsmith_client = None

        if enabled:
            try:
                from langsmith import Client
                self._langsmith_client = Client()
            except (ImportError, Exception):
                # LangSmith unavailable or import failed
                # Silently degrade to no-op (non-fatal)
                self._enabled = False

    def start_span(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> Optional[Any]:
        """
        Start a LangTrace span.

        Best-effort only. Failures are silent.
        """
        if not self._enabled:
            return None

        try:
            # Record to observability sink if available
            if self.observability_sink:
                try:
                    self.observability_sink("span_start", {
                        "trace_id": trace_metadata.trace_id,
                        "conversation_id": trace_metadata.conversation_id,
                        "node_name": name,
                    })
                except Exception:
                    pass  # Observability failure is non-fatal

            # Build trace input with allowed metadata only
            trace_input = {
                "node_name": name,
                "trace_id": trace_metadata.trace_id,
            }

            # Add optional fields if present
            if trace_metadata.conversation_id:
                trace_input["conversation_id"] = trace_metadata.conversation_id
            if trace_metadata.user_id:
                trace_input["user_id"] = trace_metadata.user_id

            # Add structural metadata (only)
            # Filter out any sensitive fields
            safe_metadata = self._filter_safe_metadata(metadata)
            trace_input.update(safe_metadata)

            # Note: Actual LangTrace integration happens at the caller level
            # (e.g., via LangGraph's built-in tracing or manual context managers)
            # This method returns metadata for the caller to use

            return {
                "trace_id": trace_metadata.trace_id,
                "span_name": name,
                "input": trace_input,
                "start_time": None,  # Set by caller
            }
        except Exception:
            # Tracing failure is non-fatal
            return None

    def end_span(self, span: Any, status: str, metadata: Dict[str, Any]) -> None:
        """
        End a LangTrace span.

        Best-effort only. Failures are silent.
        """
        if not self._enabled or span is None:
            return

        try:
            # Record to observability sink if available
            if self.observability_sink:
                try:
                    self.observability_sink("span_end", {
                        "trace_id": span.get("trace_id"),
                        "span_name": span.get("span_name"),
                        "status": status,
                        "duration_ms": metadata.get("duration_ms"),
                    })
                except Exception:
                    pass  # Observability failure is non-fatal

            # Build safe output
            safe_metadata = self._filter_safe_metadata(metadata)

            # Update span with execution results
            span["status"] = status
            span["output"] = safe_metadata
            span["end_time"] = None  # Set by caller

            # Note: Actual span finalization happens at caller level
        except Exception:
            # Tracing failure is non-fatal
            pass

    def record_event(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> None:
        """
        Record a point-in-time event (e.g., invariant violation).

        Best-effort only. Failures are silent.
        """
        if not self._enabled:
            return

        try:
            event_data = {
                "event_name": name,
                "trace_id": trace_metadata.trace_id,
            }

            if trace_metadata.conversation_id:
                event_data["conversation_id"] = trace_metadata.conversation_id
            if trace_metadata.user_id:
                event_data["user_id"] = trace_metadata.user_id

            safe_metadata = self._filter_safe_metadata(metadata)
            event_data.update(safe_metadata)

            # Note: Event recording happens at caller level
            # This method is for compatibility and future expansion
        except Exception:
            # Tracing failure is non-fatal
            pass

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    @staticmethod
    def _filter_safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to include only safe, non-sensitive fields.

        Enforces the DENY LIST from design/observability_invariants.md

        Forbidden:
        - raw_input, preprocessing_result, final_output
        - prompts, model outputs
        - memory payloads
        - embeddings, vectors
        - secrets, tokens, API keys
        - authorization flags
        - decision internals

        Allowed:
        - node name, timestamps, duration
        - status, failure type
        - operation flags (attempted/succeeded)
        """
        deny_fields = {
            # User content
            "raw_input",
            "preprocessing_result",
            "final_output",
            # Model intelligence
            "prompts",
            "model_output",
            "model_response",
            "tokens",
            # Memory payloads
            "memory_read_result",
            "memory_write_data",
            "long_term_memory_read_result",
            "long_term_memory_write_data",
            # Vectors and embeddings
            "embeddings",
            "vectors",
            "embedding_vector",
            # Secrets
            "api_key",
            "token",
            "password",
            "secret",
            # Authorization
            "memory_read_authorized",
            "memory_write_authorized",
            "authorization",
            # Decision internals
            "decision_logic",
            "internal_state",
            "debug_info",
        }

        safe_metadata = {}
        for key, value in metadata.items():
            if key.lower() in deny_fields:
                continue
            safe_metadata[key] = value

        return safe_metadata
