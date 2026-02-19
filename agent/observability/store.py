"""
In-memory observability store for local development.

Collects metadata from tracer and memory operations.
- Bounded size (FIFO eviction)
- Thread-safe
- Non-blocking
- No persistence
- No sensitive data (metadata only)
"""

from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """Metadata from a trace execution."""
    trace_id: str
    conversation_id: str
    start_time: str
    end_time: Optional[str] = None
    status: Optional[str] = None  # success, error, in-progress
    node_count: int = 0


@dataclass
class SpanRecord:
    """Metadata from a span (node execution)."""
    span_id: str
    trace_id: str
    node_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "in-progress"  # success, error, in-progress


@dataclass
class MemoryEventRecord:
    """Metadata from a memory operation."""
    timestamp: str
    operation: str  # read, write
    conversation_id: str
    authorized: bool
    backend: str  # sqlite, qdrant, stub
    status: str  # success, failed, unavailable, unauthorized


class ObservabilityStore:
    """
    Thread-safe, bounded in-memory store for observability metadata.
    
    Stores only:
    - Trace metadata (IDs, timestamps, status)
    - Span metadata (node execution times, status)
    - Memory operation metadata (operation type, backend, status)
    
    Does NOT store:
    - Prompts
    - Model outputs
    - Memory contents
    - User inputs
    - Decision logic
    """

    def __init__(self, max_traces: int = 100, max_spans: int = 500, 
                 max_events: int = 500, max_memory_events: int = 200):
        """
        Initialize bounded store.
        
        Args:
            max_traces: Maximum number of recent traces to keep
            max_spans: Maximum number of recent spans to keep
            max_events: Maximum number of recent events to keep
            max_memory_events: Maximum number of recent memory events to keep
        """
        self.max_traces = max_traces
        self.max_spans = max_spans
        self.max_events = max_events
        self.max_memory_events = max_memory_events
        
        self.traces: deque = deque(maxlen=max_traces)
        self.spans: deque = deque(maxlen=max_spans)
        self.events: deque = deque(maxlen=max_events)
        self.memory_events: deque = deque(maxlen=max_memory_events)
        
        self._lock = threading.RLock()
        self._active_spans: Dict[str, SpanRecord] = {}  # span_id -> SpanRecord
        self._active_traces: Dict[str, TraceRecord] = {}  # trace_id -> TraceRecord

    def record_trace_start(self, trace_id: str, conversation_id: str) -> None:
        """Record trace start."""
        try:
            with self._lock:
                record = TraceRecord(
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    start_time=datetime.now().isoformat(),
                )
                self._active_traces[trace_id] = record
        except Exception as e:
            logger.debug(f"Failed to record trace start: {e}")

    def record_trace_end(self, trace_id: str, status: str = "success") -> None:
        """Record trace end and move to completed."""
        try:
            with self._lock:
                if trace_id in self._active_traces:
                    record = self._active_traces.pop(trace_id)
                    record.end_time = datetime.now().isoformat()
                    record.status = status
                    self.traces.append(record)
        except Exception as e:
            logger.debug(f"Failed to record trace end: {e}")

    def record_span_start(self, span_id: str, trace_id: str, node_name: str) -> None:
        """Record span start."""
        try:
            with self._lock:
                record = SpanRecord(
                    span_id=span_id,
                    trace_id=trace_id,
                    node_name=node_name,
                    start_time=datetime.now().isoformat(),
                )
                self._active_spans[span_id] = record
                
                # Update trace node count
                if trace_id in self._active_traces:
                    self._active_traces[trace_id].node_count += 1
        except Exception as e:
            logger.debug(f"Failed to record span start: {e}")

    def record_span_end(self, span_id: str, status: str = "success") -> None:
        """Record span end and move to completed."""
        try:
            with self._lock:
                if span_id in self._active_spans:
                    record = self._active_spans.pop(span_id)
                    record.end_time = datetime.now().isoformat()
                    record.status = status
                    
                    # Calculate duration
                    try:
                        start = datetime.fromisoformat(record.start_time)
                        end = datetime.fromisoformat(record.end_time)
                        record.duration_ms = (end - start).total_seconds() * 1000
                    except Exception:
                        pass
                    
                    self.spans.append(record)
        except Exception as e:
            logger.debug(f"Failed to record span end: {e}")

    def record_memory_event(self, operation: str, conversation_id: str,
                           authorized: bool, backend: str, status: str) -> None:
        """Record memory operation (read/write)."""
        try:
            with self._lock:
                record = MemoryEventRecord(
                    timestamp=datetime.now().isoformat(),
                    operation=operation,
                    conversation_id=conversation_id,
                    authorized=authorized,
                    backend=backend,
                    status=status,
                )
                self.memory_events.append(record)
        except Exception as e:
            logger.debug(f"Failed to record memory event: {e}")

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent completed traces."""
        try:
            with self._lock:
                traces = list(self.traces)[-limit:]
                return [asdict(t) for t in traces]
        except Exception as e:
            logger.debug(f"Failed to get traces: {e}")
            return []

    def get_recent_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent completed spans."""
        try:
            with self._lock:
                spans = list(self.spans)[-limit:]
                return [asdict(s) for s in spans]
        except Exception as e:
            logger.debug(f"Failed to get spans: {e}")
            return []

    def get_memory_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memory operations."""
        try:
            with self._lock:
                events = list(self.memory_events)[-limit:]
                return [asdict(e) for e in events]
        except Exception as e:
            logger.debug(f"Failed to get memory events: {e}")
            return []

    def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get currently active traces."""
        try:
            with self._lock:
                return [asdict(t) for t in self._active_traces.values()]
        except Exception as e:
            logger.debug(f"Failed to get active traces: {e}")
            return []

    def clear(self) -> None:
        """Clear all stored data."""
        try:
            with self._lock:
                self.traces.clear()
                self.spans.clear()
                self.events.clear()
                self.memory_events.clear()
                self._active_spans.clear()
                self._active_traces.clear()
        except Exception as e:
            logger.debug(f"Failed to clear store: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get store statistics."""
        try:
            with self._lock:
                return {
                    "completed_traces": len(self.traces),
                    "active_traces": len(self._active_traces),
                    "completed_spans": len(self.spans),
                    "active_spans": len(self._active_spans),
                    "memory_events": len(self.memory_events),
                }
        except Exception as e:
            logger.debug(f"Failed to get stats: {e}")
            return {}
