"""
Intelligence Metrics & Instrumentation

Track per-invocation metrics for evaluation.
Emit structured JSON logs.
Measure intelligence effectiveness.

Tracks:
- Retrieval usage & accuracy
- Memory writes
- Conflict detection
- Tool usage
- Latency
- Hallucination proxies
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class IntelligenceMetrics(BaseModel):
    """
    Per-invocation intelligence metrics.
    
    Tracks all aspects of intelligent behavior.
    """
    
    # Identity
    trace_id: str
    conversation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Retrieval metrics
    retrieval_attempted: bool = False
    retrieval_used: bool = False
    facts_retrieved: int = 0
    facts_injected: int = 0
    retrieval_correct: Optional[bool] = None  # Human-evaluated
    retrieval_precision: Optional[float] = None  # facts_relevant / facts_injected
    retrieval_recall: Optional[float] = None  # facts_relevant / facts_available
    
    # Memory metrics
    facts_extracted: int = 0
    facts_stored: int = 0
    memory_written_count: int = 0
    
    # Conflict metrics
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    
    # Tool metrics
    tool_calls_attempted: int = 0
    tool_calls_successful: int = 0
    tool_names_used: list = Field(default_factory=list)
    
    # Quality metrics
    hallucination_proxy: Optional[float] = None  # 0 = no hallucinations, 1 = likely hallucinating
    
    # Performance metrics
    latency_ms: int = 0
    
    # Observability
    error: Optional[str] = None


class MetricsCollector:
    """
    Collect and emit intelligence metrics.
    
    Emits structured JSON logs.
    Thread-safe.
    Non-blocking.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._current_metrics: Optional[IntelligenceMetrics] = None
    
    @property
    def current_metrics(self) -> Optional[IntelligenceMetrics]:
        """Get current metrics object."""
        return self._current_metrics
    
    def start_collection(
        self,
        trace_id: str,
        conversation_id: str,
    ) -> None:
        """
        Start collecting metrics for a turn.
        
        Args:
            trace_id: Unique trace ID
            conversation_id: Conversation ID
        """
        
        self._current_metrics = IntelligenceMetrics(
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    
    def record_retrieval_attempt(self, attempted: bool, facts_retrieved: int) -> None:
        """Record that retrieval was attempted."""
        if self._current_metrics:
            self._current_metrics.retrieval_attempted = attempted
            self._current_metrics.facts_retrieved = facts_retrieved
    
    def record_facts_injected(self, count: int) -> None:
        """Record number of facts injected into prompt."""
        if self._current_metrics:
            self._current_metrics.retrieval_used = count > 0
            self._current_metrics.facts_injected = count
    
    def record_facts_extracted(self, count: int, stored: int = 0) -> None:
        """Record number of facts extracted and stored."""
        if self._current_metrics:
            self._current_metrics.facts_extracted = count
            if stored > 0:
                self._current_metrics.facts_stored = stored
                self._current_metrics.memory_written_count += stored
    
    def record_conflicts_detected(self, count: int) -> None:
        """Record conflict detection events."""
        if self._current_metrics:
            self._current_metrics.conflicts_detected = count
    
    def record_conflicts_resolved(self, count: int) -> None:
        """Record conflict resolution events."""
        if self._current_metrics:
            self._current_metrics.conflicts_resolved = count
    
    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """Record tool execution."""
        if self._current_metrics:
            self._current_metrics.tool_calls_attempted += 1
            if success:
                self._current_metrics.tool_calls_successful += 1
            if tool_name not in self._current_metrics.tool_names_used:
                self._current_metrics.tool_names_used.append(tool_name)
    
    def record_latency_ms(self, latency_ms: int) -> None:
        """Record execution latency."""
        if self._current_metrics:
            self._current_metrics.latency_ms = latency_ms
    
    def record_retrieval_precision_recall(
        self,
        precision: float,
        recall: float,
    ) -> None:
        """Record retrieval precision and recall."""
        if self._current_metrics:
            self._current_metrics.retrieval_precision = precision
            self._current_metrics.retrieval_recall = recall
    
    def record_retrieval_correct(self, count: int) -> None:
        """Record number of retrieved facts that were correct."""
        if self._current_metrics:
            self._current_metrics.retrieval_correct = count
    def record_error(self, error_msg: str) -> None:
        """Record error event."""
        if self._current_metrics:
            self._current_metrics.error = error_msg
    
    def emit(self) -> Optional[Dict[str, Any]]:
        """
        Emit metrics as JSON and reset.
        
        Returns:
            Metrics dict (or None if not collecting)
        """
        
        if not self._current_metrics:
            return None
        
        try:
            metrics_dict = self._current_metrics.dict(
                exclude_none=False,
                exclude_unset=False,
            )
            
            # Convert datetime to ISO string
            metrics_dict["timestamp"] = self._current_metrics.timestamp.isoformat()
            
            # Log as structured JSON
            logger.info(
                f"Intelligence metrics",
                extra={
                    "metrics": metrics_dict,
                    "trace_id": self._current_metrics.trace_id,
                }
            )
            
            # Reset for next collection
            self._current_metrics = None
            
            return metrics_dict
        
        except Exception as e:
            logger.error(f"Failed to emit metrics: {e}")
            self._current_metrics = None
            return None


def get_metrics_collector() -> MetricsCollector:
    """Get singleton metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


_collector: Optional[MetricsCollector] = None
