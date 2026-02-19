"""
Tests for Metrics & Instrumentation

Unit and integration tests for metrics collection and structured logging.
"""

import pytest
import json
from datetime import datetime

from agent.intelligence.metrics import (
    IntelligenceMetrics,
    MetricsCollector,
    get_metrics_collector,
)


class TestIntelligenceMetrics:
    """Test IntelligenceMetrics schema."""
    
    def test_valid_metrics(self):
        """Create valid metrics object."""
        metrics = IntelligenceMetrics(
            trace_id="trace_123",
            conversation_id="conv_1",
            timestamp=datetime.utcnow(),
            retrieval_attempted=True,
            retrieval_used=True,
            facts_retrieved=2,
            facts_injected=2,
            retrieval_correct=1,
            retrieval_precision=0.5,
            retrieval_recall=1.0,
            facts_extracted=3,
            facts_stored=3,
            memory_written_count=1,
            conflicts_detected=0,
            conflicts_resolved=0,
            tool_calls_attempted=0,
            tool_calls_successful=0,
            tool_names_used=[],
            hallucination_proxy=0.0,
            latency_ms=250,
            error=None,
        )
        
        assert metrics.trace_id == "trace_123"
        assert metrics.retrieval_precision == 0.5
        assert metrics.latency_ms == 250
    
    def test_metrics_with_error(self):
        """Metrics can include error."""
        metrics = IntelligenceMetrics(
            trace_id="trace_456",
            conversation_id="conv_2",
            timestamp=datetime.utcnow(),
            retrieval_attempted=False,
            retrieval_used=False,
            facts_retrieved=0,
            facts_injected=0,
            retrieval_correct=0,
            retrieval_precision=0.0,
            retrieval_recall=0.0,
            facts_extracted=0,
            facts_stored=0,
            memory_written_count=0,
            conflicts_detected=0,
            conflicts_resolved=0,
            tool_calls_attempted=0,
            tool_calls_successful=0,
            tool_names_used=[],
            hallucination_proxy=0.0,
            latency_ms=50,
            error="Memory write failed",
        )
        
        assert metrics.error == "Memory write failed"
        assert metrics.latency_ms == 50


class TestMetricsCollector:
    """Test MetricsCollector."""
    
    def test_start_collection(self):
        """Start metrics collection."""
        collector = MetricsCollector()
        
        collector.start_collection(
            trace_id="trace_1",
            conversation_id="conv_1"
        )
        
        assert collector.current_metrics is not None
        assert collector.current_metrics.trace_id == "trace_1"
    
    def test_record_retrieval(self):
        """Record retrieval metrics."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        
        collector.record_retrieval_attempt(True, 2)
        
        assert collector.current_metrics.retrieval_attempted == True
        assert collector.current_metrics.facts_retrieved == 2
    
    def test_record_facts_extracted(self):
        """Record facts extraction."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        
        collector.record_facts_extracted(3, stored=3)
        
        assert collector.current_metrics.facts_extracted == 3
        assert collector.current_metrics.facts_stored == 3
    
    def test_record_tool_call(self):
        """Record tool call."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        
        collector.record_tool_call("web_search", success=True)
        
        assert collector.current_metrics.tool_calls_attempted == 1
        assert collector.current_metrics.tool_calls_successful == 1
        assert "web_search" in collector.current_metrics.tool_names_used
    
    def test_record_conflicts(self):
        """Record conflict detection."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        
        collector.record_conflicts_detected(2)
        collector.record_conflicts_resolved(1)
        
        assert collector.current_metrics.conflicts_detected == 2
        assert collector.current_metrics.conflicts_resolved == 1
    
    def test_record_latency(self):
        """Record latency."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        
        collector.record_latency_ms(450)
        
        assert collector.current_metrics.latency_ms == 450


class TestMetricsEmission:
    """Test metrics emission and logging."""
    
    def test_emit_json(self):
        """Emit metrics as JSON."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        collector.record_retrieval_attempt(True, 1)
        collector.record_latency_ms(100)
        
        metrics_dict = collector.emit()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["trace_id"] == "trace_1"
        assert metrics_dict["latency_ms"] == 100
        assert metrics_dict["retrieval_attempted"] == True
    
    def test_emit_serializable(self):
        """Emitted metrics are JSON serializable."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        collector.record_retrieval_attempt(True, 2)
        
        metrics_dict = collector.emit()
        
        # Should be JSON serializable
        json_str = json.dumps(metrics_dict, default=str)
        assert json_str is not None
        assert "trace_1" in json_str


class TestMetricsCollectorSingleton:
    """Test MetricsCollector singleton."""
    
    def test_get_collector(self):
        """Get global metrics collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should be same instance
        assert collector1 is collector2
    
    def test_global_collector_works(self):
        """Global collector is functional."""
        collector = get_metrics_collector()
        
        collector.start_collection("trace_1", "conv_1")
        collector.record_retrieval_attempt(True, 1)
        
        metrics = collector.emit()
        
        assert metrics["trace_id"] == "trace_1"
        assert metrics["retrieval_attempted"] == True


class TestMetricsIntegration:
    """Integration tests for metrics collection."""
    
    def test_full_collection_flow(self):
        """Full metrics collection flow."""
        collector = MetricsCollector()
        
        # Start collection
        collector.start_collection("trace_123", "conv_456")
        
        # Simulate turn
        collector.record_retrieval_attempt(True, 2)
        collector.record_facts_extracted(3, 3)
        collector.record_conflicts_detected(1)
        collector.record_conflicts_resolved(1)
        collector.record_tool_call("web_search", True)
        collector.record_latency_ms(350)
        
        # Emit
        metrics = collector.emit()
        
        # Verify
        assert metrics["trace_id"] == "trace_123"
        assert metrics["conversation_id"] == "conv_456"
        assert metrics["retrieval_attempted"] == True
        assert metrics["facts_retrieved"] == 2
        assert metrics["facts_extracted"] == 3
        assert metrics["conflicts_detected"] == 1
        assert metrics["tool_calls_attempted"] == 1
        assert metrics["latency_ms"] == 350
    
    def test_metrics_precision(self):
        """Metrics calculate precision and recall."""
        collector = MetricsCollector()
        collector.start_collection("trace_1", "conv_1")
        
        # 2 facts retrieved, 1 correct
        collector.record_retrieval_attempt(True, 2)
        collector.record_retrieval_precision_recall(
            precision=0.5,  # 1/2 correct
            recall=1.0,     # All relevant facts retrieved
        )
        
        metrics = collector.emit()
        
        assert metrics["retrieval_precision"] == 0.5
        assert metrics["retrieval_recall"] == 1.0
    
    def test_metrics_does_not_affect_routing(self):
        """Metrics collection is transparent to routing."""
        collector = MetricsCollector()
        
        # Collection should succeed even with no metrics
        collector.start_collection("trace_1", "conv_1")
        metrics = collector.emit()
        
        # No errors, silent metrics collection
        assert metrics is not None
        assert metrics["trace_id"] == "trace_1"
    
    def test_multiple_collectors_independent(self):
        """Multiple MetricsCollector instances are independent."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        
        collector1.start_collection("trace_1", "conv_1")
        collector2.start_collection("trace_2", "conv_2")
        
        assert collector1.current_metrics.trace_id == "trace_1"
        assert collector2.current_metrics.trace_id == "trace_2"
