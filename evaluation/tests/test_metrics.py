"""
Phase 5.2 evaluation tests

Validates that:
1. Metric computation is deterministic
2. Results are identical on repeated runs
3. Regression detection works correctly
4. Zero agent runtime dependency
5. All metrics are pure functions
"""

import pytest
from datetime import datetime, timedelta
from evaluation.schemas.trace_schema import (
    Trace,
    Span,
    Event,
    Alarm,
    SpanStatus,
    EventType,
    AlarmSeverity,
    TraceMetadata,
)
from evaluation.schemas.metric_schema import (
    MetricResult,
    get_metric_by_id,
    get_metrics_for_dimension,
)
from evaluation.metrics.task_completion import (
    compute_task_completion_rate,
    compute_correction_rate,
)
from evaluation.metrics.retry_pressure import compute_retry_rate
from evaluation.metrics.memory_usefulness import compute_memory_reference_rate
from evaluation.metrics.latency_quality import compute_response_time_ms
from evaluation.aggregation.per_session import compute_per_session_metrics
from evaluation.aggregation.per_run import compute_per_run_metrics
from evaluation.compare_runs import RegressionDetector


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────


def create_test_trace(
    trace_id: str = "trace-123",
    conversation_id: str = "conv-123",
    with_correction: bool = False,
    with_retry: bool = False,
    duration_ms: float = 2000.0,
) -> Trace:
    """Create a minimal valid Trace for testing."""
    now = datetime.utcnow()

    # Create basic span
    span = Span(
        span_id="span-1",
        node_name="response_generation_node",
        status=SpanStatus.SUCCESS,
        start_time=now,
        end_time=now + timedelta(milliseconds=duration_ms),
        duration_ms=duration_ms,
        metadata={"node_name": "response_generation_node"},
    )

    # Create events
    events = [
        Event(
            event_id="evt-1",
            event_type=EventType.MODEL_CALL_COMPLETED,
            timestamp=now,
            trace_metadata=TraceMetadata(trace_id=trace_id, conversation_id=conversation_id),
            context={},
        ),
    ]

    if with_correction:
        events.append(
            Event(
                event_id="evt-2",
                event_type=EventType.CORRECTION_RECEIVED,
                timestamp=now + timedelta(seconds=1),
                trace_metadata=TraceMetadata(
                    trace_id=trace_id, conversation_id=conversation_id
                ),
                context={},
            )
        )

    if with_retry:
        events.append(
            Event(
                event_id="evt-3",
                event_type=EventType.RETRY_REQUESTED,
                timestamp=now + timedelta(seconds=2),
                trace_metadata=TraceMetadata(
                    trace_id=trace_id, conversation_id=conversation_id
                ),
                context={},
            )
        )

    return Trace(
        trace_id=trace_id,
        conversation_id=conversation_id,
        user_id=None,
        created_at=now,
        completed_at=now + timedelta(milliseconds=duration_ms),
        spans=[span],
        events=events,
        alarms=[],
        final_output="Test output",
    )


# ─────────────────────────────────────────────────────
# TESTS: METRIC SCHEMAS
# ─────────────────────────────────────────────────────


class TestMetricSchemas:
    """Test metric schema definitions."""

    def test_all_metrics_retrievable(self):
        """All metrics can be retrieved by ID."""
        metric_ids = [
            "task_completion_rate",
            "correction_rate",
            "retry_rate",
            "memory_reference_rate",
            "response_time_ms",
        ]

        for metric_id in metric_ids:
            metric = get_metric_by_id(metric_id)
            assert metric is not None
            assert metric.metric_id == metric_id

    def test_dimension_metrics(self):
        """Metrics can be grouped by dimension."""
        completion_metrics = get_metrics_for_dimension("task_completion_effectiveness")
        assert len(completion_metrics) > 0
        assert all(m.dimension == "task_completion_effectiveness" for m in completion_metrics)

    def test_metric_direction_defined(self):
        """All metrics have clear direction (higher/lower is better)."""
        for metric_id in [
            "task_completion_rate",
            "correction_rate",
            "response_time_ms",
        ]:
            metric = get_metric_by_id(metric_id)
            assert metric.direction in ["higher", "lower"]


# ─────────────────────────────────────────────────────
# TESTS: DETERMINISTIC COMPUTATION
# ─────────────────────────────────────────────────────


class TestDeterministicComputation:
    """Verify metric computation is deterministic."""

    def test_task_completion_deterministic(self):
        """Same traces produce same metric value."""
        traces = [
            create_test_trace("t1", with_correction=False),
            create_test_trace("t2", with_correction=False),
            create_test_trace("t3", with_correction=True),
        ]

        result1 = compute_task_completion_rate(traces)
        result2 = compute_task_completion_rate(traces)

        assert result1.value == result2.value
        assert result1.samples == result2.samples

    def test_correction_rate_deterministic(self):
        """Correction rate is deterministic."""
        traces = [
            create_test_trace("t1", with_correction=True),
            create_test_trace("t2", with_correction=False),
        ]

        result1 = compute_correction_rate(traces)
        result2 = compute_correction_rate(traces)

        assert result1.value == result2.value

    def test_retry_rate_deterministic(self):
        """Retry rate is deterministic."""
        traces = [
            create_test_trace("t1", with_retry=True),
            create_test_trace("t2", with_retry=False),
        ]

        result1 = compute_retry_rate(traces)
        result2 = compute_retry_rate(traces)

        assert result1.value == result2.value

    def test_latency_deterministic(self):
        """Response time computation is deterministic."""
        traces = [
            create_test_trace("t1", duration_ms=1000.0),
            create_test_trace("t2", duration_ms=2000.0),
            create_test_trace("t3", duration_ms=3000.0),
        ]

        result1 = compute_response_time_ms(traces)
        result2 = compute_response_time_ms(traces)

        assert result1.value == result2.value


# ─────────────────────────────────────────────────────
# TESTS: AGGREGATION
# ─────────────────────────────────────────────────────


class TestAggregation:
    """Test metric aggregation."""

    def test_per_session_aggregation(self):
        """Per-session metrics can be computed."""
        traces = [
            create_test_trace("t1"),
            create_test_trace("t2"),
            create_test_trace("t3"),
        ]

        results = compute_per_session_metrics(traces)

        assert results.session_id == "conv-123"
        assert len(results.results) > 0
        assert results.get_metric("task_completion_rate") is not None

    def test_per_run_aggregation(self):
        """Per-run metrics aggregate across sessions."""
        traces_by_session = {
            "sess-1": [
                create_test_trace("t1", conversation_id="sess-1"),
                create_test_trace("t2", conversation_id="sess-1"),
            ],
            "sess-2": [
                create_test_trace("t3", conversation_id="sess-2"),
                create_test_trace("t4", conversation_id="sess-2"),
            ],
        }

        run_results = compute_per_run_metrics(traces_by_session)

        assert run_results["session_count"] == 2
        assert "metrics" in run_results
        assert "task_completion_rate" in run_results["metrics"]

        # Check statistics
        task_completion = run_results["metrics"]["task_completion_rate"]
        assert "mean" in task_completion
        assert "median" in task_completion
        assert "stdev" in task_completion
        assert "p25" in task_completion
        assert "p75" in task_completion

    def test_per_run_percentiles(self):
        """Per-run results include proper percentile distributions."""
        traces_by_session = {
            f"sess-{i}": [create_test_trace(f"t{i}")]
            for i in range(10)
        }

        run_results = compute_per_run_metrics(traces_by_session)

        # Pick any valid metric
        metric = run_results["metrics"]["task_completion_rate"]
        if metric["valid"]:
            # Median should be between p25 and p75
            assert metric["p25"] <= metric["median"] <= metric["p75"]
            # Min should be <= mean <= max
            assert metric["min"] <= metric["mean"] <= metric["max"]


# ─────────────────────────────────────────────────────
# TESTS: REGRESSION DETECTION
# ─────────────────────────────────────────────────────


class TestRegressionDetection:
    """Test regression detection against baseline."""

    def test_detector_loads_baseline(self):
        """RegressionDetector loads baseline successfully."""
        detector = RegressionDetector()
        assert detector.baseline is not None
        assert "metrics" in detector.baseline

    def test_baseline_comparison_ok(self):
        """Metrics within baseline produce 'ok' status."""
        detector = RegressionDetector()

        # Create run with metrics matching baseline
        traces_by_session = {
            f"sess-{i}": [create_test_trace(f"t{i}", with_correction=False)]
            for i in range(5)
        }

        run_results = compute_per_run_metrics(traces_by_session)
        report = detector.compare_to_baseline(run_results)

        # Most metrics should be ok
        ok_count = len(
            [c for c in report["comparisons"].values() if c["status"] == "ok"]
        )
        assert ok_count > 0

    def test_baseline_comparison_detects_regression(self):
        """Baseline comparison detects metric regressions."""
        detector = RegressionDetector()

        # Create run with many corrections (bad)
        traces_by_session = {
            f"sess-{i}": [create_test_trace(f"t{i}", with_correction=True)]
            for i in range(5)
        }

        run_results = compute_per_run_metrics(traces_by_session)
        report = detector.compare_to_baseline(run_results)

        # Correction_rate should be high (bad for lower-is-better metric)
        if "correction_rate" in report["comparisons"]:
            comp = report["comparisons"]["correction_rate"]
            # High correction rate is regression
            assert comp["status"] in ["warning", "rollback_candidate"]

    def test_run_to_run_comparison(self):
        """Can compare two runs for regressions/improvements."""
        detector = RegressionDetector()

        # Run A: few corrections
        run_a_traces = {
            f"sess-{i}": [create_test_trace(f"a{i}", with_correction=False)]
            for i in range(3)
        }
        run_a = compute_per_run_metrics(run_a_traces)

        # Run B: many corrections
        run_b_traces = {
            f"sess-{i}": [create_test_trace(f"b{i}", with_correction=True)]
            for i in range(3)
        }
        run_b = compute_per_run_metrics(run_b_traces)

        # Compare
        comparison = detector.compare_runs(run_a, run_b)

        # Should detect regression in correction_rate
        if "correction_rate" in comparison["comparisons"]:
            comp = comparison["comparisons"]["correction_rate"]
            assert comp["status"] == "regression"
            assert "correction_rate" in comparison["regressions"]

    def test_format_report(self):
        """Report formatting produces human-readable output."""
        detector = RegressionDetector()

        traces_by_session = {
            f"sess-{i}": [create_test_trace(f"t{i}")]
            for i in range(3)
        }

        run_results = compute_per_run_metrics(traces_by_session)
        report = detector.compare_to_baseline(run_results)

        formatted = detector.format_report(report)

        assert "BASELINE COMPARISON" in formatted
        assert "✅" in formatted  # Should have OK status


# ─────────────────────────────────────────────────────
# TESTS: NO AGENT DEPENDENCY
# ─────────────────────────────────────────────────────


class TestNoAgentDependency:
    """Verify evaluation is independent of agent runtime."""

    def test_metrics_import_only_evaluation_schemas(self):
        """Metric modules don't import agent code."""
        import evaluation.metrics.task_completion as task_module

        # Check that agent is not imported
        imports = dir(task_module)
        assert "agent" not in str(imports)

    def test_aggregation_pure_computation(self):
        """Aggregation is pure computation (no I/O except traces)."""
        traces_by_session = {
            f"sess-{i}": [create_test_trace(f"t{i}")]
            for i in range(5)
        }

        # Should complete without any agent interaction
        run_results = compute_per_run_metrics(traces_by_session)
        assert run_results is not None

    def test_metric_result_immutable(self):
        """MetricResult is immutable (frozen dataclass)."""
        result = MetricResult(
            metric_id="test",
            value=0.5,
            samples=10,
            valid=True,
        )

        # Should not allow mutation
        with pytest.raises(AttributeError):
            result.value = 0.6


# ─────────────────────────────────────────────────────
# TESTS: EDGE CASES
# ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_traces_handled(self):
        """Metrics handle empty trace lists gracefully."""
        result = compute_task_completion_rate([])

        assert not result.valid
        assert result.error is not None

    def test_single_trace(self):
        """Metrics work with single trace."""
        traces = [create_test_trace()]

        result = compute_task_completion_rate(traces)
        assert result.valid
        assert result.samples == 1

    def test_per_session_requires_same_session(self):
        """Per-session aggregation validates all traces are from same session."""
        traces = [
            create_test_trace("t1", conversation_id="sess-1"),
            create_test_trace("t2", conversation_id="sess-2"),  # Different session!
        ]

        with pytest.raises(ValueError):
            compute_per_session_metrics(traces)

    def test_baseline_missing_metric(self):
        """Comparison handles metrics not in baseline."""
        detector = RegressionDetector()

        # Create a result with extra metric
        traces = {
            "sess-1": [create_test_trace()]
        }
        run_results = compute_per_run_metrics(traces)

        report = detector.compare_to_baseline(run_results)

        # Should still complete without error
        assert "comparisons" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
