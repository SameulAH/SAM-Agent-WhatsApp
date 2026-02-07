"""
Phase 5.2 aggregation - Per-Session Metrics

Aggregates all metrics for a single session (conversation).
Produces one record per conversation_id.
"""

from typing import List, Dict, Any
from datetime import datetime
from evaluation.schemas.trace_schema import Trace
from evaluation.schemas.metric_schema import ComputationResults, MetricResult

# Import all metric functions
from evaluation.metrics.task_completion import (
    compute_task_completion_rate,
    compute_correction_rate,
    compute_follow_up_rate,
    compute_turns_to_completion,
)
from evaluation.metrics.retry_pressure import (
    compute_retry_rate,
    compute_correction_feedback_rate,
    compute_recovery_turns,
    compute_backtracking_frequency,
)
from evaluation.metrics.memory_usefulness import (
    compute_memory_reference_rate,
    compute_memory_improvement_rate,
    compute_repetition_reduction_rate,
)
from evaluation.metrics.hallucination_proxies import (
    compute_hallucination_proxy_rate,
    compute_self_correction_frequency,
    compute_factual_consistency_rate,
    compute_confidence_accuracy_correlation,
    compute_uncertainty_expression_rate,
)
from evaluation.metrics.latency_quality import (
    compute_response_time_ms,
    compute_quality_adjusted_response_time,
    compute_premature_optimization_rate,
    compute_over_elaboration_rate,
)


def compute_per_session_metrics(traces: List[Trace]) -> ComputationResults:
    """
    Compute all intelligence metrics for a session.

    Args:
        traces: List of Trace objects for a single session

    Returns:
        ComputationResults containing all computed metrics
    """
    if not traces:
        raise ValueError("Must provide at least one trace")

    session_id = traces[0].conversation_id

    # Ensure all traces are from same session
    for trace in traces:
        if trace.conversation_id != session_id:
            raise ValueError("All traces must be from same session")

    # Compute all metrics
    results = {
        # Task Completion
        "task_completion_rate": compute_task_completion_rate(traces),
        "correction_rate": compute_correction_rate(traces),
        "follow_up_rate": compute_follow_up_rate(traces),
        "turns_to_completion": compute_turns_to_completion(traces),
        # Retry Pressure
        "retry_rate": compute_retry_rate(traces),
        "correction_feedback_rate": compute_correction_feedback_rate(traces),
        "recovery_turns": compute_recovery_turns(traces),
        "backtracking_frequency": compute_backtracking_frequency(traces),
        # Memory Usefulness
        "memory_reference_rate": compute_memory_reference_rate(traces),
        "memory_improvement_rate": compute_memory_improvement_rate(traces),
        "repetition_reduction_rate": compute_repetition_reduction_rate(traces),
        # Hallucination Risk
        "hallucination_proxy_rate": compute_hallucination_proxy_rate(traces),
        "self_correction_frequency": compute_self_correction_frequency(traces),
        "factual_consistency_rate": compute_factual_consistency_rate(traces),
        "confidence_accuracy_correlation": compute_confidence_accuracy_correlation(traces),
        "uncertainty_expression_rate": compute_uncertainty_expression_rate(traces),
        # Latency-Quality
        "response_time_ms": compute_response_time_ms(traces),
        "quality_adjusted_response_time": compute_quality_adjusted_response_time(traces),
        "premature_optimization_rate": compute_premature_optimization_rate(traces),
        "over_elaboration_rate": compute_over_elaboration_rate(traces),
    }

    return ComputationResults(
        session_id=session_id,
        results=results,
        timestamp=datetime.utcnow().isoformat(),
    )


def to_dict(computation_results: ComputationResults) -> Dict[str, Any]:
    """Convert ComputationResults to JSON-serializable dictionary."""
    return {
        "session_id": computation_results.session_id,
        "timestamp": computation_results.timestamp,
        "metrics": {
            metric_id: {
                "value": result.value,
                "samples": result.samples,
                "valid": result.valid,
                "error": result.error,
            }
            for metric_id, result in computation_results.results.items()
        },
    }
