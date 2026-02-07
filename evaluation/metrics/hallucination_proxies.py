"""
Phase 5.2 metric extractors - Hallucination Risk Signals

Computes heuristic metrics for hallucination risk. These are proxies only.
No semantic analysis - only observable signals from Phase 4.2.
All functions are pure, deterministic, side-effect free.
"""

from typing import List
from evaluation.schemas.trace_schema import (
    Trace,
    EventType,
    get_all_events_of_type,
)
from evaluation.schemas.metric_schema import MetricResult


def compute_hallucination_proxy_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Heuristic rate of invented details or unsupported claims.

    Proxy signal: ERROR_RAISED events where agent cannot justify response.
    Note: This is a heuristic only. Actual hallucination requires semantic analysis.
    """
    if not traces:
        return MetricResult(
            metric_id="hallucination_proxy_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_responses = 0
    hallucination_signals = 0

    for trace in traces:
        # Count completed model calls
        completed_calls = get_all_events_of_type(
            trace, EventType.MODEL_CALL_COMPLETED
        )
        total_responses += len(completed_calls)

        # Count errors (proxy for hallucination signals)
        errors = get_all_events_of_type(trace, EventType.ERROR_RAISED)
        hallucination_signals += len(errors)

    hallucination_rate = (
        hallucination_signals / total_responses if total_responses > 0 else 0.0
    )

    return MetricResult(
        metric_id="hallucination_proxy_rate",
        value=hallucination_rate,
        samples=len(traces),
        valid=True,
    )


def compute_self_correction_frequency(traces: List[Trace]) -> MetricResult:
    """
    Compute: How often agent retracts or revises claims when questioned.

    Signal: Correction events following model output events.
    """
    if not traces:
        return MetricResult(
            metric_id="self_correction_frequency",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_responses = 0
    retraction_count = 0

    for trace in traces:
        # Count responses
        completed_calls = get_all_events_of_type(
            trace, EventType.MODEL_CALL_COMPLETED
        )
        total_responses += len(completed_calls)

        # Count corrections (proxy for retractions)
        corrections = get_all_events_of_type(trace, EventType.CORRECTION_RECEIVED)
        retraction_count += len(corrections)

    retraction_rate = (
        retraction_count / total_responses if total_responses > 0 else 0.0
    )

    return MetricResult(
        metric_id="self_correction_frequency",
        value=retraction_rate,
        samples=len(traces),
        valid=True,
    )


def compute_factual_consistency_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of responses without error markers.

    Heuristic: No ERROR_RAISED events = consistent output.
    Note: This is a weak proxy. Actual consistency requires semantic comparison.
    """
    if not traces:
        return MetricResult(
            metric_id="factual_consistency_rate",
            value=1.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_responses = 0
    error_free_responses = 0

    for trace in traces:
        # Count responses
        completed_calls = get_all_events_of_type(
            trace, EventType.MODEL_CALL_COMPLETED
        )
        total_responses += len(completed_calls)

        # Count responses without errors
        errors = get_all_events_of_type(trace, EventType.ERROR_RAISED)
        error_count = len(errors)

        error_free = max(0, len(completed_calls) - error_count)
        error_free_responses += error_free

    consistency_rate = (
        error_free_responses / total_responses if total_responses > 0 else 1.0
    )

    return MetricResult(
        metric_id="factual_consistency_rate",
        value=consistency_rate,
        samples=len(traces),
        valid=True,
    )


def compute_confidence_accuracy_correlation(traces: List[Trace]) -> MetricResult:
    """
    Compute: When agent expresses high confidence, how often is claim correct?

    Heuristic: High confidence = no errors in same span.
    Note: Requires agent to mark confidence levels in spans (metadata field).
    """
    if not traces:
        return MetricResult(
            metric_id="confidence_accuracy_correlation",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    high_confidence_spans = 0
    accurate_spans = 0

    for trace in traces:
        # Find spans with confidence metadata
        for span in trace.spans:
            confidence = span.metadata.get("confidence_level", None)

            # High confidence defined as >0.7
            if confidence and float(confidence) > 0.7:
                high_confidence_spans += 1

                # Check if span succeeded
                if span.status.value == "success":
                    accurate_spans += 1

    if high_confidence_spans > 0:
        correlation = accurate_spans / high_confidence_spans
    else:
        correlation = 0.0

    return MetricResult(
        metric_id="confidence_accuracy_correlation",
        value=correlation,
        samples=high_confidence_spans,
        valid=high_confidence_spans > 0,
    )


def compute_uncertainty_expression_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of responses that appropriately express uncertainty.

    Heuristic: Spans with explicit "uncertainty" or "confidence" metadata.
    Note: Requires agent to emit confidence_level in span metadata.
    """
    if not traces:
        return MetricResult(
            metric_id="uncertainty_expression_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_spans = 0
    uncertainty_expressed = 0

    for trace in traces:
        for span in trace.spans:
            total_spans += 1

            # Check for uncertainty/confidence metadata
            confidence = span.metadata.get("confidence_level", None)
            has_uncertainty = span.metadata.get("expresses_uncertainty", False)

            # Uncertainty expressed if confidence < 0.8 OR explicitly marked
            if (confidence and float(confidence) < 0.8) or has_uncertainty:
                uncertainty_expressed += 1

    uncertainty_rate = (
        uncertainty_expressed / total_spans if total_spans > 0 else 0.0
    )

    return MetricResult(
        metric_id="uncertainty_expression_rate",
        value=uncertainty_rate,
        samples=total_spans,
        valid=total_spans > 0,
    )
