"""
Phase 5.2 metric extractors - Retry Pressure & Correction Overhead

Computes metrics related to how often users must retry or correct the agent.
All functions are pure, deterministic, side-effect free.
"""

from typing import List
from evaluation.schemas.trace_schema import (
    Trace,
    EventType,
    get_all_events_of_type,
)
from evaluation.schemas.metric_schema import MetricResult


def compute_retry_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of turns where user retries/rephrases due to unsatisfactory answer.

    Signal: RETRY_REQUESTED events in traces.
    """
    if not traces:
        return MetricResult(
            metric_id="retry_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_turns = 0
    retry_turns = 0

    for trace in traces:
        # Count turns (approximate as events)
        total_turns += len(trace.events)

        # Count retry turns
        retry_events = get_all_events_of_type(trace, EventType.RETRY_REQUESTED)
        retry_turns += len(retry_events)

    retry_rate = retry_turns / total_turns if total_turns > 0 else 0.0

    return MetricResult(
        metric_id="retry_rate",
        value=retry_rate,
        samples=len(traces),
        valid=True,
    )


def compute_correction_feedback_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of turns where user provides explicit correction.

    Signal: CORRECTION_RECEIVED events in traces.
    """
    if not traces:
        return MetricResult(
            metric_id="correction_feedback_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_turns = 0
    correction_turns = 0

    for trace in traces:
        total_turns += len(trace.events)

        # Count corrections
        correction_events = get_all_events_of_type(
            trace, EventType.CORRECTION_RECEIVED
        )
        correction_turns += len(correction_events)

    correction_rate = correction_turns / total_turns if total_turns > 0 else 0.0

    return MetricResult(
        metric_id="correction_feedback_rate",
        value=correction_rate,
        samples=len(traces),
        valid=True,
    )


def compute_recovery_turns(traces: List[Trace]) -> MetricResult:
    """
    Compute: Average turns needed after a failed attempt to recover.

    Heuristic: Turns between retry event and next successful completion.
    """
    if not traces:
        return MetricResult(
            metric_id="recovery_turns",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    recovery_turn_counts = []

    for trace in traces:
        retry_events = get_all_events_of_type(trace, EventType.RETRY_REQUESTED)

        # For each retry, measure turns until next correction or completion
        for retry_event in retry_events:
            # Find events after retry
            retry_turn_number = retry_event.turn_number or 0

            # Count events until next correction
            events_after_retry = [
                e
                for e in trace.events
                if (e.turn_number or 0) > retry_turn_number
            ]

            if events_after_retry:
                recovery_turn_counts.append(len(events_after_retry))

    avg_recovery = (
        sum(recovery_turn_counts) / len(recovery_turn_counts)
        if recovery_turn_counts
        else 0.0
    )

    return MetricResult(
        metric_id="recovery_turns",
        value=avg_recovery,
        samples=len(recovery_turn_counts),
        valid=len(recovery_turn_counts) > 0,
    )


def compute_backtracking_frequency(traces: List[Trace]) -> MetricResult:
    """
    Compute: How often user must restart or re-explain context.

    Heuristic: Multiple retries in same trace indicates backtracking.
    """
    if not traces:
        return MetricResult(
            metric_id="backtracking_frequency",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    backtracking_counts = []

    for trace in traces:
        # Count retry and correction combinations
        retry_count = len(get_all_events_of_type(trace, EventType.RETRY_REQUESTED))
        correction_count = len(
            get_all_events_of_type(trace, EventType.CORRECTION_RECEIVED)
        )

        # Backtracking: multiple retries + corrections
        backtrack_score = min(retry_count + correction_count, 10)  # Cap at 10
        backtracking_counts.append(backtrack_score)

    avg_backtracking = (
        sum(backtracking_counts) / len(backtracking_counts)
        if backtracking_counts
        else 0.0
    )

    return MetricResult(
        metric_id="backtracking_frequency",
        value=avg_backtracking,
        samples=len(traces),
        valid=True,
    )
