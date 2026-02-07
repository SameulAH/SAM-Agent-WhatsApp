"""
Phase 5.2 metric extractors - Task Completion Effectiveness

Computes metrics related to how completely the agent addresses user needs.
All functions are pure, deterministic, side-effect free.
"""

from typing import List
from evaluation.schemas.trace_schema import (
    Trace,
    EventType,
    get_all_events_of_type,
    has_correction_marker,
)
from evaluation.schemas.metric_schema import MetricResult


def compute_task_completion_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of sessions where user confirms task solved.

    Signal: Terminal node reached without correction markers.
    """
    if not traces:
        return MetricResult(
            metric_id="task_completion_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    completed_count = 0

    for trace in traces:
        # Completion marker: response_generation_node executed AND no corrections
        response_events = get_all_events_of_type(
            trace, EventType.MODEL_CALL_COMPLETED
        )
        has_response = len(response_events) > 0
        has_correction = has_correction_marker(trace)

        if has_response and not has_correction:
            completed_count += 1

    completion_rate = completed_count / len(traces) if traces else 0.0

    return MetricResult(
        metric_id="task_completion_rate",
        value=completion_rate,
        samples=len(traces),
        valid=True,
    )


def compute_correction_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of turns where user corrects agent understanding.

    Signal: CORRECTION_RECEIVED events in conversation.
    """
    if not traces:
        return MetricResult(
            metric_id="correction_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_turns = 0
    correction_turns = 0

    for trace in traces:
        # Count turns (approximate as events)
        total_turns += len(trace.events)

        # Count correction turns
        correction_events = get_all_events_of_type(
            trace, EventType.CORRECTION_RECEIVED
        )
        correction_turns += len(correction_events)

    correction_rate = correction_turns / total_turns if total_turns > 0 else 0.0

    return MetricResult(
        metric_id="correction_rate",
        value=correction_rate,
        samples=len(traces),
        valid=True,
    )


def compute_follow_up_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of turns where user asks clarification question.

    Heuristic: Follow-ups inferred from correction events that are not retries.
    """
    if not traces:
        return MetricResult(
            metric_id="follow_up_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_events = 0
    clarification_events = 0

    for trace in traces:
        # Total events as proxy for turns
        total_events += len(trace.events)

        # Clarification: corrections but not retries
        corrections = get_all_events_of_type(trace, EventType.CORRECTION_RECEIVED)
        retries = get_all_events_of_type(trace, EventType.RETRY_REQUESTED)

        # Approximate: clarifications = corrections without retry
        clarification_count = max(0, len(corrections) - len(retries))
        clarification_events += clarification_count

    follow_up_rate = (
        clarification_events / total_events if total_events > 0 else 0.0
    )

    return MetricResult(
        metric_id="follow_up_rate",
        value=follow_up_rate,
        samples=len(traces),
        valid=True,
    )


def compute_turns_to_completion(traces: List[Trace]) -> MetricResult:
    """
    Compute: Average number of turns needed to reach user confirmation.

    Signal: Count events until terminal node with no corrections.
    """
    if not traces:
        return MetricResult(
            metric_id="turns_to_completion",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    completion_turns = []

    for trace in traces:
        # Completion: terminal node reached without later corrections
        if has_correction_marker(trace):
            # Still include in average (includes correction overhead)
            turn_count = len(trace.events)
        else:
            turn_count = len(trace.events)

        if turn_count > 0:
            completion_turns.append(turn_count)

    avg_turns = sum(completion_turns) / len(completion_turns) if completion_turns else 0.0

    return MetricResult(
        metric_id="turns_to_completion",
        value=avg_turns,
        samples=len(traces),
        valid=len(completion_turns) > 0,
    )
