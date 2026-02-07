"""
Phase 5.2 metric extractors - Memory Usefulness (Advisory Only)

Computes metrics related to how effectively agent uses retrieved memory.
Memory is advisory only - never authoritative.
All functions are pure, deterministic, side-effect free.
"""

from typing import List
from evaluation.schemas.trace_schema import (
    Trace,
    EventType,
    get_all_events_of_type,
)
from evaluation.schemas.metric_schema import MetricResult


def compute_memory_reference_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of turns where agent reads from memory.

    Signal: MEMORY_READ_ATTEMPTED or LONG_TERM_MEMORY_READ_ATTEMPTED events.
    """
    if not traces:
        return MetricResult(
            metric_id="memory_reference_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    total_turns = 0
    memory_read_turns = 0

    for trace in traces:
        # Total turns
        total_turns += len(trace.events)

        # Count memory reads (short-term or long-term)
        short_term_reads = get_all_events_of_type(
            trace, EventType.MEMORY_READ_ATTEMPTED
        )
        long_term_reads = get_all_events_of_type(
            trace, EventType.LONG_TERM_MEMORY_READ_ATTEMPTED
        )

        memory_read_turns += len(short_term_reads) + len(long_term_reads)

    reference_rate = memory_read_turns / total_turns if total_turns > 0 else 0.0

    return MetricResult(
        metric_id="memory_reference_rate",
        value=reference_rate,
        samples=len(traces),
        valid=True,
    )


def compute_memory_improvement_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of sessions improved by memory use.

    Heuristic: Memory read followed by no corrections = memory helped.
    """
    if not traces:
        return MetricResult(
            metric_id="memory_improvement_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    improved_sessions = 0

    for trace in traces:
        # Check if memory was read
        short_term_reads = get_all_events_of_type(
            trace, EventType.MEMORY_READ_ATTEMPTED
        )
        long_term_reads = get_all_events_of_type(
            trace, EventType.LONG_TERM_MEMORY_READ_ATTEMPTED
        )

        memory_was_used = len(short_term_reads) > 0 or len(long_term_reads) > 0

        # Check if no corrections after memory use
        corrections = get_all_events_of_type(trace, EventType.CORRECTION_RECEIVED)
        no_corrections = len(corrections) == 0

        # Improved if memory used AND no corrections
        if memory_was_used and no_corrections:
            improved_sessions += 1

    improvement_rate = improved_sessions / len(traces) if traces else 0.0

    return MetricResult(
        metric_id="memory_improvement_rate",
        value=improvement_rate,
        samples=len(traces),
        valid=True,
    )


def compute_repetition_reduction_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Percentage of unique information vs total information in conversation.

    Heuristic: Count unique event types and content to estimate uniqueness.
    Note: This is a proxy metric. Full text analysis would require semantic analysis.
    """
    if not traces:
        return MetricResult(
            metric_id="repetition_reduction_rate",
            value=1.0,  # No traces = no repetition
            samples=0,
            valid=False,
            error="No traces provided",
        )

    uniqueness_scores = []

    for trace in traces:
        # Proxy: Count unique event types in trace
        event_types = set(e.event_type for e in trace.events)
        total_events = len(trace.events)

        if total_events > 0:
            # More unique event types = less repetition
            unique_ratio = len(event_types) / total_events
            uniqueness_scores.append(unique_ratio)

    avg_uniqueness = (
        sum(uniqueness_scores) / len(uniqueness_scores)
        if uniqueness_scores
        else 1.0
    )

    return MetricResult(
        metric_id="repetition_reduction_rate",
        value=avg_uniqueness,
        samples=len(traces),
        valid=len(uniqueness_scores) > 0,
    )
