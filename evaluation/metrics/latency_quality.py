"""
Phase 5.2 metric extractors - Latency-Quality Tradeoff

Computes metrics related to balancing response speed with answer quality.
All functions are pure, deterministic, side-effect free.
"""

from typing import List
from evaluation.schemas.trace_schema import (
    Trace,
    get_terminal_spans,
)
from evaluation.schemas.metric_schema import MetricResult


def compute_response_time_ms(traces: List[Trace]) -> MetricResult:
    """
    Compute: Median response time from user message to agent response.

    Signal: Duration of terminal node spans (response_generation_node, error_handler_node).
    """
    if not traces:
        return MetricResult(
            metric_id="response_time_ms",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    response_times = []

    for trace in traces:
        # Get terminal spans (where response is generated)
        terminal_spans = get_terminal_spans(trace)

        for span in terminal_spans:
            response_times.append(span.duration_ms)

    if not response_times:
        return MetricResult(
            metric_id="response_time_ms",
            value=0.0,
            samples=0,
            valid=False,
            error="No terminal spans found",
        )

    # Compute median
    sorted_times = sorted(response_times)
    median_idx = len(sorted_times) // 2
    median_time = sorted_times[median_idx]

    return MetricResult(
        metric_id="response_time_ms",
        value=median_time,
        samples=len(response_times),
        valid=True,
    )


def compute_quality_adjusted_response_time(traces: List[Trace]) -> MetricResult:
    """
    Compute: Ratio of turns-to-completion to response latency.

    Higher = better tradeoff (more work done per unit time).
    Formula: avg_turns / median_latency_ms * 1000
    """
    if not traces:
        return MetricResult(
            metric_id="quality_adjusted_response_time",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    # Get turns to completion
    total_turns = 0
    trace_count = 0

    for trace in traces:
        total_turns += len(trace.events)
        trace_count += 1

    avg_turns = total_turns / trace_count if trace_count > 0 else 0.0

    # Get median latency
    response_times = []
    for trace in traces:
        terminal_spans = get_terminal_spans(trace)
        for span in terminal_spans:
            response_times.append(span.duration_ms)

    if not response_times:
        return MetricResult(
            metric_id="quality_adjusted_response_time",
            value=0.0,
            samples=0,
            valid=False,
            error="No terminal spans found",
        )

    sorted_times = sorted(response_times)
    median_idx = len(sorted_times) // 2
    median_latency = sorted_times[median_idx]

    # Compute quality-adjusted time (higher is better)
    if median_latency > 0:
        quality_adjusted = (avg_turns / median_latency) * 1000
    else:
        quality_adjusted = 0.0

    return MetricResult(
        metric_id="quality_adjusted_response_time",
        value=quality_adjusted,
        samples=len(traces),
        valid=median_latency > 0,
    )


def compute_premature_optimization_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Responses that are fast but incomplete, requiring follow-up.

    Heuristic: Fast response (< 1000ms) followed by follow-up question.
    """
    if not traces:
        return MetricResult(
            metric_id="premature_optimization_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    fast_responses = 0
    premature_optimizations = 0

    for trace in traces:
        terminal_spans = get_terminal_spans(trace)

        for span in terminal_spans:
            # Fast response threshold: 1000ms
            if span.duration_ms < 1000:
                fast_responses += 1

                # Check if followed by follow-up question (from events)
                follow_up_events = [
                    e for e in trace.events if "follow_up" in str(e.event_type).lower()
                ]

                if follow_up_events:
                    premature_optimizations += 1

    premature_rate = (
        premature_optimizations / fast_responses if fast_responses > 0 else 0.0
    )

    return MetricResult(
        metric_id="premature_optimization_rate",
        value=premature_rate,
        samples=fast_responses,
        valid=fast_responses > 0,
    )


def compute_over_elaboration_rate(traces: List[Trace]) -> MetricResult:
    """
    Compute: Responses that take much longer but don't improve task completion.

    Heuristic: Slow response (> 3000ms) with no improvement in completion metrics.
    """
    if not traces:
        return MetricResult(
            metric_id="over_elaboration_rate",
            value=0.0,
            samples=0,
            valid=False,
            error="No traces provided",
        )

    slow_responses = 0
    over_elaborated = 0

    for trace in traces:
        terminal_spans = get_terminal_spans(trace)

        for span in terminal_spans:
            # Slow response threshold: 3000ms
            if span.duration_ms > 3000:
                slow_responses += 1

                # Check if trace has corrections (no improvement)
                corrections = [
                    e
                    for e in trace.events
                    if "correction" in str(e.event_type).lower()
                ]

                # Over-elaborated if slow AND still has corrections
                if corrections:
                    over_elaborated += 1

    over_elaboration = (
        over_elaborated / slow_responses if slow_responses > 0 else 0.0
    )

    return MetricResult(
        metric_id="over_elaboration_rate",
        value=over_elaboration,
        samples=slow_responses,
        valid=slow_responses > 0,
    )
