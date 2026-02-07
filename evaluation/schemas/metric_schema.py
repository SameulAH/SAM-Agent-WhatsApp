"""
Metric Schema - Self-describing metric definitions.

This module defines what each intelligence metric is, how to interpret it,
and what direction is "good".

Metrics are evaluative only - never operational.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class MetricDirection(str, Enum):
    """Direction that indicates improvement."""

    HIGHER_IS_BETTER = "higher"  # Increase is good
    LOWER_IS_BETTER = "lower"  # Decrease is good


class MetricType(str, Enum):
    """Type of metric value."""

    RATE = "rate"  # Percentage or ratio (0.0-1.0)
    COUNT = "count"  # Absolute count
    RATIO = "ratio"  # Numeric ratio (not bounded)
    DURATION = "duration"  # Milliseconds
    COMPOSITE = "composite"  # Combination of multiple metrics


@dataclass(frozen=True)
class MetricDefinition:
    """Self-describing metric specification."""

    metric_id: str  # Unique identifier
    dimension: str  # Which intelligence dimension
    name: str  # Human-readable name
    description: str  # What it measures
    metric_type: MetricType  # Type of value
    direction: MetricDirection  # What direction is good
    acceptable_range: tuple  # (min, max) for acceptable values
    anti_metrics: List[str]  # Related anti-metrics to watch


# DIMENSION 1: Task Completion Effectiveness

TASK_COMPLETION_RATE = MetricDefinition(
    metric_id="task_completion_rate",
    dimension="task_completion_effectiveness",
    name="Task Completion Rate",
    description="Percentage of sessions where user confirms task is solved",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.7, 1.0),
    anti_metrics=["correction_rate", "follow_up_rate", "turns_to_completion"],
)

CORRECTION_RATE = MetricDefinition(
    metric_id="correction_rate",
    dimension="task_completion_effectiveness",
    name="Correction Rate",
    description="Percentage of turns where user corrects agent understanding",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.1),
    anti_metrics=["task_completion_rate"],
)

FOLLOW_UP_RATE = MetricDefinition(
    metric_id="follow_up_rate",
    dimension="task_completion_effectiveness",
    name="Follow-Up Rate",
    description="Percentage of turns where user asks clarification question",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.15),
    anti_metrics=["task_completion_rate"],
)

TURNS_TO_COMPLETION = MetricDefinition(
    metric_id="turns_to_completion",
    dimension="task_completion_effectiveness",
    name="Turns to Completion",
    description="Average number of turns needed to reach user confirmation",
    metric_type=MetricType.COUNT,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(1.0, 5.0),
    anti_metrics=["correction_rate", "follow_up_rate"],
)


# DIMENSION 2: Correction & Retry Pressure

RETRY_RATE = MetricDefinition(
    metric_id="retry_rate",
    dimension="correction_retry_pressure",
    name="Retry Rate",
    description="Percentage of turns where user retries/rephrases due to unsatisfactory answer",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.08),
    anti_metrics=["task_completion_rate"],
)

CORRECTION_FEEDBACK_RATE = MetricDefinition(
    metric_id="correction_feedback_rate",
    dimension="correction_retry_pressure",
    name="Correction Feedback Rate",
    description="Percentage of turns where user provides explicit correction",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.05),
    anti_metrics=["task_completion_rate"],
)

RECOVERY_TURNS = MetricDefinition(
    metric_id="recovery_turns",
    dimension="correction_retry_pressure",
    name="Recovery Turns",
    description="Average turns needed after a failed attempt to recover",
    metric_type=MetricType.COUNT,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 3.0),
    anti_metrics=["retry_rate"],
)

BACKTRACKING_FREQUENCY = MetricDefinition(
    metric_id="backtracking_frequency",
    dimension="correction_retry_pressure",
    name="Backtracking Frequency",
    description="How often user must restart or re-explain context",
    metric_type=MetricType.COUNT,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 2.0),
    anti_metrics=["retry_rate"],
)


# DIMENSION 3: Memory Usefulness (Advisory Only)

MEMORY_REFERENCE_RATE = MetricDefinition(
    metric_id="memory_reference_rate",
    dimension="memory_usefulness_advisory",
    name="Memory Reference Rate",
    description="Percentage of turns where agent uses retrieved memory (explicit or implicit)",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.3, 1.0),
    anti_metrics=["memory_irrelevance_rate"],
)

MEMORY_IMPROVEMENT_RATE = MetricDefinition(
    metric_id="memory_improvement_rate",
    dimension="memory_usefulness_advisory",
    name="Memory Improvement Rate",
    description="Percentage of sessions improved by memory use (no correction after memory read)",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.5, 1.0),
    anti_metrics=["correction_rate"],
)

REPETITION_REDUCTION_RATE = MetricDefinition(
    metric_id="repetition_reduction_rate",
    dimension="memory_usefulness_advisory",
    name="Repetition Reduction Rate",
    description="Percentage of unique information vs total information in conversation",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.7, 1.0),
    anti_metrics=[],
)


# DIMENSION 4: Hallucination Risk Signals

FACTUAL_CONSISTENCY_RATE = MetricDefinition(
    metric_id="factual_consistency_rate",
    dimension="hallucination_risk_signals",
    name="Factual Consistency Rate",
    description="Percentage of specific claims verifiable against context or marked as uncertain",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.85, 1.0),
    anti_metrics=["contradiction_rate", "unsupported_claim_rate"],
)

CONFIDENCE_ACCURACY_CORRELATION = MetricDefinition(
    metric_id="confidence_accuracy_correlation",
    dimension="hallucination_risk_signals",
    name="Confidence-Accuracy Correlation",
    description="When agent expresses high confidence, how often is the claim correct?",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.8, 1.0),
    anti_metrics=["hallucination_rate"],
)

HALLUCINATION_PROXY_RATE = MetricDefinition(
    metric_id="hallucination_proxy_rate",
    dimension="hallucination_risk_signals",
    name="Hallucination Proxy Rate",
    description="Heuristic rate of invented details or unsupported claims",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.05),
    anti_metrics=["factual_consistency_rate"],
)

SELF_CORRECTION_FREQUENCY = MetricDefinition(
    metric_id="self_correction_frequency",
    dimension="hallucination_risk_signals",
    name="Self-Correction Frequency",
    description="How often agent retracts or revises claims when questioned",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.1),
    anti_metrics=["factual_consistency_rate"],
)

UNCERTAINTY_EXPRESSION_RATE = MetricDefinition(
    metric_id="uncertainty_expression_rate",
    dimension="hallucination_risk_signals",
    name="Uncertainty Expression Rate",
    description="Percentage of answers where agent appropriately expresses confidence bounds",
    metric_type=MetricType.RATE,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.3, 1.0),
    anti_metrics=[],
)


# DIMENSION 5: Latency-Quality Tradeoff

RESPONSE_TIME_MS = MetricDefinition(
    metric_id="response_time_ms",
    dimension="latency_quality_tradeoff",
    name="Response Time",
    description="Median time from user message to agent response in milliseconds",
    metric_type=MetricType.DURATION,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(500.0, 5000.0),
    anti_metrics=["quality_per_second"],
)

QUALITY_ADJUSTED_RESPONSE_TIME = MetricDefinition(
    metric_id="quality_adjusted_response_time",
    dimension="latency_quality_tradeoff",
    name="Quality-Adjusted Response Time",
    description="Ratio of turns-to-completion to response latency (higher = better tradeoff)",
    metric_type=MetricType.RATIO,
    direction=MetricDirection.HIGHER_IS_BETTER,
    acceptable_range=(0.0, 1000.0),
    anti_metrics=[],
)

PREMATURE_OPTIMIZATION_RATE = MetricDefinition(
    metric_id="premature_optimization_rate",
    dimension="latency_quality_tradeoff",
    name="Premature Optimization Rate",
    description="Responses that are fast but incomplete, requiring follow-up",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.1),
    anti_metrics=["response_time_ms"],
)

OVER_ELABORATION_RATE = MetricDefinition(
    metric_id="over_elaboration_rate",
    dimension="latency_quality_tradeoff",
    name="Over-Elaboration Rate",
    description="Responses that take much longer but don't improve task completion",
    metric_type=MetricType.RATE,
    direction=MetricDirection.LOWER_IS_BETTER,
    acceptable_range=(0.0, 0.1),
    anti_metrics=["response_time_ms"],
)


# All metric definitions registry
ALL_METRICS = [
    # Task Completion
    TASK_COMPLETION_RATE,
    CORRECTION_RATE,
    FOLLOW_UP_RATE,
    TURNS_TO_COMPLETION,
    # Retry Pressure
    RETRY_RATE,
    CORRECTION_FEEDBACK_RATE,
    RECOVERY_TURNS,
    BACKTRACKING_FREQUENCY,
    # Memory Usefulness
    MEMORY_REFERENCE_RATE,
    MEMORY_IMPROVEMENT_RATE,
    REPETITION_REDUCTION_RATE,
    # Hallucination Risk
    FACTUAL_CONSISTENCY_RATE,
    CONFIDENCE_ACCURACY_CORRELATION,
    HALLUCINATION_PROXY_RATE,
    SELF_CORRECTION_FREQUENCY,
    UNCERTAINTY_EXPRESSION_RATE,
    # Latency-Quality
    RESPONSE_TIME_MS,
    QUALITY_ADJUSTED_RESPONSE_TIME,
    PREMATURE_OPTIMIZATION_RATE,
    OVER_ELABORATION_RATE,
]


def get_metric_by_id(metric_id: str) -> Optional[MetricDefinition]:
    """Look up a metric definition by ID."""
    for metric in ALL_METRICS:
        if metric.metric_id == metric_id:
            return metric
    return None


def get_metrics_for_dimension(dimension: str) -> List[MetricDefinition]:
    """Get all metrics for a given intelligence dimension."""
    return [m for m in ALL_METRICS if m.dimension == dimension]


@dataclass(frozen=True)
class MetricResult:
    """Result of computing a single metric."""

    metric_id: str  # Which metric
    value: float  # Computed value
    samples: int  # Number of data points
    valid: bool  # Whether result is valid
    error: Optional[str] = None  # If invalid, why


@dataclass(frozen=True)
class ComputationResults:
    """Results of computing all metrics for a session."""

    session_id: str  # Session identifier
    results: dict  # metric_id -> MetricResult
    timestamp: str  # When computed

    def get_metric(self, metric_id: str) -> Optional[MetricResult]:
        """Get a specific metric result."""
        return self.results.get(metric_id)
