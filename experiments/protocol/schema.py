"""
Phase 5.3.1: Experiment Protocol Schema

Immutable dataclass definitions for experiments.
All fields are frozen (immutable). No execution logic.

This schema makes experiments first-class, inspectable objects.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ExperimentOutcome(str, Enum):
    """Valid experiment decision outcomes."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ExperimentSpec:
    """
    Specification for a proposed experiment.

    This is the declaration of what will be tested before any evaluation occurs.

    Fields:
        experiment_id: Unique identifier (e.g., "exp-001")
        hypothesis: Falsifiable prediction about the change
        changed_variable: The single variable being modified (REQUIRED, singular)
        baseline_id: Reference to Phase 5.2 baseline (e.g., "v1")
        variant_id: Label for this variant (e.g., "variant-1", "prompt-v2")
        metrics_used: List of Phase 5.2 metric IDs that will be evaluated
        minimum_runs: Minimum number of traces per variant (≥30 recommended)
        created_at: ISO 8601 timestamp of creation
        author: Who proposed this experiment
    """

    experiment_id: str
    hypothesis: str
    changed_variable: str
    baseline_id: str
    variant_id: str
    metrics_used: List[str]
    minimum_runs: int
    created_at: str
    author: str

    def __post_init__(self):
        """Validate basic constraints at instantiation."""
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty")
        if not self.hypothesis or len(self.hypothesis.strip()) < 10:
            raise ValueError("hypothesis must be at least 10 characters")
        if not self.changed_variable or not self.changed_variable.strip():
            raise ValueError("changed_variable cannot be empty")
        if not self.baseline_id or not self.baseline_id.strip():
            raise ValueError("baseline_id cannot be empty")
        if not self.variant_id or not self.variant_id.strip():
            raise ValueError("variant_id cannot be empty")
        if not self.metrics_used or len(self.metrics_used) == 0:
            raise ValueError("metrics_used cannot be empty")
        if self.minimum_runs < 1:
            raise ValueError("minimum_runs must be at least 1")
        if not self.created_at or not self.created_at.strip():
            raise ValueError("created_at cannot be empty")
        if not self.author or not self.author.strip():
            raise ValueError("author cannot be empty")


@dataclass(frozen=True)
class ExperimentDecision:
    """
    Decision outcome for a completed experiment.

    Fields:
        outcome: ACCEPT | REJECT | INCONCLUSIVE
        justification: Why this outcome? (required for all outcomes, min 50 chars)
        reviewer: Who reviewed and approved this decision
        decided_at: ISO 8601 timestamp of decision
    """

    outcome: str  # Must be one of ExperimentOutcome values
    justification: str
    reviewer: str
    decided_at: str

    def __post_init__(self):
        """Validate decision constraints."""
        if self.outcome not in {e.value for e in ExperimentOutcome}:
            raise ValueError(
                f"outcome must be one of {[e.value for e in ExperimentOutcome]}, got {self.outcome}"
            )
        if not self.justification or len(self.justification.strip()) < 50:
            raise ValueError(
                "justification is required and must be at least 50 characters"
            )
        if not self.reviewer or not self.reviewer.strip():
            raise ValueError("reviewer cannot be empty")
        if not self.decided_at or not self.decided_at.strip():
            raise ValueError("decided_at cannot be empty")


@dataclass(frozen=True)
class ExperimentRecord:
    """
    Complete audit trail for an experiment.

    This is the immutable record of:
    - What was proposed (spec)
    - What was tested (results references)
    - What was decided (decision)

    Fields:
        spec: The experiment specification
        baseline_results_ref: Path/ID to baseline metrics (Phase 5.2)
        variant_results_ref: Path/ID to variant results
        decision: The decision outcome and justification
    """

    spec: ExperimentSpec
    baseline_results_ref: str
    variant_results_ref: str
    decision: ExperimentDecision

    def __post_init__(self):
        """Validate record constraints."""
        if not self.baseline_results_ref or not self.baseline_results_ref.strip():
            raise ValueError("baseline_results_ref cannot be empty")
        if not self.variant_results_ref or not self.variant_results_ref.strip():
            raise ValueError("variant_results_ref cannot be empty")
        # spec and decision are validated by their own __post_init__


@dataclass(frozen=True)
class ExperimentRegistryEntry:
    """
    Minimal registry entry for traceability.

    Used in experiments/registry/experiments.json as append-only log.

    Fields:
        experiment_id: Unique identifier
        status: "proposed" | "evaluated" | "accepted" | "rejected"
        created_at: When proposed
        decided_at: When decision made (null if still proposed)
    """

    experiment_id: str
    status: str  # "proposed" | "evaluated" | "accepted" | "rejected"
    created_at: str
    decided_at: Optional[str] = None

    def __post_init__(self):
        """Validate registry entry."""
        if not self.experiment_id or not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty")
        valid_statuses = {"proposed", "evaluated", "accepted", "rejected"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got {self.status}"
            )
        if not self.created_at or not self.created_at.strip():
            raise ValueError("created_at cannot be empty")
        if self.decided_at is not None and not self.decided_at.strip():
            raise ValueError("decided_at must be non-empty if provided")


# Registry of valid Phase 5.2 metric IDs
# This is imported from evaluation/schemas/metric_schema.py
# For now, we define the expected metrics (must match Phase 5.2)

VALID_PHASE_5_2_METRICS = {
    # Task Completion Effectiveness
    "task_completion_rate",
    "correction_rate",
    "follow_up_rate",
    "turns_to_completion",
    # Correction & Retry Pressure
    "retry_rate",
    "correction_feedback_rate",
    "recovery_turns",
    "backtracking_frequency",
    # Memory Usefulness (Advisory)
    "memory_reference_rate",
    "memory_improvement_rate",
    "repetition_reduction_rate",
    # Hallucination Risk Signals
    "hallucination_proxy_rate",
    "self_correction_frequency",
    "factual_consistency_rate",
    "confidence_accuracy_correlation",
    "uncertainty_expression_rate",
    # Latency-Quality Tradeoff
    "response_time_ms",
    "quality_adjusted_response_time",
    "premature_optimization_rate",
    "over_elaboration_rate",
}


def is_valid_phase_5_2_metric(metric_id: str) -> bool:
    """Check if metric_id is registered in Phase 5.2."""
    return metric_id in VALID_PHASE_5_2_METRICS
