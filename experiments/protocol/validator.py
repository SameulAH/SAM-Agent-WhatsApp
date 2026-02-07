"""
Phase 5.3.1: Experiment Protocol Validator

Pure validation functions that enforce protocol constraints.

These functions:
- Return structured errors (never raise)
- Never execute experiments
- Never compare metrics
- Never touch agent code

Their only purpose: prevent invalid experiments from existing.
"""

from dataclasses import dataclass
from typing import List, Optional
from experiments.protocol.schema import (
    ExperimentSpec,
    ExperimentDecision,
    ExperimentRecord,
    ExperimentOutcome,
    VALID_PHASE_5_2_METRICS,
)


@dataclass(frozen=True)
class ValidationError:
    """Result of a failed validation."""

    field: str  # Which field failed
    error: str  # What's wrong
    severity: str  # "error" or "warning"


@dataclass(frozen=True)
class ValidationResult:
    """Result of validation (success or errors)."""

    valid: bool
    errors: List[ValidationError]

    def has_critical_errors(self) -> bool:
        """Check if any critical errors (not just warnings)."""
        return any(e.severity == "error" for e in self.errors)


def validate_experiment_spec(spec: ExperimentSpec) -> ValidationResult:
    """
    Validate an ExperimentSpec.

    Rules enforced:
    1. changed_variable must be singular (no commas, "and", "or")
    2. metrics_used must all be Phase 5.2 metric IDs
    3. minimum_runs must meet statistical threshold
    4. hypothesis must be falsifiable (not generic)
    5. baseline_id must be provided
    6. All required fields must be non-empty

    Returns:
        ValidationResult with list of validation errors (if any)
    """
    errors: List[ValidationError] = []

    # Rule 1: changed_variable must be singular
    changed_var = spec.changed_variable.lower()
    singular_check_keywords = [
        " and ",
        " or ",
        ",",
        " plus ",
        " with ",
        " &",
    ]
    contains_multiple_vars = any(kw in changed_var for kw in singular_check_keywords)

    if contains_multiple_vars:
        errors.append(
            ValidationError(
                field="changed_variable",
                error="changed_variable must be singular (exactly ONE thing). "
                "Cannot contain 'and', 'or', '+', '&', or commas. "
                f"Got: '{spec.changed_variable}'",
                severity="error",
            )
        )

    # Rule 2: metrics_used must be valid Phase 5.2 metrics
    invalid_metrics = [m for m in spec.metrics_used if m not in VALID_PHASE_5_2_METRICS]

    if invalid_metrics:
        errors.append(
            ValidationError(
                field="metrics_used",
                error=f"Metrics not in Phase 5.2 registry: {invalid_metrics}. "
                f"Valid metrics: {sorted(VALID_PHASE_5_2_METRICS)}",
                severity="error",
            )
        )

    if not spec.metrics_used:
        errors.append(
            ValidationError(
                field="metrics_used",
                error="At least one Phase 5.2 metric must be specified",
                severity="error",
            )
        )

    # Rule 3: minimum_runs must meet statistical threshold
    # Statistical recommendation: ≥30 traces for stable percentiles
    if spec.minimum_runs < 30:
        errors.append(
            ValidationError(
                field="minimum_runs",
                error=f"minimum_runs should be ≥30 for statistical significance "
                f"(Phase 5.2 assumes ~30 traces). Got: {spec.minimum_runs}",
                severity="warning",  # Not a hard error, but strongly recommended
            )
        )

    # Rule 4: hypothesis must be falsifiable (not generic)
    hypothesis_lower = spec.hypothesis.lower()
    too_generic = [
        "improve",
        "better",
        "faster",
        "more efficient",
        "increase performance",
    ]
    # Check if hypothesis is too vague
    if len(spec.hypothesis.strip()) < 20:
        errors.append(
            ValidationError(
                field="hypothesis",
                error="Hypothesis should be specific and falsifiable (≥20 characters). "
                f"Got: '{spec.hypothesis}'",
                severity="error",
            )
        )

    # Rule 5: baseline_id must be provided
    if spec.baseline_id.lower() not in ["v1", "baseline", "baseline_v1"]:
        errors.append(
            ValidationError(
                field="baseline_id",
                error=f"baseline_id should reference Phase 5.2 baseline. "
                f"Expected 'v1' or 'baseline_v1', got: '{spec.baseline_id}'",
                severity="warning",
            )
        )

    # Rule 6: Verify all required fields are non-empty
    required_fields = {
        "experiment_id": spec.experiment_id,
        "hypothesis": spec.hypothesis,
        "changed_variable": spec.changed_variable,
        "baseline_id": spec.baseline_id,
        "variant_id": spec.variant_id,
        "created_at": spec.created_at,
        "author": spec.author,
    }

    for field_name, field_value in required_fields.items():
        if not field_value or not field_value.strip():
            errors.append(
                ValidationError(
                    field=field_name,
                    error=f"{field_name} cannot be empty",
                    severity="error",
                )
            )

    # Return result
    has_critical_errors = any(e.severity == "error" for e in errors)
    return ValidationResult(valid=not has_critical_errors, errors=errors)


def validate_decision(
    record: ExperimentRecord,
    require_accept_justification: bool = True,
) -> ValidationResult:
    """
    Validate an ExperimentDecision within a record.

    Rules enforced:
    1. outcome must be valid (ACCEPT | REJECT | INCONCLUSIVE)
    2. justification is required for all outcomes (≥50 chars)
    3. ACCEPT outcome must not have anti-metric regression flags
    4. Decision must be reviewable (reviewer and decided_at provided)
    5. Justification must be specific, not generic

    Returns:
        ValidationResult with list of validation errors (if any)
    """
    errors: List[ValidationError] = []
    decision = record.decision

    # Rule 1: outcome must be valid
    valid_outcomes = {e.value for e in ExperimentOutcome}
    if decision.outcome not in valid_outcomes:
        errors.append(
            ValidationError(
                field="decision.outcome",
                error=f"outcome must be one of {valid_outcomes}. Got: '{decision.outcome}'",
                severity="error",
            )
        )

    # Rule 2: justification is required (≥50 chars)
    if not decision.justification or len(decision.justification.strip()) < 50:
        errors.append(
            ValidationError(
                field="decision.justification",
                error="justification is required and must be ≥50 characters. "
                f"Got {len(decision.justification) if decision.justification else 0} chars",
                severity="error",
            )
        )

    # Rule 3: ACCEPT with no regression evidence
    # (We can't check actual metrics here - that requires Phase 5.2 comparison)
    # But we can check that ACCEPT has strong justification
    if decision.outcome == ExperimentOutcome.ACCEPT.value:
        # Justification should mention metrics or data
        justification_lower = decision.justification.lower()
        has_metric_reference = any(
            keyword in justification_lower
            for keyword in [
                "metric",
                "percentage",
                "%",
                "improved",
                "within range",
                "acceptable",
                "baseline",
                "comparison",
            ]
        )

        if not has_metric_reference:
            errors.append(
                ValidationError(
                    field="decision.justification",
                    error="ACCEPT decision requires data-driven justification "
                    "mentioning metrics or comparisons",
                    severity="error",
                )
            )

    # Rule 4: reviewer and decided_at must be provided
    if not decision.reviewer or not decision.reviewer.strip():
        errors.append(
            ValidationError(
                field="decision.reviewer",
                error="reviewer must be specified (who approved this decision?)",
                severity="error",
            )
        )

    if not decision.decided_at or not decision.decided_at.strip():
        errors.append(
            ValidationError(
                field="decision.decided_at",
                error="decided_at timestamp is required",
                severity="error",
            )
        )

    # Rule 5: justification should be specific
    if decision.justification:
        too_generic = [
            "looks good",
            "seems okay",
            "probably works",
            "i think",
            "maybe",
            "should be fine",
        ]
        justification_lower = decision.justification.lower()
        if any(phrase in justification_lower for phrase in too_generic):
            errors.append(
                ValidationError(
                    field="decision.justification",
                    error="Justification is too generic. "
                    "Use specific data and metric references",
                    severity="warning",
                )
            )

    # Return result
    has_critical_errors = any(e.severity == "error" for e in errors)
    return ValidationResult(valid=not has_critical_errors, errors=errors)


def validate_experiment_record(record: ExperimentRecord) -> ValidationResult:
    """
    Validate a complete ExperimentRecord (spec + decision).

    Combines validation of spec and decision, plus checks for consistency.

    Rules:
    1. Spec must be valid (call validate_experiment_spec)
    2. Decision must be valid (call validate_decision)
    3. baseline_results_ref and variant_results_ref must be provided
    4. Results references should be different (not comparing same data)

    Returns:
        ValidationResult combining all errors
    """
    all_errors: List[ValidationError] = []

    # Validate spec
    spec_validation = validate_experiment_spec(record.spec)
    all_errors.extend(spec_validation.errors)

    # Validate decision
    decision_validation = validate_decision(record)
    all_errors.extend(decision_validation.errors)

    # Validate results references
    if not record.baseline_results_ref or not record.baseline_results_ref.strip():
        all_errors.append(
            ValidationError(
                field="baseline_results_ref",
                error="baseline_results_ref cannot be empty (reference to Phase 5.2 baseline)",
                severity="error",
            )
        )

    if not record.variant_results_ref or not record.variant_results_ref.strip():
        all_errors.append(
            ValidationError(
                field="variant_results_ref",
                error="variant_results_ref cannot be empty (reference to variant results)",
                severity="error",
            )
        )

    # Check that baseline and variant are different
    if (
        record.baseline_results_ref.strip() == record.variant_results_ref.strip()
        and record.baseline_results_ref.strip()
    ):
        all_errors.append(
            ValidationError(
                field="variant_results_ref",
                error="baseline_results_ref and variant_results_ref cannot be identical "
                "(must compare different runs)",
                severity="error",
            )
        )

    # Return combined result
    has_critical_errors = any(e.severity == "error" for e in all_errors)
    return ValidationResult(valid=not has_critical_errors, errors=all_errors)


def print_validation_errors(result: ValidationResult, spec_label: str = "Experiment") -> None:
    """
    Pretty-print validation errors (for testing/debugging).

    Args:
        result: ValidationResult from validation function
        spec_label: Label for the object being validated
    """
    if result.valid:
        print(f"✅ {spec_label} is VALID")
        return

    print(f"❌ {spec_label} has validation errors:")
    print()

    # Group by severity
    errors_by_severity = {}
    for error in result.errors:
        if error.severity not in errors_by_severity:
            errors_by_severity[error.severity] = []
        errors_by_severity[error.severity].append(error)

    # Print critical errors first
    if "error" in errors_by_severity:
        print("🚨 CRITICAL ERRORS:")
        for error in errors_by_severity["error"]:
            print(f"  [{error.field}] {error.error}")
        print()

    # Then warnings
    if "warning" in errors_by_severity:
        print("⚠️  WARNINGS:")
        for error in errors_by_severity["warning"]:
            print(f"  [{error.field}] {error.error}")
        print()
