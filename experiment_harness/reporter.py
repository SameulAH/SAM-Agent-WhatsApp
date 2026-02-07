"""
Phase 5.3.2: Comparison Reporter

Compares baseline vs variant metrics and detects regressions.
Uses Phase 5.2 regression detector for analysis.

Responsibilities:
- Compare baseline vs variant metrics
- Use Phase 5.2 RegressionDetector
- Classify outcomes (improvement, regression, inconclusive)
- Generate human-readable comparison report
- Output evidence (no automatic decisions)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

from experiments.protocol.schema import ExperimentSpec, ExperimentOutcome
from experiment_harness.evaluator import MetricsResult
from evaluation.schemas.metric_schema import (
    get_metric_by_id,
    MetricDirection,
)


@dataclass(frozen=True)
class ReporterError(Exception):
    """Reporter error with structured information."""

    step: str  # "comparison", "classification"
    error_message: str
    experiment_id: Optional[str] = None
    details: Optional[str] = None

    def __str__(self) -> str:
        msg = f"[{self.step}] {self.error_message}"
        if self.details:
            msg += f"\n  Details: {self.details}"
        return msg


@dataclass(frozen=True)
class ComparisonReport:
    """Result of comparing baseline vs variant."""

    experiment_id: str
    baseline_run_id: str
    variant_run_id: str
    created_at: str
    metrics_compared: List[str]
    improvements: List[Dict[str, Any]]
    regressions: List[Dict[str, Any]]
    inconclusive: List[Dict[str, Any]]
    recommendation: str  # "ACCEPT", "REJECT", "INCONCLUSIVE" (recommendation only)
    evidence: Dict[str, Any]  # Detailed evidence for recommendation

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Experiment: {self.experiment_id}",
            f"Baseline Run: {self.baseline_run_id}",
            f"Variant Run: {self.variant_run_id}",
            f"Created: {self.created_at}",
            "",
            f"Metrics Compared: {len(self.metrics_compared)}",
            f"  Improvements: {len(self.improvements)}",
            f"  Regressions: {len(self.regressions)}",
            f"  Inconclusive: {len(self.inconclusive)}",
            "",
            f"Recommendation: {self.recommendation}",
            "(Note: This is evidence-based recommendation only.",
            " Final decision requires human review.)",
        ]
        return "\n".join(lines)


class ComparisonReporter:
    """Generate comparison reports."""

    def __init__(self):
        """Initialize reporter."""
        pass

    def compare_runs(
        self,
        spec: ExperimentSpec,
        baseline_result: MetricsResult,
        variant_result: MetricsResult,
    ) -> ComparisonReport:
        """
        Compare baseline vs variant metrics.

        Args:
            spec: ExperimentSpec (validated)
            baseline_result: MetricsResult for baseline
            variant_result: MetricsResult for variant

        Returns:
            ComparisonReport with analysis

        Raises:
            ReporterError: If comparison fails
        """
        try:
            improvements: List[Dict[str, Any]] = []
            regressions: List[Dict[str, Any]] = []
            inconclusive: List[Dict[str, Any]] = []
            metrics_compared: List[str] = []

            # Compare each metric in spec.metrics_used
            for metric_id in spec.metrics_used:
                baseline_metric = baseline_result.get_metric(metric_id)
                variant_metric = variant_result.get_metric(metric_id)

                if not baseline_metric or not variant_metric:
                    inconclusive.append({
                        "metric_id": metric_id,
                        "reason": "Missing baseline or variant result",
                    })
                    continue

                # Check metric validity
                if not baseline_metric.valid or not variant_metric.valid:
                    inconclusive.append({
                        "metric_id": metric_id,
                        "reason": f"Invalid metric results "
                                 f"(baseline: {baseline_metric.valid}, "
                                 f"variant: {variant_metric.valid})",
                    })
                    continue

                metrics_compared.append(metric_id)

                # Get metric definition
                metric_def = get_metric_by_id(metric_id)
                if not metric_def:
                    inconclusive.append({
                        "metric_id": metric_id,
                        "reason": "Metric definition not found",
                    })
                    continue

                # Determine if improvement or regression
                direction = metric_def.direction
                baseline_value = baseline_metric.value
                variant_value = variant_metric.value
                delta = variant_value - baseline_value

                comparison_entry = {
                    "metric_id": metric_id,
                    "metric_name": metric_def.name,
                    "baseline": baseline_value,
                    "variant": variant_value,
                    "delta": delta,
                    "direction": direction,
                    "acceptable_range": metric_def.acceptable_range,
                }

                # Check acceptable range
                min_acceptable, max_acceptable = metric_def.acceptable_range

                if direction == MetricDirection.HIGHER_IS_BETTER:
                    if variant_value > baseline_value:
                        comparison_entry["classification"] = "improvement"
                        improvements.append(comparison_entry)
                    elif variant_value < baseline_value:
                        comparison_entry["classification"] = "regression"
                        regressions.append(comparison_entry)
                    else:
                        comparison_entry["classification"] = "neutral"
                        inconclusive.append(comparison_entry)
                else:  # LOWER_IS_BETTER
                    if variant_value < baseline_value:
                        comparison_entry["classification"] = "improvement"
                        improvements.append(comparison_entry)
                    elif variant_value > baseline_value:
                        comparison_entry["classification"] = "regression"
                        regressions.append(comparison_entry)
                    else:
                        comparison_entry["classification"] = "neutral"
                        inconclusive.append(comparison_entry)

            # Generate recommendation (evidence-based, NOT final decision)
            recommendation = self._generate_recommendation(
                improvements=improvements,
                regressions=regressions,
                inconclusive=inconclusive,
                metrics_compared=metrics_compared,
            )

            # Build evidence
            evidence = {
                "improvements": improvements,
                "regressions": regressions,
                "inconclusive": inconclusive,
                "metrics_compared": metrics_compared,
                "improvement_count": len(improvements),
                "regression_count": len(regressions),
                "inconclusive_count": len(inconclusive),
                "total_metrics": len(spec.metrics_used),
                "note": "This is evidence-based recommendation only. "
                       "Final decision requires human review.",
            }

            return ComparisonReport(
                experiment_id=spec.experiment_id,
                baseline_run_id=baseline_result.run_id,
                variant_run_id=variant_result.run_id,
                created_at=datetime.utcnow().isoformat(),
                metrics_compared=metrics_compared,
                improvements=improvements,
                regressions=regressions,
                inconclusive=inconclusive,
                recommendation=recommendation,
                evidence=evidence,
            )

        except ReporterError:
            raise
        except Exception as e:
            raise ReporterError(
                step="comparison",
                error_message="Failed to compare runs",
                experiment_id=spec.experiment_id,
                details=str(e),
            )

    def _generate_recommendation(
        self,
        improvements: List[Dict[str, Any]],
        regressions: List[Dict[str, Any]],
        inconclusive: List[Dict[str, Any]],
        metrics_compared: List[str],
    ) -> str:
        """
        Generate recommendation based on comparison results.

        This is EVIDENCE-BASED RECOMMENDATION ONLY.
        Final decision requires human review (Phase 5.3.3).

        Args:
            improvements: List of improved metrics
            regressions: List of regressed metrics
            inconclusive: List of inconclusive metrics
            metrics_compared: Total metrics compared

        Returns:
            Recommendation string ("ACCEPT", "REJECT", "INCONCLUSIVE")
        """
        # REJECT if any regressions
        if regressions:
            return "REJECT"

        # INCONCLUSIVE if missing metrics or all neutral
        if inconclusive or not metrics_compared:
            return "INCONCLUSIVE"

        # ACCEPT if all improvements or all valid metrics pass
        if improvements and len(improvements) == len(metrics_compared):
            return "ACCEPT"

        # Default to INCONCLUSIVE if mixed signals
        return "INCONCLUSIVE"

    def format_report_markdown(self, report: ComparisonReport) -> str:
        """
        Format comparison report as Markdown.

        Args:
            report: ComparisonReport

        Returns:
            Markdown string
        """
        lines = [
            f"# Experiment Comparison Report",
            "",
            f"**Experiment ID:** {report.experiment_id}",
            f"**Baseline Run:** {report.baseline_run_id}",
            f"**Variant Run:** {report.variant_run_id}",
            f"**Created:** {report.created_at}",
            "",
            "## Summary",
            "",
            f"- **Metrics Compared:** {len(report.metrics_compared)}",
            f"- **Improvements:** {len(report.improvements)}",
            f"- **Regressions:** {len(report.regressions)}",
            f"- **Inconclusive:** {len(report.inconclusive)}",
            "",
            "## Recommendation",
            "",
            f"**{report.recommendation}**",
            "",
            "> ⚠️ This is an evidence-based recommendation only.",
            "> Final decision requires human review.",
            "",
        ]

        if report.improvements:
            lines.extend([
                "## Improvements ✅",
                "",
            ])
            for imp in report.improvements:
                lines.append(f"- **{imp['metric_name']}** ({imp['metric_id']})")
                lines.append(f"  - Baseline: {imp['baseline']}")
                lines.append(f"  - Variant: {imp['variant']}")
                lines.append(f"  - Delta: {imp['delta']:+.4f}")
                lines.append("")

        if report.regressions:
            lines.extend([
                "## Regressions ❌",
                "",
            ])
            for reg in report.regressions:
                lines.append(f"- **{reg['metric_name']}** ({reg['metric_id']})")
                lines.append(f"  - Baseline: {reg['baseline']}")
                lines.append(f"  - Variant: {reg['variant']}")
                lines.append(f"  - Delta: {reg['delta']:+.4f}")
                lines.append("")

        if report.inconclusive:
            lines.extend([
                "## Inconclusive ❓",
                "",
            ])
            for inc in report.inconclusive:
                if isinstance(inc, dict) and "metric_id" in inc:
                    lines.append(f"- {inc.get('metric_id', 'unknown')}: {inc.get('reason', 'unknown')}")
                else:
                    lines.append(f"- {inc}")
            lines.append("")

        lines.extend([
            "## Evidence",
            "",
            "Full comparison details stored in JSON format.",
            "Review evidence before making final decision.",
        ])

        return "\n".join(lines)
