"""
Phase 5.2 comparison - Run Comparison & Regression Detection

Compares two runs and detects regressions against baseline.
Reports only - no auto-rollback, no operational feedback.
"""

import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from evaluation.aggregation.per_run import compute_per_run_metrics, extract_metric_values
from evaluation.schemas.trace_schema import Trace


class RegressionDetector:
    """Detects metric regressions between runs."""

    def __init__(self, baseline_path: str = "evaluation/baselines/baseline_v1.json"):
        """Load baseline metrics."""
        with open(baseline_path, "r") as f:
            self.baseline = json.load(f)
        self.baseline_metrics = self.baseline.get("metrics", {})

    def compare_to_baseline(
        self, run_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare a run against baseline and detect regressions.

        Args:
            run_results: Output from compute_per_run_metrics

        Returns:
            Comparison report with warning and rollback flags
        """
        report = {
            "baseline_id": self.baseline.get("baseline_id"),
            "run_id": run_results.get("run_id"),
            "session_count": run_results.get("session_count"),
            "comparisons": {},
            "warnings": [],
            "rollback_candidates": [],
        }

        run_metrics = run_results.get("metrics", {})

        for metric_id, baseline_spec in self.baseline_metrics.items():
            if metric_id not in run_metrics:
                report["comparisons"][metric_id] = {
                    "status": "missing",
                    "message": "Metric not computed in this run",
                }
                continue

            run_metric = run_metrics[metric_id]

            if not run_metric.get("valid", False):
                report["comparisons"][metric_id] = {
                    "status": "invalid",
                    "message": run_metric.get("error", "Unknown error"),
                }
                continue

            # Get values
            baseline_mean = baseline_spec.get("baseline_mean", 0.0)
            run_mean = run_metric.get("mean", 0.0)
            warning_thresh = baseline_spec.get("warning_threshold", 0.0)
            rollback_thresh = baseline_spec.get("rollback_threshold", 0.0)

            # Compute difference
            diff = run_mean - baseline_mean
            direction = baseline_spec.get("direction", "higher")

            # Determine if "lower is better" (diff should be negative or positive)
            is_lower_better = direction == "lower"

            # Check thresholds
            status = "ok"
            message = f"Baseline: {baseline_mean:.4f}, Run: {run_mean:.4f}, Diff: {diff:.4f}"

            # For "lower is better" metrics: positive diff is bad
            # For "higher is better" metrics: negative diff is bad
            regression_severity = diff if is_lower_better else -diff

            if abs(regression_severity) >= abs(rollback_thresh):
                status = "rollback_candidate"
                message += f" [ROLLBACK: exceeded threshold {rollback_thresh}]"
                report["rollback_candidates"].append(metric_id)
            elif abs(regression_severity) >= abs(warning_thresh):
                status = "warning"
                message += f" [WARNING: exceeded threshold {warning_thresh}]"
                report["warnings"].append(metric_id)

            report["comparisons"][metric_id] = {
                "status": status,
                "baseline_mean": baseline_mean,
                "run_mean": run_mean,
                "diff": diff,
                "message": message,
                "direction": direction,
                "samples": run_metric.get("count", 0),
                "stdev": run_metric.get("stdev", 0.0),
            }

        return report

    def compare_runs(
        self, run_a_results: Dict[str, Any], run_b_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two runs and detect regressions.

        Args:
            run_a_results: Output from compute_per_run_metrics for run A
            run_b_results: Output from compute_per_run_metrics for run B

        Returns:
            Comparison report
        """
        report = {
            "run_a_id": run_a_results.get("run_id"),
            "run_b_id": run_b_results.get("run_id"),
            "comparisons": {},
            "regressions": [],
            "improvements": [],
        }

        run_a_metrics = run_a_results.get("metrics", {})
        run_b_metrics = run_b_results.get("metrics", {})

        for metric_id in set(list(run_a_metrics.keys()) + list(run_b_metrics.keys())):
            if metric_id not in run_a_metrics or metric_id not in run_b_metrics:
                continue

            metric_a = run_a_metrics[metric_id]
            metric_b = run_b_metrics[metric_id]

            if not metric_a.get("valid") or not metric_b.get("valid"):
                continue

            mean_a = metric_a.get("mean", 0.0)
            mean_b = metric_b.get("mean", 0.0)

            # Get baseline spec for direction
            baseline_spec = self.baseline_metrics.get(metric_id, {})
            direction = baseline_spec.get("direction", "higher")
            is_lower_better = direction == "lower"

            diff = mean_b - mean_a

            # Determine if regression or improvement
            if is_lower_better:
                is_regression = diff > 0  # Lower is better, so increase is bad
            else:
                is_regression = diff < 0  # Higher is better, so decrease is bad

            status = "regression" if is_regression else "improvement"

            report["comparisons"][metric_id] = {
                "status": status,
                "run_a_mean": mean_a,
                "run_b_mean": mean_b,
                "diff": diff,
                "percent_change": (diff / mean_a * 100) if mean_a != 0 else 0,
                "direction": direction,
            }

            if is_regression:
                report["regressions"].append(metric_id)
            else:
                report["improvements"].append(metric_id)

        return report


def format_report(comparison_report: Dict[str, Any]) -> str:
    """Format comparison report as human-readable text."""
    lines = []

    if "baseline_id" in comparison_report:
        # vs. baseline report
        lines.append("=" * 80)
        lines.append(
            f"BASELINE COMPARISON: {comparison_report['baseline_id']} vs Run {comparison_report['run_id'][:8]}"
        )
        lines.append(f"Sessions: {comparison_report['session_count']}")
        lines.append("=" * 80)

        if comparison_report["rollback_candidates"]:
            lines.append("\n🚨 ROLLBACK CANDIDATES:")
            for metric_id in comparison_report["rollback_candidates"]:
                comp = comparison_report["comparisons"][metric_id]
                lines.append(f"  - {metric_id}: {comp['message']}")

        if comparison_report["warnings"]:
            lines.append("\n⚠️  WARNINGS:")
            for metric_id in comparison_report["warnings"]:
                comp = comparison_report["comparisons"][metric_id]
                lines.append(f"  - {metric_id}: {comp['message']}")

        ok_count = len(
            [
                c
                for c in comparison_report["comparisons"].values()
                if c["status"] == "ok"
            ]
        )
        lines.append(
            f"\n✅ {ok_count} metrics within acceptable range"
        )

    else:
        # vs. run report
        lines.append("=" * 80)
        lines.append(
            f"RUN COMPARISON: {comparison_report['run_a_id'][:8]} vs {comparison_report['run_b_id'][:8]}"
        )
        lines.append("=" * 80)

        if comparison_report["regressions"]:
            lines.append("\n📉 REGRESSIONS:")
            for metric_id in comparison_report["regressions"]:
                comp = comparison_report["comparisons"][metric_id]
                lines.append(
                    f"  - {metric_id}: {comp['run_a_mean']:.4f} → {comp['run_b_mean']:.4f} "
                    f"({comp['percent_change']:+.1f}%)"
                )

        if comparison_report["improvements"]:
            lines.append("\n📈 IMPROVEMENTS:")
            for metric_id in comparison_report["improvements"]:
                comp = comparison_report["comparisons"][metric_id]
                lines.append(
                    f"  - {metric_id}: {comp['run_a_mean']:.4f} → {comp['run_b_mean']:.4f} "
                    f"({comp['percent_change']:+.1f}%)"
                )

    lines.append("=" * 80)
    return "\n".join(lines)
