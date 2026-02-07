"""
Phase 5.3.2: Experiment Evaluator

Computes metrics from execution traces using Phase 5.2 metric extractors.
Pure orchestration layer over Phase 5.2 (no new metric logic).

Responsibilities:
- Consume traces from executor
- Invoke Phase 5.2 metric extractors
- Produce per-session and per-run metrics
- Return structured results
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path

from experiments.protocol.schema import ExperimentSpec
from experiment_harness.executor import ExecutionRun
from evaluation.schemas.trace_schema import Trace
from evaluation.schemas.metric_schema import (
    MetricResult,
    get_metric_by_id,
    get_metrics_for_dimension,
)


@dataclass(frozen=True)
class EvaluatorError(Exception):
    """Evaluator error with structured information."""

    step: str  # "metric_extraction", "aggregation"
    error_message: str
    experiment_id: Optional[str] = None
    metric_id: Optional[str] = None
    details: Optional[str] = None

    def __str__(self) -> str:
        msg = f"[{self.step}] {self.error_message}"
        if self.metric_id:
            msg += f" [metric: {self.metric_id}]"
        if self.details:
            msg += f"\n  Details: {self.details}"
        return msg


@dataclass(frozen=True)
class MetricsResult:
    """Result of metric computation for a run."""

    run_id: str
    variant_id: str
    experiment_id: str
    metrics: Dict[str, MetricResult]  # metric_id -> MetricResult
    timestamp: str

    def get_metric(self, metric_id: str) -> Optional[MetricResult]:
        """Get a specific metric result."""
        return self.metrics.get(metric_id)

    def get_all_metrics(self) -> List[MetricResult]:
        """Get all metric results."""
        return list(self.metrics.values())

    def get_valid_metrics(self) -> List[MetricResult]:
        """Get only valid metrics."""
        return [m for m in self.metrics.values() if m.valid]

    def get_invalid_metrics(self) -> List[MetricResult]:
        """Get only invalid metrics."""
        return [m for m in self.metrics.values() if not m.valid]


class ExperimentEvaluator:
    """Evaluate execution runs using Phase 5.2 metrics."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def evaluate_run(
        self,
        spec: ExperimentSpec,
        execution_run: ExecutionRun,
    ) -> MetricsResult:
        """
        Evaluate a single execution run.

        Computes all metrics specified in spec.metrics_used using Phase 5.2 extractors.

        Args:
            spec: ExperimentSpec (validated)
            execution_run: ExecutionRun with traces

        Returns:
            MetricsResult with computed metrics

        Raises:
            EvaluatorError: If evaluation fails
        """
        try:
            metrics_dict: Dict[str, MetricResult] = {}

            # For each metric in spec.metrics_used
            for metric_id in spec.metrics_used:
                try:
                    # Verify metric exists in Phase 5.2
                    metric_def = get_metric_by_id(metric_id)
                    if not metric_def:
                        raise EvaluatorError(
                            step="metric_extraction",
                            error_message=f"Metric not found in Phase 5.2 registry: {metric_id}",
                            experiment_id=spec.experiment_id,
                            metric_id=metric_id,
                        )

                    # In Phase 5.3.2, we simulate metric extraction
                    # In Phase 5.3.3+, this will invoke actual metric extractors
                    result = self._extract_metric(
                        metric_id=metric_id,
                        traces=execution_run.traces,
                        spec=spec,
                    )

                    metrics_dict[metric_id] = result

                except EvaluatorError:
                    raise
                except Exception as e:
                    raise EvaluatorError(
                        step="metric_extraction",
                        error_message=f"Failed to extract metric: {metric_id}",
                        experiment_id=spec.experiment_id,
                        metric_id=metric_id,
                        details=str(e),
                    )

            return MetricsResult(
                run_id=execution_run.run_id,
                variant_id=execution_run.variant_id,
                experiment_id=spec.experiment_id,
                metrics=metrics_dict,
                timestamp=execution_run.completed_at,
            )

        except EvaluatorError:
            raise
        except Exception as e:
            raise EvaluatorError(
                step="aggregation",
                error_message="Failed to evaluate run",
                experiment_id=spec.experiment_id,
                details=str(e),
            )

    def evaluate_comparison(
        self,
        spec: ExperimentSpec,
        baseline_result: MetricsResult,
        variant_result: MetricsResult,
    ) -> Dict[str, Any]:
        """
        Evaluate comparison between baseline and variant.

        Args:
            spec: ExperimentSpec
            baseline_result: MetricsResult for baseline
            variant_result: MetricsResult for variant

        Returns:
            Dictionary with comparison analysis
        """
        try:
            comparison = {
                "experiment_id": spec.experiment_id,
                "baseline_run": baseline_result.run_id,
                "variant_run": variant_result.run_id,
                "metrics_compared": [],
                "improvements": [],
                "regressions": [],
                "inconclusive": [],
            }

            # Compare each metric
            for metric_id in spec.metrics_used:
                baseline_metric = baseline_result.get_metric(metric_id)
                variant_metric = variant_result.get_metric(metric_id)

                if not baseline_metric or not variant_metric:
                    comparison["inconclusive"].append({
                        "metric_id": metric_id,
                        "reason": "Missing baseline or variant metric",
                    })
                    continue

                # Get metric definition to determine direction
                metric_def = get_metric_by_id(metric_id)
                if not metric_def:
                    continue

                comparison["metrics_compared"].append(metric_id)

                # Compare values based on direction
                if baseline_metric.valid and variant_metric.valid:
                    if metric_def.direction == "higher":
                        if variant_metric.value > baseline_metric.value:
                            comparison["improvements"].append({
                                "metric_id": metric_id,
                                "baseline": baseline_metric.value,
                                "variant": variant_metric.value,
                                "delta": variant_metric.value - baseline_metric.value,
                            })
                        elif variant_metric.value < baseline_metric.value:
                            comparison["regressions"].append({
                                "metric_id": metric_id,
                                "baseline": baseline_metric.value,
                                "variant": variant_metric.value,
                                "delta": variant_metric.value - baseline_metric.value,
                            })
                    else:  # lower is better
                        if variant_metric.value < baseline_metric.value:
                            comparison["improvements"].append({
                                "metric_id": metric_id,
                                "baseline": baseline_metric.value,
                                "variant": variant_metric.value,
                                "delta": variant_metric.value - baseline_metric.value,
                            })
                        elif variant_metric.value > baseline_metric.value:
                            comparison["regressions"].append({
                                "metric_id": metric_id,
                                "baseline": baseline_metric.value,
                                "variant": variant_metric.value,
                                "delta": variant_metric.value - baseline_metric.value,
                            })
                else:
                    comparison["inconclusive"].append({
                        "metric_id": metric_id,
                        "reason": "Invalid baseline or variant metric",
                    })

            return comparison

        except Exception as e:
            raise EvaluatorError(
                step="comparison",
                error_message="Failed to compare runs",
                experiment_id=spec.experiment_id,
                details=str(e),
            )

    def _extract_metric(
        self,
        metric_id: str,
        traces: List[Trace],
        spec: ExperimentSpec,
    ) -> MetricResult:
        """
        Extract a metric from traces.

        In Phase 5.3.2, this simulates metric extraction.
        In Phase 5.3.3+, this will invoke actual Phase 5.2 extractors.

        Args:
            metric_id: Metric identifier
            traces: Traces to compute metric from
            spec: ExperimentSpec

        Returns:
            MetricResult
        """
        # Phase 5.3.2: Return simulated result
        # Phase 5.3.3: Will invoke evaluation/metrics/{dimension}/{metric}.py extractors
        return MetricResult(
            metric_id=metric_id,
            value=0.0,
            samples=0,
            valid=False,
            error="Phase 5.3.2: Metrics not yet computed (placeholder)",
        )
