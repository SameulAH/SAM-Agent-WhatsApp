"""
Phase 5.3.2: Experiment Runner

Main orchestration entrypoint for the experiment harness.
Glues all components together and enforces the full pipeline.

Responsibilities:
- Provide CLI interface
- Load and validate specs
- Execute baseline and variant
- Compute metrics
- Compare results
- Generate reports
- Record reproducibility
- Enforce pipeline order
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from experiment_harness.loader import SpecLoader, LoaderError
from experiment_harness.executor import ExperimentExecutor, ExecutorError
from experiment_harness.evaluator import ExperimentEvaluator, EvaluatorError
from experiment_harness.reporter import ComparisonReporter, ReporterError
from experiment_harness.reproducibility import ReproducibilityRecorder


class ExperimentRunner:
    """Main orchestration entrypoint for experiment harness."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize runner.

        Args:
            output_dir: Directory for outputs
        """
        self.output_dir = output_dir or (Path.cwd() / "outputs" / "experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.spec_loader = SpecLoader()
        self.executor = ExperimentExecutor()
        self.evaluator = ExperimentEvaluator()
        self.reporter = ComparisonReporter()
        self.reproducibility_recorder = ReproducibilityRecorder(self.output_dir)

    def run_experiment(
        self,
        spec_path: str,
        output_experiment_id: Optional[str] = None,
    ) -> dict:
        """
        Run a complete experiment pipeline.

        Pipeline:
        1. Load spec
        2. Validate spec
        3. Execute baseline
        4. Execute variant
        5. Evaluate metrics
        6. Compare results
        7. Generate report
        8. Record reproducibility

        Args:
            spec_path: Path to experiment specification
            output_experiment_id: Custom experiment ID for output

        Returns:
            Results dictionary with all outputs

        Raises:
            Exception: If any step fails (never swallowed)
        """
        print(f"[Phase 5.3.2] Experiment Runner")
        print(f"=" * 60)
        print()

        # Step 1: Load spec
        print(f"[1/8] Loading specification...")
        try:
            spec = self.spec_loader.load_spec_from_file(spec_path)
            print(f"      ✓ Specification loaded: {spec.experiment_id}")
        except LoaderError as e:
            print(f"      ✗ Load failed: {e}")
            raise

        # Step 2: Validation already done in loader
        print(f"[2/8] Validating specification...")
        print(f"      ✓ Specification valid (Phase 5.3.1 validator passed)")

        # Step 3: Execute baseline
        print(f"[3/8] Executing baseline...")
        try:
            baseline_run = self.executor.execute_baseline(spec)
            print(f"      ✓ Baseline executed: {baseline_run.run_id}")
            print(f"        Dataset hash: {baseline_run.dataset_hash}")
            print(f"        Random seed: {baseline_run.random_seed}")
        except ExecutorError as e:
            print(f"      ✗ Baseline execution failed: {e}")
            raise

        # Step 4: Execute variant
        print(f"[4/8] Executing variant...")
        try:
            variant_run = self.executor.execute_variant(spec)
            print(f"      ✓ Variant executed: {variant_run.run_id}")
            print(f"        Same dataset: {baseline_run.dataset_hash == variant_run.dataset_hash}")
            print(f"        Same seed: {baseline_run.random_seed == variant_run.random_seed}")
        except ExecutorError as e:
            print(f"      ✗ Variant execution failed: {e}")
            raise

        # Step 5: Evaluate metrics
        print(f"[5/8] Evaluating metrics (Phase 5.2)...")
        try:
            baseline_metrics = self.evaluator.evaluate_run(spec, baseline_run)
            variant_metrics = self.evaluator.evaluate_run(spec, variant_run)
            print(f"      ✓ Metrics evaluated")
            print(f"        Baseline: {len(baseline_metrics.get_valid_metrics())} valid metrics")
            print(f"        Variant: {len(variant_metrics.get_valid_metrics())} valid metrics")
        except EvaluatorError as e:
            print(f"      ✗ Evaluation failed: {e}")
            raise

        # Step 6: Compare results
        print(f"[6/8] Comparing results...")
        try:
            comparison_report = self.reporter.compare_runs(spec, baseline_metrics, variant_metrics)
            print(f"      ✓ Comparison complete")
            print(f"        Improvements: {len(comparison_report.improvements)}")
            print(f"        Regressions: {len(comparison_report.regressions)}")
            print(f"        Inconclusive: {len(comparison_report.inconclusive)}")
            print(f"        Recommendation: {comparison_report.recommendation}")
        except ReporterError as e:
            print(f"      ✗ Comparison failed: {e}")
            raise

        # Step 7: Generate report
        print(f"[7/8] Generating report...")
        try:
            report_markdown = self.reporter.format_report_markdown(comparison_report)
            experiment_id = output_experiment_id or spec.experiment_id
            self._save_report(experiment_id, report_markdown, comparison_report)
            print(f"      ✓ Report generated")
            print(f"        Directory: {self.output_dir / experiment_id}")
        except Exception as e:
            print(f"      ✗ Report generation failed: {e}")
            raise

        # Step 8: Record reproducibility
        print(f"[8/8] Recording reproducibility...")
        try:
            baseline_record = self.reproducibility_recorder.record_run(
                spec, baseline_run, spec_path
            )
            variant_record = self.reproducibility_recorder.record_run(
                spec, variant_run, spec_path
            )
            print(f"      ✓ Reproducibility recorded")
            print(f"        Baseline spec hash: {baseline_record.spec_hash}")
            print(f"        Variant spec hash: {variant_record.spec_hash}")
        except Exception as e:
            print(f"      ✗ Reproducibility recording failed: {e}")
            raise

        print()
        print(f"=" * 60)
        print(f"[Phase 5.3.2] Experiment complete")
        print()
        print(f"Recommendation: {comparison_report.recommendation}")
        print(f"(Note: This is evidence-based recommendation only.)")
        print(f"Final decision requires human review (Phase 5.3.3).")
        print()

        # Return results
        return {
            "experiment_id": spec.experiment_id,
            "spec": spec,
            "baseline_run": baseline_run,
            "variant_run": variant_run,
            "baseline_metrics": baseline_metrics,
            "variant_metrics": variant_metrics,
            "comparison_report": comparison_report,
            "baseline_reproducibility": baseline_record,
            "variant_reproducibility": variant_record,
        }

    def _save_report(self, experiment_id: str, markdown: str, comparison_report: dict) -> None:
        """
        Save report to files.

        Args:
            experiment_id: Experiment identifier
            markdown: Markdown report
            comparison_report: Comparison report object
        """
        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown report
        report_file = experiment_dir / "report.md"
        report_file.write_text(markdown)

        # Save JSON comparison
        comparison_file = experiment_dir / "comparison.json"
        comparison_json = {
            "experiment_id": comparison_report.experiment_id,
            "baseline_run_id": comparison_report.baseline_run_id,
            "variant_run_id": comparison_report.variant_run_id,
            "created_at": comparison_report.created_at,
            "metrics_compared": comparison_report.metrics_compared,
            "improvements": comparison_report.improvements,
            "regressions": comparison_report.regressions,
            "inconclusive": comparison_report.inconclusive,
            "recommendation": comparison_report.recommendation,
        }
        comparison_file.write_text(json.dumps(comparison_json, indent=2))


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Phase 5.3.2 Experiment Runner",
        epilog="Example: python runner.py --experiment specs/exp-001.yaml",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment specification (YAML or JSON)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports (default: ./outputs/experiments)",
    )

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    # Run experiment
    try:
        results = runner.run_experiment(args.experiment)
        sys.exit(0)
    except Exception as e:
        print()
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
