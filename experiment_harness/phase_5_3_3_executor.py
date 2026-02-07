"""
Phase 5.3.3: First Controlled Intelligence Experiment Executor

Extends Phase 5.3.2 harness with actual agent integration.

This executor:
- Loads the fixed dataset
- Runs baseline with current prompt
- Runs variant with modified prompt (single sentence added)
- Captures Phase 4.2 traces
- Computes Phase 5.2 metrics
- Records reproducibility metadata

Invariant: Exactly one variable changes (the prompt).
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from experiment_harness.executor import ExperimentExecutor, ExecutionRun
from experiment_harness.loader import SpecLoader
from experiments.protocol.schema import ExperimentSpec


class Phase533Executor:
    """Phase 5.3.3 executor with actual agent integration."""

    def __init__(self, output_dir: Path = None):
        """Initialize Phase 5.3.3 executor."""
        self.output_dir = output_dir or (Path.cwd() / "outputs" / "experiments")
        self.harness_executor = ExperimentExecutor()
        self.spec_loader = SpecLoader()

    def load_fixed_dataset(self, dataset_path: Path) -> tuple[List[Dict], str]:
        """
        Load the fixed, frozen dataset.

        Args:
            dataset_path: Path to dataset JSON

        Returns:
            (conversations, dataset_hash)
        """
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        conversations = dataset["conversations"]

        # Compute deterministic hash
        dataset_json = json.dumps(dataset, sort_keys=True)
        dataset_hash = hashlib.sha256(dataset_json.encode()).hexdigest()[:16]

        print(f"[5.3.3] Loaded fixed dataset: {len(conversations)} conversations")
        print(f"        Dataset hash: {dataset_hash}")

        return conversations, dataset_hash

    def get_baseline_prompt(self) -> str:
        """
        Get the baseline prompt (unchanged).

        In production, this would load from the actual agent config.
        For Phase 5.3.3, we use a minimal default.
        """
        baseline_prompt = (
            "You are a helpful AI assistant. "
            "Answer questions accurately and completely."
        )
        return baseline_prompt

    def get_variant_prompt(self, baseline_prompt: str) -> str:
        """
        Get the variant prompt (baseline + ONE sentence).

        EXACTLY ONE CHANGE: append sentence at the end.

        Args:
            baseline_prompt: The baseline prompt

        Returns:
            Variant prompt = baseline_prompt + clarity sentence
        """
        clarity_sentence = "Respond concisely and directly. Avoid unnecessary verbosity."
        variant_prompt = baseline_prompt + " " + clarity_sentence
        return variant_prompt

    def validate_prompt_change(
        self, baseline_prompt: str, variant_prompt: str
    ) -> bool:
        """
        Validate that ONLY one sentence was added.

        Hard rule: exactly one variable changes.

        Args:
            baseline_prompt: Original prompt
            variant_prompt: Modified prompt

        Returns:
            True if change is valid (only append), False otherwise
        """
        clarity_sentence = "Respond concisely and directly. Avoid unnecessary verbosity."

        # Variant must be baseline + clarity sentence (with spacing)
        expected_variant = baseline_prompt + " " + clarity_sentence

        if variant_prompt != expected_variant:
            print(
                "[5.3.3] ❌ PROMPT VALIDATION FAILED: More than one change detected"
            )
            print(f"        Baseline: {baseline_prompt}")
            print(f"        Expected variant: {expected_variant}")
            print(f"        Actual variant: {variant_prompt}")
            return False

        print("[5.3.3] ✅ Prompt validation PASSED: Exactly one sentence appended")
        return True

    def execute_experiment(self, spec_path: Path) -> Dict[str, Any]:
        """
        Execute the complete experiment pipeline.

        Pipeline:
        1. Load spec
        2. Validate spec (Phase 5.3.1)
        3. Load dataset
        4. Execute baseline run (N=5)
        5. Execute variant run (N=5)
        6. Capture Phase 4.2 traces
        7. Compute Phase 5.2 metrics
        8. Compare baseline vs variant
        9. Generate report
        10. Record reproducibility metadata

        Args:
            spec_path: Path to experiment spec YAML

        Returns:
            Results dictionary with all outputs
        """
        print("[Phase 5.3.3] FIRST CONTROLLED INTELLIGENCE EXPERIMENT")
        print("=" * 70)
        print()

        # Step 1: Load spec
        print("[1/10] Loading experiment specification...")
        spec = self.spec_loader.load_spec_from_file(str(spec_path))
        print(f"      ✓ Spec loaded: {spec.experiment_id}")
        print(f"      Hypothesis: {spec.hypothesis[:60]}...")
        print()

        # Step 2: Validate spec (Phase 5.3.1)
        print("[2/10] Validating specification (Phase 5.3.1)...")
        from experiments.protocol.validator import validate_experiment_spec

        validation_result = validate_experiment_spec(spec)
        if not validation_result.valid:
            print(f"      ✗ Validation FAILED")
            for error in validation_result.errors:
                print(f"        [{error.field}] {error.error}")
            raise ValueError("Spec validation failed - experiment invalid")

        print(f"      ✓ Specification is VALID")
        print()

        # Step 3: Load fixed dataset
        print("[3/10] Loading fixed, frozen dataset...")
        dataset_dir = spec_path.parent
        dataset_path = dataset_dir / "EXP-001-fixed-dataset.json"
        conversations, dataset_hash = self.load_fixed_dataset(dataset_path)
        print()

        # Step 4: Prepare prompts
        print("[4/10] Preparing prompts...")
        baseline_prompt = self.get_baseline_prompt()
        variant_prompt = self.get_variant_prompt(baseline_prompt)

        # Validate prompt change
        if not self.validate_prompt_change(baseline_prompt, variant_prompt):
            raise ValueError("Prompt change validation failed - more than one change")

        print(f"        Baseline length: {len(baseline_prompt)}")
        print(f"        Variant length: {len(variant_prompt)}")
        print(f"        Difference: {len(variant_prompt) - len(baseline_prompt)} chars")
        print()

        # Step 5: Execute baseline run (N=5)
        print("[5/10] Executing baseline run (N=5 traces)...")
        baseline_run = self.harness_executor.execute_baseline(spec)
        print(f"      ✓ Baseline executed: {baseline_run.run_id}")
        print(f"        Traces collected: {len(baseline_run.traces)}")
        print(f"        Dataset hash: {baseline_run.dataset_hash}")
        print(f"        Random seed: {baseline_run.random_seed}")
        print()

        # Step 6: Execute variant run (N=5)
        print("[6/10] Executing variant run (N=5 traces)...")
        variant_run = self.harness_executor.execute_variant(spec)
        print(f"      ✓ Variant executed: {variant_run.run_id}")
        print(f"        Traces collected: {len(variant_run.traces)}")
        print(f"        Same dataset: {baseline_run.dataset_hash == variant_run.dataset_hash}")
        print(f"        Same seed: {baseline_run.random_seed == variant_run.random_seed}")
        print()

        # Step 7: Capture Phase 4.2 traces
        print("[7/10] Verifying Phase 4.2 trace capture...")
        baseline_trace_count = len(baseline_run.traces)
        variant_trace_count = len(variant_run.traces)
        print(f"      ✓ Baseline traces: {baseline_trace_count}")
        print(f"      ✓ Variant traces: {variant_trace_count}")
        print()

        # Step 8: Compute Phase 5.2 metrics
        print("[8/10] Computing Phase 5.2 metrics...")
        from experiment_harness.evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator()
        baseline_metrics = evaluator.evaluate_run(spec, baseline_run)
        variant_metrics = evaluator.evaluate_run(spec, variant_run)

        print(f"      ✓ Baseline metrics computed")
        print(f"        Valid metrics: {len(baseline_metrics.get_valid_metrics())}")
        print(f"      ✓ Variant metrics computed")
        print(f"        Valid metrics: {len(variant_metrics.get_valid_metrics())}")
        print()

        # Step 9: Compare baseline vs variant
        print("[9/10] Comparing results and detecting regressions...")
        from experiment_harness.reporter import ComparisonReporter

        reporter = ComparisonReporter()
        comparison_report = reporter.compare_runs(spec, baseline_metrics, variant_metrics)

        print(f"      ✓ Comparison complete")
        print(f"        Improvements: {len(comparison_report.improvements)}")
        print(f"        Regressions: {len(comparison_report.regressions)}")
        print(f"        Inconclusive: {len(comparison_report.inconclusive)}")
        print(f"        Recommendation: {comparison_report.recommendation}")
        print()

        # Step 10: Record reproducibility metadata
        print("[10/10] Recording reproducibility metadata...")
        from experiment_harness.reproducibility import ReproducibilityRecorder

        recorder = ReproducibilityRecorder(self.output_dir)
        baseline_record = recorder.record_run(spec, baseline_run, str(spec_path))
        variant_record = recorder.record_run(spec, variant_run, str(spec_path))

        print(f"      ✓ Reproducibility recorded")
        print(f"        Baseline spec hash: {baseline_record.spec_hash}")
        print(f"        Variant spec hash: {variant_record.spec_hash}")
        print()

        # Generate report
        print("=" * 70)
        print("[Phase 5.3.3] EXPERIMENT RESULTS")
        print("=" * 70)
        print()

        report_markdown = reporter.format_report_markdown(comparison_report)
        self._save_outputs(
            spec.experiment_id,
            report_markdown,
            comparison_report,
            baseline_metrics,
            variant_metrics,
            baseline_record,
            variant_record,
        )

        print(f"Recommendation: {comparison_report.recommendation}")
        print(f"(Note: This is evidence-based. Humans make final decision.)")
        print()

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
            "report_markdown": report_markdown,
        }

    def _save_outputs(
        self,
        experiment_id: str,
        report_markdown: str,
        comparison_report: Dict,
        baseline_metrics,
        variant_metrics,
        baseline_record,
        variant_record,
    ) -> None:
        """Save all output artifacts."""
        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown report
        report_file = experiment_dir / "report.md"
        report_file.write_text(report_markdown)
        print(f"✓ Report saved: {report_file}")

        # Save comparison JSON
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
            "evidence": comparison_report.evidence,
        }
        comparison_file.write_text(json.dumps(comparison_json, indent=2))
        print(f"✓ Comparison saved: {comparison_file}")

        # Save reproducibility records
        baseline_repro_file = (
            experiment_dir / f"{baseline_record.run_id}-reproducibility.json"
        )
        baseline_repro_file.write_text(baseline_record.to_json())
        print(f"✓ Baseline reproducibility saved: {baseline_repro_file}")

        variant_repro_file = (
            experiment_dir / f"{variant_record.run_id}-reproducibility.json"
        )
        variant_repro_file.write_text(variant_record.to_json())
        print(f"✓ Variant reproducibility saved: {variant_repro_file}")

        print()
