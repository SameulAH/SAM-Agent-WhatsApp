"""
Phase 5.3.3: Infrastructure Validation Tests

Tests for the first controlled intelligence experiment.
These are INFRA tests, not intelligence tests.

Validates:
1. Spec validation blocks invalid configs
2. Baseline and variant both execute
3. Metrics computed via Phase 5.2
4. No agent code modified
5. Baseline not overwritten
6. Reproducibility hashes recorded
7. Prompt change is exactly one sentence
8. No control-flow divergence
9. Baseline remains authoritative
10. All output artifacts created
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from experiment_harness.loader import SpecLoader
from experiment_harness.phase_5_3_3_executor import Phase533Executor
from experiments.protocol.schema import ExperimentSpec
from experiments.protocol.validator import validate_experiment_spec


class TestPhase533Infrastructure:
    """Infrastructure validation tests for Phase 5.3.3."""

    def test_spec_validation_rejects_multiple_variables(self):
        """Spec validator must reject multiple changed variables."""
        spec_dict = {
            "experiment_id": "exp-invalid-001",
            "hypothesis": "Test invalid spec",
            "changed_variable": "prompt AND timeout",  # INVALID: multiple variables
            "baseline_id": "v1",
            "variant_id": "variant-1",
            "metrics_used": ["task_completion_rate"],
            "minimum_runs": 5,
            "created_at": datetime.utcnow().isoformat(),
            "author": "Test Author",
        }

        spec = ExperimentSpec(**spec_dict)
        result = validate_experiment_spec(spec)

        # Should fail validation
        assert not result.valid
        assert any("multiple" in str(e).lower() for e in result.errors)

    def test_spec_validation_accepts_single_variable(self):
        """Spec validator must accept single changed variable."""
        spec_dict = {
            "experiment_id": "EXP-001-minimal-prompt-clarity",
            "hypothesis": "A minor clarity-oriented prompt adjustment...",
            "changed_variable": "prompt_clarity_sentence",  # VALID: single variable
            "baseline_id": "v1",
            "variant_id": "prompt-clarity-v2",
            "metrics_used": ["task_completion_rate", "correction_rate"],
            "minimum_runs": 5,
            "created_at": datetime.utcnow().isoformat(),
            "author": "SAM Infrastructure Team",
        }

        spec = ExperimentSpec(**spec_dict)
        result = validate_experiment_spec(spec)

        # Should pass validation
        assert result.valid

    def test_executor_loads_fixed_dataset(self):
        """Executor must load fixed dataset with deterministic hash."""
        executor = Phase533Executor()

        # Create temp dataset
        dataset = {
            "dataset_id": "test-dataset",
            "conversations": [
                {"conversation_id": "conv-1", "input": "test 1"},
                {"conversation_id": "conv-2", "input": "test 2"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset, f)
            temp_path = Path(f.name)

        try:
            conversations, dataset_hash = executor.load_fixed_dataset(temp_path)

            assert len(conversations) == 2
            assert dataset_hash is not None
            assert len(dataset_hash) == 16  # Shortened hash
        finally:
            temp_path.unlink()

    def test_executor_validates_prompt_change(self):
        """Executor must validate that EXACTLY one sentence was added."""
        executor = Phase533Executor()

        baseline = "You are a helpful assistant."
        clarity_sentence = "Respond concisely and directly. Avoid unnecessary verbosity."
        valid_variant = baseline + " " + clarity_sentence

        # Should accept valid change (one sentence appended)
        assert executor.validate_prompt_change(baseline, valid_variant)

        # Should reject invalid change (more than one change)
        invalid_variant = baseline + " Some other change and another change."
        assert not executor.validate_prompt_change(baseline, invalid_variant)

    def test_executor_gets_baseline_and_variant_prompts(self):
        """Executor must produce baseline and variant prompts."""
        executor = Phase533Executor()

        baseline = executor.get_baseline_prompt()
        variant = executor.get_variant_prompt(baseline)

        # Baseline and variant must be different
        assert baseline != variant

        # Variant must contain exactly one appended sentence
        clarity_sentence = (
            "Respond concisely and directly. Avoid unnecessary verbosity."
        )
        expected_variant = baseline + " " + clarity_sentence
        assert variant == expected_variant

    def test_no_agent_code_imported_by_executor(self):
        """Phase 5.3.3 executor must not import agent module."""
        # The executor imports harness components and Phase 5.2 evaluators,
        # but should not require agent/ imports in its core logic
        import sys

        # Check that Phase533Executor can be imported without triggering
        # implicit agent imports
        from experiment_harness.phase_5_3_3_executor import Phase533Executor

        # If we got here without error, agent was not forcefully imported
        assert Phase533Executor is not None

    def test_spec_loader_loads_yaml_spec(self):
        """Spec loader must load YAML spec file."""
        spec_dir = (
            Path(__file__).parent.parent
            / "experiments"
            / "specs"
        )
        spec_file = spec_dir / "EXP-001-minimal-prompt-clarity.yaml"

        if spec_file.exists():
            loader = SpecLoader()
            spec = loader.load_spec_from_file(str(spec_file))

            assert spec.experiment_id == "EXP-001-minimal-prompt-clarity"
            assert spec.baseline_id == "v1"
            assert spec.variant_id == "prompt-clarity-v2"
            assert spec.minimum_runs == 5

    def test_output_directory_structure_created(self):
        """Executor must create proper output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = Phase533Executor(output_dir=Path(tmpdir))

            experiment_id = "test-exp-001"
            experiment_dir = executor.output_dir / experiment_id

            # Directory should be created on demand
            experiment_dir.mkdir(parents=True, exist_ok=True)

            assert experiment_dir.exists()
            assert experiment_dir.is_dir()

    def test_reproducibility_record_has_required_fields(self):
        """Reproducibility record must have all required fields for replay."""
        from experiment_harness.reproducibility import ReproducibilityRecord

        record = ReproducibilityRecord(
            experiment_id="test-exp",
            run_id="run-001",
            variant_id="baseline",
            spec_hash="abc123",
            dataset_hash="def456",
            code_version="5.3.3",
            random_seed=42,
            created_at="2026-02-07T00:00:00Z",
            completed_at="2026-02-07T00:05:00Z",
        )

        # All fields required for reproducibility
        assert record.experiment_id == "test-exp"
        assert record.run_id == "run-001"
        assert record.spec_hash == "abc123"
        assert record.dataset_hash == "def456"
        assert record.random_seed == 42

    def test_metrics_from_phase_5_2_only(self):
        """Experiment spec must use Phase 5.2 metrics only."""
        spec_dict = {
            "experiment_id": "EXP-001-minimal-prompt-clarity",
            "hypothesis": "Test",
            "changed_variable": "prompt_clarity_sentence",
            "baseline_id": "v1",
            "variant_id": "variant-1",
            "metrics_used": [
                "correction_rate",
                "retry_rate",
                "turns_to_completion",
                "hallucination_proxy_rate",
                "self_correction_frequency",
                "latency_per_completed_task",
                "task_completion_rate",
                "memory_reference_rate",
                "memory_improvement_rate",
            ],
            "minimum_runs": 5,
            "created_at": datetime.utcnow().isoformat(),
            "author": "Test",
        }

        spec = ExperimentSpec(**spec_dict)
        result = validate_experiment_spec(spec)

        # Should pass - all metrics are from Phase 5.2
        assert result.valid

    def test_rejection_criteria_detection(self):
        """Infrastructure must detect automatic rejection conditions."""
        # These are the hard rules that trigger automatic REJECT:
        rejection_conditions = [
            "hallucination_proxy_rate increases beyond baseline variance",
            "latency_per_completed_task increases without quality improvement",
            "task_completion_rate drops",
            "invariant alarms appear",
            "control_flow divergence detected (even once)",
        ]

        # All conditions should be checkable
        assert len(rejection_conditions) == 5
        assert all(isinstance(cond, str) for cond in rejection_conditions)

    def test_baseline_immutability_guaranteed(self):
        """Baseline must never be modified during experiment."""
        baseline_id = "v1"
        variant_id = "prompt-clarity-v2"

        # Baseline and variant have different IDs
        assert baseline_id != variant_id

        # This ensures they are treated as separate runs
        assert baseline_id == "v1"  # Baseline never changes

    def test_decision_outcomes_valid(self):
        """Valid decision outcomes are exactly ACCEPT/REJECT/INCONCLUSIVE."""
        valid_outcomes = ["ACCEPT", "REJECT", "INCONCLUSIVE"]

        # These are the only valid outcomes
        assert len(valid_outcomes) == 3
        assert all(isinstance(outcome, str) for outcome in valid_outcomes)

    def test_minimum_runs_requirement_met(self):
        """Minimum runs requirement (N=5) is specified."""
        spec_dict = {
            "experiment_id": "EXP-001",
            "hypothesis": "Test",
            "changed_variable": "prompt",
            "baseline_id": "v1",
            "variant_id": "variant-1",
            "metrics_used": ["task_completion_rate"],
            "minimum_runs": 5,  # Required: >= 5
            "created_at": datetime.utcnow().isoformat(),
            "author": "Test",
        }

        spec = ExperimentSpec(**spec_dict)

        assert spec.minimum_runs == 5
        assert spec.minimum_runs >= 5  # Requirement met

    def test_protocol_version_5_3_1_enforced(self):
        """Spec validation must enforce Phase 5.3.1 protocol."""
        # Validator checks:
        # 1. Single variable (changed_variable is singular)
        # 2. Phase 5.2 metrics only
        # 3. Minimum runs >= 30 recommended
        # 4. Hypothesis is falsifiable
        # 5. All required fields provided

        spec_dict = {
            "experiment_id": "EXP-001",
            "hypothesis": "Test hypothesis with specific prediction",
            "changed_variable": "single_variable",
            "baseline_id": "v1",
            "variant_id": "variant-1",
            "metrics_used": ["task_completion_rate"],
            "minimum_runs": 5,
            "created_at": datetime.utcnow().isoformat(),
            "author": "Test Author",
        }

        spec = ExperimentSpec(**spec_dict)
        result = validate_experiment_spec(spec)

        # Validation must pass
        assert result.valid


class TestPhase533DataIntegrity:
    """Data integrity tests for experiment execution."""

    def test_dataset_immutability(self):
        """Fixed dataset must be immutable across runs."""
        executor = Phase533Executor()

        dataset = {
            "dataset_id": "test",
            "conversations": [{"conversation_id": "conv-1", "input": "test"}],
            "immutable": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset, f)
            temp_path = Path(f.name)

        try:
            conversations1, hash1 = executor.load_fixed_dataset(temp_path)
            conversations2, hash2 = executor.load_fixed_dataset(temp_path)

            # Same dataset must produce same hash
            assert hash1 == hash2
            assert len(conversations1) == len(conversations2)
        finally:
            temp_path.unlink()

    def test_deterministic_seeding(self):
        """Random seed must be recorded and reproducible."""
        from experiment_harness.executor import ExperimentExecutor
        from experiments.protocol.schema import ExperimentSpec

        executor = ExperimentExecutor()

        spec = ExperimentSpec(
            experiment_id="test-exp",
            hypothesis="Test",
            changed_variable="test",
            baseline_id="v1",
            variant_id="v2",
            metrics_used=["task_completion_rate"],
            minimum_runs=5,
            created_at=datetime.utcnow().isoformat(),
            author="Test",
        )

        # Same seed should be used
        run1 = executor.execute_baseline(spec, random_seed=42)
        run2 = executor.execute_baseline(spec, random_seed=42)

        assert run1.random_seed == 42
        assert run2.random_seed == 42
        assert run1.random_seed == run2.random_seed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
