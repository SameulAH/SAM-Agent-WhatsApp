"""
Phase 5.3.2: Experiment Executor

Executes baseline and variant experiments offline on fixed datasets.
Does NOT modify prompts, models, or agent code.

Responsibilities:
- Run agent on fixed, reproducible dataset
- Capture Phase 4.2 traces
- Ensure baseline and variant runs are isolated
- Control randomness (record seeds)
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import hashlib
import json

from experiments.protocol.schema import ExperimentSpec
from evaluation.schemas.trace_schema import Trace


@dataclass(frozen=True)
class ExecutorError(Exception):
    """Executor error with structured information."""

    step: str  # "baseline_run", "variant_run"
    error_message: str
    experiment_id: Optional[str] = None
    details: Optional[str] = None

    def __str__(self) -> str:
        msg = f"[{self.step}] {self.error_message}"
        if self.details:
            msg += f"\n  Details: {self.details}"
        return msg


@dataclass(frozen=True)
class ExecutionRun:
    """Result of executing a single variant (baseline or variant)."""

    run_id: str  # Unique identifier for this run
    variant_id: str  # Identifies which variant (baseline or variant)
    experiment_id: str  # Which experiment
    traces: List[Trace]  # Captured traces from execution
    started_at: str  # ISO timestamp
    completed_at: str  # ISO timestamp
    random_seed: int  # Seed used for reproducibility
    dataset_hash: str  # Hash of input dataset
    code_version: str  # Hash of executor code


class ExperimentExecutor:
    """Execute experiments offline on fixed datasets."""

    def __init__(self, dataset: Optional[List[dict]] = None):
        """
        Initialize executor.

        Args:
            dataset: Fixed dataset to use for all runs
                     If None, uses default test dataset
        """
        self.dataset = dataset or self._get_default_dataset()
        self.dataset_hash = self._compute_dataset_hash(self.dataset)

    def execute_baseline(
        self,
        spec: ExperimentSpec,
        random_seed: int = 42,
    ) -> ExecutionRun:
        """
        Execute baseline experiment on fixed dataset.

        Args:
            spec: ExperimentSpec (already validated)
            random_seed: Seed for reproducibility

        Returns:
            ExecutionRun with traces

        Raises:
            ExecutorError: If execution fails
        """
        try:
            # In Phase 5.3.2, we simulate execution
            # In Phase 5.3.3+, this will actually run the agent
            traces = self._simulate_execution(
                experiment_id=spec.experiment_id,
                variant_id=spec.baseline_id,
                random_seed=random_seed,
            )

            return ExecutionRun(
                run_id=f"{spec.experiment_id}-baseline-{random_seed}",
                variant_id=spec.baseline_id,
                experiment_id=spec.experiment_id,
                traces=traces,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat(),
                random_seed=random_seed,
                dataset_hash=self.dataset_hash,
                code_version=self._get_code_version(),
            )
        except Exception as e:
            raise ExecutorError(
                step="baseline_run",
                error_message="Failed to execute baseline",
                experiment_id=spec.experiment_id,
                details=str(e),
            )

    def execute_variant(
        self,
        spec: ExperimentSpec,
        random_seed: int = 42,
    ) -> ExecutionRun:
        """
        Execute variant experiment on fixed dataset.

        Args:
            spec: ExperimentSpec (already validated)
            random_seed: Seed for reproducibility

        Returns:
            ExecutionRun with traces

        Raises:
            ExecutorError: If execution fails
        """
        try:
            # In Phase 5.3.2, we simulate execution
            # In Phase 5.3.3+, this will actually run the variant
            traces = self._simulate_execution(
                experiment_id=spec.experiment_id,
                variant_id=spec.variant_id,
                random_seed=random_seed,
                is_variant=True,
            )

            return ExecutionRun(
                run_id=f"{spec.experiment_id}-variant-{random_seed}",
                variant_id=spec.variant_id,
                experiment_id=spec.experiment_id,
                traces=traces,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat(),
                random_seed=random_seed,
                dataset_hash=self.dataset_hash,
                code_version=self._get_code_version(),
            )
        except Exception as e:
            raise ExecutorError(
                step="variant_run",
                error_message="Failed to execute variant",
                experiment_id=spec.experiment_id,
                details=str(e),
            )

    def _simulate_execution(
        self,
        experiment_id: str,
        variant_id: str,
        random_seed: int,
        is_variant: bool = False,
    ) -> List[Trace]:
        """
        Simulate execution (placeholder for actual agent execution).

        In Phase 5.3.3, this will actually run the orchestrator.
        For now, returns empty trace list to validate pipeline.

        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            random_seed: Random seed for reproducibility
            is_variant: Whether this is variant run

        Returns:
            List of Trace objects (empty in Phase 5.3.2)
        """
        # Phase 5.3.2: Return empty traces
        # Phase 5.3.3: Will integrate with agent orchestrator
        return []

    def _get_default_dataset(self) -> List[dict]:
        """Get default test dataset."""
        return [
            {"input": "test input 1", "conversation_id": "conv-001"},
            {"input": "test input 2", "conversation_id": "conv-002"},
            {"input": "test input 3", "conversation_id": "conv-003"},
        ]

    def _compute_dataset_hash(self, dataset: List[dict]) -> str:
        """Compute hash of dataset for reproducibility."""
        dataset_json = json.dumps(dataset, sort_keys=True)
        return hashlib.sha256(dataset_json.encode()).hexdigest()[:16]

    def _get_code_version(self) -> str:
        """Get version hash of executor code."""
        # In Phase 5.3.3, this will hash the actual agent code
        # For now, return phase version
        return "5.3.2"


def execute_baseline(
    spec: ExperimentSpec,
    dataset: Optional[List[dict]] = None,
    random_seed: int = 42,
) -> ExecutionRun:
    """
    Convenience function to execute baseline.

    Args:
        spec: ExperimentSpec
        dataset: Optional fixed dataset
        random_seed: Optional seed

    Returns:
        ExecutionRun

    Raises:
        ExecutorError: If execution fails
    """
    executor = ExperimentExecutor(dataset=dataset)
    return executor.execute_baseline(spec, random_seed=random_seed)


def execute_variant(
    spec: ExperimentSpec,
    dataset: Optional[List[dict]] = None,
    random_seed: int = 42,
) -> ExecutionRun:
    """
    Convenience function to execute variant.

    Args:
        spec: ExperimentSpec
        dataset: Optional fixed dataset
        random_seed: Optional seed

    Returns:
        ExecutionRun

    Raises:
        ExecutorError: If execution fails
    """
    executor = ExperimentExecutor(dataset=dataset)
    return executor.execute_variant(spec, random_seed=random_seed)
