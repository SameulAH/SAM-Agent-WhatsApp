"""
Phase 5.3.2: Reproducibility Recorder

Records full reproducibility information for experiment runs.
Ensures every run can be replayed exactly.

Responsibilities:
- Record experiment spec hash
- Record dataset hash
- Record code version
- Record random seeds
- Ensure run can be replayed
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from experiments.protocol.schema import ExperimentSpec
from experiment_harness.executor import ExecutionRun


@dataclass(frozen=True)
class ReproducibilityRecord:
    """Complete reproducibility information for a run."""

    experiment_id: str
    run_id: str
    variant_id: str
    spec_hash: str  # SHA256 of ExperimentSpec
    dataset_hash: str  # Hash of input dataset
    code_version: str  # Version of harness code
    random_seed: int  # Seed used
    created_at: str  # ISO timestamp
    completed_at: str  # ISO timestamp
    spec_path: Optional[str] = None  # Path to spec file (for reference)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ReproducibilityRecorder:
    """Record and verify reproducibility information."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize recorder.

        Args:
            output_dir: Directory to store reproducibility records
                       Defaults to ./outputs/experiments/
        """
        self.output_dir = output_dir or (Path.cwd() / "outputs" / "experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_run(
        self,
        spec: ExperimentSpec,
        execution_run: ExecutionRun,
        spec_path: Optional[str] = None,
    ) -> ReproducibilityRecord:
        """
        Record reproducibility information for a run.

        Args:
            spec: ExperimentSpec
            execution_run: ExecutionRun with execution details
            spec_path: Optional path to spec file

        Returns:
            ReproducibilityRecord
        """
        # Compute spec hash
        spec_json = self._spec_to_json(spec)
        spec_hash = hashlib.sha256(spec_json.encode()).hexdigest()[:16]

        record = ReproducibilityRecord(
            experiment_id=spec.experiment_id,
            run_id=execution_run.run_id,
            variant_id=execution_run.variant_id,
            spec_hash=spec_hash,
            dataset_hash=execution_run.dataset_hash,
            code_version=execution_run.code_version,
            random_seed=execution_run.random_seed,
            created_at=execution_run.started_at,
            completed_at=execution_run.completed_at,
            spec_path=spec_path,
        )

        # Save record to file
        self._save_record(record)

        return record

    def verify_reproducibility(self, record: ReproducibilityRecord) -> bool:
        """
        Verify that reproducibility information is sufficient.

        Args:
            record: ReproducibilityRecord to verify

        Returns:
            True if reproducible, False otherwise
        """
        # Check all required fields are present
        required_fields = [
            "experiment_id",
            "run_id",
            "variant_id",
            "spec_hash",
            "dataset_hash",
            "code_version",
            "random_seed",
            "created_at",
            "completed_at",
        ]

        for field in required_fields:
            value = getattr(record, field, None)
            if value is None or value == "":
                return False

        return True

    def _spec_to_json(self, spec: ExperimentSpec) -> str:
        """Convert ExperimentSpec to JSON string."""
        spec_dict = {
            "experiment_id": spec.experiment_id,
            "hypothesis": spec.hypothesis,
            "changed_variable": spec.changed_variable,
            "baseline_id": spec.baseline_id,
            "variant_id": spec.variant_id,
            "metrics_used": spec.metrics_used,
            "minimum_runs": spec.minimum_runs,
            "created_at": spec.created_at,
            "author": spec.author,
        }
        return json.dumps(spec_dict, sort_keys=True)

    def _save_record(self, record: ReproducibilityRecord) -> None:
        """Save reproducibility record to file."""
        experiment_dir = self.output_dir / record.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        record_file = experiment_dir / f"{record.run_id}-reproducibility.json"
        record_file.write_text(record.to_json())

    @staticmethod
    def load_record(record_path: Path) -> ReproducibilityRecord:
        """
        Load reproducibility record from file.

        Args:
            record_path: Path to reproducibility record file

        Returns:
            ReproducibilityRecord
        """
        with open(record_path, "r") as f:
            data = json.load(f)
        return ReproducibilityRecord(**data)
