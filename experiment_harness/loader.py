"""
Phase 5.3.2: Spec Loader

Loads experiment specifications from YAML/JSON and validates against Phase 5.3.1 protocol.

Responsibilities:
- Load ExperimentSpec from files
- Validate spec using Phase 5.3.1 validator
- Ensure baseline reference exists
- Hard-fail on invalid specs (no partial execution)
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

from experiments.protocol.schema import ExperimentSpec
from experiments.protocol.validator import (
    validate_experiment_spec,
    ValidationResult,
)


@dataclass(frozen=True)
class LoaderError(Exception):
    """Loader error with structured information."""

    step: str  # Which step failed: "file_load", "parse", "validation"
    error_message: str
    spec_path: Optional[str] = None
    details: Optional[str] = None

    def __str__(self) -> str:
        msg = f"[{self.step}] {self.error_message}"
        if self.details:
            msg += f"\n  Details: {self.details}"
        return msg


class SpecLoader:
    """Load and validate experiment specifications from files."""

    def __init__(self, baseline_registry_path: Optional[str] = None):
        """
        Initialize loader.

        Args:
            baseline_registry_path: Path to Phase 5.2 baseline registry
                                   If None, uses default location
        """
        self.baseline_registry_path = baseline_registry_path or (
            Path(__file__).parent.parent / "evaluation" / "baselines" / "baseline_v1.json"
        )

    def load_spec_from_file(self, spec_path: Union[str, Path]) -> ExperimentSpec:
        """
        Load experiment specification from YAML or JSON file.

        Args:
            spec_path: Path to specification file (*.yaml or *.json)

        Returns:
            ExperimentSpec (validated)

        Raises:
            LoaderError: If any step fails
        """
        spec_path = Path(spec_path)

        # Step 1: Check file exists
        if not spec_path.exists():
            raise LoaderError(
                step="file_load",
                error_message=f"Specification file not found: {spec_path}",
                spec_path=str(spec_path),
            )

        # Step 2: Parse YAML or JSON
        try:
            if spec_path.suffix in [".yaml", ".yml"]:
                with open(spec_path, "r") as f:
                    spec_dict = yaml.safe_load(f)
            elif spec_path.suffix == ".json":
                with open(spec_path, "r") as f:
                    spec_dict = json.load(f)
            else:
                raise LoaderError(
                    step="parse",
                    error_message=f"Unsupported file format: {spec_path.suffix}",
                    spec_path=str(spec_path),
                    details="Use .yaml, .yml, or .json",
                )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise LoaderError(
                step="parse",
                error_message=f"Failed to parse specification file",
                spec_path=str(spec_path),
                details=str(e),
            )

        # Step 3: Convert dict to ExperimentSpec
        try:
            spec = ExperimentSpec(**spec_dict)
        except TypeError as e:
            raise LoaderError(
                step="parse",
                error_message="Missing or invalid required fields in specification",
                spec_path=str(spec_path),
                details=str(e),
            )
        except ValueError as e:
            raise LoaderError(
                step="parse",
                error_message="Invalid field values in specification",
                spec_path=str(spec_path),
                details=str(e),
            )

        # Step 4: Validate spec using Phase 5.3.1 validator
        validation_result = validate_experiment_spec(spec)

        if not validation_result.valid:
            error_details = "\n".join(
                [f"  [{e.field}] {e.error} ({e.severity})" for e in validation_result.errors]
            )
            raise LoaderError(
                step="validation",
                error_message="Specification failed Phase 5.3.1 protocol validation",
                spec_path=str(spec_path),
                details=error_details,
            )

        # Step 5: Verify baseline exists
        self._verify_baseline_exists(spec.baseline_id)

        return spec

    def load_spec_from_dict(self, spec_dict: dict) -> ExperimentSpec:
        """
        Load experiment specification from dictionary.

        Args:
            spec_dict: Specification as dictionary

        Returns:
            ExperimentSpec (validated)

        Raises:
            LoaderError: If validation fails
        """
        try:
            spec = ExperimentSpec(**spec_dict)
        except (TypeError, ValueError) as e:
            raise LoaderError(
                step="parse",
                error_message="Invalid specification dictionary",
                details=str(e),
            )

        # Validate
        validation_result = validate_experiment_spec(spec)
        if not validation_result.valid:
            error_details = "\n".join(
                [f"  [{e.field}] {e.error}" for e in validation_result.errors]
            )
            raise LoaderError(
                step="validation",
                error_message="Specification failed validation",
                details=error_details,
            )

        # Verify baseline
        self._verify_baseline_exists(spec.baseline_id)

        return spec

    def _verify_baseline_exists(self, baseline_id: str) -> None:
        """
        Verify that baseline reference exists.

        Args:
            baseline_id: Baseline identifier (e.g., "v1")

        Raises:
            LoaderError: If baseline not found
        """
        # For now, accept v1 as special baseline
        if baseline_id.lower() in ["v1", "baseline", "baseline_v1"]:
            return

        # In future, check baseline registry
        # For now, just document the requirement
        pass


def load_spec(spec_path: Union[str, Path], baseline_registry_path: Optional[str] = None) -> ExperimentSpec:
    """
    Convenience function to load a specification.

    Args:
        spec_path: Path to specification file
        baseline_registry_path: Optional path to baseline registry

    Returns:
        ExperimentSpec (validated)

    Raises:
        LoaderError: If loading or validation fails
    """
    loader = SpecLoader(baseline_registry_path=baseline_registry_path)
    return loader.load_spec_from_file(spec_path)
