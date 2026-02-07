"""
Phase 5.3.2: Experiment Harness

Offline, safe, governed experiment execution framework.
Executes experiments without affecting agent runtime or automatically changing behavior.

This harness:
- Loads and validates experiment specifications (Phase 5.3.1)
- Runs baseline and variant experiments offline
- Computes metrics using Phase 5.2
- Compares results and detects regressions
- Generates human-readable reports
- Records full reproducibility information

What it does NOT do:
- Modify agent code
- Change prompts or models automatically
- Auto-accept or auto-rollback
- Feed metrics back into execution
- Skip protocol validation
"""

__version__ = "5.3.2"

from experiment_harness.loader import SpecLoader, LoaderError
from experiment_harness.executor import ExperimentExecutor, ExecutorError
from experiment_harness.evaluator import ExperimentEvaluator, EvaluatorError
from experiment_harness.reporter import ComparisonReporter, ReporterError
from experiment_harness.reproducibility import ReproducibilityRecorder
from experiment_harness.runner import ExperimentRunner

__all__ = [
    "SpecLoader",
    "LoaderError",
    "ExperimentExecutor",
    "ExecutorError",
    "ExperimentEvaluator",
    "EvaluatorError",
    "ComparisonReporter",
    "ReporterError",
    "ReproducibilityRecorder",
    "ExperimentRunner",
]
