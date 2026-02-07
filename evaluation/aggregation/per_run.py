"""
Phase 5.2 aggregation - Per-Run Metrics

Aggregates metrics across multiple sessions to produce run-level statistics.
Computes means, percentiles, and distributions.
"""

from typing import List, Dict, Any
from datetime import datetime
import statistics
from evaluation.aggregation.per_session import compute_per_session_metrics
from evaluation.schemas.trace_schema import Trace
from evaluation.schemas.metric_schema import ComputationResults


def compute_per_run_metrics(traces_by_session: Dict[str, List[Trace]]) -> Dict[str, Any]:
    """
    Compute run-level statistics across multiple sessions.

    Args:
        traces_by_session: Dictionary mapping session_id -> List[Trace]

    Returns:
        Dictionary containing run-level metrics
    """
    if not traces_by_session:
        raise ValueError("Must provide at least one session")

    # Compute per-session metrics
    session_results = {}
    all_metric_ids = set()

    for session_id, traces in traces_by_session.items():
        results = compute_per_session_metrics(traces)
        session_results[session_id] = results
        all_metric_ids.update(results.results.keys())

    # Aggregate across sessions
    run_metrics = {}

    for metric_id in all_metric_ids:
        values = []

        # Collect values from all sessions
        for session_id, results in session_results.items():
            if metric_id in results.results:
                metric_result = results.results[metric_id]
                if metric_result.valid:
                    values.append(metric_result.value)

        if not values:
            run_metrics[metric_id] = {
                "valid": False,
                "error": "No valid samples",
            }
            continue

        # Compute statistics
        sorted_values = sorted(values)
        n = len(sorted_values)

        run_metrics[metric_id] = {
            "valid": True,
            "count": n,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p25": sorted_values[n // 4] if n > 0 else None,
            "p50": sorted_values[n // 2] if n > 0 else None,
            "p75": sorted_values[(3 * n) // 4] if n > 0 else None,
            "p95": sorted_values[int(0.95 * n)] if n > 0 else None,
        }

    return {
        "run_id": datetime.utcnow().isoformat(),
        "session_count": len(traces_by_session),
        "metrics": run_metrics,
        "session_results": {
            session_id: {
                "session_id": results.session_id,
                "timestamp": results.timestamp,
                "metrics": {
                    mid: {
                        "value": mres.value,
                        "samples": mres.samples,
                        "valid": mres.valid,
                    }
                    for mid, mres in results.results.items()
                },
            }
            for session_id, results in session_results.items()
        },
    }


def extract_metric_values(
    run_results: Dict[str, Any], metric_id: str
) -> List[float]:
    """
    Extract all values for a specific metric across all sessions.

    Args:
        run_results: Output from compute_per_run_metrics
        metric_id: Which metric to extract

    Returns:
        List of metric values from all sessions
    """
    values = []

    for session_id, session_data in run_results.get("session_results", {}).items():
        metrics = session_data.get("metrics", {})
        if metric_id in metrics:
            metric_data = metrics[metric_id]
            if metric_data.get("valid", False):
                values.append(metric_data["value"])

    return values
