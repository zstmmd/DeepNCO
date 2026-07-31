from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _matching_verified_event(
    row: Mapping[str, Any],
    *,
    case_id: str,
    target_cmax: float,
    run_id: str,
    tolerance: float,
) -> bool:
    if str(row.get("case", "")) != str(case_id):
        return False
    if run_id and str(row.get("run_id", "")) != str(run_id):
        return False
    if not bool(row.get("internal_feasible", False)):
        return False
    verified_cmax = _finite_float(row.get("verified_cmax"))
    return math.isfinite(verified_cmax) and abs(verified_cmax - target_cmax) <= tolerance


def first_verified_target_time_from_events(
    rows: Iterable[Mapping[str, Any]],
    *,
    case_id: str,
    target_cmax: float,
    run_id: str = "",
    tolerance: float = 1e-5,
) -> float:
    """Return the first verifier-complete target event without exposing it to search."""

    target = _finite_float(target_cmax)
    if not math.isfinite(target):
        return float("nan")
    matching_times = []
    for row in rows:
        if not _matching_verified_event(
            row,
            case_id=case_id,
            target_cmax=target,
            run_id=run_id,
            tolerance=max(0.0, float(tolerance)),
        ):
            continue
        timestamp = _finite_float(row.get("wall_timestamp_sec"))
        if math.isfinite(timestamp):
            matching_times.append(timestamp)
    return min(matching_times) if matching_times else float("nan")


def summarize_verified_events(
    rows: Iterable[Mapping[str, Any]],
    *,
    case_id: str,
    target_cmax: float,
    run_id: str = "",
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Summarize target-aware acceptance evidence after the solver process exits."""

    target = _finite_float(target_cmax)
    if not math.isfinite(target):
        return {
            "case": str(case_id),
            "run_id": str(run_id),
            "target_cmax": target,
            "best_verified_cmax": float("nan"),
            "cmax_equal": False,
            "lower_than_target": False,
            "first_verified_target_time_sec": float("nan"),
            "first_solver_incumbent_target_time_sec": float("nan"),
        }

    tolerance_value = max(0.0, float(tolerance))
    verified_values = []
    matching_wall_times = []
    matching_solver_times = []
    for row in rows:
        if str(row.get("case", "")) != str(case_id):
            continue
        if run_id and str(row.get("run_id", "")) != str(run_id):
            continue
        if not bool(row.get("internal_feasible", False)):
            continue
        verified_cmax = _finite_float(row.get("verified_cmax"))
        if not math.isfinite(verified_cmax):
            continue
        verified_values.append(verified_cmax)
        if abs(verified_cmax - target) > tolerance_value:
            continue
        wall_timestamp = _finite_float(row.get("wall_timestamp_sec"))
        if math.isfinite(wall_timestamp):
            matching_wall_times.append(wall_timestamp)
        solver_timestamp = _finite_float(row.get("solver_incumbent_timestamp_sec"))
        if math.isfinite(solver_timestamp):
            matching_solver_times.append(solver_timestamp)

    best_cmax = min(verified_values) if verified_values else float("nan")
    return {
        "case": str(case_id),
        "run_id": str(run_id),
        "target_cmax": target,
        "best_verified_cmax": best_cmax,
        "cmax_equal": bool(
            math.isfinite(best_cmax) and abs(best_cmax - target) <= tolerance_value
        ),
        "lower_than_target": bool(
            math.isfinite(best_cmax) and best_cmax < target - tolerance_value
        ),
        "first_verified_target_time_sec": (
            min(matching_wall_times) if matching_wall_times else float("nan")
        ),
        "first_solver_incumbent_target_time_sec": (
            min(matching_solver_times) if matching_solver_times else float("nan")
        ),
    }


def time_to_target_from_iter_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_cmax: float,
    tolerance: float = 1e-6,
) -> float:
    """Compute acceptance time after a run without exposing the target to search."""
    target = _finite_float(target_cmax)
    if not math.isfinite(target):
        return float("nan")

    elapsed = 0.0
    for row in rows:
        iter_runtime = _finite_float(row.get("iter_runtime_sec"))
        if math.isfinite(iter_runtime):
            elapsed += max(0.0, iter_runtime)
        validated = _finite_float(row.get("validated_makespan"))
        if not math.isfinite(validated) and bool(row.get("incumbent_internal_feasible", False)):
            validated = _finite_float(row.get("best_z"))
        if math.isfinite(validated) and abs(validated - target) <= max(0.0, float(tolerance)):
            return float(elapsed)
    return float("nan")
