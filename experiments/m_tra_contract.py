from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


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
