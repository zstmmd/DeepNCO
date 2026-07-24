from __future__ import annotations

import math
from typing import Any


def _finite_or_inf(value: float) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else float("inf")


def comproc_candidate_key(
    candidate: Any,
) -> tuple[int, float, float, float, float, float, str]:
    result = getattr(candidate, "comproc", None)
    feasible = result is not None and bool(result.feasible)
    cmax_score = (
        getattr(result, "projected_cmax", float("inf"))
        if feasible
        else getattr(result, "verified_cmax", float("inf"))
    )
    return (
        0 if feasible else 1,
        _finite_or_inf(getattr(result, "recourse_score", cmax_score)),
        _finite_or_inf(cmax_score),
        _finite_or_inf(getattr(result, "projected_objective", float("inf"))),
        _finite_or_inf(candidate.repair_risk.total),
        _finite_or_inf(candidate.relaxed_objective),
        str(candidate.shell.sha256),
    )
