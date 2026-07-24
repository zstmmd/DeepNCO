from __future__ import annotations

from typing import Any


def configure_outer_certification_search(model: Any) -> None:
    """Favor verified incumbents inside the fixed 4% certification slices."""

    params = model.Params
    params.MIPFocus = 1
    params.MIPGap = 0.0
    params.PoolSearchMode = 0
    params.Heuristics = max(0.35, float(params.Heuristics))
    params.PumpPasses = max(20, int(params.PumpPasses))
    params.StartNodeLimit = max(2000, int(params.StartNodeLimit))
    params.RINS = 10
