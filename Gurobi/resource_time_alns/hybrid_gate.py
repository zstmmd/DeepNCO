from __future__ import annotations

import math
from typing import Iterable


def should_run_hybrid_exact(
    *,
    iter_id: int,
    last_exact_iter: int,
    layer: str,
    allowed_layers: Iterable[str],
    period: int,
    margin_ratio: float,
    current_best: float,
    proxy_value: float,
    revolving_lb: float,
) -> bool:
    if int(iter_id) <= 0:
        return True
    if int(last_exact_iter) == int(iter_id):
        return False
    allowed = {str(value).strip().upper() for value in allowed_layers if str(value).strip()}
    if str(layer).upper() not in allowed:
        return False
    if int(iter_id) % max(1, int(period)) != 0:
        return False
    margin = max(0.0, float(margin_ratio))
    promising_proxy = math.isfinite(current_best) and proxy_value <= current_best * (1.0 + margin)
    promising_lb = math.isfinite(current_best) and math.isfinite(revolving_lb) and revolving_lb <= current_best
    return bool(promising_proxy or promising_lb or not math.isfinite(current_best))
