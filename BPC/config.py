from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


DEFAULT_SCALES: Tuple[str, ...] = tuple(f"GUROBI-S{i}" for i in range(1, 10))


@dataclass
class BPCConfig:
    seed: int = 42
    scales: Tuple[str, ...] = DEFAULT_SCALES
    small_scale_time_limit_sec: float = 600.0
    large_scale_time_limit_sec: float = 3600.0
    large_scale_start_index: int = 6
    gurobi_baseline_dir: str = "result/gurobi_s1_s9_current_200s_20260516"
    output_dir: str = ""
    pricing_time_limit_sec: float = 30.0
    pricing_max_labels: int = 200000
    pricing_reduced_cost_tol: float = 1e-9
    exact_gap_tol: float = 1e-9
    max_branch_nodes: int = 100000
    enable_sp4_greedy_fallback: bool = True
    experiment_name: str = "bpc_s1_s9"
    metadata: Dict[str, object] = field(default_factory=dict)

    def time_limit_for_scale(self, scale: str) -> float:
        idx = self.scale_index(scale)
        if idx >= int(self.large_scale_start_index):
            return float(self.large_scale_time_limit_sec)
        return float(self.small_scale_time_limit_sec)

    @staticmethod
    def scale_index(scale: str) -> int:
        text = str(scale or "").upper()
        prefix = "GUROBI-S"
        if not text.startswith(prefix):
            return 0
        digits = []
        for ch in text[len(prefix):]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        return int("".join(digits) or "0")

    @classmethod
    def normalize_scales(cls, values: Iterable[str] | str) -> Tuple[str, ...]:
        if isinstance(values, str):
            raw = values.split(",")
        else:
            raw = list(values)
        out = tuple(str(item).strip().upper() for item in raw if str(item).strip())
        return out or DEFAULT_SCALES
