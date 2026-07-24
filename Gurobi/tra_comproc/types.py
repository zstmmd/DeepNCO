from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from Gurobi.tra_outer_start import FullStartVector


@dataclass(frozen=True)
class DP1RouteResult:
    feasible: bool
    route_end_sec: float
    slot_arrival_lower: Mapping[int, float]
    robot_paths: Mapping[int, tuple[int, ...]]
    error_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DP2ServiceResult:
    feasible: bool
    slot_arrival: Mapping[int, float]
    slot_process_duration: Mapping[int, float]
    station_by_slot: Mapping[int, int]
    error_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DP3RecoveryResult:
    feasible: bool
    no_wait_cmax: float
    feasible_start_cmax: float
    recourse_score: float
    station_overlap_sec: float
    station_workload_imbalance: float
    active_slot_count: int
    station_orders: Mapping[int, tuple[int, ...]]
    error_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComProcResult:
    feasible: bool
    projected_cmax: float
    recourse_score: float
    verified_cmax: float
    projected_objective: float
    feasibility_residual: float
    source: str
    dp1: DP1RouteResult
    dp2: DP2ServiceResult
    dp3: DP3RecoveryResult
    full_start: Optional[FullStartVector]
    error_codes: tuple[str, ...] = ()
