from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BPCRouteTask:
    task_key: int
    source_task_id: int
    subtask_id: int
    order_id: int
    stack_id: int
    station_id: int
    pickup_xy: Tuple[float, float]
    delivery_xy: Tuple[float, float]
    service_time: float
    load: int = 1
    station_rank: int = 0


@dataclass
class BPCRouteColumn:
    column_id: int
    robot_id: int
    task_keys: Tuple[int, ...]
    sequence: Tuple[int, ...]
    arrival_at_station: Dict[int, float]
    finish_time: float
    travel_time: float
    service_time: float
    reduced_cost: float = 0.0


@dataclass
class PricingResult:
    columns: List[BPCRouteColumn] = field(default_factory=list)
    exact: bool = False
    timed_out: bool = False
    label_limit_hit: bool = False
    expanded_labels: int = 0
    best_reduced_cost: float = 0.0


@dataclass
class MasterResult:
    status: str
    objective: float
    lower_bound: float
    dual_task_cover: Dict[int, float] = field(default_factory=dict)
    selected_columns: Dict[int, float] = field(default_factory=dict)
    integer: bool = False


@dataclass
class BranchNode:
    node_id: int
    depth: int = 0
    fixed_task_robot: Dict[int, int] = field(default_factory=dict)
    forbidden_task_robot: Dict[Tuple[int, int], bool] = field(default_factory=dict)
    lower_bound: float = 0.0
    status: str = "OPEN"


@dataclass
class BPCCertificate:
    exact: bool
    reason: str
    incumbent_found: bool
    integer_solution: bool
    all_nodes_closed: bool
    pricing_exact: bool
    no_negative_reduced_cost: bool
    open_nodes: int
    upper_bound: float
    lower_bound: float
    gap: float

    @classmethod
    def evaluate(
        cls,
        incumbent_found: bool,
        integer_solution: bool,
        all_nodes_closed: bool,
        pricing_exact: bool,
        no_negative_reduced_cost: bool,
        open_nodes: int,
        upper_bound: float,
        lower_bound: float,
        tol: float = 1e-9,
    ) -> "BPCCertificate":
        ub = float(upper_bound)
        lb = float(lower_bound)
        if math.isfinite(ub) and math.isfinite(lb) and abs(ub) > 1e-12:
            gap = max(0.0, (ub - lb) / max(1.0, abs(ub)))
        elif math.isfinite(ub) and math.isfinite(lb):
            gap = max(0.0, ub - lb)
        else:
            gap = float("inf")
        exact = bool(
            incumbent_found
            and integer_solution
            and all_nodes_closed
            and pricing_exact
            and no_negative_reduced_cost
            and int(open_nodes) == 0
            and math.isfinite(ub)
            and math.isfinite(lb)
            and abs(ub - lb) <= float(tol)
        )
        reason = "proved_full_space_optimal"
        if not exact:
            checks = [
                ("no_incumbent", not incumbent_found),
                ("non_integer_incumbent", not integer_solution),
                ("open_nodes_remaining", not all_nodes_closed or int(open_nodes) != 0),
                ("pricing_not_exact", not pricing_exact),
                ("negative_reduced_cost_remaining", not no_negative_reduced_cost),
                ("gap_not_closed", not (math.isfinite(ub) and math.isfinite(lb) and abs(ub - lb) <= float(tol))),
            ]
            reason = next(name for name, failed in checks if failed)
        return cls(
            exact=exact,
            reason=reason,
            incumbent_found=bool(incumbent_found),
            integer_solution=bool(integer_solution),
            all_nodes_closed=bool(all_nodes_closed),
            pricing_exact=bool(pricing_exact),
            no_negative_reduced_cost=bool(no_negative_reduced_cost),
            open_nodes=int(open_nodes),
            upper_bound=ub,
            lower_bound=lb,
            gap=float(gap),
        )


@dataclass
class BPCResult:
    status: str
    objective: float
    lower_bound: float
    gap: float
    runtime_sec: float
    exact: bool
    certificate: BPCCertificate
    route_columns: Sequence[BPCRouteColumn] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)
