from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .state import ResourceConfig
from .u_repair_solver import UFastRepairSolver, URoutePlan


@dataclass
class RevolvingCandidate:
    config: ResourceConfig
    u_plan: Optional[URoutePlan]
    released_layer: str
    fixed_layers: str
    inner_relaxed_obj: float
    revolving_lb: float
    lb_gate_skipped: bool
    changed_subtask_ids: Iterable[int]
    changed_robot_ids: Iterable[int]

    def metadata(self) -> Dict[str, Any]:
        u_meta = self.u_plan.to_metadata() if self.u_plan is not None else {}
        return {
            **u_meta,
            "revolving_enabled": True,
            "released_layer": str(self.released_layer).upper(),
            "fixed_layers": str(self.fixed_layers),
            "inner_relaxed_obj": float(self.inner_relaxed_obj),
            "revolving_lb": float(self.revolving_lb),
            "lb_gate_skipped": bool(self.lb_gate_skipped),
            "changed_subtask_ids": [int(x) for x in (self.changed_subtask_ids or [])],
            "changed_robot_ids": [int(x) for x in (self.changed_robot_ids or [])],
        }


def _station_workload_lb(config: ResourceConfig) -> float:
    station_loads: Dict[int, float] = {}
    for row in config.subtasks.values():
        if int(row.station_id) < 0:
            continue
        service = 0.0
        for task in row.z_tasks or []:
            service += float(getattr(task, "station_service_time", 0.0) or 0.0)
            service += float(max(1, int(getattr(task, "sku_pick_count", 0) or 0)))
        station_loads[int(row.station_id)] = float(station_loads.get(int(row.station_id), 0.0) + service)
    return float(max(station_loads.values()) if station_loads else 0.0)


def _order_service_lb(config: ResourceConfig) -> float:
    order_loads: Dict[int, float] = {}
    for row in config.subtasks.values():
        service = 0.0
        for task in row.z_tasks or []:
            service += float(getattr(task, "robot_service_time", 0.0) or 0.0)
            service += float(getattr(task, "station_service_time", 0.0) or 0.0)
            service += float(max(1, int(getattr(task, "sku_pick_count", 0) or 0)))
        order_loads[int(row.order_id)] = float(order_loads.get(int(row.order_id), 0.0) + service)
    return float(max(order_loads.values()) if order_loads else 0.0)


class RevolvingSolver:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.u_solver = UFastRepairSolver(opt)

    @staticmethod
    def fixed_layers_for_release(layer: str) -> str:
        layer_name = str(layer or "").upper()
        if layer_name == "X":
            return "Y,Z,U"
        if layer_name == "Y":
            return "X,Z,U"
        if layer_name == "Z":
            return "X,Y,U"
        if layer_name == "U":
            return "X,Y,Z"
        return "X,Y,Z,U"

    def attach_u_plan(
        self,
        config: ResourceConfig,
        *,
        released_layer: str,
        affected_subtask_ids: Iterable[int] | None = None,
        incumbent_value: float = float("inf"),
    ) -> RevolvingCandidate:
        normalized = config.clone().rebuild_indices()
        prev_route_plan = dict((getattr(normalized, "metadata", {}) or {}).get("fixed_route_task_sequence_by_robot", {}) or {})
        u_plan = self.u_solver.repair(
            normalized,
            previous_route_plan=prev_route_plan,
            affected_subtask_ids=affected_subtask_ids or [],
        )
        station_lb = _station_workload_lb(normalized)
        order_lb = _order_service_lb(normalized)
        u_lb = float(getattr(u_plan, "u_route_lb", 0.0) or 0.0)
        lb = float(max(float(station_lb), float(order_lb), float(u_lb)))
        lb = float(min(lb, float(getattr(u_plan, "u_fast_cmax", lb) or lb))) if math.isfinite(lb) else float("inf")
        lb_gate_skipped = bool(math.isfinite(float(incumbent_value)) and math.isfinite(lb) and lb >= float(incumbent_value) - 1e-6)
        changed_robot_ids = list(getattr(u_plan, "changed_robot_ids", []) or [])
        metadata = dict(getattr(normalized, "metadata", {}) or {})
        metadata.update(u_plan.to_metadata())
        metadata.update(
            {
                "revolving_enabled": True,
                "released_layer": str(released_layer).upper(),
                "fixed_layers": self.fixed_layers_for_release(released_layer),
                "inner_relaxed_obj": float(getattr(u_plan, "u_fast_cmax", float("inf"))),
                "revolving_lb": float(lb),
                "lb_gate_skipped": bool(lb_gate_skipped),
                "changed_subtask_ids": [int(x) for x in (affected_subtask_ids or [])],
                "changed_robot_ids": [int(x) for x in changed_robot_ids],
            }
        )
        normalized.metadata = metadata
        return RevolvingCandidate(
            config=normalized,
            u_plan=u_plan,
            released_layer=str(released_layer).upper(),
            fixed_layers=self.fixed_layers_for_release(released_layer),
            inner_relaxed_obj=float(getattr(u_plan, "u_fast_cmax", float("inf"))),
            revolving_lb=float(lb),
            lb_gate_skipped=bool(lb_gate_skipped),
            changed_subtask_ids=[int(x) for x in (affected_subtask_ids or [])],
            changed_robot_ids=[int(x) for x in changed_robot_ids],
        )
