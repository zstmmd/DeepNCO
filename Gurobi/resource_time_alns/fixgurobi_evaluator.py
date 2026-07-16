from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver

from config.ofs_config import OFSConfig

from .state import ResourceConfig, ResourceSubtask, UpperEvalResult
from .route_edge_audit import allowed_route_edges_from_global_payload, audit_fixed_route_edges


class FixGurobiEvaluator:
    """Evaluate TRA resource configs by fixing layer-specific decisions in GlobalXYZU."""

    def __init__(self, opt, surrogate_scorer=None) -> None:
        self.opt = opt
        self.cfg = opt.cfg
        self.surrogate_scorer = surrogate_scorer
        self.cache: OrderedDict[Tuple[Any, ...], UpperEvalResult] = OrderedDict()
        self.compiled_cache: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()
        self.route_signature_cache: OrderedDict[Tuple[Any, ...], UpperEvalResult] = OrderedDict()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.compiled_cache_hit_count = 0
        self.compiled_cache_miss_count = 0
        self.cheap_lb_gate_reject_count = 0
        self.route_signature_cache_hit_count = 0
        self.current_best_value = float("inf")

    def _remaining_wall_budget_sec(self) -> float:
        wall_limit = float(getattr(self.cfg, "resource_wall_time_limit_sec", 0.0) or 0.0)
        if wall_limit <= 0.0:
            return float("inf")
        elapsed = 0.0
        try:
            elapsed = float(self.opt._runtime_elapsed_sec())
        except Exception:
            elapsed = 0.0
        return float(wall_limit - elapsed)
    def _cache_size(self) -> int:
        return max(1, int(getattr(self.cfg, "fixgurobi_cache_size", 128) or 128))

    def _compiled_cache_size(self) -> int:
        return max(1, int(getattr(self.cfg, "fixgurobi_compiled_cache_size", 8) or 8))

    def _route_signature_cache_size(self) -> int:
        return max(1, int(getattr(self.cfg, "fixgurobi_route_signature_cache_size", 256) or 256))

    def _route_signature(self, config: ResourceConfig, scope: str) -> Optional[Tuple[Any, ...]]:
        """Cheap signature of the fixed route sequence (only meaningful when U is fixed)."""
        if not self._fixes_u(scope):
            return None
        metadata = getattr(config, "metadata", {}) or {}
        route_nodes = dict(metadata.get("fixed_route_node_sequence_by_robot", {}) or {})
        route_tasks = dict(metadata.get("fixed_route_task_sequence_by_robot", {}) or {})
        if not route_nodes and not route_tasks:
            return None
        node_sig = tuple(
            (
                int(robot_id),
                tuple(
                    (
                        str(row.get("kind", "")),
                        int(row.get("subtask_id", row.get("local_slot_index", -1))),
                        int(row.get("order_id", -1)),
                        int(row.get("stack_id", -1)),
                        int(row.get("station_id", -1)),
                    )
                    for row in (rows or [])
                ),
            )
            for robot_id, rows in sorted(route_nodes.items(), key=lambda item: int(item[0]))
        )
        task_sig = tuple(
            (
                int(robot_id),
                tuple(
                    (
                        int(row.get("subtask_id", -1)),
                        int(row.get("order_id", -1)),
                        int(row.get("stack_id", -1)),
                        int(row.get("station_id", -1)),
                    )
                    for row in (rows or [])
                ),
            )
            for robot_id, rows in sorted(route_tasks.items(), key=lambda item: int(item[0]))
        )
        return (node_sig, task_sig)

    def _route_signature_cache_get(self, sig_key: Tuple[Any, ...]) -> Optional[UpperEvalResult]:
        cached = self.route_signature_cache.get(sig_key)
        if cached is None:
            return None
        cache_t0 = time.perf_counter()
        self.route_signature_cache.move_to_end(sig_key)
        self.route_signature_cache_hit_count += 1
        out = copy.deepcopy(cached)
        out.metadata["fixgurobi_route_signature_cache_hit"] = True
        out.metadata["fixgurobi_solve_time"] = 0.0
        out.metadata["fixgurobi_wall_time"] = float(time.perf_counter() - cache_t0)
        out.metadata["fixgurobi_route_signature_cache_hit_count"] = int(self.route_signature_cache_hit_count)
        return out

    def _route_signature_cache_put(self, sig_key: Tuple[Any, ...], value: UpperEvalResult) -> None:
        self.route_signature_cache[sig_key] = copy.deepcopy(value)
        self.route_signature_cache.move_to_end(sig_key)
        while len(self.route_signature_cache) > self._route_signature_cache_size():
            self.route_signature_cache.popitem(last=False)

    def _cheap_cmax_lower_bound(self, config: ResourceConfig, scope: str) -> float:
        """Admissible (conservative) lower bound on Cmax for a fully X/Y/Z-fixed candidate.

        The global model serialises all slots assigned to the same station (StationSeq
        clocks) and sets each slot's processing time to
        PICKING_TIME * sku_pick_count + station_service_time (FinishDef / FCFS replay).
        Therefore the total processing load on any single station is a true lower bound on
        Cmax. We deliberately do NOT use a per-order span bound: an order's slots may be
        assigned to different stations and run in parallel, so summing per-order load would
        not be admissible. Only valid when the scope fixes X, Y and Z (otherwise station
        assignment is free) -> returns -inf to disable pruning in that case.
        """
        scope_name = str(scope or "").upper()
        if scope_name in {"LOCALXYZ", "LOCALYZ"}:
            return float("-inf")
        if not (self._fixes_x(scope) and self._fixes_y(scope) and self._fixes_z(scope)):
            return float("-inf")
        pick_time = float(getattr(OFSConfig, "PICKING_TIME", 1.0) or 0.0)
        load_by_station: Dict[int, float] = defaultdict(float)
        for row in config.subtasks.values():
            station_id = int(row.station_id)
            if station_id < 0:
                continue
            row_load = 0.0
            for task in row.z_tasks or []:
                row_load += float(max(0, int(getattr(task, "sku_pick_count", 0) or 0))) * pick_time
                if getattr(task, "noise_tote_ids", None):
                    row_load += float(getattr(task, "station_service_time", 0.0) or 0.0)
            load_by_station[station_id] += float(row_load)
        if not load_by_station:
            return float("-inf")
        return float(max(load_by_station.values()))

    def _cache_get(self, cache_key: Tuple[Any, ...]) -> Optional[UpperEvalResult]:
        cached = self.cache.get(cache_key)
        if cached is None:
            return None
        cache_t0 = time.perf_counter()
        self.cache.move_to_end(cache_key)
        self.cache_hit_count += 1
        out = copy.deepcopy(cached)
        original_solve_time = float(out.metadata.get("fixgurobi_solve_time", 0.0) or 0.0)
        out.metadata["fixgurobi_cache_hit"] = True
        out.metadata["fixgurobi_cached_original_solve_time"] = float(original_solve_time)
        out.metadata["fixgurobi_solve_time"] = 0.0
        out.metadata["fixgurobi_wall_time"] = float(time.perf_counter() - cache_t0)
        out.metadata["fixgurobi_cache_hit_count"] = int(self.cache_hit_count)
        return out

    def _cache_put(self, cache_key: Tuple[Any, ...], value: UpperEvalResult) -> None:
        self.cache[cache_key] = copy.deepcopy(value)
        self.cache.move_to_end(cache_key)
        while len(self.cache) > self._cache_size():
            self.cache.popitem(last=False)

    def _cache_context_signature(self, scope: str) -> Tuple[Any, ...]:
        best = float(getattr(self, "current_best_value", float("inf")) or float("inf"))
        local_release_scope = str(scope or "").upper() in {"LOCALXYZ", "LOCALYZ"}
        if bool(getattr(self.cfg, "resource_revolving_allow_nonimproving_exact", False)) and not bool(local_release_scope):
            best = float("inf")
        use_two_stage = bool(getattr(self.cfg, "fixgurobi_enable_two_stage", True)) and math.isfinite(best)
        use_cutoff = bool(getattr(self.cfg, "fixgurobi_enable_cutoff", True)) and math.isfinite(best)
        cutoff = float(best - 1e-6) if use_cutoff else float("nan")
        target = float(getattr(self.cfg, "resource_target_cmax", float("nan")))
        best_obj_stop = float("nan")
        if bool(getattr(self.cfg, "fixgurobi_enable_best_obj_stop", False)) and math.isfinite(target):
            slack = float(getattr(self.cfg, "fixgurobi_best_obj_stop_slack", 0.999) or 0.999)
            best_obj_stop = float(target + slack)
        return (
            bool(use_two_stage),
            bool(use_cutoff),
            round(float(cutoff), 6) if math.isfinite(cutoff) else None,
            round(float(best_obj_stop), 6) if math.isfinite(best_obj_stop) else None,
            round(float(getattr(self.cfg, "fixgurobi_time_limit_sec", 1200.0) or 1200.0), 6),
            round(float(getattr(self.cfg, "fixgurobi_mip_gap", 0.01) or 0.01), 8),
            round(float(getattr(self.cfg, "fixgurobi_coarse_time_limit_sec", 8.0) or 8.0), 6),
            round(float(getattr(self.cfg, "fixgurobi_coarse_mip_gap", 0.05) or 0.05), 8),
            bool(getattr(self.cfg, "fixgurobi_accept_first_improvement", True)),
        )

    def _scope_for_layer(self, layer: str) -> str:
        layer_name = str(layer or "").upper()
        if bool(getattr(self.cfg, "resource_revolving_mode", False)) and bool(getattr(self.cfg, "resource_revolving_enable_u_layer", False)):
            if layer_name == "XYZ" and bool(getattr(self.cfg, "fixgurobi_global_outer_on_xyz", False)):
                return "GLOBAL"
            if layer_name == "X":
                return "Y"
            if layer_name == "XZ":
                return "Y"
            if layer_name == "Y":
                return "XZ"
            if layer_name == "YZ":
                yz_scope = str(getattr(self.cfg, "resource_revolving_yz_fix_scope", "") or "").upper()
                if yz_scope in {"X", "LOCALYZ", "XYZ", "XY", "XZ", "YZ"}:
                    return yz_scope
                if yz_scope in {"", "AUTO"}:
                    return self._auto_yz_scope()
                return "LOCALYZ"
            if layer_name == "XY":
                return "Z"
            if layer_name == "Z":
                return "XY"
            if layer_name == "U":
                return "XYZ"
            if layer_name in {"XYZ", "XYZU"}:
                return "LOCALXYZ" if layer_name == "XYZ" else "XYZU"
        if (
            bool(getattr(self.cfg, "sp1_no_split", False))
            and str(getattr(self.cfg, "resource_operator_profile", "") or "").strip().lower() == "no_split_y_focus"
            and layer_name in {"Y", "Z"}
        ):
            return "XY"
        if layer_name == "U":
            return "XYZU"
        if layer_name == "XYZU":
            return "XYZU"
        if bool(getattr(self.cfg, "fixgurobi_force_xyz_scope", False)):
            return "XYZ"
        if layer_name in {"X", "Y", "Z", "XYZ", "XYZU"}:
            return layer_name
        return "XYZ"

    @staticmethod
    def _scope_signature(config: ResourceConfig, scope: str, release_subtask_ids: Optional[Iterable[int]] = None) -> Tuple[Any, ...]:
        scope_name = str(scope or "").upper()
        release_ids = frozenset(int(x) for x in (release_subtask_ids or []) if int(x) >= 0)
        rows_by_order: Dict[int, List[ResourceSubtask]] = defaultdict(list)
        for row in config.subtasks.values():
            rows_by_order[int(row.order_id)].append(row)

        def x_sig(row: ResourceSubtask) -> Tuple[Any, ...]:
            unit_keys = []
            for work_unit_id in row.work_unit_ids or ():
                work_unit = config.work_units.get(str(work_unit_id))
                if work_unit is None:
                    continue
                unit_keys.append((int(work_unit.order_id), int(work_unit.sku_id)))
            return tuple(sorted(unit_keys))

        def z_task_sig(task) -> Tuple[Any, ...]:
            sort_range = getattr(task, "sort_layer_range", None)
            return (
                int(getattr(task, "stack_id", -1)),
                str(getattr(task, "mode", "")).upper(),
                tuple(sorted(int(v) for v in (getattr(task, "target_tote_ids", ()) or ()))),
                tuple(sorted(int(v) for v in (getattr(task, "hit_tote_ids", ()) or ()))),
                tuple(sorted(int(v) for v in (getattr(task, "noise_tote_ids", ()) or ()))),
                None if sort_range is None else (int(sort_range[0]), int(sort_range[1])),
            )

        def z_sig(row: ResourceSubtask) -> Tuple[Any, ...]:
            return tuple(sorted(z_task_sig(task) for task in (row.z_tasks or ())))

        rows = []
        for order_id, order_rows in sorted(rows_by_order.items()):
            order_rows = sorted(
                order_rows,
                key=lambda row: (
                    -int(len(getattr(row, "work_unit_ids", []) or [])),
                    int(row.station_id if row.station_id >= 0 else 10**9),
                    int(row.station_rank if row.station_rank >= 0 else 10**9),
                    int(row.subtask_id),
                )
            )
            slot_parts: List[Any] = []
            for row in order_rows:
                released = int(row.subtask_id) in release_ids
                row_parts: List[Any] = [int(row.subtask_id)]
                if "X" in scope_name and not (released and scope_name in {"LOCALXYZ"}):
                    row_parts.append(("X", x_sig(row)))
                if "Y" in scope_name and not (released and scope_name in {"LOCALXYZ", "LOCALYZ"}):
                    row_parts.append(("Y", int(row.station_id), int(row.station_rank)))
                if "Z" in scope_name and not (released and scope_name in {"LOCALXYZ", "LOCALYZ"}):
                    row_parts.append(("Z", z_sig(row)))
                if len(row_parts) == 1:
                    row_parts.extend(
                        [
                            ("X", x_sig(row)),
                            ("Y", int(row.station_id), int(row.station_rank)),
                            ("Z", z_sig(row)),
                        ]
                    )
                slot_parts.append(tuple(row_parts))
            parts: List[Any] = [int(order_id), ("SLOTS", tuple(slot_parts))]
            rows.append(tuple(parts))
        route_sig = ()
        if scope_name in {"U", "XYZU"}:
            route_rows = dict((getattr(config, "metadata", {}) or {}).get("fixed_route_task_sequence_by_robot", {}) or {})
            route_nodes = dict((getattr(config, "metadata", {}) or {}).get("fixed_route_node_sequence_by_robot", {}) or {})
            route_sig = tuple(
                (
                    int(robot_id),
                    tuple(
                        (
                            int(row.get("order_id", -1)),
                            int(row.get("local_slot_index", -1)),
                            int(row.get("stack_id", -1)),
                            int(row.get("station_id", -1)),
                        )
                        for row in (rows or [])
                    ),
                )
                for robot_id, rows in sorted(route_rows.items(), key=lambda item: int(item[0]))
            ) + tuple(
                (
                    int(robot_id),
                    tuple(
                        (
                            str(row.get("kind", "")),
                            int(row.get("order_id", -1)),
                            int(row.get("local_slot_index", -1)),
                            int(row.get("stack_id", -1)),
                            int(row.get("station_id", -1)),
                        )
                        for row in (rows or [])
                    ),
                )
                for robot_id, rows in sorted(route_nodes.items(), key=lambda item: int(item[0]))
            )
        if scope_name in {"LOCALXYZ", "LOCALYZ"}:
            rows.append(("RELEASE", tuple(sorted(int(x) for x in release_ids))))
        return tuple(rows) + (("U", route_sig),) if route_sig else tuple(rows)

    @staticmethod
    def _fixes_x(scope: str) -> bool:
        return "X" in str(scope or "").upper()

    @staticmethod
    def _fixes_y(scope: str) -> bool:
        return "Y" in str(scope or "").upper()

    @staticmethod
    def _fixes_z(scope: str) -> bool:
        return "Z" in str(scope or "").upper()

    @staticmethod
    def _fixes_u(scope: str) -> bool:
        return "U" in str(scope or "").upper()

    def _include_forced_stacks(self, scope: str) -> bool:
        scope_name = str(scope or "").upper()
        if scope_name in {"LOCALXYZ", "LOCALYZ"}:
            return False
        if bool(getattr(self.cfg, "fixgurobi_force_candidate_stacks", False)):
            return self._fixes_z(scope_name)
        return self._fixes_z(scope_name)

    def _include_fixed_used_stacks(self, scope: str) -> bool:
        return self._fixes_z(scope) and bool(getattr(self.cfg, "fixgurobi_fix_used_stack_ids", False))

    @staticmethod
    def _route_sequence_audit(route_task_sequence: Optional[Dict[Any, Any]], route_node_sequence: Optional[Dict[Any, Any]]) -> Dict[str, Any]:
        task_rows = dict(route_task_sequence or {})
        node_rows = dict(route_node_sequence or {})
        if not task_rows and not node_rows:
            return {"ok": True, "reason": "empty", "robot_count": 0, "checked_task_count": 0}
        if node_rows and not task_rows:
            return {"ok": True, "reason": "node_sequence_only", "robot_count": len(node_rows), "checked_task_count": 0}
        if not task_rows or not node_rows:
            return {"ok": False, "reason": "missing_task_or_node_sequence", "robot_count": len(task_rows or node_rows), "checked_task_count": 0}
        task_robot_ids = {int(robot_id) for robot_id in task_rows.keys()}
        node_robot_ids = {int(robot_id) for robot_id in node_rows.keys()}
        if task_robot_ids != node_robot_ids:
            return {
                "ok": False,
                "reason": "robot_set_mismatch",
                "robot_count": len(task_robot_ids | node_robot_ids),
                "checked_task_count": 0,
            }
        checked = 0
        for robot_id, rows in sorted(task_rows.items(), key=lambda item: int(item[0])):
            nodes = list(node_rows.get(robot_id, node_rows.get(str(robot_id), [])) or [])
            task_list = [dict(row) for row in list(rows or []) if int(row.get("task_id", -1)) >= 0]
            expected_events: List[Tuple[float, int, int]] = []
            for row in task_list:
                task_id = int(row.get("task_id", -1))
                expected_events.append((float(row.get("arrival_stack", 0.0) or 0.0), 0, task_id))
                expected_events.append((float(row.get("arrival_station", 0.0) or 0.0), 1, task_id))
            expected_events.sort(key=lambda item: (float(item[0]), int(item[1]), int(item[2])))
            expected_sequence = [("pickup" if int(kind_rank) == 0 else "delivery", int(task_id)) for _, kind_rank, task_id in expected_events]
            actual_sequence: List[Tuple[str, int]] = []
            pickup_by_task = {}
            delivery_by_task = {}
            for node in nodes:
                task_id = int(node.get("task_id", -1))
                if task_id < 0:
                    continue
                kind = str(node.get("kind", node.get("node_type", "")) or "").lower()
                if kind == "pickup":
                    pickup_by_task[task_id] = node
                    actual_sequence.append(("pickup", int(task_id)))
                elif kind in {"delivery", "station"}:
                    delivery_by_task[task_id] = node
                    actual_sequence.append(("delivery", int(task_id)))
            if actual_sequence != expected_sequence:
                return {
                    "ok": False,
                    "reason": f"node_sequence_order_mismatch:robot={robot_id}",
                    "robot_count": len(task_rows),
                    "checked_task_count": checked,
                }
            if len(pickup_by_task) != len(task_list) or len(delivery_by_task) != len(task_list):
                return {
                    "ok": False,
                    "reason": f"node_sequence_count_mismatch:robot={robot_id}",
                    "robot_count": len(task_rows),
                    "checked_task_count": checked,
                }
            for row in task_list:
                task_id = int(row.get("task_id", -1))
                checked += 1
                pickup = pickup_by_task.get(task_id)
                delivery = delivery_by_task.get(task_id)
                if pickup is None or delivery is None:
                    return {"ok": False, "reason": f"missing_pickup_or_delivery:robot={robot_id}:task={task_id}", "robot_count": len(task_rows), "checked_task_count": checked}
                for key in ("subtask_id", "order_id", "stack_id"):
                    if key in row and key in pickup and int(row.get(key, -1)) != int(pickup.get(key, -1)):
                        return {"ok": False, "reason": f"pickup_{key}_mismatch:robot={robot_id}:task={task_id}", "robot_count": len(task_rows), "checked_task_count": checked}
                if "station_id" in row and "station_id" in delivery and int(row.get("station_id", -1)) != int(delivery.get("station_id", -1)):
                    return {"ok": False, "reason": f"delivery_station_mismatch:robot={robot_id}:task={task_id}", "robot_count": len(task_rows), "checked_task_count": checked}
                if "arrival_stack" in row and "time" in pickup and abs(float(row.get("arrival_stack", 0.0) or 0.0) - float(pickup.get("time", 0.0) or 0.0)) > 1e-6:
                    return {"ok": False, "reason": f"pickup_time_mismatch:robot={robot_id}:task={task_id}", "robot_count": len(task_rows), "checked_task_count": checked}
                if "arrival_station" in row and "time" in delivery and abs(float(row.get("arrival_station", 0.0) or 0.0) - float(delivery.get("time", 0.0) or 0.0)) > 1e-6:
                    return {"ok": False, "reason": f"delivery_time_mismatch:robot={robot_id}:task={task_id}", "robot_count": len(task_rows), "checked_task_count": checked}
        return {"ok": True, "reason": "", "robot_count": len(task_rows), "checked_task_count": int(checked)}

    def _scope_label(self, scope: str) -> str:
        scope_name = str(scope or "").upper()
        if scope_name in {"X", "Y", "Z", "XY", "XZ", "YZ", "XYZ", "LOCALXYZ", "LOCALYZ"}:
            return scope_name
        if scope_name in {"U", "XYZU"}:
            return "XYZU"
        return "XYZ"

    def _result_from_base(self, base_eval: Optional[UpperEvalResult], *, value: float, metadata: Dict[str, Any]) -> UpperEvalResult:
        if base_eval is None:
            base_eval = UpperEvalResult(Sx=0.0, Sy=0.0, Sz=0.0, F_raw=float(value), F_cal=float(value))
        merged_metadata = dict(getattr(base_eval, "metadata", {}) or {})
        merged_metadata.update(metadata)
        return UpperEvalResult(
            Sx=float(getattr(base_eval, "Sx", 0.0)),
            Sy=float(getattr(base_eval, "Sy", 0.0)),
            Sz=float(getattr(base_eval, "Sz", 0.0)),
            F_raw=float(value),
            F_cal=float(value),
            Sy_frozen=float(getattr(base_eval, "Sy_frozen", 0.0)),
            Sy_affected=float(getattr(base_eval, "Sy_affected", 0.0)),
            Sz_frozen=float(getattr(base_eval, "Sz_frozen", 0.0)),
            Sz_affected=float(getattr(base_eval, "Sz_affected", 0.0)),
            fallback_penalty=float(getattr(base_eval, "fallback_penalty", 0.0)),
            feasibility_penalty=float(getattr(base_eval, "feasibility_penalty", 0.0)),
            duplicate_tote_count=int(getattr(base_eval, "duplicate_tote_count", 0)),
            duplicate_tote_penalty=float(getattr(base_eval, "duplicate_tote_penalty", 0.0)),
            coverage_feasible=bool(getattr(base_eval, "coverage_feasible", True)) and math.isfinite(float(value)),
            unmet_sku_total=int(getattr(base_eval, "unmet_sku_total", 0)),
            residual_hat=0.0,
            residual_std=0.0,
            residual_decay_alpha=0.0,
            residual_conf_alpha=0.0,
            uncertainty=0.0,
            subtask_y_contribs=dict(getattr(base_eval, "subtask_y_contribs", {}) or {}),
            subtask_z_contribs=dict(getattr(base_eval, "subtask_z_contribs", {}) or {}),
            affected_subtask_ids=frozenset(getattr(base_eval, "affected_subtask_ids", frozenset()) or frozenset()),
            metadata=merged_metadata,
        )

    @staticmethod
    def _unit_key(config: ResourceConfig, work_unit_id: str) -> Optional[str]:
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            return None
        return f"{int(work_unit.order_id)}:{int(work_unit.sku_id)}"

    def _subtasks_by_order(self, config: ResourceConfig) -> Dict[int, List[ResourceSubtask]]:
        rows_by_order: Dict[int, List[ResourceSubtask]] = defaultdict(list)
        for row in config.subtasks.values():
            rows_by_order[int(row.order_id)].append(row)
        for rows in rows_by_order.values():
            rows.sort(
                key=lambda row: (
                    -int(len(getattr(row, "work_unit_ids", []) or [])),
                    int(row.station_id if row.station_id >= 0 else 10**9),
                    int(row.station_rank if row.station_rank >= 0 else 10**9),
                    int(row.subtask_id),
                )
            )
        return rows_by_order

    def _fixed_payload(self, config: ResourceConfig, scope: str, release_subtask_ids: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        config = config.clone().rebuild_indices()
        release_ids = frozenset(int(x) for x in (release_subtask_ids or []) if int(x) >= 0)
        rows_by_order = self._subtasks_by_order(config)
        fixed_slot_count_by_order: Dict[int, int] = {}
        fixed_work_units_by_order_slot: Dict[int, List[List[str]]] = {}
        fixed_station_rank_by_order_slot: Dict[int, List[Optional[Tuple[int, int]]]] = {}
        fixed_z_descriptors_by_order_slot: Dict[int, List[List[Dict[str, Any]]]] = {}
        forced_candidate_stacks_by_order: Dict[int, List[int]] = {}
        invalid_reasons: List[str] = []

        for order_id, rows in rows_by_order.items():
            fixed_slot_count_by_order[int(order_id)] = int(len(rows))
            unit_rows: List[List[str]] = []
            y_rows: List[Optional[Tuple[int, int]]] = []
            z_rows: List[List[Dict[str, Any]]] = []
            used_stack_ids: Set[int] = set()
            for row in rows:
                scope_name = str(scope or "").upper()
                released_x = int(row.subtask_id) in release_ids and scope_name in {"LOCALXYZ"}
                released_y = int(row.subtask_id) in release_ids and scope_name in {"LOCALXYZ", "LOCALYZ"}
                released_z = int(row.subtask_id) in release_ids and scope_name in {"LOCALXYZ", "LOCALYZ"}
                unit_keys: List[str] = []
                seen_units: Set[str] = set()
                for work_unit_id in row.work_unit_ids or ():
                    key = self._unit_key(config, str(work_unit_id))
                    if key is None or key in seen_units:
                        continue
                    seen_units.add(key)
                    unit_keys.append(key)
                unit_rows.append(None if bool(released_x) else sorted(unit_keys))
                y_rows.append(None if bool(released_y) else (int(row.station_id), int(row.station_rank)))
                descriptors: List[Dict[str, Any]] = []
                for task in row.z_tasks or []:
                    stack_id = int(task.stack_id)
                    if stack_id >= 0:
                        used_stack_ids.add(stack_id)
                    descriptors.append(
                        {
                            "task_id": int(getattr(task, "task_id", -1)),
                            "stack_id": stack_id,
                            "mode": str(task.mode).upper(),
                            "target_tote_ids": [int(v) for v in (task.target_tote_ids or ())],
                            "hit_tote_ids": [int(v) for v in (task.hit_tote_ids or ())],
                            "noise_tote_ids": [int(v) for v in (task.noise_tote_ids or ())],
                            "sort_layer_range": None
                            if task.sort_layer_range is None
                            else [int(task.sort_layer_range[0]), int(task.sort_layer_range[1])],
                        }
                    )
                z_rows.append(None if bool(released_z) else descriptors)
            if self._fixes_x(scope):
                fixed_work_units_by_order_slot[int(order_id)] = unit_rows
            if self._include_forced_stacks(scope):
                forced_candidate_stacks_by_order[int(order_id)] = sorted(used_stack_ids)
            if self._fixes_y(scope):
                fixed_station_rank_by_order_slot[int(order_id)] = y_rows
            if self._fixes_z(scope):
                fixed_z_descriptors_by_order_slot[int(order_id)] = z_rows

        if str(scope or "").upper() == "GLOBAL":
            manifest_slot_counts = dict(
                dict(getattr(self.cfg, "master_domain_manifest", None) or {}).get("slot_count_by_order", {}) or {}
            )
            if manifest_slot_counts:
                fixed_slot_count_by_order = {
                    int(order_id): int(count) for order_id, count in manifest_slot_counts.items()
                }

        fixed_used_stack_ids_by_order = (
            {int(k): list(v) for k, v in forced_candidate_stacks_by_order.items()}
            if self._include_fixed_used_stacks(scope)
            else None
        )
        route_task_sequence = (
            dict((getattr(config, "metadata", {}) or {}).get("fixed_route_task_sequence_by_robot", {}) or {})
            if self._fixes_u(scope)
            else None
        )
        route_node_sequence = (
            dict((getattr(config, "metadata", {}) or {}).get("fixed_route_node_sequence_by_robot", {}) or {})
            if self._fixes_u(scope)
            else None
        )
        if route_node_sequence:
            # Keep route fixing consistent with run_fixgurobi_replay.py: node sequence
            # is the authoritative route seed, task sequence is only a fallback.
            route_task_sequence = None
        seed_stacks = dict((getattr(config, "metadata", {}) or {}).get("structure_seed_stack_ids_by_order", {}) or {})
        if self._include_forced_stacks(scope) and seed_stacks:
            for order_id, stack_ids in seed_stacks.items():
                merged = set(int(x) for x in forced_candidate_stacks_by_order.get(int(order_id), []) if int(x) >= 0)
                merged.update(int(x) for x in (stack_ids or []) if int(x) >= 0)
                if merged:
                    forced_candidate_stacks_by_order[int(order_id)] = sorted(merged)
        route_audit = self._route_sequence_audit(route_task_sequence, route_node_sequence) if self._fixes_u(scope) else {"ok": True}
        if self._fixes_u(scope) and bool(getattr(self.cfg, "fixgurobi_route_sequence_exact_replay_gate", True)) and not bool(route_audit.get("ok", False)):
            invalid_reasons.append(f"route_sequence_exact_replay:{route_audit.get('reason', '')}")
        return {
            "fixed_slot_count_by_order": fixed_slot_count_by_order,
            "fixed_work_units_by_order_slot": fixed_work_units_by_order_slot or None,
            "fixed_station_rank_by_order_slot": fixed_station_rank_by_order_slot or None,
            "fixed_z_descriptors_by_order_slot": fixed_z_descriptors_by_order_slot or None,
            "forced_candidate_stacks_by_order": forced_candidate_stacks_by_order or None,
            "fixed_used_stack_ids_by_order": fixed_used_stack_ids_by_order,
            "fixed_route_task_sequence_by_robot": route_task_sequence,
            "fixed_route_node_sequence_by_robot": route_node_sequence,
            "invalid_reasons": invalid_reasons,
            "route_sequence_audit": route_audit,
        }

    def precompile(self, config: ResourceConfig, *, layer: str) -> Dict[str, Any]:
        scope = self._scope_for_layer(layer)
        normalized = config.clone().rebuild_indices()
        fixed_payload = self._fixed_payload(normalized, scope, ())
        started = time.perf_counter()
        compiled, diagnostics, _base_cfg = self._get_compiled_model(fixed_payload, scope)
        return {
            "ok": compiled is not None,
            "scope": str(scope),
            "runtime_sec": float(time.perf_counter() - started),
            **dict(diagnostics or {}),
        }

    def _build_global_cfg(self, fixed_payload: Dict[str, Any]) -> GlobalXYZUConfig:
        has_fixed_route = bool(
            fixed_payload.get("fixed_route_node_sequence_by_robot")
            or fixed_payload.get("fixed_route_task_sequence_by_robot")
        )
        master_domain = dict(getattr(self.cfg, "master_domain_manifest", None) or {})
        canonical_warm = dict(master_domain.get("canonical_warm_config", {}) or {})
        cfg = GlobalXYZUConfig(
            time_limit_sec=float(getattr(self.cfg, "fixgurobi_time_limit_sec", 20.0) or 20.0),
            mip_gap=float(getattr(self.cfg, "fixgurobi_mip_gap", 0.01) or 0.01),
            candidate_stack_topk=int(getattr(self.cfg, "fixgurobi_candidate_stack_topk", 999) or 999),
            max_candidate_stacks_per_order=int(getattr(self.cfg, "fixgurobi_max_candidate_stacks_per_order", 0) or 0),
            enable_warm_candidate_stack_prune=bool(getattr(self.cfg, "fixgurobi_enable_warm_candidate_stack_prune", False)),
            candidate_station_topk_per_stack=(
                999
                if has_fixed_route
                else int(getattr(self.cfg, "fixgurobi_candidate_station_topk_per_stack", 999) or 999)
            ),
            warm_start_sp4_time_limit_sec=int(canonical_warm.get("warm_start_sp4_time_limit_sec", 0) or 0),
            warm_start_subtask_ordering=str(
                canonical_warm.get(
                    "warm_start_subtask_ordering",
                    getattr(self.cfg, "fixgurobi_warm_start_subtask_ordering", "default"),
                )
                or "default"
            ),
            warm_start_use_sp2_mip_initial=bool(canonical_warm.get("warm_start_use_sp2_mip_initial", False)),
            warm_start_sp2_mip_time_limit_sec=float(
                canonical_warm.get("warm_start_sp2_mip_time_limit_sec", 30.0) or 30.0
            ),
            warm_start_refine_sp2_after_sp4=bool(
                canonical_warm.get("warm_start_refine_sp2_after_sp4", False)
            ),
            route_pickup_neighbor_limit=int(getattr(self.cfg, "fixgurobi_route_pickup_neighbor_limit", 0) or 0),
            sort_hit_tote_threshold=int(
                canonical_warm.get(
                    "sort_hit_tote_threshold",
                    getattr(self.cfg, "fixgurobi_sort_hit_tote_threshold", 3),
                )
                or 3
            ),
            enable_scale_adaptive_candidate_prune=bool(getattr(self.cfg, "fixgurobi_enable_scale_adaptive_candidate_prune", False)),
            gurobi_output=bool(getattr(self.cfg, "fixgurobi_output", False)),
            enable_warm_start=bool(getattr(self.cfg, "enable_warm_start", False) or getattr(self.cfg, "fixgurobi_use_warm_bound", True)),
            warm_start_use_sp4=bool(
                canonical_warm.get(
                    "warm_start_use_sp4",
                    bool(getattr(self.cfg, "enable_warm_start", False) or getattr(self.cfg, "fixgurobi_use_warm_bound", True)),
                )
            ),
            integrate_u_route=True,
            route_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_arc_prune", True)),
            enable_route_time_window_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_time_window_arc_prune", True)),
            enable_route_load_interval_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_load_interval_arc_prune", True)),
            enable_resource_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_slot_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_tote_equivalence_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_station_global_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_robot_finish_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_selected_workload_lbs=True,
            enable_order_time_windows=bool(getattr(self.cfg, "fixgurobi_enable_order_time_windows", True)),
            kitting_span_penalty_weight=float(getattr(self.cfg, "kitting_span_penalty_weight", 5.0) or 5.0),
            deadline_penalty_weight=float(getattr(self.cfg, "deadline_penalty_weight", 1000.0) or 1000.0),
            fixed_slot_count_by_order=fixed_payload.get("fixed_slot_count_by_order"),
            fixed_work_units_by_order_slot=fixed_payload.get("fixed_work_units_by_order_slot"),
            fixed_station_rank_by_order_slot=fixed_payload.get("fixed_station_rank_by_order_slot"),
            fixed_z_descriptors_by_order_slot=fixed_payload.get("fixed_z_descriptors_by_order_slot"),
            fixed_used_stack_ids_by_order=fixed_payload.get("fixed_used_stack_ids_by_order"),
            forced_candidate_stacks_by_order=fixed_payload.get("forced_candidate_stacks_by_order"),
            fixed_route_arcs_by_robot=None,
            fixed_route_task_sequence_by_robot=fixed_payload.get("fixed_route_task_sequence_by_robot"),
            fixed_route_node_sequence_by_robot=fixed_payload.get("fixed_route_node_sequence_by_robot"),
            extra_protected_route_edges=list(getattr(self.cfg, "fixgurobi_extra_protected_route_edges", []) or []),
            fixed_route_arc_fix_nonselected=not bool(getattr(self.cfg, "resource_revolving_mode", False)),
            fixgurobi_relax_sort_tote_fix=bool(getattr(self.cfg, "fixgurobi_relax_sort_tote_fix", False)),
            fixgurobi_no_warm_start=not bool(getattr(self.cfg, "enable_warm_start", False)),
            fixgurobi_allow_warm_start_fallback=bool(getattr(self.cfg, "fixgurobi_allow_warm_start_fallback", False)),
            fixgurobi_warm_bound_only=bool((not bool(getattr(self.cfg, "enable_warm_start", False))) and getattr(self.cfg, "fixgurobi_use_warm_bound", True)),
            master_domain_manifest=getattr(self.cfg, "master_domain_manifest", None),
            master_domain_strict=bool(getattr(self.cfg, "master_domain_strict", False)),
        )
        cfg.route_big_m_time = max(
            float(getattr(cfg, "big_m_time", 2000.0) or 2000.0),
            float(getattr(cfg, "route_big_m_time", 0.0) or 0.0),
        )
        if has_fixed_route:
            cfg.enable_resource_lex_symmetry = False
            cfg.enable_robot_finish_lex_symmetry = False
        remaining_budget = float(self._remaining_wall_budget_sec())
        if math.isfinite(remaining_budget):
            cfg.time_limit_sec = max(0.05, min(float(cfg.time_limit_sec), float(remaining_budget)))
        target = float(getattr(self.cfg, "resource_target_cmax", float("nan")))
        if bool(getattr(self.cfg, "fixgurobi_enable_best_obj_stop", False)) and math.isfinite(target):
            slack = float(getattr(self.cfg, "fixgurobi_best_obj_stop_slack", 0.999) or 0.999)
            cfg.gurobi_best_obj_stop = float(target + slack)
        return cfg

    def _compiled_base_cfg(self, fixed_payload: Dict[str, Any]) -> GlobalXYZUConfig:
        base_payload = dict(fixed_payload or {})
        for key in (
            "fixed_work_units_by_order_slot",
            "fixed_station_rank_by_order_slot",
            "fixed_z_descriptors_by_order_slot",
            "fixed_used_stack_ids_by_order",
        ):
            base_payload[key] = None
        return self._build_global_cfg(base_payload)

    def _compiled_cache_key(self, scope: str, fixed_payload: Dict[str, Any], base_cfg: GlobalXYZUConfig) -> Tuple[Any, ...]:
        problem = getattr(self.opt, "problem", None)
        scale = str(getattr(problem, "scale_name", getattr(self.cfg, "scale", "")) or "")
        seed = int(getattr(self.cfg, "seed", 0) or 0)
        master_domain_sha256 = str(
            dict(getattr(base_cfg, "master_domain_manifest", None) or {}).get("manifest_sha256", "")
        )
        slot_count_key = tuple(
            (int(order_id), int(count))
            for order_id, count in sorted(dict(fixed_payload.get("fixed_slot_count_by_order") or {}).items())
        )
        forced = fixed_payload.get("forced_candidate_stacks_by_order")
        forced_key = tuple(
            (int(order_id), tuple(sorted(int(v) for v in (stack_ids or ()))))
            for order_id, stack_ids in sorted(dict(forced or {}).items())
        )
        route_nodes = dict(fixed_payload.get("fixed_route_node_sequence_by_robot") or {})
        route_tasks = dict(fixed_payload.get("fixed_route_task_sequence_by_robot") or {})
        route_key = tuple(
            (
                int(robot_id),
                tuple(
                    (
                        str(row.get("kind", "")),
                        int(row.get("subtask_id", -1)),
                        int(row.get("stack_id", -1)),
                        int(row.get("station_id", -1)),
                    )
                    for row in (rows or [])
                ),
            )
            for robot_id, rows in sorted(route_nodes.items(), key=lambda item: int(item[0]))
        ) or tuple(
            (
                int(robot_id),
                tuple(
                    (
                        int(row.get("subtask_id", -1)),
                        int(row.get("stack_id", -1)),
                        int(row.get("station_id", -1)),
                    )
                    for row in (rows or [])
                ),
            )
            for robot_id, rows in sorted(route_tasks.items(), key=lambda item: int(item[0]))
        )
        extra_edges_key = tuple(
            (
                tuple(edge.get("src", [])),
                tuple(edge.get("dst", [])),
            )
            for edge in list(getattr(base_cfg, "extra_protected_route_edges", None) or [])
            if isinstance(edge, dict)
        )
        return (
            str(scale).upper(),
            int(seed),
            master_domain_sha256,
            slot_count_key,
            int(getattr(base_cfg, "candidate_stack_topk", 999) or 999),
            int(getattr(base_cfg, "candidate_station_topk_per_stack", 999) or 999),
            int(getattr(base_cfg, "max_candidate_stacks_per_order", 0) or 0),
            int(getattr(base_cfg, "route_pickup_neighbor_limit", 0) or 0),
            bool(getattr(base_cfg, "route_arc_prune", True)),
            bool(getattr(base_cfg, "enable_route_time_window_arc_prune", False)),
            bool(getattr(base_cfg, "enable_route_load_interval_arc_prune", True)),
            bool(getattr(base_cfg, "enable_resource_lex_symmetry", True)),
            forced_key,
            route_key,
            extra_edges_key,
        )

    def _get_compiled_model(self, fixed_payload: Dict[str, Any], scope: str) -> Tuple[Optional[Any], Dict[str, Any], GlobalXYZUConfig]:
        base_cfg = self._compiled_base_cfg(fixed_payload)
        if not bool(getattr(self.cfg, "fixgurobi_enable_compiled_cache", True)):
            return None, {"fixgurobi_compile_cache_hit": False, "fixgurobi_compile_disabled": True}, base_cfg
        key = self._compiled_cache_key(scope, fixed_payload, base_cfg)
        cached = self.compiled_cache.get(key)
        if cached is not None:
            self.compiled_cache.move_to_end(key)
            self.compiled_cache_hit_count += 1
            return cached, {
                "fixgurobi_compile_cache_hit": True,
                "fixgurobi_compile_time": 0.0,
                "fixgurobi_compile_cache_hit_count": int(self.compiled_cache_hit_count),
                "fixgurobi_compile_cache_miss_count": int(self.compiled_cache_miss_count),
            }, base_cfg
        self.compiled_cache_miss_count += 1
        t0 = time.perf_counter()
        compiled = GlobalXYZUSolver().compile_model(copy.deepcopy(self.opt.problem), base_cfg)
        compile_time = float(time.perf_counter() - t0)
        self.compiled_cache[key] = compiled
        self.compiled_cache.move_to_end(key)
        while len(self.compiled_cache) > self._compiled_cache_size():
            self.compiled_cache.popitem(last=False)
        return compiled, {
            "fixgurobi_compile_cache_hit": False,
            "fixgurobi_compile_time": float(compile_time),
            "fixgurobi_compile_cache_hit_count": int(self.compiled_cache_hit_count),
            "fixgurobi_compile_cache_miss_count": int(self.compiled_cache_miss_count),
        }, base_cfg

    def _solve_with_optional_compiled(self, fixed_payload: Dict[str, Any], global_cfg: GlobalXYZUConfig, scope: str, stage: str):
        compiled_meta: Dict[str, Any] = {}
        if bool(getattr(self.cfg, "fixgurobi_enable_compiled_cache", True)):
            try:
                compiled, compiled_meta, _base_cfg = self._get_compiled_model(fixed_payload, scope)
                if compiled is not None:
                    result = GlobalXYZUSolver().solve_compiled(compiled, fixed_cfg=global_cfg, solve_cfg=global_cfg)
                    diag = dict(getattr(result, "diagnostics", {}) or {})
                    diag.update(compiled_meta)
                    diag["fixgurobi_compiled_fallback_used"] = False
                    result.diagnostics = diag
                    return result
            except Exception as exc:
                compiled_meta["fixgurobi_compiled_fallback_used"] = True
                compiled_meta["fixgurobi_compiled_fallback_reason"] = str(exc)
        root = str(getattr(self.opt.problem, "runtime_result_dir", "") or "").strip()
        if (
            root
            and str(stage).lower() in {"full", "refine"}
            and bool(getattr(self.cfg, "fixgurobi_debug_iis", False))
        ):
            try:
                out_dir = os.path.join(tempfile.gettempdir(), "deepnco_fixgurobi_iis")
                os.makedirs(out_dir, exist_ok=True)
                idx = int(getattr(self, "_iis_dump_count", 0) or 0)
                if idx < 5:
                    setattr(global_cfg, "debug_iis_path", os.path.join(out_dir, f"iis_{idx:03d}.ilp"))
                    setattr(self, "_iis_dump_count", idx + 1)
            except Exception:
                pass
        result = GlobalXYZUSolver().solve(copy.deepcopy(self.opt.problem), global_cfg)
        diag = dict(getattr(result, "diagnostics", {}) or {})
        diag.update(compiled_meta)
        diag.setdefault("fixgurobi_compile_cache_hit", False)
        diag.setdefault("fixgurobi_compile_time", 0.0)
        diag["fixgurobi_stage"] = str(stage)
        if str(getattr(result, "status", "")).upper() not in {"OPTIMAL", "TIME_LIMIT"}:
            self._dump_failed_payload(fixed_payload, global_cfg, scope, stage, diag)
        result.diagnostics = diag
        return result

    def _dump_failed_payload(
        self,
        fixed_payload: Dict[str, Any],
        global_cfg: GlobalXYZUConfig,
        scope: str,
        stage: str,
        diagnostics: Dict[str, Any],
    ) -> None:
        try:
            root = str(getattr(self.opt.problem, "runtime_result_dir", "") or "").strip()
            if not root:
                return
            out_dir = os.path.join(root, "fixgurobi_failed_payloads")
            os.makedirs(out_dir, exist_ok=True)
            idx = len([name for name in os.listdir(out_dir) if name.endswith(".json")])
            if idx >= 20:
                return
            diag_keys = (
                "stage",
                "u_fallback_reason",
                "model_status_code",
                "model_best_bound",
                "model_cmax",
                "gurobi_solve_time_sec",
                "u_arc_count",
                "u_active_task_count",
                "route_big_m",
                "route_big_m_source",
                "slot_count",
                "work_unit_count",
                "warm_start_makespan",
                "route_time_window_prune_warm_model_cmax",
                "u_time_window_latest_source",
                "debug_iis_path",
                "debug_iis_error",
            )
            payload = {
                "scope": str(scope),
                "stage": str(stage),
                "diagnostics": {key: diagnostics.get(key) for key in diag_keys if key in diagnostics},
                "cfg": {
                    "candidate_stack_topk": int(getattr(global_cfg, "candidate_stack_topk", 0)),
                    "max_candidate_stacks_per_order": int(getattr(global_cfg, "max_candidate_stacks_per_order", 0)),
                    "candidate_station_topk_per_stack": int(getattr(global_cfg, "candidate_station_topk_per_stack", 0)),
                    "route_pickup_neighbor_limit": int(getattr(global_cfg, "route_pickup_neighbor_limit", 0)),
                    "route_arc_prune": bool(getattr(global_cfg, "route_arc_prune", False)),
                    "enable_route_time_window_arc_prune": bool(getattr(global_cfg, "enable_route_time_window_arc_prune", False)),
                    "enable_route_load_interval_arc_prune": bool(getattr(global_cfg, "enable_route_load_interval_arc_prune", False)),
                    "enable_order_time_windows": bool(getattr(global_cfg, "enable_order_time_windows", False)),
                    "enable_selected_workload_lbs": bool(getattr(global_cfg, "enable_selected_workload_lbs", False)),
                    "fixgurobi_no_warm_start": bool(getattr(global_cfg, "fixgurobi_no_warm_start", False)),
                    "fixgurobi_warm_bound_only": bool(getattr(global_cfg, "fixgurobi_warm_bound_only", False)),
                },
                "fixed_payload": fixed_payload,
            }
            path = os.path.join(out_dir, f"failed_payload_{idx:03d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return

    def _solve_fixgurobi(self, fixed_payload: Dict[str, Any], global_cfg: GlobalXYZUConfig, scope: str):
        best = float(getattr(self, "current_best_value", float("inf")) or float("inf"))
        local_release_scope = str(scope or "").upper() in {"LOCALXYZ", "LOCALYZ"}
        if bool(getattr(self.cfg, "resource_revolving_allow_nonimproving_exact", False)) and not bool(local_release_scope):
            best = float("inf")
        use_two_stage = bool(getattr(self.cfg, "fixgurobi_enable_two_stage", True)) and math.isfinite(best)
        use_cutoff = bool(getattr(self.cfg, "fixgurobi_enable_cutoff", True)) and math.isfinite(best)
        cutoff = float(best - 1e-6) if use_cutoff else None
        if use_two_stage:
            coarse_cfg = copy.copy(global_cfg)
            coarse_cfg.time_limit_sec = float(getattr(self.cfg, "fixgurobi_coarse_time_limit_sec", 8.0) or 8.0)
            coarse_cfg.mip_gap = float(getattr(self.cfg, "fixgurobi_coarse_mip_gap", 0.05) or 0.05)
            coarse_cfg.gurobi_cutoff = cutoff
            coarse = self._solve_with_optional_compiled(fixed_payload, coarse_cfg, scope, "coarse")
            coarse_value = self._objective_from_result(coarse)
            coarse_diag = dict(getattr(coarse, "diagnostics", {}) or {})
            coarse_time = float(coarse_diag.get("gurobi_solve_time_sec", getattr(coarse, "runtime_sec", 0.0)) or 0.0)
            if (
                bool(getattr(self.cfg, "fixgurobi_accept_first_improvement", True))
                and math.isfinite(coarse_value)
                and coarse_value + 1e-9 < best
            ):
                coarse_diag["fixgurobi_stage"] = "coarse_accept"
                coarse_diag["fixgurobi_refined"] = False
                coarse_diag["fixgurobi_cutoff"] = float(cutoff) if cutoff is not None else float("nan")
                coarse_diag["fixgurobi_coarse_time"] = float(coarse_time)
                coarse_diag["fixgurobi_refine_time"] = 0.0
                coarse_diag["fixgurobi_first_improvement_accepted"] = True
                coarse.diagnostics = coarse_diag
                return coarse
            if not (math.isfinite(coarse_value) and coarse_value + 1e-9 < best):
                status_code = int(coarse_diag.get("model_status_code", -1) or -1)
                bound = float(coarse_diag.get("model_best_bound", float("nan")))
                # A short coarse solve may report infeasible under aggressive pruning/cutoff
                # before the full fixed model has been tested.  Only CUTOFF or a valid bound
                # can safely prove that the candidate cannot improve the incumbent.
                proven_no_improve = status_code in {6}
                if cutoff is not None and math.isfinite(bound) and bound >= float(cutoff) - 1e-9:
                    proven_no_improve = True
                if math.isfinite(coarse_value) or proven_no_improve:
                    coarse_diag["fixgurobi_stage"] = "coarse"
                    coarse_diag["fixgurobi_refined"] = False
                    coarse_diag["fixgurobi_cutoff"] = float(cutoff) if cutoff is not None else float("nan")
                    coarse_diag["fixgurobi_coarse_proven_no_improve"] = bool(proven_no_improve)
                    coarse_diag["fixgurobi_coarse_time"] = float(coarse_time)
                    coarse_diag["fixgurobi_refine_time"] = 0.0
                    coarse.diagnostics = coarse_diag
                    return coarse
            refine_cfg = copy.copy(global_cfg)
            refine_cfg.gurobi_cutoff = cutoff
            refined = self._solve_with_optional_compiled(fixed_payload, refine_cfg, scope, "refine")
            refined_diag = dict(getattr(refined, "diagnostics", {}) or {})
            refined_diag["fixgurobi_stage"] = "refine"
            refined_diag["fixgurobi_refined"] = True
            refined_diag["fixgurobi_cutoff"] = float(cutoff) if cutoff is not None else float("nan")
            refined_diag["fixgurobi_coarse_obj"] = float(coarse_value)
            refined_diag["fixgurobi_coarse_status"] = str(getattr(coarse, "status", ""))
            refined_diag["fixgurobi_coarse_gap"] = float(getattr(coarse, "gap", float("nan")))
            refined_diag["fixgurobi_coarse_time"] = float(coarse_time)
            refined_diag["fixgurobi_refine_time"] = float(
                refined_diag.get("gurobi_solve_time_sec", getattr(refined, "runtime_sec", 0.0)) or 0.0
            )
            refined.diagnostics = refined_diag
            return refined
        full_cfg = copy.copy(global_cfg)
        full_cfg.gurobi_cutoff = cutoff
        result = self._solve_with_optional_compiled(fixed_payload, full_cfg, scope, "full")
        diag = dict(getattr(result, "diagnostics", {}) or {})
        diag["fixgurobi_stage"] = "full"
        diag["fixgurobi_refined"] = False
        diag["fixgurobi_cutoff"] = float(cutoff) if cutoff is not None else float("nan")
        diag["fixgurobi_coarse_time"] = 0.0
        diag["fixgurobi_refine_time"] = 0.0
        diag["fixgurobi_full_time"] = float(diag.get("gurobi_solve_time_sec", getattr(result, "runtime_sec", 0.0)) or 0.0)
        result.diagnostics = diag
        return result

    def _full_global_route_edge_audit(self, fixed_payload: Dict[str, Any], scope: str) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "fixgurobi_full_global_route_edge_gate", True)):
            return {"ok": True, "enabled": False, "reason": "disabled"}
        if not self._fixes_u(scope):
            return {"ok": True, "enabled": False, "reason": "scope_without_u"}
        route_tasks = fixed_payload.get("fixed_route_task_sequence_by_robot")
        route_nodes = fixed_payload.get("fixed_route_node_sequence_by_robot")
        if not route_tasks and not route_nodes:
            return {"ok": True, "enabled": True, "reason": "empty_route_sequence"}
        base_cfg = self._compiled_base_cfg(fixed_payload)
        compiled, compile_meta, _base_cfg = self._get_compiled_model(fixed_payload, scope)
        payload = getattr(compiled, "vars_payload", {}) if compiled is not None else {}
        allowed_edges = allowed_route_edges_from_global_payload(dict(payload or {}))
        audit = audit_fixed_route_edges(
            allowed_edges,
            route_task_sequence=route_tasks,
            route_node_sequence=route_nodes,
        )
        audit.update(
            {
                "enabled": True,
                "source": "global_xyzu_compile_model",
                "route_pickup_neighbor_limit": int(getattr(base_cfg, "route_pickup_neighbor_limit", 0) or 0),
                "route_arc_prune": bool(getattr(base_cfg, "route_arc_prune", True)),
                "route_edge_gate_compile_cache_hit": bool(compile_meta.get("fixgurobi_compile_cache_hit", False)),
                "route_edge_gate_compile_time": float(compile_meta.get("fixgurobi_compile_time", 0.0) or 0.0),
            }
        )
        return audit

    @staticmethod
    def _objective_from_result(result) -> float:
        diag = dict(getattr(result, "diagnostics", {}) or {})
        status = str(getattr(result, "status", "") or "").upper()
        span_overrun = float(diag.get("total_span_overrun", diag.get("warm_start_total_span_overrun", 0.0)) or 0.0)
        deadline_overrun = float(diag.get("total_deadline_overrun", diag.get("warm_start_total_deadline_overrun", 0.0)) or 0.0)
        if span_overrun > 1e-9 or deadline_overrun > 1e-9:
            return float("inf")
        if status == "WARM_START_FALLBACK":
            for key in ("warm_start_model_cmax", "model_cmax", "validated_global_makespan", "true_global_makespan"):
                value = diag.get(key, None)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    return float(value)
        for key in ("model_cmax", "validated_global_makespan", "true_global_makespan"):
            value = diag.get(key, None)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        value = float(getattr(result, "objective", float("inf")))
        return value if math.isfinite(value) else float("inf")

    @staticmethod
    def _fixed_payload_diag(config: ResourceConfig, fixed_payload: Dict[str, Any], release_ids: Iterable[int]) -> Dict[str, int]:
        fixed_units_diag = dict(fixed_payload.get("fixed_work_units_by_order_slot") or {})
        fixed_y_diag = dict(fixed_payload.get("fixed_station_rank_by_order_slot") or {})
        fixed_z_diag = dict(fixed_payload.get("fixed_z_descriptors_by_order_slot") or {})
        return {
            "fixgurobi_config_subtask_count": int(len(getattr(config, "subtasks", {}) or {})),
            "fixgurobi_release_subtask_count": int(len(tuple(release_ids or ()))),
            "fixgurobi_payload_fixed_x_order_count": int(len(fixed_units_diag)),
            "fixgurobi_payload_fixed_y_order_count": int(len(fixed_y_diag)),
            "fixgurobi_payload_fixed_z_order_count": int(len(fixed_z_diag)),
            "fixgurobi_payload_fixed_x_row_count": int(sum(len(rows or []) for rows in fixed_units_diag.values())),
            "fixgurobi_payload_fixed_y_row_count": int(sum(len(rows or []) for rows in fixed_y_diag.values())),
            "fixgurobi_payload_fixed_z_row_count": int(sum(len(rows or []) for rows in fixed_z_diag.values())),
        }

    @staticmethod
    def _local_fallback_scope(scope: str) -> str:
        scope_name = str(scope or "").upper()
        if scope_name == "LOCALYZ":
            return "X"
        if scope_name == "LOCALXYZ":
            return "XYZ"
        return ""

    def _auto_yz_scope(self) -> str:
        problem = getattr(self.opt, "problem", None)
        order_num = int(getattr(problem, "order_num", 0) or 0)
        skus_num = int(getattr(problem, "skus_num", 0) or 0)
        sku_per_order = float(skus_num) / float(order_num) if order_num > 0 else float("inf")
        if order_num <= 6:
            return "X"
        if order_num <= 7 and sku_per_order <= 6.5:
            return "X"
        return "LOCALYZ"

    def evaluate(
        self,
        config: ResourceConfig,
        *,
        layer: str,
        base_eval: Optional[UpperEvalResult] = None,
        affected_subtask_ids: Optional[Iterable[int]] = None,
        current_best_value: Optional[float] = None,
        bypass_cache: bool = False,
    ) -> UpperEvalResult:
        if current_best_value is not None:
            try:
                best_value = float(current_best_value)
                if math.isfinite(best_value):
                    self.current_best_value = float(best_value)
            except Exception:
                pass
        scope = self._scope_for_layer(layer)
        normalized_config = config.clone().rebuild_indices()
        release_ids = tuple(sorted(int(x) for x in (affected_subtask_ids or []) if int(x) >= 0))
        cache_key = (
            str(scope),
            self._cache_context_signature(scope),
            self._scope_signature(normalized_config, scope, release_ids),
            release_ids if str(scope).upper() in {"LOCALXYZ", "LOCALYZ"} else (),
        )
        cached = None if bool(bypass_cache) else self._cache_get(cache_key)
        if cached is not None:
            return cached

        self.cache_miss_count += 1
        remaining_budget = float(self._remaining_wall_budget_sec())
        if math.isfinite(remaining_budget) and remaining_budget <= 0.0:
            metadata = {
                "eval_backend": "fixgurobi_prefix",
                "fixgurobi_status": "WALL_TIME_LIMIT",
                "fixgurobi_obj": float("inf"),
                "fixgurobi_bound": float("nan"),
                "fixgurobi_gap": float("nan"),
                "fixgurobi_solve_time": 0.0,
                "fixgurobi_wall_time": 0.0,
                "fixgurobi_fixed_scope": str(scope),
                "fixgurobi_infeasible_reason": "wall_time_limit_exhausted",
                "fixgurobi_cache_hit": False,
            }
            return self._result_from_base(base_eval, value=float("inf"), metadata=metadata)
        if bool(getattr(self.cfg, "fixgurobi_cheap_gate", True)) and not bool(bypass_cache):
            cheap_reasons: List[str] = []
            if base_eval is not None:
                local_release_scope = str(scope or "").upper() in {"LOCALXYZ", "LOCALYZ"}
                if self._fixes_x(scope) and self._fixes_z(scope) and not bool(local_release_scope):
                    if not bool(getattr(base_eval, "coverage_feasible", True)):
                        cheap_reasons.append("coverage_infeasible")
                    if int(getattr(base_eval, "unmet_sku_total", 0) or 0) > 0:
                        cheap_reasons.append("unmet_sku")
                if self._fixes_z(scope) and not bool(local_release_scope) and int(getattr(base_eval, "duplicate_tote_count", 0) or 0) > 0:
                    cheap_reasons.append("duplicate_tote")
            if cheap_reasons:
                metadata = {
                    "eval_backend": "fixgurobi_prefix",
                    "fixgurobi_status": "CHEAP_GATE_REJECT",
                    "fixgurobi_obj": float("inf"),
                    "fixgurobi_bound": float("nan"),
                    "fixgurobi_gap": float("nan"),
                    "fixgurobi_solve_time": 0.0,
                    "fixgurobi_wall_time": 0.0,
                    "fixgurobi_fixed_scope": str(scope),
                    "fixgurobi_infeasible_reason": ",".join(cheap_reasons),
                    "fixgurobi_cache_hit": False,
                    "fixgurobi_cheap_gate_reject": True,
                    "fixgurobi_cheap_gate_reasons": ",".join(cheap_reasons),
                }
                return self._result_from_base(base_eval, value=float("inf"), metadata=metadata)
        # (a) Cheap admissible lower-bound gate: pure pruning of provably non-improving
        # candidates. Never accepts anything; only rejects with F_raw=inf when the cheap
        # station-load lower bound already exceeds the validated-best Cmax.
        if (
            bool(getattr(self.cfg, "fixgurobi_enable_cheap_lb_gate", True))
            and not bool(bypass_cache)
        ):
            lb_best = float(getattr(self, "current_best_value", float("inf")) or float("inf"))
            lb_local_release_scope = str(scope or "").upper() in {"LOCALXYZ", "LOCALYZ"}
            if bool(getattr(self.cfg, "resource_revolving_allow_nonimproving_exact", False)) and not bool(lb_local_release_scope):
                lb_best = float("inf")
            if math.isfinite(lb_best):
                cheap_lb = float(self._cheap_cmax_lower_bound(normalized_config, scope))
                if math.isfinite(cheap_lb) and cheap_lb > lb_best + 1e-9:
                    self.cheap_lb_gate_reject_count += 1
                    metadata = {
                        "eval_backend": "fixgurobi_prefix",
                        "fixgurobi_status": "CHEAP_LB_GATE_REJECT",
                        "fixgurobi_obj": float("inf"),
                        "fixgurobi_bound": float(cheap_lb),
                        "fixgurobi_gap": float("nan"),
                        "fixgurobi_solve_time": 0.0,
                        "fixgurobi_wall_time": 0.0,
                        "fixgurobi_fixed_scope": str(scope),
                        "fixgurobi_infeasible_reason": "cheap_lb_exceeds_best",
                        "fixgurobi_cache_hit": False,
                        "fixgurobi_hard_gate_reject": True,
                        "fixgurobi_cheap_lb_gate_reject": True,
                        "fixgurobi_cheap_lb": float(cheap_lb),
                        "fixgurobi_cheap_lb_best": float(lb_best),
                        "fixgurobi_cheap_lb_gate_reject_count": int(self.cheap_lb_gate_reject_count),
                    }
                    return self._result_from_base(base_eval, value=float("inf"), metadata=metadata)
        # (b) Duplicate route-signature cache: reuse a prior exact evaluation for an
        # identical fixed problem (keyed on the full scope signature plus a cheap route
        # signature and the solve context) without recompiling/resolving.
        route_signature = self._route_signature(normalized_config, scope)
        route_sig_key: Optional[Tuple[Any, ...]] = None
        if (
            route_signature is not None
            and bool(getattr(self.cfg, "fixgurobi_enable_route_signature_cache", True))
            and not bool(bypass_cache)
        ):
            route_sig_key = (cache_key, route_signature)
            cached_route = self._route_signature_cache_get(route_sig_key)
            if cached_route is not None:
                return cached_route
        fixed_payload = self._fixed_payload(normalized_config, scope, release_ids)
        fixed_payload_diag = self._fixed_payload_diag(normalized_config, fixed_payload, release_ids)
        invalid_reasons = [str(x) for x in (fixed_payload.get("invalid_reasons", []) or []) if str(x)]
        route_edge_audit: Dict[str, Any] = {"ok": True, "enabled": False}
        if not invalid_reasons:
            try:
                route_edge_audit = self._full_global_route_edge_audit(fixed_payload, scope)
                if not bool(route_edge_audit.get("ok", True)):
                    invalid_reasons.append(
                        f"full_global_route_edge_missing:{int(route_edge_audit.get('missing_edge_count', 0) or 0)}"
                    )
            except Exception as exc:
                if self._fixes_u(scope) and (
                    fixed_payload.get("fixed_route_node_sequence_by_robot")
                    or fixed_payload.get("fixed_route_task_sequence_by_robot")
                ):
                    route_edge_audit = {"ok": False, "enabled": True, "reason": "audit_exception", "error": str(exc)}
                    invalid_reasons.append(f"full_global_route_edge_audit_exception:{exc}")
        if invalid_reasons:
            metadata = {
                "eval_backend": "fixgurobi_prefix",
                "fixgurobi_status": "HARD_GATE_REJECT",
                "fixgurobi_obj": float("inf"),
                "fixgurobi_bound": float("nan"),
                "fixgurobi_gap": float("nan"),
                "fixgurobi_solve_time": 0.0,
                "fixgurobi_wall_time": 0.0,
                "fixgurobi_fixed_scope": str(scope),
                "fixgurobi_infeasible_reason": ",".join(invalid_reasons),
                "fixgurobi_cache_hit": False,
                "fixgurobi_hard_gate_reject": True,
                "fixgurobi_route_sequence_audit": dict(fixed_payload.get("route_sequence_audit", {}) or {}),
                "fixgurobi_full_global_route_edge_audit": dict(route_edge_audit or {}),
            }
            metadata.update(fixed_payload_diag)
            out = self._result_from_base(base_eval, value=float("inf"), metadata=metadata)
            self._cache_put(cache_key, out)
            return out
        global_cfg = self._build_global_cfg(fixed_payload)
        problem = copy.deepcopy(self.opt.problem)
        t0 = time.perf_counter()
        try:
            result = self._solve_fixgurobi(fixed_payload, global_cfg, scope)
            value = self._objective_from_result(result)
            status = str(getattr(result, "status", "UNKNOWN") or "UNKNOWN")
            diagnostics = dict(getattr(result, "diagnostics", {}) or {})
            materialized_problem = getattr(result, "materialized_problem", None)
            gap = float(getattr(result, "gap", float("nan")))
            bound = float(diagnostics.get("model_best_bound", float("nan")))
            infeasible_reason = "" if math.isfinite(value) else str(diagnostics.get("fallback_reason", status))
            target_cmax = float(getattr(self.cfg, "resource_target_cmax", float("nan")))
            if math.isfinite(float(value)) and math.isfinite(target_cmax) and float(value) < float(target_cmax) - 1e-9:
                diagnostics["fixgurobi_below_target_rejected"] = True
                diagnostics["fixgurobi_rejected_below_target_cmax"] = float(value)
                infeasible_reason = "below_target_cmax_rejected"
                value = float("inf")
        except Exception as exc:
            value = float("inf")
            status = "EXCEPTION"
            gap = float("nan")
            bound = float("nan")
            diagnostics = {"exception": str(exc)}
            materialized_problem = None
            infeasible_reason = str(exc)
        runtime = float(time.perf_counter() - t0)
        metadata = {
            "eval_backend": "fixgurobi_prefix",
            "fixgurobi_status": status,
            "fixgurobi_obj": float(value),
            "fixgurobi_bound": float(bound),
            "fixgurobi_gap": float(gap),
            "fixgurobi_solve_time": float(runtime),
            "fixgurobi_wall_time": float(runtime),
            "fixgurobi_fixed_scope": str(scope),
            "fixgurobi_infeasible_reason": str(infeasible_reason),
            "fixgurobi_cache_hit": False,
            "fixgurobi_cache_hit_count": int(self.cache_hit_count),
            "fixgurobi_cache_miss_count": int(self.cache_miss_count),
            "fixgurobi_diagnostics": diagnostics,
            "fixgurobi_route_sequence_audit": dict(fixed_payload.get("route_sequence_audit", {}) or {}),
            "fixgurobi_full_global_route_edge_audit": dict(route_edge_audit or {}),
        }
        if materialized_problem is not None and math.isfinite(float(value)):
            metadata["fixgurobi_materialized_problem"] = copy.deepcopy(materialized_problem)
        metadata.update(fixed_payload_diag)
        for key in (
            "fixgurobi_compile_cache_hit",
            "fixgurobi_compile_time",
            "fixgurobi_compile_cache_hit_count",
            "fixgurobi_compile_cache_miss_count",
            "fixgurobi_stage",
            "fixgurobi_cutoff",
            "fixgurobi_refined",
            "fixgurobi_coarse_obj",
            "fixgurobi_coarse_status",
            "fixgurobi_coarse_proven_no_improve",
            "fixgurobi_coarse_time",
            "fixgurobi_refine_time",
            "fixgurobi_full_time",
            "fixgurobi_first_improvement_accepted",
            "fixgurobi_compiled_fallback_used",
            "fixgurobi_compiled_fallback_reason",
            "fixgurobi_fixed_constraint_count",
            "fixgurobi_invalid_fix_count",
            "fixgurobi_fixed_route_arc_count_from_cfg",
            "fixgurobi_fixed_route_arc_robot_count",
            "fixgurobi_fixed_route_sequence_robot_count",
            "fixgurobi_fixed_route_sequence_missing_count",
            "fixgurobi_fixed_route_sequence_missing_rows",
            "compiled_model_used",
            "compiled_model_copy_time_sec",
            "gurobi_solve_time_sec",
            "gurobi_runtime_sec",
            "model_objective",
            "objective_value",
            "total_span_overrun",
            "total_deadline_overrun",
            "order_time_windows",
        ):
            if key in diagnostics:
                metadata[key] = diagnostics.get(key)
        for key in (
            "revolving_enabled",
            "released_layer",
            "fixed_layers",
            "inner_relaxed_obj",
            "u_fast_cmax",
            "u_route_lb",
            "u_repair_time",
            "u_changed_robot_count",
            "revolving_lb",
            "lb_gate_skipped",
            "changed_subtask_ids",
            "changed_robot_ids",
        ):
            if key in (getattr(normalized_config, "metadata", {}) or {}):
                metadata[key] = (getattr(normalized_config, "metadata", {}) or {}).get(key)
        fallback_scope = self._local_fallback_scope(scope)
        best_for_fallback = float(getattr(self, "current_best_value", float("inf")) or float("inf"))
        use_local_fallback = (
            bool(fallback_scope)
            and bool(getattr(self.cfg, "resource_local_fixgurobi_enable_fallback_scope", True))
            and (not math.isfinite(float(value)) or (math.isfinite(best_for_fallback) and float(value) + 1e-9 >= best_for_fallback))
        )
        if use_local_fallback:
            local_metadata = dict(metadata)
            fallback_release_ids: Tuple[int, ...] = ()
            fallback_payload = self._fixed_payload(normalized_config, fallback_scope, fallback_release_ids)
            fallback_cfg = self._build_global_cfg(fallback_payload)
            fallback_cache_key = (
                str(fallback_scope),
                self._cache_context_signature(fallback_scope),
                self._scope_signature(normalized_config, fallback_scope, fallback_release_ids),
                (),
            )
            cached_fallback = None if bool(bypass_cache) else self._cache_get(fallback_cache_key)
            if cached_fallback is not None:
                fallback_value = float(cached_fallback.F_raw)
                fallback_runtime = 0.0
                fallback_metadata = dict(getattr(cached_fallback, "metadata", {}) or {})
                fallback_metadata["fixgurobi_solve_time"] = float(runtime)
                fallback_metadata["fixgurobi_wall_time"] = float(runtime)
                fallback_metadata["fixgurobi_local_fallback_cache_hit"] = True
            else:
                fallback_t0 = time.perf_counter()
                try:
                    fallback_result = self._solve_fixgurobi(fallback_payload, fallback_cfg, fallback_scope)
                    fallback_value = self._objective_from_result(fallback_result)
                    fallback_status = str(getattr(fallback_result, "status", "UNKNOWN") or "UNKNOWN")
                    fallback_diagnostics = dict(getattr(fallback_result, "diagnostics", {}) or {})
                    fallback_materialized_problem = getattr(fallback_result, "materialized_problem", None)
                    fallback_gap = float(getattr(fallback_result, "gap", float("nan")))
                    fallback_bound = float(fallback_diagnostics.get("model_best_bound", float("nan")))
                    fallback_infeasible_reason = "" if math.isfinite(fallback_value) else str(
                        fallback_diagnostics.get("fallback_reason", fallback_status)
                    )
                except Exception as exc:
                    fallback_value = float("inf")
                    fallback_status = "EXCEPTION"
                    fallback_gap = float("nan")
                    fallback_bound = float("nan")
                    fallback_diagnostics = {"exception": str(exc)}
                    fallback_materialized_problem = None
                    fallback_infeasible_reason = str(exc)
                fallback_runtime = float(time.perf_counter() - fallback_t0)
                fallback_metadata = {
                    "eval_backend": "fixgurobi_prefix",
                    "fixgurobi_status": fallback_status,
                    "fixgurobi_obj": float(fallback_value),
                    "fixgurobi_bound": float(fallback_bound),
                    "fixgurobi_gap": float(fallback_gap),
                    "fixgurobi_solve_time": float(runtime + fallback_runtime),
                    "fixgurobi_wall_time": float(runtime + fallback_runtime),
                    "fixgurobi_fixed_scope": str(fallback_scope),
                    "fixgurobi_infeasible_reason": str(fallback_infeasible_reason),
                    "fixgurobi_cache_hit": False,
                    "fixgurobi_cache_hit_count": int(self.cache_hit_count),
                    "fixgurobi_cache_miss_count": int(self.cache_miss_count),
                    "fixgurobi_diagnostics": fallback_diagnostics,
                    "fixgurobi_local_fallback_cache_hit": False,
                }
                if fallback_materialized_problem is not None and math.isfinite(float(fallback_value)):
                    fallback_metadata["fixgurobi_materialized_problem"] = copy.deepcopy(fallback_materialized_problem)
                fallback_metadata.update(self._fixed_payload_diag(normalized_config, fallback_payload, fallback_release_ids))
                for key in (
                    "fixgurobi_compile_cache_hit",
                    "fixgurobi_compile_time",
                    "fixgurobi_compile_cache_hit_count",
                    "fixgurobi_compile_cache_miss_count",
                    "fixgurobi_stage",
                    "fixgurobi_cutoff",
                    "fixgurobi_refined",
                    "fixgurobi_coarse_obj",
                    "fixgurobi_coarse_status",
                    "fixgurobi_coarse_proven_no_improve",
                    "fixgurobi_coarse_time",
                    "fixgurobi_refine_time",
                    "fixgurobi_full_time",
                    "fixgurobi_first_improvement_accepted",
                    "fixgurobi_compiled_fallback_used",
                    "fixgurobi_compiled_fallback_reason",
                    "fixgurobi_fixed_constraint_count",
                    "fixgurobi_invalid_fix_count",
                    "fixgurobi_fixed_route_arc_count_from_cfg",
                    "fixgurobi_fixed_route_arc_robot_count",
                    "fixgurobi_fixed_route_sequence_robot_count",
                    "fixgurobi_fixed_route_sequence_missing_count",
                    "fixgurobi_fixed_route_sequence_missing_rows",
                    "compiled_model_used",
                    "compiled_model_copy_time_sec",
                    "gurobi_solve_time_sec",
                    "gurobi_runtime_sec",
            "model_objective",
            "objective_value",
            "total_span_overrun",
            "total_deadline_overrun",
            "order_time_windows",
                ):
                    if key in fallback_diagnostics:
                        fallback_metadata[key] = fallback_diagnostics.get(key)
                if not bool(bypass_cache):
                    fallback_out = self._result_from_base(base_eval, value=float(fallback_value), metadata=fallback_metadata)
                    self._cache_put(fallback_cache_key, fallback_out)
            for key in (
                "revolving_enabled",
                "released_layer",
                "fixed_layers",
                "inner_relaxed_obj",
                "u_fast_cmax",
                "u_route_lb",
                "u_repair_time",
                "u_changed_robot_count",
                "revolving_lb",
                "lb_gate_skipped",
                "changed_subtask_ids",
                "changed_robot_ids",
            ):
                if key in (getattr(normalized_config, "metadata", {}) or {}):
                    fallback_metadata[key] = (getattr(normalized_config, "metadata", {}) or {}).get(key)
            fallback_metadata["fixgurobi_local_attempt_metadata"] = local_metadata
            fallback_metadata["fixgurobi_local_fallback_used"] = True
            fallback_metadata["fixgurobi_local_attempt_scope"] = str(scope)
            fallback_metadata["fixgurobi_local_attempt_obj"] = float(value)
            fallback_metadata["fixgurobi_local_attempt_status"] = str(status)
            fallback_metadata["fixgurobi_local_attempt_solve_time"] = float(runtime)
            fallback_metadata["fixgurobi_local_fallback_scope"] = str(fallback_scope)
            fallback_metadata["fixgurobi_local_fallback_solve_time"] = float(fallback_runtime)
            value = float(fallback_value)
            metadata = fallback_metadata
        metadata["fixgurobi_cheap_lb_gate_reject_count"] = int(self.cheap_lb_gate_reject_count)
        metadata["fixgurobi_route_signature_cache_hit_count"] = int(self.route_signature_cache_hit_count)
        out = self._result_from_base(base_eval, value=float(value), metadata=metadata)
        out.affected_subtask_ids = frozenset(int(x) for x in (affected_subtask_ids or getattr(out, "affected_subtask_ids", frozenset()) or []))
        if not bool(bypass_cache):
            self._cache_put(cache_key, out)
            if route_sig_key is not None:
                self._route_signature_cache_put(route_sig_key, out)
        return out
