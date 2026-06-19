from __future__ import annotations

import copy
import math
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver

from .state import ResourceConfig, ResourceSubtask, UpperEvalResult


class FixGurobiEvaluator:
    """Evaluate TRA resource configs by fixing layer-specific decisions in GlobalXYZU."""

    def __init__(self, opt, surrogate_scorer=None) -> None:
        self.opt = opt
        self.cfg = opt.cfg
        self.surrogate_scorer = surrogate_scorer
        self.cache: OrderedDict[Tuple[Any, ...], UpperEvalResult] = OrderedDict()
        self.compiled_cache: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.compiled_cache_hit_count = 0
        self.compiled_cache_miss_count = 0
        self.current_best_value = float("inf")

    def _cache_size(self) -> int:
        return max(1, int(getattr(self.cfg, "fixgurobi_cache_size", 128) or 128))

    def _compiled_cache_size(self) -> int:
        return max(1, int(getattr(self.cfg, "fixgurobi_compiled_cache_size", 8) or 8))

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
        if bool(getattr(self.cfg, "fixgurobi_force_xyz_scope", False)):
            return "XYZ"
        if layer_name == "U":
            return "XYZU"
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

        def y_sig(order_rows: List[ResourceSubtask]) -> Tuple[Any, ...]:
            by_station: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for local_idx, row in enumerate(order_rows):
                by_station[int(row.station_id)].append((int(row.station_rank), int(local_idx)))
            out = []
            for station_id, rank_rows in sorted(by_station.items()):
                ordered = [local_idx for _rank, local_idx in sorted(rank_rows)]
                out.append((int(station_id), tuple(int(v) for v in ordered)))
            return tuple(out)

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
            order_rows = sorted(order_rows, key=lambda row: (int(row.station_rank), int(row.subtask_id)))
            parts: List[Any] = [int(order_id)]
            fixed_rows = [row for row in order_rows if int(row.subtask_id) not in release_ids]
            x_rows = fixed_rows if scope_name in {"LOCALXYZ"} else order_rows
            y_rows = fixed_rows if scope_name in {"LOCALXYZ", "LOCALYZ"} else order_rows
            z_rows = fixed_rows if scope_name in {"LOCALXYZ", "LOCALYZ"} else order_rows
            if "X" in scope_name:
                parts.append(("X", tuple(sorted(x_sig(row) for row in x_rows))))
            if "Y" in scope_name:
                parts.append(("Y", y_sig(y_rows)))
            if "Z" in scope_name:
                parts.append(("Z", tuple(sorted(z_sig(row) for row in z_rows))))
            if len(parts) == 1:
                parts.extend(
                    [
                        ("X", tuple(sorted(x_sig(row) for row in order_rows))),
                        ("Y", y_sig(order_rows)),
                        ("Z", tuple(sorted(z_sig(row) for row in order_rows))),
                    ]
                )
            rows.append(tuple(parts))
        route_sig = ()
        if scope_name in {"U", "XYZU"}:
            route_rows = dict((getattr(config, "metadata", {}) or {}).get("fixed_route_task_sequence_by_robot", {}) or {})
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
            rows.sort(key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)))
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

        fixed_used_stack_ids_by_order = (
            {int(k): list(v) for k, v in forced_candidate_stacks_by_order.items()}
            if self._include_fixed_used_stacks(scope)
            else None
        )
        return {
            "fixed_slot_count_by_order": fixed_slot_count_by_order,
            "fixed_work_units_by_order_slot": fixed_work_units_by_order_slot or None,
            "fixed_station_rank_by_order_slot": fixed_station_rank_by_order_slot or None,
            "fixed_z_descriptors_by_order_slot": fixed_z_descriptors_by_order_slot or None,
            "forced_candidate_stacks_by_order": forced_candidate_stacks_by_order or None,
            "fixed_used_stack_ids_by_order": fixed_used_stack_ids_by_order,
            "fixed_route_task_sequence_by_robot": (
                dict((getattr(config, "metadata", {}) or {}).get("fixed_route_task_sequence_by_robot", {}) or {})
                if self._fixes_u(scope)
                else None
            ),
            "invalid_reasons": invalid_reasons,
        }

    def _build_global_cfg(self, fixed_payload: Dict[str, Any]) -> GlobalXYZUConfig:
        cfg = GlobalXYZUConfig(
            time_limit_sec=float(getattr(self.cfg, "fixgurobi_time_limit_sec", 20.0) or 20.0),
            mip_gap=float(getattr(self.cfg, "fixgurobi_mip_gap", 0.01) or 0.01),
            candidate_stack_topk=int(getattr(self.cfg, "fixgurobi_candidate_stack_topk", 999) or 999),
            max_candidate_stacks_per_order=int(getattr(self.cfg, "fixgurobi_max_candidate_stacks_per_order", 0) or 0),
            enable_warm_candidate_stack_prune=bool(getattr(self.cfg, "fixgurobi_enable_warm_candidate_stack_prune", False)),
            candidate_station_topk_per_stack=int(getattr(self.cfg, "fixgurobi_candidate_station_topk_per_stack", 999) or 999),
            warm_start_subtask_ordering=str(getattr(self.cfg, "fixgurobi_warm_start_subtask_ordering", "default") or "default"),
            route_pickup_neighbor_limit=int(getattr(self.cfg, "fixgurobi_route_pickup_neighbor_limit", 0) or 0),
            enable_scale_adaptive_candidate_prune=bool(getattr(self.cfg, "fixgurobi_enable_scale_adaptive_candidate_prune", False)),
            gurobi_output=bool(getattr(self.cfg, "fixgurobi_output", False)),
            enable_warm_start=bool(getattr(self.cfg, "fixgurobi_use_warm_bound", True)),
            warm_start_use_sp4=bool(getattr(self.cfg, "fixgurobi_use_warm_bound", True)),
            integrate_u_route=True,
            route_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_arc_prune", True)),
            enable_route_time_window_arc_prune=False,
            enable_route_load_interval_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_load_interval_arc_prune", True)),
            enable_resource_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_slot_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_tote_equivalence_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_station_global_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_robot_finish_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
            enable_selected_workload_lbs=True,
            fixed_slot_count_by_order=fixed_payload.get("fixed_slot_count_by_order"),
            fixed_work_units_by_order_slot=fixed_payload.get("fixed_work_units_by_order_slot"),
            fixed_station_rank_by_order_slot=fixed_payload.get("fixed_station_rank_by_order_slot"),
            fixed_z_descriptors_by_order_slot=fixed_payload.get("fixed_z_descriptors_by_order_slot"),
            fixed_used_stack_ids_by_order=fixed_payload.get("fixed_used_stack_ids_by_order"),
            forced_candidate_stacks_by_order=fixed_payload.get("forced_candidate_stacks_by_order"),
            fixed_route_arcs_by_robot=None,
            fixed_route_task_sequence_by_robot=fixed_payload.get("fixed_route_task_sequence_by_robot"),
            fixed_route_arc_fix_nonselected=not bool(getattr(self.cfg, "resource_revolving_mode", False)),
            fixgurobi_no_warm_start=True,
            fixgurobi_allow_warm_start_fallback=bool(getattr(self.cfg, "fixgurobi_allow_warm_start_fallback", False)),
            fixgurobi_warm_bound_only=bool(getattr(self.cfg, "fixgurobi_use_warm_bound", True)),
        )
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
            "fixed_route_task_sequence_by_robot",
        ):
            base_payload[key] = None
        return self._build_global_cfg(base_payload)

    def _compiled_cache_key(self, scope: str, fixed_payload: Dict[str, Any], base_cfg: GlobalXYZUConfig) -> Tuple[Any, ...]:
        problem = getattr(self.opt, "problem", None)
        scale = str(getattr(problem, "scale_name", getattr(self.cfg, "scale", "")) or "")
        seed = int(getattr(self.cfg, "seed", 0) or 0)
        slot_count_key = tuple(
            (int(order_id), int(count))
            for order_id, count in sorted(dict(fixed_payload.get("fixed_slot_count_by_order") or {}).items())
        )
        forced = fixed_payload.get("forced_candidate_stacks_by_order")
        forced_key = tuple(
            (int(order_id), tuple(sorted(int(v) for v in (stack_ids or ()))))
            for order_id, stack_ids in sorted(dict(forced or {}).items())
        )
        return (
            str(scale).upper(),
            int(seed),
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
        result = GlobalXYZUSolver().solve(copy.deepcopy(self.opt.problem), global_cfg)
        diag = dict(getattr(result, "diagnostics", {}) or {})
        diag.update(compiled_meta)
        diag.setdefault("fixgurobi_compile_cache_hit", False)
        diag.setdefault("fixgurobi_compile_time", 0.0)
        diag["fixgurobi_stage"] = str(stage)
        result.diagnostics = diag
        return result

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
                proven_no_improve = status_code in {3, 4, 6}
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

    @staticmethod
    def _objective_from_result(result) -> float:
        diag = dict(getattr(result, "diagnostics", {}) or {})
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
        fixed_payload = self._fixed_payload(normalized_config, scope, release_ids)
        fixed_payload_diag = self._fixed_payload_diag(normalized_config, fixed_payload, release_ids)
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
        out = self._result_from_base(base_eval, value=float(value), metadata=metadata)
        out.affected_subtask_ids = frozenset(int(x) for x in (affected_subtask_ids or getattr(out, "affected_subtask_ids", frozenset()) or []))
        if not bool(bypass_cache):
            self._cache_put(cache_key, out)
        return out
