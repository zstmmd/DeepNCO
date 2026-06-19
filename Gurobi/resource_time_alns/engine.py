from __future__ import annotations

import copy
from collections import deque
import itertools
import math
import random
import statistics
import time
from typing import Dict, List, Optional, Tuple

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from problemDto.createInstance import CreateOFSProblem

from .initializer import build_initial_resource_config, build_resource_config_from_problem
from .operators_x import (
    X_DESTROY_OPERATORS,
    X_FALLBACK_OPERATOR,
    X_REPAIR_OPERATORS,
    x_repair_greedy_fallback,
)
from .operators_y import (
    Y_DESTROY_OPERATORS,
    Y_FALLBACK_OPERATOR,
    Y_REPAIR_OPERATORS,
    apply_exact_y_plan,
    plan_y_candidate,
)
from .operators_z import (
    Z_DESTROY_OPERATORS,
    Z_FALLBACK_OPERATOR,
    Z_REPAIR_OPERATORS,
    apply_joint_colocated_sort_postprocess,
    apply_single_flip_sortify_polish,
    apply_exact_z_plan,
    build_full_z_assignment,
    plan_z_candidate,
)
from .projection import apply_projection_repair
from .reporting import build_iter_row
from .state import OperatorArm, ResourceConfig, UpperEvalResult, ValidatedIncumbent
from .surrogate import ResourceSurrogateScorer, config_distance
from .validator import ResourceValidator
from .fixgurobi_evaluator import FixGurobiEvaluator
from .revolving_solver import RevolvingSolver


class ResourceTimeALNSEngine:
    def __init__(self, opt):
        self.opt = opt
        self.cfg = opt.cfg
        self.rng = random.Random(int(getattr(self.cfg, "seed", 42)) + 7919)
        self.validator = ResourceValidator(opt)
        self.scorer = ResourceSurrogateScorer(opt)
        self.revolving_solver = RevolvingSolver(opt)
        self.eval_backend = str(getattr(self.cfg, "resource_eval_backend", "surrogate") or "surrogate").strip().lower()
        self.fixgurobi_evaluator = FixGurobiEvaluator(opt, surrogate_scorer=self.scorer) if self.eval_backend == "fixgurobi_prefix" else None
        self.fixgurobi_only_eval = bool(
            self.eval_backend == "fixgurobi_prefix"
            and bool(getattr(self.cfg, "resource_fixgurobi_skip_ortools_validation", False))
        )
        self.current_config: ResourceConfig = build_initial_resource_config(opt)
        self.initial_config: ResourceConfig = self.current_config.clone()
        skip_initial_fixgurobi = bool(
            self.eval_backend == "fixgurobi_prefix"
            and bool(getattr(self.cfg, "resource_skip_initial_fixgurobi_eval", False))
        )
        self.current_eval = (
            self.scorer.evaluate(self.current_config)
            if bool(skip_initial_fixgurobi)
            else self._evaluate_config(self.current_config, layer="XYZ")
        )
        if self.fixgurobi_only_eval:
            initial_hard_reject_reason = ""
            initial_makespan = float("inf") if bool(skip_initial_fixgurobi) else float(self.current_eval.F_raw)
            self.best_validated = ValidatedIncumbent(
                config=self.current_config.clone(),
                makespan=float(initial_makespan),
                iter_id=0,
                snapshot=None,
            )
            initial_validated_makespans = [] if bool(skip_initial_fixgurobi) else [float(initial_makespan)]
            if not bool(skip_initial_fixgurobi):
                self._sync_fixgurobi_best_snapshot(float(initial_makespan), 0)
        else:
            initial_validation = self.validator.validate(self.current_config, 0)
            initial_hard_reject_reason = str(initial_validation.get("hard_reject_reason", "") or "")
            initial_validated_makespans: List[float] = []
            if not initial_hard_reject_reason:
                initial_makespan = float(initial_validation.get("makespan", float(getattr(opt.best, "z", float("inf")))))
                initial_snapshot = initial_validation.get("snapshot", getattr(opt, "best", None))
                self.best_validated = ValidatedIncumbent(
                    config=self.current_config.clone(),
                    makespan=float(initial_makespan),
                    iter_id=0,
                    snapshot=initial_snapshot,
                )
                initial_validated_makespans = [float(initial_makespan)]
                if initial_snapshot is not None:
                    self.opt.best = initial_snapshot
                    self.opt.work = initial_snapshot
                    self.opt.work_z = float(initial_makespan)
            else:
                self.best_validated = ValidatedIncumbent(
                    config=self.current_config.clone(),
                    makespan=float("inf"),
                    iter_id=0,
                    snapshot=None,
                )
        self.current_eval.metadata.update(
            {
                "last_validation_iter": 0,
                "last_validation_f_raw": float(self.current_eval.F_raw),
                "recent_validated_makespans": list(initial_validated_makespans),
                "initial_hard_reject_reason": str(initial_hard_reject_reason),
            }
        )
        self.last_validated_config = self.current_config.clone()
        self.last_validated_signature = self.current_config.validation_signature()
        self.last_validation_iter = 0
        self.last_validation_f_raw = float(self.current_eval.F_raw)
        self.recent_validated_makespans: List[float] = list(initial_validated_makespans)
        self.temperature = float(getattr(self.cfg, "resource_sa_init_temp", max(1.0, 0.05 * float(self.current_eval.F_raw))))
        self.resource_layers = ["X", "Y", "Z"]
        if bool(getattr(self.cfg, "resource_revolving_mode", False)):
            self.resource_layers.append("XZ")
            self.resource_layers.append("YZ")
        if bool(getattr(self.cfg, "resource_revolving_enable_u_layer", False)):
            self.resource_layers.append("U")
        if bool(getattr(self.cfg, "resource_enable_xyz_operator", False)):
            self.resource_layers.append("XYZ")
        self.layer_ema_improve = {layer: 1.0 for layer in self.resource_layers}
        self.layer_stagnation = {layer: 0.0 for layer in self.resource_layers}
        self.layer_exec_since_update = {layer: 0 for layer in self.resource_layers}
        self.layer_last_update_iter = {layer: 0 for layer in self.resource_layers}
        self.layer_cooldown_until_iter = {layer: 0 for layer in self.resource_layers}
        self.layer_failure_cooldown_until_iter = {layer: 0 for layer in self.resource_layers}
        self.layer_dynamic_multiplier = {layer: 1.0 for layer in self.resource_layers}
        self.consecutive_fail_count = {layer: 0 for layer in self.resource_layers}
        self.z_exact_fail_streak = 0
        self.z_operator_pick_count = 0
        self.forced_layer_queue: deque[str] = deque()
        self.global_decomp_repair_used = False
        if bool(getattr(self.cfg, "resource_global_decomp_repair_enabled", False)) and "XYZ" in self.resource_layers:
            self.forced_layer_queue.append("XYZ")
        self.last_selected_layer = ""
        self.last_available_layers: List[str] = list(self.resource_layers)
        self.last_selected_layer_source = ""
        self.last_xyz_skip_reason = ""
        self.last_x_repartition_iter = -10**9
        self.last_critical_path_subtask_ids: List[int] = []
        self.no_improve_rounds = 0.0
        self.no_best_z_change_rounds = 0.0
        self.validated_best_no_change_rounds = 0
        self.best_f_raw = float(self.current_eval.F_raw)
        lahc_len = max(1, int(getattr(self.cfg, "resource_lahc_history_length", 20)))
        self.lahc_history: List[float] = [float(self.current_eval.F_cal)] * lahc_len
        self.lahc_index = 0
        self.lahc_threshold = float(self.current_eval.F_cal)
        self.multi_start_restart_count = 0
        self.consecutive_exact_cache_hit_count = 0
        self.adaptive_destroy_bonus = 0.0
        self.coverage_hard_reject_count = 0
        self.x_failure_decapitation_count = 0
        self.lkh_call_count = 0
        self.lkh_budget_consumed_by_rollback = 0
        self.operator_arms = self._init_operator_arms()
        history_size = max(1, int(getattr(self.cfg, "resource_action_signature_history_size", 30)))
        self.action_signature_history = {layer: deque(maxlen=history_size) for layer in self.resource_layers}
        self.action_signature_seen = {layer: set() for layer in self.resource_layers}
        self.opt.candidate_iter_log = []
        self.opt.stop_reason = ""
        self.joint_colocated_sort_postprocess_stats = {
            "triggered": 0.0,
            "candidate_groups": 0.0,
            "submitted": 0.0,
            "applied": 0.0,
            "makespan_improvement": 0.0,
            "rejected_capacity": 0.0,
            "rejected_interval_illegal": 0.0,
            "rejected_noise": 0.0,
            "rejected_eval_not_better": 0.0,
            "rejected_validation": 0.0,
            "rejected_target_conflict": 0.0,
        }
        self.opt.joint_colocated_sort_postprocess_stats = self.joint_colocated_sort_postprocess_stats
        self._refresh_operator_stats_payload()

    def _evaluate_config(
        self,
        config: ResourceConfig,
        *,
        layer: str,
        score_cache=None,
        affected_subtask_ids=None,
        fallback_penalty: float = 0.0,
        iterations_since_last_validation: int = 0,
        distance_to_last_validated: float = 0.0,
        bypass_cache: bool = False,
    ) -> UpperEvalResult:
        eval_config = config
        layer_name = str(layer).upper()
        if (
            bool(getattr(self.cfg, "resource_revolving_mode", False))
            and bool(getattr(self.cfg, "resource_revolving_enable_u_layer", False))
            and layer_name in {"X", "Y", "Z", "XZ", "YZ", "XY"}
            and not (getattr(config, "metadata", {}) or {}).get("fixed_route_task_sequence_by_robot")
        ):
            current_best = float("inf")
            if hasattr(self, "best_validated"):
                try:
                    current_best = float(self.best_validated.makespan)
                except Exception:
                    current_best = float("inf")
            revolving = self.revolving_solver.attach_u_plan(
                config,
                released_layer=layer_name,
                affected_subtask_ids=affected_subtask_ids or [],
                incumbent_value=float(current_best),
            )
            eval_config = revolving.config
        base_eval = self.scorer.evaluate(
            config=eval_config,
            score_cache=score_cache,
            affected_subtask_ids=affected_subtask_ids,
            fallback_penalty=float(fallback_penalty),
            iterations_since_last_validation=int(iterations_since_last_validation),
            distance_to_last_validated=float(distance_to_last_validated),
        )
        if self.fixgurobi_evaluator is None:
            return base_eval
        current_best = float("inf")
        if hasattr(self, "best_validated"):
            try:
                current_best = float(self.best_validated.makespan)
            except Exception:
                current_best = float("inf")
        result = self.fixgurobi_evaluator.evaluate(
            eval_config,
            layer=str(layer),
            base_eval=base_eval,
            affected_subtask_ids=affected_subtask_ids,
            current_best_value=current_best,
            bypass_cache=bool(bypass_cache),
        )
        result.metadata.update(dict(getattr(eval_config, "metadata", {}) or {}))
        return result

    def _sync_fixgurobi_best_snapshot(self, value: float, iter_id: int) -> None:
        """Keep TRA summary fields aligned when FixGurobi is the only evaluator."""
        self.opt.work_z = float(value)
        for attr in ("best", "work"):
            snapshot = getattr(self.opt, attr, None)
            if snapshot is None:
                continue
            try:
                snapshot.z = float(value)
            except Exception:
                pass
            try:
                snapshot.iter_id = int(iter_id)
            except Exception:
                pass

    def _snapshot_tail_metrics(self, snapshot, config: Optional[ResourceConfig] = None) -> Dict[str, object]:
        rows = list(getattr(snapshot, "subtask_state", []) or [])
        active_robots = set()
        latest_robot_finish = 0.0
        order_completion: Dict[int, float] = {}
        for subtask in rows:
            subtask_id = int(getattr(subtask, "id", getattr(subtask, "subtask_id", -1)))
            order_id = -1
            if config is not None and int(subtask_id) in config.subtasks:
                order_id = int(config.subtasks[int(subtask_id)].order_id)
            parent_order = getattr(subtask, "parent_order", None)
            if order_id < 0 and parent_order is not None:
                order_id = int(getattr(parent_order, "order_id", getattr(parent_order, "id", -1)))
            task_rows = list(getattr(subtask, "execution_tasks", []) or [])
            max_end = 0.0
            for task in task_rows:
                robot_id = int(getattr(task, "robot_id", -1))
                if robot_id >= 0:
                    active_robots.add(int(robot_id))
                latest_robot_finish = max(
                    float(latest_robot_finish),
                    float(getattr(task, "arrival_time_at_station", 0.0) or 0.0),
                    float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0),
                )
                max_end = max(float(max_end), float(getattr(task, "end_process_time", 0.0) or 0.0))
            if order_id >= 0:
                order_completion[int(order_id)] = max(float(order_completion.get(int(order_id), 0.0)), float(max_end))
        return {
            "latest_robot_finish": float(latest_robot_finish),
            "max_order_completion": float(max(order_completion.values(), default=0.0)),
            "active_robot_count": int(len(active_robots)),
            "active_robot_ids": sorted(active_robots),
        }

    def _tail_guard_reason(self, candidate_validation: Dict[str, object], candidate_config: ResourceConfig) -> Tuple[str, Dict[str, object]]:
        if not bool(getattr(self.cfg, "resource_tail_guard_enabled", True)):
            return "", {}
        candidate_makespan = float(candidate_validation.get("makespan", float("inf")) or float("inf"))
        improves_makespan = bool(candidate_makespan + 1e-9 < float(getattr(self.best_validated, "makespan", float("inf"))))
        incumbent_metrics = self._snapshot_tail_metrics(getattr(self.best_validated, "snapshot", None), self.best_validated.config)
        candidate_metrics = self._snapshot_tail_metrics(candidate_validation.get("snapshot", None), candidate_config)
        ratio = float(getattr(self.cfg, "resource_tail_guard_ratio", 1.05))
        reason = ""
        incumbent_latest = float(incumbent_metrics.get("latest_robot_finish", 0.0) or 0.0)
        candidate_latest = float(candidate_metrics.get("latest_robot_finish", 0.0) or 0.0)
        incumbent_order = float(incumbent_metrics.get("max_order_completion", 0.0) or 0.0)
        candidate_order = float(candidate_metrics.get("max_order_completion", 0.0) or 0.0)
        if (not improves_makespan) and incumbent_latest > 1e-9 and candidate_latest > incumbent_latest * ratio + 1e-9:
            reason = "latest_robot_finish_regression"
        elif (not improves_makespan) and incumbent_order > 1e-9 and candidate_order > incumbent_order * ratio + 1e-9:
            reason = "max_order_completion_regression"
        else:
            incumbent_active = set(int(x) for x in (incumbent_metrics.get("active_robot_ids", []) or []))
            candidate_active = set(int(x) for x in (candidate_metrics.get("active_robot_ids", []) or []))
            if len(candidate_active) < len(incumbent_active) and bool(incumbent_active - candidate_active):
                reason = "active_robot_count_regression"
        meta = {
            "candidate_latest_robot_finish": float(candidate_latest),
            "candidate_max_order_completion": float(candidate_order),
            "candidate_active_robot_count": int(candidate_metrics.get("active_robot_count", 0) or 0),
            "incumbent_latest_robot_finish": float(incumbent_latest),
            "incumbent_max_order_completion": float(incumbent_order),
            "incumbent_active_robot_count": int(incumbent_metrics.get("active_robot_count", 0) or 0),
        }
        return str(reason), meta

    def _snapshot_local_window_context(self) -> Tuple[Dict[int, Dict[str, object]], Dict[int, List[int]], Dict[int, List[int]]]:
        snapshot = getattr(self.opt, "best", None) or getattr(self.opt, "work", None)
        rows = list(getattr(snapshot, "subtask_state", []) or []) if snapshot is not None else []
        metrics: Dict[int, Dict[str, object]] = {}
        station_rows: Dict[int, List[Tuple[int, float, int]]] = {}
        robot_rows: Dict[int, List[Tuple[float, float, int]]] = {}
        for subtask in rows:
            subtask_id = int(getattr(subtask, "id", getattr(subtask, "subtask_id", -1)))
            if subtask_id < 0:
                continue
            task_rows = list(getattr(subtask, "execution_tasks", []) or [])
            station_id = int(getattr(subtask, "assigned_station_id", -1))
            rank = int(getattr(subtask, "station_sequence_rank", -1))
            start_time = min((float(getattr(task, "start_process_time", 0.0) or 0.0) for task in task_rows), default=0.0)
            completion_time = max((float(getattr(task, "end_process_time", 0.0) or 0.0) for task in task_rows), default=0.0)
            arrival_stack = min((float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0) for task in task_rows), default=0.0)
            arrival_station = min((float(getattr(task, "arrival_time_at_station", 0.0) or 0.0) for task in task_rows), default=0.0)
            robot_ids = sorted({int(getattr(task, "robot_id", -1)) for task in task_rows if int(getattr(task, "robot_id", -1)) >= 0})
            metrics[int(subtask_id)] = {
                "station_id": int(station_id),
                "station_rank": int(rank),
                "robot_ids": list(robot_ids),
                "start_time": float(start_time),
                "completion_time": float(completion_time),
                "arrival_stack": float(arrival_stack),
                "arrival_station": float(arrival_station),
            }
            if station_id >= 0:
                station_rows.setdefault(int(station_id), []).append((rank if rank >= 0 else 10**9, float(start_time), int(subtask_id)))
            for robot_id in robot_ids:
                robot_rows.setdefault(int(robot_id), []).append((float(arrival_stack), float(arrival_station), int(subtask_id)))
        station_chains = {
            int(station_id): [int(item[2]) for item in sorted(chain, key=lambda item: (int(item[0]), float(item[1]), int(item[2])))]
            for station_id, chain in station_rows.items()
        }
        robot_chains = {
            int(robot_id): [int(item[2]) for item in sorted(chain, key=lambda item: (float(item[0]), float(item[1]), int(item[2])))]
            for robot_id, chain in robot_rows.items()
        }
        return metrics, station_chains, robot_chains

    def _coverage_issue_subtask_ids(self, config: ResourceConfig, *, limit: int = 0) -> List[int]:
        tote_map = dict(getattr(getattr(self.opt, "problem", None), "id_to_tote", {}) or {})
        coverage = config.coverage_summary(tote_map)
        if bool(coverage.get("coverage_ok", False)):
            return []
        metrics, _station_chains, _robot_chains = self._snapshot_local_window_context()
        rows = []
        for row in list(coverage.get("subtasks", []) or []):
            subtask_id = int(row.get("subtask_id", -1))
            unmet = int(row.get("unmet_sku_units", 0) or 0)
            if subtask_id < 0 or unmet <= 0:
                continue
            rows.append(
                (
                    -int(unmet),
                    -float(metrics.get(int(subtask_id), {}).get("completion_time", 0.0)),
                    int(subtask_id),
                )
            )
        ranked = [int(item[2]) for item in sorted(rows)]
        if int(limit) > 0:
            return ranked[: int(limit)]
        return ranked

    def _local_retry_caps(self, attr_name: str, *, base_cap: int) -> List[int]:
        raw = getattr(self.cfg, attr_name, "")
        if raw is None or str(raw).strip() == "":
            raw = "48,64"
        caps: List[int] = []
        if isinstance(raw, (list, tuple, set)):
            parts = list(raw)
        else:
            parts = str(raw).replace(";", ",").split(",")
        for part in parts:
            try:
                value = int(float(str(part).strip()))
            except Exception:
                continue
            if value > int(base_cap):
                caps.append(int(value))
        return sorted(set(caps))

    def _expand_local_release_window(self, seed_ids, *, cap: int, radius: int = 1, config: Optional[ResourceConfig] = None) -> Set[int]:
        base_config = config if config is not None else self.current_config
        available_ids = set(int(x) for x in base_config.subtasks.keys())
        seed_list = [int(x) for x in (seed_ids or [])]
        seed_rank: Dict[int, int] = {}
        for idx, subtask_id in enumerate(seed_list):
            seed_rank.setdefault(int(subtask_id), int(idx))
        selected: List[int] = []
        seen: Set[int] = set()

        def add(subtask_id: int) -> None:
            subtask_id = int(subtask_id)
            if subtask_id in available_ids and subtask_id not in seen:
                seen.add(subtask_id)
                selected.append(subtask_id)

        for subtask_id in seed_list:
            add(int(subtask_id))
        metrics, station_chains, robot_chains = self._snapshot_local_window_context()
        for subtask_id in list(selected):
            meta = dict(metrics.get(int(subtask_id), {}) or {})
            station_id = int(meta.get("station_id", -1))
            chain = list(station_chains.get(int(station_id), []) or [])
            if int(subtask_id) in chain:
                idx = chain.index(int(subtask_id))
                for offset in range(-int(radius), int(radius) + 1):
                    if offset and 0 <= idx + offset < len(chain):
                        add(chain[idx + offset])
            for robot_id in list(meta.get("robot_ids", []) or []):
                robot_chain = list(robot_chains.get(int(robot_id), []) or [])
                if int(subtask_id) in robot_chain:
                    idx = robot_chain.index(int(subtask_id))
                    for offset in range(-int(radius), int(radius) + 1):
                        if offset and 0 <= idx + offset < len(robot_chain):
                            add(robot_chain[idx + offset])
        late_rows = sorted(
            ((-float(meta.get("completion_time", 0.0)), int(subtask_id)) for subtask_id, meta in metrics.items()),
            key=lambda item: (float(item[0]), int(item[1])),
        )
        late_take = max(0, int(getattr(self.cfg, "resource_local_window_late_subtask_count", 2)))
        for _score, subtask_id in late_rows[:late_take]:
            add(int(subtask_id))
        if len(selected) <= int(cap):
            return set(int(x) for x in selected)
        ranked = sorted(
            selected,
            key=lambda subtask_id: (
                0 if int(subtask_id) in seed_rank else 1,
                int(seed_rank.get(int(subtask_id), 10**9)),
                -float(metrics.get(int(subtask_id), {}).get("completion_time", 0.0)),
                int(subtask_id),
            ),
        )
        return set(int(x) for x in ranked[: max(1, int(cap))])

    def _candidate_validation_trigger(self, iter_id: int, candidate_eval, candidate_signature, precomputed_validation) -> str:
        if precomputed_validation is not None:
            return str(precomputed_validation.get("_trigger_reason", "joint_colocated_sort_postprocess"))
        trigger = str(self._should_validate(iter_id, candidate_eval, candidate_signature) or "")
        if trigger:
            return trigger
        # Every accepted exact candidate must clear a route-feasibility screen before
        # it can become the current incumbent; otherwise unassigned SP4 tasks can slip
        # through between periodic validations.
        return "operator_route_precheck"

    def _init_operator_arms(self) -> Dict[str, Dict[str, Dict[str, OperatorArm]]]:
        arms = {
            "X": {
                "destroy": {name: OperatorArm(name=name) for name in X_DESTROY_OPERATORS.keys()},
                "repair": {name: OperatorArm(name=name) for name in list(X_REPAIR_OPERATORS.keys()) + [X_FALLBACK_OPERATOR]},
            },
            "Y": {
                "destroy": {name: OperatorArm(name=name) for name in Y_DESTROY_OPERATORS.keys()},
                "repair": {name: OperatorArm(name=name) for name in list(Y_REPAIR_OPERATORS.keys()) + [Y_FALLBACK_OPERATOR]},
            },
            "Z": {
                "destroy": {name: OperatorArm(name=name) for name in Z_DESTROY_OPERATORS.keys()},
                "repair": {name: OperatorArm(name=name) for name in list(Z_REPAIR_OPERATORS.keys()) + [Z_FALLBACK_OPERATOR]},
            },
        }
        if "XYZ" in getattr(self, "resource_layers", []):
            arms["XYZ"] = {
                "destroy": {"xyz_destroy_sequential": OperatorArm(name="xyz_destroy_sequential")},
                "repair": {"xyz_repair_sequential": OperatorArm(name="xyz_repair_sequential")},
            }
        if "XZ" in getattr(self, "resource_layers", []):
            arms["XZ"] = {
                "destroy": {"xz_destroy_joint": OperatorArm(name="xz_destroy_joint")},
                "repair": {"xz_repair_joint": OperatorArm(name="xz_repair_joint")},
            }
        if "YZ" in getattr(self, "resource_layers", []):
            arms["YZ"] = {
                "destroy": {"yz_destroy_joint": OperatorArm(name="yz_destroy_joint")},
                "repair": {"yz_repair_joint": OperatorArm(name="yz_repair_joint")},
            }
        if "XY" in getattr(self, "resource_layers", []):
            arms["XY"] = {
                "destroy": {"xy_destroy_joint": OperatorArm(name="xy_destroy_joint")},
                "repair": {"xy_repair_joint": OperatorArm(name="xy_repair_joint")},
            }
        if "U" in getattr(self, "resource_layers", []):
            arms["U"] = {
                "destroy": {"u_destroy_heavy_robot_route": OperatorArm(name="u_destroy_heavy_robot_route")},
                "repair": {"u_fast_repair_dispatch": OperatorArm(name="u_fast_repair_dispatch")},
            }
        if not bool(getattr(self.cfg, "resource_enable_experimental_z_shared_stack", False)):
            arms["Z"]["destroy"].pop("z_destroy_shared_stack_window", None)
            arms["Z"]["repair"].pop("z_repair_cross_subtask_shared_stack", None)
            arms["Z"]["repair"].pop("z_repair_multistack_cover_compact", None)
        profile = str(getattr(self.cfg, "resource_operator_profile", "baseline_safe") or "baseline_safe").strip().lower()
        if profile == "baseline_safe":
            # Keep the default Z search space aligned with the older 161626 baseline.
            # Later-added diversification operators change the sampled operator pool and
            # split probability mass away from the historical sort/flip repair path.
            for name in (
                "z_destroy_spread_hotspot_window",
                "z_destroy_random_window",
                "z_destroy_related_stack_window",
            ):
                arms["Z"]["destroy"].pop(name, None)
            for name in (
                "z_repair_same_stack_window",
                "z_repair_load_balance_idle_robot",
            ):
                arms["Z"]["repair"].pop(name, None)
        if not bool(getattr(self.cfg, "resource_enable_experimental_x_repartition", False)):
            arms["X"]["destroy"].pop("x_destroy_order_repartition", None)
            arms["X"]["destroy"].pop("x_destroy_critical_order_cluster", None)
            arms["X"]["repair"].pop("x_repair_partition_dp", None)
            arms["X"]["repair"].pop("x_repair_station_balanced_partition", None)
            arms["X"]["repair"].pop("x_repair_sku_cluster_beam", None)
        if not bool(getattr(self.cfg, "resource_enable_critical_path_xyz", False)):
            arms["X"]["destroy"].pop("x_destroy_critical_order_cluster", None)
            arms["X"]["repair"].pop("x_repair_sku_cluster_beam", None)
            arms["Y"]["destroy"].pop("y_destroy_critical_path_block", None)
            arms["Y"]["repair"].pop("y_repair_ejection_chain_balance", None)
            arms["Z"]["destroy"].pop("z_destroy_critical_path_window", None)
        large_case_polish_enabled = bool(
            bool(getattr(self.cfg, "resource_enable_experimental_y_rank_permutation", False))
            and len(list(getattr(getattr(self.opt, "problem", None), "station_list", []) or [])) >= 4
            and len(list(getattr(getattr(self.opt, "problem", None), "subtask_list", []) or [])) >= 5
        )
        if not large_case_polish_enabled:
            arms["Y"]["repair"].pop("y_repair_station_rank_permutation", None)
        if not bool(getattr(self.cfg, "resource_enable_experimental_z_joint_polish", False)):
            arms["Z"]["repair"].pop("z_repair_stack_mode_joint_polish", None)
        if profile == "route_polish_exact":
            if "z_destroy_critical_path_window" in arms["Z"]["destroy"]:
                arms["Z"]["destroy"]["z_destroy_critical_path_window"].weight = 6.0
            if "y_destroy_critical_path_block" in arms["Y"]["destroy"]:
                arms["Y"]["destroy"]["y_destroy_critical_path_block"].weight = 3.5
            if "x_destroy_critical_order_cluster" in arms["X"]["destroy"]:
                arms["X"]["destroy"]["x_destroy_critical_order_cluster"].weight = 4.0
            if "z_repair_mode_toggle_contextual" in arms["Z"]["repair"]:
                arms["Z"]["repair"]["z_repair_mode_toggle_contextual"].weight = 6.0
            if "z_repair_gurobi_like_sort" in arms["Z"]["repair"]:
                arms["Z"]["repair"]["z_repair_gurobi_like_sort"].weight = 4.0
        for name, weight in {
            "z_repair_gurobi_like_sort": 3.0,
            "z_repair_sort_range_shrink_first": 2.0,
            "z_repair_mode_toggle_contextual": 2.5,
            "z_repair_joint_sort_colocated_flip": 2.0,
            "z_repair_spread_region_balance": 2.5,
            "z_repair_load_balance_idle_robot": 2.5,
            "z_repair_cross_subtask_shared_stack": 2.75,
            "z_repair_multistack_cover_compact": 2.75,
            "z_repair_stack_mode_joint_polish": 3.0,
        }.items():
            if name in arms["Z"]["repair"]:
                arms["Z"]["repair"][name].weight = float(weight)
        if profile == "route_polish_exact":
            if "z_repair_mode_toggle_contextual" in arms["Z"]["repair"]:
                arms["Z"]["repair"]["z_repair_mode_toggle_contextual"].weight = 6.0
            if "z_repair_gurobi_like_sort" in arms["Z"]["repair"]:
                arms["Z"]["repair"]["z_repair_gurobi_like_sort"].weight = 4.0
        if "y_repair_station_rank_permutation" in arms["Y"]["repair"]:
            arms["Y"]["repair"]["y_repair_station_rank_permutation"].weight = 3.0
        if "y_repair_station_rank_permutation" in arms["Y"]["repair"] and "y_destroy_rank_window_release" in arms["Y"]["destroy"]:
            arms["Y"]["destroy"]["y_destroy_rank_window_release"].weight = max(
                float(arms["Y"]["destroy"]["y_destroy_rank_window_release"].weight),
                2.5,
            )
        if "z_destroy_spread_hotspot_window" in arms["Z"]["destroy"]:
            arms["Z"]["destroy"]["z_destroy_spread_hotspot_window"].weight = 2.5
        if "z_destroy_shared_stack_window" in arms["Z"]["destroy"]:
            arms["Z"]["destroy"]["z_destroy_shared_stack_window"].weight = 3.0
        if "y_destroy_heavy_robot_tail" in arms["Y"]["destroy"]:
            arms["Y"]["destroy"]["y_destroy_heavy_robot_tail"].weight = 2.0
        if "x_destroy_order_repartition" in arms["X"]["destroy"]:
            arms["X"]["destroy"]["x_destroy_order_repartition"].weight = 3.0
        for name in ("x_repair_partition_dp", "x_repair_station_balanced_partition"):
            if name in arms["X"]["repair"]:
                arms["X"]["repair"][name].weight = 2.75
        if "x_destroy_critical_order_cluster" in arms["X"]["destroy"]:
            arms["X"]["destroy"]["x_destroy_critical_order_cluster"].weight = 3.5
        if "x_repair_sku_cluster_beam" in arms["X"]["repair"]:
            arms["X"]["repair"]["x_repair_sku_cluster_beam"].weight = 3.0
        if "y_destroy_critical_path_block" in arms["Y"]["destroy"]:
            arms["Y"]["destroy"]["y_destroy_critical_path_block"].weight = 3.0
        if "y_repair_ejection_chain_balance" in arms["Y"]["repair"]:
            arms["Y"]["repair"]["y_repair_ejection_chain_balance"].weight = 2.75
        if "z_destroy_critical_path_window" in arms["Z"]["destroy"]:
            arms["Z"]["destroy"]["z_destroy_critical_path_window"].weight = 3.25
        if profile == "route_polish_exact":
            if "z_destroy_critical_path_window" in arms["Z"]["destroy"]:
                arms["Z"]["destroy"]["z_destroy_critical_path_window"].weight = 6.0
        self._apply_operator_weight_floors(arms)
        return arms

    def _z_diversification_names(self) -> Tuple[set[str], set[str]]:
        return (
            {"z_destroy_spread_hotspot_window"},
            {"z_repair_spread_region_balance", "z_repair_load_balance_idle_robot"},
        )

    def _apply_operator_weight_floors(self, arms=None) -> None:
        arms = self.operator_arms if arms is None else arms
        destroy_names, repair_names = self._z_diversification_names()
        floor = float(getattr(getattr(self, "cfg", None), "resource_z_diversification_weight_floor", 1.25))
        for name in destroy_names:
            arm = arms.get("Z", {}).get("destroy", {}).get(str(name))
            if arm is not None:
                arm.weight = float(max(float(arm.weight), floor))
        for name in repair_names:
            arm = arms.get("Z", {}).get("repair", {}).get(str(name))
            if arm is not None:
                arm.weight = float(max(float(arm.weight), floor))

    def _refresh_operator_stats_payload(self) -> None:
        payload: Dict[str, Dict[str, Dict[str, float]]] = {}
        for layer, groups in self.operator_arms.items():
            payload[layer] = {}
            for arm_group in groups.values():
                for name, arm in arm_group.items():
                    avg_reward = float(sum(arm.pending_rewards) / max(1, len(arm.pending_rewards))) if arm.pending_rewards else 0.0
                    payload[layer][name] = {
                        "reward_mean": float(avg_reward),
                        "weight": float(arm.weight),
                        "execution_count": float(arm.execution_count),
                    }
        self.opt.operator_stats = payload

    def _weighted_pick(self, arms: Dict[str, OperatorArm]) -> str:
        rows = list(arms.values())
        total = float(sum(max(0.0, float(arm.weight)) for arm in rows))
        if total <= 1e-9:
            return str(rows[0].name)
        draw = self.rng.random() * total
        acc = 0.0
        for arm in rows:
            acc += max(0.0, float(arm.weight))
            if draw <= acc:
                return str(arm.name)
        return str(rows[-1].name)

    def _available_layers(self, iter_id: int) -> List[str]:
        all_layers = list(getattr(self, "resource_layers", ["X", "Y", "Z"]))
        if bool(getattr(self.cfg, "sp1_no_split", False)):
            all_layers = [layer for layer in all_layers if str(layer).upper() not in {"X", "XZ"}]
        self.last_xyz_skip_reason = ""
        configured_order_text = str(getattr(self.cfg, "revolving_layer_order", "") or "").strip()
        configured_revolving_order = {
            part.strip().upper()
            for part in configured_order_text.split(",")
            if part.strip()
        }
        if configured_order_text.upper() == "AUTO":
            configured_revolving_order = set(self._auto_revolving_layer_order())
        if (
            "XYZ" in all_layers
            and bool(getattr(self.cfg, "resource_xyz_stagnation_gate", True))
            and "XYZ" not in configured_revolving_order
            and not (
                bool(getattr(self.cfg, "resource_global_decomp_repair_enabled", False))
                and not bool(getattr(self, "global_decomp_repair_used", False))
            )
        ):
            trigger = int(getattr(self.cfg, "resource_xyz_trigger_stagnation_rounds", 8))
            recent_repartition = int(iter_id) - int(getattr(self, "last_x_repartition_iter", -10**9)) <= 1
            stagnant = float(getattr(self, "no_improve_rounds", 0.0)) >= float(trigger) or float(getattr(self, "no_best_z_change_rounds", 0.0)) >= float(trigger)
            if not (bool(recent_repartition) or bool(stagnant)):
                all_layers = [layer for layer in all_layers if str(layer).upper() != "XYZ"]
                self.last_xyz_skip_reason = f"stagnation_gate<{trigger}"
        available = [
            layer
            for layer in all_layers
            if max(
                int(self.layer_cooldown_until_iter.get(layer, 0)),
                int(self.layer_failure_cooldown_until_iter.get(layer, 0)),
            )
            < int(iter_id)
        ]
        selected = available if available else all_layers
        self.last_available_layers = list(selected)
        return selected

    def _auto_revolving_layer_order(self) -> List[str]:
        problem = getattr(self.opt, "problem", None)
        order_num = int(getattr(problem, "order_num", 0) or 0)
        if order_num <= 6:
            return ["YZ", "Y", "XZ"]
        if order_num <= 7:
            return ["Y", "YZ", "U"]
        return ["Y", "U", "Y", "YZ"]

    def _round_robin_next(self, available_layers: Optional[List[str]] = None) -> str:
        order = list(getattr(self, "resource_layers", ["X", "Y", "Z"]))
        available = list(available_layers or order)
        if str(self.last_selected_layer) not in order:
            return available[0]
        idx = order.index(str(self.last_selected_layer))
        for offset in range(1, len(order) + 1):
            candidate = order[(idx + offset) % len(order)]
            if candidate in available:
                return candidate
        return available[0]

    def _current_layer_cooldown_remaining(self, layer: str, iter_id: int) -> int:
        until_iter = int(self.layer_cooldown_until_iter.get(str(layer), 0))
        return int(max(0, until_iter - int(iter_id) + 1))

    def _current_failure_cooldown_remaining(self, layer: str, iter_id: int) -> int:
        until_iter = int(self.layer_failure_cooldown_until_iter.get(str(layer), 0))
        return int(max(0, until_iter - int(iter_id) + 1))

    def _select_layer(self, iter_id: int) -> Tuple[str, bool]:
        available_layers = self._available_layers(int(iter_id))
        forced_queue = getattr(self, "forced_layer_queue", None)
        if forced_queue is None:
            forced_queue = deque()
            self.forced_layer_queue = forced_queue
        if bool(getattr(self.cfg, "resource_revolving_mode", False)):
            configured_order = str(getattr(self.cfg, "revolving_layer_order", "") or "").strip()
            if configured_order.upper() == "AUTO":
                order = self._auto_revolving_layer_order()
            elif configured_order:
                order = [part.strip().upper() for part in configured_order.split(",") if part.strip()]
            else:
                order = ["X", "Y", "Z"]
                if bool(getattr(self.cfg, "resource_revolving_enable_u_layer", False)):
                    order.append("U")
            order = [layer for layer in order if layer in available_layers]
            if order:
                layer = str(order[(int(iter_id) - 1) % len(order)])
                self.last_selected_layer = layer
                self.last_selected_layer_source = "revolving_cycle"
                return layer, True
        while forced_queue:
            layer = str(forced_queue.popleft()).upper()
            if layer in available_layers:
                self.last_selected_layer = layer
                self.last_selected_layer_source = "forced_queue"
                return layer, True
        wx = float(getattr(self.cfg, "resource_component_weight_x", 1.0))
        wy = float(getattr(self.cfg, "resource_component_weight_y", 1.0))
        wz = float(getattr(self.cfg, "resource_component_weight_z", 1.0))
        wxyz = float(getattr(self.cfg, "resource_component_weight_xyz", 1.0))
        wu = float(getattr(self.cfg, "resource_component_weight_u", 1.0))
        bx = float(getattr(self.cfg, "resource_layer_base_weight_x", 0.10))
        by = float(getattr(self.cfg, "resource_layer_base_weight_y", 0.45))
        bz = float(getattr(self.cfg, "resource_layer_base_weight_z", 0.45))
        bxyz = float(getattr(self.cfg, "resource_layer_base_weight_xyz", 0.15))
        bu = float(getattr(self.cfg, "resource_layer_base_weight_u", 0.35))
        pressure = {
            "X": float(self.current_eval.Sx),
            "Y": float(self.current_eval.Sy),
            "Z": float(self.current_eval.Sz),
            "XZ": float((float(self.current_eval.Sx) + float(self.current_eval.Sz)) / 2.0),
            "YZ": float((float(self.current_eval.Sy) + float(self.current_eval.Sz)) / 2.0),
            "XY": float((float(self.current_eval.Sx) + float(self.current_eval.Sy)) / 2.0),
            "XYZ": float((float(self.current_eval.Sx) + float(self.current_eval.Sy) + float(self.current_eval.Sz)) / 3.0),
            "U": float(max(1.0, float(getattr(self.best_validated, "makespan", 1.0) or 1.0)) / max(1.0, float(getattr(self.current_eval, "F_raw", 1.0) or 1.0))),
        }
        base_weight = {
            "X": float(bx * wx),
            "Y": float(by * wy),
            "Z": float(bz * wz),
            "XZ": float(0.5 * (bx * wx + bz * wz)),
            "YZ": float(0.5 * (by * wy + bz * wz)),
            "XY": float(0.5 * (bx * wx + by * wy)),
            "XYZ": float(bxyz * wxyz),
            "U": float(bu * wu),
        }
        boost = float(getattr(self.cfg, "resource_stagnation_boost", 0.15))
        eps = max(1e-9, float(getattr(self.cfg, "resource_layer_score_epsilon", 0.05)))
        if any(float(self.layer_stagnation[layer]) >= float(getattr(self.cfg, "resource_force_rotate_threshold", 20)) for layer in available_layers):
            layer = self._round_robin_next(available_layers)
            self.last_selected_layer = str(layer)
            self.last_selected_layer_source = "force_rotate"
            return str(layer), True
        if self.rng.random() < float(getattr(self.cfg, "resource_layer_explore_eps", 0.10)):
            layer = str(self.rng.choice(available_layers))
            self.last_selected_layer = layer
            self.last_selected_layer_source = "explore"
            return layer, False
        scores: Dict[str, float] = {}
        for layer in available_layers:
            score = float(base_weight[layer] * float(self.layer_dynamic_multiplier.get(layer, 1.0)) * pressure[layer] / (float(self.layer_ema_improve[layer]) + eps))
            if float(self.layer_stagnation[layer]) > 0.0:
                score *= float(1.0 + boost * min(5.0, float(self.layer_stagnation[layer])))
            scores[layer] = max(score, eps)
        total = float(sum(scores.values()))
        draw = self.rng.random() * total
        acc = 0.0
        for layer in available_layers:
            acc += float(scores[layer])
            if draw <= acc:
                self.last_selected_layer = layer
                self.last_selected_layer_source = "weighted"
                return layer, False
        fallback_layer = available_layers[-1]
        self.last_selected_layer = fallback_layer
        self.last_selected_layer_source = "weighted_fallback"
        return fallback_layer, False

    def _current_destroy_mu(self) -> Tuple[float, bool, str]:
        medium_trigger = int(getattr(self.cfg, "resource_destroy_mu_medium_trigger", 30))
        heavy_trigger = int(getattr(self.cfg, "resource_heavy_destroy_trigger", 50))
        if float(self.no_improve_rounds) >= float(heavy_trigger):
            tier_mu = float(getattr(self.cfg, "resource_destroy_mu_heavy", 0.35))
            destroy_tier = "heavy"
            heavy = True
        elif float(self.no_improve_rounds) >= float(medium_trigger):
            tier_mu = float(getattr(self.cfg, "resource_destroy_mu_medium", 0.20))
            destroy_tier = "medium"
            heavy = False
        else:
            tier_mu = float(getattr(self.cfg, "resource_destroy_mu_base", 0.10))
            destroy_tier = "base"
            heavy = False
        cap = float(getattr(self.cfg, "resource_adaptive_destroy_bonus_cap", 0.20))
        effective_mu = float(min(0.40, tier_mu + min(cap, float(getattr(self, "adaptive_destroy_bonus", 0.0)))))
        return float(effective_mu), bool(heavy), str(destroy_tier)

    def _layer_population(self, layer: str) -> int:
        if str(layer) == "X":
            return int(sum(len(row.work_unit_ids or ()) for row in self.current_config.subtasks.values()))
        if str(layer) == "Y":
            return int(len(self.current_config.subtasks))
        if str(layer).upper() == "XYZ":
            return int(
                max(
                    sum(len(row.work_unit_ids or ()) for row in self.current_config.subtasks.values()),
                    len(self.current_config.subtasks),
                    sum(len(row.z_tasks or []) for row in self.current_config.subtasks.values()),
                )
            )
        if str(layer).upper() == "XZ":
            return int(
                max(
                    sum(len(row.work_unit_ids or ()) for row in self.current_config.subtasks.values()),
                    sum(len(row.z_tasks or []) for row in self.current_config.subtasks.values()),
                )
            )
        if str(layer).upper() == "YZ":
            return int(
                max(
                    len(self.current_config.subtasks),
                    sum(len(row.z_tasks or []) for row in self.current_config.subtasks.values()),
                )
            )
        if str(layer).upper() == "XY":
            return int(
                max(
                    sum(len(row.work_unit_ids or ()) for row in self.current_config.subtasks.values()),
                    len(self.current_config.subtasks),
                )
            )
        return int(sum(len(row.z_tasks or []) for row in self.current_config.subtasks.values()))

    def _effective_destroy_budget(self, layer: str, mu: float) -> int:
        base = int(getattr(self.cfg, f"resource_destroy_degree_{str(layer).lower()}", 1))
        population = max(1, int(self._layer_population(layer)))
        dynamic = int(math.ceil(float(mu) * float(population)))
        return int(max(base, dynamic))

    def _sample_operator_pair(self, layer: str) -> Tuple[str, str]:
        if str(layer).upper() == "Y":
            profile = str(getattr(self.cfg, "resource_operator_profile", "") or "").strip().lower()
            if (
                profile == "no_split_y_focus"
                and "y_destroy_global_station_release" in self.operator_arms["Y"]["destroy"]
                and "y_repair_global_route_balance" in self.operator_arms["Y"]["repair"]
                and int(getattr(self, "layer_exec_since_update", {}).get("Y", 0)) % 3 == 0
            ):
                return "y_destroy_global_station_release", "y_repair_global_route_balance"
            if (
                "y_destroy_rank_window_release" in self.operator_arms["Y"]["destroy"]
                and "y_repair_station_rank_permutation" in self.operator_arms["Y"]["repair"]
                and int(getattr(self, "layer_exec_since_update", {}).get("Y", 0)) % 5 == 4
            ):
                return "y_destroy_rank_window_release", "y_repair_station_rank_permutation"
        if str(layer).upper() == "Z":
            self.z_operator_pick_count = int(getattr(self, "z_operator_pick_count", 0)) + 1
            shared_period = max(0, int(getattr(self.cfg, "z_shared_stack_destroy_period", 5)))
            if (
                shared_period > 0
                and self.z_operator_pick_count % shared_period == 0
                and "z_destroy_shared_stack_window" in self.operator_arms["Z"]["destroy"]
                and "z_repair_cross_subtask_shared_stack" in self.operator_arms["Z"]["repair"]
            ):
                return "z_destroy_shared_stack_window", "z_repair_cross_subtask_shared_stack"
            period = max(0, int(getattr(self.cfg, "resource_z_diversification_force_period", 5)))
            destroy_names, repair_names = self._z_diversification_names()
            if period > 0 and self.z_operator_pick_count % period == 0:
                destroy_available = [name for name in sorted(destroy_names) if name in self.operator_arms["Z"]["destroy"]]
                repair_available = [name for name in sorted(repair_names) if name in self.operator_arms["Z"]["repair"]]
                if destroy_available and repair_available:
                    return str(destroy_available[0]), str(repair_available[(self.z_operator_pick_count // period) % len(repair_available)])
            if (
                "z_destroy_mode_window" in self.operator_arms["Z"]["destroy"]
                and "z_repair_stack_mode_joint_polish" in self.operator_arms["Z"]["repair"]
                and self.z_operator_pick_count % 4 == 2
            ):
                return "z_destroy_mode_window", "z_repair_stack_mode_joint_polish"
        destroy_name = self._weighted_pick(self.operator_arms[layer]["destroy"])
        repair_candidates = {
            name: arm
            for name, arm in self.operator_arms[layer]["repair"].items()
            if not str(name).endswith("greedy_fallback")
        }
        repair_name = self._weighted_pick(repair_candidates)
        return str(destroy_name), str(repair_name)

    def _candidate_signature_text(self, signature) -> str:
        return repr(signature)

    def _action_signature_text(self, signature) -> str:
        return repr(signature)

    def _action_signature_known(self, layer: str, signature) -> bool:
        return self._action_signature_text(signature) in self.action_signature_seen[str(layer)]

    def _remember_action_signature(self, layer: str, signature) -> None:
        layer_name = str(layer)
        signature_text = self._action_signature_text(signature)
        if signature_text in self.action_signature_seen[layer_name]:
            return
        history = self.action_signature_history[layer_name]
        seen = self.action_signature_seen[layer_name]
        if len(history) >= int(history.maxlen or 0) and history:
            evicted = history.popleft()
            seen.discard(str(evicted))
        history.append(signature_text)
        seen.add(signature_text)

    def _candidate_sort_key(self, row: Dict[str, object]) -> Tuple[float, float, int, str, str, str]:
        return (
            float(row.get("F_cal", float("inf"))),
            float(row.get("F_raw", float("inf"))),
            1 if bool(row.get("fallback_used", False)) else 0,
            str(row.get("destroy_operator", "")),
            str(row.get("repair_operator", "")),
            str(row.get("candidate_signature", "")),
        )

    def _select_best_candidate(self, candidate_rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not list(candidate_rows or []):
            return None
        ordered = sorted(candidate_rows, key=self._candidate_sort_key)
        for rank, row in enumerate(ordered, start=1):
            row["candidate_rank"] = int(rank)
            row["selected_for_sa"] = False
        return ordered[0]

    def _score_rough_candidate(self, layer: str, rough_features: Dict[str, object], fallback_used: bool, iter_id: int) -> UpperEvalResult:
        if str(layer) == "Y":
            return self.scorer.score_rough_y_action(
                self.current_eval,
                rough_features,
                fallback_penalty=0.15 if bool(fallback_used) else 0.0,
                iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                distance_to_last_validated=0.0,
            )
        return self.scorer.score_rough_z_action(
            self.current_eval,
            rough_features,
            fallback_penalty=0.15 if bool(fallback_used) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=0.0,
        )

    def _build_x_action_signature(self, destroy_name: str, repair_name: str) -> Tuple[object, ...]:
        return ("X", str(destroy_name), str(repair_name), self.current_config.validation_signature())

    def _apply_x_candidate(self, iter_id: int, destroy_name: str, repair_name: str, degree: int) -> Optional[Dict[str, object]]:
        return self._apply_x_candidate_to_config(
            base_config=self.current_config,
            base_eval=self.current_eval,
            iter_id=int(iter_id),
            destroy_name=str(destroy_name),
            repair_name=str(repair_name),
            degree=int(degree),
        )

    def _apply_x_candidate_to_config(
        self,
        base_config: ResourceConfig,
        base_eval,
        iter_id: int,
        destroy_name: str,
        repair_name: str,
        degree: int,
    ) -> Optional[Dict[str, object]]:
        if bool(getattr(self.cfg, "sp1_no_split", False)):
            return None
        candidate = base_config.clone()
        destroy_ctx = X_DESTROY_OPERATORS[str(destroy_name)](self.opt, candidate, self.rng, degree)
        if not bool(destroy_ctx.get("success", False)):
            return None
        repair_result = X_REPAIR_OPERATORS[str(repair_name)](self.opt, candidate, destroy_ctx, self.rng)
        fallback_used = False
        if not bool(repair_result.get("success", False)):
            repair_result = x_repair_greedy_fallback(self.opt, candidate, destroy_ctx, self.rng)
            fallback_used = bool(repair_result.get("success", False))
        if not bool(repair_result.get("success", False)):
            return None
        affected_ids = set(int(x) for x in (repair_result.get("affected_subtask_ids", set()) or set()))
        candidate, score_cache, projection_meta = apply_projection_repair(
            opt=self.opt,
            previous_config=base_config,
            candidate_config=candidate,
            previous_eval=base_eval,
            affected_subtask_ids=sorted(affected_ids),
            iter_id=int(iter_id),
            rng=self.rng,
        )
        fallback_used = bool(fallback_used or projection_meta.get("fallback_used", False))
        return {
            "config": candidate,
            "score_cache": score_cache,
            "affected_ids": affected_ids,
            "fallback_used": bool(fallback_used),
            "projection_mode": str(projection_meta.get("projection_mode", "")),
            "projection_repaired_subtask_count": int(projection_meta.get("projection_repaired_subtask_count", 0)),
            "projection_z_repair_fail_count": int(projection_meta.get("projection_z_repair_fail_count", 0) or 0),
            "projection_z_repair_fail_reasons": dict(projection_meta.get("projection_z_repair_fail_reasons", {}) or {}),
            "projection_z_repair_failed_subtask_ids": list(projection_meta.get("projection_z_repair_failed_subtask_ids", []) or []),
        }

    def _build_x_exact_candidate(self, iter_id: int, destroy_name: str, repair_name: str, budget: int) -> Optional[Dict[str, object]]:
        action_signature = self._build_x_action_signature(str(destroy_name), str(repair_name))
        if self._action_signature_known("X", action_signature):
            return None
        self._remember_action_signature("X", action_signature)
        payload = self._apply_x_candidate(int(iter_id), str(destroy_name), str(repair_name), int(budget))
        if payload is None:
            return None
        candidate_signature = payload["config"].validation_signature()
        candidate_eval = self._evaluate_config(
            config=payload["config"],
            layer="X",
            score_cache=payload.get("score_cache", None),
            affected_subtask_ids=payload.get("affected_ids", set()),
            fallback_penalty=0.15 if bool(payload.get("fallback_used", False)) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(payload["config"], self.last_validated_config),
        )
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
            }
        )
        return {
            "iter": int(iter_id),
            "layer": "X",
            "candidate_stage": "exact",
            "candidate_rank": 0,
            "destroy_operator": str(destroy_name),
            "repair_operator": str(repair_name),
            "fallback_used": bool(payload.get("fallback_used", False)),
            "projection_mode": str(payload.get("projection_mode", "")),
            "projection_repaired_subtask_count": int(payload.get("projection_repaired_subtask_count", 0)),
            "projection_z_repair_fail_count": int(payload.get("projection_z_repair_fail_count", 0) or 0),
            "projection_z_repair_fail_reasons": str(payload.get("projection_z_repair_fail_reasons", {}) or {}),
            "projection_z_repair_failed_subtask_ids": str(payload.get("projection_z_repair_failed_subtask_ids", []) or []),
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_fixed_constraint_count": candidate_eval.metadata.get("fixgurobi_fixed_constraint_count", ""),
            "fixgurobi_invalid_fix_count": candidate_eval.metadata.get("fixgurobi_invalid_fix_count", ""),
            "fixgurobi_fixed_route_arc_count_from_cfg": candidate_eval.metadata.get("fixgurobi_fixed_route_arc_count_from_cfg", ""),
            "fixgurobi_fixed_route_sequence_robot_count": candidate_eval.metadata.get("fixgurobi_fixed_route_sequence_robot_count", ""),
            "fixgurobi_fixed_route_sequence_missing_count": candidate_eval.metadata.get("fixgurobi_fixed_route_sequence_missing_count", ""),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(action_signature),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
        }

    def _generate_x_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(target_size))
        max_attempts = max(target, int(getattr(self.cfg, "resource_candidate_pool_max_attempts", 12)))
        attempts = 0
        generated_count = 0
        pool: List[Dict[str, object]] = []
        candidate_rows: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        penalized_pairs: List[Tuple[str, str, float]] = []
        seen_validation_signatures = set()
        coverage_hard_reject_count = 0
        duplicate_hard_reject_count = 0
        while attempts < max_attempts and len(pool) < target:
            attempts += 1
            destroy_name, repair_name = self._sample_operator_pair("X")
            attempted_pairs.append((str(destroy_name), str(repair_name)))
            row = self._build_x_exact_candidate(int(iter_id), str(destroy_name), str(repair_name), int(budget))
            if row is None:
                continue
            generated_count += 1
            row_fixes_z = str(row.get("fixgurobi_fixed_scope", "")).upper() in {"Z", "XYZ", "XYZU"}
            if bool(row_fixes_z) and (not bool(row.get("coverage_feasible", True)) or int(row.get("unmet_sku_total", 0) or 0) > 0):
                penalized_pairs.append((str(destroy_name), str(repair_name), -6.0))
                coverage_hard_reject_count += 1
                row["candidate_stage"] = "exact_reject"
                row["exact_fail_reason"] = "coverage_hard_reject"
                row["selected_for_sa"] = False
                candidate_rows.append(row)
                continue
            if bool(row_fixes_z) and int(row["duplicate_tote_count"]) > 0:
                duplicate_hard_reject_count += 1
                row["candidate_stage"] = "exact_reject"
                row["exact_fail_reason"] = "duplicate_tote_hard_reject"
                row["selected_for_sa"] = False
                candidate_rows.append(row)
                continue
            candidate_signature = row["candidate_signature_tuple"]
            if candidate_signature in seen_validation_signatures:
                row["candidate_stage"] = "exact_reject"
                row["exact_fail_reason"] = "duplicate_signature"
                row["selected_for_sa"] = False
                candidate_rows.append(row)
                continue
            seen_validation_signatures.add(candidate_signature)
            candidate_rows.append(row)
            pool.append(row)
        selected = self._select_best_candidate(pool)
        if selected is not None:
            selected["selected_for_sa"] = True
            selected_signature = selected.get("candidate_signature_tuple", None)
            for row in candidate_rows:
                if row.get("candidate_signature_tuple") == selected_signature and str(row.get("candidate_stage", "")) == "exact":
                    row["selected_for_sa"] = True
                    break
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(len(pool)),
            "exact_count": int(len(pool)),
            "rows": candidate_rows,
            "selected": selected,
            "hard_reject_reason": "coverage_hard_reject" if int(coverage_hard_reject_count) > 0 and not pool else "",
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": penalized_pairs,
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
            "exact_fail_count": 0,
            "exact_fail_reasons": {},
            "duplicate_hard_reject_count": int(duplicate_hard_reject_count),
        }

    def _build_xyz_exact_candidate(self, iter_id: int, budget: int) -> Optional[Dict[str, object]]:
        critical_path_mode = bool(getattr(self.cfg, "resource_enable_critical_path_xyz", False))
        if critical_path_mode:
            if bool(getattr(self.cfg, "resource_xyz_use_local_x_window", False)):
                x_destroy = "x_destroy_random_units" if "x_destroy_random_units" in self.operator_arms["X"]["destroy"] else self._sample_operator_pair("X")[0]
                x_repair = "x_repair_affinity_pack" if "x_repair_affinity_pack" in self.operator_arms["X"]["repair"] else self._sample_operator_pair("X")[1]
            else:
                x_destroy = "x_destroy_critical_order_cluster" if "x_destroy_critical_order_cluster" in self.operator_arms["X"]["destroy"] else self._sample_operator_pair("X")[0]
                x_repair = "x_repair_sku_cluster_beam" if "x_repair_sku_cluster_beam" in self.operator_arms["X"]["repair"] else self._sample_operator_pair("X")[1]
            y_destroy = "y_destroy_critical_path_block" if "y_destroy_critical_path_block" in self.operator_arms["Y"]["destroy"] else "y_destroy_max_tardiness_blocker"
            y_repair = "y_repair_ejection_chain_balance" if "y_repair_ejection_chain_balance" in self.operator_arms["Y"]["repair"] else "y_repair_arrival_aware_rank"
            z_destroy = "z_destroy_critical_path_window" if "z_destroy_critical_path_window" in self.operator_arms["Z"]["destroy"] else "z_destroy_shared_stack_window"
            z_repair = "z_repair_stack_mode_joint_polish" if "z_repair_stack_mode_joint_polish" in self.operator_arms["Z"]["repair"] else "z_repair_multistack_cover_compact"
        else:
            x_destroy, x_repair = self._sample_operator_pair("X")
            y_destroy, y_repair = self._sample_operator_pair("Y")
            z_destroy, z_repair = self._sample_operator_pair("Z")
        no_split_full_yz_mode = bool(
            False
            and
            bool(getattr(self.cfg, "sp1_no_split", False))
            and str(getattr(self.cfg, "resource_operator_profile", "") or "").strip().lower() == "no_split_y_focus"
            and "y_destroy_global_station_release" in self.operator_arms["Y"]["destroy"]
            and "y_repair_global_route_balance" in self.operator_arms["Y"]["repair"]
        )
        if no_split_full_yz_mode:
            y_destroy = "y_destroy_global_station_release"
            y_repair = "y_repair_global_route_balance"
            z_destroy = "z_destroy_full_rebuild"
            full_z_strategies = [
                "z_repair_gurobi_like_sort",
                "z_repair_flip_compact",
                "z_repair_multistack_cover_compact",
                "z_repair_greedy_fallback",
            ]
            full_z_pick = int(getattr(self, "no_split_full_z_pick_count", 0))
            self.no_split_full_z_pick_count = int(full_z_pick) + 1
            z_repair = str(full_z_strategies[int(full_z_pick) % len(full_z_strategies)])
        action_signature = (
            "XYZ",
            str(x_destroy),
            str(x_repair),
            str(y_destroy),
            str(y_repair),
            str(z_destroy),
            str(z_repair),
            self.current_config.validation_signature(),
        )
        if self._action_signature_known("XYZ", action_signature) and not bool(critical_path_mode):
            self.last_xyz_skip_reason = "duplicate_action_signature"
            return None

        x_degree = max(1, int(budget))
        local_xyz_mode = bool(getattr(self.cfg, "resource_xyz_use_local_x_window", False))
        local_xyz_degree = max(1, int(getattr(self.cfg, "resource_xyz_local_yz_degree", 3)))
        if bool(getattr(self.cfg, "resource_xyz_use_local_x_window", False)):
            x_degree = max(1, int(getattr(self.cfg, "resource_xyz_local_x_degree", 1)))
        if bool(getattr(self.cfg, "sp1_no_split", False)):
            candidate = self.current_config.clone()
            touched_ids = set()
            fallback_used = False
            projection_count = 0
            projection_mode = "sp1_no_split_skip_x"
        else:
            x_payload = self._apply_x_candidate_to_config(
                base_config=self.current_config,
                base_eval=self.current_eval,
                iter_id=int(iter_id),
                destroy_name=str(x_destroy),
                repair_name=str(x_repair),
                degree=int(x_degree),
            )
            if x_payload is None:
                self.last_xyz_skip_reason = f"x_payload_fail:{x_destroy}+{x_repair}"
                return None
            candidate = x_payload["config"]
            touched_ids = set(int(x) for x in (x_payload.get("affected_ids", set()) or set()))
            fallback_used = bool(x_payload.get("fallback_used", False))
            projection_count = int(x_payload.get("projection_repaired_subtask_count", 0))
            projection_mode = str(x_payload.get("projection_mode", ""))

        yz_degree = int(local_xyz_degree) if bool(local_xyz_mode) else max(1, int(budget))
        y_plan = plan_y_candidate(self.opt, candidate, str(y_destroy), str(y_repair), self.rng, max(1, int(yz_degree)))
        if not bool(y_plan.get("success", False)):
            self.last_xyz_skip_reason = f"y_plan_fail:{y_destroy}+{y_repair}"
            return None
        critical_subtasks = list(y_plan.get("destroy_ctx", {}).get("critical_path_subtask_ids", []) or [])
        y_payload = apply_exact_y_plan(self.opt, candidate, y_plan, self.rng)
        if not bool(y_payload.get("success", False)):
            self.last_xyz_skip_reason = f"y_apply_fail:{y_destroy}+{y_repair}"
            return None
        candidate = y_payload["config"]
        touched_ids.update(int(x) for x in (y_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(fallback_used or y_payload.get("fallback_used", False))

        z_repair_used = str(z_repair)
        if bool(no_split_full_yz_mode):
            candidate = candidate.clone()
            for row in candidate.subtasks.values():
                row.z_tasks = []
            candidate.next_task_id = 0
            used_totes = set()
            rebuilt_ids = set()
            full_z_failed_reason = ""
            for row in sorted(
                candidate.subtasks.values(),
                key=lambda item: (
                    int(item.station_id if int(item.station_id) >= 0 else 10**9),
                    int(item.station_rank if int(item.station_rank) >= 0 else 10**9),
                    int(item.subtask_id),
                ),
            ):
                ok, descriptors, meta = build_full_z_assignment(
                    self.opt,
                    candidate,
                    int(row.subtask_id),
                    preferred_stack_ids=[],
                    strategy=str(z_repair),
                    allow_fallback=True,
                    external_used_totes=set(int(x) for x in used_totes),
                    rng=self.rng,
                )
                if not bool(ok):
                    full_z_failed_reason = f"full_z_fail:{row.subtask_id}:{meta.get('reason', '')}"
                    break
                row.z_tasks = list(descriptors)
                rebuilt_ids.add(int(row.subtask_id))
                fallback_used = bool(fallback_used or meta.get("fallback_used", False))
                for descriptor in descriptors:
                    for tote_id in getattr(descriptor, "target_tote_ids", ()) or ():
                        used_totes.add(int(tote_id))
            if full_z_failed_reason:
                self.last_xyz_skip_reason = full_z_failed_reason
                return None
            candidate.rebuild_indices()
            touched_ids.update(int(x) for x in rebuilt_ids)
            z_repair_used = f"full_z:{z_repair}"
        else:
            z_repair_attempts = [
                str(z_repair),
                "z_repair_multistack_cover_compact",
                "z_repair_flip_compact",
                "z_repair_gurobi_like_sort",
                "z_repair_spread_region_balance",
                "z_repair_greedy_fallback",
            ]
            z_repair_attempts = list(dict.fromkeys(z_repair_attempts))
            z_plan = None
            z_payload = None
            z_failure_reason = ""
            for z_repair_try in z_repair_attempts:
                z_plan_i = plan_z_candidate(self.opt, candidate, str(z_destroy), str(z_repair_try), self.rng, max(1, int(yz_degree)))
                if not bool(z_plan_i.get("success", False)):
                    z_failure_reason = f"plan_fail:{z_destroy}+{z_repair_try}"
                    continue
                z_payload_i = apply_exact_z_plan(self.opt, candidate, z_plan_i, self.rng)
                if not bool(z_payload_i.get("success", False)):
                    z_failure_reason = f"apply_fail:{z_destroy}+{z_repair_try}:{z_payload_i.get('reason', '')}"
                    continue
                z_plan = z_plan_i
                z_payload = z_payload_i
                z_repair_used = str(z_repair_try)
                break
            if z_plan is None or z_payload is None:
                self.last_xyz_skip_reason = f"z_repair_fail:{z_failure_reason}"
                return None
            candidate = z_payload["config"]
            touched_ids.update(int(x) for x in (z_payload.get("affected_ids", set()) or set()))
            fallback_used = bool(fallback_used or z_payload.get("fallback_used", False))
        affected_cap = max(
            1,
            int(
                getattr(
                    self.cfg,
                    "resource_critical_xyz_exact_validation_subtask_cap" if bool(critical_path_mode) else "resource_xyz_exact_validation_subtask_cap",
                    16 if bool(critical_path_mode) else 6,
                )
            ),
        )
        if bool(local_xyz_mode):
            affected_cap = max(1, int(getattr(self.cfg, "resource_local_xyz_exact_validation_subtask_cap", min(4, int(affected_cap)))))
        coverage_seed_limit = max(0, int(getattr(self.cfg, "resource_local_window_coverage_seed_limit", max(4, int(affected_cap) // 2))))
        coverage_issue_ids = self._coverage_issue_subtask_ids(candidate, limit=int(coverage_seed_limit))
        expand_seeds = [int(x) for x in coverage_issue_ids]
        seen_expand = set(int(x) for x in expand_seeds)
        for subtask_id in list(int(x) for x in critical_subtasks):
            if int(subtask_id) not in seen_expand:
                expand_seeds.append(int(subtask_id))
                seen_expand.add(int(subtask_id))
        for subtask_id in sorted(int(x) for x in touched_ids):
            if int(subtask_id) not in seen_expand:
                expand_seeds.append(int(subtask_id))
                seen_expand.add(int(subtask_id))
        if expand_seeds:
            touched_ids = self._expand_local_release_window(
                expand_seeds,
                cap=int(affected_cap),
                radius=int(
                    getattr(
                        self.cfg,
                        "resource_local_xyz_window_neighbor_radius",
                        0 if bool(local_xyz_mode) else getattr(self.cfg, "resource_local_window_neighbor_radius", 1),
                    )
                ),
                config=candidate,
            )
        if len(touched_ids) > int(affected_cap):
            self.last_xyz_skip_reason = f"affected_cap:{len(touched_ids)}>{affected_cap}"
            return None

        payload = {
            "config": candidate,
            "score_cache": None,
            "affected_ids": touched_ids,
            "fallback_used": bool(fallback_used),
            "projection_mode": projection_mode,
            "projection_repaired_subtask_count": int(projection_count),
        }
        candidate_signature = candidate.validation_signature()
        candidate_eval = self._evaluate_config(
            config=candidate,
            layer="XYZ",
            score_cache=None,
            affected_subtask_ids=touched_ids,
            fallback_penalty=0.20 if bool(fallback_used) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(candidate, self.last_validated_config),
        )
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
            }
        )
        local_release_retry_count = 0
        local_release_initial_count = int(len(touched_ids))
        if not math.isfinite(float(candidate_eval.F_raw)):
            for retry_cap in self._local_retry_caps("resource_local_xyz_retry_caps", base_cap=int(affected_cap)):
                retry_ids = self._expand_local_release_window(
                    expand_seeds,
                    cap=int(retry_cap),
                    radius=int(
                        getattr(
                            self.cfg,
                            "resource_local_xyz_window_retry_neighbor_radius",
                            getattr(self.cfg, "resource_local_window_retry_neighbor_radius", 2),
                        )
                    ),
                    config=candidate,
                )
                if retry_ids == touched_ids:
                    continue
                retry_eval = self._evaluate_config(
                    config=candidate,
                    layer="XYZ",
                    score_cache=None,
                    affected_subtask_ids=retry_ids,
                    fallback_penalty=0.20 if bool(fallback_used) else 0.0,
                    iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                    distance_to_last_validated=config_distance(candidate, self.last_validated_config),
                    bypass_cache=True,
                )
                local_release_retry_count += 1
                if math.isfinite(float(retry_eval.F_raw)) or str(retry_eval.metadata.get("fixgurobi_status", "")) != str(candidate_eval.metadata.get("fixgurobi_status", "")):
                    touched_ids = set(int(x) for x in retry_ids)
                    candidate_eval = retry_eval
                if math.isfinite(float(candidate_eval.F_raw)):
                    break
        payload["affected_ids"] = touched_ids
        candidate_eval.metadata.update(
            {
                "local_release_initial_count": int(local_release_initial_count),
                "local_release_retry_count": int(local_release_retry_count),
            }
        )
        return {
            "iter": int(iter_id),
            "layer": "XYZ",
            "candidate_stage": "exact",
            "candidate_rank": 0,
            "destroy_operator": "xyz_destroy_sequential",
            "repair_operator": "xyz_repair_sequential",
            "fallback_used": bool(fallback_used),
            "projection_mode": str(projection_mode),
            "projection_repaired_subtask_count": int(projection_count),
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(action_signature),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
            "xyz_x_operator": f"{x_destroy}+{x_repair}",
            "xyz_y_operator": f"{y_destroy}+{y_repair}",
            "xyz_z_operator": f"{z_destroy}+{z_repair_used}",
            "repartition_mode": bool(str(x_destroy) == "x_destroy_order_repartition"),
            "critical_path_operator_used": bool(critical_path_mode),
            "critical_path_subtask_ids": list(int(x) for x in critical_subtasks),
            "coverage_issue_subtask_ids": list(int(x) for x in coverage_issue_ids),
            "local_release_subtask_count": int(len(touched_ids)),
            "local_release_initial_count": int(local_release_initial_count),
            "local_release_retry_count": int(local_release_retry_count),
        }

    def _build_global_decomp_repair_candidate(self, iter_id: int) -> Optional[Dict[str, object]]:
        if self.global_decomp_repair_used:
            return None
        self.global_decomp_repair_used = True
        target_cmax = float(getattr(self.cfg, "resource_target_cmax", float("nan")))
        total_limit = float(
            getattr(
                self.cfg,
                "resource_global_decomp_repair_time_limit_sec",
                getattr(self.cfg, "fixgurobi_time_limit_sec", 1200.0),
            )
            or getattr(self.cfg, "fixgurobi_time_limit_sec", 1200.0)
        )
        stage_limit = float(getattr(self.cfg, "resource_global_decomp_repair_stage_time_limit_sec", 0.0) or 0.0)
        attempts: List[Dict[str, object]] = []
        if bool(getattr(self.cfg, "resource_global_decomp_repair_staged", True)) and stage_limit > 0.0:
            attempts.append(
                {
                    "name": "narrow",
                    "time_limit": min(float(stage_limit), float(total_limit)),
                    "candidate_stack_topk": int(getattr(self.cfg, "resource_global_decomp_repair_candidate_stack_topk", 3) or 3),
                    "candidate_station_topk_per_stack": int(
                        getattr(self.cfg, "resource_global_decomp_repair_candidate_station_topk_per_stack", 2) or 2
                    ),
                    "max_candidate_stacks_per_order": int(
                        getattr(self.cfg, "resource_global_decomp_repair_max_candidate_stacks_per_order", 24) or 24
                    ),
                }
            )
        attempts.append(
            {
                "name": "full",
                "time_limit": float(total_limit),
                "candidate_stack_topk": 999,
                "candidate_station_topk_per_stack": 999,
                "max_candidate_stacks_per_order": 0,
            }
        )
        t0 = time.perf_counter()
        attempt_rows: List[Dict[str, object]] = []
        result = None
        problem_scale = str(getattr(self.cfg, "scale", "") or getattr(getattr(self.opt, "problem", None), "scale_name", "") or "")
        problem_seed = int(getattr(self.cfg, "seed", 42) or 42)
        for attempt in attempts:
            elapsed = float(time.perf_counter() - t0)
            remaining = max(1.0, float(total_limit) - float(elapsed))
            cfg = GlobalXYZUConfig(
                time_limit_sec=min(float(attempt["time_limit"]), remaining),
                mip_gap=float(getattr(self.cfg, "fixgurobi_mip_gap", 0.01) or 0.01),
                candidate_stack_topk=int(attempt["candidate_stack_topk"]),
                max_candidate_stacks_per_order=int(attempt["max_candidate_stacks_per_order"]),
                enable_warm_candidate_stack_prune=False,
                candidate_station_topk_per_stack=int(attempt["candidate_station_topk_per_stack"]),
                route_pickup_neighbor_limit=int(getattr(self.cfg, "resource_global_decomp_repair_route_pickup_neighbor_limit", 0) or 0),
                enable_scale_adaptive_candidate_prune=False,
                gurobi_output=bool(getattr(self.cfg, "fixgurobi_output", False)),
                enable_warm_start=False,
                warm_start_use_sp4=False,
                integrate_u_route=True,
                route_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_arc_prune", True)),
                enable_route_time_window_arc_prune=bool(
                    getattr(self.cfg, "resource_global_decomp_repair_route_time_window_arc_prune", True)
                ),
                enable_route_load_interval_arc_prune=bool(getattr(self.cfg, "fixgurobi_route_load_interval_arc_prune", True)),
                enable_route_directional_arc_prune=False,
                enable_resource_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
                enable_slot_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
                enable_tote_equivalence_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
                enable_station_global_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
                enable_robot_finish_lex_symmetry=bool(getattr(self.cfg, "fixgurobi_enable_symmetry", True)),
                enable_selected_workload_lbs=True,
                fixgurobi_no_warm_start=True,
                fixgurobi_allow_warm_start_fallback=False,
                fixgurobi_warm_bound_only=False,
            )
            if bool(getattr(self.cfg, "resource_global_decomp_repair_best_obj_stop", True)) and math.isfinite(target_cmax):
                slack = float(getattr(self.cfg, "resource_global_decomp_repair_obj_slack", 0.999) or 0.999)
                cfg.gurobi_best_obj_stop = float(target_cmax + slack)
            attempt_t0 = time.perf_counter()
            try:
                if bool(getattr(self.cfg, "resource_global_decomp_repair_use_fresh_problem", True)) and problem_scale:
                    repair_problem = CreateOFSProblem.generate_problem_by_scale(problem_scale, seed=problem_seed)
                else:
                    repair_problem = copy.deepcopy(self.opt.problem)
                result = GlobalXYZUSolver().solve(repair_problem, cfg)
            except Exception as exc:
                attempt_rows.append({"attempt": str(attempt["name"]), "runtime_sec": float(time.perf_counter() - attempt_t0), "status": "EXCEPTION", "reason": str(exc)})
                continue
            diag_i = dict(getattr(result, "diagnostics", {}) or {})
            cmax_i = float(diag_i.get("model_cmax", getattr(result, "objective", float("inf"))) or float("inf"))
            attempt_rows.append(
                {
                    "attempt": str(attempt["name"]),
                    "runtime_sec": float(time.perf_counter() - attempt_t0),
                    "status": str(getattr(result, "status", "")),
                    "cmax": float(cmax_i),
                    "gap": float(getattr(result, "gap", float("nan"))),
                }
            )
            if math.isfinite(cmax_i) and (not math.isfinite(target_cmax) or cmax_i <= target_cmax + 1e-9):
                break
        repair_runtime = float(time.perf_counter() - t0)
        if result is None:
            self.last_xyz_skip_reason = "global_decomp_repair_no_attempt_result"
            return None
        diag = dict(getattr(result, "diagnostics", {}) or {})
        materialized_problem = getattr(result, "materialized_problem", None)
        if materialized_problem is None or not math.isfinite(float(diag.get("model_cmax", getattr(result, "objective", float("inf"))) or float("inf"))):
            self.last_xyz_skip_reason = f"global_decomp_repair_no_solution:{getattr(result, 'status', '')}"
            return None
        candidate = build_resource_config_from_problem(self.opt, materialized_problem)
        candidate.metadata["global_decomp_repair"] = True
        candidate_signature = candidate.validation_signature()
        touched_ids = set(int(x) for x in candidate.subtasks.keys())
        candidate_eval = self._evaluate_config(
            config=candidate,
            layer="XYZ",
            score_cache=None,
            affected_subtask_ids=touched_ids,
            fallback_penalty=0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(candidate, self.last_validated_config),
        )
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
                "global_decomp_repair_status": str(getattr(result, "status", "")),
                "global_decomp_repair_runtime_sec": float(repair_runtime),
                "global_decomp_repair_model_cmax": float(diag.get("model_cmax", getattr(result, "objective", float("nan")))),
                "global_decomp_repair_gap": float(getattr(result, "gap", float("nan"))),
                "global_decomp_repair_attempts": list(attempt_rows),
            }
        )
        payload = {
            "config": candidate,
            "score_cache": None,
            "affected_ids": touched_ids,
            "fallback_used": False,
            "projection_mode": "global_decomp_repair",
            "projection_repaired_subtask_count": int(len(touched_ids)),
        }
        return {
            "iter": int(iter_id),
            "layer": "XYZ",
            "candidate_stage": "global_decomp_repair",
            "candidate_rank": 0,
            "destroy_operator": "xyz_destroy_sequential",
            "repair_operator": "xyz_repair_sequential",
            "fallback_used": False,
            "projection_mode": "global_decomp_repair",
            "projection_repaired_subtask_count": int(len(touched_ids)),
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(("XYZ", "global_decomp_repair", int(iter_id))),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
            "xyz_x_operator": "global_decomp_repair",
            "xyz_y_operator": "global_decomp_repair",
            "xyz_z_operator": "global_decomp_repair",
            "repartition_mode": True,
            "critical_path_operator_used": False,
            "critical_path_subtask_ids": [],
            "global_decomp_repair": True,
            "global_decomp_repair_runtime_sec": float(repair_runtime),
            "global_decomp_repair_status": str(getattr(result, "status", "")),
            "global_decomp_repair_attempts": str(attempt_rows),
        }

    def _build_xz_exact_candidate(self, iter_id: int, budget: int) -> Optional[Dict[str, object]]:
        if bool(getattr(self.cfg, "sp1_no_split", False)):
            self.last_xyz_skip_reason = "xz_disabled_sp1_no_split"
            return None
        x_destroy, x_repair = self._sample_operator_pair("X")
        z_destroy, z_repair = self._sample_operator_pair("Z")
        action_signature = (
            "XZ",
            str(x_destroy),
            str(x_repair),
            str(z_destroy),
            str(z_repair),
            self.current_config.validation_signature(),
        )
        if self._action_signature_known("XZ", action_signature):
            self.last_xyz_skip_reason = "xz_duplicate_action_signature"
            return None
        self._remember_action_signature("XZ", action_signature)

        x_payload = self._apply_x_candidate_to_config(
            base_config=self.current_config,
            base_eval=self.current_eval,
            iter_id=int(iter_id),
            destroy_name=str(x_destroy),
            repair_name=str(x_repair),
            degree=max(1, int(budget)),
        )
        if x_payload is None:
            self.last_xyz_skip_reason = f"xz_x_payload_fail:{x_destroy}+{x_repair}"
            return None
        candidate = x_payload["config"]
        touched_ids = set(int(x) for x in (x_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(x_payload.get("fallback_used", False))
        projection_count = int(x_payload.get("projection_repaired_subtask_count", 0))
        projection_mode = str(x_payload.get("projection_mode", ""))

        z_budget = max(1, int(budget), len(touched_ids))
        z_plan = plan_z_candidate(self.opt, candidate, str(z_destroy), str(z_repair), self.rng, int(z_budget))
        if not bool(z_plan.get("success", False)):
            self.last_xyz_skip_reason = f"xz_z_plan_fail:{z_destroy}+{z_repair}"
            return None
        z_payload = apply_exact_z_plan(self.opt, candidate, z_plan, self.rng)
        if not bool(z_payload.get("success", False)):
            self.last_xyz_skip_reason = f"xz_z_apply_fail:{z_destroy}+{z_repair}:{z_payload.get('reason', '')}"
            return None
        candidate = z_payload["config"]
        touched_ids.update(int(x) for x in (z_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(fallback_used or z_payload.get("fallback_used", False))

        affected_cap = max(1, int(getattr(self.cfg, "resource_xz_exact_validation_subtask_cap", 16)))
        if len(touched_ids) > int(affected_cap):
            self.last_xyz_skip_reason = f"xz_affected_cap:{len(touched_ids)}>{affected_cap}"
            return None

        payload = {
            "config": candidate,
            "score_cache": None,
            "affected_ids": touched_ids,
            "fallback_used": bool(fallback_used),
            "projection_mode": projection_mode,
            "projection_repaired_subtask_count": int(projection_count),
        }
        candidate_signature = candidate.validation_signature()
        candidate_eval = self._evaluate_config(
            config=candidate,
            layer="XZ",
            score_cache=None,
            affected_subtask_ids=touched_ids,
            fallback_penalty=0.20 if bool(fallback_used) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(candidate, self.last_validated_config),
        )
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
            }
        )
        return {
            "iter": int(iter_id),
            "layer": "XZ",
            "candidate_stage": "exact",
            "candidate_rank": 0,
            "destroy_operator": "xz_destroy_joint",
            "repair_operator": "xz_repair_joint",
            "fallback_used": bool(fallback_used),
            "projection_mode": str(projection_mode),
            "projection_repaired_subtask_count": int(projection_count),
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(action_signature),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
            "xz_x_operator": f"{x_destroy}+{x_repair}",
            "xz_z_operator": f"{z_destroy}+{z_repair}",
            "revolving_enabled": bool(candidate_eval.metadata.get("revolving_enabled", False)),
            "released_layer": str(candidate_eval.metadata.get("released_layer", "")),
            "fixed_layers": str(candidate_eval.metadata.get("fixed_layers", "")),
            "inner_relaxed_obj": candidate_eval.metadata.get("inner_relaxed_obj", ""),
            "u_fast_cmax": candidate_eval.metadata.get("u_fast_cmax", ""),
            "u_route_lb": candidate_eval.metadata.get("u_route_lb", ""),
            "u_repair_time": candidate_eval.metadata.get("u_repair_time", ""),
            "u_changed_robot_count": candidate_eval.metadata.get("u_changed_robot_count", ""),
            "revolving_lb": candidate_eval.metadata.get("revolving_lb", ""),
            "lb_gate_skipped": candidate_eval.metadata.get("lb_gate_skipped", ""),
        }

    def _generate_xz_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(getattr(self.cfg, "resource_xz_candidate_pool_size", target_size)))
        max_attempts = max(target, int(getattr(self.cfg, "resource_xz_candidate_pool_max_attempts", getattr(self.cfg, "resource_candidate_pool_max_attempts", 12))))
        attempts = 0
        generated_count = 0
        pool: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        coverage_hard_reject_count = 0
        duplicate_hard_reject_count = 0
        seen_validation_signatures = set()
        while attempts < max_attempts and len(pool) < target:
            attempts += 1
            attempted_pairs.append(("xz_destroy_joint", "xz_repair_joint"))
            row = self._build_xz_exact_candidate(int(iter_id), int(budget))
            if row is None:
                continue
            generated_count += 1
            fixed_scope = str(row.get("fixgurobi_fixed_scope", "") or "").upper()
            row_fixes_z = "Z" in fixed_scope and fixed_scope not in {"LOCALXYZ", "LOCALYZ"}
            if bool(row_fixes_z) and (not bool(row.get("coverage_feasible", True)) or int(row.get("unmet_sku_total", 0) or 0) > 0):
                coverage_hard_reject_count += 1
                continue
            if bool(row_fixes_z) and int(row["duplicate_tote_count"]) > 0:
                duplicate_hard_reject_count += 1
                continue
            candidate_signature = row["candidate_signature_tuple"]
            if candidate_signature in seen_validation_signatures:
                continue
            seen_validation_signatures.add(candidate_signature)
            pool.append(row)
        selected = self._select_best_candidate(pool)
        if selected is not None:
            selected["selected_for_sa"] = True
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(len(pool)),
            "exact_count": int(len(pool)),
            "rows": pool,
            "selected": selected,
            "hard_reject_reason": "coverage_hard_reject" if int(coverage_hard_reject_count) > 0 and not pool else "",
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": [],
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
            "exact_fail_count": 0,
            "exact_fail_reasons": {},
            "duplicate_hard_reject_count": int(duplicate_hard_reject_count),
        }

    def _build_yz_exact_candidate(self, iter_id: int, budget: int) -> Optional[Dict[str, object]]:
        y_destroy, y_repair = self._sample_operator_pair("Y")
        z_destroy, z_repair = self._sample_operator_pair("Z")
        action_signature = (
            "YZ",
            str(y_destroy),
            str(y_repair),
            str(z_destroy),
            str(z_repair),
            self.current_config.validation_signature(),
        )
        if self._action_signature_known("YZ", action_signature):
            self.last_xyz_skip_reason = "yz_duplicate_action_signature"
            return None
        self._remember_action_signature("YZ", action_signature)

        y_plan = plan_y_candidate(self.opt, self.current_config, str(y_destroy), str(y_repair), self.rng, max(1, int(budget)))
        if not bool(y_plan.get("success", False)):
            self.last_xyz_skip_reason = f"yz_y_plan_fail:{y_destroy}+{y_repair}"
            return None
        y_payload = apply_exact_y_plan(self.opt, self.current_config, y_plan, self.rng)
        if not bool(y_payload.get("success", False)):
            self.last_xyz_skip_reason = f"yz_y_apply_fail:{y_destroy}+{y_repair}"
            return None
        candidate = y_payload["config"]
        touched_ids = set(int(x) for x in (y_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(y_payload.get("fallback_used", False))

        z_budget = max(1, int(budget), len(touched_ids))
        z_plan = plan_z_candidate(self.opt, candidate, str(z_destroy), str(z_repair), self.rng, int(z_budget))
        if not bool(z_plan.get("success", False)):
            self.last_xyz_skip_reason = f"yz_z_plan_fail:{z_destroy}+{z_repair}"
            return None
        z_payload = apply_exact_z_plan(self.opt, candidate, z_plan, self.rng)
        if not bool(z_payload.get("success", False)):
            self.last_xyz_skip_reason = f"yz_z_apply_fail:{z_destroy}+{z_repair}:{z_payload.get('reason', '')}"
            return None
        candidate = z_payload["config"]
        touched_ids.update(int(x) for x in (z_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(fallback_used or z_payload.get("fallback_used", False))

        affected_cap = max(1, int(getattr(self.cfg, "resource_yz_exact_validation_subtask_cap", 18)))
        coverage_seed_limit = max(
            0,
            int(
                getattr(
                    self.cfg,
                    "resource_local_yz_window_coverage_seed_limit",
                    getattr(self.cfg, "resource_local_window_coverage_seed_limit", max(4, int(affected_cap) // 2)),
                )
            ),
        )
        coverage_issue_ids = self._coverage_issue_subtask_ids(candidate, limit=int(coverage_seed_limit))
        expand_seeds = [int(x) for x in coverage_issue_ids]
        if touched_ids:
            seen_expand = set(int(x) for x in expand_seeds)
            for subtask_id in sorted(int(x) for x in touched_ids):
                if int(subtask_id) not in seen_expand:
                    expand_seeds.append(int(subtask_id))
                    seen_expand.add(int(subtask_id))
            touched_ids = self._expand_local_release_window(
                expand_seeds,
                cap=int(affected_cap),
                radius=int(getattr(self.cfg, "resource_local_yz_window_neighbor_radius", getattr(self.cfg, "resource_local_window_neighbor_radius", 1))),
                config=candidate,
            )
        if len(touched_ids) > int(affected_cap):
            self.last_xyz_skip_reason = f"yz_affected_cap:{len(touched_ids)}>{affected_cap}"
            return None

        payload = {
            "config": candidate,
            "score_cache": None,
            "affected_ids": touched_ids,
            "fallback_used": bool(fallback_used),
            "projection_mode": "",
            "projection_repaired_subtask_count": 0,
            "coverage_issue_ids": list(int(x) for x in coverage_issue_ids),
        }
        candidate_signature = candidate.validation_signature()
        candidate_eval = self._evaluate_config(
            config=candidate,
            layer="YZ",
            score_cache=None,
            affected_subtask_ids=touched_ids,
            fallback_penalty=0.20 if bool(fallback_used) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(candidate, self.last_validated_config),
        )
        local_release_retry_count = 0
        local_release_initial_count = int(len(touched_ids))
        if not math.isfinite(float(candidate_eval.F_raw)):
            for retry_cap in self._local_retry_caps("resource_local_yz_retry_caps", base_cap=int(affected_cap)):
                retry_ids = self._expand_local_release_window(
                    expand_seeds,
                    cap=int(retry_cap),
                    radius=int(getattr(self.cfg, "resource_local_yz_window_retry_neighbor_radius", getattr(self.cfg, "resource_local_window_retry_neighbor_radius", 2))),
                    config=candidate,
                )
                if retry_ids == touched_ids:
                    continue
                retry_eval = self._evaluate_config(
                    config=candidate,
                    layer="YZ",
                    score_cache=None,
                    affected_subtask_ids=retry_ids,
                    fallback_penalty=0.20 if bool(fallback_used) else 0.0,
                    iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                    distance_to_last_validated=config_distance(candidate, self.last_validated_config),
                    bypass_cache=True,
                )
                local_release_retry_count += 1
                if math.isfinite(float(retry_eval.F_raw)) or str(retry_eval.metadata.get("fixgurobi_status", "")) != str(candidate_eval.metadata.get("fixgurobi_status", "")):
                    touched_ids = set(int(x) for x in retry_ids)
                    candidate_eval = retry_eval
                if math.isfinite(float(candidate_eval.F_raw)):
                    break
        payload["affected_ids"] = touched_ids
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
                "local_release_initial_count": int(local_release_initial_count),
                "local_release_retry_count": int(local_release_retry_count),
            }
        )
        return {
            "iter": int(iter_id),
            "layer": "YZ",
            "candidate_stage": "exact",
            "candidate_rank": 0,
            "destroy_operator": "yz_destroy_joint",
            "repair_operator": "yz_repair_joint",
            "fallback_used": bool(fallback_used),
            "projection_mode": "",
            "projection_repaired_subtask_count": 0,
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(action_signature),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
            "yz_y_operator": f"{y_destroy}+{y_repair}",
            "yz_z_operator": f"{z_destroy}+{z_repair}",
            "revolving_enabled": bool(candidate_eval.metadata.get("revolving_enabled", False)),
            "released_layer": str(candidate_eval.metadata.get("released_layer", "")),
            "fixed_layers": str(candidate_eval.metadata.get("fixed_layers", "")),
            "inner_relaxed_obj": candidate_eval.metadata.get("inner_relaxed_obj", ""),
            "u_fast_cmax": candidate_eval.metadata.get("u_fast_cmax", ""),
            "u_route_lb": candidate_eval.metadata.get("u_route_lb", ""),
            "u_repair_time": candidate_eval.metadata.get("u_repair_time", ""),
            "u_changed_robot_count": candidate_eval.metadata.get("u_changed_robot_count", ""),
            "revolving_lb": candidate_eval.metadata.get("revolving_lb", ""),
            "lb_gate_skipped": candidate_eval.metadata.get("lb_gate_skipped", ""),
            "coverage_issue_subtask_ids": list(int(x) for x in coverage_issue_ids),
            "local_release_subtask_count": int(len(touched_ids)),
            "local_release_initial_count": int(local_release_initial_count),
            "local_release_retry_count": int(local_release_retry_count),
        }

    def _generate_yz_joint_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(getattr(self.cfg, "resource_yz_candidate_pool_size", target_size)))
        max_attempts = max(target, int(getattr(self.cfg, "resource_yz_candidate_pool_max_attempts", getattr(self.cfg, "resource_candidate_pool_max_attempts", 12))))
        attempts = 0
        generated_count = 0
        pool: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        coverage_hard_reject_count = 0
        duplicate_hard_reject_count = 0
        seen_validation_signatures = set()
        while attempts < max_attempts and len(pool) < target:
            attempts += 1
            attempted_pairs.append(("yz_destroy_joint", "yz_repair_joint"))
            row = self._build_yz_exact_candidate(int(iter_id), int(budget))
            if row is None:
                continue
            generated_count += 1
            fixed_scope = str(row.get("fixgurobi_fixed_scope", "") or "").upper()
            row_fixes_z = "Z" in fixed_scope and fixed_scope not in {"LOCALXYZ", "LOCALYZ"}
            if bool(row_fixes_z) and (not bool(row.get("coverage_feasible", True)) or int(row.get("unmet_sku_total", 0) or 0) > 0):
                coverage_hard_reject_count += 1
                continue
            if bool(row_fixes_z) and int(row["duplicate_tote_count"]) > 0:
                duplicate_hard_reject_count += 1
                continue
            candidate_signature = row["candidate_signature_tuple"]
            if candidate_signature in seen_validation_signatures:
                continue
            seen_validation_signatures.add(candidate_signature)
            pool.append(row)
        selected = self._select_best_candidate(pool)
        if selected is not None:
            selected["selected_for_sa"] = True
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(len(pool)),
            "exact_count": int(len(pool)),
            "rows": pool,
            "selected": selected,
            "hard_reject_reason": "coverage_hard_reject" if int(coverage_hard_reject_count) > 0 and not pool else "",
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": [],
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
            "exact_fail_count": 0,
            "exact_fail_reasons": {},
            "duplicate_hard_reject_count": int(duplicate_hard_reject_count),
        }

    def _build_xy_exact_candidate(self, iter_id: int, budget: int) -> Optional[Dict[str, object]]:
        x_destroy, x_repair = self._sample_operator_pair("X")
        y_destroy, y_repair = self._sample_operator_pair("Y")
        action_signature = (
            "XY",
            str(x_destroy),
            str(x_repair),
            str(y_destroy),
            str(y_repair),
            self.current_config.validation_signature(),
        )
        if self._action_signature_known("XY", action_signature):
            self.last_xyz_skip_reason = "xy_duplicate_action_signature"
            return None
        self._remember_action_signature("XY", action_signature)

        x_payload = self._apply_x_candidate_to_config(
            base_config=self.current_config,
            base_eval=self.current_eval,
            iter_id=int(iter_id),
            destroy_name=str(x_destroy),
            repair_name=str(x_repair),
            degree=max(1, int(budget)),
        )
        if x_payload is None:
            self.last_xyz_skip_reason = f"xy_x_payload_fail:{x_destroy}+{x_repair}"
            return None
        candidate = x_payload["config"]
        touched_ids = set(int(x) for x in (x_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(x_payload.get("fallback_used", False))
        projection_count = int(x_payload.get("projection_repaired_subtask_count", 0))
        projection_mode = str(x_payload.get("projection_mode", ""))

        y_budget = max(1, int(budget), len(touched_ids))
        y_plan = plan_y_candidate(self.opt, candidate, str(y_destroy), str(y_repair), self.rng, int(y_budget))
        if not bool(y_plan.get("success", False)):
            self.last_xyz_skip_reason = f"xy_y_plan_fail:{y_destroy}+{y_repair}"
            return None
        y_payload = apply_exact_y_plan(self.opt, candidate, y_plan, self.rng)
        if not bool(y_payload.get("success", False)):
            self.last_xyz_skip_reason = f"xy_y_apply_fail:{y_destroy}+{y_repair}"
            return None
        candidate = y_payload["config"]
        touched_ids.update(int(x) for x in (y_payload.get("affected_ids", set()) or set()))
        fallback_used = bool(fallback_used or y_payload.get("fallback_used", False))

        affected_cap = max(1, int(getattr(self.cfg, "resource_xy_exact_validation_subtask_cap", 18)))
        if len(touched_ids) > int(affected_cap):
            self.last_xyz_skip_reason = f"xy_affected_cap:{len(touched_ids)}>{affected_cap}"
            return None

        payload = {
            "config": candidate,
            "score_cache": None,
            "affected_ids": touched_ids,
            "fallback_used": bool(fallback_used),
            "projection_mode": projection_mode,
            "projection_repaired_subtask_count": int(projection_count),
        }
        candidate_signature = candidate.validation_signature()
        candidate_eval = self._evaluate_config(
            config=candidate,
            layer="XY",
            score_cache=None,
            affected_subtask_ids=touched_ids,
            fallback_penalty=0.20 if bool(fallback_used) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(candidate, self.last_validated_config),
        )
        candidate_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
            }
        )
        return {
            "iter": int(iter_id),
            "layer": "XY",
            "candidate_stage": "exact",
            "candidate_rank": 0,
            "destroy_operator": "xy_destroy_joint",
            "repair_operator": "xy_repair_joint",
            "fallback_used": bool(fallback_used),
            "projection_mode": str(projection_mode),
            "projection_repaired_subtask_count": int(projection_count),
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "candidate_signature": self._candidate_signature_text(candidate_signature),
            "candidate_signature_tuple": candidate_signature,
            "candidate_payload": payload,
            "candidate_eval": candidate_eval,
            "selected_for_sa": False,
            "action_signature": self._action_signature_text(action_signature),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
            "xy_x_operator": f"{x_destroy}+{x_repair}",
            "xy_y_operator": f"{y_destroy}+{y_repair}",
            "revolving_enabled": bool(candidate_eval.metadata.get("revolving_enabled", False)),
            "released_layer": str(candidate_eval.metadata.get("released_layer", "")),
            "fixed_layers": str(candidate_eval.metadata.get("fixed_layers", "")),
            "inner_relaxed_obj": candidate_eval.metadata.get("inner_relaxed_obj", ""),
            "u_fast_cmax": candidate_eval.metadata.get("u_fast_cmax", ""),
            "u_route_lb": candidate_eval.metadata.get("u_route_lb", ""),
            "u_repair_time": candidate_eval.metadata.get("u_repair_time", ""),
            "u_changed_robot_count": candidate_eval.metadata.get("u_changed_robot_count", ""),
            "revolving_lb": candidate_eval.metadata.get("revolving_lb", ""),
            "lb_gate_skipped": candidate_eval.metadata.get("lb_gate_skipped", ""),
        }

    def _generate_xy_joint_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(getattr(self.cfg, "resource_xy_candidate_pool_size", target_size)))
        max_attempts = max(target, int(getattr(self.cfg, "resource_xy_candidate_pool_max_attempts", getattr(self.cfg, "resource_candidate_pool_max_attempts", 12))))
        attempts = 0
        generated_count = 0
        pool: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        coverage_hard_reject_count = 0
        duplicate_hard_reject_count = 0
        seen_validation_signatures = set()
        while attempts < max_attempts and len(pool) < target:
            attempts += 1
            attempted_pairs.append(("xy_destroy_joint", "xy_repair_joint"))
            row = self._build_xy_exact_candidate(int(iter_id), int(budget))
            if row is None:
                continue
            generated_count += 1
            fixed_scope = str(row.get("fixgurobi_fixed_scope", "") or "").upper()
            row_fixes_z = "Z" in fixed_scope and fixed_scope not in {"LOCALXYZ", "LOCALYZ"}
            if bool(row_fixes_z) and (not bool(row.get("coverage_feasible", True)) or int(row.get("unmet_sku_total", 0) or 0) > 0):
                coverage_hard_reject_count += 1
                continue
            if bool(row_fixes_z) and int(row["duplicate_tote_count"]) > 0:
                duplicate_hard_reject_count += 1
                continue
            candidate_signature = row["candidate_signature_tuple"]
            if candidate_signature in seen_validation_signatures:
                continue
            seen_validation_signatures.add(candidate_signature)
            pool.append(row)
        selected = self._select_best_candidate(pool)
        if selected is not None:
            selected["selected_for_sa"] = True
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(len(pool)),
            "exact_count": int(len(pool)),
            "rows": pool,
            "selected": selected,
            "hard_reject_reason": "coverage_hard_reject" if int(coverage_hard_reject_count) > 0 and not pool else "",
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": [],
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
            "exact_fail_count": 0,
            "exact_fail_reasons": {},
            "duplicate_hard_reject_count": int(duplicate_hard_reject_count),
        }

    def _generate_xyz_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(getattr(self.cfg, "resource_xyz_candidate_pool_size", target_size)))
        max_attempts = max(target, int(getattr(self.cfg, "resource_xyz_candidate_pool_max_attempts", 12)))
        attempts = 0
        generated_count = 0
        pool: List[Dict[str, object]] = []
        candidate_rows: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        coverage_hard_reject_count = 0
        duplicate_hard_reject_count = 0
        seen_validation_signatures = set()
        if bool(getattr(self.cfg, "resource_global_decomp_repair_enabled", False)):
            row = self._build_global_decomp_repair_candidate(int(iter_id))
            if row is not None:
                generated_count += 1
                if bool(row.get("coverage_feasible", True)) and int(row.get("unmet_sku_total", 0) or 0) <= 0 and int(row["duplicate_tote_count"]) <= 0:
                    candidate_signature = row["candidate_signature_tuple"]
                    seen_validation_signatures.add(candidate_signature)
                    pool.append(row)
        while attempts < max_attempts and len(pool) < target:
            attempts += 1
            attempted_pairs.append(("xyz_destroy_sequential", "xyz_repair_sequential"))
            row = self._build_xyz_exact_candidate(int(iter_id), int(budget))
            if row is None:
                continue
            generated_count += 1
            fixed_scope = str(row.get("fixgurobi_fixed_scope", "") or "").upper()
            row_fixes_z = "Z" in fixed_scope and fixed_scope not in {"LOCALXYZ", "LOCALYZ"}
            if bool(row_fixes_z) and (not bool(row.get("coverage_feasible", True)) or int(row.get("unmet_sku_total", 0) or 0) > 0):
                coverage_hard_reject_count += 1
                row["candidate_stage"] = "exact_reject"
                row["exact_fail_reason"] = "coverage_hard_reject"
                candidate_rows.append(row)
                continue
            if bool(row_fixes_z) and int(row["duplicate_tote_count"]) > 0:
                duplicate_hard_reject_count += 1
                row["candidate_stage"] = "exact_reject"
                row["exact_fail_reason"] = "duplicate_tote_hard_reject"
                candidate_rows.append(row)
                continue
            candidate_signature = row["candidate_signature_tuple"]
            if candidate_signature in seen_validation_signatures:
                row["candidate_stage"] = "exact_reject"
                row["exact_fail_reason"] = "duplicate_signature"
                candidate_rows.append(row)
                continue
            seen_validation_signatures.add(candidate_signature)
            candidate_rows.append(row)
            pool.append(row)
        selected = self._select_best_candidate(pool)
        if selected is not None:
            selected["selected_for_sa"] = True
            selected_signature = selected.get("candidate_signature_tuple", None)
            for row in candidate_rows:
                if row.get("candidate_signature_tuple") == selected_signature and str(row.get("candidate_stage", "")) == "exact":
                    row["selected_for_sa"] = True
                    break
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(len(pool)),
            "exact_count": int(len(pool)),
            "rows": candidate_rows,
            "selected": selected,
            "hard_reject_reason": "coverage_hard_reject" if int(coverage_hard_reject_count) > 0 and not pool else "",
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": [],
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
            "exact_fail_count": 0,
            "exact_fail_reasons": {},
            "duplicate_hard_reject_count": int(duplicate_hard_reject_count),
        }

    def _build_u_exact_candidate(self, iter_id: int) -> Optional[Dict[str, object]]:
        current_best = float(getattr(self.best_validated, "makespan", float("inf")) or float("inf"))
        revolving = self.revolving_solver.attach_u_plan(
            self.current_config,
            released_layer="U",
            affected_subtask_ids=[],
            incumbent_value=float(current_best),
        )
        if bool(revolving.lb_gate_skipped):
            return None
        candidate_config = revolving.config
        candidate_eval = self._evaluate_config(
            config=candidate_config,
            layer="U",
            score_cache=None,
            affected_subtask_ids=[],
            fallback_penalty=0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(candidate_config, self.last_validated_config),
            bypass_cache=True,
        )
        candidate_eval.metadata.update(revolving.metadata())
        payload = {"config": candidate_config, "affected_ids": set(), "score_cache": None}
        route_sig = tuple(sorted(int(x) for x in (candidate_config.metadata.get("changed_robot_ids", []) or [])))
        row = {
            "iter": int(iter_id),
            "layer": "U",
            "candidate_rank": 1,
            "candidate_stage": "exact",
            "destroy_operator": "u_destroy_heavy_robot_route",
            "repair_operator": "u_fast_repair_dispatch",
            "fallback_used": False,
            "projection_mode": "",
            "projection_repaired_subtask_count": 0,
            "candidate_eval": candidate_eval,
            "candidate_payload": payload,
            "F_raw": float(candidate_eval.F_raw),
            "F_cal": float(candidate_eval.F_cal),
            "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
            "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
            "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
            "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
            "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
            "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
            "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
            "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
            "fixgurobi_cache_hit": bool(candidate_eval.metadata.get("fixgurobi_cache_hit", False)),
            "fixgurobi_compile_cache_hit": bool(candidate_eval.metadata.get("fixgurobi_compile_cache_hit", False)),
            "fixgurobi_compile_time": candidate_eval.metadata.get("fixgurobi_compile_time", ""),
            "fixgurobi_stage": str(candidate_eval.metadata.get("fixgurobi_stage", "")),
            "fixgurobi_cutoff": candidate_eval.metadata.get("fixgurobi_cutoff", ""),
            "fixgurobi_refined": bool(candidate_eval.metadata.get("fixgurobi_refined", False)),
            "fixgurobi_coarse_time": candidate_eval.metadata.get("fixgurobi_coarse_time", ""),
            "fixgurobi_refine_time": candidate_eval.metadata.get("fixgurobi_refine_time", ""),
            "fixgurobi_full_time": candidate_eval.metadata.get("fixgurobi_full_time", ""),
            "fixgurobi_first_improvement_accepted": bool(candidate_eval.metadata.get("fixgurobi_first_improvement_accepted", False)),
            "fixgurobi_cheap_gate_reject": bool(candidate_eval.metadata.get("fixgurobi_cheap_gate_reject", False)),
            "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
            "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
            "coverage_feasible": bool(candidate_eval.coverage_feasible),
            "unmet_sku_total": int(candidate_eval.unmet_sku_total),
            "candidate_signature_tuple": candidate_config.validation_signature(),
            "candidate_signature": self._candidate_signature_text(("U", candidate_config.validation_signature(), route_sig)),
            "action_signature": self._action_signature_text(("U", "u_fast_repair_dispatch", int(iter_id), route_sig)),
            "selected_for_sa": True,
            "revolving_enabled": True,
            "released_layer": "U",
            "fixed_layers": str(candidate_config.metadata.get("fixed_layers", "X,Y,Z")),
            "inner_relaxed_obj": float(candidate_config.metadata.get("inner_relaxed_obj", float("nan"))),
            "u_fast_cmax": float(candidate_config.metadata.get("u_fast_cmax", float("nan"))),
            "u_route_lb": float(candidate_config.metadata.get("u_route_lb", float("nan"))),
            "u_repair_time": float(candidate_config.metadata.get("u_repair_time", 0.0) or 0.0),
            "u_changed_robot_count": int(candidate_config.metadata.get("u_changed_robot_count", 0) or 0),
            "revolving_lb": float(candidate_config.metadata.get("revolving_lb", float("nan"))),
            "lb_gate_skipped": False,
        }
        return row

    def _generate_u_candidate_pool(self, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        del budget, target_size
        row = self._build_u_exact_candidate(int(iter_id))
        if row is None:
            return {
                "target_size": 1,
                "attempt_count": 1,
                "generated_count": 0,
                "unique_count": 0,
                "exact_count": 0,
                "rows": [],
                "selected": None,
                "hard_reject_reason": "u_revolving_lb_gate_skip",
                "attempted_pairs": [("u_destroy_heavy_robot_route", "u_fast_repair_dispatch")],
                "penalized_pairs": [],
                "coverage_hard_reject_count": 0,
                "exact_fail_count": 0,
                "exact_fail_reasons": {},
                "duplicate_hard_reject_count": 0,
            }
        return {
            "target_size": 1,
            "attempt_count": 1,
            "generated_count": 1,
            "unique_count": 1,
            "exact_count": 1,
            "rows": [row],
            "selected": row,
            "hard_reject_reason": "",
            "attempted_pairs": [("u_destroy_heavy_robot_route", "u_fast_repair_dispatch")],
            "penalized_pairs": [],
            "coverage_hard_reject_count": 0,
            "exact_fail_count": 0,
            "exact_fail_reasons": {},
            "duplicate_hard_reject_count": 0,
        }

    def _generate_yz_candidate_pool(self, layer: str, iter_id: int, budget: int, target_size: int) -> Dict[str, object]:
        target = max(1, int(target_size))
        max_attempts = max(target, int(getattr(self.cfg, "resource_candidate_pool_max_attempts", 12)))
        planner = plan_y_candidate if str(layer) == "Y" else plan_z_candidate
        exact_applier = apply_exact_y_plan if str(layer) == "Y" else apply_exact_z_plan
        attempts = 0
        generated_count = 0
        rough_pool: List[Dict[str, object]] = []
        attempted_pairs: List[Tuple[str, str]] = []
        local_action_signatures = set()
        while attempts < max_attempts and len(rough_pool) < target:
            attempts += 1
            destroy_name, repair_name = self._sample_operator_pair(layer)
            attempted_pairs.append((str(destroy_name), str(repair_name)))
            plan = planner(self.opt, self.current_config, str(destroy_name), str(repair_name), self.rng, int(budget))
            if not bool(plan.get("success", False)):
                continue
            action_signature = plan.get("action_signature")
            action_signature_text = self._action_signature_text(action_signature)
            if action_signature_text in local_action_signatures or self._action_signature_known(layer, action_signature):
                continue
            local_action_signatures.add(action_signature_text)
            self._remember_action_signature(layer, action_signature)
            generated_count += 1
            destroy_ctx = dict(plan.get("destroy_ctx", {}) or {})
            rough_features = dict(plan.get("rough_features", {}) or {})
            if str(layer) == "Y":
                rough_features["affected_subtask_ids"] = [int(x) for x in (destroy_ctx.get("released_subtasks", {}) or {}).keys()]
            else:
                rough_features["affected_subtask_ids"] = sorted(
                    {
                        int(window_ctx.get("subtask_id", -1))
                        for window_ctx in (destroy_ctx.get("windows", []) or [])
                        if int(window_ctx.get("subtask_id", -1)) >= 0
                    }
                )
            rough_eval = self._score_rough_candidate(str(layer), rough_features, bool(plan.get("fallback_used", False)), int(iter_id))
            rough_pool.append(
                {
                    "iter": int(iter_id),
                    "layer": str(layer).upper(),
                    "candidate_stage": "rough",
                    "candidate_rank": 0,
                    "destroy_operator": str(destroy_name),
                    "repair_operator": str(repair_name),
                    "fallback_used": bool(plan.get("fallback_used", False)),
                    "projection_mode": "",
                    "projection_repaired_subtask_count": 0,
                    "F_raw": float(rough_eval.F_raw),
                    "F_cal": float(rough_eval.F_cal),
                    "duplicate_tote_count": int(rough_eval.duplicate_tote_count),
                    "duplicate_tote_penalty": float(rough_eval.duplicate_tote_penalty),
                    "candidate_signature": str(action_signature_text),
                    "candidate_signature_tuple": action_signature,
                    "candidate_payload": None,
                    "candidate_eval": rough_eval,
                    "selected_for_sa": False,
                    "action_signature": str(action_signature_text),
                    "plan": plan,
                    "z_structural_score": float(rough_features.get("z_structural_score", 0.0) or 0.0),
                    "z_choke_over_soft": float(rough_features.get("z_choke_over_soft", 0.0) or 0.0),
                    "z_station_load_soft": float(rough_features.get("z_station_load_soft", 0.0) or 0.0),
                    "z_robot_region_load_soft": float(rough_features.get("z_robot_region_load_soft", 0.0) or 0.0),
                }
            )

        self._select_best_candidate(rough_pool)
        rough_ranked = sorted(rough_pool, key=self._candidate_sort_key)
        candidate_rows: List[Dict[str, object]] = [dict(row) for row in rough_pool]
        exact_count = 0
        unique_count = 0
        selected = None
        hard_reject_reason = ""
        penalized_pairs: List[Tuple[str, str, float]] = []
        coverage_hard_reject_count = 0
        exact_fail_count = 0
        duplicate_hard_reject_count = 0
        exact_fail_reasons: Dict[str, int] = {}
        exact_trial_limit = max(1, int(getattr(self.cfg, "resource_exact_candidate_trial_limit", target)))
        exact_valid_rows: List[Dict[str, object]] = []
        for rough_row in rough_ranked[:exact_trial_limit]:
            exact_payload = exact_applier(self.opt, self.current_config, rough_row["plan"], self.rng)
            if not bool(exact_payload.get("success", False)):
                exact_fail_count += 1
                reason = str(exact_payload.get("reason", "exact_candidate_fail") or "exact_candidate_fail")
                exact_fail_reasons[reason] = int(exact_fail_reasons.get(reason, 0)) + 1
                exact_fail_detail = dict(exact_payload.get("validation_detail", {}) or {})
                hard_reject_reason = f"exact_candidate_fail:{reason}"
                candidate_rows.append(
                    {
                        "iter": int(iter_id),
                        "layer": str(layer).upper(),
                        "candidate_stage": "exact_fail",
                        "candidate_rank": int(rough_row.get("candidate_rank", 1) or 1),
                        "destroy_operator": str(rough_row.get("destroy_operator", "")),
                        "repair_operator": str(rough_row.get("repair_operator", "")),
                        "fallback_used": False,
                        "projection_mode": "",
                        "projection_repaired_subtask_count": 0,
                        "F_raw": float("nan"),
                        "F_cal": float("nan"),
                        "duplicate_tote_count": 0,
                        "duplicate_tote_penalty": 0.0,
                        "candidate_signature": str(rough_row.get("candidate_signature", "")),
                        "candidate_signature_tuple": rough_row.get("candidate_signature_tuple", None),
                        "candidate_payload": None,
                        "candidate_eval": None,
                        "selected_for_sa": False,
                        "action_signature": str(rough_row.get("action_signature", "")),
                        "exact_fail_reason": reason,
                        "exact_fail_detail": str(exact_fail_detail),
                        "z_structural_score": float(rough_row.get("z_structural_score", 0.0) or 0.0),
                        "z_choke_over_soft": float(rough_row.get("z_choke_over_soft", 0.0) or 0.0),
                        "z_station_load_soft": float(rough_row.get("z_station_load_soft", 0.0) or 0.0),
                        "z_robot_region_load_soft": float(rough_row.get("z_robot_region_load_soft", 0.0) or 0.0),
                    }
                )
                continue
            exact_count += 1
            candidate_signature = exact_payload["config"].validation_signature()
            candidate_eval = self._evaluate_config(
                config=exact_payload["config"],
                layer=str(layer).upper(),
                score_cache=exact_payload.get("score_cache", None),
                affected_subtask_ids=exact_payload.get("affected_ids", set()),
                fallback_penalty=0.15 if bool(exact_payload.get("fallback_used", False)) else 0.0,
                iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                distance_to_last_validated=config_distance(exact_payload["config"], self.last_validated_config),
            )
            candidate_eval.metadata.update(
                {
                    "last_validation_iter": int(self.last_validation_iter),
                    "last_validation_f_raw": float(self.last_validation_f_raw),
                    "recent_validated_makespans": list(self.recent_validated_makespans),
                }
            )
            exact_row = {
                "iter": int(iter_id),
                "layer": str(layer).upper(),
                "candidate_stage": "exact",
                "candidate_rank": int(rough_row.get("candidate_rank", 1) or 1),
                "destroy_operator": str(rough_row.get("destroy_operator", "")),
                "repair_operator": str(rough_row.get("repair_operator", "")),
                "fallback_used": bool(exact_payload.get("fallback_used", False)),
                "projection_mode": str(exact_payload.get("projection_mode", "")),
                "projection_repaired_subtask_count": int(exact_payload.get("projection_repaired_subtask_count", 0)),
                "F_raw": float(candidate_eval.F_raw),
                "F_cal": float(candidate_eval.F_cal),
                "eval_backend": str(candidate_eval.metadata.get("eval_backend", "surrogate")),
                "fixgurobi_status": str(candidate_eval.metadata.get("fixgurobi_status", "")),
                "fixgurobi_obj": candidate_eval.metadata.get("fixgurobi_obj", ""),
                "fixgurobi_bound": candidate_eval.metadata.get("fixgurobi_bound", ""),
                "fixgurobi_gap": candidate_eval.metadata.get("fixgurobi_gap", ""),
                "fixgurobi_solve_time": candidate_eval.metadata.get("fixgurobi_solve_time", ""),
                "fixgurobi_fixed_scope": str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")),
                "fixgurobi_fixed_constraint_count": candidate_eval.metadata.get("fixgurobi_fixed_constraint_count", ""),
                "fixgurobi_invalid_fix_count": candidate_eval.metadata.get("fixgurobi_invalid_fix_count", ""),
                "fixgurobi_fixed_route_arc_count_from_cfg": candidate_eval.metadata.get("fixgurobi_fixed_route_arc_count_from_cfg", ""),
                "fixgurobi_fixed_route_sequence_robot_count": candidate_eval.metadata.get("fixgurobi_fixed_route_sequence_robot_count", ""),
                "fixgurobi_fixed_route_sequence_missing_count": candidate_eval.metadata.get("fixgurobi_fixed_route_sequence_missing_count", ""),
                "fixgurobi_infeasible_reason": str(candidate_eval.metadata.get("fixgurobi_infeasible_reason", "")),
                "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
                "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
                "candidate_signature": self._candidate_signature_text(candidate_signature),
                "candidate_signature_tuple": candidate_signature,
                "candidate_payload": exact_payload,
                "candidate_eval": candidate_eval,
                "selected_for_sa": False,
                "action_signature": str(rough_row.get("action_signature", "")),
                "coverage_feasible": bool(candidate_eval.coverage_feasible),
                "unmet_sku_total": int(candidate_eval.unmet_sku_total),
                "z_structural_score": float(rough_row.get("z_structural_score", 0.0) or 0.0),
                "z_choke_over_soft": float(rough_row.get("z_choke_over_soft", 0.0) or 0.0),
                "z_station_load_soft": float(rough_row.get("z_station_load_soft", 0.0) or 0.0),
                "z_robot_region_load_soft": float(rough_row.get("z_robot_region_load_soft", 0.0) or 0.0),
                "revolving_enabled": bool(candidate_eval.metadata.get("revolving_enabled", False)),
                "released_layer": str(candidate_eval.metadata.get("released_layer", "")),
                "fixed_layers": str(candidate_eval.metadata.get("fixed_layers", "")),
                "inner_relaxed_obj": candidate_eval.metadata.get("inner_relaxed_obj", ""),
                "u_fast_cmax": candidate_eval.metadata.get("u_fast_cmax", ""),
                "u_route_lb": candidate_eval.metadata.get("u_route_lb", ""),
                "u_repair_time": candidate_eval.metadata.get("u_repair_time", ""),
                "u_changed_robot_count": candidate_eval.metadata.get("u_changed_robot_count", ""),
                "revolving_lb": candidate_eval.metadata.get("revolving_lb", ""),
                "lb_gate_skipped": candidate_eval.metadata.get("lb_gate_skipped", ""),
            }
            candidate_rows.append(exact_row)
            row_fixes_z = str(candidate_eval.metadata.get("fixgurobi_fixed_scope", "")).upper() in {"Z", "XYZ", "XYZU"}
            if bool(row_fixes_z) and (not bool(candidate_eval.coverage_feasible) or int(candidate_eval.unmet_sku_total) > 0):
                hard_reject_reason = "coverage_hard_reject"
                coverage_hard_reject_count += 1
                penalized_pairs.append((str(rough_row.get("destroy_operator", "")), str(rough_row.get("repair_operator", "")), -6.0))
                continue
            if bool(row_fixes_z) and int(candidate_eval.duplicate_tote_count) > 0:
                hard_reject_reason = "duplicate_tote_hard_reject"
                duplicate_hard_reject_count += 1
                continue
            exact_valid_rows.append(exact_row)
        if exact_valid_rows:
            unique_count = int(len(exact_valid_rows))
            selected = self._select_best_candidate(exact_valid_rows)
            if selected is not None:
                selected["selected_for_sa"] = True
                selected_signature = selected.get("candidate_signature_tuple", None)
                for row in candidate_rows:
                    if row.get("candidate_stage") == "exact" and row.get("candidate_signature_tuple") == selected_signature:
                        row["selected_for_sa"] = True
                        break
                hard_reject_reason = ""
        elif int(coverage_hard_reject_count) > 0:
            hard_reject_reason = "coverage_hard_reject"
        elif int(duplicate_hard_reject_count) > 0:
            hard_reject_reason = "duplicate_tote_hard_reject"
        return {
            "target_size": int(target),
            "attempt_count": int(attempts),
            "generated_count": int(generated_count),
            "unique_count": int(unique_count),
            "exact_count": int(exact_count),
            "rows": candidate_rows,
            "selected": selected,
            "hard_reject_reason": str(hard_reject_reason),
            "attempted_pairs": attempted_pairs,
            "penalized_pairs": penalized_pairs,
            "coverage_hard_reject_count": int(coverage_hard_reject_count),
            "exact_fail_count": int(exact_fail_count),
            "exact_fail_reasons": dict(exact_fail_reasons),
            "duplicate_hard_reject_count": int(duplicate_hard_reject_count),
        }

    def _sa_accept(self, candidate_eval, layer: str) -> Tuple[bool, float, float]:
        delta = float(candidate_eval.F_cal) - float(self.current_eval.F_cal)
        effective_temp = float(self.temperature)
        if str(layer) == "X":
            effective_temp *= float(getattr(self.cfg, "resource_x_sa_temp_multiplier", 2.0))
        if delta <= 0.0:
            return True, 1.0, float(effective_temp)
        temp = max(1e-6, float(effective_temp))
        accept_prob = float(math.exp(-delta / temp))
        return bool(self.rng.random() < accept_prob), float(accept_prob), float(effective_temp)

    def _accept_candidate(self, candidate_eval, layer: str) -> Tuple[bool, float, float, str, float]:
        mode = str(getattr(self.cfg, "resource_acceptance_mode", "sa") or "sa").strip().lower()
        if mode != "lahc":
            accepted, prob, temp = self._sa_accept(candidate_eval, layer)
            return bool(accepted), float(prob), float(temp), "sa", float("nan")
        if not getattr(self, "lahc_history", []):
            self.lahc_history = [float(self.current_eval.F_cal)]
            self.lahc_index = 0
        threshold = float(self.lahc_history[int(self.lahc_index) % len(self.lahc_history)])
        current_value = float(self.current_eval.F_cal)
        candidate_value = float(candidate_eval.F_cal)
        greedy_when_best = bool(getattr(self.cfg, "resource_lahc_greedy_when_best", True))
        best_gate = float(self.best_f_raw) if bool(greedy_when_best) else float("inf")
        accepted = bool(
            candidate_value <= current_value + 1e-9
            or candidate_value <= threshold + 1e-9
            or candidate_value <= best_gate + 1e-9
        )
        return bool(accepted), 1.0 if accepted else 0.0, 0.0, "lahc", float(threshold)

    def _advance_lahc_history(self) -> None:
        if not getattr(self, "lahc_history", []):
            return
        idx = int(self.lahc_index) % len(self.lahc_history)
        self.lahc_history[idx] = float(self.current_eval.F_cal)
        self.lahc_index = int(self.lahc_index) + 1
        self.lahc_threshold = float(self.lahc_history[int(self.lahc_index) % len(self.lahc_history)])

    def _restart_current_search_state(self, iter_id: int) -> None:
        self.multi_start_restart_count = int(getattr(self, "multi_start_restart_count", 0)) + 1
        self.current_config = getattr(self, "initial_config", self.current_config).clone()
        self.current_eval = self._evaluate_config(
            config=self.current_config,
            layer="XYZ",
            iterations_since_last_validation=0,
            distance_to_last_validated=config_distance(self.current_config, self.last_validated_config),
        )
        self.temperature = float(getattr(self.cfg, "resource_sa_init_temp", max(1.0, 0.05 * float(self.current_eval.F_raw))))
        self.no_improve_rounds = 0.0
        self.no_best_z_change_rounds = 0.0
        self.validated_best_no_change_rounds = 0
        self.layer_stagnation = {layer: 0.0 for layer in self.layer_stagnation}
        self.consecutive_fail_count = {layer: 0 for layer in self.consecutive_fail_count}
        self.layer_cooldown_until_iter = {layer: 0 for layer in self.layer_cooldown_until_iter}
        self.layer_failure_cooldown_until_iter = {layer: 0 for layer in self.layer_failure_cooldown_until_iter}
        self.lahc_history = [float(self.current_eval.F_cal)] * max(1, int(getattr(self.cfg, "resource_lahc_history_length", 20)))
        self.lahc_index = 0
        self.lahc_threshold = float(self.current_eval.F_cal)
        shake_steps = max(0, int(getattr(self.cfg, "resource_multi_start_shake_steps", 3)))
        for _ in range(shake_steps):
            layer = str(self.rng.choice(["X", "Y", "Z"]))
            destroy_name, repair_name = self._sample_operator_pair(layer)
            budget = max(1, self._effective_destroy_budget(layer, float(getattr(self.cfg, "resource_destroy_mu_medium", 0.20))))
            payload = None
            if layer == "X":
                payload = self._apply_x_candidate(int(iter_id), str(destroy_name), str(repair_name), int(budget))
            else:
                planner = plan_y_candidate if layer == "Y" else plan_z_candidate
                applier = apply_exact_y_plan if layer == "Y" else apply_exact_z_plan
                plan = planner(self.opt, self.current_config, str(destroy_name), str(repair_name), self.rng, int(budget))
                if bool(plan.get("success", False)):
                    payload = applier(self.opt, self.current_config, plan, self.rng)
            if payload is not None and bool(payload.get("success", True)) and payload.get("config", None) is not None:
                self.current_config = payload["config"]
                self.current_eval = self._evaluate_config(
                    config=self.current_config,
                    layer=str(layer),
                    affected_subtask_ids=payload.get("affected_ids", set()),
                    fallback_penalty=0.15 if bool(payload.get("fallback_used", False)) else 0.0,
                    iterations_since_last_validation=0,
                    distance_to_last_validated=config_distance(self.current_config, self.last_validated_config),
                )

    def _y_polish_proxy_score(self, config: ResourceConfig) -> Tuple[float, ...]:
        station_loads: Dict[int, float] = {}
        station_counts: Dict[int, int] = {}
        for row in config.subtasks.values():
            if int(row.station_id) < 0:
                continue
            station_counts[int(row.station_id)] = int(station_counts.get(int(row.station_id), 0)) + 1
            station_loads[int(row.station_id)] = float(station_loads.get(int(row.station_id), 0.0)) + float(
                sum(float(getattr(task, "station_service_time", 0.0) or 0.0) + max(1, int(getattr(task, "sku_pick_count", 0) or 0)) for task in (row.z_tasks or []))
            )
        loads = list(station_loads.values()) or [0.0]
        return (
            float(max(loads)),
            float(statistics.pstdev(loads) if len(loads) >= 2 else 0.0),
            float(max(station_counts.values(), default=0)),
            float(self.scorer.evaluate(config).F_raw),
        )

    def _enumerate_y_polish_candidates(self, base_config: ResourceConfig) -> List[Tuple[Tuple[float, ...], ResourceConfig]]:
        rows = sorted(base_config.subtasks.values(), key=lambda row: (int(row.order_id), int(row.subtask_id)))
        cap = max(1, int(getattr(self.cfg, "resource_y_polish_subtask_cap", 6)))
        station_count = max(1, len(list(getattr(getattr(self.opt, "problem", None), "station_list", []) or [])))
        if len(rows) <= 1 or len(rows) > int(cap) or station_count <= 0:
            return []
        candidate_limit = max(1, int(getattr(self.cfg, "resource_y_polish_candidate_limit", 64)))
        scored: List[Tuple[Tuple[float, ...], ResourceConfig]] = []
        seen = set()
        row_ids = [int(row.subtask_id) for row in rows]
        for station_assignment in itertools.product(range(station_count), repeat=len(rows)):
            groups: Dict[int, List[int]] = {}
            for subtask_id, station_id in zip(row_ids, station_assignment):
                groups.setdefault(int(station_id), []).append(int(subtask_id))
            per_station_orders: List[List[Tuple[int, ...]]] = []
            station_ids = sorted(groups.keys())
            for station_id in station_ids:
                group = list(groups[int(station_id)])
                if len(group) <= 1:
                    per_station_orders.append([tuple(group)])
                elif len(group) <= 4:
                    current_rank = {
                        int(subtask_id): int(base_config.subtasks[int(subtask_id)].station_rank if int(base_config.subtasks[int(subtask_id)].station_rank) >= 0 else 10**9)
                        for subtask_id in group
                    }
                    perms = sorted(
                        itertools.permutations(group),
                        key=lambda perm: (
                            sum(abs(idx - int(current_rank.get(int(subtask_id), idx))) for idx, subtask_id in enumerate(perm)),
                            tuple(int(x) for x in perm),
                        ),
                    )[: max(2, min(8, math.factorial(len(group))))]
                    per_station_orders.append([tuple(int(x) for x in perm) for perm in perms])
                else:
                    per_station_orders.append([tuple(group), tuple(reversed(group))])
            for order_combo in itertools.product(*per_station_orders):
                trial = base_config.clone_for_layer("Y", row_ids)
                for station_id, sequence in zip(station_ids, order_combo):
                    for rank, subtask_id in enumerate(sequence):
                        row = trial.subtasks.get(int(subtask_id))
                        if row is None:
                            continue
                        row.station_id = int(station_id)
                        row.station_rank = int(rank)
                trial.rebuild_indices()
                signature = trial.validation_signature()
                if signature in seen:
                    continue
                seen.add(signature)
                score = self._y_polish_proxy_score(trial)
                scored.append((score, trial))
                if len(scored) > candidate_limit * 20:
                    scored.sort(key=lambda item: item[0])
                    scored = scored[: candidate_limit * 10]
        scored.sort(key=lambda item: item[0])
        return scored[:candidate_limit]

    def _best_y_assignment_polish(self) -> Dict[str, object]:
        if not bool(getattr(self.cfg, "resource_enable_best_y_assignment_polish", False)):
            return {"enabled": False, "applied": False}
        if self.best_validated.snapshot is not None:
            self.opt.restore_snapshot(self.best_validated.snapshot)
        base_config = self.best_validated.config.clone()
        candidates = self._enumerate_y_polish_candidates(base_config)
        best_makespan = float(self.best_validated.makespan)
        best_payload = None
        exact_count = 0
        for _score, candidate in candidates:
            if candidate.validation_signature() == base_config.validation_signature():
                continue
            validation = self.validator.validate(candidate, -1)
            exact_count += int(validation.get("validation_call_count", validation.get("lkh_call_count", 1)) or 0)
            hard_reason = str(validation.get("hard_reject_reason", "") or "")
            if hard_reason:
                continue
            makespan = float(validation.get("makespan", float("inf")))
            if makespan + 1e-9 < best_makespan:
                best_makespan = float(makespan)
                best_payload = (candidate.clone(), validation)
        if best_payload is not None:
            candidate, validation = best_payload
            self.best_validated = ValidatedIncumbent(
                config=candidate.clone(),
                makespan=float(best_makespan),
                iter_id=-1,
                snapshot=validation["snapshot"],
            )
            self.current_config = candidate.clone()
            self.last_validated_config = candidate.clone()
            self.last_validated_signature = candidate.validation_signature()
            self.opt.best = validation["snapshot"]
            self.opt.work = validation["snapshot"]
            self.opt.work_z = float(best_makespan)
        stats = {
            "enabled": True,
            "applied": bool(best_payload is not None),
            "candidate_count": int(len(candidates)),
            "exact_count": int(exact_count),
            "best_makespan": float(best_makespan),
        }
        self.opt.best_y_assignment_polish_stats = stats
        return stats

    def _best_z_sortify_polish(self) -> Dict[str, object]:
        if not bool(getattr(self.cfg, "resource_enable_best_z_sortify_polish", False)):
            return {"enabled": False, "applied": False}
        if self.best_validated.snapshot is not None:
            self.opt.restore_snapshot(self.best_validated.snapshot)
        before_flip_count = int(
            sum(
                1
                for subtask in self.best_validated.config.subtasks.values()
                for descriptor in (subtask.z_tasks or [])
                if str(getattr(descriptor, "mode", "")).upper() == "FLIP"
            )
        )
        candidate, sortify_stats = apply_single_flip_sortify_polish(self.opt, self.best_validated.config)
        if candidate.validation_signature() == self.best_validated.config.validation_signature():
            stats = {"enabled": True, "applied": False, **{f"sortify_{k}": v for k, v in sortify_stats.items()}}
            self.opt.best_z_sortify_polish_stats = stats
            return stats
        validation = self.validator.validate(candidate, -1)
        hard_reason = str(validation.get("hard_reject_reason", "") or "")
        makespan = float(validation.get("makespan", float("inf")))
        applied = False
        after_flip_count = int(
            sum(
                1
                for subtask in candidate.subtasks.values()
                for descriptor in (subtask.z_tasks or [])
                if str(getattr(descriptor, "mode", "")).upper() == "FLIP"
            )
        )
        structural_tie_improved = bool(
            makespan <= float(self.best_validated.makespan) + 1e-9
            and after_flip_count < before_flip_count
        )
        if not hard_reason and makespan + 1e-9 < float(self.best_validated.makespan):
            self.best_validated = ValidatedIncumbent(
                config=candidate.clone(),
                makespan=float(makespan),
                iter_id=-1,
                snapshot=validation["snapshot"],
            )
            self.current_config = candidate.clone()
            self.last_validated_config = candidate.clone()
            self.last_validated_signature = candidate.validation_signature()
            self.opt.best = validation["snapshot"]
            self.opt.work = validation["snapshot"]
            self.opt.work_z = float(makespan)
            applied = True
        stats = {
            "enabled": True,
            "applied": bool(applied),
            "hard_reject_reason": hard_reason,
            "candidate_makespan": float(makespan),
            "before_flip_count": int(before_flip_count),
            "after_flip_count": int(after_flip_count),
            "structural_tie_improved": bool(structural_tie_improved),
            **{f"sortify_{k}": v for k, v in sortify_stats.items()},
        }
        self.opt.best_z_sortify_polish_stats = stats
        return stats

    def _update_layer_progress(self, layer: str, accepted: bool, prev_f_raw: float, new_f_raw: float, stagnation_increment: float) -> None:
        improvement = max(0.0, float(prev_f_raw) - float(new_f_raw)) if bool(accepted) else 0.0
        self.layer_ema_improve[layer] = float(0.7 * float(self.layer_ema_improve[layer]) + 0.3 * max(1e-6, improvement))
        if improvement > 1e-9:
            self.layer_stagnation[layer] = 0.0
        else:
            self.layer_stagnation[layer] = float(self.layer_stagnation[layer]) + float(stagnation_increment)
        if float(new_f_raw) + 1e-9 < float(self.best_f_raw):
            self.best_f_raw = float(new_f_raw)
            self.no_improve_rounds = 0.0
        else:
            self.no_improve_rounds = float(self.no_improve_rounds) + float(stagnation_increment)

    def _should_validate(self, iter_id: int, candidate_eval, candidate_signature) -> str:
        if int(iter_id) - int(self.last_validation_iter) >= int(getattr(self.cfg, "resource_real_eval_period", 8)):
            if candidate_signature == self.last_validated_signature:
                return "periodic_skip_same_config"
            return "periodic"
        if float(candidate_eval.F_raw) + 1e-9 < float(self.best_f_raw) and candidate_signature != self.last_validated_signature:
            return "f_raw_breakthrough"
        return ""

    def _catastrophic_threshold(self) -> float:
        vals = [float(v) for v in (self.recent_validated_makespans or []) if float(v) > 0.0]
        if len(vals) < 2:
            validated_cv = 0.0
        else:
            validated_cv = float(statistics.pstdev(vals) / max(1e-9, statistics.mean(vals)))
        return float(max(float(getattr(self.cfg, "resource_catastrophic_threshold_floor", 1.30)), 1.0 + float(getattr(self.cfg, "resource_catastrophic_cv_scale", 3.0)) * validated_cv))

    def _record_reward(self, layer: str, destroy_name: str, repair_name: str, reward: float, fallback_used: bool, iter_id: int) -> None:
        layer_name = str(layer).upper()
        self.operator_arms[layer_name]["destroy"][str(destroy_name)].record(float(reward), int(iter_id))
        self.operator_arms[layer_name]["repair"][str(repair_name)].record(float(reward), int(iter_id))
        if bool(fallback_used):
            fallback_name = {"X": X_FALLBACK_OPERATOR, "Y": Y_FALLBACK_OPERATOR, "Z": Z_FALLBACK_OPERATOR}.get(layer_name)
            if fallback_name is not None and str(fallback_name) in self.operator_arms[layer_name]["repair"]:
                self.operator_arms[layer_name]["repair"][str(fallback_name)].record(float(reward), int(iter_id))
        self.layer_exec_since_update[layer_name] = int(self.layer_exec_since_update[layer_name]) + 1

    def _apply_empty_candidate_failure(self, layer: str, attempted_pairs: List[Tuple[str, str]], iter_id: int) -> bool:
        reward = float(getattr(self.cfg, "resource_empty_candidate_reward", -2.0))
        penalized = False
        for destroy_name, repair_name in list(attempted_pairs or []):
            if str(destroy_name) not in self.operator_arms[str(layer)]["destroy"]:
                continue
            if str(repair_name) not in self.operator_arms[str(layer)]["repair"]:
                continue
            self._record_reward(str(layer), str(destroy_name), str(repair_name), reward, False, int(iter_id))
            penalized = True
        if penalized:
            cooldown = max(0, int(getattr(self.cfg, "resource_empty_candidate_layer_cooldown", 3)))
            if cooldown > 0:
                self.layer_cooldown_until_iter[str(layer)] = max(
                    int(self.layer_cooldown_until_iter.get(str(layer), 0)),
                    int(iter_id) + cooldown,
                )
        return penalized

    def _apply_pair_rewards(self, layer: str, rewards: List[Tuple[str, str, float]], iter_id: int) -> bool:
        applied = False
        for destroy_name, repair_name, reward in list(rewards or []):
            if str(destroy_name) not in self.operator_arms[str(layer)]["destroy"]:
                continue
            if str(repair_name) not in self.operator_arms[str(layer)]["repair"]:
                continue
            self._record_reward(str(layer), str(destroy_name), str(repair_name), float(reward), False, int(iter_id))
            applied = True
        return applied

    def _stagnation_increment(self, valid_candidate_scored: bool, used_exact_eval_cache: bool, improved_best: bool) -> float:
        if bool(improved_best):
            return 0.0
        if not bool(valid_candidate_scored):
            return float(getattr(self.cfg, "resource_empty_candidate_stagnation_increment", 0.0))
        if bool(used_exact_eval_cache):
            return float(getattr(self.cfg, "resource_cache_hit_stagnation_increment", 0.2))
        return 1.0

    def _update_failure_state(self, layer: str, accepted: bool, improved_best: bool, iter_id: int) -> None:
        layer_name = str(layer)
        if bool(accepted):
            self.consecutive_fail_count[layer_name] = 0
        else:
            self.consecutive_fail_count[layer_name] = int(self.consecutive_fail_count.get(layer_name, 0)) + 1
        if layer_name == "X" and int(self.consecutive_fail_count.get("X", 0)) >= int(getattr(self.cfg, "resource_layer_fail_threshold", 3)):
            factor = float(getattr(self.cfg, "resource_layer_fail_multiplier", 0.1))
            self.layer_dynamic_multiplier["X"] = float(max(1e-6, float(self.layer_dynamic_multiplier.get("X", 1.0)) * factor))
            cooldown = max(0, int(getattr(self.cfg, "resource_layer_fail_cooldown", 10)))
            if cooldown > 0:
                self.layer_failure_cooldown_until_iter["X"] = max(
                    int(self.layer_failure_cooldown_until_iter.get("X", 0)),
                    int(iter_id) + cooldown,
                )
            self.consecutive_fail_count["X"] = 0
            self.x_failure_decapitation_count += 1
        if bool(improved_best) and layer_name in ("Y", "Z"):
            self.consecutive_fail_count["X"] = 0
            self.layer_dynamic_multiplier["X"] = 1.0
            self.layer_failure_cooldown_until_iter["X"] = 0

    def _update_exact_cache_funnel(self, used_exact_eval_cache: bool, improved_best: bool) -> None:
        if bool(improved_best):
            self.consecutive_exact_cache_hit_count = 0
            self.adaptive_destroy_bonus = 0.0
            return
        if bool(used_exact_eval_cache):
            self.consecutive_exact_cache_hit_count = int(self.consecutive_exact_cache_hit_count) + 1
            trigger = int(getattr(self.cfg, "resource_adaptive_destroy_cache_hit_trigger", 3))
            if int(self.consecutive_exact_cache_hit_count) >= trigger:
                step = float(getattr(self.cfg, "resource_adaptive_destroy_bonus_step", 0.05))
                cap = float(getattr(self.cfg, "resource_adaptive_destroy_bonus_cap", 0.20))
                self.adaptive_destroy_bonus = float(min(cap, float(self.adaptive_destroy_bonus) + step))
                self.consecutive_exact_cache_hit_count = 0
        else:
            self.consecutive_exact_cache_hit_count = 0

    def _maybe_update_weights(self, layer: str, iter_id: int) -> None:
        batch_size = int(getattr(self.cfg, "resource_operator_update_batch_size", 10))
        max_stale = int(getattr(self.cfg, "resource_operator_update_max_stale_rounds", 15))
        if int(self.layer_exec_since_update[layer]) < batch_size and int(iter_id) - int(self.layer_last_update_iter[layer]) < max_stale:
            return
        rho = float(getattr(self.cfg, "resource_weight_reaction", 0.2))
        floor = float(getattr(self.cfg, "resource_operator_weight_floor", 0.1))
        for group in self.operator_arms[layer].values():
            for arm in group.values():
                if arm.pending_rewards:
                    avg_reward = float(sum(arm.pending_rewards) / max(1, len(arm.pending_rewards)))
                    target = max(floor, avg_reward)
                    arm.weight = float((1.0 - rho) * float(arm.weight) + rho * target)
                    arm.pending_rewards = []
        self._apply_operator_weight_floors()
        self.layer_exec_since_update[layer] = 0
        self.layer_last_update_iter[layer] = int(iter_id)
        self._refresh_operator_stats_payload()

    def _weight_snapshot(self, layer: str) -> Dict[str, float]:
        payload = {}
        for group in self.operator_arms[layer].values():
            for name, arm in group.items():
                payload[str(name)] = float(arm.weight)
        return payload

    def _counts_as_effective_iteration(self, candidate_pool_info: Dict[str, object]) -> bool:
        return bool(int(candidate_pool_info.get("generated_count", 0) or 0) > 0)

    def _accumulate_joint_postprocess_stats(self, stats: Dict[str, float]) -> None:
        for key, value in dict(stats or {}).items():
            if str(key) not in self.joint_colocated_sort_postprocess_stats:
                continue
            self.joint_colocated_sort_postprocess_stats[str(key)] = float(
                self.joint_colocated_sort_postprocess_stats.get(str(key), 0.0)
            ) + float(value or 0.0)
        self.opt.joint_colocated_sort_postprocess_stats = self.joint_colocated_sort_postprocess_stats

    def _validate_candidate_config(self, config: ResourceConfig, iter_id: int) -> Dict[str, object]:
        validation = dict(self.validator.validate(config, iter_id) or {})
        validation["lkh_call_count"] = int(validation.get("lkh_call_count", 1) or 0)
        return validation

    def _attempt_constrained_reentry(
        self,
        iter_id: int,
        previous_config: ResourceConfig,
        previous_eval,
        candidate_config: ResourceConfig,
        validation: Dict[str, object],
    ) -> Tuple[Optional[ResourceConfig], Optional[Dict[str, object]]]:
        if not bool(getattr(self.cfg, "resource_conflict_local_reentry_enabled", True)):
            return None, None
        conflict_summary = dict(validation.get("conflict_summary", {}) or {})
        affected_ids = {
            int(x)
            for x in (
                conflict_summary.get("failed_subtask_ids", [])
                or [row.get("subtask_id", -1) for row in (validation.get("unassigned_robot_tasks", []) or [])]
            )
            if int(x) >= 0 and int(x) in candidate_config.subtasks
        }
        if not affected_ids:
            return None, None
        repaired_config = candidate_config.clone()
        repaired_config.metadata = dict(getattr(repaired_config, "metadata", {}) or {})
        repaired_config.metadata["transient_conflict_constraints"] = dict(conflict_summary)
        try:
            repaired_config, _, _ = apply_projection_repair(
                opt=self.opt,
                previous_config=previous_config,
                candidate_config=repaired_config,
                previous_eval=previous_eval,
                affected_subtask_ids=sorted(affected_ids),
                iter_id=int(iter_id),
                rng=self.rng,
            )
        except Exception:
            return None, None
        repaired_validation = self._validate_candidate_config(repaired_config, int(iter_id))
        if str(repaired_validation.get("hard_reject_reason", "") or ""):
            return None, repaired_validation
        return repaired_config, repaired_validation

    def _maybe_apply_joint_colocated_sort_postprocess(
        self,
        iter_id: int,
        layer: str,
        candidate_row: Optional[Dict[str, object]],
    ) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]], Dict[str, float], float]:
        empty_stats = dict(self.joint_colocated_sort_postprocess_stats)
        for key in empty_stats:
            empty_stats[key] = 0.0
        if (
            candidate_row is None
            or str(layer).upper() != "Z"
            or not bool(getattr(self.cfg, "resource_enable_joint_colocated_sort_postprocess", True))
        ):
            return candidate_row, None, empty_stats, 0.0
        payload = candidate_row.get("candidate_payload", {}) or {}
        candidate_config = payload.get("config", None)
        if candidate_config is None:
            return candidate_row, None, empty_stats, 0.0
        max_groups = max(0, int(getattr(self.cfg, "resource_joint_colocated_sort_max_groups_per_candidate", 1)))
        if max_groups <= 0:
            return candidate_row, None, empty_stats, 0.0
        t0 = time.perf_counter()
        enhanced_config, stats = apply_joint_colocated_sort_postprocess(self.opt, candidate_config, max_groups=max_groups)
        if int(stats.get("applied", 0) or 0) <= 0:
            return candidate_row, None, stats, float(time.perf_counter() - t0)
        baseline_validation = self._validate_candidate_config(candidate_config, int(iter_id))
        enhanced_validation = self._validate_candidate_config(enhanced_config, int(iter_id))
        validation_call_count = int(baseline_validation.get("lkh_call_count", 0)) + int(enhanced_validation.get("lkh_call_count", 0))
        baseline_valid = not str(baseline_validation.get("hard_reject_reason", "") or "")
        enhanced_valid = not str(enhanced_validation.get("hard_reject_reason", "") or "")
        baseline_eval = candidate_row.get("candidate_eval")
        enhanced_payload = dict(payload)
        enhanced_payload["config"] = enhanced_config
        enhanced_payload["affected_ids"] = set(
            int(subtask_id)
            for subtask_id in enhanced_config.subtasks.keys()
            if enhanced_config.subtasks[int(subtask_id)].validation_signature()
            != candidate_config.subtasks.get(int(subtask_id), enhanced_config.subtasks[int(subtask_id)]).validation_signature()
        )
        enhanced_eval = self._evaluate_config(
            config=enhanced_config,
            layer="Z",
            score_cache=enhanced_payload.get("score_cache", None),
            affected_subtask_ids=enhanced_payload.get("affected_ids", set()),
            fallback_penalty=0.15 if bool(enhanced_payload.get("fallback_used", False)) else 0.0,
            iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
            distance_to_last_validated=config_distance(enhanced_config, self.last_validated_config),
        )
        enhanced_eval.metadata.update(
            {
                "last_validation_iter": int(self.last_validation_iter),
                "last_validation_f_raw": float(self.last_validation_f_raw),
                "recent_validated_makespans": list(self.recent_validated_makespans),
            }
        )
        choose_enhanced = False
        if enhanced_valid and not baseline_valid:
            choose_enhanced = True
        elif enhanced_valid and baseline_valid:
            baseline_makespan = float(baseline_validation.get("makespan", float("inf")))
            enhanced_makespan = float(enhanced_validation.get("makespan", float("inf")))
            if float(enhanced_makespan) + 1e-9 < float(baseline_makespan):
                choose_enhanced = True
            elif abs(float(enhanced_makespan) - float(baseline_makespan)) <= 1e-9 and baseline_eval is not None:
                choose_enhanced = float(enhanced_eval.F_cal) + 1e-9 < float(baseline_eval.F_cal)
        if not choose_enhanced:
            stats["rejected_eval_not_better"] = float(stats.get("rejected_eval_not_better", 0.0)) + 1.0
            baseline_validation["validation_call_count"] = int(validation_call_count)
            return candidate_row, baseline_validation, stats, float(time.perf_counter() - t0)
        updated_row = dict(candidate_row)
        updated_row["candidate_payload"] = enhanced_payload
        updated_row["candidate_eval"] = enhanced_eval
        updated_row["F_raw"] = float(enhanced_eval.F_raw)
        updated_row["F_cal"] = float(enhanced_eval.F_cal)
        updated_row["eval_backend"] = str(enhanced_eval.metadata.get("eval_backend", "surrogate"))
        updated_row["fixgurobi_status"] = str(enhanced_eval.metadata.get("fixgurobi_status", ""))
        updated_row["fixgurobi_obj"] = enhanced_eval.metadata.get("fixgurobi_obj", "")
        updated_row["fixgurobi_bound"] = enhanced_eval.metadata.get("fixgurobi_bound", "")
        updated_row["fixgurobi_gap"] = enhanced_eval.metadata.get("fixgurobi_gap", "")
        updated_row["fixgurobi_solve_time"] = enhanced_eval.metadata.get("fixgurobi_solve_time", "")
        updated_row["fixgurobi_fixed_scope"] = str(enhanced_eval.metadata.get("fixgurobi_fixed_scope", ""))
        updated_row["fixgurobi_infeasible_reason"] = str(enhanced_eval.metadata.get("fixgurobi_infeasible_reason", ""))
        updated_row["duplicate_tote_count"] = int(enhanced_eval.duplicate_tote_count)
        updated_row["duplicate_tote_penalty"] = float(enhanced_eval.duplicate_tote_penalty)
        updated_row["candidate_signature_tuple"] = enhanced_config.validation_signature()
        updated_row["candidate_signature"] = self._candidate_signature_text(updated_row["candidate_signature_tuple"])
        updated_row["joint_colocated_sort_postprocess_applied"] = True
        if baseline_valid and enhanced_valid:
            stats["makespan_improvement"] = float(stats.get("makespan_improvement", 0.0)) + max(
                0.0,
                float(baseline_validation.get("makespan", 0.0)) - float(enhanced_validation.get("makespan", 0.0)),
            )
        enhanced_validation["validation_call_count"] = int(validation_call_count)
        return updated_row, enhanced_validation, stats, float(time.perf_counter() - t0)

    def run(self) -> float:
        if self.best_validated.snapshot is not None:
            self.opt.best = self.best_validated.snapshot
            self.opt.work = self.best_validated.snapshot
            self.opt.work_z = float(self.best_validated.makespan)
        max_iters = int(getattr(self.cfg, "max_iters", 50))
        cooling = float(getattr(self.cfg, "resource_sa_cooling", 0.95))
        reheat = float(getattr(self.cfg, "resource_sa_reheat_factor", 1.25))
        for iter_id in range(1, max_iters + 1):
            t_iter0 = time.perf_counter()
            best_z_before_iter = float(self.best_validated.makespan)
            layer, force_rotate_used = self._select_layer(iter_id)
            effective_destroy_mu, heavy_destroy_active, destroy_tier = self._current_destroy_mu()
            effective_destroy_budget = self._effective_destroy_budget(layer, effective_destroy_mu)
            prev_f_raw = float(self.current_eval.F_raw)
            accepted = False
            accept_prob = 0.0
            effective_sa_temperature = float(self.temperature)
            acceptance_mode = str(getattr(self.cfg, "resource_acceptance_mode", "sa") or "sa")
            lahc_threshold = float("nan")
            fallback_used = False
            projection_mode = ""
            projection_count = 0
            destroy_name = ""
            repair_name = ""
            validation_trigger = ""
            validated_makespan = float("nan")
            catastrophic_rollback = False
            tail_guard_rejected = False
            tail_guard_reason = ""
            tail_guard_meta: Dict[str, object] = {}
            improved_best = False
            reward = -2.0
            candidate_eval = self.current_eval
            val_time = 0.0
            candidate_hard_reject_reason = ""
            x_temp_boost_used = False
            empty_candidate_penalized = False
            layer_cooldown_remaining = self._current_layer_cooldown_remaining(layer, iter_id + 1)
            x_failure_cooldown_remaining = self._current_failure_cooldown_remaining("X", iter_id + 1)
            used_exact_eval_cache = False
            exact_eval_cache_hit_count = int(getattr(self.scorer, "exact_eval_cache_hit_count", 0))
            coverage_hard_reject = False
            unmet_sku_total = 0
            stagnation_increment = 0.0
            joint_postprocess_stats = {
                "triggered": 0.0,
                "candidate_groups": 0.0,
                "submitted": 0.0,
                "applied": 0.0,
                "makespan_improvement": 0.0,
                "rejected_capacity": 0.0,
                "rejected_interval_illegal": 0.0,
                "rejected_noise": 0.0,
                "rejected_eval_not_better": 0.0,
                "rejected_validation": 0.0,
                "rejected_target_conflict": 0.0,
            }
            joint_postprocess_time_sec = 0.0
            precomputed_validation = None
            candidate_pool_target_size = int(getattr(self.cfg, "resource_candidate_pool_size", 3))
            if str(layer) == "X":
                candidate_pool_info = self._generate_x_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            elif str(layer).upper() == "XZ":
                candidate_pool_info = self._generate_xz_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            elif str(layer).upper() == "YZ":
                candidate_pool_info = self._generate_yz_joint_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            elif str(layer).upper() == "XY":
                candidate_pool_info = self._generate_xy_joint_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            elif str(layer).upper() == "XYZ":
                candidate_pool_info = self._generate_xyz_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            elif str(layer).upper() == "U":
                candidate_pool_info = self._generate_u_candidate_pool(iter_id, effective_destroy_budget, candidate_pool_target_size)
            else:
                candidate_pool_info = self._generate_yz_candidate_pool(layer, iter_id, effective_destroy_budget, candidate_pool_target_size)
            candidate_rows = list(candidate_pool_info.get("rows", []) or [])
            selected_candidate = candidate_pool_info.get("selected", None)
            pair_penalties_applied = bool(self._apply_pair_rewards(layer, candidate_pool_info.get("penalized_pairs", []), iter_id))
            if int(candidate_pool_info.get("coverage_hard_reject_count", 0) or 0) > 0:
                self.coverage_hard_reject_count += int(candidate_pool_info.get("coverage_hard_reject_count", 0) or 0)
            if bool(getattr(self.cfg, "resource_candidate_pool_log", True)):
                self.opt.candidate_iter_log.extend(candidate_rows)

            if selected_candidate is not None:
                destroy_name = str(selected_candidate.get("destroy_operator", ""))
                repair_name = str(selected_candidate.get("repair_operator", ""))
                if str(layer).upper() == "X" and (
                    str(destroy_name) in {"x_destroy_order_repartition", "x_destroy_critical_order_cluster"}
                    or bool(selected_candidate.get("repartition_mode", False))
                ):
                    self.last_x_repartition_iter = int(iter_id)
                fallback_used = bool(selected_candidate.get("fallback_used", False))
                projection_mode = str(selected_candidate.get("projection_mode", ""))
                projection_count = int(selected_candidate.get("projection_repaired_subtask_count", 0))
                candidate_eval = selected_candidate["candidate_eval"]
                candidate_payload = selected_candidate["candidate_payload"]
                candidate_config = candidate_payload["config"]
                candidate_signature = candidate_payload["config"].validation_signature()
                used_exact_eval_cache = bool(candidate_eval.metadata.get("used_exact_eval_cache", False))
                exact_eval_cache_hit_count = int(candidate_eval.metadata.get("exact_eval_cache_hit_count", getattr(self.scorer, "exact_eval_cache_hit_count", 0)))
                coverage_hard_reject = bool(not getattr(candidate_eval, "coverage_feasible", True))
                unmet_sku_total = int(getattr(candidate_eval, "unmet_sku_total", 0) or 0)
                force_validation_cap = max(1, int(getattr(self.cfg, "resource_force_exact_validation_subtask_cap", 6)))
                if not math.isfinite(float(candidate_eval.F_raw)):
                    candidate_hard_reject_reason = str(
                        candidate_eval.metadata.get("fixgurobi_infeasible_reason", "nonfinite_candidate_eval")
                    )
                    coverage_hard_reject = True
                    reward = -6.0
                elif bool(coverage_hard_reject):
                    candidate_hard_reject_reason = "coverage_hard_reject"
                    reward = -6.0
                else:
                    if (
                        not self.fixgurobi_only_eval
                        and str(layer).upper() in {"X", "XYZ"}
                        and len(set(int(x) for x in (candidate_payload.get("affected_ids", set()) or set()))) <= int(force_validation_cap)
                    ):
                        precomputed_validation = self.validator.validate(candidate_config, iter_id)
                        precomputed_validation["_trigger_reason"] = "force_exact_validation"
                        if str(precomputed_validation.get("hard_reject_reason", "") or ""):
                            coverage_hard_reject = True
                            candidate_hard_reject_reason = str(precomputed_validation.get("hard_reject_reason", "") or "pre_accept_validation_fail")
                            reward = -6.0
                    if (not self.fixgurobi_only_eval) and not bool(getattr(self.cfg, "resource_joint_colocated_sort_on_accepted_only", True)):
                        selected_candidate, precomputed_validation, joint_postprocess_stats, joint_postprocess_time_sec = self._maybe_apply_joint_colocated_sort_postprocess(
                            int(iter_id),
                            str(layer),
                            selected_candidate,
                        )
                        self._accumulate_joint_postprocess_stats(joint_postprocess_stats)
                        if selected_candidate is not None:
                            candidate_eval = selected_candidate["candidate_eval"]
                            candidate_payload = selected_candidate["candidate_payload"]
                            candidate_config = candidate_payload["config"]
                            candidate_signature = candidate_payload["config"].validation_signature()
                    accepted, accept_prob, effective_sa_temperature, acceptance_mode, lahc_threshold = self._accept_candidate(candidate_eval, layer)
                    x_temp_boost_used = bool(str(layer) == "X")
                if (
                    accepted
                    and not bool(coverage_hard_reject)
                    and (not self.fixgurobi_only_eval)
                    and bool(getattr(self.cfg, "resource_joint_colocated_sort_on_accepted_only", True))
                ):
                    selected_candidate, precomputed_validation, joint_postprocess_stats, joint_postprocess_time_sec = self._maybe_apply_joint_colocated_sort_postprocess(
                        int(iter_id),
                        str(layer),
                        selected_candidate,
                    )
                    self._accumulate_joint_postprocess_stats(joint_postprocess_stats)
                    if selected_candidate is not None:
                        candidate_eval = selected_candidate["candidate_eval"]
                        candidate_payload = selected_candidate["candidate_payload"]
                        candidate_config = candidate_payload["config"]
                        candidate_signature = candidate_payload["config"].validation_signature()
                if accepted and not bool(coverage_hard_reject):
                    pre_accept_config = self.current_config.clone()
                    pre_accept_eval = self.current_eval
                    self.current_config = candidate_config
                    self.current_eval = candidate_eval
                    if precomputed_validation is not None and precomputed_validation.get("snapshot", None) is not None:
                        self.opt.restore_snapshot(precomputed_validation["snapshot"])
                    reward = 1.0 if bool(fallback_used) else 3.0
                    if self.fixgurobi_only_eval:
                        validated_makespan = float(candidate_eval.F_raw)
                        validation_trigger = "fixgurobi_only"
                        val_time = float(candidate_eval.metadata.get("fixgurobi_solve_time", 0.0) or 0.0)
                        self.recent_validated_makespans.append(float(validated_makespan))
                        self.last_validation_iter = int(iter_id)
                        self.last_validation_f_raw = float(candidate_eval.F_raw)
                        self.last_validated_config = self.current_config.clone()
                        self.last_validated_signature = candidate_signature
                        self.opt._clear_z_detour_cache()
                        prev_best = float(self.best_validated.makespan)
                        if float(validated_makespan) + 1e-9 < float(prev_best):
                            materialized_snapshot = candidate_eval.metadata.get("fixgurobi_materialized_problem", None)
                            materialized_config = None
                            if materialized_snapshot is not None:
                                materialized_snapshot = copy.deepcopy(materialized_snapshot)
                                try:
                                    materialized_snapshot.z = float(validated_makespan)
                                    materialized_snapshot.iter_id = int(iter_id)
                                except Exception:
                                    pass
                                materialized_config = build_resource_config_from_problem(self.opt, materialized_snapshot)
                                self.current_config = materialized_config.clone()
                                self.last_validated_config = materialized_config.clone()
                                self.last_validated_signature = materialized_config.validation_signature()
                            self.best_validated = ValidatedIncumbent(
                                config=(materialized_config.clone() if materialized_config is not None else self.current_config.clone()),
                                makespan=float(validated_makespan),
                                iter_id=int(iter_id),
                                snapshot=materialized_snapshot,
                            )
                            if materialized_snapshot is not None:
                                self.opt.problem = copy.deepcopy(materialized_snapshot)
                                self.opt._rebuild_solvers()
                                if getattr(self.opt, "best", None) is not None:
                                    self.opt.best.problem_state = copy.deepcopy(materialized_snapshot)
                                    self.opt.best.subtask_state = None
                                if getattr(self.opt, "work", None) is not None:
                                    self.opt.work.problem_state = copy.deepcopy(materialized_snapshot)
                                    self.opt.work.subtask_state = None
                                self.opt.work_z = float(validated_makespan)
                            else:
                                self._sync_fixgurobi_best_snapshot(float(validated_makespan), int(iter_id))
                            improved_best = True
                            reward = 8.0
                        else:
                            reward = 6.0
                    else:
                        validation_trigger = self._candidate_validation_trigger(
                            int(iter_id),
                            candidate_eval,
                            candidate_signature,
                            precomputed_validation,
                        )
                        if validation_trigger == "periodic_skip_same_config":
                            self.last_validation_iter = int(iter_id)
                        elif validation_trigger:
                            t_val0 = time.perf_counter()
                            validation = precomputed_validation if precomputed_validation is not None else self.validator.validate(self.current_config, iter_id)
                            val_time = float(time.perf_counter() - t_val0) + float(joint_postprocess_time_sec)
                            self.opt.layer_runtime_sec_by_name["U"] = float(self.opt.layer_runtime_sec_by_name.get("U", 0.0)) + val_time
                            actual_lkh_calls = int(validation.get("validation_call_count", validation.get("lkh_call_count", 1)) or 0)
                            self.opt.global_eval_count = int(getattr(self.opt, "global_eval_count", 0)) + int(actual_lkh_calls)
                            self.lkh_call_count += int(actual_lkh_calls)
                            coverage_hard_reject = bool(validation.get("coverage_hard_reject", False))
                            unmet_sku_total = int(validation.get("unmet_sku_total", 0) or 0)
                            validation_hard_reject_reason = str(validation.get("hard_reject_reason", "") or "")
                            if str(validation_hard_reject_reason) == "coverage_hard_reject":
                                self.coverage_hard_reject_count += 1
                            if str(validation_hard_reject_reason):
                                repaired_config = None
                                repaired_validation = None
                                if str(validation_hard_reject_reason) == "unassigned_robot_task_hard_reject":
                                    repaired_config, repaired_validation = self._attempt_constrained_reentry(
                                        int(iter_id),
                                        previous_config=pre_accept_config,
                                        previous_eval=pre_accept_eval,
                                        candidate_config=self.current_config,
                                        validation=validation,
                                    )
                                if repaired_config is not None and repaired_validation is not None:
                                    validation = dict(repaired_validation)
                                    validation_hard_reject_reason = str(validation.get("hard_reject_reason", "") or "")
                                    self.current_config = repaired_config
                                    self.current_eval = self._evaluate_config(
                                        config=self.current_config,
                                        layer=str(layer),
                                        iterations_since_last_validation=int(iter_id) - int(self.last_validation_iter),
                                        distance_to_last_validated=config_distance(self.current_config, self.last_validated_config),
                                    )
                                    candidate_eval = self.current_eval
                                if str(validation_hard_reject_reason):
                                    candidate_hard_reject_reason = str(validation_hard_reject_reason)
                                    reward = -6.0
                                    self.current_config = self.best_validated.config.clone()
                                    self.last_validated_config = self.best_validated.config.clone()
                                    self.last_validated_signature = self.best_validated.config.validation_signature()
                                    self.opt._clear_z_detour_cache()
                                    self.current_eval = self._evaluate_config(
                                        config=self.current_config,
                                        layer="XYZ",
                                        iterations_since_last_validation=0,
                                        distance_to_last_validated=0.0,
                                    )
                                    accepted = False
                                    validation_trigger = str(validation_hard_reject_reason)
                                    validated_makespan = float("inf")
                            if not str(validation_hard_reject_reason):
                                guard_reason, guard_meta = self._tail_guard_reason(validation, self.current_config)
                                tail_guard_reason = str(guard_reason)
                                tail_guard_meta = dict(guard_meta or {})
                                if str(guard_reason):
                                    tail_guard_rejected = True
                                    candidate_hard_reject_reason = f"tail_guard:{guard_reason}"
                                    reward = -6.0
                                    self.current_config = self.best_validated.config.clone()
                                    self.last_validated_config = self.best_validated.config.clone()
                                    self.last_validated_signature = self.best_validated.config.validation_signature()
                                    self.opt._clear_z_detour_cache()
                                    self.current_eval = self._evaluate_config(
                                        config=self.current_config,
                                        layer="XYZ",
                                        iterations_since_last_validation=0,
                                        distance_to_last_validated=0.0,
                                    )
                                    accepted = False
                                    validation_trigger = f"tail_guard:{guard_reason}"
                                    validated_makespan = float(validation.get("makespan", float("inf")))
                            if not str(validation_hard_reject_reason) and not bool(tail_guard_rejected):
                                validated_makespan = float(validation["makespan"])
                                self.recent_validated_makespans.append(float(validated_makespan))
                                self.last_validation_iter = int(iter_id)
                                self.last_validation_f_raw = float(candidate_eval.F_raw)
                                self.last_validated_config = self.current_config.clone()
                                self.last_validated_signature = candidate_signature
                                self.opt._clear_z_detour_cache()
                                self.scorer.update_with_validation(candidate_eval, validated_makespan)
                                prev_best = float(self.best_validated.makespan)
                                if float(validated_makespan) + 1e-9 < float(prev_best):
                                    self.best_validated = ValidatedIncumbent(
                                        config=self.current_config.clone(),
                                        makespan=float(validated_makespan),
                                        iter_id=int(iter_id),
                                        snapshot=validation["snapshot"],
                                    )
                                    self.opt.best = validation["snapshot"]
                                    self.opt.work = validation["snapshot"]
                                    self.opt.work_z = float(validated_makespan)
                                    improved_best = True
                                    reward = 8.0
                                else:
                                    reward = 6.0
                                    catastrophic_threshold = self._catastrophic_threshold()
                                    if float(validated_makespan) > float(self.best_validated.makespan) * catastrophic_threshold + 1e-9:
                                        catastrophic_rollback = True
                                        reward = -6.0
                                        self.current_config = self.best_validated.config.clone()
                                        self.last_validated_config = self.best_validated.config.clone()
                                        self.last_validated_signature = self.best_validated.config.validation_signature()
                                        self.opt._clear_z_detour_cache()
                                        self.current_eval = self._evaluate_config(
                                            config=self.current_config,
                                            layer="XYZ",
                                            iterations_since_last_validation=0,
                                            distance_to_last_validated=0.0,
                                        )
                                        self.temperature = float(max(self.temperature, reheat * self.temperature))
            else:
                candidate_hard_reject_reason = str(candidate_pool_info.get("hard_reject_reason", "") or "no_candidate_pool")
                if int(candidate_pool_info.get("generated_count", 0)) <= 0:
                    empty_candidate_penalized = bool(self._apply_empty_candidate_failure(layer, candidate_pool_info.get("attempted_pairs", []), iter_id))
                    layer_cooldown_remaining = self._current_layer_cooldown_remaining(layer, iter_id + 1)
                    if bool(empty_candidate_penalized) or bool(pair_penalties_applied):
                        self._maybe_update_weights(layer, iter_id)
                elif bool(pair_penalties_applied):
                    self._maybe_update_weights(layer, iter_id)
            if str(layer).upper() == "Z" and (str(candidate_hard_reject_reason).startswith("exact_candidate_fail") or str(candidate_hard_reject_reason) == "no_candidate_pool"):
                self.z_exact_fail_streak = int(getattr(self, "z_exact_fail_streak", 0)) + 1
                threshold = int(getattr(self.cfg, "resource_z_exact_fail_force_threshold", 3))
                forced_queue = getattr(self, "forced_layer_queue", None)
                if forced_queue is None:
                    forced_queue = deque()
                    self.forced_layer_queue = forced_queue
                if threshold > 0 and int(self.z_exact_fail_streak) >= threshold and not forced_queue:
                    forced_queue.extend(["Y", "Z"])
                    self.z_exact_fail_streak = 0
            elif str(layer).upper() == "Z" and (bool(accepted) or int(candidate_pool_info.get("generated_count", 0) or 0) > 0):
                self.z_exact_fail_streak = 0
            if selected_candidate is not None:
                self._record_reward(layer, destroy_name, repair_name, reward, fallback_used, iter_id)
                self._maybe_update_weights(layer, iter_id)
            valid_candidate_scored = self._counts_as_effective_iteration(candidate_pool_info)
            stagnation_increment = float(self._stagnation_increment(valid_candidate_scored, used_exact_eval_cache, improved_best))
            if valid_candidate_scored:
                self._update_layer_progress(layer, accepted, prev_f_raw, float(candidate_eval.F_raw), stagnation_increment)
            self._update_failure_state(layer, accepted, improved_best, iter_id)
            self._update_exact_cache_funnel(used_exact_eval_cache, improved_best)
            x_failure_cooldown_remaining = self._current_failure_cooldown_remaining("X", iter_id + 1)

            current_known_z = float(validated_makespan) if validated_makespan == validated_makespan else float(self.best_validated.makespan)
            iter_runtime_sec = float(time.perf_counter() - t_iter0)
            row = build_iter_row(
                iter_id=iter_id,
                layer=layer,
                best_z=float(self.best_validated.makespan),
                current_z=float(current_known_z),
                accepted=bool(accepted),
                improved_best=bool(improved_best),
                eval_result=candidate_eval,
                destroy_operator=destroy_name,
                repair_operator=repair_name,
                fallback_used=bool(fallback_used),
                projection_mode=projection_mode,
                projection_repaired_subtask_count=int(projection_count),
                validation_trigger=validation_trigger,
                validated_makespan=float(validated_makespan),
                catastrophic_rollback=bool(catastrophic_rollback),
                lkh_budget_consumed_by_rollback=int(self.lkh_budget_consumed_by_rollback),
                extra={
                    "case": str(getattr(self.cfg, "scale", "")),
                    "prev_f_raw": float(prev_f_raw),
                    "local_obj": float(candidate_eval.F_raw),
                    "sa_temperature": float(effective_sa_temperature),
                    "sa_accept_prob": float(accept_prob),
                    "acceptance_mode": str(acceptance_mode),
                    "lahc_threshold": float(lahc_threshold),
                    "multi_start_restart_count": int(getattr(self, "multi_start_restart_count", 0)),
                    "iter_runtime_sec": float(iter_runtime_sec),
                    "global_eval_time_sec": float(val_time),
                    "operator_weight_snapshot": self._weight_snapshot(layer),
                    "global_z_before": float(self.best_validated.makespan),
                    "global_z_after": float(validated_makespan) if validated_makespan == validated_makespan else float(self.best_validated.makespan),
                    "lkh_call_count": int(self.lkh_call_count),
                    "search_scheme": "resource_time_alns",
                    "effective_destroy_mu": float(effective_destroy_mu),
                    "effective_destroy_budget": int(effective_destroy_budget),
                    "heavy_destroy_active": bool(heavy_destroy_active),
                    "destroy_tier": str(destroy_tier),
                    "force_rotate_used": bool(force_rotate_used),
                    "available_layers": "|".join(str(x) for x in getattr(self, "last_available_layers", [])),
                    "selected_layer_source": str(getattr(self, "last_selected_layer_source", "")),
                    "xyz_skip_reason": str(getattr(self, "last_xyz_skip_reason", "")),
                    "critical_path_operator_used": bool(selected_candidate.get("critical_path_operator_used", False)) if selected_candidate is not None else False,
                    "critical_path_subtask_ids": str(selected_candidate.get("critical_path_subtask_ids", [])) if selected_candidate is not None else "",
                    "x_temp_boost_used": bool(x_temp_boost_used),
                    "duplicate_tote_count": int(candidate_eval.duplicate_tote_count),
                    "duplicate_tote_penalty": float(candidate_eval.duplicate_tote_penalty),
                    "candidate_hard_reject_reason": str(candidate_hard_reject_reason),
                    "candidate_pool_target_size": int(candidate_pool_info.get("target_size", 0)),
                    "candidate_pool_generated_count": int(candidate_pool_info.get("generated_count", 0)),
                    "candidate_pool_unique_count": int(candidate_pool_info.get("unique_count", 0)),
                    "candidate_pool_exact_count": int(candidate_pool_info.get("exact_count", 0)),
                    "candidate_pool_attempt_count": int(candidate_pool_info.get("attempt_count", 0)),
                    "candidate_pool_exact_fail_count": int(candidate_pool_info.get("exact_fail_count", 0)),
                    "candidate_pool_exact_fail_reasons": str(candidate_pool_info.get("exact_fail_reasons", {}) or {}),
                    "candidate_pool_duplicate_hard_reject_count": int(candidate_pool_info.get("duplicate_hard_reject_count", 0)),
                    "z_structural_score": float(selected_candidate.get("z_structural_score", 0.0)) if selected_candidate is not None else 0.0,
                    "z_choke_over_soft": float(selected_candidate.get("z_choke_over_soft", 0.0)) if selected_candidate is not None else 0.0,
                    "z_station_load_soft": float(selected_candidate.get("z_station_load_soft", 0.0)) if selected_candidate is not None else 0.0,
                    "z_robot_region_load_soft": float(selected_candidate.get("z_robot_region_load_soft", 0.0)) if selected_candidate is not None else 0.0,
                    "candidate_pool_best_f_raw": float(selected_candidate.get("F_raw", float("nan"))) if selected_candidate is not None else float("nan"),
                    "candidate_pool_best_f_cal": float(selected_candidate.get("F_cal", float("nan"))) if selected_candidate is not None else float("nan"),
                    "selected_candidate_rank": int(selected_candidate.get("candidate_rank", 0)) if selected_candidate is not None else 0,
                    "global_eval_triggered": bool(validation_trigger not in ("", "periodic_skip_same_config")),
                    "tail_guard_rejected": bool(tail_guard_rejected),
                    "tail_guard_reason": str(tail_guard_reason),
                    "candidate_latest_robot_finish": float(tail_guard_meta.get("candidate_latest_robot_finish", 0.0) or 0.0),
                    "candidate_active_robot_count": int(tail_guard_meta.get("candidate_active_robot_count", 0) or 0),
                    "empty_candidate_penalized": bool(empty_candidate_penalized),
                    "joint_colocated_sort_postprocess_triggered": float(joint_postprocess_stats.get("triggered", 0.0)),
                    "joint_colocated_sort_postprocess_candidate_groups": float(joint_postprocess_stats.get("candidate_groups", 0.0)),
                    "joint_colocated_sort_postprocess_submitted": float(joint_postprocess_stats.get("submitted", 0.0)),
                    "joint_colocated_sort_postprocess_applied": float(joint_postprocess_stats.get("applied", 0.0)),
                    "joint_colocated_sort_postprocess_makespan_gain": float(joint_postprocess_stats.get("makespan_improvement", 0.0)),
                    "joint_colocated_sort_postprocess_rejected_capacity": float(joint_postprocess_stats.get("rejected_capacity", 0.0)),
                    "joint_colocated_sort_postprocess_rejected_interval_illegal": float(joint_postprocess_stats.get("rejected_interval_illegal", 0.0)),
                    "joint_colocated_sort_postprocess_rejected_noise": float(joint_postprocess_stats.get("rejected_noise", 0.0)),
                    "joint_colocated_sort_postprocess_rejected_eval_not_better": float(joint_postprocess_stats.get("rejected_eval_not_better", 0.0)),
                    "joint_colocated_sort_postprocess_rejected_validation": float(joint_postprocess_stats.get("rejected_validation", 0.0)),
                    "joint_colocated_sort_postprocess_rejected_target_conflict": float(joint_postprocess_stats.get("rejected_target_conflict", 0.0)),
                    "layer_cooldown_remaining": int(layer_cooldown_remaining),
                    "used_exact_eval_cache": bool(used_exact_eval_cache),
                    "exact_eval_cache_hit_count": int(exact_eval_cache_hit_count),
                    "coverage_hard_reject": bool(coverage_hard_reject),
                    "unmet_sku_total": int(unmet_sku_total),
                    "stagnation_increment": float(stagnation_increment),
                    "consecutive_exact_cache_hit_count": int(self.consecutive_exact_cache_hit_count),
                    "adaptive_destroy_bonus": float(self.adaptive_destroy_bonus),
                    "consecutive_fail_count_x": int(self.consecutive_fail_count.get("X", 0)),
                    "z_exact_fail_streak": int(getattr(self, "z_exact_fail_streak", 0)),
                    "forced_layer_queue_len": int(len(getattr(self, "forced_layer_queue", []))),
                    "x_layer_dynamic_multiplier": float(self.layer_dynamic_multiplier.get("X", 1.0)),
                    "x_failure_cooldown_remaining": int(x_failure_cooldown_remaining),
                    "validated_best_no_change_rounds": int(self.validated_best_no_change_rounds),
                    "revolving_enabled": bool(getattr(self.cfg, "resource_revolving_mode", False)),
                    "released_layer": str(candidate_eval.metadata.get("released_layer", str(layer).upper())),
                    "fixed_layers": str(candidate_eval.metadata.get("fixed_layers", "")),
                    "inner_relaxed_obj": float(candidate_eval.metadata.get("inner_relaxed_obj", float("nan"))),
                    "u_fast_cmax": float(candidate_eval.metadata.get("u_fast_cmax", float("nan"))),
                    "u_route_lb": float(candidate_eval.metadata.get("u_route_lb", float("nan"))),
                    "u_repair_time": float(candidate_eval.metadata.get("u_repair_time", 0.0) or 0.0),
                    "u_changed_robot_count": int(candidate_eval.metadata.get("u_changed_robot_count", 0) or 0),
                    "revolving_lb": float(candidate_eval.metadata.get("revolving_lb", float("nan"))),
                    "lb_gate_skipped": bool(candidate_eval.metadata.get("lb_gate_skipped", False)),
                    "outer_exact_obj": float(candidate_eval.metadata.get("fixgurobi_obj", float("nan"))) if str(candidate_eval.metadata.get("fixgurobi_obj", "")) != "" else float("nan"),
                    "outer_fixgurobi_time": float(candidate_eval.metadata.get("fixgurobi_solve_time", 0.0) or 0.0),
                    "cycle_id": int((int(iter_id) - 1) // max(1, len(getattr(self, "resource_layers", []) or [1]))),
                    "mark": int(self.validated_best_no_change_rounds),
                    "target_guidance_disabled": bool(getattr(self.cfg, "resource_revolving_mode", False) and not math.isfinite(float(getattr(self.cfg, "resource_target_cmax", float("nan"))))),
                },
            )
            self.opt.iter_log.append(row)
            self.opt.layer_runtime_sec_by_name[layer] = float(self.opt.layer_runtime_sec_by_name.get(layer, 0.0)) + float(iter_runtime_sec)
            self.opt.layer_trial_count_by_name[layer] = float(self.opt.layer_trial_count_by_name.get(layer, 0.0)) + 1.0
            self.temperature = float(max(1e-6, self.temperature * cooling))
            self._advance_lahc_history()
            self._refresh_operator_stats_payload()
            if abs(float(self.best_validated.makespan) - float(best_z_before_iter)) > 1e-9:
                self.no_best_z_change_rounds = 0.0
                self.validated_best_no_change_rounds = 0
            else:
                self.no_best_z_change_rounds = float(self.no_best_z_change_rounds) + float(stagnation_increment)
                self.validated_best_no_change_rounds = int(self.validated_best_no_change_rounds) + 1
            target_cmax = float(getattr(self.cfg, "resource_target_cmax", float("nan")))
            if math.isfinite(target_cmax) and float(self.best_validated.makespan) <= float(target_cmax) + 1e-9:
                self.opt.stop_reason = "target_reached"
                break
            hard_stop_rounds = int(getattr(
                self.cfg,
                "resource_stop_if_validated_best_no_change_rounds",
                getattr(self.cfg, "resource_stop_if_best_z_no_change_rounds", 50),
            ))
            if int(hard_stop_rounds) > 0 and int(self.validated_best_no_change_rounds) >= int(hard_stop_rounds):
                max_starts = max(1, int(getattr(self.cfg, "resource_multi_start_count", 1)))
                patience = int(getattr(self.cfg, "resource_multi_start_patience", 0))
                can_restart = int(getattr(self, "multi_start_restart_count", 0)) < int(max_starts - 1)
                if bool(can_restart) and (patience <= 0 or int(self.validated_best_no_change_rounds) >= int(patience)):
                    self._restart_current_search_state(int(iter_id))
                    continue
                self.opt.stop_reason = f"validated_best_no_change_{int(hard_stop_rounds)}"
                break

        self.opt.run_total_time_sec = float(self.opt._runtime_elapsed_sec())
        self.opt.coverage_hard_reject_count = int(self.coverage_hard_reject_count)
        self.opt.x_failure_decapitation_count = int(self.x_failure_decapitation_count)
        self.opt.consecutive_exact_cache_hit_count = int(self.consecutive_exact_cache_hit_count)
        self.opt.adaptive_destroy_bonus = float(self.adaptive_destroy_bonus)
        if not str(getattr(self.opt, "stop_reason", "") or ""):
            self.opt.stop_reason = "max_iters_reached"
        if self.fixgurobi_only_eval:
            z_polish_stats = {"enabled": False, "applied": False}
            polish_stats = {"enabled": False, "applied": False}
            z_polish_stats_after_y = {"enabled": False, "applied": False}
        else:
            z_polish_stats = self._best_z_sortify_polish()
            polish_stats = self._best_y_assignment_polish()
            z_polish_stats_after_y = self._best_z_sortify_polish()
        if bool(polish_stats.get("applied", False)):
            self.opt.stop_reason = f"{self.opt.stop_reason}+best_y_polish"
        if bool(z_polish_stats.get("applied", False)) or bool(z_polish_stats_after_y.get("applied", False)):
            self.opt.stop_reason = f"{self.opt.stop_reason}+best_z_sortify"
        if self.best_validated.snapshot is not None:
            snapshot = self.best_validated.snapshot
            if hasattr(snapshot, "problem_state"):
                self.opt.restore_snapshot(snapshot)
                self.opt.best = snapshot
                self.opt.work = snapshot
            else:
                self.opt.problem = copy.deepcopy(snapshot)
                self.opt._rebuild_solvers()
                self._sync_fixgurobi_best_snapshot(float(self.best_validated.makespan), int(self.best_validated.iter_id))
                if getattr(self.opt, "best", None) is not None:
                    self.opt.best.problem_state = copy.deepcopy(snapshot)
                    self.opt.best.subtask_state = None
                if getattr(self.opt, "work", None) is not None:
                    self.opt.work.problem_state = copy.deepcopy(snapshot)
                    self.opt.work.subtask_state = None
            self.opt.work_z = float(self.best_validated.makespan)
        self.opt._write_logs()
        return float(self.best_validated.makespan)
