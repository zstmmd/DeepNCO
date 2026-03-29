import os
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import shutil
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig

from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2LayerContext, SP2LocalSolveResult, SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
from Gurobi.sp4 import SP4_Robot_Router

from entity.calculate import GlobalTimeCalculator
from entity.subTask import SubTask


@dataclass
class TRARunConfig:
    scale: str = "SMALL"
    seed: int = 42
    max_iters: int = 50
    no_improve_limit: int = 3
    epsilon: float = 0.05

    # 求解策略（可按“前期快、后期精”切换）
    sp2_use_mip: bool = True
    sp3_use_mip: bool = False
    sp4_use_mip: bool = False
    sp2_time_limit_sec: float = 15.0
    sp4_lkh_time_limit_seconds: int = 120

    # “先启发式、后精确”切换（满足同一组约束：只是改变求解策略/时间上限）
    switch_to_exact_iter: int = 999999  # 迭代号达到后，开启更精确策略
    exact_sp2_use_mip: bool = True
    exact_sp3_use_mip: bool = True
    exact_sp4_use_mip: bool = False
    exact_sp2_time_limit_sec: float = 600.0
    exact_sp4_lkh_time_limit_seconds: int = 120

    # 输出
    log_dir: str = "log"
    export_best_solution: bool = True
    write_iteration_logs: bool = True

    # --- SP1 更外层反馈 ---
    enable_sp1_feedback_analysis: bool = True
    sp1_feedback_spread_threshold: int = 40  # 子任务涉及堆垛的空间跨度阈值
    sp1_feedback_top_k: int = 10

    # --- Soft coupling and affinity switches/params ---
    enable_sp3_precheck: bool = True
    sp3_precheck_use_mip: bool = False
    sp3_precheck_fail_action: str = "log"  # "log" | "abort"

    enable_soft_mu: bool = False
    enable_soft_pi: bool = False
    enable_soft_beta: bool = False
    enable_sku_affinity: bool = False

    # μ / π / β params
    mu_value: float = 1.0
    pi_scale: float = 1.0
    pi_clip: float = 120.0
    d0_threshold: float = 20.0
    beta_base: float = 1.0
    beta_gain: float = 1.0
    beta_min: float = 0.5
    beta_max: float = 3.0
    sp2_shadow_weight: float = 1.0

    # SKU affinity
    affinity_span_threshold: int = 40
    affinity_pairs_per_task: int = 3
    enable_role_vns: bool = True
    eps_skip: float = 0.05
    eps_light: float = 0.15
    weak_accept_eta: float = 0.02
    vns_max_trials: int = 10
    mode_fail_limit: int = 3
    mode_explore_bonus: float = 0.35
    mode_rotation_bonus: float = 0.15
    surrogate_top_k_light: int = 1
    surrogate_top_k_full: int = 2
    surrogate_prune_ratio: float = 0.25
    x_full_rebuild_subtask_threshold: int = 1
    search_scheme: str = "legacy"
    force_global_after_layer: str = "U"
    global_eval_period_layers: int = 4
    early_global_if_local_gain_ge: float = 0.05
    max_shadow_layers_without_global: int = 4
    global_weak_accept_eta: float = 0.01
    acceptance_mode: str = "strict_global"
    acceptance_rho_min: float = 0.05
    acceptance_min_actual_improve: float = 1e-6
    operator_selection_mode: str = "ucb1"
    layer_operator_budget_x: int = 4
    layer_operator_budget_y: int = 6
    layer_operator_budget_z: int = 4
    layer_operator_budget_u: int = 4
    layer_restart_patience: int = 2
    layer_shake_strength_init: int = 1
    layer_shake_strength_max: int = 3
    y_operator_topk: int = 3
    y_destroy_fraction: float = 0.25
    enable_fast_x_gate: bool = True
    fast_x_gate_margin: float = 0.0
    fast_x_arrival_weight: float = 4.0
    fast_x_station_drift_weight: float = 2.0
    fast_x_route_pressure_weight: float = 1.0
    fast_x_penalty_max: float = 20.0
    fast_x_arrival_shift_cap: float = 5.0
    fast_x_subtask_delta_weight: float = 3.0
    fast_x_soft_penalty_cap: float = 0.20
    fast_x_hard_penalty_cap: float = 0.60
    x_fast_borderline_gain_ratio: float = 0.03
    x_fast_soft_relax: float = 0.20
    x_fast_hard_relax: float = 0.60
    x_fast_arrival_shift_rel_cap: float = 0.35
    x_force_eval_period: int = 2
    x_affinity_update_decay: float = 0.85
    x_affinity_co_subtask_weight: float = 1.0
    x_affinity_route_weight: float = 0.5
    x_affinity_finish_time_weight: float = 0.5
    x_affinity_max_pairs_per_subtask: int = 20
    x_operator_pair_budget: int = 6
    x_destroy_size_min: int = 1
    x_destroy_size_max: int = 3
    x_low_affinity_destroy_fraction: float = 0.25
    x_random_repair_temperature: float = 0.15
    lambda_x_affinity: float = 0.5
    lambda_x_route: float = 0.5
    lambda_x_time: float = 0.5
    x_prox_weight: float = 0.25
    y_recent_signature_window: int = 3
    y_intensify_station_change_limit: int = 3
    y_intensify_rank_window: int = 2
    y_diversify_stagnation_rounds: int = 2
    y_fast_wait_weight: float = 0.35
    y_fast_queue_weight: float = 1.0
    y_fast_misalignment_cap: float = 30.0
    y_signature_record_mode: str = "global_eval_only"
    y_precheck_topk: int = 2
    enable_y_sp3_precheck: bool = True
    y_precheck_arrival_slack_weight: float = 2.0
    y_precheck_sorting_cost_weight: float = 1.0
    y_global_eval_topk: int = 2
    y_dual_eval_gap_ratio: float = 0.05
    y_block_move_size: int = 3
    y_destroy_fraction_min: float = 0.15
    y_destroy_fraction_mid: float = 0.25
    y_destroy_fraction_max: float = 0.40
    y_route_eval_mode: str = "replay_then_polish"
    y_route_eval_topk: int = 2
    y_route_eval_full_sp4_topk: int = 1
    y_route_arrival_weight: float = 1.0
    y_route_late_task_weight: float = 8.0
    y_route_load_balance_weight: float = 0.5
    y_incremental_route_subtask_cap: int = 4
    y_incremental_route_trip_cap: int = 3
    y_route_sim_cache_size: int = 16
    enable_triggered_zu_budget: bool = True
    z_trigger_station_load_std: float = 2.0
    z_trigger_noise_ratio: float = 0.20
    z_trigger_multi_stack_pen: float = 1.0
    u_trigger_arrival_slack_mean: float = 60.0
    u_trigger_late_task_count: int = 1
    u_default_budget_when_triggered: int = 1
    u_aggressive_trigger_arrival_slack_mean: float = 180.0
    u_aggressive_trigger_late_task_count: int = 24
    z_budget_after_streak_reject: int = 1
    z_hotspot_batch_size: int = 3
    z_destroy_fraction: float = 0.30
    z_min_budget: int = 2
    operator_reward_reject_surrogate: float = -1.0
    operator_reward_reject_global: float = -2.0
    operator_reward_pred_scale: float = 1.0
    operator_reward_actual_scale: float = 1.0
    lambda_init: float = 1.0
    lambda_step: float = 0.05
    lambda_decay: float = 0.95
    lambda_min: float = 0.05
    lambda_max: float = 10.0
    lambda_reject_ema_cap: float = 20.0
    lambda_reset_after_reject_streak: int = 3
    tau_x: float = 1.0
    tau_y: float = 1.0
    tau_z: float = 1.0
    tau_u: float = 1.0
    u_global_sp4_polish: bool = False
    progress_callback: Optional[Any] = None


@dataclass
class SolutionSnapshot:
    z: float
    iter_id: int
    seed: int
    subtask_station_rank: Dict[int, Tuple[int, int]]  # subtask_id -> (station_id, rank)
    sp1_capacity_limits: Dict[int, int]              # order_id -> cap
    sp1_incompatibility_pairs: List[Tuple[int, int]]
    subtask_state: Optional[List[Any]] = None
    problem_state: Optional[OFSProblemDTO] = None
    last_sp4_arrival_times: Optional[Dict[int, float]] = None
    last_sp3_tote_selection: Optional[Dict[int, List[int]]] = None
    last_sp3_sorting_costs: Optional[Dict[int, float]] = None
    last_station_start_times: Optional[Dict[int, float]] = None
    last_beta_value: Optional[float] = None


@dataclass
class FastYIntegratedEvalResult:
    objective_value: float
    approx_makespan: float
    station_cmax: float
    waiting_penalty: float
    queue_penalty: float
    arrival_misalignment_penalty: float
    load_balance_penalty: float
    station_preference_penalty: float
    prox_station_penalty: float
    prox_rank_penalty: float
    station_loads: Dict[int, float]
    station_finish_times: Dict[int, float]
    subtask_start_times: Dict[int, float]


@dataclass
class YPrecheckEvalResult:
    objective_value: float
    sorting_cost_delta: float
    station_cmax: float
    arrival_slack_delta: float


@dataclass
class YRouteSimEvalResult:
    objective_value: float
    route_arrival_score: float
    station_makespan: float
    global_makespan_proxy: float
    arrival_slack_mean: float
    late_task_count: float
    station_wait_total: float
    station_load_std: float
    replayed_sp4: bool
    used_incremental_route: bool


@dataclass
class FastXRolloutEvalResult:
    objective_value: float
    delta_station_load_drift: float
    delta_arrival_shift: float
    delta_route_pressure: float
    delta_subtask_count: float


@dataclass
class XSplitProposal:
    order_to_subtask_sku_sets: Dict[int, List[List[int]]]
    subtask_count: int
    sku_to_subtask_assignment: Dict[Tuple[int, int], int]
    touched_orders: Set[int]
    touched_subtask_count: int
    unassigned_skus: Dict[int, List[int]]


class RankAwareGlobalTimeCalculator(GlobalTimeCalculator):
    def _simulate_station_fcfs(self, all_tasks: List):
        station_to_tasks: Dict[int, List] = defaultdict(list)
        for task in all_tasks:
            station_to_tasks[int(getattr(task, "target_station_id", 0))].append(task)

        for station_id, tasks in station_to_tasks.items():
            station = self.problem.station_list[station_id]
            ordered_tasks = sorted(
                tasks,
                key=lambda t: (
                    int(getattr(t, "station_sequence_rank", -1)) if int(getattr(t, "station_sequence_rank", -1)) >= 0 else 10 ** 9,
                    float(getattr(t, "arrival_time_at_station", 0.0)),
                    int(getattr(t, "task_id", -1)),
                ),
            )

            for task in ordered_tasks:
                sku_count_in_task = self._calculate_sku_count(task)
                task.picking_duration = sku_count_in_task * self.t_pick

                extra_service = task.station_service_time if getattr(task, "noise_tote_ids", None) else 0.0
                total_process_duration = task.picking_duration + extra_service

                start_time = max(float(getattr(task, "arrival_time_at_station", 0.0)), station.next_available_time)
                if start_time > station.next_available_time:
                    station.total_idle_time += (start_time - station.next_available_time)

                task.tote_wait_time = start_time - float(getattr(task, "arrival_time_at_station", 0.0))
                task.start_process_time = start_time
                task.end_process_time = start_time + total_process_duration
                task.extra_service_used = float(extra_service)
                task.total_process_duration = float(total_process_duration)

                station.next_available_time = task.end_process_time
                station.processed_tasks.append(task)


class TRAOptimizer:
    """
    旋转外循环：
    - 每轮只重点优化一个阶段（SP4 / SP3 / SP2）
    - 每轮用 GlobalTimeCalculator 计算 z = global_makespan
    - 用近似 LB + ε 做跳过
    """

    def __init__(self, cfg: TRARunConfig):
        self.cfg = cfg
        self.problem: Optional[OFSProblemDTO] = None

        self.sp1: Optional[SP1_BOM_Splitter] = None
        self.sp2: Optional[SP2_Station_Assigner] = None
        self.sp3: Optional[SP3_Bin_Hitter] = None
        self.sp4: Optional[SP4_Robot_Router] = None
        self.sim: Optional[GlobalTimeCalculator] = None

        self.best: Optional[SolutionSnapshot] = None
        self.work: Optional[SolutionSnapshot] = None
        self.work_z: float = float("inf")
        self.anchor: Optional[SolutionSnapshot] = None
        self.shadow: Optional[SolutionSnapshot] = None
        self.anchor_z: float = float("inf")
        self.shadow_depth: int = 0
        self.shadow_last_layer: str = ""
        self.anchor_reference: Dict[str, Any] = {}
        self.layer_names: List[str] = ["X", "Y", "Z", "U"]
        self.layer_lambda_weights: Dict[str, float] = {
            key: float(self.cfg.lambda_init)
            for key in ["x_affinity", "x_route", "x_time", "yx", "yz", "yu", "zx", "zy", "zu", "uy", "uz"]
        }
        self.layer_residual_ema: Dict[str, float] = {key: 0.0 for key in self.layer_lambda_weights}
        self.x_sku_affinity: Dict[Tuple[int, int], float] = {}
        self.x_sku_affinity_last_iter: int = -1
        self._simulate_call_count: int = 0
        self._last_global_eval_iter: int = 0
        self.iter_log: List[Dict] = []
        self.mode_names: List[str] = ["M_R", "M_Y", "M_B", "M_X", "M_XYB"]
        self.mode_stats: Dict[str, Dict[str, float]] = {
            m: {"calls": 0.0, "success": 0.0, "fail": 0.0, "skip": 0.0, "last_gap": 1.0}
            for m in self.mode_names
        }
        self.last_selected_mode: Optional[str] = None

        # --- 旋转过程中用于“反馈耦合”的缓存 ---
        self.last_sp4_arrival_times: Dict[int, float] = {}
        self.last_sp3_tote_selection: Dict[int, List[int]] = {}
        self.last_sp3_sorting_costs: Dict[int, float] = {}
        self.last_sp2_local_result: Optional[SP2LocalSolveResult] = None
        # soft coupling caches
        self.last_station_start_times: Dict[int, float] = {}
        self.last_beta_value: Optional[float] = None
        # precheck results
        self.precheck_result: Optional[Dict] = None
        self.precheck_aborted: bool = False
        self.precheck_status: Optional[str] = None
        self.run_start_time_sec: float = 0.0
        self.run_total_time_sec: float = 0.0
        self.layer_runtime_sec_by_name: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.layer_trial_count_by_name: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.global_eval_count: int = 0
        self.operator_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.stagnation_stats: Dict[str, Dict[str, float]] = {}
        self.layer_operator_catalog: Dict[str, List[str]] = {}
        self.y_recent_signatures: Deque[str] = deque(maxlen=max(1, int(getattr(self.cfg, "y_recent_signature_window", 3))))
        self.layer_reject_surrogate_streak: Dict[str, int] = {}
        self.layer_global_reject_streak: Dict[str, int] = {}
        self.layer_fast_gate_reject_streak: Dict[str, int] = {}
        self.last_layer_accept: str = ""
        self._reset_runtime_caches()

    # ----------------------------
    # 基础设施
    # ----------------------------
    def _set_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _ensure_log_dir(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(root, self.cfg.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _log_path(self, filename: str) -> str:
        return os.path.join(self._ensure_log_dir(), filename)

    def _rebuild_solvers(self):
        assert self.problem is not None
        self.sp1 = SP1_BOM_Splitter(self.problem)
        self.sp2 = SP2_Station_Assigner(self.problem)
        self.sp3 = SP3_Bin_Hitter(self.problem)
        self.sp4 = SP4_Robot_Router(self.problem)
        self.sim = RankAwareGlobalTimeCalculator(self.problem)

    def _reset_runtime_caches(self):
        self.dirty_x = False
        self.dirty_y = False
        self.dirty_b = False
        self.dirty_r = False
        self.last_sp2_local_result = None
        self.cached_eval: Optional[float] = None
        self.cached_metrics: Dict[str, float] = {}
        self.cached_signature_by_layer: Dict[str, Tuple] = {}
        self.surrogate_stats: Dict[str, float] = {
            "evaluated": 0.0,
            "promoted": 0.0,
            "signature_skip": 0.0,
            "pruned": 0.0,
        }
        self.snapshot_time_sec = 0.0
        self.restore_time_sec = 0.0
        self.lightweight_snapshot_count = 0
        self.heavy_snapshot_count = 0

    def _set_dirty(self, x: bool = False, y: bool = False, b: bool = False, r: bool = False):
        self.dirty_x = self.dirty_x or bool(x)
        self.dirty_y = self.dirty_y or bool(y)
        self.dirty_b = self.dirty_b or bool(b)
        self.dirty_r = self.dirty_r or bool(r)

    def _clear_dirty(self):
        self.dirty_x = False
        self.dirty_y = False
        self.dirty_b = False
        self.dirty_r = False

    def _task_signature(self, task) -> Tuple:
        return (
            int(getattr(task, "sub_task_id", -1)),
            int(getattr(task, "target_stack_id", -1)),
            int(getattr(task, "target_station_id", -1)),
            str(getattr(task, "operation_mode", "")),
            tuple(sorted(int(tid) for tid in (getattr(task, "target_tote_ids", []) or []))),
            tuple(sorted(int(tid) for tid in (getattr(task, "hit_tote_ids", []) or []))),
            tuple(sorted(int(tid) for tid in (getattr(task, "noise_tote_ids", []) or []))),
            tuple(getattr(task, "sort_layer_range", None) or ()),
            int(getattr(task, "station_sequence_rank", -1)),
        )

    def _layer_signature(self, layer: str) -> Tuple:
        if self.problem is None:
            return tuple()
        layer = str(layer).upper()
        if layer == "X":
            rows = []
            for st in getattr(self.problem, "subtask_list", []) or []:
                rows.append((
                    int(getattr(st, "id", -1)),
                    int(getattr(getattr(st, "parent_order", None), "order_id", -1)),
                    tuple(sorted(int(getattr(sku, "id", -1)) for sku in (getattr(st, "sku_list", []) or []))),
                ))
            return tuple(sorted(rows))
        if layer == "Y":
            rows = []
            for st in getattr(self.problem, "subtask_list", []) or []:
                rows.append((
                    int(getattr(st, "id", -1)),
                    int(getattr(st, "assigned_station_id", -1)),
                    int(getattr(st, "station_sequence_rank", -1)),
                ))
            return tuple(sorted(rows))
        if layer == "B":
            rows = []
            for task in self._collect_all_tasks():
                rows.append(self._task_signature(task))
            return tuple(sorted(rows))
        if layer == "R":
            rows = []
            for task in self._collect_all_tasks():
                rows.append((
                    int(getattr(task, "task_id", -1)),
                    int(getattr(task, "robot_id", -1)),
                    int(getattr(task, "trip_id", 0)),
                    round(float(getattr(task, "arrival_time_at_stack", 0.0)), 6),
                    round(float(getattr(task, "arrival_time_at_station", 0.0)), 6),
                ))
            return tuple(sorted(rows))
        return tuple()

    def _sp1_feedback_signature(self) -> Tuple:
        caps = tuple(sorted((int(k), int(v)) for k, v in (getattr(self.sp1, "order_capacity_limits", {}) or {}).items()))
        incompat = tuple(sorted(tuple(int(x) for x in pair) for pair in (getattr(self.sp1, "incompatibility_pairs", set()) or set())))
        return caps, incompat

    def _capture_all_signatures(self) -> Dict[str, Tuple]:
        return {
            "X": self._layer_signature("X"),
            "XF": self._sp1_feedback_signature(),
            "Y": self._layer_signature("Y"),
            "B": self._layer_signature("B"),
            "R": self._layer_signature("R"),
        }

    def _sync_task_assignments_from_subtasks(self):
        for st in getattr(self.problem, "subtask_list", []) or []:
            station_id = int(getattr(st, "assigned_station_id", -1))
            rank = int(getattr(st, "station_sequence_rank", -1))
            for task in getattr(st, "execution_tasks", []) or []:
                task.target_station_id = station_id
                task.station_sequence_rank = rank

    def _rebuild_problem_task_list(self):
        all_tasks: List[Any] = []
        for st in getattr(self.problem, "subtask_list", []) or []:
            all_tasks.extend(getattr(st, "execution_tasks", []) or [])
        self.problem.task_list = all_tasks
        self.problem.task_num = len(all_tasks)

    def _compute_changed_orders(self, before_x: Tuple, after_x: Tuple) -> Set[int]:
        before_map = {int(item[0]): item for item in before_x}
        after_map = {int(item[0]): item for item in after_x}
        changed_orders: Set[int] = set()
        for sid in set(before_map.keys()) | set(after_map.keys()):
            if before_map.get(sid) != after_map.get(sid):
                row = after_map.get(sid) or before_map.get(sid)
                changed_orders.add(int(row[1]))
        return changed_orders

    def _derive_dirty_from_changes(self, mode: str, before_sig: Dict[str, Tuple], after_sig: Dict[str, Tuple]) -> Dict[str, Any]:
        changed = {name: before_sig.get(name) != after_sig.get(name) for name in ["X", "Y", "B", "R"]}
        changed["XF"] = before_sig.get("XF") != after_sig.get("XF")
        dirty = {"x": False, "y": False, "b": False, "r": False}
        if changed["X"] or changed["XF"]:
            dirty["x"] = True
            dirty["y"] = True
            dirty["b"] = True
            dirty["r"] = True
        elif changed["B"]:
            dirty["b"] = True
            dirty["r"] = True
        elif changed["Y"]:
            dirty["y"] = True
            dirty["r"] = True
        elif changed["R"]:
            dirty["r"] = True

        mode_upper = str(mode).upper()
        if mode_upper == "M_R":
            dirty["x"] = False
            dirty["b"] = False
            dirty["y"] = changed["Y"]
            dirty["r"] = changed["Y"] or changed["R"]
        elif mode_upper == "M_Y":
            dirty["x"] = False
            dirty["b"] = False
            dirty["r"] = changed["Y"] or changed["R"]
        elif mode_upper == "M_B":
            dirty["x"] = False
            dirty["y"] = False
            dirty["b"] = True
            dirty["r"] = True

        return {
            "changed": changed,
            "dirty": dirty,
            "changed_orders": self._compute_changed_orders(before_sig["X"], after_sig["X"]) if changed["X"] else set(),
            "signature_changed": any(changed.values()),
        }

    def _compute_surrogate_score(self, mode: str, baseline_metrics: Dict[str, float], trial_metrics: Dict[str, float], change_info: Dict[str, Any]) -> float:
        mode = str(mode).upper()
        changed = change_info.get("changed", {})
        penalty = 0.0
        if mode == "M_R":
            penalty += trial_metrics.get("robot_path_length_total", 0.0) - baseline_metrics.get("robot_path_length_total", 0.0)
            penalty += 40.0 * (trial_metrics.get("robot_finish_ratio", 0.0) - baseline_metrics.get("robot_finish_ratio", 0.0))
            penalty += 10.0 * (trial_metrics.get("arrival_slack_mean", 0.0) - baseline_metrics.get("arrival_slack_mean", 0.0))
        elif mode == "M_Y":
            penalty += 100.0 * (trial_metrics.get("station_load_max_ratio", 0.0) - baseline_metrics.get("station_load_max_ratio", 0.0))
            penalty += 0.2 * (trial_metrics.get("station_idle_total", 0.0) - baseline_metrics.get("station_idle_total", 0.0))
            penalty += 10.0 * (trial_metrics.get("arrival_slack_mean", 0.0) - baseline_metrics.get("arrival_slack_mean", 0.0))
        elif mode == "M_B":
            penalty += 120.0 * (trial_metrics.get("noise_ratio", 0.0) - baseline_metrics.get("noise_ratio", 0.0))
            penalty += 5.0 * (trial_metrics.get("avg_stack_span", 0.0) - baseline_metrics.get("avg_stack_span", 0.0))
            penalty += 0.5 * (trial_metrics.get("sorting_cost_proxy", 0.0) - baseline_metrics.get("sorting_cost_proxy", 0.0))
        elif mode == "M_X":
            penalty += 20.0 * (trial_metrics.get("avg_sku_per_subtask", 0.0) - baseline_metrics.get("avg_sku_per_subtask", 0.0))
            penalty += 30.0 * (trial_metrics.get("max_sku_per_subtask", 0.0) - baseline_metrics.get("max_sku_per_subtask", 0.0))
            penalty += 2.0 * (trial_metrics.get("subtask_count", 0.0) - baseline_metrics.get("subtask_count", 0.0))
        else:
            penalty += 15.0 * (trial_metrics.get("avg_sku_per_subtask", 0.0) - baseline_metrics.get("avg_sku_per_subtask", 0.0))
            penalty += 80.0 * (trial_metrics.get("station_load_max_ratio", 0.0) - baseline_metrics.get("station_load_max_ratio", 0.0))
            penalty += 120.0 * (trial_metrics.get("noise_ratio", 0.0) - baseline_metrics.get("noise_ratio", 0.0))

        if changed.get("X"):
            penalty += 5.0 * len(change_info.get("changed_orders", set()))
        if not change_info.get("signature_changed", False):
            penalty += 1e9
        return float(penalty)

    def _refresh_runtime_cache(self, z: Optional[float] = None):
        self.cached_eval = None if z is None else float(z)
        self.cached_metrics = dict(self._collect_layer_metrics())
        self.cached_signature_by_layer = self._capture_all_signatures()
        self._clear_dirty()

    def _sync_tasks_and_maybe_rebuild(self, dirty: Dict[str, bool], change_info: Dict[str, Any]) -> int:
        full_rebuild_called = 0
        self._set_dirty(**dirty)

        if self.dirty_x:
            full_rebuild_called = 1
            self._run_sp1()
            if self.cfg.sp2_use_mip:
                self._run_sp2_mip()
            else:
                self._run_sp2_initial()
            self._run_sp3()
            self._run_sp4()
            self._rebuild_problem_task_list()
            return full_rebuild_called

        if self.dirty_y:
            self._sync_task_assignments_from_subtasks()

        if self.dirty_b:
            full_rebuild_called = 1
            self._run_sp3()

        if self.dirty_y or self.dirty_b or self.dirty_r:
            self._run_sp4()

        self._rebuild_problem_task_list()
        return full_rebuild_called

    def _mode_roles(self, mode: str) -> Dict[str, str]:
        roles = {
            "M_R": {"X": "Frozen", "Y": "Anchored", "B": "Frozen", "R": "Active"},
            "M_Y": {"X": "Frozen", "Y": "Active", "B": "Frozen", "R": "Anchored"},
            "M_B": {"X": "Frozen", "Y": "Frozen", "B": "Active", "R": "Anchored"},
            "M_X": {"X": "Active", "Y": "Anchored", "B": "Anchored", "R": "Frozen"},
            "M_XYB": {"X": "Active", "Y": "Active", "B": "Anchored", "R": "Frozen"},
        }
        return roles.get(mode, {"X": "Frozen", "Y": "Frozen", "B": "Frozen", "R": "Frozen"})

    def _trust_region_tau(self, layer: str) -> float:
        layer = str(layer).upper()
        if layer == "X":
            return float(self.cfg.tau_x)
        if layer == "Y":
            return float(self.cfg.tau_y)
        if layer == "Z":
            return float(self.cfg.tau_z)
        return float(self.cfg.tau_u)

    def _layer_coupling_keys(self, layer: str) -> List[str]:
        layer = str(layer).upper()
        if layer == "X":
            return ["x_affinity", "x_route", "x_time"]
        if layer == "Y":
            return ["yx", "yu", "yz"]
        if layer == "Z":
            return ["zx", "zy", "zu"]
        return ["uy", "uz"]

    def _flatten_lambda_weights(self) -> Dict[str, float]:
        return {f"lambda_{k}": float(v) for k, v in self.layer_lambda_weights.items()}

    def _init_layer_augmented_state(self, z0: float):
        self.anchor = self.snapshot(z0, iter_id=0, lightweight=True)
        self.shadow = self.snapshot(z0, iter_id=0, lightweight=True)
        self.anchor_z = float(z0)
        self.shadow_depth = 0
        self.shadow_last_layer = ""
        self._last_global_eval_iter = 0
        self.anchor_reference = {}
        self.y_recent_signatures = deque(maxlen=max(1, int(getattr(self.cfg, "y_recent_signature_window", 3))))
        self.layer_reject_surrogate_streak = {layer: 0 for layer in self.layer_names}
        self.layer_global_reject_streak = {layer: 0 for layer in self.layer_names}
        self.layer_fast_gate_reject_streak = {layer: 0 for layer in self.layer_names}
        self.last_layer_accept = ""
        self.x_sku_affinity = {}
        self.x_sku_affinity_last_iter = -1
        self.y_route_sim_cache = {}
        self._refresh_anchor_reference()
        self.x_destroy_catalog = [
            "x_destroy_random_subtasks",
            "x_destroy_low_affinity_subtasks",
            "x_destroy_route_conflict_subtasks",
            "x_destroy_time_window_outliers",
            "x_destroy_order_boundary_merge_split",
        ]
        self.x_repair_catalog = [
            "x_repair_affinity_greedy",
            "x_repair_finish_time_cluster",
            "x_repair_route_cluster",
            "x_repair_randomized_best_fit",
        ]
        self.layer_operator_catalog = {
            "X": list(self.x_destroy_catalog) + list(self.x_repair_catalog),
            "Y": [
                "station_reassign_single",
                "station_swap_pair",
                "rank_reinsert_within_station",
                "cross_station_reinsert",
                "station_block_relocate",
                "order_block_reinsert",
                "congested_station_destroy_repair",
                "order_cohesion_destroy_repair",
            ],
            "Z": [
                "stack_replace",
                "tote_replace_within_stack",
                "mode_flip_sort_toggle",
                "range_shrink_expand",
                "task_merge_split",
                "z_hotspot_destroy_repair",
            ],
            "U": [
                "u_trip_relocate",
                "u_trip_swap",
                "u_segment_reverse",
                "u_late_task_pull_forward",
                "u_cross_robot_relocate",
                "u_trip_split_merge",
            ],
        }
        self.operator_stats = {}
        self.stagnation_stats = {}
        for layer, ops in self.layer_operator_catalog.items():
            self.operator_stats[layer] = {}
            self.stagnation_stats[layer] = {
                "no_improve_rounds": 0.0,
                "shake_strength": float(max(1, int(self.cfg.layer_shake_strength_init))),
                "restart_triggered": 0.0,
                "round_index": 0.0,
            }
            for op in ops:
                self.operator_stats[layer][op] = {
                    "pulls": 0.0,
                    "reward_mean": 0.0,
                    "last_improve": -1.0,
                    "last_global_accept": -1.0,
                }

    def _runtime_elapsed_sec(self) -> float:
        if self.run_start_time_sec <= 0.0:
            return 0.0
        return float(max(0.0, time.perf_counter() - self.run_start_time_sec))

    def _runtime_stats_payload(self) -> Dict[str, Any]:
        reward_summary: Dict[str, Dict[str, float]] = {}
        for layer, rows in (self.operator_stats or {}).items():
            reward_summary[str(layer)] = {
                str(op): float(meta.get("reward_mean", 0.0)) for op, meta in rows.items()
            }
        return {
            "run_start_time_sec": float(self.run_start_time_sec),
            "run_total_time_sec": float(self.run_total_time_sec if self.run_total_time_sec > 0.0 else self._runtime_elapsed_sec()),
            "layer_runtime_sec_by_name": {str(k): float(v) for k, v in self.layer_runtime_sec_by_name.items()},
            "layer_trial_count_by_name": {str(k): float(v) for k, v in self.layer_trial_count_by_name.items()},
            "global_eval_count": int(self.global_eval_count),
            "simulate_call_count": int(self._simulate_call_count),
            "snapshot_time_sec": float(getattr(self, "snapshot_time_sec", 0.0)),
            "restore_time_sec": float(getattr(self, "restore_time_sec", 0.0)),
            "lightweight_snapshot_count": int(getattr(self, "lightweight_snapshot_count", 0)),
            "heavy_snapshot_count": int(getattr(self, "heavy_snapshot_count", 0)),
            "layer_augmented_acceptance_mode": str(getattr(self.cfg, "acceptance_mode", "strict_global")),
            "operator_selection_mode": str(getattr(self.cfg, "operator_selection_mode", "ucb1")),
            "operator_reward_mean_by_layer": reward_summary,
            "layer_augmented_ignored_legacy_commit_controls": [
                "global_eval_period_layers",
                "max_shadow_layers_without_global",
                "global_weak_accept_eta",
            ],
            "layer_augmented_deprecated_y_precheck_controls": [
                "enable_y_sp3_precheck",
                "y_precheck_topk",
                "y_precheck_arrival_slack_weight",
                "y_precheck_sorting_cost_weight",
            ],
            "layer_augmented_deprecated_fast_x_absolute_caps": [
                "fast_x_soft_penalty_cap",
                "fast_x_hard_penalty_cap",
                "fast_x_arrival_shift_cap",
                "fast_x_penalty_max",
                "fast_x_gate_margin",
            ],
        }

    def _compute_y_assignment_signature(self) -> str:
        rows = []
        for st in sorted(getattr(self.problem, "subtask_list", []) or [], key=lambda item: int(getattr(item, "id", -1))):
            rows.append((
                int(getattr(st, "id", -1)),
                int(getattr(st, "assigned_station_id", -1)),
                int(getattr(st, "station_sequence_rank", -1)),
            ))
        return repr(tuple(rows))

    def _compute_y_change_counts(self) -> Dict[str, int]:
        anchor_station = {}
        anchor_rank = {}
        for st in self._iter_snapshot_subtasks(self.anchor):
            sid = int(getattr(st, "id", -1))
            anchor_station[sid] = int(getattr(st, "assigned_station_id", -1))
            anchor_rank[sid] = int(getattr(st, "station_sequence_rank", -1))
        station_change_count = 0
        rank_change_count = 0
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            curr_station = int(getattr(st, "assigned_station_id", -1))
            curr_rank = int(getattr(st, "station_sequence_rank", -1))
            if curr_station != int(anchor_station.get(sid, curr_station)):
                station_change_count += 1
            if curr_rank != int(anchor_rank.get(sid, curr_rank)):
                rank_change_count += abs(curr_rank - int(anchor_rank.get(sid, curr_rank)))
        return {
            "station_change_count": int(station_change_count),
            "rank_change_count": int(rank_change_count),
        }

    def _collect_changed_y_subtasks(self) -> List[int]:
        changed: List[int] = []
        anchor_station = {}
        anchor_rank = {}
        for st in self._iter_snapshot_subtasks(self.anchor):
            sid = int(getattr(st, "id", -1))
            anchor_station[sid] = int(getattr(st, "assigned_station_id", -1))
            anchor_rank[sid] = int(getattr(st, "station_sequence_rank", -1))
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            curr_station = int(getattr(st, "assigned_station_id", -1))
            curr_rank = int(getattr(st, "station_sequence_rank", -1))
            if curr_station != int(anchor_station.get(sid, curr_station)) or curr_rank != int(anchor_rank.get(sid, curr_rank)):
                changed.append(sid)
        return changed

    def _collect_affected_y_robot_trips(self, changed_subtask_ids: List[int]) -> Set[Tuple[int, int]]:
        changed_set = {int(x) for x in changed_subtask_ids}
        affected: Set[Tuple[int, int]] = set()
        for task in self._collect_all_tasks():
            if int(getattr(task, "sub_task_id", -1)) in changed_set:
                rid = int(getattr(task, "robot_id", -1))
                trip = int(getattr(task, "trip_id", 0))
                if rid >= 0:
                    affected.add((rid, trip))
        return affected

    def _compute_late_task_count(self) -> int:
        late = 0
        for st in getattr(self.problem, "subtask_list", []) or []:
            start_ref = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            for task in getattr(st, "execution_tasks", []) or []:
                if float(getattr(task, "arrival_time_at_station", 0.0)) > start_ref + 1e-9:
                    late += 1
        return int(late)

    def _compute_priority_z_multi_stack_pen(self, limit: int = 2) -> float:
        rows: List[float] = []
        for sid in self._select_priority_z_subtasks(limit=limit):
            st = next((item for item in (getattr(self.problem, "subtask_list", []) or []) if int(getattr(item, "id", -1)) == int(sid)), None)
            if st is None:
                continue
            subtask_stack_ids = {
                int(getattr(task, "target_stack_id", -1))
                for task in getattr(st, "execution_tasks", []) or []
                if int(getattr(task, "target_stack_id", -1)) >= 0
            }
            rows.append(max(0.0, float(len(subtask_stack_ids)) - 1.0))
        return float(max(rows) if rows else 0.0)

    def _iter_snapshot_subtasks(self, snap: Optional[SolutionSnapshot]) -> List[Any]:
        if snap is None:
            return []
        if snap.subtask_state is not None:
            return list(snap.subtask_state or [])
        if snap.problem_state is not None:
            return list(getattr(snap.problem_state, "subtask_list", []) or [])
        return []

    def _iter_snapshot_tasks(self, snap: Optional[SolutionSnapshot]) -> List[Any]:
        tasks: List[Any] = []
        for st in self._iter_snapshot_subtasks(snap):
            tasks.extend(getattr(st, "execution_tasks", []) or [])
        return tasks

    def _layer_operator_budget(self, layer: str) -> int:
        layer = str(layer).upper()
        base_budget = {
            "X": max(1, int(getattr(self.cfg, "layer_operator_budget_x", 4))),
            "Y": max(1, int(getattr(self.cfg, "layer_operator_budget_y", 6))),
            "Z": max(1, int(getattr(self.cfg, "layer_operator_budget_z", 4))),
            "U": max(1, int(getattr(self.cfg, "layer_operator_budget_u", 4))),
        }.get(layer, 1)
        if bool(getattr(self.cfg, "enable_triggered_zu_budget", True)):
            metrics = self._collect_layer_metrics() if self.problem is not None else {}
            if layer == "Z":
                trigger_reasons: List[str] = []
                if float(metrics.get("station_load_std", 0.0)) >= float(getattr(self.cfg, "z_trigger_station_load_std", 2.0)):
                    trigger_reasons.append("station_load_std")
                if float(metrics.get("noise_ratio", 0.0)) >= float(getattr(self.cfg, "z_trigger_noise_ratio", 0.20)):
                    trigger_reasons.append("noise_ratio")
                if self._compute_priority_z_multi_stack_pen(limit=2) >= float(getattr(self.cfg, "z_trigger_multi_stack_pen", 1.0)):
                    trigger_reasons.append("multi_stack_pen")
                self.current_trigger_gate = {
                    "layer": "Z",
                    "open": bool(trigger_reasons),
                    "reason": ",".join(trigger_reasons) if trigger_reasons else "below_threshold",
                }
                if not trigger_reasons:
                    return 0
                if int(self.layer_reject_surrogate_streak.get("Z", 0)) >= 2:
                    return max(
                        int(getattr(self.cfg, "z_min_budget", 2)),
                        int(getattr(self.cfg, "z_budget_after_streak_reject", 1)),
                    )
                if float(self.stagnation_stats.get("Z", {}).get("restart_triggered", 0.0)) > 0.0:
                    return max(int(getattr(self.cfg, "z_min_budget", 2)), min(base_budget, 3))
            if layer == "U":
                late_task_count = self._compute_late_task_count()
                trigger_reasons = []
                if float(metrics.get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "u_trigger_arrival_slack_mean", 60.0)):
                    trigger_reasons.append("arrival_slack_mean")
                if int(late_task_count) >= int(getattr(self.cfg, "u_trigger_late_task_count", 1)):
                    trigger_reasons.append("late_task_count")
                if str(self.last_layer_accept).upper() == "Y":
                    trigger_reasons.append("post_y_accept")
                self.current_trigger_gate = {
                    "layer": "U",
                    "open": bool(trigger_reasons),
                    "reason": ",".join(trigger_reasons) if trigger_reasons else "below_threshold",
                    "late_task_count": int(late_task_count),
                }
                if not trigger_reasons:
                    return 0
                strong_trigger = (
                    float(metrics.get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "u_aggressive_trigger_arrival_slack_mean", 180.0))
                    or int(late_task_count) >= int(getattr(self.cfg, "u_aggressive_trigger_late_task_count", 24))
                )
                self.current_trigger_gate["strong"] = bool(strong_trigger)
                if str(self.last_layer_accept).upper() != "Y" and not strong_trigger:
                    return 0
                return max(1, int(getattr(self.cfg, "u_default_budget_when_triggered", 1)))
        return base_budget

    def _select_operator_sequence(self, layer: str, budget: int) -> List[str]:
        layer = str(layer).upper()
        ops = list(self.layer_operator_catalog.get(layer, []))
        if not ops:
            return []
        stats = self.operator_stats.setdefault(layer, {})
        total_pulls = sum(float(stats.get(op, {}).get("pulls", 0.0)) for op in ops)
        selected: List[str] = []
        unseen = [op for op in ops if float(stats.get(op, {}).get("pulls", 0.0)) <= 0.0]
        while len(selected) < int(budget):
            if unseen:
                op = unseen.pop(0)
                selected.append(op)
                continue
            best_op = ops[0]
            best_score = -float("inf")
            total_term = math.log(max(2.0, total_pulls + 1.0))
            for op in ops:
                row = stats.get(op, {})
                pulls = max(1.0, float(row.get("pulls", 0.0)))
                mean = float(row.get("reward_mean", 0.0))
                score = mean + math.sqrt(2.0 * total_term / pulls)
                if score > best_score:
                    best_score = score
                    best_op = op
            selected.append(best_op)
            total_pulls += 1.0
        return selected

    def _record_operator_reward(self, layer: str, operator: str, pred_reward: float, actual_reward: float, iter_id: int, accepted: bool, status: str = "") -> float:
        layer = str(layer).upper()
        operator = str(operator)
        if not operator:
            return 0.0
        row = self.operator_stats.setdefault(layer, {}).setdefault(operator, {
            "pulls": 0.0,
            "reward_mean": 0.0,
            "last_improve": -1.0,
            "last_global_accept": -1.0,
        })
        row["pulls"] = float(row.get("pulls", 0.0)) + 1.0
        clip_pred = max(0.0, min(float(pred_reward), 50.0))
        clip_actual = max(0.0, min(float(actual_reward), 200.0))
        if status == "accept":
            reward = float(clip_pred + clip_actual)
        elif status == "reject_global":
            reward = float(-20.0 - min(abs(float(actual_reward)), 200.0))
        elif status == "reject_surrogate":
            reward = float(-5.0 - min(max(0.0, float(pred_reward)), 50.0))
        else:
            reward = 0.0
        old_mean = float(row.get("reward_mean", 0.0))
        row["reward_mean"] = old_mean + (reward - old_mean) / max(1.0, float(row["pulls"]))
        if pred_reward > 1e-9:
            row["last_improve"] = float(iter_id)
        if accepted:
            row["last_global_accept"] = float(iter_id)
        return float(reward)

    def _apply_layer_stagnation_update(self, layer: str, improved: bool):
        layer = str(layer).upper()
        row = self.stagnation_stats.setdefault(layer, {
            "no_improve_rounds": 0.0,
            "shake_strength": float(max(1, int(self.cfg.layer_shake_strength_init))),
            "restart_triggered": 0.0,
            "round_index": 0.0,
        })
        row["round_index"] = float(row.get("round_index", 0.0)) + 1.0
        if improved:
            row["no_improve_rounds"] = 0.0
            row["restart_triggered"] = 0.0
            row["shake_strength"] = float(max(1, int(self.cfg.layer_shake_strength_init)))
            return
        row["no_improve_rounds"] = float(row.get("no_improve_rounds", 0.0)) + 1.0
        patience = max(1, int(getattr(self.cfg, "layer_restart_patience", 2)))
        if row["no_improve_rounds"] >= float(patience):
            row["restart_triggered"] = 1.0
            row["shake_strength"] = float(min(
                int(getattr(self.cfg, "layer_shake_strength_max", 3)),
                int(row.get("shake_strength", 1.0)) + 1,
            ))
        else:
            row["restart_triggered"] = 0.0

    def _layer_shake_strength(self, layer: str) -> int:
        return max(1, int(self.stagnation_stats.get(str(layer).upper(), {}).get("shake_strength", float(self.cfg.layer_shake_strength_init))))

    def _clone_x_split_proposal(self, proposal: XSplitProposal) -> XSplitProposal:
        return copy.deepcopy(proposal)

    def _refresh_x_split_proposal_metadata(self, proposal: XSplitProposal):
        assignment: Dict[Tuple[int, int], int] = {}
        subtask_count = 0
        touched_subtask_count = 0
        for order_id, groups in (proposal.order_to_subtask_sku_sets or {}).items():
            clean_groups: List[List[int]] = []
            for group_idx, group in enumerate(groups):
                uniq_group = sorted({int(sku_id) for sku_id in group if int(sku_id) >= 0})
                if not uniq_group:
                    continue
                clean_groups.append(uniq_group)
                subtask_count += 1
                if int(order_id) in proposal.touched_orders:
                    touched_subtask_count += 1
                for sku_id in uniq_group:
                    assignment[(int(order_id), int(sku_id))] = int(group_idx)
            proposal.order_to_subtask_sku_sets[int(order_id)] = clean_groups
        proposal.subtask_count = int(subtask_count)
        proposal.sku_to_subtask_assignment = assignment
        proposal.touched_subtask_count = int(touched_subtask_count)
        for order_id, rows in list((proposal.unassigned_skus or {}).items()):
            proposal.unassigned_skus[int(order_id)] = sorted({int(sku_id) for sku_id in rows if int(sku_id) >= 0})

    def _extract_x_split_solution(self) -> XSplitProposal:
        order_to_subtask_sku_sets: Dict[int, List[List[int]]] = defaultdict(list)
        seen_orders: Set[int] = set()
        for st in getattr(self.problem, "subtask_list", []) or []:
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            if order_id < 0:
                continue
            seen_orders.add(order_id)
            sku_ids = sorted({int(getattr(sku, "id", -1)) for sku in getattr(st, "unique_sku_list", []) or [] if int(getattr(sku, "id", -1)) >= 0})
            if sku_ids:
                order_to_subtask_sku_sets[order_id].append(sku_ids)
        for order in getattr(self.problem, "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            if order_id in seen_orders:
                continue
            fallback_ids = sorted({int(getattr(sku, "id", -1)) for sku in getattr(order, "unique_sku_list", []) or [] if int(getattr(sku, "id", -1)) >= 0})
            if fallback_ids:
                order_to_subtask_sku_sets[order_id].append(fallback_ids)
        proposal = XSplitProposal(
            order_to_subtask_sku_sets={int(k): [list(group) for group in rows] for k, rows in order_to_subtask_sku_sets.items()},
            subtask_count=0,
            sku_to_subtask_assignment={},
            touched_orders=set(),
            touched_subtask_count=0,
            unassigned_skus={},
        )
        self._refresh_x_split_proposal_metadata(proposal)
        return proposal

    def _get_order_unique_sku_ids(self, order_id: int) -> List[int]:
        order = next((row for row in (getattr(self.problem, "order_list", []) or []) if int(getattr(row, "order_id", -1)) == int(order_id)), None)
        if order is None:
            return []
        return sorted({int(getattr(sku, "id", -1)) for sku in getattr(order, "unique_sku_list", []) or [] if int(getattr(sku, "id", -1)) >= 0})

    def _get_order_capacity_limit(self, order_id: int) -> int:
        default_cap = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 6)))
        if self.sp1 is None:
            return default_cap
        return max(1, int(getattr(self.sp1, "order_capacity_limits", {}).get(int(order_id), default_cap)))

    def _x_sku_feature_map(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        features: Dict[Tuple[int, int], Dict[str, float]] = {}
        robot_route_pos: Dict[int, Dict[int, float]] = defaultdict(dict)
        robot_task_rows: Dict[int, List[Any]] = defaultdict(list)
        for task in self._collect_all_tasks():
            rid = int(getattr(task, "robot_id", -1))
            if rid >= 0:
                robot_task_rows[rid].append(task)
        for rid, rows in robot_task_rows.items():
            rows.sort(key=lambda item: (float(getattr(item, "arrival_time_at_station", 0.0)), int(getattr(item, "task_id", -1))))
            for pos, task in enumerate(rows):
                robot_route_pos[rid][int(getattr(task, "sub_task_id", -1))] = float(pos)
        for st in getattr(self.problem, "subtask_list", []) or []:
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            if order_id < 0:
                continue
            task_rows = list(getattr(st, "execution_tasks", []) or [])
            robot_ids = sorted({int(getattr(task, "robot_id", -1)) for task in task_rows if int(getattr(task, "robot_id", -1)) >= 0})
            route_positions = []
            for rid in robot_ids:
                route_positions.append(float(robot_route_pos.get(rid, {}).get(int(getattr(st, "id", -1)), 0.0)))
            completion = float(getattr(st, "completion_time", 0.0))
            if completion <= 0.0:
                completion = float(self._estimate_subtask_start(st) + self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
            for sku in getattr(st, "unique_sku_list", []) or []:
                sku_id = int(getattr(sku, "id", -1))
                if sku_id < 0:
                    continue
                features[(order_id, sku_id)] = {
                    "completion": float(completion),
                    "robot_id": float(robot_ids[0]) if robot_ids else -1.0,
                    "route_pos": float(sum(route_positions) / len(route_positions)) if route_positions else 0.0,
                    "subtask_id": float(getattr(st, "id", -1)),
                }
        return features

    def _proposal_group_mean_affinity(self, order_id: int, sku_ids: List[int]) -> float:
        rows = [int(sid) for sid in sku_ids if int(sid) >= 0]
        if len(rows) <= 1:
            return 1.0
        pair_scores: List[float] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = sorted((rows[i], rows[j]))
                pair_scores.append(float(self.x_sku_affinity.get((a, b), 0.0)))
        return float(sum(pair_scores) / len(pair_scores)) if pair_scores else 1.0

    def _normalize_x_split_proposal(self, proposal: XSplitProposal):
        for order_id in list((proposal.order_to_subtask_sku_sets or {}).keys()):
            groups = [sorted({int(sku_id) for sku_id in group if int(sku_id) >= 0}) for group in proposal.order_to_subtask_sku_sets.get(order_id, [])]
            groups = [group for group in groups if group]
            assigned = {sku_id for group in groups for sku_id in group}
            missing = [sku_id for sku_id in self._get_order_unique_sku_ids(int(order_id)) if int(sku_id) not in assigned and int(sku_id) not in set(proposal.unassigned_skus.get(int(order_id), []))]
            if missing:
                proposal.unassigned_skus.setdefault(int(order_id), []).extend(missing)
            proposal.order_to_subtask_sku_sets[int(order_id)] = groups
        self._refresh_x_split_proposal_metadata(proposal)

    def _compute_x_destroy_size(self, group_size: int, strength: int) -> int:
        lo = max(1, int(getattr(self.cfg, "x_destroy_size_min", 1)))
        hi = max(lo, int(getattr(self.cfg, "x_destroy_size_max", 3)))
        hi = min(hi, max(1, int(group_size)))
        if hi <= lo:
            return hi
        return max(lo, min(hi, lo + max(0, int(strength) - 1)))

    def _apply_x_destroy_operator(self, proposal: XSplitProposal, operator: str, rng: random.Random, strength: int) -> int:
        self._normalize_x_split_proposal(proposal)
        features = self._x_sku_feature_map()
        changed = 0
        if operator == "x_destroy_order_boundary_merge_split":
            candidate_orders = [(oid, rows) for oid, rows in proposal.order_to_subtask_sku_sets.items() if len(rows) >= 2]
            if not candidate_orders:
                return 0
            order_id, rows = min(candidate_orders, key=lambda item: min(len(group) for group in item[1]))
            rows = sorted(rows, key=lambda group: (len(group), self._proposal_group_mean_affinity(order_id, group)))
            merge_rows = rows[:2]
            merged = sorted({sku_id for group in merge_rows for sku_id in group})
            proposal.order_to_subtask_sku_sets[order_id] = [group for group in proposal.order_to_subtask_sku_sets.get(order_id, []) if group not in merge_rows]
            proposal.unassigned_skus.setdefault(order_id, []).extend(merged)
            proposal.touched_orders.add(int(order_id))
            changed = len(merged)
            self._refresh_x_split_proposal_metadata(proposal)
            return int(changed)

        all_groups: List[Tuple[int, int, List[int]]] = []
        for order_id, groups in proposal.order_to_subtask_sku_sets.items():
            for group_idx, group in enumerate(groups):
                if group:
                    all_groups.append((int(order_id), int(group_idx), list(group)))
        if not all_groups:
            return 0

        if operator == "x_destroy_random_subtasks":
            group_count = min(len(all_groups), self._compute_x_destroy_size(len(all_groups), strength))
            selected = rng.sample(all_groups, group_count)
        elif operator == "x_destroy_low_affinity_subtasks":
            selected = sorted(
                all_groups,
                key=lambda item: (
                    self._proposal_group_mean_affinity(item[0], item[2]),
                    len(item[2]),
                    item[0],
                    item[1],
                ),
            )[: max(1, min(len(all_groups), int(strength)))]
        elif operator == "x_destroy_route_conflict_subtasks":
            def route_conflict(item: Tuple[int, int, List[int]]) -> float:
                positions = [float(features.get((item[0], sku_id), {}).get("route_pos", 0.0)) for sku_id in item[2]]
                robots = {int(features.get((item[0], sku_id), {}).get("robot_id", -1)) for sku_id in item[2]}
                return (max(positions) - min(positions) if positions else 0.0) + max(0.0, float(len([rid for rid in robots if rid >= 0]) - 1))
            selected = sorted(all_groups, key=route_conflict, reverse=True)[: max(1, min(len(all_groups), int(strength)))]
        else:
            def time_outlier(item: Tuple[int, int, List[int]]) -> float:
                comps = [float(features.get((item[0], sku_id), {}).get("completion", 0.0)) for sku_id in item[2]]
                return (max(comps) - min(comps)) if comps else 0.0
            selected = sorted(all_groups, key=time_outlier, reverse=True)[: max(1, min(len(all_groups), int(strength)))]

        destroy_fraction = max(0.05, float(getattr(self.cfg, "x_low_affinity_destroy_fraction", 0.25)))
        for order_id, group_idx, group in selected:
            current_group = list(proposal.order_to_subtask_sku_sets.get(order_id, [])[group_idx])
            destroy_n = max(1, min(len(current_group), int(math.ceil(len(current_group) * destroy_fraction))))
            if operator == "x_destroy_low_affinity_subtasks":
                ordered = sorted(
                    current_group,
                    key=lambda sku_id: self._proposal_group_mean_affinity(order_id, [sid for sid in current_group if sid != sku_id]),
                )
            elif operator == "x_destroy_route_conflict_subtasks":
                ordered = sorted(current_group, key=lambda sku_id: float(features.get((order_id, sku_id), {}).get("route_pos", 0.0)))
                if len(ordered) >= 2:
                    ordered = [ordered[0], ordered[-1]] + ordered[1:-1]
            elif operator == "x_destroy_time_window_outliers":
                group_mean = sum(float(features.get((order_id, sku_id), {}).get("completion", 0.0)) for sku_id in current_group) / max(1, len(current_group))
                ordered = sorted(current_group, key=lambda sku_id: abs(float(features.get((order_id, sku_id), {}).get("completion", group_mean)) - group_mean), reverse=True)
            else:
                ordered = list(current_group)
                rng.shuffle(ordered)
            removed = ordered[:destroy_n]
            remain = [sku_id for sku_id in current_group if sku_id not in set(removed)]
            proposal.order_to_subtask_sku_sets[order_id][group_idx] = remain
            proposal.unassigned_skus.setdefault(order_id, []).extend(removed)
            proposal.touched_orders.add(int(order_id))
            changed += len(removed)
        self._normalize_x_split_proposal(proposal)
        return int(changed)

    def _x_group_route_score(self, order_id: int, group: List[int], sku_id: int, features: Dict[Tuple[int, int], Dict[str, float]]) -> float:
        if not group:
            return 0.0
        sku_feature = features.get((order_id, sku_id), {})
        sku_robot = int(sku_feature.get("robot_id", -1))
        sku_pos = float(sku_feature.get("route_pos", 0.0))
        total = 0.0
        for other in group:
            other_feature = features.get((order_id, int(other)), {})
            other_robot = int(other_feature.get("robot_id", -1))
            other_pos = float(other_feature.get("route_pos", 0.0))
            same_robot_bonus = 1.0 if sku_robot >= 0 and sku_robot == other_robot else 0.0
            total += same_robot_bonus + 1.0 / (1.0 + abs(sku_pos - other_pos))
        return float(total / max(1, len(group)))

    def _x_group_time_score(self, order_id: int, group: List[int], sku_id: int, features: Dict[Tuple[int, int], Dict[str, float]]) -> float:
        if not group:
            return 0.0
        sku_completion = float(features.get((order_id, sku_id), {}).get("completion", 0.0))
        total = 0.0
        for other in group:
            other_completion = float(features.get((order_id, int(other)), {}).get("completion", sku_completion))
            total += 1.0 / (1.0 + abs(sku_completion - other_completion))
        return float(total / max(1, len(group)))

    def _x_repair_score(self, order_id: int, group: List[int], sku_id: int, features: Dict[Tuple[int, int], Dict[str, float]], mode: str) -> float:
        affinity_score = float(sum(self.x_sku_affinity.get(tuple(sorted((int(sku_id), int(other)))), 0.0) for other in group)) / max(1, len(group))
        route_score = self._x_group_route_score(order_id, group, sku_id, features)
        time_score = self._x_group_time_score(order_id, group, sku_id, features)
        if mode == "affinity":
            return affinity_score + 0.25 * route_score + 0.25 * time_score
        if mode == "finish":
            return time_score + 0.25 * affinity_score + 0.25 * route_score
        if mode == "route":
            return route_score + 0.25 * affinity_score + 0.25 * time_score
        return 0.5 * affinity_score + 0.25 * route_score + 0.25 * time_score

    def _apply_x_repair_operator(self, proposal: XSplitProposal, operator: str, rng: random.Random, strength: int) -> bool:
        self._normalize_x_split_proposal(proposal)
        features = self._x_sku_feature_map()
        changed = False
        for order_id in sorted(list((proposal.unassigned_skus or {}).keys())):
            pending = list(sorted({int(sku_id) for sku_id in proposal.unassigned_skus.get(order_id, []) if int(sku_id) >= 0}))
            if not pending:
                continue
            groups = proposal.order_to_subtask_sku_sets.setdefault(int(order_id), [])
            capacity = self._get_order_capacity_limit(int(order_id))
            while pending:
                sku_id = pending.pop(0)
                feasible_groups = [group for group in groups if len(group) < capacity]
                mode = "random"
                if operator == "x_repair_affinity_greedy":
                    mode = "affinity"
                elif operator == "x_repair_finish_time_cluster":
                    mode = "finish"
                elif operator == "x_repair_route_cluster":
                    mode = "route"
                ranked: List[Tuple[float, Optional[List[int]]]] = []
                for group in feasible_groups:
                    ranked.append((self._x_repair_score(order_id, group, sku_id, features, mode), group))
                ranked.sort(key=lambda item: item[0], reverse=True)
                target_group: Optional[List[int]] = None
                if ranked:
                    if operator == "x_repair_randomized_best_fit" and len(ranked) > 1:
                        temperature = max(0.01, float(getattr(self.cfg, "x_random_repair_temperature", 0.15)))
                        k = min(len(ranked), max(1, int(math.ceil(len(ranked) * temperature * 4.0))))
                        target_group = rng.choice([row[1] for row in ranked[:k]])
                    else:
                        target_group = ranked[0][1]
                if target_group is None:
                    groups.append([int(sku_id)])
                else:
                    target_group.append(int(sku_id))
                    target_group[:] = sorted({int(row) for row in target_group})
                proposal.touched_orders.add(int(order_id))
                changed = True
            proposal.unassigned_skus[int(order_id)] = []
        self._normalize_x_split_proposal(proposal)
        return bool(changed)

    def _project_x_split_solution_to_problem(self, proposal: XSplitProposal):
        sub_tasks: List[Any] = []
        next_subtask_id = 0
        for order in getattr(self.problem, "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            groups = list((proposal.order_to_subtask_sku_sets or {}).get(order_id, []))
            if not groups:
                fallback = self._get_order_unique_sku_ids(order_id)
                groups = [fallback] if fallback else []
            sku_groups: Dict[int, List[Any]] = defaultdict(list)
            for sku_id in getattr(order, "order_product_id_list", []) or []:
                sku_obj = self.problem.id_to_sku.get(int(sku_id))
                if sku_obj is not None:
                    sku_groups[int(sku_id)].append(sku_obj)
            for group in groups:
                sku_list: List[Any] = []
                for sku_id in group:
                    sku_list.extend(list(sku_groups.get(int(sku_id), [])))
                if not sku_list:
                    continue
                sub_tasks.append(SubTask(
                    id=int(next_subtask_id),
                    parent_order=order,
                    sku_list=sku_list,
                ))
                next_subtask_id += 1
        self.problem.subtask_list = sub_tasks
        self.problem.subtask_num = len(sub_tasks)

    def _compute_x_affinity_matrix_from_current_solution(self, iter_id: int):
        decay = max(0.0, min(1.0, float(getattr(self.cfg, "x_affinity_update_decay", 0.85))))
        max_pairs = max(1, int(getattr(self.cfg, "x_affinity_max_pairs_per_subtask", 20)))
        co_weight = float(getattr(self.cfg, "x_affinity_co_subtask_weight", 1.0))
        route_weight = float(getattr(self.cfg, "x_affinity_route_weight", 0.5))
        finish_weight = float(getattr(self.cfg, "x_affinity_finish_time_weight", 0.5))
        pair_scores: Dict[Tuple[int, int], float] = defaultdict(float)
        pair_counts: Dict[Tuple[int, int], float] = defaultdict(float)
        for st in getattr(self.problem, "subtask_list", []) or []:
            sku_ids = sorted({int(getattr(sku, "id", -1)) for sku in getattr(st, "unique_sku_list", []) or [] if int(getattr(sku, "id", -1)) >= 0})
            completion = float(getattr(st, "completion_time", 0.0))
            if completion <= 0.0:
                completion = float(self._estimate_subtask_start(st) + self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
            pairs_seen = 0
            for i in range(len(sku_ids)):
                for j in range(i + 1, len(sku_ids)):
                    if pairs_seen >= max_pairs:
                        break
                    key = (int(sku_ids[i]), int(sku_ids[j]))
                    pair_scores[key] += co_weight + finish_weight * (1.0 / (1.0 + completion))
                    pair_counts[key] += 1.0
                    pairs_seen += 1
                if pairs_seen >= max_pairs:
                    break
        robot_rows: Dict[int, List[Any]] = defaultdict(list)
        for task in self._collect_all_tasks():
            rid = int(getattr(task, "robot_id", -1))
            if rid >= 0:
                robot_rows[rid].append(task)
        for rid, rows in robot_rows.items():
            rows.sort(key=lambda task: (float(getattr(task, "arrival_time_at_station", 0.0)), int(getattr(task, "task_id", -1))))
            for idx, task in enumerate(rows):
                sid = int(getattr(task, "sub_task_id", -1))
                st = next((row for row in (getattr(self.problem, "subtask_list", []) or []) if int(getattr(row, "id", -1)) == sid), None)
                if st is None:
                    continue
                left = sorted({int(getattr(sku, "id", -1)) for sku in getattr(st, "unique_sku_list", []) or [] if int(getattr(sku, "id", -1)) >= 0})
                for next_idx in range(idx + 1, min(len(rows), idx + 3)):
                    next_sid = int(getattr(rows[next_idx], "sub_task_id", -1))
                    next_st = next((row for row in (getattr(self.problem, "subtask_list", []) or []) if int(getattr(row, "id", -1)) == next_sid), None)
                    if next_st is None:
                        continue
                    right = sorted({int(getattr(sku, "id", -1)) for sku in getattr(next_st, "unique_sku_list", []) or [] if int(getattr(sku, "id", -1)) >= 0})
                    bonus = route_weight * (1.0 / (1.0 + float(next_idx - idx)))
                    for a in left:
                        for b in right:
                            if a == b:
                                continue
                            key = tuple(sorted((int(a), int(b))))
                            pair_scores[key] += bonus
                            pair_counts[key] += 1.0
        new_affinity: Dict[Tuple[int, int], float] = {}
        for key, value in pair_scores.items():
            new_affinity[key] = float(value / max(1.0, pair_counts.get(key, 1.0)))
        if new_affinity:
            max_value = max(new_affinity.values()) or 1.0
            new_affinity = {key: float(min(1.0, value / max_value)) for key, value in new_affinity.items()}
        merged: Dict[Tuple[int, int], float] = {}
        all_keys = set(self.x_sku_affinity.keys()) | set(new_affinity.keys())
        for key in all_keys:
            prev = float(self.x_sku_affinity.get(key, 0.0))
            curr = float(new_affinity.get(key, 0.0))
            merged[key] = float(decay * prev + (1.0 - decay) * curr)
        self.x_sku_affinity = merged
        self.x_sku_affinity_last_iter = int(iter_id)

    def _score_x_split_proposal(self, proposal: Optional[XSplitProposal] = None) -> Dict[str, float]:
        features = self._x_sku_feature_map()
        current = proposal if proposal is not None else self._extract_x_split_solution()
        self._refresh_x_split_proposal_metadata(current)
        low_affinity_penalty = 0.0
        route_conflict_penalty = 0.0
        finish_time_dispersion_penalty = 0.0
        for order_id, groups in (current.order_to_subtask_sku_sets or {}).items():
            for group in groups:
                group = sorted({int(sku_id) for sku_id in group if int(sku_id) >= 0})
                if not group:
                    continue
                mean_affinity = self._proposal_group_mean_affinity(int(order_id), group)
                low_affinity_penalty += max(0.0, 1.0 - mean_affinity)
                route_positions = [float(features.get((int(order_id), sku_id), {}).get("route_pos", 0.0)) for sku_id in group]
                completions = [float(features.get((int(order_id), sku_id), {}).get("completion", 0.0)) for sku_id in group]
                robots = {int(features.get((int(order_id), sku_id), {}).get("robot_id", -1)) for sku_id in group if int(features.get((int(order_id), sku_id), {}).get("robot_id", -1)) >= 0}
                if route_positions:
                    route_conflict_penalty += max(route_positions) - min(route_positions)
                route_conflict_penalty += max(0.0, float(len(robots) - 1))
                if completions:
                    finish_time_dispersion_penalty += (max(completions) - min(completions)) / max(1.0, float(self.anchor_z if math.isfinite(self.anchor_z) else 1.0))
        prox_penalty = float(current.touched_subtask_count) * float(getattr(self.cfg, "x_prox_weight", 0.25))
        local_obj = float(current.subtask_count)
        coupling_penalty = (
            float(self.layer_lambda_weights.get("x_affinity", float(getattr(self.cfg, "lambda_x_affinity", 0.5)))) * float(low_affinity_penalty)
            + float(self.layer_lambda_weights.get("x_route", float(getattr(self.cfg, "lambda_x_route", 0.5)))) * float(route_conflict_penalty)
            + float(self.layer_lambda_weights.get("x_time", float(getattr(self.cfg, "lambda_x_time", 0.5)))) * float(finish_time_dispersion_penalty)
        )
        augmented_obj = float(local_obj + coupling_penalty + float(self._trust_region_tau("X")) * prox_penalty)
        return {
            "local_obj": float(local_obj),
            "coupling_penalty": float(coupling_penalty),
            "prox_penalty": float(prox_penalty),
            "augmented_obj": float(augmented_obj),
            "x_affinity_penalty": float(low_affinity_penalty),
            "x_route_conflict_penalty": float(route_conflict_penalty),
            "x_finish_time_dispersion_penalty": float(finish_time_dispersion_penalty),
            "x_subtask_count": float(current.subtask_count),
            "x_touched_subtask_count": float(current.touched_subtask_count),
            "couplings": {
                "x_affinity": float(low_affinity_penalty),
                "x_route": float(route_conflict_penalty),
                "x_time": float(finish_time_dispersion_penalty),
            },
        }

    def _select_x_operator_pairs(self, budget: int) -> List[Tuple[str, str]]:
        destroy_ops = list(getattr(self, "x_destroy_catalog", []))
        repair_ops = list(getattr(self, "x_repair_catalog", []))
        if not destroy_ops or not repair_ops:
            return []
        destroy_sequence = self._select_operator_sequence("X", max(1, int(budget)))
        destroy_sequence = [op for op in destroy_sequence if op in destroy_ops]
        repair_sequence = self._select_operator_sequence("X", max(1, int(budget)))
        repair_sequence = [op for op in repair_sequence if op in repair_ops]
        if not destroy_sequence:
            destroy_sequence = destroy_ops[:max(1, int(budget))]
        if not repair_sequence:
            repair_sequence = repair_ops[:max(1, int(budget))]
        pairs: List[Tuple[str, str]] = []
        pair_budget = max(1, int(getattr(self.cfg, "x_operator_pair_budget", budget)))
        for idx in range(pair_budget):
            pairs.append((
                str(destroy_sequence[idx % len(destroy_sequence)]),
                str(repair_sequence[idx % len(repair_sequence)]),
            ))
        return pairs

    def _refresh_anchor_reference(self):
        if self.anchor is None:
            self.anchor_reference = {}
            return

        subtasks = self._iter_snapshot_subtasks(self.anchor)
        if not subtasks:
            self.anchor_reference = {}
            return
        ref: Dict[str, Any] = {
            "order_major_station": {},
            "order_station_counts": {},
            "station_subtask_count": {},
            "order_subtask_count": {},
            "order_proc_avg": {},
            "order_arrival_avg": {},
            "order_start_avg": {},
            "order_slack_avg": {},
            "order_route_pressure": {},
            "subtask_proc": {},
            "subtask_arrival": {},
            "subtask_start": {},
            "subtask_slack": {},
            "stack_route_cost": {},
            "anchor_task_count": 0,
            "anchor_stack_set": set(),
            "anchor_mode_set": set(),
            "robot_path_length": 0.0,
        }
        order_station_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        station_subtask_count: Dict[int, int] = defaultdict(int)
        order_proc_rows: Dict[int, List[float]] = defaultdict(list)
        order_arrival_rows: Dict[int, List[float]] = defaultdict(list)
        order_start_rows: Dict[int, List[float]] = defaultdict(list)
        order_slack_rows: Dict[int, List[float]] = defaultdict(list)
        order_route_rows: Dict[int, List[float]] = defaultdict(list)
        stack_route_cost: Dict[int, List[float]] = defaultdict(list)

        anchor_tasks: List[Any] = []
        for st in subtasks:
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            sid = int(getattr(st, "assigned_station_id", -1))
            if oid >= 0:
                ref["order_subtask_count"][oid] = ref["order_subtask_count"].get(oid, 0) + 1
            if sid >= 0:
                station_subtask_count[sid] += 1
                if oid >= 0:
                    order_station_counts[oid][sid] += 1

            proc_val = 0.0
            arr_val = 0.0
            start_val = float(getattr(st, "estimated_process_start_time", 0.0))
            start_found = False
            route_val = 0.0
            for task in getattr(st, "execution_tasks", []) or []:
                anchor_tasks.append(task)
                ref["anchor_stack_set"].add(int(getattr(task, "target_stack_id", -1)))
                ref["anchor_mode_set"].add(str(getattr(task, "operation_mode", "")))
                pick = float(getattr(task, "total_process_duration", 0.0))
                if pick <= 1e-9:
                    pick = float(getattr(task, "picking_duration", 0.0)) + float(getattr(task, "station_service_time", 0.0))
                if pick <= 1e-9:
                    pick = len(getattr(st, "sku_list", []) or []) * float(OFSConfig.PICKING_TIME)
                proc_val += pick
                arr_val = max(arr_val, float(getattr(task, "arrival_time_at_station", 0.0)))
                task_start = float(getattr(task, "start_process_time", 0.0))
                if task_start > 0.0:
                    start_val = task_start if not start_found else min(start_val, task_start)
                    start_found = True
                stack_id = int(getattr(task, "target_stack_id", -1))
                if stack_id >= 0:
                    stack_route_cost[stack_id].append(
                        float(getattr(task, "arrival_time_at_stack", 0.0)) + float(getattr(task, "robot_service_time", 0.0))
                    )
                route_val += float(getattr(task, "robot_service_time", 0.0))
                route_val += float(getattr(task, "arrival_time_at_stack", 0.0))

            if proc_val <= 1e-9:
                proc_val = len(getattr(st, "sku_list", []) or []) * float(OFSConfig.PICKING_TIME)
            slack_val = max(0.0, start_val - arr_val)
            ref["subtask_proc"][int(getattr(st, "id", -1))] = float(proc_val)
            ref["subtask_arrival"][int(getattr(st, "id", -1))] = float(arr_val)
            ref["subtask_start"][int(getattr(st, "id", -1))] = float(start_val)
            ref["subtask_slack"][int(getattr(st, "id", -1))] = float(slack_val)

            if oid >= 0:
                order_proc_rows[oid].append(float(proc_val))
                order_arrival_rows[oid].append(float(arr_val))
                order_start_rows[oid].append(float(start_val))
                order_slack_rows[oid].append(float(slack_val))
                order_route_rows[oid].append(float(route_val))

        for oid, station_counts in order_station_counts.items():
            ref["order_station_counts"][oid] = dict(station_counts)
            if station_counts:
                ref["order_major_station"][oid] = max(station_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        ref["station_subtask_count"] = dict(station_subtask_count)
        ref["anchor_task_count"] = int(len(anchor_tasks))
        for oid, rows in order_proc_rows.items():
            ref["order_proc_avg"][oid] = float(sum(rows) / len(rows)) if rows else 0.0
        for oid, rows in order_arrival_rows.items():
            ref["order_arrival_avg"][oid] = float(sum(rows) / len(rows)) if rows else 0.0
        for oid, rows in order_start_rows.items():
            ref["order_start_avg"][oid] = float(sum(rows) / len(rows)) if rows else 0.0
        for oid, rows in order_slack_rows.items():
            ref["order_slack_avg"][oid] = float(sum(rows) / len(rows)) if rows else 0.0
        for oid, rows in order_route_rows.items():
            ref["order_route_pressure"][oid] = float(sum(rows) / len(rows)) if rows else 0.0
        ref["stack_route_cost"] = {
            sid: float(sum(rows) / len(rows)) if rows else 0.0 for sid, rows in stack_route_cost.items()
        }
        ref["robot_path_length"] = float(self._compute_robot_path_length_from_tasks(anchor_tasks))
        self.anchor_reference = ref

    def _estimate_subtask_processing_time(
        self,
        st: Any,
        sorting_costs: Optional[Dict[int, float]] = None,
        fallback: Optional[float] = None,
    ) -> float:
        sid = int(getattr(st, "id", -1))
        oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
        if sorting_costs and sid in sorting_costs:
            return len(getattr(st, "sku_list", []) or []) * float(OFSConfig.PICKING_TIME) + float(sorting_costs[sid])
        if sid in self.anchor_reference.get("subtask_proc", {}):
            return float(self.anchor_reference["subtask_proc"][sid])
        if oid in self.anchor_reference.get("order_proc_avg", {}):
            return float(self.anchor_reference["order_proc_avg"][oid])
        if fallback is not None:
            return float(fallback)
        return len(getattr(st, "sku_list", []) or []) * float(OFSConfig.PICKING_TIME)

    def _estimate_subtask_arrival(self, st: Any) -> float:
        sid = int(getattr(st, "id", -1))
        oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
        if sid in self.anchor_reference.get("subtask_arrival", {}):
            return float(self.anchor_reference["subtask_arrival"][sid])
        if oid in self.anchor_reference.get("order_arrival_avg", {}):
            return float(self.anchor_reference["order_arrival_avg"][oid])
        return 0.0

    def _estimate_subtask_start(self, st: Any) -> float:
        sid = int(getattr(st, "id", -1))
        oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
        if sid in self.anchor_reference.get("subtask_start", {}):
            return float(self.anchor_reference["subtask_start"][sid])
        if oid in self.anchor_reference.get("order_start_avg", {}):
            return float(self.anchor_reference["order_start_avg"][oid])
        return 0.0

    def _estimate_subtask_slack(self, st: Any) -> float:
        sid = int(getattr(st, "id", -1))
        oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
        if sid in self.anchor_reference.get("subtask_slack", {}):
            return float(self.anchor_reference["subtask_slack"][sid])
        if oid in self.anchor_reference.get("order_slack_avg", {}):
            return float(self.anchor_reference["order_slack_avg"][oid])
        return 0.0

    def _estimate_subtask_stack_span(self, st: Any) -> Dict[str, float]:
        points: Dict[int, Tuple[float, float]] = {}
        for sku in getattr(st, "unique_sku_list", []) or []:
            for tote_id in getattr(sku, "storeToteList", []) or []:
                tote = self.problem.id_to_tote.get(tote_id) if self.problem is not None else None
                if tote is None or getattr(tote, "store_point", None) is None:
                    continue
                pt = tote.store_point
                points[int(pt.idx)] = (float(pt.x), float(pt.y))
        coords = list(points.values())
        max_span = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dx = abs(coords[i][0] - coords[j][0])
                dy = abs(coords[i][1] - coords[j][1])
                max_span = max(max_span, dx + dy)
        return {"stack_count": float(len(points)), "span": float(max_span)}

    def _normalize_station_assignments(self):
        station_groups: Dict[int, List[Any]] = defaultdict(list)
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid >= 0:
                station_groups[sid].append(st)
        for sid, rows in station_groups.items():
            rows.sort(key=lambda item: (int(getattr(item, "station_sequence_rank", 0)), int(getattr(item, "id", -1))))
            for rank, st in enumerate(rows):
                st.assigned_station_id = sid
                st.station_sequence_rank = rank

    def _seed_station_assignments_from_anchor(self):
        if self.problem is None:
            return
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(getattr(self.problem, "station_list", []) or [])]
        if not station_ids:
            return
        loads: Dict[int, int] = defaultdict(int)
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid >= 0:
                loads[sid] += 1
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid >= 0:
                continue
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            preferred = int(self.anchor_reference.get("order_major_station", {}).get(oid, -1))
            if preferred < 0 or preferred not in station_ids:
                preferred = min(station_ids, key=lambda idx: (loads[idx], idx))
            st.assigned_station_id = preferred
            st.station_sequence_rank = loads[preferred]
            loads[preferred] += 1
        self._normalize_station_assignments()

    def _build_sp2_layer_context(self) -> SP2LayerContext:
        arrival_time_by_subtask: Dict[int, float] = {}
        processing_time_by_subtask: Dict[int, float] = {}
        order_station_penalty: Dict[Tuple[int, int], float] = {}
        anchor_station_by_subtask: Dict[int, int] = {}
        anchor_rank_by_subtask: Dict[int, int] = {}

        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(getattr(self.problem, "station_list", []) or [])]
        order_station_counts = self.anchor_reference.get("order_station_counts", {})
        order_major_station = self.anchor_reference.get("order_major_station", {})

        for st in getattr(self.problem, "subtask_list", []) or []:
            subtask_id = int(getattr(st, "id", -1))
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            arrival_time_by_subtask[subtask_id] = float(self._estimate_subtask_arrival(st))
            processing_time_by_subtask[subtask_id] = float(
                self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)
            )

            station_counts = dict(order_station_counts.get(order_id, {}) or {})
            total = float(sum(float(v) for v in station_counts.values()))
            major_station = int(order_major_station.get(order_id, -1))
            for station_id in station_ids:
                if total > 1e-9:
                    share = float(station_counts.get(station_id, 0.0)) / total
                    penalty = max(0.0, 1.0 - share)
                elif major_station >= 0:
                    penalty = 0.0 if station_id == major_station else 1.0
                else:
                    penalty = 0.0
                order_station_penalty[(order_id, station_id)] = float(penalty)

        for anchor_st in self._iter_snapshot_subtasks(self.anchor):
            subtask_id = int(getattr(anchor_st, "id", -1))
            anchor_station_by_subtask[subtask_id] = int(getattr(anchor_st, "assigned_station_id", -1))
            anchor_rank_by_subtask[subtask_id] = int(getattr(anchor_st, "station_sequence_rank", -1))

        return SP2LayerContext(
            arrival_time_by_subtask=arrival_time_by_subtask,
            processing_time_by_subtask=processing_time_by_subtask,
            order_station_penalty=order_station_penalty,
            anchor_station_by_subtask=anchor_station_by_subtask,
            anchor_rank_by_subtask=anchor_rank_by_subtask,
            lambda_yx=float(self.layer_lambda_weights.get("yx", float(self.cfg.lambda_init))),
            lambda_yu=float(self.layer_lambda_weights.get("yu", float(self.cfg.lambda_init))),
            lambda_yz=float(self.layer_lambda_weights.get("yz", float(self.cfg.lambda_init))),
            tau_y=float(self._trust_region_tau("Y")),
        )

    def _fast_y_integrated_eval(self, context: SP2LayerContext) -> FastYIntegratedEvalResult:
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(getattr(self.problem, "station_list", []) or [])]
        station_loads: Dict[int, float] = {sid: 0.0 for sid in station_ids}
        station_finish_times: Dict[int, float] = {sid: 0.0 for sid in station_ids}
        subtask_start_times: Dict[int, float] = {}

        if not station_ids:
            return FastYIntegratedEvalResult(
                objective_value=0.0,
                approx_makespan=0.0,
                station_cmax=0.0,
                waiting_penalty=0.0,
                queue_penalty=0.0,
                arrival_misalignment_penalty=0.0,
                load_balance_penalty=0.0,
                station_preference_penalty=0.0,
                prox_station_penalty=0.0,
                prox_rank_penalty=0.0,
                station_loads={},
                station_finish_times={},
                subtask_start_times={},
            )

        station_groups: Dict[int, List[Any]] = defaultdict(list)
        anchor_route_tail = 0.0
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid not in station_loads:
                sid = station_ids[0]
                st.assigned_station_id = sid
            station_groups[sid].append(st)
            task_arrivals = [
                float(getattr(task, "arrival_time_at_station", 0.0))
                for task in getattr(st, "execution_tasks", []) or []
                if float(getattr(task, "arrival_time_at_station", 0.0)) > 0.0
            ]
            if task_arrivals:
                anchor_route_tail = max(anchor_route_tail, max(task_arrivals))
            else:
                anchor_route_tail = max(anchor_route_tail, float(self._estimate_subtask_arrival(st)))

        waiting_penalty = 0.0
        queue_penalty = 0.0
        arrival_misalignment_penalty = 0.0
        station_preference_penalty = 0.0
        prox_station_penalty = 0.0
        prox_rank_penalty = 0.0

        for station_id in station_ids:
            rows = list(station_groups.get(station_id, []))
            rows.sort(
                key=lambda item: (
                    int(getattr(item, "station_sequence_rank", -1))
                    if int(getattr(item, "station_sequence_rank", -1)) >= 0
                    else 10 ** 9,
                    int(getattr(item, "id", -1)),
                )
            )
            current_finish = 0.0
            for rank, st in enumerate(rows):
                st.assigned_station_id = int(station_id)
                st.station_sequence_rank = int(rank)
                subtask_id = int(getattr(st, "id", -1))
                arrival_candidates = [
                    float(getattr(task, "arrival_time_at_station", 0.0))
                    for task in getattr(st, "execution_tasks", []) or []
                    if float(getattr(task, "arrival_time_at_station", 0.0)) > 0.0
                ]
                arrival = max(arrival_candidates) if arrival_candidates else float(self._estimate_subtask_arrival(st))
                proc = float(self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
                start = max(current_finish, arrival)
                finish = start + proc
                waiting = max(0.0, start - arrival)
                anchor_start = float(self.anchor_reference.get("subtask_start", {}).get(subtask_id, self._estimate_subtask_start(st)))
                anchor_slack = float(self.anchor_reference.get("subtask_slack", {}).get(subtask_id, self._estimate_subtask_slack(st)))

                waiting_penalty += waiting
                queue_penalty += max(0.0, waiting - anchor_slack)
                arrival_misalignment_penalty += max(0.0, start - anchor_start)
                station_preference_penalty += float(self.sp2._local_station_preference(st, station_id, context))
                prox_station_penalty += float(self.sp2._local_anchor_station_penalty(st, station_id, context))
                prox_rank_penalty += float(self.sp2._local_anchor_rank_penalty(st, rank, context))

                station_loads[station_id] += proc
                station_finish_times[station_id] = finish
                subtask_start_times[subtask_id] = start
                current_finish = finish

        station_cmax = max(station_finish_times.values()) if station_finish_times else 0.0
        approx_makespan = max(float(station_cmax), float(anchor_route_tail))
        load_balance_penalty = max(station_loads.values()) - min(station_loads.values()) if station_loads else 0.0
        prox_penalty = float(prox_station_penalty) + float(self.sp2.local_prox_rank_weight) * float(prox_rank_penalty)
        timing_penalty = float(waiting_penalty) + float(queue_penalty) + float(arrival_misalignment_penalty)
        wait_weight = float(getattr(self.cfg, "y_fast_wait_weight", 0.35))
        queue_weight = float(getattr(self.cfg, "y_fast_queue_weight", 1.0))
        misalignment_cap = float(getattr(self.cfg, "y_fast_misalignment_cap", 30.0))
        timing_penalty = (
            wait_weight * float(waiting_penalty)
            + queue_weight * float(queue_penalty)
            + min(float(arrival_misalignment_penalty), misalignment_cap)
        )
        objective_value = (
            float(approx_makespan)
            + float(context.lambda_yx) * float(station_preference_penalty)
            + float(context.lambda_yu) * float(timing_penalty)
            + float(context.lambda_yz) * float(load_balance_penalty)
            + float(context.tau_y) * float(prox_penalty)
        )

        return FastYIntegratedEvalResult(
            objective_value=float(objective_value),
            approx_makespan=float(approx_makespan),
            station_cmax=float(station_cmax),
            waiting_penalty=float(waiting_penalty),
            queue_penalty=float(queue_penalty),
            arrival_misalignment_penalty=float(arrival_misalignment_penalty),
            load_balance_penalty=float(load_balance_penalty),
            station_preference_penalty=float(station_preference_penalty),
            prox_station_penalty=float(prox_station_penalty),
            prox_rank_penalty=float(prox_rank_penalty),
            station_loads={int(k): float(v) for k, v in station_loads.items()},
            station_finish_times={int(k): float(v) for k, v in station_finish_times.items()},
            subtask_start_times={int(k): float(v) for k, v in subtask_start_times.items()},
        )

    def _y_precheck_eval(self, context: SP2LayerContext) -> YPrecheckEvalResult:
        station_stats = self._recompute_station_schedule(
            arrival_by_subtask=context.arrival_time_by_subtask,
            proc_by_subtask=context.processing_time_by_subtask,
        )
        baseline_slack = float(self._collect_layer_metrics().get("arrival_slack_mean", 0.0))
        current_slacks: List[float] = []
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            arrival = float(context.arrival_time_by_subtask.get(sid, self._estimate_subtask_arrival(st)))
            start = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            current_slacks.append(max(0.0, start - arrival))
        arrival_slack_mean = float(sum(current_slacks) / len(current_slacks)) if current_slacks else 0.0
        arrival_slack_delta = max(0.0, arrival_slack_mean - baseline_slack)
        baseline_sorting = float(sum((self.last_sp3_sorting_costs or {}).values()))
        current_sorting = 0.0
        for st in getattr(self.problem, "subtask_list", []) or []:
            current_sorting += float(self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
        sorting_cost_delta = max(0.0, current_sorting - baseline_sorting)
        objective = (
            float(station_stats.get("station_makespan", 0.0))
            + float(getattr(self.cfg, "y_precheck_sorting_cost_weight", 1.0)) * sorting_cost_delta
            + float(getattr(self.cfg, "y_precheck_arrival_slack_weight", 2.0)) * arrival_slack_delta
        )
        return YPrecheckEvalResult(
            objective_value=float(objective),
            sorting_cost_delta=float(sorting_cost_delta),
            station_cmax=float(station_stats.get("station_makespan", 0.0)),
            arrival_slack_delta=float(arrival_slack_delta),
        )

    def _fast_x_rollout_eval(self) -> FastXRolloutEvalResult:
        template_station_load: Dict[int, float] = defaultdict(float)
        station_load_drift = 0.0
        arrival_shift_penalty = 0.0
        route_pressure_upper_bound = 0.0
        fallback_route = float(sum(self.anchor_reference.get("stack_route_cost", {}).values()) / max(1, len(self.anchor_reference.get("stack_route_cost", {})) or 1))
        baseline_route_pressure = 0.0
        proposal_subtask_count = float(len(getattr(self.problem, "subtask_list", []) or []))
        anchor_order_counts = dict(self.anchor_reference.get("order_subtask_count", {}) or {})
        current_order_counts: Dict[int, int] = defaultdict(int)
        for st in getattr(self.problem, "subtask_list", []) or []:
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            major_sid = int(self.anchor_reference.get("order_major_station", {}).get(order_id, -1))
            if major_sid >= 0:
                template_station_load[major_sid] += 1.0
            current_order_counts[order_id] += 1
            proc = float(self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
            anchor_arrival = float(self._estimate_subtask_arrival(st))
            anchor_start = float(self._estimate_subtask_start(st))
            anchor_proc = float(self.anchor_reference.get("order_proc_avg", {}).get(order_id, proc))
            order_count_now = float(current_order_counts.get(order_id, 1))
            order_count_anchor = float(anchor_order_counts.get(order_id, max(1.0, order_count_now)))
            shifted_start = max(anchor_arrival, anchor_arrival + max(0.0, proc - anchor_proc) + max(0.0, order_count_now - order_count_anchor))
            arrival_shift_penalty += max(0.0, shifted_start - anchor_start)
            stack_ids: Set[int] = set()
            for sku in getattr(st, "unique_sku_list", []) or []:
                for tote_id in getattr(sku, "storeToteList", []) or []:
                    tote = self.problem.id_to_tote.get(int(tote_id)) if self.problem is not None else None
                    if tote is not None and getattr(tote, "store_point", None) is not None:
                        stack_ids.add(int(getattr(tote.store_point, "idx", -1)))
            route_pressure_upper_bound += sum(float(self.anchor_reference.get("stack_route_cost", {}).get(stack_id, fallback_route)) for stack_id in stack_ids if int(stack_id) >= 0)
            baseline_route_pressure += float(self.anchor_reference.get("order_route_pressure", {}).get(order_id, 0.0))
        for sid, load in template_station_load.items():
            anchor_count = float(self.anchor_reference.get("station_subtask_count", {}).get(sid, 0.0))
            station_load_drift += abs(float(load) - anchor_count)
        arrival_weight = float(getattr(self.cfg, "fast_x_arrival_weight", 4.0))
        station_weight = float(getattr(self.cfg, "fast_x_station_drift_weight", 2.0))
        route_weight = float(getattr(self.cfg, "fast_x_route_pressure_weight", 1.0))
        subtask_delta = abs(float(proposal_subtask_count) - float(len(self._iter_snapshot_subtasks(self.anchor))))
        anchor_station_load_sum = max(1.0, float(sum(self.anchor_reference.get("station_subtask_count", {}).values())))
        anchor_makespan = max(1.0, float(self.anchor_z))
        anchor_route_pressure_sum = max(1.0, float(sum(self.anchor_reference.get("order_route_pressure", {}).values())))
        delta_station_load_drift = max(0.0, float(station_load_drift)) / anchor_station_load_sum
        delta_arrival_shift = max(0.0, float(arrival_shift_penalty)) / anchor_makespan
        delta_route_pressure = max(0.0, float(route_pressure_upper_bound - baseline_route_pressure)) / anchor_route_pressure_sum
        fast_penalty = station_weight * delta_station_load_drift
        fast_penalty += arrival_weight * delta_arrival_shift
        fast_penalty += route_weight * delta_route_pressure
        fast_penalty += float(getattr(self.cfg, "fast_x_subtask_delta_weight", 3.0)) * subtask_delta
        return FastXRolloutEvalResult(
            objective_value=float(fast_penalty),
            delta_station_load_drift=float(delta_station_load_drift),
            delta_arrival_shift=float(delta_arrival_shift),
            delta_route_pressure=float(delta_route_pressure),
            delta_subtask_count=float(subtask_delta),
        )

    def _recompute_station_schedule(
        self,
        arrival_by_subtask: Optional[Dict[int, float]] = None,
        proc_by_subtask: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        arrival_by_subtask = arrival_by_subtask or {}
        proc_by_subtask = proc_by_subtask or {}
        self._normalize_station_assignments()
        station_groups: Dict[int, List[Any]] = defaultdict(list)
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid >= 0:
                station_groups[sid].append(st)
        station_ends: Dict[int, float] = {}
        station_idle_total = 0.0
        loads: List[float] = []
        for sid, rows in station_groups.items():
            rows.sort(key=lambda item: (int(getattr(item, "station_sequence_rank", 0)), int(getattr(item, "id", -1))))
            current = 0.0
            for st in rows:
                sid_i = int(getattr(st, "id", -1))
                arrival = float(arrival_by_subtask.get(sid_i, self._estimate_subtask_arrival(st)))
                proc = float(proc_by_subtask.get(sid_i, self._estimate_subtask_processing_time(st)))
                start = max(current, arrival)
                if start > current:
                    station_idle_total += start - current
                st.estimated_process_start_time = start
                current = start + proc
            station_ends[sid] = current
            loads.append(float(len(rows)))
        station_makespan = max(station_ends.values()) if station_ends else 0.0
        load_mean = (sum(loads) / len(loads)) if loads else 0.0
        load_std = math.sqrt(sum((x - load_mean) ** 2 for x in loads) / len(loads)) if loads else 0.0
        load_max_ratio = (max(loads) / load_mean) if load_mean > 1e-9 and loads else 0.0
        return {
            "station_makespan": float(station_makespan),
            "station_idle_total": float(station_idle_total),
            "station_load_std": float(load_std),
            "station_load_max_ratio": float(load_max_ratio),
        }

    def _compute_x_delta(self) -> Dict[str, float]:
        if self.anchor is None:
            return {"changed_orders": 0.0, "cap_delta": 0.0, "incompat_delta": 0.0}
        curr_caps = dict(getattr(self.sp1, "order_capacity_limits", {}) or {})
        anch_caps = dict(self.anchor.sp1_capacity_limits or {})
        changed_orders = {
            int(k)
            for k in set(curr_caps.keys()) | set(anch_caps.keys())
            if int(curr_caps.get(k, -1)) != int(anch_caps.get(k, -1))
        }
        curr_incompat = {tuple(int(x) for x in row) for row in (getattr(self.sp1, "incompatibility_pairs", set()) or set())}
        anch_incompat = {tuple(int(x) for x in row) for row in (self.anchor.sp1_incompatibility_pairs or [])}
        cap_delta = sum(abs(int(curr_caps.get(k, 0)) - int(anch_caps.get(k, 0))) for k in set(curr_caps.keys()) | set(anch_caps.keys()))
        incompat_delta = len(curr_incompat.symmetric_difference(anch_incompat))
        return {
            "changed_orders": float(len(changed_orders)),
            "cap_delta": float(cap_delta),
            "incompat_delta": float(incompat_delta),
        }

    def _current_task_signature_set(self) -> Set[Tuple]:
        return {
            (
                int(getattr(t, "sub_task_id", -1)),
                int(getattr(t, "target_stack_id", -1)),
                str(getattr(t, "operation_mode", "")),
            )
            for t in self._collect_all_tasks()
        }

    def _current_robot_assignment_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for task in self._collect_all_tasks():
            mapping[int(getattr(task, "task_id", -1))] = int(getattr(task, "robot_id", -1))
        return mapping

    def _compute_augmented_layer_objective(self, layer: str) -> Dict[str, Any]:
        layer = str(layer).upper()
        couplings: Dict[str, float] = {}
        local_obj = 0.0
        prox_penalty = 0.0

        if layer == "X":
            x_score = self._score_x_split_proposal()
            local_obj = float(x_score["local_obj"])
            couplings.update({str(k): float(v) for k, v in (x_score.get("couplings", {}) or {}).items()})
            prox_penalty = float(x_score["prox_penalty"])
            x_fast_eval = self._fast_x_rollout_eval()

        elif layer == "Y":
            context = self._build_sp2_layer_context()
            result = self._fast_y_integrated_eval(context)
            local_obj = float(result.approx_makespan)
            couplings["yx"] = float(result.station_preference_penalty)
            couplings["yu"] = float(result.waiting_penalty + result.queue_penalty + result.arrival_misalignment_penalty)
            couplings["yz"] = float(result.load_balance_penalty)
            prox_penalty = float(result.prox_station_penalty) + float(self.sp2.local_prox_rank_weight) * float(result.prox_rank_penalty)

        elif layer == "Z":
            metrics = self._collect_layer_metrics()
            coverage = self._compute_solution_coverage()
            used_stack_ids = {
                int(getattr(task, "target_stack_id", -1))
                for task in self._collect_all_tasks()
                if int(getattr(task, "target_stack_id", -1)) >= 0
            }
            stack_multi_pen = 0.0
            proc_overflow = 0.0
            for st in getattr(self.problem, "subtask_list", []) or []:
                subtask_stack_ids = {
                    int(getattr(task, "target_stack_id", -1))
                    for task in getattr(st, "execution_tasks", []) or []
                    if int(getattr(task, "target_stack_id", -1)) >= 0
                }
                stack_multi_pen += max(0.0, float(len(subtask_stack_ids)) - 1.0)
                proc_curr = self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)
                proc_overflow += max(0.0, proc_curr - self._estimate_subtask_slack(st))
            route_proxy_default = float(sum(self.anchor_reference.get("stack_route_cost", {}).values()) / max(1, len(self.anchor_reference.get("stack_route_cost", {}))))
            route_proxy = sum(float(self.anchor_reference.get("stack_route_cost", {}).get(sid, route_proxy_default)) for sid in used_stack_ids)
            local_obj = float(sum((self.last_sp3_sorting_costs or {}).values())) + 25.0 * float(metrics.get("noise_ratio", 0.0))
            couplings["zx"] = float(coverage.get("unmet_sku_total", 0)) + float(stack_multi_pen)
            couplings["zy"] = float(proc_overflow)
            couplings["zu"] = float(route_proxy / max(1, len(used_stack_ids) or 1))
            anchor_sig = self.anchor_reference.get("anchor_stack_set", set())
            curr_sig = used_stack_ids
            mode_set = {str(getattr(task, "operation_mode", "")) for task in self._collect_all_tasks()}
            prox_penalty = float(abs(len(self._collect_all_tasks()) - int(self.anchor_reference.get("anchor_task_count", 0))))
            prox_penalty += 0.25 * float(len(curr_sig.symmetric_difference(anchor_sig)))
            prox_penalty += 0.25 * float(len(mode_set.symmetric_difference(self.anchor_reference.get("anchor_mode_set", set()))))

        else:
            tasks = self._collect_all_tasks()
            route_finish = 0.0
            lateness = 0.0
            subtask_latest_arrival: Dict[int, float] = defaultdict(float)
            for task in tasks:
                route_finish = max(
                    route_finish,
                    float(getattr(task, "arrival_time_at_station", 0.0)),
                    float(getattr(task, "arrival_time_at_stack", 0.0)),
                )
                sid = int(getattr(task, "sub_task_id", -1))
                subtask_latest_arrival[sid] = max(subtask_latest_arrival[sid], float(getattr(task, "arrival_time_at_station", 0.0)))
            for st in getattr(self.problem, "subtask_list", []) or []:
                sid = int(getattr(st, "id", -1))
                lateness += max(0.0, subtask_latest_arrival.get(sid, 0.0) - float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st))))
            path_len = float(self._compute_robot_path_length())
            local_obj = float(route_finish)
            couplings["uy"] = float(lateness)
            couplings["uz"] = max(0.0, float(path_len) - float(self.anchor_reference.get("robot_path_length", 0.0)))
            anchor_assign: Dict[int, int] = {}
            anchor_arrivals: Dict[int, float] = {}
            for task in self._iter_snapshot_tasks(self.anchor):
                anchor_assign[int(getattr(task, "task_id", -1))] = int(getattr(task, "robot_id", -1))
                anchor_arrivals[int(getattr(task, "task_id", -1))] = float(getattr(task, "arrival_time_at_station", 0.0))
            curr_assign = self._current_robot_assignment_map()
            arrival_delta = 0.0
            reassign_count = 0.0
            for tid, rid in curr_assign.items():
                if int(anchor_assign.get(tid, rid)) != int(rid):
                    reassign_count += 1.0
                arrival_delta += abs(
                    float(anchor_arrivals.get(tid, 0.0))
                    - float(
                        next(
                            (
                                float(getattr(task, "arrival_time_at_station", 0.0))
                                for task in tasks
                                if int(getattr(task, "task_id", -1)) == tid
                            ),
                            0.0,
                        )
                    )
                )
            prox_penalty = float(reassign_count + 0.01 * arrival_delta)

        coupling_penalty = 0.0
        for key, value in couplings.items():
            coupling_penalty += float(self.layer_lambda_weights.get(key, float(self.cfg.lambda_init))) * float(value)
        augmented_obj = float(local_obj + coupling_penalty + self._trust_region_tau(layer) * prox_penalty)
        return {
            "layer": layer,
            "local_obj": float(local_obj),
            "coupling_penalty": float(coupling_penalty),
            "prox_penalty": float(prox_penalty),
            "augmented_obj": float(augmented_obj),
            "couplings": {key: float(val) for key, val in couplings.items()},
            **(
                {
                    "x_destroy_operator": "",
                    "x_repair_operator": "",
                    "x_destroy_size": float("nan"),
                    "x_subtask_count_before": float(len(self._iter_snapshot_subtasks(self.anchor)) if self.anchor is not None else 0),
                    "x_subtask_count_after": float(x_score.get("x_subtask_count", float("nan"))),
                    "x_affinity_penalty": float(x_score.get("x_affinity_penalty", float("nan"))),
                    "x_route_conflict_penalty": float(x_score.get("x_route_conflict_penalty", float("nan"))),
                    "x_finish_time_dispersion_penalty": float(x_score.get("x_finish_time_dispersion_penalty", float("nan"))),
                    "x_fast_penalty": float(x_fast_eval.objective_value),
                    "x_fast_delta_station_load_drift": float(x_fast_eval.delta_station_load_drift),
                    "x_fast_delta_arrival_shift": float(x_fast_eval.delta_arrival_shift),
                    "x_fast_delta_route_pressure": float(x_fast_eval.delta_route_pressure),
                    "x_fast_delta_subtask_count": float(x_fast_eval.delta_subtask_count),
                }
                if layer == "X"
                else {}
            ),
            **(
                {
                    "y_fast_objective": float(result.objective_value),
                    "y_fast_approx_makespan": float(result.approx_makespan),
                    "y_fast_station_cmax": float(result.station_cmax),
                    "y_fast_waiting_penalty": float(result.waiting_penalty),
                    "y_fast_queue_penalty": float(result.queue_penalty),
                    "y_fast_arrival_misalignment_penalty": float(result.arrival_misalignment_penalty),
                    "y_fast_load_balance_penalty": float(result.load_balance_penalty),
                    "y_fast_station_finish_max": float(max(result.station_finish_times.values()) if result.station_finish_times else 0.0),
                    "y_precheck_score": float("nan"),
                    "y_precheck_sorting_cost_delta": float("nan"),
                    "y_precheck_station_cmax": float("nan"),
                    "y_precheck_arrival_slack_delta": float("nan"),
                }
                if layer == "Y"
                else {}
            ),
        }

    def _run_sp4_augmented(self, mu_override: Optional[float] = None):
        soft_windows: Dict[int, float] = {}
        for st in getattr(self.problem, "subtask_list", []) or []:
            latest = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            for task in getattr(st, "execution_tasks", []) or []:
                soft_windows[int(getattr(task, "task_id", -1))] = latest
        mu = float(mu_override if mu_override is not None else max(0.0, self.cfg.mu_value))
        arrival_times, robot_assign = self.sp4.solve(
            self.problem.subtask_list,
            use_mip=self.cfg.sp4_use_mip,
            lkh_time_limit_seconds=self.cfg.sp4_lkh_time_limit_seconds if not self.cfg.sp4_use_mip else None,
            soft_time_windows=soft_windows if soft_windows else None,
            mu=mu,
        )
        self.last_sp4_arrival_times = {int(k): float(v) for k, v in (arrival_times or {}).items()}
        st_map = {st.id: st for st in self.problem.subtask_list}
        for st_id, robot_id in (robot_assign or {}).items():
            if st_id in st_map:
                st_map[st_id].assigned_robot_id = int(robot_id)

    def _u_route_state_to_plan(self, state: Dict[int, Dict[str, int]]) -> Dict[int, List[List[int]]]:
        grouped: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        for task_id, row in (state or {}).items():
            rid = int(row.get("robot_id", -1))
            trip_id = int(row.get("trip_id", 0))
            seq = int(row.get("robot_visit_sequence", 0))
            grouped[rid][trip_id].append((seq, int(task_id)))
        plan: Dict[int, List[List[int]]] = {}
        for rid, trip_map in grouped.items():
            plan[int(rid)] = []
            for trip_id in sorted(trip_map.keys()):
                trip = [tid for _, tid in sorted(trip_map[trip_id], key=lambda item: (item[0], item[1]))]
                if trip:
                    plan[int(rid)].append(trip)
        return plan

    def _u_route_plan_to_state(self, plan: Dict[int, List[List[int]]]) -> Dict[int, Dict[str, int]]:
        state: Dict[int, Dict[str, int]] = {}
        for rid, trips in (plan or {}).items():
            for trip_id, trip in enumerate(trips):
                for seq, task_id in enumerate(trip):
                    state[int(task_id)] = {
                        "robot_id": int(rid),
                        "trip_id": int(trip_id),
                        "robot_visit_sequence": int(seq),
                    }
        return state

    def _extract_u_route_state(self) -> Dict[int, Dict[str, int]]:
        robot_ids = [int(getattr(r, "id", idx)) for idx, r in enumerate(getattr(self.problem, "robot_list", []) or [])]
        fallback_robot = robot_ids[0] if robot_ids else -1
        tasks = sorted(
            self._collect_all_tasks(),
            key=lambda t: (
                int(getattr(t, "robot_id", fallback_robot)),
                int(getattr(t, "trip_id", 0)),
                int(getattr(t, "robot_visit_sequence", -1)) if int(getattr(t, "robot_visit_sequence", -1)) >= 0 else 10 ** 6,
                int(getattr(t, "task_id", -1)),
            ),
        )
        state: Dict[int, Dict[str, int]] = {}
        seq_counter: Dict[Tuple[int, int], int] = defaultdict(int)
        for task in tasks:
            tid = int(getattr(task, "task_id", -1))
            rid = int(getattr(task, "robot_id", fallback_robot))
            trip_id = int(getattr(task, "trip_id", 0))
            seq = int(getattr(task, "robot_visit_sequence", -1))
            if seq < 0:
                seq = seq_counter[(rid, trip_id)]
            seq_counter[(rid, trip_id)] = max(seq_counter[(rid, trip_id)], seq + 1)
            state[tid] = {"robot_id": rid, "trip_id": trip_id, "robot_visit_sequence": seq}
        return self._normalize_u_route_state(state)

    def _normalize_u_route_state(self, state: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        return self._u_route_plan_to_state(self._u_route_state_to_plan(state))

    def _apply_u_route_state(self, state: Dict[int, Dict[str, int]]):
        task_map = {int(getattr(t, "task_id", -1)): t for t in self._collect_all_tasks()}
        for task_id, row in (state or {}).items():
            task = task_map.get(int(task_id))
            if task is None:
                continue
            task.robot_id = int(row.get("robot_id", -1))
            task.trip_id = int(row.get("trip_id", 0))
            task.robot_visit_sequence = int(row.get("robot_visit_sequence", 0))

    def _replay_u_routes(self) -> bool:
        if self.problem is None:
            return False
        task_map = {int(getattr(t, "task_id", -1)): t for t in self._collect_all_tasks()}
        robots = getattr(self.problem, "robot_list", []) or []
        robot_map = {int(getattr(r, "id", idx)): r for idx, r in enumerate(robots)}
        speed = max(1e-9, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0)))
        plan = self._u_route_state_to_plan(self._extract_u_route_state())

        for task in task_map.values():
            task.arrival_time_at_stack = 0.0
            task.arrival_time_at_station = 0.0
            task.detailed_path = []

        for rid, trips in plan.items():
            robot = robot_map.get(int(rid))
            if robot is None or getattr(robot, "start_point", None) is None:
                return False
            depot = robot.start_point
            curr_time = 0.0
            curr_pos = (float(depot.x), float(depot.y))
            for trip in trips:
                if curr_pos != (float(depot.x), float(depot.y)):
                    curr_time += (abs(curr_pos[0] - float(depot.x)) + abs(curr_pos[1] - float(depot.y))) / speed
                    curr_pos = (float(depot.x), float(depot.y))
                for seq, task_id in enumerate(trip):
                    task = task_map.get(int(task_id))
                    if task is None:
                        return False
                    stack = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1)))
                    sid = int(getattr(task, "target_station_id", -1))
                    if stack is None or getattr(stack, "store_point", None) is None:
                        return False
                    if sid < 0 or sid >= len(getattr(self.problem, "station_list", []) or []):
                        return False
                    station = self.problem.station_list[sid]
                    stack_pos = (float(stack.store_point.x), float(stack.store_point.y))
                    station_pos = (float(station.point.x), float(station.point.y))
                    curr_time += (abs(curr_pos[0] - stack_pos[0]) + abs(curr_pos[1] - stack_pos[1])) / speed
                    task.arrival_time_at_stack = float(curr_time)
                    curr_time += float(getattr(task, "robot_service_time", 0.0))
                    curr_time += (abs(stack_pos[0] - station_pos[0]) + abs(stack_pos[1] - station_pos[1])) / speed
                    task.arrival_time_at_station = float(curr_time)
                    task.robot_visit_sequence = int(seq)
                    task.detailed_path = [
                        (float(curr_pos[0]), float(curr_pos[1]), max(0.0, float(task.arrival_time_at_stack) - float(getattr(task, "robot_service_time", 0.0)))),
                        (float(stack_pos[0]), float(stack_pos[1]), float(task.arrival_time_at_stack)),
                        (float(station_pos[0]), float(station_pos[1]), float(task.arrival_time_at_station)),
                    ]
                    curr_pos = station_pos
            if curr_pos != (float(depot.x), float(depot.y)):
                curr_time += (abs(curr_pos[0] - float(depot.x)) + abs(curr_pos[1] - float(depot.y))) / speed

        for st in getattr(self.problem, "subtask_list", []) or []:
            robot_ids = [int(getattr(t, "robot_id", -1)) for t in (getattr(st, "execution_tasks", []) or []) if int(getattr(t, "robot_id", -1)) >= 0]
            if not robot_ids:
                st.assigned_robot_id = -1
            else:
                counts = defaultdict(int)
                for rid in robot_ids:
                    counts[int(rid)] += 1
                st.assigned_robot_id = int(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
        return True

    def _replay_y_candidate_routes_incrementally(self, changed_subtask_ids: List[int], affected_robot_trips: Set[Tuple[int, int]]) -> bool:
        self._sync_task_assignments_from_subtasks()
        if not changed_subtask_ids:
            return True
        # 当前先复用现有 route replay，保持 task/robot/trip 不变，只按新的 station/rank 回放到站时间。
        # 虽然实现上是全量回放，但语义上属于固定 route skeleton 的增量模式。
        return bool(self._replay_u_routes())

    def _run_partial_sp4_for_y_candidate(self) -> bool:
        try:
            self._sync_task_assignments_from_subtasks()
            self._run_sp4_augmented()
            return True
        except Exception:
            return False

    def _evaluate_y_candidate_with_route_sim(self, y_signature: str) -> YRouteSimEvalResult:
        cache_size = max(1, int(getattr(self.cfg, "y_route_sim_cache_size", 16)))
        cache = getattr(self, "y_route_sim_cache", {})
        if y_signature and y_signature in cache:
            return copy.deepcopy(cache[y_signature])

        self._sync_task_assignments_from_subtasks()
        changed_subtasks = self._collect_changed_y_subtasks()
        affected_trips = self._collect_affected_y_robot_trips(changed_subtasks)
        incremental_cap = max(1, int(getattr(self.cfg, "y_incremental_route_subtask_cap", 4)))
        trip_cap = max(1, int(getattr(self.cfg, "y_incremental_route_trip_cap", 3)))
        use_incremental = bool(
            len(changed_subtasks) <= incremental_cap
            and len(affected_trips) <= trip_cap
            and str(getattr(self.cfg, "y_route_eval_mode", "replay_then_polish")).lower() == "replay_then_polish"
        )
        replayed_sp4 = False
        used_incremental_route = False
        ok = False
        if use_incremental:
            ok = self._replay_y_candidate_routes_incrementally(changed_subtasks, affected_trips)
            used_incremental_route = bool(ok)
        if not ok:
            ok = self._run_partial_sp4_for_y_candidate()
            replayed_sp4 = bool(ok)
        if not ok:
            self._run_sp4_augmented()
            replayed_sp4 = True

        proxy = float(self.sim.calculate_with_existing_arrivals())
        metrics = self._collect_layer_metrics()
        station_makespan = float(proxy)
        route_arrival_score = float(metrics.get("arrival_slack_mean", 0.0)) + 2.0 * float(self._compute_late_task_count())
        result = YRouteSimEvalResult(
            objective_value=float(
                proxy
                + float(getattr(self.cfg, "y_route_arrival_weight", 1.0)) * float(metrics.get("arrival_slack_mean", 0.0))
                + float(getattr(self.cfg, "y_route_late_task_weight", 8.0)) * float(self._compute_late_task_count())
                + float(getattr(self.cfg, "y_route_load_balance_weight", 0.5)) * float(metrics.get("station_load_std", 0.0))
            ),
            route_arrival_score=float(route_arrival_score),
            station_makespan=float(station_makespan),
            global_makespan_proxy=float(proxy),
            arrival_slack_mean=float(metrics.get("arrival_slack_mean", 0.0)),
            late_task_count=float(self._compute_late_task_count()),
            station_wait_total=float(sum(float(getattr(t, "tote_wait_time", 0.0)) for t in self._collect_all_tasks())),
            station_load_std=float(metrics.get("station_load_std", 0.0)),
            replayed_sp4=bool(replayed_sp4),
            used_incremental_route=bool(used_incremental_route),
        )
        if y_signature:
            cache[str(y_signature)] = copy.deepcopy(result)
            while len(cache) > cache_size:
                first_key = next(iter(cache.keys()))
                del cache[first_key]
            self.y_route_sim_cache = cache
        return result

    def _propose_u_route_neighbor(self, rng: random.Random, priority_robot_ids: Optional[List[int]] = None, forced_move_type: Optional[str] = None) -> Dict[str, Any]:
        plan = self._u_route_state_to_plan(self._extract_u_route_state())
        robot_ids = sorted(plan.keys())
        if priority_robot_ids:
            preferred = [rid for rid in robot_ids if int(rid) in {int(x) for x in priority_robot_ids}]
            if preferred:
                robot_ids = preferred
        if not robot_ids:
            return {"feasible": False, "move_type": "none", "state": {}, "changed_task_count": 0, "trip_count_delta": 0}
        move_type = str(forced_move_type) if forced_move_type else rng.choice(["same_robot_swap", "cross_robot_relocate", "cross_robot_swap", "trip_split_merge"])
        changed_tasks: Set[int] = set()
        before_trip_count = sum(len(trips) for trips in plan.values())

        if move_type == "same_robot_swap":
            candidates = [(rid, idx) for rid, trips in plan.items() for idx, trip in enumerate(trips) if len(trip) >= 2]
            if not candidates:
                return {"feasible": False, "move_type": move_type, "state": {}, "changed_task_count": 0, "trip_count_delta": 0}
            rid, trip_idx = rng.choice(candidates)
            trip = plan[rid][trip_idx]
            i, j = sorted(rng.sample(range(len(trip)), 2))
            trip[i], trip[j] = trip[j], trip[i]
            changed_tasks.update([int(trip[i]), int(trip[j])])
        elif move_type == "cross_robot_relocate":
            if len(robot_ids) < 2:
                return {"feasible": False, "move_type": move_type, "state": {}, "changed_task_count": 0, "trip_count_delta": 0}
            src, dst = rng.sample(robot_ids, 2)
            src_choices = [idx for idx, trip in enumerate(plan.get(src, [])) if trip]
            if not src_choices:
                return {"feasible": False, "move_type": move_type, "state": {}, "changed_task_count": 0, "trip_count_delta": 0}
            src_trip_idx = rng.choice(src_choices)
            src_trip = plan[src][src_trip_idx]
            pos = rng.randrange(len(src_trip))
            task_id = int(src_trip.pop(pos))
            changed_tasks.add(task_id)
            if not plan.get(dst):
                plan[dst] = [[task_id]]
            else:
                dst_trip_idx = rng.randrange(len(plan[dst]))
                plan[dst][dst_trip_idx].append(task_id)
            if not src_trip:
                del plan[src][src_trip_idx]
        elif move_type == "cross_robot_swap":
            if len(robot_ids) < 2:
                return {"feasible": False, "move_type": move_type, "state": {}, "changed_task_count": 0, "trip_count_delta": 0}
            r1, r2 = rng.sample(robot_ids, 2)
            choices1 = [(idx, trip) for idx, trip in enumerate(plan.get(r1, [])) if trip]
            choices2 = [(idx, trip) for idx, trip in enumerate(plan.get(r2, [])) if trip]
            if not choices1 or not choices2:
                return {"feasible": False, "move_type": move_type, "state": {}, "changed_task_count": 0, "trip_count_delta": 0}
            idx1, trip1 = rng.choice(choices1)
            idx2, trip2 = rng.choice(choices2)
            p1 = rng.randrange(len(trip1))
            p2 = rng.randrange(len(trip2))
            trip1[p1], trip2[p2] = trip2[p2], trip1[p1]
            changed_tasks.update([int(trip1[p1]), int(trip2[p2])])
        else:
            split_candidates = [(rid, idx) for rid, trips in plan.items() for idx, trip in enumerate(trips) if len(trip) >= 2]
            merge_candidates = [(rid, idx) for rid, trips in plan.items() for idx in range(len(trips) - 1)]
            if split_candidates and (not merge_candidates or rng.random() < 0.5):
                rid, trip_idx = rng.choice(split_candidates)
                trip = plan[rid][trip_idx]
                cut = rng.randint(1, len(trip) - 1)
                new_trip = trip[cut:]
                plan[rid][trip_idx] = trip[:cut]
                plan[rid].insert(trip_idx + 1, new_trip)
                changed_tasks.update(int(tid) for tid in trip)
            elif merge_candidates:
                rid, trip_idx = rng.choice(merge_candidates)
                merged = list(plan[rid][trip_idx]) + list(plan[rid][trip_idx + 1])
                plan[rid][trip_idx] = merged
                del plan[rid][trip_idx + 1]
                changed_tasks.update(int(tid) for tid in merged)
            else:
                return {"feasible": False, "move_type": move_type, "state": {}, "changed_task_count": 0, "trip_count_delta": 0}

        new_state = self._normalize_u_route_state(self._u_route_plan_to_state(plan))
        after_trip_count = sum(len(trips) for trips in self._u_route_state_to_plan(new_state).values())
        return {
            "feasible": True,
            "move_type": move_type,
            "state": new_state,
            "changed_task_count": int(len(changed_tasks)),
            "trip_count_delta": int(after_trip_count - before_trip_count),
        }

    def _estimate_task_sorting_cost(self, task: Any) -> float:
        noise = float(len(getattr(task, "noise_tote_ids", []) or []))
        span = 0.0
        if getattr(task, "sort_layer_range", None) is not None:
            lo, hi = getattr(task, "sort_layer_range", (0, 0))
            span = float(max(0, hi - lo + 1))
        base = float(getattr(task, "station_service_time", 0.0))
        if str(getattr(task, "operation_mode", "")).upper() == "SORT":
            base += 0.25 * span + 0.5 * noise
        return float(base)

    def _refresh_subtask_execution_details(self, st: Any):
        st.assigned_tote_ids = []
        st.involved_stacks = []
        st.visit_points = []
        seen_totes: Set[int] = set()
        seen_stacks: Set[int] = set()
        for task in getattr(st, "execution_tasks", []) or []:
            task.sub_task_id = int(getattr(st, "id", -1))
            task.target_station_id = int(getattr(st, "assigned_station_id", -1))
            task.station_sequence_rank = int(getattr(st, "station_sequence_rank", -1))
            task.target_tote_ids = list(dict.fromkeys(int(x) for x in (getattr(task, "target_tote_ids", []) or [])))
            hit_set = set(int(x) for x in (getattr(task, "hit_tote_ids", []) or []))
            task.hit_tote_ids = [int(x) for x in task.target_tote_ids if int(x) in hit_set]
            task.noise_tote_ids = [int(x) for x in task.target_tote_ids if int(x) not in hit_set]
            for tote_id in task.target_tote_ids:
                if int(tote_id) not in seen_totes:
                    seen_totes.add(int(tote_id))
                    st.assigned_tote_ids.append(int(tote_id))
            stack_id = int(getattr(task, "target_stack_id", -1))
            stack_obj = self.problem.point_to_stack.get(stack_id) if self.problem is not None else None
            if stack_obj is not None and stack_id not in seen_stacks:
                seen_stacks.add(stack_id)
                st.involved_stacks.append(stack_obj)
                if getattr(stack_obj, "store_point", None) is not None:
                    st.visit_points.append(stack_obj.store_point)

    def _sync_sp3_caches_from_problem(self):
        tote_selection: Dict[int, List[int]] = {}
        sorting_costs: Dict[int, float] = {}
        self._rebuild_problem_task_list()
        for st in getattr(self.problem, "subtask_list", []) or []:
            self._refresh_subtask_execution_details(st)
            sid = int(getattr(st, "id", -1))
            tote_selection[sid] = [int(x) for x in (getattr(st, "assigned_tote_ids", []) or [])]
            sorting_costs[sid] = float(sum(self._estimate_task_sorting_cost(task) for task in (getattr(st, "execution_tasks", []) or [])))
        self.last_sp3_tote_selection = tote_selection
        self.last_sp3_sorting_costs = sorting_costs

    def _next_task_id(self) -> int:
        tasks = self._collect_all_tasks()
        return (max(int(getattr(t, "task_id", -1)) for t in tasks) + 1) if tasks else 0

    def _compute_subtask_target_coverage(self, st: Any) -> Dict[int, int]:
        req: Dict[int, int] = defaultdict(int)
        prov: Dict[int, int] = defaultdict(int)
        tote_map = getattr(self.problem, "id_to_tote", {}) if self.problem is not None else {}
        for sku in getattr(st, "sku_list", []) or []:
            sid = int(getattr(sku, "id", -1))
            if sid >= 0:
                req[sid] += 1
        for task in getattr(st, "execution_tasks", []) or []:
            for tote_id in getattr(task, "target_tote_ids", []) or []:
                tote = tote_map.get(int(tote_id))
                if tote is None:
                    continue
                for sid, qty in getattr(tote, "sku_quantity_map", {}).items():
                    sid_i = int(sid)
                    if sid_i in req:
                        prov[sid_i] += int(qty)
        return {sid: max(0, int(req[sid] - prov.get(sid, 0))) for sid in req}

    def _validate_z_subtask_candidate(self, st: Any) -> Tuple[bool, Dict[str, Any]]:
        if self.problem is None:
            return False, {"reason": "no_problem"}
        tote_map = getattr(self.problem, "id_to_tote", {}) or {}
        tasks = list(getattr(st, "execution_tasks", []) or [])
        if not tasks:
            return False, {"reason": "no_tasks"}

        seen_target_totes: Set[int] = set()
        seen_hit_totes: Set[int] = set()
        used_stack_ids: Set[int] = set()
        mode_changes = 0
        for task in tasks:
            stack_id = int(getattr(task, "target_stack_id", -1))
            stack = self.problem.point_to_stack.get(stack_id)
            if stack is None or getattr(stack, "store_point", None) is None:
                return False, {"reason": "missing_stack", "task_id": int(getattr(task, "task_id", -1))}
            stack_tote_ids = [int(t.id) for t in (getattr(stack, "totes", []) or [])]
            target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or [])]
            hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or [])]
            if not target_ids:
                return False, {"reason": "empty_target_totes", "task_id": int(getattr(task, "task_id", -1))}
            if len(target_ids) != len(set(target_ids)):
                return False, {"reason": "duplicate_target_totes", "task_id": int(getattr(task, "task_id", -1))}
            if any(tid not in stack_tote_ids for tid in target_ids):
                return False, {"reason": "target_tote_not_in_stack", "task_id": int(getattr(task, "task_id", -1))}
            if any(tid not in set(target_ids) for tid in hit_ids):
                return False, {"reason": "hit_not_subset_target", "task_id": int(getattr(task, "task_id", -1))}
            if any(tid in seen_target_totes for tid in target_ids):
                return False, {"reason": "reused_target_tote", "task_id": int(getattr(task, "task_id", -1))}
            if any(tid in seen_hit_totes for tid in hit_ids):
                return False, {"reason": "reused_hit_tote", "task_id": int(getattr(task, "task_id", -1))}
            seen_target_totes.update(target_ids)
            seen_hit_totes.update(hit_ids)
            used_stack_ids.add(stack_id)

            op_mode = str(getattr(task, "operation_mode", "")).upper()
            if op_mode == "FLIP":
                if getattr(task, "sort_layer_range", None) is not None:
                    return False, {"reason": "flip_has_sort_range", "task_id": int(getattr(task, "task_id", -1))}
                if set(target_ids) != set(hit_ids):
                    return False, {"reason": "flip_targets_not_equal_hits", "task_id": int(getattr(task, "task_id", -1))}
            elif op_mode == "SORT":
                layer_range = getattr(task, "sort_layer_range", None)
                if layer_range is None:
                    return False, {"reason": "sort_missing_range", "task_id": int(getattr(task, "task_id", -1))}
                lo, hi = int(layer_range[0]), int(layer_range[1])
                if lo < 0 or hi < lo or hi >= len(stack_tote_ids):
                    return False, {"reason": "sort_range_invalid", "task_id": int(getattr(task, "task_id", -1))}
                expected_ids = [int(t.id) for t in (getattr(stack, "totes", []) or [])[lo:hi + 1]]
                if target_ids != expected_ids:
                    return False, {"reason": "sort_targets_not_contiguous_range", "task_id": int(getattr(task, "task_id", -1))}
                mode_changes += 1
            else:
                return False, {"reason": "unknown_mode", "task_id": int(getattr(task, "task_id", -1))}

            for tote_id in target_ids:
                tote = tote_map.get(int(tote_id))
                if tote is None:
                    return False, {"reason": "missing_tote", "task_id": int(getattr(task, "task_id", -1))}
                if getattr(tote, "store_point", None) is None or int(getattr(tote.store_point, "idx", -1)) != stack_id:
                    return False, {"reason": "tote_stack_mismatch", "task_id": int(getattr(task, "task_id", -1))}

        unmet = self._compute_subtask_target_coverage(st)
        unmet_total = int(sum(unmet.values()))
        if unmet_total > 0:
            return False, {"reason": "subtask_unmet_coverage", "unmet_total": unmet_total, "unmet": unmet}
        return True, {
            "used_stack_count": int(len(used_stack_ids)),
            "task_count": int(len(tasks)),
            "sort_task_count": int(mode_changes),
        }

    def _build_z_candidate_from_subtasks(self, rng: random.Random, priority_subtask_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "execution_tasks", None)]
        if priority_subtask_ids:
            preferred_ids = {int(x) for x in priority_subtask_ids}
            preferred = [st for st in subtasks if int(getattr(st, "id", -1)) in preferred_ids]
            if preferred:
                subtasks = preferred
        if not subtasks:
            return {"feasible": False, "move_type": "none", "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        st = rng.choice(subtasks)
        tasks = list(getattr(st, "execution_tasks", []) or [])
        if not tasks:
            return {"feasible": False, "move_type": "none", "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        before_task_count = len(tasks)
        before_stacks = {int(getattr(t, "target_stack_id", -1)) for t in tasks}
        before_modes = {str(getattr(t, "operation_mode", "")) for t in tasks}
        move_type = rng.choice(["stack_replace", "tote_replace_within_stack", "mode_flip_sort_toggle", "range_shrink_expand", "task_merge_split"])

        if move_type == "stack_replace":
            stack_groups: Dict[int, List[Any]] = defaultdict(list)
            for task in tasks:
                stack_groups[int(getattr(task, "target_stack_id", -1))].append(task)
            if len(stack_groups) < 2:
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            a, b = rng.sample(list(stack_groups.keys()), 2)
            task_a = rng.choice(stack_groups[a])
            task_b = rng.choice(stack_groups[b])
            for name in ["target_stack_id", "target_tote_ids", "hit_tote_ids", "noise_tote_ids", "sort_layer_range", "robot_service_time", "station_service_time", "operation_mode"]:
                tmp = copy.deepcopy(getattr(task_a, name))
                setattr(task_a, name, copy.deepcopy(getattr(task_b, name)))
                setattr(task_b, name, tmp)
        elif move_type == "tote_replace_within_stack":
            task = rng.choice(tasks)
            stack = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1))) if self.problem is not None else None
            if stack is None or not getattr(stack, "totes", None):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            current = [int(x) for x in (getattr(task, "target_tote_ids", []) or [])]
            if not current:
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            hit_set = set(int(x) for x in (getattr(task, "hit_tote_ids", []) or []))
            pool = [int(t.id) for t in getattr(stack, "totes", []) if int(t.id) not in hit_set]
            if not pool:
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            replace_idx = next((idx for idx, tote_id in enumerate(current) if int(tote_id) not in hit_set), 0)
            current[replace_idx] = int(rng.choice(pool))
            task.target_tote_ids = list(dict.fromkeys(current))
        elif move_type == "mode_flip_sort_toggle":
            task = rng.choice(tasks)
            stack = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1))) if self.problem is not None else None
            hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or [])]
            if stack is None:
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            if str(getattr(task, "operation_mode", "")).upper() == "SORT":
                task.operation_mode = "FLIP"
                task.target_tote_ids = list(hit_ids)
                task.sort_layer_range = None
                task.station_service_time = 0.0
            else:
                layers = [int(stack.get_tote_layer(int(tid))) for tid in hit_ids]
                layers = [x for x in layers if x >= 0]
                if not layers:
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
                lo = max(0, min(layers) - 1)
                hi = min(len(getattr(stack, "totes", []) or []) - 1, max(layers) + 1)
                task.operation_mode = "SORT"
                task.sort_layer_range = (int(lo), int(hi))
                task.target_tote_ids = [int(t.id) for t in getattr(stack, "totes", [])[lo:hi + 1]]
                task.station_service_time = float(len([x for x in task.target_tote_ids if int(x) not in set(hit_ids)])) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0))
        elif move_type == "range_shrink_expand":
            task = rng.choice(tasks)
            stack = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1))) if self.problem is not None else None
            if stack is None or not getattr(stack, "totes", None):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            layers = [int(stack.get_tote_layer(int(tid))) for tid in (getattr(task, "target_tote_ids", []) or [])]
            layers = [x for x in layers if x >= 0]
            if not layers:
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            lo, hi = min(layers), max(layers)
            if rng.random() < 0.5 and hi - lo >= 1:
                lo = lo + 1 if rng.random() < 0.5 else lo
                hi = hi - 1 if rng.random() >= 0.5 else hi
            else:
                lo = max(0, lo - (1 if rng.random() < 0.5 else 0))
                hi = min(len(getattr(stack, "totes", []) or []) - 1, hi + (1 if rng.random() >= 0.5 else 0))
            new_target = [int(t.id) for t in getattr(stack, "totes", [])[lo:hi + 1]]
            hit_set = set(int(x) for x in (getattr(task, "hit_tote_ids", []) or []))
            if not hit_set.issubset(set(new_target)):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            task.operation_mode = "SORT"
            task.sort_layer_range = (int(lo), int(hi))
            task.target_tote_ids = new_target
            task.station_service_time = float(len([x for x in new_target if int(x) not in hit_set])) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0))
        else:
            merge_candidates = [(i, j) for i in range(len(tasks)) for j in range(i + 1, len(tasks)) if int(getattr(tasks[i], "target_stack_id", -1)) == int(getattr(tasks[j], "target_stack_id", -1))]
            if merge_candidates and rng.random() < 0.5:
                i, j = rng.choice(merge_candidates)
                left = tasks[i]
                right = tasks[j]
                left.target_tote_ids = list(dict.fromkeys(list(getattr(left, "target_tote_ids", []) or []) + list(getattr(right, "target_tote_ids", []) or [])))
                left.hit_tote_ids = list(dict.fromkeys(list(getattr(left, "hit_tote_ids", []) or []) + list(getattr(right, "hit_tote_ids", []) or [])))
                left.robot_service_time = float(getattr(left, "robot_service_time", 0.0)) + float(getattr(right, "robot_service_time", 0.0))
                left.station_service_time = float(getattr(left, "station_service_time", 0.0)) + float(getattr(right, "station_service_time", 0.0))
                st.execution_tasks.remove(right)
            else:
                task = rng.choice(tasks)
                target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or [])]
                if len(target_ids) < 2:
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
                cut = len(target_ids) // 2
                new_task = copy.deepcopy(task)
                new_task.task_id = int(self._next_task_id())
                task.target_tote_ids = target_ids[:cut]
                new_task.target_tote_ids = target_ids[cut:]
                st.execution_tasks.append(new_task)

        for task in getattr(st, "execution_tasks", []) or []:
            hit_set = set(int(x) for x in (getattr(task, "hit_tote_ids", []) or []))
            task.target_tote_ids = list(dict.fromkeys(int(x) for x in (getattr(task, "target_tote_ids", []) or [])))
            task.hit_tote_ids = [int(x) for x in task.target_tote_ids if int(x) in hit_set]
            task.noise_tote_ids = [int(x) for x in task.target_tote_ids if int(x) not in hit_set]
            task.station_service_time = max(float(getattr(task, "station_service_time", 0.0)), float(len(task.noise_tote_ids)) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)))
            task.robot_service_time = max(float(getattr(task, "robot_service_time", 0.0)), float(len(task.target_tote_ids)) * 0.5)
        valid_subtask, subtask_info = self._validate_z_subtask_candidate(st)
        if not valid_subtask:
            return {
                "feasible": False,
                "move_type": move_type,
                "changed_subtask_count": 0,
                "task_delta": 0,
                "stack_delta": 0,
                "mode_delta": 0,
                "reject_reason": str(subtask_info.get("reason", "invalid_subtask")),
            }
        self._sync_sp3_caches_from_problem()
        coverage = self._compute_solution_coverage()
        if not bool(coverage.get("coverage_ok", False)):
            return {
                "feasible": False,
                "move_type": move_type,
                "changed_subtask_count": 0,
                "task_delta": 0,
                "stack_delta": 0,
                "mode_delta": 0,
                "reject_reason": "global_coverage_fail",
            }
        after_tasks = list(getattr(st, "execution_tasks", []) or [])
        after_stacks = {int(getattr(t, "target_stack_id", -1)) for t in after_tasks}
        after_modes = {str(getattr(t, "operation_mode", "")) for t in after_tasks}
        return {
            "feasible": True,
            "move_type": move_type,
            "changed_subtask_count": 1,
            "task_delta": int(len(after_tasks) - before_task_count),
            "stack_delta": int(len(after_stacks.symmetric_difference(before_stacks))),
            "mode_delta": int(len(after_modes.symmetric_difference(before_modes))),
            "reject_reason": "",
            "used_stack_count": int(subtask_info.get("used_stack_count", len(after_stacks))),
        }

    def _select_priority_z_subtasks(self, limit: int = 2) -> List[int]:
        rows: List[Tuple[float, int]] = []
        route_proxy_default = float(sum(self.anchor_reference.get("stack_route_cost", {}).values()) / max(1, len(self.anchor_reference.get("stack_route_cost", {})) or 1))
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            subtask_stack_ids = {
                int(getattr(task, "target_stack_id", -1))
                for task in getattr(st, "execution_tasks", []) or []
                if int(getattr(task, "target_stack_id", -1)) >= 0
            }
            multi_stack_pen = max(0.0, float(len(subtask_stack_ids)) - 1.0)
            proc_curr = self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)
            proc_overflow = max(0.0, proc_curr - self._estimate_subtask_slack(st))
            noise_total = sum(float(len(getattr(task, "noise_tote_ids", []) or [])) for task in getattr(st, "execution_tasks", []) or [])
            route_proxy = sum(float(self.anchor_reference.get("stack_route_cost", {}).get(stack_id, route_proxy_default)) for stack_id in subtask_stack_ids)
            score = 3.0 * multi_stack_pen + 2.0 * proc_overflow + 0.5 * noise_total + 0.1 * route_proxy
            rows.append((float(score), sid))
        rows.sort(key=lambda item: (-item[0], item[1]))
        return [sid for _, sid in rows[:max(1, int(limit))]]

    def _select_priority_u_robot_ids(self, limit: int = 2) -> List[int]:
        robot_scores: Dict[int, float] = defaultdict(float)
        trip_lateness: Dict[Tuple[int, int], float] = defaultdict(float)
        for st in getattr(self.problem, "subtask_list", []) or []:
            start_time = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            for task in getattr(st, "execution_tasks", []) or []:
                rid = int(getattr(task, "robot_id", -1))
                trip_id = int(getattr(task, "trip_id", 0))
                if rid < 0:
                    continue
                arrival_station = float(getattr(task, "arrival_time_at_station", 0.0))
                arrival_stack = float(getattr(task, "arrival_time_at_stack", 0.0))
                lateness = max(0.0, arrival_station - start_time)
                trip_lateness[(rid, trip_id)] = max(trip_lateness[(rid, trip_id)], lateness)
                robot_scores[rid] += 0.5 * arrival_station + 0.2 * arrival_stack + 4.0 * lateness + float(len(getattr(task, "target_tote_ids", []) or []))
        for (rid, _), lateness in trip_lateness.items():
            robot_scores[rid] += 6.0 * lateness
        rows = sorted(((float(score), int(rid)) for rid, score in robot_scores.items()), key=lambda item: (-item[0], item[1]))
        return [rid for _, rid in rows[:max(1, int(limit))]]

    def _candidate_top_k(self, layer: str) -> int:
        layer = str(layer).upper()
        if layer == "Y":
            return max(1, int(getattr(self.cfg, "y_operator_topk", 3)))
        return max(1, int(self.cfg.surrogate_top_k_full))

    def _should_trigger_global_eval(self, layer: str, local_gain: float) -> Tuple[bool, str]:
        layer = str(layer).upper()
        if layer == str(self.cfg.force_global_after_layer).upper():
            return True, "force_after_layer"
        if local_gain >= float(self.cfg.early_global_if_local_gain_ge):
            return True, "early_gain_gate"
        if self.shadow_depth >= int(self.cfg.max_shadow_layers_without_global):
            return True, "shadow_depth_limit"
        return False, ""

    def _evaluate_shadow_candidate_globally(self, last_layer: str, iter_id: int) -> Tuple[SolutionSnapshot, float, Dict[str, float]]:
        runtime = {
            "simulate_calls": 0.0,
            "sp1_called": 0.0,
            "sp2_called": 0.0,
            "sp3_called": 0.0,
            "sp4_called": 0.0,
            "sp1_time_sec": 0.0,
            "sp2_time_sec": 0.0,
            "sp3_time_sec": 0.0,
            "sp4_time_sec": 0.0,
            "simulate_time_sec": 0.0,
        }
        last_layer = str(last_layer).upper()
        if last_layer == "X":
            t0 = time.perf_counter()
            if self.cfg.sp2_use_mip:
                self._run_sp2_mip()
            else:
                self._run_sp2_initial()
                self._recompute_station_schedule(
                    arrival_by_subtask=self.anchor_reference.get("subtask_arrival", {}),
                    proc_by_subtask={
                        int(getattr(st, "id", -1)): self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)
                        for st in getattr(self.problem, "subtask_list", []) or []
                    },
                )
            runtime["sp2_called"] += 1.0
            runtime["sp2_time_sec"] += float(time.perf_counter() - t0)
        if last_layer in {"X", "Y"}:
            t0 = time.perf_counter()
            self._run_sp3()
            runtime["sp3_called"] += 1.0
            runtime["sp3_time_sec"] += float(time.perf_counter() - t0)
        if last_layer in {"X", "Y"}:
            t0 = time.perf_counter()
            self._run_sp4_augmented()
            runtime["sp4_called"] += 1.0
            runtime["sp4_time_sec"] += float(time.perf_counter() - t0)
        elif last_layer == "Z":
            if bool(getattr(self.cfg, "u_global_sp4_polish", False)):
                t0 = time.perf_counter()
                self._run_sp4_augmented()
                runtime["sp4_called"] += 1.0
                runtime["sp4_time_sec"] += float(time.perf_counter() - t0)
            else:
                self._replay_u_routes()

        t0 = time.perf_counter()
        z = float(self.evaluate())
        runtime["simulate_calls"] = 1.0
        runtime["simulate_time_sec"] = float(time.perf_counter() - t0)
        self._harvest_station_start_times()
        self._update_beta_from_station()
        snap = self.snapshot(z, iter_id=iter_id, lightweight=True)
        return snap, float(z), runtime

    def _layer_augmented_acceptance_decision(self, actual_reduction: float, rho: float) -> Tuple[bool, str]:
        min_actual = float(getattr(self.cfg, "acceptance_min_actual_improve", 1e-6))
        if not math.isfinite(actual_reduction):
            return False, "invalid_actual_reduction"
        if actual_reduction <= min_actual:
            return False, "no_global_improvement"

        mode = str(getattr(self.cfg, "acceptance_mode", "strict_global")).lower()
        if mode == "ratio_gated":
            rho_min = float(getattr(self.cfg, "acceptance_rho_min", 0.05))
            if not math.isfinite(rho):
                return False, "invalid_rho"
            if rho < rho_min:
                return False, "rho_below_threshold"
            return True, "ratio_gated_accept"
        return True, "strict_global_accept"

    def _update_coupling_weights(self, score: Dict[str, Any], accepted: bool):
        step = float(self.cfg.lambda_step)
        decay = float(self.cfg.lambda_decay)
        lo = float(self.cfg.lambda_min)
        hi = float(self.cfg.lambda_max)
        reject_ema_cap = float(getattr(self.cfg, "lambda_reject_ema_cap", 20.0))
        layer = str(score.get("layer", "")).upper()
        for key, val in (score.get("couplings", {}) or {}).items():
            residual = float(val)
            prev = float(self.layer_residual_ema.get(key, 0.0))
            ema = 0.7 * prev + 0.3 * residual
            self.layer_residual_ema[key] = ema
            curr = float(self.layer_lambda_weights.get(key, float(self.cfg.lambda_init)))
            if accepted:
                curr = max(lo, curr * decay)
            else:
                curr = min(hi, curr + step * min(ema, reject_ema_cap))
            self.layer_lambda_weights[key] = max(lo, min(hi, curr))
        reset_after = max(1, int(getattr(self.cfg, "lambda_reset_after_reject_streak", 3)))
        if not accepted and layer and int(self.layer_global_reject_streak.get(layer, 0)) >= reset_after:
            init_val = float(getattr(self.cfg, "lambda_init", 1.0))
            for key in self._layer_coupling_keys(layer):
                curr = float(self.layer_lambda_weights.get(key, init_val))
                self.layer_lambda_weights[key] = max(lo, min(hi, 0.5 * (curr + init_val)))

    def _run_layer_vns(self, layer: str, iter_id: int) -> Tuple[Optional[SolutionSnapshot], Dict[str, Any], Dict[str, float]]:
        assert self.anchor is not None
        self.restore_snapshot(self.anchor)
        if layer == "Y":
            if any(int(getattr(st, "assigned_station_id", -1)) < 0 for st in getattr(self.problem, "subtask_list", []) or []):
                self._seed_station_assignments_from_anchor()
        y_context = self._build_sp2_layer_context() if layer == "Y" else None
        baseline = self._compute_augmented_layer_objective(layer)
        best_score: Optional[Dict[str, Any]] = None
        best_snap: Optional[SolutionSnapshot] = None
        candidate_pool: List[Dict[str, Any]] = []
        runtime = {
            "trial_count": 0.0,
            "sp1_called": 0.0,
            "sp2_called": 0.0,
            "sp3_called": 0.0,
            "sp4_called": 0.0,
            "sp1_time_sec": 0.0,
            "sp2_time_sec": 0.0,
            "sp3_time_sec": 0.0,
            "sp4_time_sec": 0.0,
            "simulate_calls": 0.0,
            "simulate_time_sec": 0.0,
            "u_move_type": "",
            "u_changed_task_count": 0.0,
            "u_trip_count_delta": 0.0,
            "u_replay_feasible": False,
            "z_move_type": "",
            "z_changed_subtask_count": 0.0,
            "z_candidate_task_delta": 0.0,
            "z_candidate_stack_delta": 0.0,
            "z_candidate_mode_delta": 0.0,
            "candidate_count": 0.0,
            "candidate_topk": 0.0,
            "selected_candidate_rank": 0.0,
            "selected_operator": "",
            "selected_operator_rank": 0.0,
            "operator_budget": 0.0,
            "operator_reward_pred": 0.0,
            "operator_reward_actual": 0.0,
            "operator_reward_total": 0.0,
            "shake_strength": float(self._layer_shake_strength(layer)),
            "restart_triggered": float(self.stagnation_stats.get(str(layer).upper(), {}).get("restart_triggered", 0.0)),
            "proposal_pass_fast_gate": True,
            "fast_gate_reason": "",
            "x_fast_gate_band": "",
            "trigger_gate_open": True,
            "trigger_gate_reason": "",
            "late_task_count": float(self._compute_late_task_count()),
            "y_search_mode": "",
            "y_duplicate_skipped": False,
            "y_signature": "",
            "y_station_change_count": 0.0,
            "y_rank_change_count": 0.0,
            "forced_eval_origin_layer": "",
            "global_eval_candidate_count": 0.0,
            "y_fast_rank": 0.0,
            "y_route_sim_rank": 0.0,
            "y_route_sim_score": float("nan"),
            "y_route_sim_global_makespan_proxy": float("nan"),
            "y_route_sim_arrival_slack_mean": float("nan"),
            "y_route_sim_late_task_count": float("nan"),
            "y_route_sim_replayed_sp4": False,
            "y_route_sim_used_incremental_route": False,
            "x_destroy_operator": "",
            "x_repair_operator": "",
            "x_destroy_size": 0.0,
            "x_subtask_count_before": float(len(self._iter_snapshot_subtasks(self.anchor))),
            "x_subtask_count_after": float("nan"),
            "x_affinity_penalty": float("nan"),
            "x_route_conflict_penalty": float("nan"),
            "x_finish_time_dispersion_penalty": float("nan"),
            "x_candidates_generated_count": 0.0,
            "x_candidates_pruned_count": 0.0,
            "x_candidate_details": [],
        }

        self.current_trigger_gate = {"layer": layer, "open": True, "reason": ""}
        y_search_mode = self._y_search_mode() if layer == "Y" else ""
        runtime["y_search_mode"] = y_search_mode
        budget = self._layer_operator_budget(layer)
        operator_sequence = self._select_operator_sequence(layer, budget)
        x_operator_pairs: List[Tuple[str, str]] = []
        if layer == "X":
            x_operator_pairs = self._select_x_operator_pairs(max(1, int(getattr(self.cfg, "x_operator_pair_budget", budget))))
            operator_sequence = [f"{destroy}|{repair}" for destroy, repair in x_operator_pairs]
        if layer == "Y":
            operator_sequence = self._filter_y_operator_sequence(operator_sequence, y_search_mode, len(operator_sequence))
        elif layer == "U":
            aggressive_u = (
                float(runtime.get("late_task_count", 0.0)) >= float(getattr(self.cfg, "u_aggressive_trigger_late_task_count", 24))
                or float(self._collect_layer_metrics().get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "u_aggressive_trigger_arrival_slack_mean", 180.0))
            )
            if str(self.last_layer_accept).upper() != "Y" and not aggressive_u:
                operator_sequence = []
            elif int(budget) <= 1 or not aggressive_u:
                operator_sequence = [op for op in operator_sequence if op in {"u_late_task_pull_forward", "u_segment_reverse"}][:1]
        runtime["operator_budget"] = float(len(operator_sequence))
        if layer in {"Z", "U"}:
            runtime["trigger_gate_open"] = bool(self.current_trigger_gate.get("open", True))
            runtime["trigger_gate_reason"] = str(self.current_trigger_gate.get("reason", ""))
            runtime["late_task_count"] = float(self.current_trigger_gate.get("late_task_count", runtime["late_task_count"]))
        trial_limit = len(operator_sequence)
        z_priority = self._select_priority_z_subtasks(limit=2) if layer == "Z" else []
        u_priority = self._select_priority_u_robot_ids(limit=2) if layer == "U" else []
        shake_strength = self._layer_shake_strength(layer)
        if layer == "Y" and float(self.stagnation_stats.get("Y", {}).get("restart_triggered", 0.0)) > 0.0 and y_context is not None:
            y_context.tau_y = max(0.25, float(y_context.tau_y) * 0.5)
        for trial, operator_name in enumerate(operator_sequence):
            self.restore_snapshot(self.anchor)
            runtime["trial_count"] += 1.0
            rng = random.Random(int(self.cfg.seed) + iter_id * 1009 + trial * 131 + sum(ord(c) for c in str(layer)))

            if layer == "X":
                if trial >= len(x_operator_pairs):
                    continue
                destroy_operator, repair_operator = x_operator_pairs[trial]
                proposal = self._extract_x_split_solution()
                destroy_size = self._apply_x_destroy_operator(proposal, destroy_operator, rng, shake_strength)
                repaired = self._apply_x_repair_operator(proposal, repair_operator, rng, shake_strength)
                if destroy_size <= 0 or not repaired:
                    continue
                self._project_x_split_solution_to_problem(proposal)
                score = self._compute_augmented_layer_objective(layer)
                score["proposal_pass_fast_gate"] = True
                score["fast_gate_reason"] = ""
                score["x_fast_gate_band"] = ""
                score["x_destroy_operator"] = str(destroy_operator)
                score["x_repair_operator"] = str(repair_operator)
                score["x_destroy_size"] = float(destroy_size)
            elif layer == "Y":
                if any(int(getattr(st, "assigned_station_id", -1)) < 0 for st in getattr(self.problem, "subtask_list", []) or []):
                    self._seed_station_assignments_from_anchor()
                assert y_context is not None
                t0 = time.perf_counter()
                changed = self._apply_y_operator(
                    operator_name,
                    rng,
                    shake_strength,
                    y_context,
                    search_mode=y_search_mode,
                )
                if not changed:
                    continue
                y_signature = self._compute_y_assignment_signature()
                if y_signature in self.y_recent_signatures:
                    runtime["y_duplicate_skipped"] = True
                    runtime["y_signature"] = y_signature
                    continue
                local_result = self.sp2.summarize_local_layer(
                    getattr(self.problem, "subtask_list", []) or [],
                    y_context,
                    apply_to_tasks=False,
                )
                self.last_sp2_local_result = local_result
                runtime["sp2_called"] += 1.0
                runtime["sp2_time_sec"] += float(time.perf_counter() - t0)
                fast_eval = self._fast_y_integrated_eval(y_context)
                prox_penalty = float(fast_eval.prox_station_penalty) + float(self.sp2.local_prox_rank_weight) * float(fast_eval.prox_rank_penalty)
                coupling_penalty = (
                    float(y_context.lambda_yx) * float(fast_eval.station_preference_penalty)
                    + float(y_context.lambda_yu) * float(fast_eval.waiting_penalty + fast_eval.queue_penalty + fast_eval.arrival_misalignment_penalty)
                    + float(y_context.lambda_yz) * float(fast_eval.load_balance_penalty)
                )
                score = {
                    "layer": "Y",
                    "local_obj": float(fast_eval.approx_makespan),
                    "coupling_penalty": float(coupling_penalty),
                    "prox_penalty": float(prox_penalty),
                    "augmented_obj": float(fast_eval.objective_value),
                    "couplings": {
                        "yx": float(fast_eval.station_preference_penalty),
                        "yu": float(fast_eval.waiting_penalty + fast_eval.queue_penalty + fast_eval.arrival_misalignment_penalty),
                        "yz": float(fast_eval.load_balance_penalty),
                    },
                    "y_fast_objective": float(fast_eval.objective_value),
                    "y_fast_approx_makespan": float(fast_eval.approx_makespan),
                    "y_fast_station_cmax": float(fast_eval.station_cmax),
                    "y_fast_waiting_penalty": float(fast_eval.waiting_penalty),
                    "y_fast_queue_penalty": float(fast_eval.queue_penalty),
                    "y_fast_arrival_misalignment_penalty": float(fast_eval.arrival_misalignment_penalty),
                    "y_fast_load_balance_penalty": float(fast_eval.load_balance_penalty),
                    "y_fast_station_finish_max": float(max(fast_eval.station_finish_times.values()) if fast_eval.station_finish_times else 0.0),
                    "y_precheck_score": float("nan"),
                    "y_precheck_sorting_cost_delta": float("nan"),
                    "y_precheck_station_cmax": float("nan"),
                    "y_precheck_arrival_slack_delta": float("nan"),
                    "y_route_sim_score": float("nan"),
                    "y_route_sim_global_makespan_proxy": float("nan"),
                    "y_route_sim_arrival_slack_mean": float("nan"),
                    "y_route_sim_late_task_count": float("nan"),
                    "y_route_sim_replayed_sp4": False,
                    "y_route_sim_used_incremental_route": False,
                    "proposal_pass_fast_gate": True,
                    "y_local_cmax": float(local_result.cmax_value),
                    "y_waiting_penalty": float(local_result.waiting_penalty),
                    "y_load_balance_penalty": float(local_result.load_balance_penalty),
                    "y_station_preference_penalty": float(local_result.station_preference_penalty),
                    "y_prox_station_penalty": float(local_result.prox_station_penalty),
                    "y_prox_rank_penalty": float(local_result.prox_rank_penalty),
                    "y_signature": y_signature,
                }
                y_counts = self._compute_y_change_counts()
                score["y_station_change_count"] = float(y_counts["station_change_count"])
                score["y_rank_change_count"] = float(y_counts["rank_change_count"])
            elif layer == "Z":
                z_move = self._apply_z_operator(operator_name, rng, shake_strength, priority_subtask_ids=z_priority)
                if not bool(z_move.get("feasible", False)):
                    continue
                score = self._compute_augmented_layer_objective(layer)
            else:
                u_move = self._apply_u_operator(operator_name, rng, shake_strength, priority_robot_ids=u_priority)
                if not bool(u_move.get("feasible", False)):
                    continue
                self._apply_u_route_state(u_move.get("state", {}))
                if not self._replay_u_routes():
                    continue
                score = self._compute_augmented_layer_objective(layer)
            move_meta: Dict[str, Any] = {}
            if layer == "Z":
                move_meta = {
                    "z_move_type": str(z_move.get("move_type", "")),
                    "z_changed_subtask_count": float(z_move.get("changed_subtask_count", 0)),
                    "z_candidate_task_delta": float(z_move.get("task_delta", 0)),
                    "z_candidate_stack_delta": float(z_move.get("stack_delta", 0)),
                    "z_candidate_mode_delta": float(z_move.get("mode_delta", 0)),
                }
            elif layer == "U":
                move_meta = {
                    "u_move_type": str(u_move.get("move_type", "")),
                    "u_changed_task_count": float(u_move.get("changed_task_count", 0)),
                    "u_trip_count_delta": float(u_move.get("trip_count_delta", 0)),
                    "u_replay_feasible": True,
                }
            elif layer == "X":
                move_meta = {
                    "x_destroy_operator": str(score.get("x_destroy_operator", "")),
                    "x_repair_operator": str(score.get("x_repair_operator", "")),
                    "x_destroy_size": float(score.get("x_destroy_size", 0.0)),
                    "x_subtask_count_after": float(score.get("x_subtask_count_after", score.get("x_subtask_count", float("nan")))),
                    "x_affinity_penalty": float(score.get("x_affinity_penalty", float("nan"))),
                    "x_route_conflict_penalty": float(score.get("x_route_conflict_penalty", float("nan"))),
                    "x_finish_time_dispersion_penalty": float(score.get("x_finish_time_dispersion_penalty", float("nan"))),
                }
            candidate_pool.append({
                "score": dict(score),
                "snapshot": self.snapshot(
                    self.anchor_z if math.isfinite(self.anchor_z) else self.work_z,
                    iter_id=iter_id,
                    lightweight=True,
                ),
                "meta": move_meta,
                "operator": str(operator_name),
                "operator_rank": int(trial + 1),
            })

        if candidate_pool:
            if layer == "X" and bool(getattr(self.cfg, "enable_fast_x_gate", True)):
                min_penalty = min(float(item["score"].get("x_fast_penalty", float("inf"))) for item in candidate_pool)
                soft_relax = float(getattr(self.cfg, "x_fast_soft_relax", 0.20))
                hard_relax = float(getattr(self.cfg, "x_fast_hard_relax", 0.60))
                borderline_gain_ratio = float(getattr(self.cfg, "x_fast_borderline_gain_ratio", 0.03))
                arrival_cap = float(getattr(self.cfg, "x_fast_arrival_shift_rel_cap", 0.35))
                for item in candidate_pool:
                    score = item["score"]
                    penalty = float(score.get("x_fast_penalty", float("inf")))
                    predicted_gain_ratio = max(
                        0.0,
                        float(baseline.get("local_obj", float("inf"))) - float(score.get("local_obj", float("inf"))),
                    ) / max(1.0, abs(float(baseline.get("local_obj", 1.0))))
                    arrival_shift = float(score.get("x_fast_delta_arrival_shift", float("inf")))
                    gate_pass = False
                    gate_band = "reject_hard"
                    gate_reason = ""
                    if arrival_shift > arrival_cap + 1e-9:
                        gate_reason = "fast_x_arrival_shift_cap"
                    elif penalty <= min_penalty * (1.0 + soft_relax) + 1e-9:
                        gate_pass = True
                        gate_band = "pass_soft"
                    elif penalty <= min_penalty * (1.0 + hard_relax) + 1e-9 and predicted_gain_ratio >= borderline_gain_ratio:
                        gate_pass = True
                        gate_band = "pass_borderline"
                    else:
                        gate_reason = "fast_x_relative_penalty"
                    score["proposal_pass_fast_gate"] = bool(gate_pass)
                    score["fast_gate_reason"] = str(gate_reason)
                    score["x_fast_gate_band"] = str(gate_band)
            candidate_pool.sort(key=lambda item: (float(item["score"].get("augmented_obj", float("inf"))), float(item["score"].get("local_obj", float("inf")))))
            top_k = min(len(candidate_pool), self._candidate_top_k(layer))
            runtime["candidate_count"] = float(len(candidate_pool))
            runtime["candidate_topk"] = float(top_k)
            if layer == "X":
                runtime["x_candidates_generated_count"] = float(len(candidate_pool))
            for rank_idx, item in enumerate(candidate_pool, start=1):
                item["rank"] = int(rank_idx)
            shortlisted = list(candidate_pool[:top_k])
            selection_pool = shortlisted
            if layer == "X":
                passing = [item for item in shortlisted if bool(item["score"].get("proposal_pass_fast_gate", False))]
                if passing:
                    selection_pool = passing
                candidate_details = []
                passing_ids = {id(item) for item in passing}
                shortlisted_ids = {id(item) for item in shortlisted}
                for item in candidate_pool:
                    score = dict(item.get("score", {}) or {})
                    item_reason = ""
                    if id(item) not in shortlisted_ids:
                        item_reason = "pruned_topk"
                    elif not bool(score.get("proposal_pass_fast_gate", False)):
                        item_reason = str(score.get("fast_gate_reason", "fast_gate_reject"))
                    candidate_details.append({
                        "rank": int(item.get("rank", 0)),
                        "destroy_operator": str(score.get("x_destroy_operator", "")),
                        "repair_operator": str(score.get("x_repair_operator", "")),
                        "combined_operator": str(item.get("operator", "")),
                        "destroy_size": float(score.get("x_destroy_size", 0.0)),
                        "local_obj": float(score.get("local_obj", float("nan"))),
                        "augmented_obj": float(score.get("augmented_obj", float("nan"))),
                        "x_fast_penalty": float(score.get("x_fast_penalty", float("nan"))),
                        "x_fast_gate_band": str(score.get("x_fast_gate_band", "")),
                        "proposal_pass_fast_gate": bool(score.get("proposal_pass_fast_gate", False)),
                        "pruned_reason": item_reason,
                        "would_shortlist": bool(id(item) in shortlisted_ids),
                        "would_pass_fast_gate": bool(id(item) in passing_ids),
                        "z_evaluated": None,
                    })
                runtime["x_candidates_pruned_count"] = float(sum(1 for row in candidate_details if str(row.get("pruned_reason", "")).strip()))
                runtime["x_candidate_details"] = candidate_details
            elif layer == "Y":
                route_eval_topk = max(1, int(getattr(self.cfg, "y_route_eval_topk", 2)))
                route_eval_candidates = list(shortlisted[: min(len(shortlisted), route_eval_topk)])
                route_scored: List[Dict[str, Any]] = []
                for fast_rank, item in enumerate(route_eval_candidates, start=1):
                    snap = item.get("snapshot")
                    if snap is None:
                        continue
                    self.restore_snapshot(snap)
                    y_signature = str((item.get("score", {}) or {}).get("y_signature", ""))
                    route_eval = self._evaluate_y_candidate_with_route_sim(y_signature)
                    item_score = dict(item.get("score", {}) or {})
                    item_score["y_fast_rank"] = float(fast_rank)
                    item_score["y_route_sim_score"] = float(route_eval.objective_value)
                    item_score["y_route_sim_global_makespan_proxy"] = float(route_eval.global_makespan_proxy)
                    item_score["y_route_sim_arrival_slack_mean"] = float(route_eval.arrival_slack_mean)
                    item_score["y_route_sim_late_task_count"] = float(route_eval.late_task_count)
                    item_score["y_route_sim_replayed_sp4"] = bool(route_eval.replayed_sp4)
                    item_score["y_route_sim_used_incremental_route"] = bool(route_eval.used_incremental_route)
                    item["score"] = item_score
                    route_scored.append(item)
                if route_scored:
                    route_scored.sort(
                        key=lambda item: (
                            float((item.get("score", {}) or {}).get("y_route_sim_score", float("inf"))),
                            float((item.get("score", {}) or {}).get("y_fast_objective", float("inf"))),
                            int(item.get("rank", 10 ** 9)),
                        )
                    )
                    for route_rank, item in enumerate(route_scored, start=1):
                        (item.get("score", {}) or {})["y_route_sim_rank"] = float(route_rank)
                    selection_pool = route_scored
            chosen = min(
                selection_pool,
                key=lambda item: (
                    float(item["score"].get("y_route_sim_score", item["score"].get("augmented_obj", float("inf")))),
                    float(item["score"].get("augmented_obj", float("inf"))),
                    int(item.get("rank", 10 ** 9)),
                ),
            )
            best_score = dict(chosen["score"])
            best_snap = chosen["snapshot"]
            if layer == "X":
                best_score["_x_all_candidates"] = [
                    {
                        "snapshot": item["snapshot"],
                        "score": dict(item["score"]),
                        "operator": str(item.get("operator", "")),
                        "operator_rank": int(item.get("operator_rank", 0)),
                        "rank": int(item.get("rank", 0)),
                        "meta": dict(item.get("meta", {}) or {}),
                    }
                    for item in candidate_pool
                ]
            runtime["selected_candidate_rank"] = float(chosen.get("rank", 0))
            runtime["selected_operator"] = str(chosen.get("operator", ""))
            runtime["selected_operator_rank"] = float(chosen.get("operator_rank", 0))
            runtime["proposal_pass_fast_gate"] = bool(best_score.get("proposal_pass_fast_gate", layer != "X"))
            runtime["fast_gate_reason"] = str(best_score.get("fast_gate_reason", ""))
            runtime["x_fast_gate_band"] = str(best_score.get("x_fast_gate_band", ""))
            runtime["y_signature"] = str(best_score.get("y_signature", ""))
            runtime["y_station_change_count"] = float(best_score.get("y_station_change_count", 0.0))
            runtime["y_rank_change_count"] = float(best_score.get("y_rank_change_count", 0.0))
            runtime["y_fast_rank"] = float(best_score.get("y_fast_rank", 0.0))
            runtime["y_route_sim_rank"] = float(best_score.get("y_route_sim_rank", 0.0))
            runtime["y_route_sim_score"] = float(best_score.get("y_route_sim_score", float("nan")))
            runtime["y_route_sim_global_makespan_proxy"] = float(best_score.get("y_route_sim_global_makespan_proxy", float("nan")))
            runtime["y_route_sim_arrival_slack_mean"] = float(best_score.get("y_route_sim_arrival_slack_mean", float("nan")))
            runtime["y_route_sim_late_task_count"] = float(best_score.get("y_route_sim_late_task_count", float("nan")))
            runtime["y_route_sim_replayed_sp4"] = bool(best_score.get("y_route_sim_replayed_sp4", False))
            runtime["y_route_sim_used_incremental_route"] = bool(best_score.get("y_route_sim_used_incremental_route", False))
            for key, value in (chosen.get("meta", {}) or {}).items():
                runtime[key] = value
            if layer == "Y":
                eval_candidates: List[Dict[str, Any]] = [chosen]
                max_eval_topk = max(1, int(getattr(self.cfg, "y_global_eval_topk", 2)))
                gap_ratio_limit = float(getattr(self.cfg, "y_dual_eval_gap_ratio", 0.05))
                baseline_aug = float(baseline.get("augmented_obj", float("inf")))
                compare_pool = list(selection_pool[: max_eval_topk])
                if max_eval_topk >= 2 and len(compare_pool) >= 2:
                    top1 = compare_pool[0]
                    top2 = compare_pool[1]
                    top1_score = float(top1["score"].get("y_route_sim_score", top1["score"].get("y_fast_objective", top1["score"].get("augmented_obj", float("inf")))))
                    top2_score = float(top2["score"].get("y_route_sim_score", top2["score"].get("y_fast_objective", top2["score"].get("augmented_obj", float("inf")))))
                    gap_ratio = abs(top2_score - top1_score) / max(1.0, abs(top1_score))
                    if (
                        gap_ratio <= gap_ratio_limit + 1e-9
                        and str(top1["score"].get("y_signature", "")) != str(top2["score"].get("y_signature", ""))
                        and float(top2["score"].get("augmented_obj", float("inf"))) < baseline_aug - 1e-9
                    ):
                        eval_candidates.append(top2)
                best_score["_global_eval_candidates"] = [
                    {
                        "snapshot": item["snapshot"],
                        "score": dict(item["score"]),
                        "operator": str(item.get("operator", "")),
                        "operator_rank": int(item.get("operator_rank", 0)),
                        "rank": int(item.get("rank", 0)),
                        "meta": dict(item.get("meta", {}) or {}),
                    }
                    for item in eval_candidates
                ]

        if best_score is None:
            best_score = dict(baseline)
        best_score["baseline_augmented_obj"] = float(baseline["augmented_obj"])
        best_score["baseline_local_obj"] = float(baseline["local_obj"])
        return best_snap, best_score, runtime

    def _run_layer_augmented_main(self) -> float:
        assert self.best is not None
        assert self.anchor is not None
        assert self.shadow is not None

        mark = 0
        cycle_global_eval_count = 0
        cycle_best_skipped: Optional[Dict[str, Any]] = None
        self._notify_progress(0, self.cfg.max_iters, "init")
        for it in range(1, self.cfg.max_iters + 1):
            if it >= self.cfg.switch_to_exact_iter:
                self.cfg.sp2_use_mip = self.cfg.exact_sp2_use_mip
                self.cfg.sp3_use_mip = self.cfg.exact_sp3_use_mip
                self.cfg.sp4_use_mip = self.cfg.exact_sp4_use_mip
                self.cfg.sp2_time_limit_sec = self.cfg.exact_sp2_time_limit_sec
                self.cfg.sp4_lkh_time_limit_seconds = self.cfg.exact_sp4_lkh_time_limit_seconds

            t_iter0 = time.perf_counter()
            layer = self.layer_names[(it - 1) % len(self.layer_names)]
            self.shadow = self.anchor
            self.work = self.anchor
            self.restore_snapshot(self.anchor)
            best_snap, best_score, runtime = self._run_layer_vns(layer, it)
            base_aug = float(best_score.get("baseline_augmented_obj", float("inf")))
            base_local = float(best_score.get("baseline_local_obj", float("nan")))
            cand_aug = float(best_score.get("augmented_obj", float("inf")))
            proposal_pass_fast_gate = bool(runtime.get("proposal_pass_fast_gate", True))
            global_z_before = float(self.anchor_z)
            global_z_after = float(self.anchor_z)
            predicted_reduction = 0.0
            actual_reduction = 0.0
            rho = float("nan")
            resolved_eval_layer = str(layer)
            forced_eval_origin_layer = ""
            if layer == "X":
                proposal_pass_surrogate = bool(best_snap is not None and cand_aug < base_aug - 1e-9 and proposal_pass_fast_gate)
            else:
                proposal_pass_surrogate = bool(best_snap is not None and cand_aug < base_aug - 1e-9)
            global_eval_triggered = False
            global_eval_reason = ""
            global_accept = "reject"
            rollback_happened = False
            commit_decision = "reject_surrogate"
            commit_reason = "proposal_not_better_than_baseline" if best_snap is not None else "no_candidate"
            improved = False
            iter_z_real = float(self.anchor_z)
            iter_metrics = dict(self._collect_layer_metrics())
            runtime["global_eval_candidate_count"] = 0.0
            if runtime.get("operator_budget", 0.0) <= 0.0 and layer in {"Z", "U"}:
                commit_decision = "skip_trigger_gate"
                commit_reason = str(runtime.get("trigger_gate_reason", "skip_trigger_gate"))
            elif best_snap is not None and cand_aug < base_aug - 1e-9 and not proposal_pass_fast_gate:
                commit_decision = "reject_fast_x_gate"
                commit_reason = str(runtime.get("fast_gate_reason", "fast_gate_reject"))

            if proposal_pass_surrogate and math.isfinite(base_aug) and math.isfinite(cand_aug):
                predicted_reduction = float(base_aug - cand_aug)

            if best_snap is not None and cand_aug < base_aug - 1e-9 and not proposal_pass_surrogate:
                gain_ratio = float((base_aug - cand_aug) / max(1.0, abs(base_aug)))
                skipped_candidate = {
                    "origin_layer": str(layer),
                    "snapshot": best_snap,
                    "score": dict(best_score),
                    "operator": str(runtime.get("selected_operator", "")),
                    "operator_rank": int(runtime.get("selected_operator_rank", 0.0)),
                    "rank": int(runtime.get("selected_candidate_rank", 0.0)),
                    "meta": {},
                    "gain_ratio": float(gain_ratio),
                }
                if (
                    cycle_best_skipped is None
                    or float(skipped_candidate["gain_ratio"]) > float(cycle_best_skipped.get("gain_ratio", -float("inf"))) + 1e-12
                    or (
                        abs(float(skipped_candidate["gain_ratio"]) - float(cycle_best_skipped.get("gain_ratio", -float("inf")))) <= 1e-12
                        and float(skipped_candidate["score"].get("augmented_obj", float("inf"))) < float(cycle_best_skipped.get("score", {}).get("augmented_obj", float("inf")))
                    )
                ):
                    cycle_best_skipped = skipped_candidate

            eval_candidates: List[Dict[str, Any]] = []
            if layer == "X" and bool(runtime.get("x_candidate_details")):
                global_eval_triggered = True
                global_eval_reason = "all_x_candidates"
                for detail in list(runtime.get("x_candidate_details", []) or []):
                    score_item = None
                    snap_item = None
                    for item in list(best_score.get("_x_all_candidates", []) or []):
                        item_score = dict(item.get("score", {}) or {})
                        if (
                            str(item_score.get("x_destroy_operator", "")) == str(detail.get("destroy_operator", ""))
                            and str(item_score.get("x_repair_operator", "")) == str(detail.get("repair_operator", ""))
                            and int(item.get("rank", 0)) == int(detail.get("rank", 0))
                        ):
                            score_item = item_score
                            snap_item = item.get("snapshot")
                            break
                    if snap_item is None or score_item is None:
                        continue
                    eval_candidates.append({
                        "origin_layer": "X",
                        "snapshot": snap_item,
                        "score": score_item,
                        "operator": f"{score_item.get('x_destroy_operator', '')}|{score_item.get('x_repair_operator', '')}",
                        "operator_rank": int(detail.get("rank", 0)),
                        "rank": int(detail.get("rank", 0)),
                        "meta": {
                            "x_destroy_operator": str(score_item.get("x_destroy_operator", "")),
                            "x_repair_operator": str(score_item.get("x_repair_operator", "")),
                            "x_destroy_size": float(score_item.get("x_destroy_size", 0.0)),
                            "x_subtask_count_after": float(score_item.get("x_subtask_count_after", float("nan"))),
                            "x_affinity_penalty": float(score_item.get("x_affinity_penalty", float("nan"))),
                            "x_route_conflict_penalty": float(score_item.get("x_route_conflict_penalty", float("nan"))),
                            "x_finish_time_dispersion_penalty": float(score_item.get("x_finish_time_dispersion_penalty", float("nan"))),
                        },
                    })
            elif proposal_pass_surrogate and best_snap is not None:
                global_eval_triggered = True
                global_eval_reason = "surrogate_pass"
                if layer == "Y":
                    for item in list(best_score.get("_global_eval_candidates", []) or []):
                        score_item = dict(item.get("score", {}) or {})
                        if float(score_item.get("augmented_obj", float("inf"))) < base_aug - 1e-9:
                            eval_candidates.append({
                                "origin_layer": "Y",
                                "snapshot": item.get("snapshot"),
                                "score": score_item,
                                "operator": str(item.get("operator", "")),
                                "operator_rank": int(item.get("operator_rank", 0)),
                                "rank": int(item.get("rank", 0)),
                                "meta": dict(item.get("meta", {}) or {}),
                            })
                if not eval_candidates:
                    eval_candidates = [{
                        "origin_layer": str(layer),
                        "snapshot": best_snap,
                        "score": dict(best_score),
                        "operator": str(runtime.get("selected_operator", "")),
                        "operator_rank": int(runtime.get("selected_operator_rank", 0.0)),
                        "rank": int(runtime.get("selected_candidate_rank", 0.0)),
                        "meta": {},
                    }]

            if (
                not eval_candidates
                and layer == "X"
                and best_snap is not None
                and cand_aug < base_aug - 1e-9
                and not proposal_pass_fast_gate
                and int(self.layer_fast_gate_reject_streak.get("X", 0)) + 1 >= max(1, int(getattr(self.cfg, "x_force_eval_period", 2)))
            ):
                eval_candidates = [{
                    "origin_layer": "X",
                    "snapshot": best_snap,
                    "score": dict(best_score),
                    "operator": str(runtime.get("selected_operator", "")),
                    "operator_rank": int(runtime.get("selected_operator_rank", 0.0)),
                    "rank": int(runtime.get("selected_candidate_rank", 0.0)),
                    "meta": {},
                }]
                global_eval_triggered = True
                global_eval_reason = "forced:x_fast_gate_period"
                forced_eval_origin_layer = "X"
                runtime["forced_eval_origin_layer"] = "X"

            if not eval_candidates:
                force_allowed = (
                    cycle_global_eval_count <= 0
                    and cycle_best_skipped is not None
                    and str(layer).upper() == str(getattr(self.cfg, "force_global_after_layer", "U")).upper()
                )
                if force_allowed:
                    should_force, force_reason = self._should_trigger_global_eval(str(layer), float(cycle_best_skipped.get("gain_ratio", 0.0)))
                    if should_force:
                        eval_candidates = [dict(cycle_best_skipped)]
                        global_eval_triggered = True
                        global_eval_reason = f"forced:{force_reason}"
                        forced_eval_origin_layer = str(cycle_best_skipped.get("origin_layer", ""))
                        runtime["forced_eval_origin_layer"] = forced_eval_origin_layer
                        runtime["selected_operator"] = str(cycle_best_skipped.get("operator", ""))
                        runtime["selected_operator_rank"] = float(cycle_best_skipped.get("operator_rank", 0))
                        runtime["selected_candidate_rank"] = float(cycle_best_skipped.get("rank", 0))
                        best_score = dict(cycle_best_skipped.get("score", {}) or {})
                        best_snap = cycle_best_skipped.get("snapshot")
                        base_aug = float(best_score.get("baseline_augmented_obj", base_aug))
                        cand_aug = float(best_score.get("augmented_obj", cand_aug))
                        proposal_pass_fast_gate = bool(best_score.get("proposal_pass_fast_gate", proposal_pass_fast_gate))
                        predicted_reduction = max(0.0, float(base_aug - cand_aug)) if math.isfinite(base_aug) and math.isfinite(cand_aug) else 0.0
                        commit_decision = "reject_surrogate"
                        commit_reason = "forced_eval_pending"

            if eval_candidates:
                runtime["global_eval_candidate_count"] = float(len(eval_candidates))
                best_eval: Optional[Dict[str, Any]] = None
                for candidate in eval_candidates:
                    snap = candidate.get("snapshot")
                    if snap is None:
                        continue
                    origin_layer = str(candidate.get("origin_layer", layer)).upper()
                    self.shadow = snap
                    self.restore_snapshot(snap)
                    full_snap, z_new, eval_runtime = self._evaluate_shadow_candidate_globally(origin_layer, it)
                    if origin_layer == "X":
                        self._compute_x_affinity_matrix_from_current_solution(int(it))
                    self.global_eval_count += 1
                    cycle_global_eval_count += 1
                    for key, value in eval_runtime.items():
                        runtime[key] = float(runtime.get(key, 0.0)) + float(value)
                    eval_row = {
                        "origin_layer": origin_layer,
                        "full_snap": full_snap,
                        "z": float(z_new),
                        "score": dict(candidate.get("score", {}) or {}),
                        "operator": str(candidate.get("operator", "")),
                        "operator_rank": int(candidate.get("operator_rank", 0)),
                        "rank": int(candidate.get("rank", 0)),
                        "meta": dict(candidate.get("meta", {}) or {}),
                    }
                    if origin_layer == "X":
                        for detail in list(runtime.get("x_candidate_details", []) or []):
                            if (
                                int(detail.get("rank", 0)) == int(candidate.get("rank", 0))
                                and str(detail.get("destroy_operator", "")) == str((candidate.get("score", {}) or {}).get("x_destroy_operator", ""))
                                and str(detail.get("repair_operator", "")) == str((candidate.get("score", {}) or {}).get("x_repair_operator", ""))
                            ):
                                detail["z_evaluated"] = float(z_new)
                                break
                    if best_eval is None or float(eval_row["z"]) < float(best_eval["z"]) - 1e-9:
                        best_eval = eval_row
                if best_eval is not None:
                    resolved_eval_layer = str(best_eval.get("origin_layer", layer)).upper()
                    best_score = dict(best_eval.get("score", {}) or {})
                    best_score["baseline_augmented_obj"] = float(base_aug)
                    best_score["baseline_local_obj"] = float(base_local)
                    best_snap = best_eval.get("full_snap")
                    global_z_after = float(best_eval.get("z", global_z_before))
                    actual_reduction = float(global_z_before - global_z_after)
                    predicted_reduction = max(
                        0.0,
                        float(best_score.get("baseline_augmented_obj", base_aug)) - float(best_score.get("augmented_obj", cand_aug)),
                    )
                    rho = float(actual_reduction / max(predicted_reduction, 1e-9)) if math.isfinite(actual_reduction) else float("nan")
                    runtime["selected_operator"] = str(best_eval.get("operator", runtime.get("selected_operator", "")))
                    runtime["selected_operator_rank"] = float(best_eval.get("operator_rank", runtime.get("selected_operator_rank", 0.0)))
                    runtime["selected_candidate_rank"] = float(best_eval.get("rank", runtime.get("selected_candidate_rank", 0.0)))
                    for key, value in (best_eval.get("meta", {}) or {}).items():
                        runtime[key] = value
                    if resolved_eval_layer == "Y":
                        for candidate in eval_candidates:
                            sig = str((candidate.get("score", {}) or {}).get("y_signature", "")).strip()
                            if sig:
                                self.y_recent_signatures.append(sig)
                    accepted, accept_reason = self._layer_augmented_acceptance_decision(actual_reduction, rho)
                    if accepted:
                        global_accept = "accept"
                        commit_decision = "accept"
                        commit_reason = accept_reason
                        self.layer_global_reject_streak[resolved_eval_layer] = 0
                        self.anchor = best_snap
                        self.shadow = best_snap
                        self.anchor_z = float(global_z_after)
                        self.restore_snapshot(best_snap)
                        self.work = best_snap
                        self.work_z = float(global_z_after)
                        self._update_coupling_weights(best_score, accepted=True)
                        self._refresh_anchor_reference()
                        if global_z_after < float(self.best.z) - 1e-6:
                            self.best = best_snap
                        improved = bool(global_z_after < global_z_before - 1e-6)
                        self.last_layer_accept = str(resolved_eval_layer)
                        mark = 0
                        self.layer_global_reject_streak[resolved_eval_layer] = 0
                    else:
                        global_accept = "reject"
                        commit_decision = "reject_global"
                        commit_reason = accept_reason
                        rollback_happened = True
                        self.layer_global_reject_streak[resolved_eval_layer] = int(self.layer_global_reject_streak.get(resolved_eval_layer, 0)) + 1
                        self._update_coupling_weights(best_score, accepted=False)
                        self.shadow = self.anchor
                        self.restore_snapshot(self.anchor)
                        self.work = self.anchor
                        self.work_z = float(self.anchor_z)
                        self.last_layer_accept = ""
                        mark += 1
                    iter_metrics = dict(self._collect_layer_metrics())
                    if global_eval_reason.startswith("forced:"):
                        cycle_best_skipped = None
                else:
                    self.shadow = self.anchor
                    self.restore_snapshot(self.anchor)
                    self.work = self.anchor
                    self.work_z = float(self.anchor_z)
                    self.last_layer_accept = ""
                    mark += 1
                    iter_metrics = dict(self._collect_layer_metrics())
            else:
                self.shadow = self.anchor
                self.restore_snapshot(self.anchor)
                self.work = self.anchor
                self.work_z = float(self.anchor_z)
                self.last_layer_accept = ""
                mark += 1
                iter_metrics = dict(self._collect_layer_metrics())

            if layer == "X":
                if commit_decision == "reject_fast_x_gate":
                    self.layer_fast_gate_reject_streak["X"] = int(self.layer_fast_gate_reject_streak.get("X", 0)) + 1
                else:
                    self.layer_fast_gate_reject_streak["X"] = 0
            if commit_decision == "reject_surrogate":
                self.layer_reject_surrogate_streak[layer] = int(self.layer_reject_surrogate_streak.get(layer, 0)) + 1
            else:
                self.layer_reject_surrogate_streak[layer] = 0
            if commit_decision == "accept":
                self.layer_global_reject_streak[resolved_eval_layer] = 0
            elif not global_eval_triggered and resolved_eval_layer not in self.layer_global_reject_streak:
                self.layer_global_reject_streak[resolved_eval_layer] = 0

            if str(layer).upper() == str(getattr(self.cfg, "force_global_after_layer", "U")).upper():
                cycle_global_eval_count = 0
                cycle_best_skipped = None

            self.shadow_depth = 0
            self.shadow_last_layer = ""
            iter_z_real = float(self.anchor_z)
            pred_reward_log = float(
                predicted_reduction
                if proposal_pass_surrogate
                else max(0.0, cand_aug - base_aug) if math.isfinite(cand_aug) and math.isfinite(base_aug) else 0.0
            )
            actual_reward_log = float(actual_reduction if global_eval_triggered else 0.0)
            resolved_reward_layer = str(resolved_eval_layer if global_eval_triggered else layer).upper()
            selected_operator_name = str(runtime.get("selected_operator", ""))
            total_reward = 0.0
            if resolved_reward_layer == "X" and (str(runtime.get("x_destroy_operator", "")) or str(runtime.get("x_repair_operator", ""))):
                destroy_reward = self._record_operator_reward(
                    resolved_reward_layer,
                    str(runtime.get("x_destroy_operator", "")),
                    pred_reward_log,
                    actual_reward_log,
                    int(it),
                    bool(commit_decision == "accept"),
                    status=str(commit_decision),
                )
                repair_reward = self._record_operator_reward(
                    resolved_reward_layer,
                    str(runtime.get("x_repair_operator", "")),
                    pred_reward_log,
                    actual_reward_log,
                    int(it),
                    bool(commit_decision == "accept"),
                    status=str(commit_decision),
                )
                total_reward = float(0.5 * (destroy_reward + repair_reward))
            else:
                total_reward = self._record_operator_reward(
                    resolved_reward_layer,
                    selected_operator_name,
                    pred_reward_log,
                    actual_reward_log,
                    int(it),
                    bool(commit_decision == "accept"),
                    status=str(commit_decision),
                )
            runtime["operator_reward_pred"] = float(pred_reward_log)
            runtime["operator_reward_actual"] = float(actual_reward_log)
            runtime["operator_reward_total"] = float(total_reward)
            if forced_eval_origin_layer and str(forced_eval_origin_layer).upper() != str(layer).upper():
                self._apply_layer_stagnation_update(layer, False)
                self._apply_layer_stagnation_update(str(forced_eval_origin_layer).upper(), bool(commit_decision == "accept"))
            else:
                self._apply_layer_stagnation_update(layer, bool(commit_decision == "accept"))

            self.layer_runtime_sec_by_name[layer] = float(self.layer_runtime_sec_by_name.get(layer, 0.0)) + float(time.perf_counter() - t_iter0)
            self.layer_trial_count_by_name[layer] = float(self.layer_trial_count_by_name.get(layer, 0.0)) + float(runtime.get("trial_count", 0.0))

            row = {
                "iter": int(it),
                "focus": layer,
                "layer": layer,
                "z": float(iter_z_real),
                "z_evaluated": bool(global_eval_triggered),
                "accepted_type": str(commit_decision),
                "improved": bool(improved),
                "skipped": False,
                "lb": None,
                "local_accept": bool(proposal_pass_surrogate),
                "proposal_pass_surrogate": bool(proposal_pass_surrogate),
                "proposal_pass_fast_gate": bool(proposal_pass_fast_gate),
                "predicted_reduction": float(predicted_reduction),
                "actual_reduction": float(actual_reduction),
                "rho": float(rho),
                "commit_decision": str(commit_decision),
                "commit_reason": str(commit_reason),
                "global_eval_triggered": bool(global_eval_triggered),
                "global_eval_reason": global_eval_reason,
                "global_accept": global_accept,
                "rollback_happened": bool(rollback_happened),
                "global_z_before": float(global_z_before),
                "global_z_after": float(global_z_after),
                "committed_z": float(self.anchor_z),
                "best_z": float(self.best.z),
                "local_obj": float(best_score.get("local_obj", float("nan"))),
                "baseline_local_obj": float(best_score.get("baseline_local_obj", float("nan"))),
                "coupling_penalty": float(best_score.get("coupling_penalty", float("nan"))),
                "prox_penalty": float(best_score.get("prox_penalty", float("nan"))),
                "augmented_obj": float(best_score.get("augmented_obj", float("nan"))),
                "baseline_augmented_obj": float(best_score.get("baseline_augmented_obj", float("nan"))),
                "x_fast_penalty": float(best_score.get("x_fast_penalty", float("nan"))),
                "x_fast_delta_station_load_drift": float(best_score.get("x_fast_delta_station_load_drift", float("nan"))),
                "x_fast_delta_arrival_shift": float(best_score.get("x_fast_delta_arrival_shift", float("nan"))),
                "x_fast_delta_route_pressure": float(best_score.get("x_fast_delta_route_pressure", float("nan"))),
                "x_fast_delta_subtask_count": float(best_score.get("x_fast_delta_subtask_count", float("nan"))),
                "x_fast_gate_band": str(best_score.get("x_fast_gate_band", "")),
                "x_destroy_operator": str(runtime.get("x_destroy_operator", best_score.get("x_destroy_operator", ""))),
                "x_repair_operator": str(runtime.get("x_repair_operator", best_score.get("x_repair_operator", ""))),
                "x_destroy_size": float(runtime.get("x_destroy_size", best_score.get("x_destroy_size", float("nan")))),
                "x_subtask_count_before": float(runtime.get("x_subtask_count_before", float("nan"))),
                "x_subtask_count_after": float(runtime.get("x_subtask_count_after", best_score.get("x_subtask_count_after", float("nan")))),
                "x_affinity_penalty": float(runtime.get("x_affinity_penalty", best_score.get("x_affinity_penalty", float("nan")))),
                "x_route_conflict_penalty": float(runtime.get("x_route_conflict_penalty", best_score.get("x_route_conflict_penalty", float("nan")))),
                "x_finish_time_dispersion_penalty": float(runtime.get("x_finish_time_dispersion_penalty", best_score.get("x_finish_time_dispersion_penalty", float("nan")))),
                "x_candidates_generated_count": float(runtime.get("x_candidates_generated_count", 0.0)),
                "x_candidates_pruned_count": float(runtime.get("x_candidates_pruned_count", 0.0)),
                "x_candidate_details": list(runtime.get("x_candidate_details", []) or []),
                "y_local_cmax": float(best_score.get("y_local_cmax", float("nan"))),
                "y_waiting_penalty": float(best_score.get("y_waiting_penalty", float("nan"))),
                "y_load_balance_penalty": float(best_score.get("y_load_balance_penalty", float("nan"))),
                "y_station_preference_penalty": float(best_score.get("y_station_preference_penalty", float("nan"))),
                "y_prox_station_penalty": float(best_score.get("y_prox_station_penalty", float("nan"))),
                "y_prox_rank_penalty": float(best_score.get("y_prox_rank_penalty", float("nan"))),
                "y_fast_objective": float(best_score.get("y_fast_objective", float("nan"))),
                "y_fast_approx_makespan": float(best_score.get("y_fast_approx_makespan", float("nan"))),
                "y_fast_station_cmax": float(best_score.get("y_fast_station_cmax", float("nan"))),
                "y_fast_waiting_penalty": float(best_score.get("y_fast_waiting_penalty", float("nan"))),
                "y_fast_queue_penalty": float(best_score.get("y_fast_queue_penalty", float("nan"))),
                "y_fast_arrival_misalignment_penalty": float(best_score.get("y_fast_arrival_misalignment_penalty", float("nan"))),
                "y_fast_load_balance_penalty": float(best_score.get("y_fast_load_balance_penalty", float("nan"))),
                "y_fast_station_finish_max": float(best_score.get("y_fast_station_finish_max", float("nan"))),
                "y_fast_rank": float(runtime.get("y_fast_rank", best_score.get("y_fast_rank", float("nan")))),
                "y_route_sim_rank": float(runtime.get("y_route_sim_rank", best_score.get("y_route_sim_rank", float("nan")))),
                "y_route_sim_score": float(runtime.get("y_route_sim_score", best_score.get("y_route_sim_score", float("nan")))),
                "y_route_sim_global_makespan_proxy": float(runtime.get("y_route_sim_global_makespan_proxy", best_score.get("y_route_sim_global_makespan_proxy", float("nan")))),
                "y_route_sim_arrival_slack_mean": float(runtime.get("y_route_sim_arrival_slack_mean", best_score.get("y_route_sim_arrival_slack_mean", float("nan")))),
                "y_route_sim_late_task_count": float(runtime.get("y_route_sim_late_task_count", best_score.get("y_route_sim_late_task_count", float("nan")))),
                "y_route_sim_replayed_sp4": bool(runtime.get("y_route_sim_replayed_sp4", best_score.get("y_route_sim_replayed_sp4", False))),
                "y_route_sim_used_incremental_route": bool(runtime.get("y_route_sim_used_incremental_route", best_score.get("y_route_sim_used_incremental_route", False))),
                "y_precheck_score": float(best_score.get("y_precheck_score", float("nan"))),
                "y_precheck_sorting_cost_delta": float(best_score.get("y_precheck_sorting_cost_delta", float("nan"))),
                "y_precheck_station_cmax": float(best_score.get("y_precheck_station_cmax", float("nan"))),
                "y_precheck_arrival_slack_delta": float(best_score.get("y_precheck_arrival_slack_delta", float("nan"))),
                "u_move_type": str(runtime.get("u_move_type", "")),
                "u_changed_task_count": float(runtime.get("u_changed_task_count", 0.0)),
                "u_trip_count_delta": float(runtime.get("u_trip_count_delta", 0.0)),
                "u_replay_feasible": bool(runtime.get("u_replay_feasible", False)),
                "z_move_type": str(runtime.get("z_move_type", "")),
                "z_changed_subtask_count": float(runtime.get("z_changed_subtask_count", 0.0)),
                "z_candidate_task_delta": float(runtime.get("z_candidate_task_delta", 0.0)),
                "z_candidate_stack_delta": float(runtime.get("z_candidate_stack_delta", 0.0)),
                "z_candidate_mode_delta": float(runtime.get("z_candidate_mode_delta", 0.0)),
                "shadow_depth": int(self.shadow_depth),
                "selected_operator": str(runtime.get("selected_operator", "")),
                "selected_operator_rank": float(runtime.get("selected_operator_rank", 0.0)),
                "operator_budget": float(runtime.get("operator_budget", 0.0)),
                "operator_reward_pred": float(pred_reward_log),
                "operator_reward_actual": float(actual_reward_log),
                "operator_reward_total": float(total_reward),
                "shake_strength": float(self.stagnation_stats.get(layer, {}).get("shake_strength", runtime.get("shake_strength", 1.0))),
                "restart_triggered": bool(self.stagnation_stats.get(layer, {}).get("restart_triggered", runtime.get("restart_triggered", 0.0))),
                "forced_eval_origin_layer": str(runtime.get("forced_eval_origin_layer", "")),
                "global_eval_candidate_count": float(runtime.get("global_eval_candidate_count", 0.0)),
                "iter_runtime_sec": float(time.perf_counter() - t_iter0),
                "run_total_time_sec_so_far": float(self._runtime_elapsed_sec()),
                "epsilon": float(self.cfg.epsilon),
                **runtime,
                **self._flatten_lambda_weights(),
                **iter_metrics,
            }
            self.iter_log.append(row)
            self._notify_progress(it, self.cfg.max_iters, layer)

            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            self._write_logs()
        if self.cfg.export_best_solution:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            self.export_best()
        return float(self.best.z)

    # ----------------------------
    # 评价函数与下界
    # ----------------------------
    def evaluate(self) -> float:
        self._simulate_call_count += 1
        return self.sim.calculate()

    def _lb_transport(self) -> float:
        # 运输下界：各任务到站时间的最大值
        max_t = 0.0
        for st in self.problem.subtask_list:
            for t in getattr(st, "execution_tasks", []) or []:
                max_t = max(max_t, float(getattr(t, "arrival_time_at_station", 0.0)))
        return max_t

    def _lb_station_workload(self) -> float:
        # 工作站下界：总工作量/站数 与 单站最大工作量 的 max
        station_num = max(1, len(self.problem.station_list))
        per_station = [0.0 for _ in range(station_num)]

        total = 0.0
        for st in self.problem.subtask_list:
            for t in getattr(st, "execution_tasks", []) or []:
                pick = float(getattr(t, "picking_duration", 0.0))
                extra = float(getattr(t, "station_service_time", 0.0)) if getattr(t, "noise_tote_ids", None) else 0.0
                w = pick + extra
                sid = int(getattr(t, "target_station_id", 0))
                if 0 <= sid < station_num:
                    per_station[sid] += w
                total += w

        return max(total / station_num, max(per_station) if per_station else 0.0)

    def compute_lb(self, focus: str) -> float:
        # focus in {"sp2","sp3","sp4"}：按轮次挑一个下界（或组合）
        lb_t = self._lb_transport()
        lb_s = self._lb_station_workload()
        if focus == "sp4":
            return lb_t
        if focus == "sp2":
            return lb_s
        # sp3 改变选箱会同时影响运输与工作量结构：取 max 更稳
        return max(lb_t, lb_s)

    def _collect_all_tasks(self) -> List[Any]:
        tasks: List[Any] = []
        if self.problem is None:
            return tasks
        for st in getattr(self.problem, "subtask_list", []) or []:
            tasks.extend(getattr(st, "execution_tasks", []) or [])
        return tasks

    def _compute_robot_path_length(self) -> float:
        if self.problem is None:
            return 0.0
        return self._compute_robot_path_length_from_tasks(self._collect_all_tasks())

    def _compute_robot_path_length_from_tasks(self, tasks: List[Any]) -> float:
        if self.problem is None:
            return 0.0
        robots = getattr(self.problem, "robot_list", []) or []
        robot_map = {int(getattr(r, "id", -1)): r for r in robots}
        events_by_robot: Dict[int, List[Tuple[float, int, int]]] = {}
        for task in tasks:
            rid = int(getattr(task, "robot_id", -1))
            if rid < 0:
                continue
            stack_obj = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1)))
            if stack_obj is not None and getattr(stack_obj, "store_point", None) is not None:
                events_by_robot.setdefault(rid, []).append((
                    float(getattr(task, "arrival_time_at_stack", 0.0)),
                    int(stack_obj.store_point.x),
                    int(stack_obj.store_point.y),
                ))
            sid = int(getattr(task, "target_station_id", -1))
            if 0 <= sid < len(getattr(self.problem, "station_list", []) or []):
                pt = self.problem.station_list[sid].point
                events_by_robot.setdefault(rid, []).append((
                    float(getattr(task, "arrival_time_at_station", 0.0)),
                    int(pt.x),
                    int(pt.y),
                ))
        total = 0.0
        for rid, events in events_by_robot.items():
            robot = robot_map.get(rid)
            if robot is None or getattr(robot, "start_point", None) is None:
                continue
            events.sort(key=lambda x: x[0])
            x0 = int(robot.start_point.x)
            y0 = int(robot.start_point.y)
            last_x, last_y = x0, y0
            for _, x, y in events:
                total += abs(x - last_x) + abs(y - last_y)
                last_x, last_y = x, y
            total += abs(last_x - x0) + abs(last_y - y0)
        return float(total)

    def _structural_score(self, metrics: Dict[str, float]) -> float:
        return (
            float(metrics.get("station_load_max_ratio", 0.0)) * 2.0 +
            float(metrics.get("robot_finish_ratio", 0.0)) * 1.5 +
            float(metrics.get("noise_ratio", 0.0)) * 2.0 +
            float(metrics.get("avg_stack_span", 0.0)) * 0.05 +
            float(metrics.get("station_idle_total", 0.0)) * 0.002
        )

    def _collect_layer_metrics(self) -> Dict[str, float]:
        if self.problem is None:
            return {}
        tasks = self._collect_all_tasks()
        subtasks = getattr(self.problem, "subtask_list", []) or []
        stations = getattr(self.problem, "station_list", []) or []
        robots = getattr(self.problem, "robot_list", []) or []
        makespan = float(getattr(self.problem, "global_makespan", 0.0))

        subtask_sizes = [len(getattr(st, "unique_sku_list", []) or []) for st in subtasks]
        station_loads = [0.0 for _ in stations] if stations else [0.0]
        station_busy = []
        station_idle_total = 0.0
        for idx, s in enumerate(stations):
            seq = getattr(s, "processed_tasks", []) or []
            busy = sum(float(getattr(t, "total_process_duration", 0.0)) for t in seq)
            station_busy.append(busy)
            station_idle_total += float(getattr(s, "total_idle_time", 0.0))
            station_loads[idx] = float(len(seq))
        station_util_mean = (sum((min(1.0, b / makespan) for b in station_busy)) / len(station_busy)) if station_busy and makespan > 1e-9 else 0.0
        station_load_max = max(station_loads) if station_loads else 0.0
        station_load_mean = (sum(station_loads) / len(station_loads)) if station_loads else 0.0
        station_load_std = math.sqrt(sum((x - station_load_mean) ** 2 for x in station_loads) / len(station_loads)) if station_loads else 0.0

        hit_stack_ids = set()
        noise_total = 0
        target_total = 0
        stack_spans = []
        latest_robot_finish = 0.0
        arrival_slacks = []
        robot_busy: Dict[int, float] = {int(getattr(r, "id", idx)): 0.0 for idx, r in enumerate(robots)}
        for t in tasks:
            if len(getattr(t, "hit_tote_ids", []) or []) > 0:
                hit_stack_ids.add(int(getattr(t, "target_stack_id", -1)))
            noise_total += len(getattr(t, "noise_tote_ids", []) or [])
            target_total += len(getattr(t, "target_tote_ids", []) or [])
            if getattr(t, "sort_layer_range", None) is not None:
                lo, hi = getattr(t, "sort_layer_range", (0, 0))
                stack_spans.append(float(hi - lo + 1))
            rid = int(getattr(t, "robot_id", -1))
            if rid in robot_busy:
                robot_busy[rid] += float(getattr(t, "robot_service_time", 0.0))
            latest_robot_finish = max(
                latest_robot_finish,
                float(getattr(t, "arrival_time_at_station", 0.0)),
                float(getattr(t, "arrival_time_at_stack", 0.0)),
            )
            arrival_slacks.append(float(getattr(t, "start_process_time", 0.0)) - float(getattr(t, "arrival_time_at_station", 0.0)))

        robot_util_mean = (sum((min(1.0, v / makespan) for v in robot_busy.values())) / len(robot_busy)) if robot_busy and makespan > 1e-9 else 0.0
        robot_finish_ratio = (latest_robot_finish / makespan) if makespan > 1e-9 else 0.0
        station_load_max_ratio = (station_load_max / station_load_mean) if station_load_mean > 1e-9 else 0.0

        return {
            "subtask_count": float(len(subtasks)),
            "avg_sku_per_subtask": float(sum(subtask_sizes) / len(subtask_sizes)) if subtask_sizes else 0.0,
            "max_sku_per_subtask": float(max(subtask_sizes)) if subtask_sizes else 0.0,
            "station_idle_total": float(station_idle_total),
            "station_utilization_mean": float(station_util_mean),
            "station_load_max": float(station_load_max),
            "station_load_std": float(station_load_std),
            "station_load_max_ratio": float(station_load_max_ratio),
            "hit_stack_count": float(len([x for x in hit_stack_ids if x >= 0])),
            "noise_ratio": float(noise_total / target_total) if target_total > 0 else 0.0,
            "avg_stack_span": float(sum(stack_spans) / len(stack_spans)) if stack_spans else 0.0,
            "sorting_cost_proxy": float(sum((self.last_sp3_sorting_costs or {}).values())),
            "robot_path_length_total": float(self._compute_robot_path_length()),
            "robot_utilization_mean": float(robot_util_mean),
            "latest_robot_finish": float(latest_robot_finish),
            "robot_finish_ratio": float(robot_finish_ratio),
            "arrival_slack_mean": float(sum(arrival_slacks) / len(arrival_slacks)) if arrival_slacks else 0.0,
            "global_makespan": float(makespan),
            "UB": float(self.best.z) if self.best is not None else float(makespan),
        }

    def _lb_robot_workload(self) -> float:
        tasks = self._collect_all_tasks()
        robot_num = max(1, len(getattr(self.problem, "robot_list", []) or []))
        total = sum(float(getattr(t, "robot_service_time", 0.0)) for t in tasks)
        per_robot: Dict[int, float] = {}
        for t in tasks:
            rid = int(getattr(t, "robot_id", -1))
            if rid >= 0:
                per_robot[rid] = per_robot.get(rid, 0.0) + float(getattr(t, "robot_service_time", 0.0))
        return max(total / robot_num, max(per_robot.values()) if per_robot else 0.0)

    def _lb_order_chain(self) -> float:
        order_to_work: Dict[int, float] = {}
        for st in getattr(self.problem, "subtask_list", []) or []:
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            subtotal = 0.0
            for t in getattr(st, "execution_tasks", []) or []:
                subtotal += float(getattr(t, "robot_service_time", 0.0))
                subtotal += float(getattr(t, "picking_duration", 0.0))
                subtotal += float(getattr(t, "station_service_time", 0.0))
            if subtotal <= 1e-9:
                subtotal = float(len(getattr(st, "sku_list", []) or []))
            order_to_work[oid] = order_to_work.get(oid, 0.0) + subtotal
        return max(order_to_work.values()) if order_to_work else 0.0

    def _lb_stack_blocking(self) -> float:
        vals = []
        for t in self._collect_all_tasks():
            span = 0.0
            if getattr(t, "sort_layer_range", None) is not None:
                lo, hi = getattr(t, "sort_layer_range", (0, 0))
                span = float(max(0, hi - lo + 1))
            vals.append(span + float(len(getattr(t, "noise_tote_ids", []) or [])))
        return max(vals) if vals else 0.0

    def _compute_lb_bundle(self, mode: str) -> Dict[str, float]:
        lb_move = float(self._lb_transport())
        lb_sta = float(self._lb_station_workload())
        lb_rob = float(self._lb_robot_workload())
        lb_ord = float(self._lb_order_chain())
        lb_stack = float(self._lb_stack_blocking())
        lb0 = max(lb_move, lb_sta, lb_rob, lb_ord, lb_stack)
        mode_to_lb = {
            "M_R": max(lb_move, 0.85 * lb_sta, lb_ord),
            "M_Y": max(lb_sta, lb_move, lb_ord),
            "M_B": max(lb_stack, lb_sta, lb_rob),
            "M_X": max(lb_ord, lb_sta, lb_stack),
            "M_XYB": max(lb0, 0.5 * (lb_sta + lb_stack)),
        }
        lb_mode = float(mode_to_lb.get(mode, lb0))
        ub = float(self.best.z) if self.best is not None else float(getattr(self.problem, "global_makespan", 0.0))
        gap = ((ub - lb_mode) / ub) if ub > 1e-9 else 0.0
        return {
            "LB_0": float(lb0),
            "LB_mode": float(lb_mode),
            "gap_mode": float(gap),
            "LB_move_0": float(lb_move),
            "LB_sta_0": float(lb_sta),
            "LB_rob_0": float(lb_rob),
            "LB_ord_0": float(lb_ord),
            "LB_stack_0": float(lb_stack),
        }

    # ----------------------------
    # 快照（保存最优解用于导出/复现）
    # ----------------------------
    def snapshot(self, z: float, iter_id: int, lightweight: bool = False) -> SolutionSnapshot:
        t0 = time.perf_counter()
        subtask_station_rank = {}
        for st in self.problem.subtask_list:
            subtask_station_rank[int(st.id)] = (int(st.assigned_station_id), int(st.station_sequence_rank))
        snap = SolutionSnapshot(
            z=float(z),
            iter_id=int(iter_id),
            seed=int(self.cfg.seed),
            subtask_station_rank=subtask_station_rank,
            sp1_capacity_limits=dict(getattr(self.sp1, "order_capacity_limits", {}) or {}),
            sp1_incompatibility_pairs=sorted(list(getattr(self.sp1, "incompatibility_pairs", set()) or set())),
            subtask_state=copy.deepcopy(getattr(self.problem, "subtask_list", []) or []) if lightweight else None,
            problem_state=None if lightweight else copy.deepcopy(self.problem),
            last_sp4_arrival_times=dict(self.last_sp4_arrival_times or {}),
            last_sp3_tote_selection={int(k): list(v) for k, v in (self.last_sp3_tote_selection or {}).items()},
            last_sp3_sorting_costs={int(k): float(v) for k, v in (self.last_sp3_sorting_costs or {}).items()},
            last_station_start_times=dict(self.last_station_start_times or {}),
            last_beta_value=None if self.last_beta_value is None else float(self.last_beta_value),
        )
        self.snapshot_time_sec += float(time.perf_counter() - t0)
        if lightweight:
            self.lightweight_snapshot_count += 1
        else:
            self.heavy_snapshot_count += 1
        return snap

    def restore_snapshot(self, snap: SolutionSnapshot):
        t0 = time.perf_counter()
        if snap.problem_state is not None:
            self.problem = copy.deepcopy(snap.problem_state)
            self._rebuild_solvers()
            if self.sp1:
                self.sp1.order_capacity_limits = dict(snap.sp1_capacity_limits or {})
                self.sp1.incompatibility_pairs = set(tuple(x) for x in (snap.sp1_incompatibility_pairs or []))
            self.last_sp4_arrival_times = dict(snap.last_sp4_arrival_times or {})
            self.last_sp3_tote_selection = {
                int(k): list(v) for k, v in (snap.last_sp3_tote_selection or {}).items()
            }
            self.last_sp3_sorting_costs = {
                int(k): float(v) for k, v in (snap.last_sp3_sorting_costs or {}).items()
            }
            self.last_sp2_local_result = None
            self.last_station_start_times = dict(snap.last_station_start_times or {})
            self.last_beta_value = None if snap.last_beta_value is None else float(snap.last_beta_value)
            self.restore_time_sec += float(time.perf_counter() - t0)
            return

        if snap.subtask_state is not None:
            self.problem.subtask_list = copy.deepcopy(snap.subtask_state)
            self.problem.subtask_num = len(getattr(self.problem, "subtask_list", []) or [])
            self._rebuild_problem_task_list()
            self._sync_task_assignments_from_subtasks()
            if self.sp1:
                self.sp1.order_capacity_limits = dict(snap.sp1_capacity_limits or {})
                self.sp1.incompatibility_pairs = set(tuple(x) for x in (snap.sp1_incompatibility_pairs or []))
            self.last_sp4_arrival_times = dict(snap.last_sp4_arrival_times or {})
            self.last_sp3_tote_selection = {
                int(k): list(v) for k, v in (snap.last_sp3_tote_selection or {}).items()
            }
            self.last_sp3_sorting_costs = {
                int(k): float(v) for k, v in (snap.last_sp3_sorting_costs or {}).items()
            }
            self.last_sp2_local_result = None
            self.last_station_start_times = dict(snap.last_station_start_times or {})
            self.last_beta_value = None if snap.last_beta_value is None else float(snap.last_beta_value)
            self.restore_time_sec += float(time.perf_counter() - t0)
            return

        # 恢复 SP1 容量反馈（影响后续 SP1 重拆分时的上限）
        if self.sp1 and snap.sp1_capacity_limits:
            self.sp1.order_capacity_limits = dict(snap.sp1_capacity_limits)
        if self.sp1:
            self.sp1.incompatibility_pairs = set(tuple(x) for x in (snap.sp1_incompatibility_pairs or []))
            self._run_sp1()

        # 恢复 SP2 的站点与顺位，再重建 SP3→SP4→仿真
        st_map = {st.id: st for st in self.problem.subtask_list}
        for st_id, (sid, rank) in snap.subtask_station_rank.items():
            if st_id in st_map:
                st_map[st_id].assigned_station_id = sid
                st_map[st_id].station_sequence_rank = rank
        self.restore_time_sec += float(time.perf_counter() - t0)

    # ----------------------------
    # 各阶段求解与回填
    # ----------------------------
    def _run_sp1(self):
        sub_tasks = self.sp1.solve(use_mip=False)
        self.problem.subtask_list = sub_tasks
        self.problem.subtask_num = len(sub_tasks)

    def _run_sp2_initial(self):
        self.sp2.solve_initial_heuristic()

    def _run_sp2_mip(self):
        shadow = self._compute_shadow_prices() if self.cfg.enable_soft_pi else None
        try:
            self.sp2.solve_mip_with_feedback(
                tasks=self.problem.subtask_list,
                sp4_robot_arrival_times=self.last_sp4_arrival_times,
                sp3_tote_selection=self.last_sp3_tote_selection,
                sp3_sorting_costs=self.last_sp3_sorting_costs,
                shadow_assignment_penalty=shadow,
                shadow_weight=float(self.cfg.sp2_shadow_weight),
                time_limit_sec=self.cfg.sp2_time_limit_sec,
            )
        except Exception as exc:
            print(f"  >>> [TRA] SP2 MIP failed, fallback to heuristic: {exc}")
            self._run_sp2_initial()

    def _run_sp3(self):
        if self.cfg.sp3_use_mip:
            physical_tasks, tote_selection, sorting_costs = self.sp3.solve(
                self.problem.subtask_list,
                beta_congestion=float(self.last_beta_value or 1.0),
                sp4_routing_costs=None,
            )
        else:
            heuristic = self.sp3.SP3_Heuristic_Solver(self.problem)
            physical_tasks, tote_selection, sorting_costs = heuristic.solve(
                self.problem.subtask_list,
                beta_congestion=float(self.last_beta_value or 1.0),
            )

        self.problem.task_list = physical_tasks
        self.problem.task_num = len(physical_tasks)
        self.last_sp3_tote_selection = {int(k): list(v) for k, v in tote_selection.items()}
        self.last_sp3_sorting_costs = {int(k): float(v) for k, v in sorting_costs.items()}

    def _run_sp4(self):
        # soft time windows
        soft_windows = None
        mu = 0.0
        if self.cfg.enable_soft_mu and self.last_station_start_times:
            soft_windows = dict(self.last_station_start_times)
            mu = float(self.cfg.mu_value)
        arrival_times, robot_assign = self.sp4.solve(
            self.problem.subtask_list,
            use_mip=self.cfg.sp4_use_mip,
            lkh_time_limit_seconds=self.cfg.sp4_lkh_time_limit_seconds if not self.cfg.sp4_use_mip else None,
            soft_time_windows=soft_windows,
            mu=mu,
        )
        self.last_sp4_arrival_times = {int(k): float(v) for k, v in (arrival_times or {}).items()}

        # 回填 subtask 的 assigned_robot_id（供日志/仿真输出使用）
        st_map = {st.id: st for st in self.problem.subtask_list}
        for st_id, robot_id in (robot_assign or {}).items():
            if st_id in st_map:
                st_map[st_id].assigned_robot_id = int(robot_id)

    # ----------------------------
    # 主入口
    # ----------------------------
    def initialize(self):
        self._reset_runtime_caches()
        self.run_start_time_sec = float(time.perf_counter())
        self.run_total_time_sec = 0.0
        self.layer_runtime_sec_by_name = {name: 0.0 for name in self.layer_names}
        self.layer_trial_count_by_name = {name: 0.0 for name in self.layer_names}
        self.global_eval_count = 0
        self._set_seed(self.cfg.seed)
        self.problem = CreateOFSProblem.generate_problem_by_scale(self.cfg.scale, seed=self.cfg.seed)
        self._rebuild_solvers()

        # SP3 coverage precheck (deepcopy; non-intrusive)
        if self.cfg.enable_sp3_precheck:
            try:
                self.precheck_result = self._precheck_sp3_coverage()
                unmet = int(self.precheck_result.get("unmet_sku_total", 0)) if self.precheck_result else 0
                if unmet > 0 and str(self.cfg.sp3_precheck_fail_action).lower() == "abort":
                    self.precheck_aborted = True
                    self.precheck_status = f"precheck_unmet:{unmet}"
                    return
            except Exception as e:
                try:
                    path = self._log_path("sp3_precheck_error.txt")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(e))
                except Exception:
                    pass

        self._run_sp1()
        self._run_sp2_initial()
        self._run_sp3()
        self._run_sp4()
        z0 = self.evaluate()
        # harvest soft-coupling caches
        self._harvest_station_start_times()
        self._update_beta_from_station()

        use_light_snapshot = str(getattr(self.cfg, "search_scheme", "legacy")).lower() == "layer_augmented"
        self.best = self.snapshot(z0, iter_id=0, lightweight=use_light_snapshot)
        self.work = self.snapshot(z0, iter_id=0, lightweight=use_light_snapshot)
        self.work_z = float(z0)
        self._init_layer_augmented_state(z0)
        self._refresh_runtime_cache(z0)
        self._append_iter_log(0, focus="init", z=z0, improved=True, skipped=False, lb=None)

    def _append_iter_log(self, iter_id: int, focus: str, z: float, improved: bool, skipped: bool,
                         lb: Optional[float]):
        self.iter_log.append({
            "iter": int(iter_id),
            "focus": focus,
            "z": float(z),
            "best_z": float(self.best.z if self.best else z),
            "improved": bool(improved),
            "skipped": bool(skipped),
            "lb": None if lb is None else float(lb),
            "epsilon": float(self.cfg.epsilon),
        })

    def _notify_progress(self, iter_id: int, total_iters: int, focus: str):
        cb = getattr(self.cfg, "progress_callback", None)
        if cb is None:
            return
        try:
            cb({
                "iter": int(iter_id),
                "total_iters": int(total_iters),
                "focus": str(focus),
                "best_z": float(self.best.z) if self.best is not None else float("nan"),
                "scale": str(getattr(self.cfg, "scale", "")),
            })
        except Exception:
            pass

    def _select_mode(self) -> Tuple[str, Dict[str, Dict[str, float]]]:
        mode_eval: Dict[str, Dict[str, float]] = {}
        best_mode = self.mode_names[0]
        best_score = -1e18
        current_metrics = self._collect_layer_metrics()
        min_calls = min(float(self.mode_stats[m]["calls"]) for m in self.mode_names) if self.mode_names else 0.0
        for mode in self.mode_names:
            bundle = self._compute_lb_bundle(mode)
            stats = self.mode_stats[mode]
            success_rate = float(stats["success"]) / max(1.0, float(stats["calls"]))
            fail_pen = min(1.0, float(stats["fail"]) / max(1.0, float(self.cfg.mode_fail_limit)))
            explore_bonus = float(self.cfg.mode_explore_bonus) if float(stats["calls"]) <= min_calls + 1e-9 else 0.0
            rotation_bonus = float(self.cfg.mode_rotation_bonus) if mode != self.last_selected_mode else 0.0
            diversity = 0.0
            if mode == "M_R":
                diversity = (
                    current_metrics.get("robot_finish_ratio", 0.0)
                    + 0.5 * current_metrics.get("arrival_slack_mean", 0.0) / max(1.0, current_metrics.get("global_makespan", 1.0))
                )
            elif mode == "M_Y":
                diversity = (
                    current_metrics.get("station_load_max_ratio", 0.0)
                    + 0.5 * current_metrics.get("station_idle_total", 0.0) / max(1.0, current_metrics.get("global_makespan", 1.0))
                )
            elif mode == "M_B":
                diversity = current_metrics.get("noise_ratio", 0.0) + 0.05 * current_metrics.get("avg_stack_span", 0.0)
            elif mode == "M_X":
                diversity = current_metrics.get("max_sku_per_subtask", 0.0) / max(1.0, current_metrics.get("avg_sku_per_subtask", 1.0))
            else:
                diversity = current_metrics.get("station_load_max_ratio", 0.0) + current_metrics.get("noise_ratio", 0.0)
            score = (
                0.30 * bundle["gap_mode"]
                + 0.15 * success_rate
                + 0.20 * diversity
                - 0.15 * fail_pen
                + explore_bonus
                + rotation_bonus
            )
            bundle["score"] = float(score)
            bundle["explore_bonus"] = float(explore_bonus)
            bundle["rotation_bonus"] = float(rotation_bonus)
            mode_eval[mode] = bundle
            if score > best_score:
                best_score = score
                best_mode = mode
        self.last_selected_mode = best_mode
        return best_mode, mode_eval

    def _run_downstream_for_mode(self, mode: str, change_info: Optional[Dict[str, Any]] = None) -> int:
        before_sig = self._capture_all_signatures()
        if change_info is None:
            self._apply_mode_perturbation(mode, random.Random(int(self.cfg.seed)))
            after_sig = self._capture_all_signatures()
            change_info = self._derive_dirty_from_changes(mode, before_sig, after_sig)
        return self._sync_tasks_and_maybe_rebuild(change_info.get("dirty", {}), change_info)

    def _perturb_station_assignments(self, rng: random.Random, max_count: int = 3):
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "assigned_station_id", -1) >= 0]
        if not subtasks or not getattr(self.problem, "station_list", None):
            return
        station_ids = [int(getattr(s, "id", idx)) for idx, s in enumerate(self.problem.station_list)]
        count = min(max_count, len(subtasks))
        for st in rng.sample(subtasks, count):
            alt = [sid for sid in station_ids if sid != int(getattr(st, "assigned_station_id", -1))]
            if alt:
                st.assigned_station_id = rng.choice(alt)
            st.station_sequence_rank = max(0, int(getattr(st, "station_sequence_rank", 0)) + rng.choice([-1, 0, 1]))

    def _perturb_routing_anchor(self, rng: random.Random, max_count: int = 3):
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "assigned_station_id", -1) >= 0]
        if not subtasks:
            return
        for st in rng.sample(subtasks, min(max_count, len(subtasks))):
            st.station_sequence_rank = max(0, int(getattr(st, "station_sequence_rank", 0)) + rng.choice([-1, 1]))

    def _perturb_stack_behavior(self, rng: random.Random):
        base = float(self.last_beta_value or 1.0)
        self.last_beta_value = max(0.5, min(3.0, base * rng.choice([0.85, 1.15, 1.30])))

    def _perturb_split_structure(self, rng: random.Random, operator: str = "x_cap_shrink_expand", strength: int = 1):
        if self.sp1 is None or self.problem is None:
            return
        orders = getattr(self.problem, "order_list", []) or []
        if not orders:
            return
        max_cap = int(getattr(self.problem, "robot_num", 1)) + 6
        chosen_orders = list(rng.sample(orders, min(max(1, int(strength)), len(orders))))
        if operator == "x_incompatibility_flip":
            for order in chosen_orders:
                sku_ids = sorted({int(s.id) for s in getattr(order, "unique_sku_list", []) or []})
                if len(sku_ids) >= 2:
                    a, b = rng.sample(sku_ids, 2)
                    self.sp1.add_incompatibility(a, b)
            return
        for order in chosen_orders:
            oid = int(getattr(order, "order_id", -1))
            curr = int(self.sp1.order_capacity_limits.get(oid, 1))
            if operator == "x_order_targeted_resplit":
                delta = rng.choice([-2, -1, 1, 2])
            else:
                delta = rng.choice([-1, 1])
            new_cap = max(1, min(curr + delta, max_cap))
            self.sp1.order_capacity_limits[oid] = new_cap
            if operator == "x_order_targeted_resplit":
                sku_ids = sorted({int(s.id) for s in getattr(order, "unique_sku_list", []) or []})
                if len(sku_ids) >= 2 and rng.random() < 0.7:
                    a, b = rng.sample(sku_ids, 2)
                    self.sp1.add_incompatibility(a, b)

    def _y_priority_subtasks(self, limit: int = 3) -> List[Any]:
        rows: List[Tuple[float, Any]] = []
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            arrival = float(self._estimate_subtask_arrival(st))
            start = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            wait = max(0.0, start - arrival)
            anchor_slack = float(self.anchor_reference.get("subtask_slack", {}).get(sid, self._estimate_subtask_slack(st)))
            mismatch = max(0.0, start - float(self.anchor_reference.get("subtask_start", {}).get(sid, self._estimate_subtask_start(st))))
            rows.append((wait + max(0.0, wait - anchor_slack) + mismatch, st))
        rows.sort(key=lambda item: (-item[0], int(getattr(item[1], "id", -1))))
        return [st for _, st in rows[:max(1, int(limit))]]

    def _y_search_mode(self) -> str:
        stagnation = float(self.stagnation_stats.get("Y", {}).get("no_improve_rounds", 0.0))
        if stagnation >= float(getattr(self.cfg, "y_diversify_stagnation_rounds", 2)):
            return "diversify"
        return "intensify"

    def _current_y_destroy_fraction(self) -> float:
        stagnation = float(self.stagnation_stats.get("Y", {}).get("no_improve_rounds", 0.0))
        if stagnation >= 4.0:
            return float(getattr(self.cfg, "y_destroy_fraction_max", 0.40))
        if stagnation >= 2.0:
            return float(getattr(self.cfg, "y_destroy_fraction_mid", 0.25))
        return float(getattr(self.cfg, "y_destroy_fraction_min", 0.15))

    def _filter_y_operator_sequence(self, operator_sequence: List[str], search_mode: str, budget: int) -> List[str]:
        local_ops = [
            "station_reassign_single",
            "station_swap_pair",
            "rank_reinsert_within_station",
            "cross_station_reinsert",
            "station_block_relocate",
            "order_block_reinsert",
        ]
        destroy_ops = [
            "congested_station_destroy_repair",
            "order_cohesion_destroy_repair",
        ]
        if search_mode == "diversify":
            return list(operator_sequence[:budget])
        filtered: List[str] = []
        destroy_used = 0
        for op in operator_sequence:
            if op in local_ops:
                filtered.append(op)
            elif op in destroy_ops and destroy_used < 1:
                filtered.append(op)
                destroy_used += 1
            if len(filtered) >= int(budget):
                break
        if not filtered:
            filtered = list(local_ops[:max(1, int(budget))])
        return filtered

    def _enforce_y_trust_region(self):
        anchor_station = {}
        anchor_rank = {}
        for st in self._iter_snapshot_subtasks(self.anchor):
            sid = int(getattr(st, "id", -1))
            anchor_station[sid] = int(getattr(st, "assigned_station_id", -1))
            anchor_rank[sid] = int(getattr(st, "station_sequence_rank", -1))
        station_limit = max(1, int(getattr(self.cfg, "y_intensify_station_change_limit", 2)))
        rank_window = max(0, int(getattr(self.cfg, "y_intensify_rank_window", 1)))
        changed_station_rows = []
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            anc_station = int(anchor_station.get(sid, int(getattr(st, "assigned_station_id", -1))))
            anc_rank = int(anchor_rank.get(sid, int(getattr(st, "station_sequence_rank", -1))))
            if int(getattr(st, "assigned_station_id", -1)) != anc_station:
                changed_station_rows.append(st)
            curr_rank = int(getattr(st, "station_sequence_rank", -1))
            st.station_sequence_rank = max(0, min(curr_rank, anc_rank + rank_window))
            st.station_sequence_rank = max(st.station_sequence_rank, max(0, anc_rank - rank_window))
        if len(changed_station_rows) > station_limit:
            for st in changed_station_rows[station_limit:]:
                sid = int(getattr(st, "id", -1))
                st.assigned_station_id = int(anchor_station.get(sid, int(getattr(st, "assigned_station_id", -1))))
                st.station_sequence_rank = int(anchor_rank.get(sid, int(getattr(st, "station_sequence_rank", -1))))
        self._normalize_station_assignments()

    def _apply_y_operator(self, operator: str, rng: random.Random, strength: int, context: SP2LayerContext, search_mode: str = "intensify") -> bool:
        subtasks = list(getattr(self.problem, "subtask_list", []) or [])
        station_ids = [int(getattr(st, "id", idx)) for idx, st in enumerate(getattr(self.problem, "station_list", []) or [])]
        if not subtasks or not station_ids:
            return False
        self._normalize_station_assignments()
        changed = False
        target_rows = self._y_priority_subtasks(limit=max(2, strength + 1))
        block_size = max(2, int(getattr(self.cfg, "y_block_move_size", 3)))
        destroy_fraction = self._current_y_destroy_fraction()
        order_groups: Dict[int, List[Any]] = defaultdict(list)
        station_groups: Dict[int, List[Any]] = defaultdict(list)
        for st in subtasks:
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            order_groups[oid].append(st)
            station_groups[int(getattr(st, "assigned_station_id", -1))].append(st)
        if operator == "station_reassign_single":
            st = rng.choice(target_rows or subtasks)
            curr_sid = int(getattr(st, "assigned_station_id", -1))
            candidate_station_ids = [sid for sid in station_ids if sid != curr_sid]
            if not candidate_station_ids:
                return False
            best_sid = min(candidate_station_ids, key=lambda sid: float(context.order_station_penalty.get((int(getattr(getattr(st, "parent_order", None), "order_id", -1)), sid), 0.0)))
            st.assigned_station_id = int(best_sid)
            changed = True
        elif operator == "station_swap_pair":
            rows = target_rows or subtasks
            if len(rows) < 2:
                return False
            a, b = rng.sample(rows, 2)
            if int(getattr(a, "assigned_station_id", -1)) == int(getattr(b, "assigned_station_id", -1)):
                return False
            a.assigned_station_id, b.assigned_station_id = int(getattr(b, "assigned_station_id", -1)), int(getattr(a, "assigned_station_id", -1))
            changed = True
        elif operator == "rank_reinsert_within_station":
            crowded = [sid for sid, rows in station_groups.items() if len(rows) >= 2]
            if not crowded:
                return False
            sid = rng.choice(crowded)
            rows = sorted(station_groups[sid], key=lambda st: (int(getattr(st, "station_sequence_rank", -1)), int(getattr(st, "id", -1))))
            chosen = rng.choice(rows)
            rows.remove(chosen)
            insert_pos = rng.randrange(len(rows) + 1)
            rows.insert(insert_pos, chosen)
            for rank, st in enumerate(rows):
                st.station_sequence_rank = int(rank)
            changed = True
        elif operator == "cross_station_reinsert":
            st = rng.choice(target_rows or subtasks)
            curr_sid = int(getattr(st, "assigned_station_id", -1))
            alt = [sid for sid in station_ids if sid != curr_sid]
            if not alt:
                return False
            new_sid = rng.choice(alt)
            st.assigned_station_id = int(new_sid)
            st.station_sequence_rank = int(len(station_groups.get(new_sid, [])))
            changed = True
        elif operator == "station_block_relocate":
            source_sid = max(station_groups.keys(), key=lambda sid: len(station_groups.get(sid, []))) if station_groups else -1
            rows = list(sorted(station_groups.get(source_sid, []), key=lambda st: (int(getattr(st, "station_sequence_rank", -1)), int(getattr(st, "id", -1)))))
            if len(rows) < 2:
                return False
            move_n = max(2, min(len(rows), int(math.ceil(float(block_size) * max(1.0, destroy_fraction / 0.15)))))
            moving = rows[-move_n:]
            load_now = {sid: len(station_groups.get(sid, [])) for sid in station_ids}
            for st in moving:
                oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
                target_sid = min(
                    [sid for sid in station_ids if sid != source_sid] or station_ids,
                    key=lambda sid: (
                        float(context.order_station_penalty.get((oid, sid), 0.0)),
                        load_now.get(sid, 0),
                        sid,
                    ),
                )
                st.assigned_station_id = int(target_sid)
                st.station_sequence_rank = int(load_now.get(target_sid, 0))
                load_now[target_sid] = int(load_now.get(target_sid, 0)) + 1
            _ = self.sp2.solve_local_layer(subtasks, context, use_mip=False, time_limit_sec=float(self.cfg.sp2_time_limit_sec))
            changed = True
        elif operator == "order_block_reinsert":
            candidate_orders = [oid for oid, rows in order_groups.items() if oid >= 0 and len(rows) >= 2]
            if not candidate_orders:
                return False
            oid = rng.choice(candidate_orders)
            rows = list(sorted(order_groups[oid], key=lambda st: (int(getattr(st, "station_sequence_rank", -1)), int(getattr(st, "id", -1)))))
            move_n = max(2, min(len(rows), int(math.ceil(float(block_size) * max(1.0, destroy_fraction / 0.15)))))
            chosen_rows = rows[:move_n]
            target_sid = min(
                station_ids,
                key=lambda sid: (
                    float(context.order_station_penalty.get((oid, sid), 0.0)),
                    len([x for x in subtasks if int(getattr(x, "assigned_station_id", -1)) == sid]),
                    sid,
                ),
            )
            base_rank = len([x for x in subtasks if int(getattr(x, "assigned_station_id", -1)) == target_sid])
            for offset, st in enumerate(chosen_rows):
                st.assigned_station_id = int(target_sid)
                st.station_sequence_rank = int(base_rank + offset)
            _ = self.sp2.solve_local_layer(subtasks, context, use_mip=False, time_limit_sec=float(self.cfg.sp2_time_limit_sec))
            changed = True
        elif operator == "congested_station_destroy_repair":
            if search_mode != "diversify" and float(self.stagnation_stats.get("Y", {}).get("restart_triggered", 0.0)) <= 0.0:
                return False
            heavy_sid = max(station_groups.keys(), key=lambda sid: len(station_groups.get(sid, []))) if station_groups else -1
            rows = list(station_groups.get(heavy_sid, []))
            if not rows:
                return False
            destroy_n = max(1, min(len(rows), int(math.ceil(len(subtasks) * destroy_fraction))))
            removed = rows[-destroy_n:]
            for st in removed:
                target_sid = min(station_ids, key=lambda sid: (len([x for x in subtasks if int(getattr(x, "assigned_station_id", -1)) == sid]), sid))
                st.assigned_station_id = int(target_sid)
                st.station_sequence_rank = int(len([x for x in subtasks if int(getattr(x, "assigned_station_id", -1)) == target_sid]))
            _ = self.sp2.solve_local_layer(subtasks, context, use_mip=False, time_limit_sec=float(self.cfg.sp2_time_limit_sec))
            changed = True
        elif operator == "order_cohesion_destroy_repair":
            if search_mode != "diversify" and float(self.stagnation_stats.get("Y", {}).get("restart_triggered", 0.0)) <= 0.0:
                return False
            bad_orders = [oid for oid, rows in order_groups.items() if oid >= 0 and len({int(getattr(st, "assigned_station_id", -1)) for st in rows}) >= 2]
            if not bad_orders:
                return False
            oid = rng.choice(bad_orders)
            rows = list(order_groups[oid])
            target_sid = min(
                station_ids,
                key=lambda sid: (
                    float(context.order_station_penalty.get((oid, sid), 0.0)),
                    len([x for x in subtasks if int(getattr(x, "assigned_station_id", -1)) == sid]),
                    sid,
                ),
            )
            for st in rows:
                st.assigned_station_id = int(target_sid)
            _ = self.sp2.solve_local_layer(subtasks, context, use_mip=False, time_limit_sec=float(self.cfg.sp2_time_limit_sec))
            changed = True
        if changed:
            if search_mode == "intensify":
                self._enforce_y_trust_region()
            self._normalize_station_assignments()
        return bool(changed)

    def _apply_z_operator(self, operator: str, rng: random.Random, strength: int, priority_subtask_ids: Optional[List[int]]) -> Dict[str, Any]:
        if operator != "z_hotspot_destroy_repair":
            return self._build_z_candidate_from_subtasks(rng, priority_subtask_ids=priority_subtask_ids)
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "execution_tasks", None)]
        if not subtasks:
            return {"feasible": False, "move_type": operator, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        batch_size = max(1, int(getattr(self.cfg, "z_hotspot_batch_size", 3)))
        destroy_fraction = max(0.05, float(getattr(self.cfg, "z_destroy_fraction", 0.30)))
        selected_ids = self._select_priority_z_subtasks(limit=max(batch_size, strength))
        selected_ids = list(selected_ids[:batch_size])
        changed = 0
        agg = {"task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        for sid in selected_ids:
            self._normalize_station_assignments()
            st = next((item for item in subtasks if int(getattr(item, "id", -1)) == int(sid)), None)
            if st is not None:
                tasks = list(getattr(st, "execution_tasks", []) or [])
                if tasks:
                    target_count = max(1, int(math.ceil(len(tasks) * destroy_fraction)))
                    noisy_tasks = sorted(
                        tasks,
                        key=lambda task: (
                            -float(len(getattr(task, "noise_tote_ids", []) or [])),
                            -float(self._estimate_task_sorting_cost(task)),
                            int(getattr(task, "task_id", -1)),
                        ),
                    )
                    for task in noisy_tasks[:target_count]:
                        hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or [])]
                        if hit_ids:
                            task.operation_mode = "FLIP"
                            task.target_tote_ids = list(hit_ids)
                            task.sort_layer_range = None
                            task.station_service_time = 0.0
                            task.noise_tote_ids = []
            candidate = self._build_z_candidate_from_subtasks(rng, priority_subtask_ids=[sid])
            if not bool(candidate.get("feasible", False)):
                continue
            changed += int(candidate.get("changed_subtask_count", 0))
            for key in agg:
                agg[key] += int(candidate.get(key, 0))
        if changed <= 0:
            return {"feasible": False, "move_type": operator, "changed_subtask_count": 0, **agg}
        return {"feasible": True, "move_type": operator, "changed_subtask_count": changed, **agg}

    def _apply_u_operator(self, operator: str, rng: random.Random, strength: int, priority_robot_ids: Optional[List[int]]) -> Dict[str, Any]:
        if operator == "u_cross_robot_relocate":
            mapped = "cross_robot_relocate"
        elif operator == "u_trip_swap":
            mapped = "cross_robot_swap"
        elif operator == "u_trip_split_merge":
            mapped = "trip_split_merge"
        elif operator == "u_trip_relocate":
            mapped = "same_robot_swap"
        elif operator == "u_segment_reverse":
            mapped = "same_robot_swap"
        else:
            mapped = "same_robot_swap"
        move = self._propose_u_route_neighbor(rng, priority_robot_ids=priority_robot_ids, forced_move_type=mapped)
        if not bool(move.get("feasible", False)):
            return move
        if operator == "u_segment_reverse":
            plan = self._u_route_state_to_plan(move.get("state", {}))
            robot_ids = [rid for rid, trips in plan.items() if any(len(trip) >= 2 for trip in trips)]
            if robot_ids:
                rid = rng.choice(robot_ids)
                trip_candidates = [trip for trip in plan[rid] if len(trip) >= 2]
                if trip_candidates:
                    trip = rng.choice(trip_candidates)
                    i = rng.randrange(len(trip) - 1)
                    j = rng.randrange(i + 1, len(trip))
                    trip[i:j + 1] = list(reversed(trip[i:j + 1]))
                    move["state"] = self._normalize_u_route_state(self._u_route_plan_to_state(plan))
                    move["move_type"] = operator
        elif operator == "u_late_task_pull_forward":
            plan = self._u_route_state_to_plan(move.get("state", {}))
            late_rows = self._select_priority_u_robot_ids(limit=max(1, strength))
            for rid in late_rows:
                trips = plan.get(int(rid), [])
                if not trips:
                    continue
                trip = max(trips, key=lambda row: len(row))
                if len(trip) >= 2:
                    task_id = trip.pop(-1)
                    trip.insert(0, task_id)
                    move["state"] = self._normalize_u_route_state(self._u_route_plan_to_state(plan))
                    move["move_type"] = operator
                    break
        return move

    def _apply_mode_perturbation(self, mode: str, rng: random.Random):
        if mode == "M_R":
            self._perturb_routing_anchor(rng, max_count=3)
        elif mode == "M_Y":
            self._perturb_station_assignments(rng, max_count=4)
        elif mode == "M_B":
            self._perturb_stack_behavior(rng)
        elif mode == "M_X":
            self._perturb_split_structure(rng)
        elif mode == "M_XYB":
            self._perturb_split_structure(rng)
            self._perturb_station_assignments(rng, max_count=2)
            self._perturb_stack_behavior(rng)

    def _run_vns_for_mode(self, iter_id: int, mode: str, vns_type: str) -> Tuple[Optional[SolutionSnapshot], float, Dict[str, float], int, Dict[str, float]]:
        assert self.work is not None
        current_metrics = self._collect_layer_metrics()
        best_local_snap: Optional[SolutionSnapshot] = None
        best_local_z = float("inf")
        best_local_metrics: Dict[str, float] = {}
        runtime_stats = {
            "trial_count": 0.0,
            "surrogate_evaluated_count": 0.0,
            "full_rebuild_called": 0.0,
            "sp1_called": 0.0,
            "sp2_called": 0.0,
            "sp3_called": 0.0,
            "sp4_called": 0.0,
            "surrogate_score": float("nan"),
            "surrogate_rank": float("nan"),
            "reject_reason": "",
        }
        trials = int(self.cfg.vns_max_trials)
        candidate_rows: List[Dict[str, Any]] = []
        for trial in range(trials):
            self.restore_snapshot(self.work)
            rng = random.Random(int(self.cfg.seed) + iter_id * 1000 + trial * 37 + sum(ord(c) for c in mode))
            before_sig = self._capture_all_signatures()
            self._apply_mode_perturbation(mode, rng)
            after_sig = self._capture_all_signatures()
            change_info = self._derive_dirty_from_changes(mode, before_sig, after_sig)
            runtime_stats["trial_count"] += 1.0

            if not change_info.get("signature_changed", False):
                self.surrogate_stats["signature_skip"] += 1.0
                if vns_type == "Light" and trial >= 2:
                    break
                continue

            trial_metrics = self._collect_layer_metrics()
            surrogate_score = self._compute_surrogate_score(mode, current_metrics, trial_metrics, change_info)
            runtime_stats["surrogate_evaluated_count"] += 1.0
            self.surrogate_stats["evaluated"] += 1.0
            candidate_rows.append(
                {
                    "trial": trial,
                    "surrogate_score": float(surrogate_score),
                    "change_info": change_info,
                    "snap": self.snapshot(self.work_z, iter_id=iter_id),
                }
            )
            if vns_type == "Light" and trial >= 2:
                break

        if not candidate_rows:
            runtime_stats["reject_reason"] = "no_signature_change"
            return None, float("inf"), {}, trials, runtime_stats

        candidate_rows.sort(key=lambda row: (float(row["surrogate_score"]), int(row["trial"])))
        top_k = int(self.cfg.surrogate_top_k_light if vns_type == "Light" else self.cfg.surrogate_top_k_full)
        top_k = max(1, min(top_k, len(candidate_rows)))
        cutoff = float(candidate_rows[0]["surrogate_score"]) + abs(float(candidate_rows[0]["surrogate_score"])) * float(self.cfg.surrogate_prune_ratio)
        promoted_rows = [
            row for idx, row in enumerate(candidate_rows)
            if idx < top_k and float(row["surrogate_score"]) <= cutoff + 1e-9
        ]
        if not promoted_rows:
            promoted_rows = candidate_rows[:top_k]

        self.surrogate_stats["promoted"] += float(len(promoted_rows))
        self.surrogate_stats["pruned"] += float(max(0, len(candidate_rows) - len(promoted_rows)))

        for rank_idx, row in enumerate(promoted_rows, start=1):
            self.restore_snapshot(row["snap"])
            full_rebuild_called = self._sync_tasks_and_maybe_rebuild(row["change_info"].get("dirty", {}), row["change_info"])
            runtime_stats["full_rebuild_called"] += float(full_rebuild_called)
            if row["change_info"]["dirty"].get("x"):
                runtime_stats["sp1_called"] += 1.0
                runtime_stats["sp2_called"] += 1.0
                runtime_stats["sp3_called"] += 1.0
                runtime_stats["sp4_called"] += 1.0
            elif row["change_info"]["dirty"].get("b"):
                runtime_stats["sp3_called"] += 1.0
                runtime_stats["sp4_called"] += 1.0
            elif row["change_info"]["dirty"].get("y") or row["change_info"]["dirty"].get("r"):
                runtime_stats["sp4_called"] += 1.0

            z = float(self.evaluate())
            self._harvest_station_start_times()
            self._update_beta_from_station()
            metrics = self._collect_layer_metrics()
            metrics["z_cand"] = float(z)
            metrics["surrogate_score"] = float(row["surrogate_score"])
            metrics["surrogate_rank"] = float(rank_idx)
            if z < best_local_z - 1e-6:
                best_local_z = float(z)
                best_local_snap = self.snapshot(z, iter_id=iter_id)
                best_local_metrics = dict(metrics)

        if best_local_metrics:
            runtime_stats["surrogate_score"] = float(best_local_metrics.get("surrogate_score", float("nan")))
            runtime_stats["surrogate_rank"] = float(best_local_metrics.get("surrogate_rank", float("nan")))
        elif promoted_rows:
            runtime_stats["reject_reason"] = "promoted_but_not_improved"
        return best_local_snap, float(best_local_z), best_local_metrics, trials, runtime_stats

    def _run_role_vns_main(self) -> float:
        assert self.best is not None
        assert self.work is not None
        mark = 0
        self._notify_progress(0, self.cfg.max_iters, "init")
        for it in range(1, self.cfg.max_iters + 1):
            if it >= self.cfg.switch_to_exact_iter:
                self.cfg.sp2_use_mip = self.cfg.exact_sp2_use_mip
                self.cfg.sp3_use_mip = self.cfg.exact_sp3_use_mip
                self.cfg.sp4_use_mip = self.cfg.exact_sp4_use_mip
                self.cfg.sp2_time_limit_sec = self.cfg.exact_sp2_time_limit_sec
                self.cfg.sp4_lkh_time_limit_seconds = self.cfg.exact_sp4_lkh_time_limit_seconds

            t_iter0 = time.perf_counter()
            mode, mode_eval = self._select_mode()
            roles = self._mode_roles(mode)
            lb_bundle = mode_eval.get(mode, self._compute_lb_bundle(mode))
            lb = float(lb_bundle["LB_mode"])
            gap_ratio = float(lb_bundle["gap_mode"])
            self.mode_stats[mode]["calls"] += 1.0
            self.mode_stats[mode]["last_gap"] = float(gap_ratio)
            z_before = float(self.work_z)
            current_metrics = self._collect_layer_metrics()

            if gap_ratio <= float(self.cfg.eps_skip) or self.mode_stats[mode]["fail"] >= float(self.cfg.mode_fail_limit):
                self.mode_stats[mode]["skip"] += 1.0
                self.iter_log.append({
                    "iter": int(it),
                    "focus": mode,
                    "mode": mode,
                    "roles": dict(roles),
                    "vns_type": "Skip",
                    "z": float("nan"),
                    "z_before": float(z_before),
                    "z_cand": float("nan"),
                    "best_z": float(self.best.z),
                    "best_z_after": float(self.best.z),
                    "improved": False,
                    "skipped": True,
                    "lb": float(lb),
                    "LB_0": float(lb_bundle["LB_0"]),
                    "LB_mode": float(lb_bundle["LB_mode"]),
                    "gap_mode": float(gap_ratio),
                    "accepted_type": "skip",
                    "reject_reason": "gap_skip_or_mode_fail",
                    "iter_runtime_sec": float(time.perf_counter() - t_iter0),
                    "epsilon": float(self.cfg.epsilon),
                    "trial_count": 0.0,
                    "surrogate_evaluated_count": 0.0,
                    "full_rebuild_called": 0.0,
                    "sp1_called": 0.0,
                    "sp2_called": 0.0,
                    "sp3_called": 0.0,
                    "sp4_called": 0.0,
                    "surrogate_score": float("nan"),
                    "surrogate_rank": float("nan"),
                    **current_metrics,
                })
                self._notify_progress(it, self.cfg.max_iters, mode)
                mark += 1
                if mark >= self.cfg.no_improve_limit:
                    break
                continue

            vns_type = "Light" if gap_ratio <= float(self.cfg.eps_light) else "Full"
            cand_snap, cand_z, cand_metrics, _, runtime_stats = self._run_vns_for_mode(it, mode, vns_type)
            if cand_snap is not None:
                self.restore_snapshot(cand_snap)

            z_cand = float(cand_z) if cand_snap is not None else float("nan")
            accepted_type = "reject"
            improved = False
            if cand_snap is not None and z_cand < float(self.best.z) - 1e-6:
                self.best = cand_snap
                self.work = cand_snap
                self.work_z = float(z_cand)
                mark = 0
                improved = True
                accepted_type = "strong"
                self.mode_stats[mode]["success"] += 1.0
                self.mode_stats[mode]["fail"] = 0.0
            elif cand_snap is not None and z_cand < float(self.work_z) - 1e-6:
                self.work = cand_snap
                self.work_z = float(z_cand)
                accepted_type = "improve_work"
                self.mode_stats[mode]["success"] += 1.0
                self.mode_stats[mode]["fail"] = 0.0
                mark += 1
            elif cand_snap is not None:
                weak_ok = z_cand <= float(self.work_z) * (1.0 + float(self.cfg.weak_accept_eta))
                if weak_ok and self._structural_score(cand_metrics) < self._structural_score(current_metrics) - 1e-9:
                    self.work = cand_snap
                    self.work_z = float(z_cand)
                    accepted_type = "weak"
                    self.mode_stats[mode]["success"] += 1.0
                    self.mode_stats[mode]["fail"] = 0.0
                else:
                    self.mode_stats[mode]["fail"] += 1.0
                mark += 1
            else:
                self.mode_stats[mode]["fail"] += 1.0
                mark += 1

            row_metrics = cand_metrics if cand_metrics else current_metrics
            self.iter_log.append({
                "iter": int(it),
                "focus": mode,
                "mode": mode,
                "roles": dict(roles),
                "vns_type": vns_type,
                "z": float(z_cand) if cand_snap is not None else float("nan"),
                "z_before": float(z_before),
                "z_cand": float(z_cand) if cand_snap is not None else float("nan"),
                "best_z": float(self.best.z),
                "best_z_after": float(self.best.z),
                "improved": bool(improved),
                "skipped": False,
                "lb": float(lb),
                "LB_0": float(lb_bundle["LB_0"]),
                "LB_mode": float(lb_bundle["LB_mode"]),
                "gap_mode": float(gap_ratio),
                "accepted_type": accepted_type,
                "iter_runtime_sec": float(time.perf_counter() - t_iter0),
                "epsilon": float(self.cfg.epsilon),
                **runtime_stats,
                **row_metrics,
            })
            self._notify_progress(it, self.cfg.max_iters, mode)
            self.restore_snapshot(self.work)
            self._refresh_runtime_cache(self.work_z)

            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            self._write_logs()
        if self.cfg.export_best_solution:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            self.export_best()
        return float(self.best.z)

    def run(self) -> float:
        if self.problem is None:
            self.initialize()

        assert self.best is not None
        assert self.work is not None

        # Precheck aborted: short-circuit
        if self.precheck_aborted:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            if self.cfg.write_iteration_logs:
                self._write_logs()
            return float("nan")

        if str(getattr(self.cfg, "search_scheme", "legacy")).lower() == "layer_augmented":
            z_final = self._run_layer_augmented_main()
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            return z_final

        if self.cfg.enable_role_vns:
            z_final = self._run_role_vns_main()
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            return z_final

        mark = 0
        self._notify_progress(0, self.cfg.max_iters, "init")
        for it in range(1, self.cfg.max_iters + 1):
            # 动态切换求解策略（不改变约束，只改变求解器/时间限制）
            if it >= self.cfg.switch_to_exact_iter:
                self.cfg.sp2_use_mip = self.cfg.exact_sp2_use_mip
                self.cfg.sp3_use_mip = self.cfg.exact_sp3_use_mip
                self.cfg.sp4_use_mip = self.cfg.exact_sp4_use_mip
                self.cfg.sp2_time_limit_sec = self.cfg.exact_sp2_time_limit_sec
                self.cfg.sp4_lkh_time_limit_seconds = self.cfg.exact_sp4_lkh_time_limit_seconds

            focus = ["sp4", "sp3", "sp2", "sp1"][it % 4]  # 旋转

            # 下界过滤
            lb = self.compute_lb(focus)
            gap_ratio = (self.best.z - lb) / self.best.z if self.best.z > 1e-9 else 0.0
            if gap_ratio <= self.cfg.epsilon:
                self._append_iter_log(it, focus=focus, z=float("nan"), improved=False, skipped=True, lb=lb)
                self._notify_progress(it, self.cfg.max_iters, focus)
                mark += 1
                if mark >= self.cfg.no_improve_limit:
                    break
                continue

            # 按 focus 重点优化，并重建下游
            if focus == "sp2":
                if self.cfg.sp2_use_mip:
                    self._run_sp2_mip()
                else:
                    self._run_sp2_initial()
                self._run_sp3()
                self._run_sp4()
            elif focus == "sp1":
                self._run_sp1()
                if self.cfg.sp2_use_mip:
                    self._run_sp2_mip()
                else:
                    self._run_sp2_initial()
                self._run_sp3()
                self._run_sp4()
            elif focus == "sp3":
                self._run_sp3()
                self._run_sp4()
            else:
                self._run_sp4()

            z = self.evaluate()
            # SKU affinity feedback (inject incompatibility pairs; no SP1 re-run in this iteration)
            if self.cfg.enable_sku_affinity:
                try:
                    self._apply_sku_affinity_feedback()
                except Exception:
                    pass
            # update μ/β caches for next iteration
            self._harvest_station_start_times()
            self._update_beta_from_station()
            improved = z < self.best.z - 1e-6
            if improved:
                self.best = self.snapshot(z, iter_id=it)
                mark = 0
            else:
                mark += 1

            self._append_iter_log(it, focus=focus, z=z, improved=improved, skipped=False, lb=lb)
            self._notify_progress(it, self.cfg.max_iters, focus)

            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            self._write_logs()
        if self.cfg.export_best_solution:
            self.run_total_time_sec = float(self._runtime_elapsed_sec())
            self.export_best()
        self.run_total_time_sec = float(self._runtime_elapsed_sec())
        return float(self.best.z)

    # ----------------------------
    # Precheck & soft-coupling helpers
    # ----------------------------
    def _precheck_sp3_coverage(self) -> Dict:
        import copy
        pc = copy.deepcopy(self.problem)
        sp1 = SP1_BOM_Splitter(pc)
        sp2 = SP2_Station_Assigner(pc)
        sp3 = SP3_Bin_Hitter(pc)
        sub_tasks = sp1.solve(use_mip=False)
        pc.subtask_list = sub_tasks
        pc.subtask_num = len(sub_tasks)
        sp2.solve_initial_heuristic()
        if self.cfg.sp3_precheck_use_mip:
            physical_tasks, _, sorting_costs = sp3.solve(sub_tasks, beta_congestion=1.0, sp4_routing_costs=None)
        else:
            heuristic = sp3.SP3_Heuristic_Solver(pc)
            physical_tasks, _, sorting_costs = heuristic.solve(sub_tasks, beta_congestion=1.0)
        pc.task_list = physical_tasks
        pc.task_num = len(physical_tasks)

        from collections import defaultdict
        unmet_total = 0
        unmet_subtasks = 0
        details = []
        id_to_tote = pc.id_to_tote
        for st in sub_tasks:
            req = defaultdict(int)
            for sku in st.sku_list:
                req[sku.id] += 1
            prov = defaultdict(int)
            hit_totes = []
            chosen_stacks = set()
            for t in getattr(st, "execution_tasks", []) or []:
                hit_totes.extend(list(t.hit_tote_ids or []))
                chosen_stacks.add(int(getattr(t, "target_stack_id", -1)))
            for tid in hit_totes:
                tote = id_to_tote.get(tid)
                if not tote:
                    continue
                for sid, qty in tote.sku_quantity_map.items():
                    if req.get(sid, 0) > 0 and prov[sid] < req[sid]:
                        use = min(int(qty), req[sid] - prov[sid])
                        if use > 0:
                            prov[sid] += use
            unmet_skus = {sid: max(0, req[sid] - prov.get(sid, 0)) for sid in req.keys() if req[sid] - prov.get(sid, 0) > 0}
            this_unmet = int(sum(unmet_skus.values()))
            if this_unmet > 0:
                unmet_subtasks += 1
                unmet_total += this_unmet
            details.append({
                "subtask_id": int(st.id),
                "unmet": int(this_unmet),
                "unmet_skus": unmet_skus,
                "chosen_stacks": sorted([s for s in chosen_stacks if s >= 0]),
                "hit_totes": list(sorted(set(hit_totes))),
            })

        result = {
            "unmet_subtask_count": int(unmet_subtasks),
            "unmet_sku_total": int(unmet_total),
            "details": details,
        }
        if os.environ.get("OFS_BATCH_SILENT", "0") != "1":
            try:
                with open(self._log_path("sp3_precheck.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                with open(self._log_path("sp3_precheck.txt"), "w", encoding="utf-8") as f:
                    f.write(f"unmet_subtask_count={unmet_subtasks}\n")
                    f.write(f"unmet_sku_total={unmet_total}\n")
            except Exception:
                pass
        return result

    def _harvest_station_start_times(self):
        start_map: Dict[int, float] = {}
        for station in self.problem.station_list:
            for t in getattr(station, "processed_tasks", []) or []:
                start_map[int(getattr(t, "task_id", -1))] = float(getattr(t, "start_process_time", 0.0))
        self.last_station_start_times = start_map

    def _update_beta_from_station(self):
        if not self.cfg.enable_soft_beta:
            self.last_beta_value = 1.0
            return
        makespan = float(getattr(self.problem, "global_makespan", 0.0))
        if makespan <= 1e-9:
            self.last_beta_value = 1.0
            return
        total_idle = sum(float(getattr(s, "total_idle_time", 0.0)) for s in self.problem.station_list)
        avg_idle_ratio = (total_idle / len(self.problem.station_list)) / makespan if self.problem.station_list else 0.0
        congestion = max(0.0, 1.0 - avg_idle_ratio)
        beta = self.cfg.beta_base + self.cfg.beta_gain * congestion
        beta = max(self.cfg.beta_min, min(self.cfg.beta_max, beta))
        self.last_beta_value = float(beta)

    def _compute_shadow_prices(self) -> Dict[Tuple[int, int], float]:
        if not self.cfg.enable_soft_pi:
            return {}
        pi: Dict[Tuple[int, int], float] = {}
        stations = self.problem.station_list
        D0 = float(self.cfg.d0_threshold)
        scale = float(self.cfg.pi_scale)
        clipv = float(self.cfg.pi_clip)
        for st in self.problem.subtask_list:
            order_id = int(getattr(st.parent_order, "order_id", -1))
            stacks = getattr(st, "involved_stacks", None) or []
            stack_points = [s.store_point for s in stacks if getattr(s, "store_point", None)]
            if not stack_points:
                continue
            for s in stations:
                dmin = 1e9
                for p in stack_points:
                    d = abs(p.x - s.point.x) + abs(p.y - s.point.y)
                    if d < dmin:
                        dmin = d
                val = max(0.0, float(dmin) - D0) * scale
                if clipv > 0:
                    val = min(val, clipv)
                pi[(order_id, int(s.id))] = float(val)
        return pi

    def _apply_sku_affinity_feedback(self):
        # 从 subtask 跨距筛选，向 SP1 注入有限对数的 SKU 互斥对
        if not self.sp1:
            return
        th = int(self.cfg.affinity_span_threshold)
        max_pairs = int(self.cfg.affinity_pairs_per_task)
        for st in self.problem.subtask_list:
            stacks = getattr(st, "involved_stacks", None) or []
            if len(stacks) < 2:
                continue
            pts = [s.store_point for s in stacks if getattr(s, "store_point", None)]
            if len(pts) < 2:
                continue
            max_span = 0
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = abs(pts[i].x - pts[j].x) + abs(pts[i].y - pts[j].y)
                    if d > max_span:
                        max_span = d
            if max_span < th:
                continue
            sku_ids = sorted({sku.id for sku in getattr(st, 'unique_sku_list', []) or []})
            cnt = 0
            for i in range(len(sku_ids)):
                for j in range(i+1, len(sku_ids)):
                    self.sp1.add_incompatibility(sku_ids[i], sku_ids[j])
                    cnt += 1
                    if cnt >= max_pairs:
                        break
                if cnt >= max_pairs:
                    break

    def _compute_solution_coverage(self) -> Dict[str, Any]:
        subtask_rows: List[Dict[str, Any]] = []
        unmet_total = 0
        unmet_subtasks = 0
        tote_map = getattr(self.problem, "id_to_tote", {}) if self.problem is not None else {}
        for st in getattr(self.problem, "subtask_list", []) or []:
            req: Dict[int, int] = {}
            for sku in getattr(st, "sku_list", []) or []:
                sid = int(getattr(sku, "id", -1))
                if sid >= 0:
                    req[sid] = req.get(sid, 0) + 1
            prov: Dict[int, int] = {}
            for task in getattr(st, "execution_tasks", []) or []:
                for tote_id in getattr(task, "target_tote_ids", []) or []:
                    tote = tote_map.get(int(tote_id))
                    if tote is None:
                        continue
                    for sid, qty in getattr(tote, "sku_quantity_map", {}).items():
                        sid_i = int(sid)
                        if sid_i in req:
                            prov[sid_i] = prov.get(sid_i, 0) + int(qty)
            unmet = {sid: int(max(0, req[sid] - prov.get(sid, 0))) for sid in req if req[sid] - prov.get(sid, 0) > 0}
            unmet_units = int(sum(unmet.values()))
            if unmet_units > 0:
                unmet_total += unmet_units
                unmet_subtasks += 1
            subtask_rows.append({
                "subtask_id": int(getattr(st, "id", -1)),
                "order_id": int(getattr(getattr(st, "parent_order", None), "order_id", -1)),
                "required_sku_units": int(sum(req.values())),
                "provided_sku_units": int(sum(min(req.get(sid, 0), prov.get(sid, 0)) for sid in req)),
                "unmet_sku_units": int(unmet_units),
                "unmet_skus": unmet,
                "coverage_ok": bool(unmet_units == 0),
            })
        return {
            "coverage_ok": bool(unmet_total == 0),
            "unmet_sku_total": int(unmet_total),
            "unmet_subtask_count": int(unmet_subtasks),
            "subtasks": subtask_rows,
        }

    def _write_best_solution_summary(self, out_dir: str, z: float):
        metrics = self._collect_layer_metrics()
        coverage = self._compute_solution_coverage()
        summary = {
            "best_iter": int(self.best.iter_id) if self.best is not None else -1,
            "best_z": float(self.best.z) if self.best is not None else float(z),
            "recomputed_z": float(z),
            "global_makespan": float(getattr(self.problem, "global_makespan", 0.0)),
            "run_total_time_sec": float(self.run_total_time_sec if self.run_total_time_sec > 0.0 else self._runtime_elapsed_sec()),
            "sp1": {
                "subtask_count": int(metrics.get("subtask_count", 0.0)),
                "avg_sku_per_subtask": float(metrics.get("avg_sku_per_subtask", 0.0)),
                "max_sku_per_subtask": float(metrics.get("max_sku_per_subtask", 0.0)),
            },
            "sp2": {
                "station_idle_total": float(metrics.get("station_idle_total", 0.0)),
                "station_utilization_mean": float(metrics.get("station_utilization_mean", 0.0)),
                "station_load_max": float(metrics.get("station_load_max", 0.0)),
                "station_load_std": float(metrics.get("station_load_std", 0.0)),
                "station_load_max_ratio": float(metrics.get("station_load_max_ratio", 0.0)),
            },
            "sp3": {
                "hit_stack_count": float(metrics.get("hit_stack_count", 0.0)),
                "noise_ratio": float(metrics.get("noise_ratio", 0.0)),
                "avg_stack_span": float(metrics.get("avg_stack_span", 0.0)),
                "sorting_cost_proxy": float(metrics.get("sorting_cost_proxy", 0.0)),
            },
            "sp4": {
                "robot_path_length_total": float(metrics.get("robot_path_length_total", 0.0)),
                "robot_utilization_mean": float(metrics.get("robot_utilization_mean", 0.0)),
                "latest_robot_finish": float(metrics.get("latest_robot_finish", 0.0)),
                "robot_finish_ratio": float(metrics.get("robot_finish_ratio", 0.0)),
                "arrival_slack_mean": float(metrics.get("arrival_slack_mean", 0.0)),
            },
            "coverage": coverage,
        }
        with open(os.path.join(out_dir, "best_solution_objectives.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "best_solution_objectives.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_iter={summary['best_iter']}\n")
            f.write(f"best_z={summary['best_z']:.6f}\n")
            f.write(f"global_makespan={summary['global_makespan']:.6f}\n")
            f.write(f"run_total_time_sec={summary['run_total_time_sec']:.6f}\n")
            f.write(
                f"sp1_subtask_count={summary['sp1']['subtask_count']}, "
                f"sp1_avg_sku_per_subtask={summary['sp1']['avg_sku_per_subtask']:.6f}, "
                f"sp1_max_sku_per_subtask={summary['sp1']['max_sku_per_subtask']:.6f}\n"
            )
            f.write(
                f"sp2_station_idle_total={summary['sp2']['station_idle_total']:.6f}, "
                f"sp2_station_load_max={summary['sp2']['station_load_max']:.6f}, "
                f"sp2_station_load_std={summary['sp2']['station_load_std']:.6f}\n"
            )
            f.write(
                f"sp3_hit_stack_count={summary['sp3']['hit_stack_count']:.6f}, "
                f"sp3_noise_ratio={summary['sp3']['noise_ratio']:.6f}, "
                f"sp3_avg_stack_span={summary['sp3']['avg_stack_span']:.6f}, "
                f"sp3_sorting_cost_proxy={summary['sp3']['sorting_cost_proxy']:.6f}\n"
            )
            f.write(
                f"sp4_robot_path_length_total={summary['sp4']['robot_path_length_total']:.6f}, "
                f"sp4_latest_robot_finish={summary['sp4']['latest_robot_finish']:.6f}, "
                f"sp4_arrival_slack_mean={summary['sp4']['arrival_slack_mean']:.6f}\n"
            )
            f.write(
                f"coverage_ok={summary['coverage']['coverage_ok']}, "
                f"unmet_sku_total={summary['coverage']['unmet_sku_total']}, "
                f"unmet_subtask_count={summary['coverage']['unmet_subtask_count']}\n"
            )

    def _write_logs(self):
        log_dir = self._ensure_log_dir()
        summary_path = self._log_path("tra_summary.json")
        run_stats = self._runtime_stats_payload()
        best_summary = None
        if self.best:
            best_summary = {
                "z": float(self.best.z),
                "iter_id": int(self.best.iter_id),
                "seed": int(self.best.seed),
                "subtask_station_rank": dict(self.best.subtask_station_rank),
                "sp1_capacity_limits": dict(self.best.sp1_capacity_limits),
                "sp1_incompatibility_pairs": list(self.best.sp1_incompatibility_pairs),
                "task_count": int(len(self._iter_snapshot_tasks(self.best))),
            }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": asdict(self.cfg),
                "best": best_summary,
                "run_stats": run_stats,
                "iters": self.iter_log,
            }, f, ensure_ascii=False, indent=2)

        # 另存一份更易读的 txt
        txt_path = self._log_path("tra_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== TRA Rotating Outer Loop Summary ===\n")
            f.write(f"scale={self.cfg.scale}, seed={self.cfg.seed}\n")
            f.write(f"total_runtime_sec={run_stats['run_total_time_sec']:.6f}\n")
            f.write(f"best_z={self.best.z:.3f}s @ iter={self.best.iter_id}\n\n")
            f.write(f"{'iter':>4} | {'focus':>4} | {'z':>10} | {'best':>10} | {'imp':>3} | {'skip':>4} | {'lb':>10}\n")
            f.write("-" * 72 + "\n")
            for row in self.iter_log:
                z = row["z"]
                z_str = "SKIP" if (isinstance(z, float) and math.isnan(z)) else f"{z:10.3f}"
                lb = row["lb"]
                lb_str = "   -   " if lb is None else f"{lb:10.3f}"
                f.write(f"{row['iter']:4d} | {row['focus']:>4} | {z_str} | {row['best_z']:10.3f} | "
                        f"{'Y' if row['improved'] else 'N':>3} | {('Y' if row['skipped'] else 'N'):>4} | {lb_str}\n")

        print(f"  >>> [TRA] Logs written to {log_dir}")

        if self.cfg.enable_sp1_feedback_analysis:
            self._write_sp1_feedback_suggestions()

    def _write_sp1_feedback_suggestions(self):
        """
        生成“更外层（SP1）”的软耦合反馈建议文件：
        - 识别空间跨度很大的 SubTask（通常意味着 SKU 组合跨区，SP4 运输代价高）
        - 给出建议：对其所属 Order 降低容量上限，从而让 SP1 拆得更细
        """
        suggestions = []
        for st in self.problem.subtask_list:
            stacks = getattr(st, "involved_stacks", None) or []
            if len(stacks) < 2:
                continue
            pts = [s.store_point for s in stacks if getattr(s, "store_point", None)]
            if len(pts) < 2:
                continue
            max_span = 0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = abs(pts[i].x - pts[j].x) + abs(pts[i].y - pts[j].y)
                    if d > max_span:
                        max_span = d
            if max_span >= self.cfg.sp1_feedback_spread_threshold:
                order_id = getattr(st.parent_order, "order_id", None)
                suggestions.append({
                    "subtask_id": int(st.id),
                    "order_id": int(order_id) if order_id is not None else -1,
                    "station_id": int(getattr(st, "assigned_station_id", -1)),
                    "rank": int(getattr(st, "station_sequence_rank", -1)),
                    "stack_cnt": int(len(stacks)),
                    "max_span_L1": int(max_span),
                })

        suggestions.sort(key=lambda x: (-x["max_span_L1"], x["order_id"], x["subtask_id"]))
        suggestions = suggestions[: max(0, int(self.cfg.sp1_feedback_top_k))]

        # 聚合成“建议容量上限”
        order_to_suggested_cap = {}
        if self.sp1 is not None:
            for item in suggestions:
                oid = item["order_id"]
                if oid < 0:
                    continue
                curr = self.sp1.order_capacity_limits.get(oid, None)
                if curr is None:
                    continue
                # 简单策略：每触发一次建议就把 cap-1（下限=1）
                order_to_suggested_cap[oid] = max(1, min(curr, order_to_suggested_cap.get(oid, curr)) - 1)

        path = self._log_path("tra_sp1_feedback_suggestions.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "spread_threshold": int(self.cfg.sp1_feedback_spread_threshold),
                "top_k": int(self.cfg.sp1_feedback_top_k),
                "subtask_suggestions": suggestions,
                "order_capacity_suggestions": order_to_suggested_cap,
            }, f, ensure_ascii=False, indent=2)

    def export_best(self):
        out_dir = self._log_path("tra_best_export")
        self.export_best_to(out_dir)
        print(f"  >>> [TRA] Best solution exported to {out_dir}")

    def export_best_to(self, out_dir: str):
        # Restore the frozen best state directly, then recompute simulation/export from that exact snapshot.
        assert self.best is not None
        self._set_seed(self.best.seed)
        self.restore_snapshot(self.best)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        calc = GlobalTimeCalculator(self.problem)
        z = float(calc.calculate())
        calc.calculate_and_export(out_dir)
        self._verify_makespan_breakdown(out_dir)
        self._write_best_solution_summary(out_dir, z)
        self._write_best_solution_dump(out_dir, z)

    def _write_best_solution_dump(self, out_dir: str, z: float):
        path = os.path.join(out_dir, "best_solution_full_dump.txt")
        all_tasks = []
        for st in self.problem.subtask_list:
            all_tasks.extend(getattr(st, "execution_tasks", []) or [])
        all_tasks.sort(key=lambda t: (int(getattr(t, "target_station_id", -1)), float(getattr(t, "start_process_time", 0.0)), int(getattr(t, "task_id", -1))))

        with open(path, "w", encoding="utf-8") as f:
            f.write("[TRA Best Solution Dump]\n")
            f.write(f"best_iter={int(self.best.iter_id)}\n")
            f.write(f"seed={int(self.best.seed)}\n")
            f.write(f"best_z={float(self.best.z):.6f}\n")
            f.write(f"recomputed_z={float(z):.6f}\n")
            f.write(f"global_makespan={float(getattr(self.problem, 'global_makespan', 0.0)):.6f}\n")
            f.write("\n[TRA Iter Log]\n")
            for row in self.iter_log:
                f.write(
                    f"iter={int(row.get('iter', 0))}, focus={row.get('focus', '')}, "
                    f"z={row.get('z', float('nan'))}, best_z={row.get('best_z', float('nan'))}, "
                    f"improved={bool(row.get('improved', False))}, skipped={bool(row.get('skipped', False))}, "
                    f"lb={row.get('lb', None)}, epsilon={row.get('epsilon', None)}\n"
                )

            f.write("\n[SP1 Decisions]\n")
            for st in sorted(self.problem.subtask_list, key=lambda x: int(x.id)):
                sku_ids = [int(s.id) for s in getattr(st, "sku_list", []) or []]
                unique_sku_ids = sorted({int(s.id) for s in getattr(st, "unique_sku_list", []) or []})
                f.write(
                    f"subtask_id={int(st.id)}, order_id={int(getattr(st.parent_order, 'order_id', -1))}, "
                    f"sku_units={len(sku_ids)}, unique_skus={unique_sku_ids}, sku_list={sku_ids}\n"
                )

            f.write("\n[SP2 Decisions]\n")
            for st in sorted(self.problem.subtask_list, key=lambda x: (int(getattr(x, 'assigned_station_id', -1)), int(getattr(x, 'station_sequence_rank', -1)), int(x.id))):
                f.write(
                    f"subtask_id={int(st.id)}, station_id={int(getattr(st, 'assigned_station_id', -1))}, "
                    f"rank={int(getattr(st, 'station_sequence_rank', -1))}, est_start={float(getattr(st, 'estimated_process_start_time', 0.0)):.6f}\n"
                )

            f.write("\n[SP3 Decisions]\n")
            for t in sorted(all_tasks, key=lambda x: int(getattr(x, "task_id", -1))):
                f.write(
                    f"task_id={int(t.task_id)}, subtask_id={int(t.sub_task_id)}, stack_id={int(t.target_stack_id)}, "
                    f"station_id={int(t.target_station_id)}, mode={getattr(t, 'operation_mode', '')}, "
                    f"target_totes={list(getattr(t, 'target_tote_ids', []) or [])}, "
                    f"hit_totes={list(getattr(t, 'hit_tote_ids', []) or [])}, "
                    f"noise_totes={list(getattr(t, 'noise_tote_ids', []) or [])}, "
                    f"sort_range={getattr(t, 'sort_layer_range', None)}, load={int(getattr(t, 'total_load_count', 0))}, "
                    f"sku_pick_count={int(getattr(t, 'sku_pick_count', 0))}, robot_service_time={float(getattr(t, 'robot_service_time', 0.0)):.6f}, "
                    f"station_service_time={float(getattr(t, 'station_service_time', 0.0)):.6f}\n"
                )

            f.write("\n[SP4 Decisions]\n")
            for st in sorted(self.problem.subtask_list, key=lambda x: int(x.id)):
                f.write(
                    f"subtask_id={int(st.id)}, assigned_robot_id={int(getattr(st, 'assigned_robot_id', -1))}\n"
                )
            for t in sorted(all_tasks, key=lambda x: int(getattr(x, "task_id", -1))):
                f.write(
                    f"task_id={int(t.task_id)}, robot_id={int(getattr(t, 'robot_id', -1))}, trip_id={int(getattr(t, 'trip_id', 0))}, "
                    f"arrival_stack={float(getattr(t, 'arrival_time_at_stack', 0.0)):.6f}, "
                    f"arrival_station={float(getattr(t, 'arrival_time_at_station', 0.0)):.6f}\n"
                )

            f.write("\n[Z Reproduction Fields]\n")
            f.write("z = max(task.end_process_time)\n")
            for t in all_tasks:
                f.write(
                    f"task_id={int(t.task_id)}, subtask_id={int(t.sub_task_id)}, station_id={int(t.target_station_id)}, "
                    f"robot_id={int(getattr(t, 'robot_id', -1))}, trip_id={int(getattr(t, 'trip_id', 0))}, "
                    f"arrival_stack={float(getattr(t, 'arrival_time_at_stack', 0.0)):.6f}, "
                    f"robot_service_time={float(getattr(t, 'robot_service_time', 0.0)):.6f}, "
                    f"arrival_station={float(getattr(t, 'arrival_time_at_station', 0.0)):.6f}, "
                    f"sku_pick_count={int(getattr(t, 'sku_pick_count', 0))}, "
                    f"picking_duration={float(getattr(t, 'picking_duration', 0.0)):.6f}, "
                    f"station_service_time={float(getattr(t, 'station_service_time', 0.0)):.6f}, "
                    f"extra_service_used={float(getattr(t, 'extra_service_used', 0.0)):.6f}, "
                    f"start_process_time={float(getattr(t, 'start_process_time', 0.0)):.6f}, "
                    f"end_process_time={float(getattr(t, 'end_process_time', 0.0)):.6f}, "
                    f"tote_wait_time={float(getattr(t, 'tote_wait_time', 0.0)):.6f}, "
                    f"total_process_duration={float(getattr(t, 'total_process_duration', 0.0)):.6f}, "
                    f"noise_cnt={len(getattr(t, 'noise_tote_ids', []) or [])}\n"
                )

    def _verify_makespan_breakdown(self, out_dir: str):
        all_tasks = []
        for st in self.problem.subtask_list:
            all_tasks.extend(getattr(st, "execution_tasks", []) or [])

        failures = []
        station_task_rows = []

        for station in self.problem.station_list:
            seq = sorted(getattr(station, "processed_tasks", []) or [], key=lambda t: t.start_process_time)
            prev_end = 0.0
            for t in seq:
                extra = float(getattr(t, "extra_service_used", 0.0))
                expected_end = float(t.start_process_time) + float(t.picking_duration) + extra
                if abs(expected_end - float(t.end_process_time)) > 1e-6:
                    failures.append(
                        f"Task {t.task_id}: end mismatch expected={expected_end:.6f}, actual={float(t.end_process_time):.6f}"
                    )
                if float(t.start_process_time) + 1e-6 < prev_end:
                    failures.append(
                        f"Station {station.id}: FCFS violation at task {t.task_id}, "
                        f"start={float(t.start_process_time):.6f} < prev_end={prev_end:.6f}"
                    )
                prev_end = float(t.end_process_time)
                station_task_rows.append({
                    "station_id": int(station.id),
                    "task_id": int(t.task_id),
                    "start": float(t.start_process_time),
                    "end": float(t.end_process_time),
                    "wait": float(getattr(t, "tote_wait_time", 0.0)),
                    "pick": float(getattr(t, "picking_duration", 0.0)),
                    "extra": extra,
                })

        max_end = max((float(getattr(t, "end_process_time", 0.0)) for t in all_tasks), default=0.0)
        global_makespan = float(getattr(self.problem, "global_makespan", 0.0))
        if abs(max_end - global_makespan) > 1e-6:
            failures.append(
                f"Global makespan mismatch: max_task_end={max_end:.6f}, global_makespan={global_makespan:.6f}"
            )

        coverage = self._compute_solution_coverage()
        if int(coverage.get("unmet_sku_total", 0)) > 0:
            failures.append(
                f"SKU coverage unmet: unmet_sku_total={int(coverage.get('unmet_sku_total', 0))}, "
                f"unmet_subtask_count={int(coverage.get('unmet_subtask_count', 0))}"
            )

        result = {
            "status": "PASS" if not failures else "FAIL",
            "task_count": len(all_tasks),
            "max_task_end": max_end,
            "global_makespan": global_makespan,
            "coverage": coverage,
            "failures": failures,
            "station_task_rows": station_task_rows,
        }

        json_path = os.path.join(out_dir, "tra_makespan_verification.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        txt_path = os.path.join(out_dir, "tra_makespan_verification.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("[TRA Makespan Verification]\n")
            f.write(f"status={result['status']}\n")
            f.write(f"task_count={result['task_count']}\n")
            f.write(f"max_task_end={max_end:.6f}\n")
            f.write(f"global_makespan={global_makespan:.6f}\n")
            f.write(f"coverage_ok={bool(coverage.get('coverage_ok', False))}\n")
            f.write(f"unmet_sku_total={int(coverage.get('unmet_sku_total', 0))}\n")
            f.write(f"unmet_subtask_count={int(coverage.get('unmet_subtask_count', 0))}\n")
            if failures:
                f.write("failures:\n")
                for item in failures:
                    f.write(f"- {item}\n")


if __name__ == "__main__":
    cfg = TRARunConfig(
        scale="SMALL",
        seed=42,
        max_iters=6,
        no_improve_limit=3,
        epsilon=0.05,
        sp2_use_mip=True,
        sp3_use_mip=False,
        sp4_use_mip=False,
        sp2_time_limit_sec=10.0,
        sp4_lkh_time_limit_seconds=5,
        export_best_solution=True,
    )
    opt = TRAOptimizer(cfg)
    best_z = opt.run()
    print(f"Best z = {best_z:.3f}s")

