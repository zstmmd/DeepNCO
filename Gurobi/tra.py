import os
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
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
from Gurobi.layer_surrogate import (
    CandidatePrediction,
    F1EvalResult,
    SurrogateLayerState,
    TrainingSample,
    OnlineBinaryRankEnsemble,
    OnlineFeatureScaler,
    OnlineResidualEnsemble,
)
from Gurobi.resource_time_alns import ResourceTimeALNSEngine
from Gurobi.resource_time_alns.reporting import (
    write_resource_time_best_runtime_txt,
    write_resource_time_candidates_csv,
    write_resource_time_iters_csv,
)
from Gurobi.resource_time_alns.runtime_support import init_resource_time_runtime_state

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

    # “先启发式、后精确”切换（满足同一组约束：只改变求解策略/时间上限）
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

    # 渭 / 蟺 / 尾 params
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
    enable_role_vns: bool = False
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
    search_scheme: str = "resource_time_alns"
    resource_real_eval_period: int = 8
    resource_validation_delta_thresh: float = 0.03
    resource_sa_init_temp: float = 10.0
    resource_sa_cooling: float = 0.95
    resource_sa_reheat_factor: float = 1.25
    resource_weight_reaction: float = 0.2
    resource_operator_update_batch_size: int = 10
    resource_operator_update_max_stale_rounds: int = 15
    resource_operator_weight_floor: float = 0.1
    resource_component_weight_x: float = 1.0
    resource_component_weight_y: float = 1.0
    resource_component_weight_z: float = 1.0
    resource_layer_base_weight_x: float = 0.10
    resource_layer_base_weight_y: float = 0.45
    resource_layer_base_weight_z: float = 0.45
    resource_use_surrogate_calibrator: bool = True
    resource_destroy_candidate_pool_size: int = 3
    resource_destroy_candidate_pool_weights: Tuple[float, ...] = ()
    resource_candidate_pool_size: int = 3
    resource_candidate_pool_max_attempts: int = 12
    resource_candidate_pool_log: bool = True
    resource_action_signature_history_size: int = 30
    resource_destroy_degree_x: int = 2
    resource_destroy_degree_y: int = 2
    resource_destroy_degree_z: int = 1
    resource_destroy_mu_base: float = 0.10
    resource_destroy_mu_medium: float = 0.20
    resource_destroy_mu_medium_trigger: int = 30
    resource_destroy_mu_heavy: float = 0.35
    resource_heavy_destroy_trigger: int = 50
    resource_residual_half_life: float = 3.0
    resource_residual_uncertainty_cap: float = 80.0
    resource_surrogate_trust_radius: float = 0.35
    resource_layer_explore_eps: float = 0.10
    resource_layer_score_epsilon: float = 0.05
    resource_stagnation_boost: float = 0.15
    resource_force_rotate_threshold: int = 20
    resource_projection_full_y_refresh_prob: float = 0.10
    resource_x_sa_temp_multiplier: float = 2.0
    resource_duplicate_tote_penalty: float = 100000.0
    resource_empty_candidate_reward: float = -2.0
    resource_empty_candidate_layer_cooldown: int = 3
    resource_stop_if_best_z_no_change_rounds: int = 50
    resource_cache_hit_stagnation_increment: float = 0.2
    resource_empty_candidate_stagnation_increment: float = 0.0
    resource_layer_fail_threshold: int = 3
    resource_layer_fail_multiplier: float = 0.1
    resource_layer_fail_cooldown: int = 10
    resource_adaptive_destroy_bonus_step: float = 0.05
    resource_adaptive_destroy_cache_hit_trigger: int = 3
    resource_adaptive_destroy_bonus_cap: float = 0.20
    resource_soft_greedy_topk: int = 3
    resource_soft_greedy_noise: float = 0.05
    resource_catastrophic_threshold_floor: float = 1.30
    resource_catastrophic_cv_scale: float = 3.0
    resource_rollback_budget_ratio: float = 0.15
    resource_z_window_size: int = 3
    resource_z_candidate_stack_topk: int = 5
    xz_evaluator_mode: str = "neural"
    enable_z_positive_mining_verify: bool = False
    z_positive_mining_verify_budget_base: int = 1
    z_positive_mining_verify_budget_zrich: int = 2
    z_positive_mining_proxy_margin_ratio: float = 0.12
    z_positive_mining_proxy_margin_abs: float = 10.0
    z_positive_mining_arrival_cap_ratio: float = 0.25
    z_positive_mining_wait_cap_ratio: float = 0.30
    z_positive_mining_tail_cap_ratio: float = 0.18
    z_positive_mining_route_gap_cap_ratio: float = 0.16
    z_positive_mining_min_hit_preservation_ratio: float = 0.75
    z_positive_mining_max_mode_delta: float = 1.0
    z_positive_mining_max_noise_tote_delta: float = 3.0
    z_positive_mining_route_gap_weight: float = 0.08
    z_positive_mining_arrival_weight: float = 0.10
    z_positive_mining_wait_weight: float = 0.12
    z_positive_mining_tail_weight: float = 0.08
    z_positive_mining_hit_bonus_weight: float = 0.06
    z_positive_mining_stack_locality_weight: float = 0.05
    z_positive_mining_demand_bonus_weight: float = 0.04
    z_positive_mining_preservation_bonus_weight: float = 0.10
    z_positive_mining_noise_penalty_weight: float = 0.06
    z_positive_mining_fallback_penalty_weight: float = 0.10
    z_positive_mining_mode_penalty_weight: float = 0.08
    enable_shadow_chain: bool = False
    shadow_chain_max_depth: int = 3
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
    x_eval_all_candidates: bool = False
    x_global_eval_topk: int = 2
    x_dual_eval_gap_ratio: float = 0.04
    x_f0_topk: int = 4
    x_f1_topk: int = 2
    x_f2_topk: int = 2
    x_uncertainty_probe_period: int = 3
    x_f1_station_overload_cap: float = 6.0
    x_f1_mapping_coverage_min: float = 0.35
    x_station_overload_soft_cap: float = 6.0
    x_station_overload_hard_cap: float = 45.0
    x_template_change_soft_cap: int = 2
    x_template_change_hard_cap: int = 5
    x_random_destroy_prob_small: float = 0.15
    x_repair_temperature: float = 0.20
    x_min_unique_candidates_per_round: int = 2
    x_changed_orders_cap: int = 2
    x_delta_subtask_cap: int = 1
    x_station_template_change_cap: int = 2
    x_robot_trip_template_change_cap: int = 2
    x_group_route_span_cap: float = 8.0
    x_group_completion_span_cap: float = 240.0
    x_disable_random_destroy_small: bool = True
    x_surrogate_station_weight: float = 6.0
    x_surrogate_arrival_weight: float = 10.0
    x_surrogate_route_weight: float = 4.0
    x_surrogate_affinity_weight: float = 0.75
    x_surrogate_finish_weight: float = 1.5
    x_surrogate_subtask_weight: float = 0.35
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
    y_load_skew_station_load_std_threshold: float = 2.0
    y_load_skew_station_load_ratio_threshold: float = 1.6
    y_load_skew_arrival_slack_mean_threshold: float = 180.0
    y_route_arrival_weight: float = 1.0
    y_route_late_task_weight: float = 8.0
    y_route_load_balance_weight: float = 0.5
    y_incremental_route_subtask_cap: int = 4
    y_incremental_route_trip_cap: int = 3
    y_route_sim_cache_size: int = 16
    enable_triggered_zu_budget: bool = True
    z_trigger_station_load_std: float = 1.5
    z_trigger_noise_ratio: float = 0.12
    z_trigger_multi_stack_pen: float = 0.5
    u_trigger_arrival_slack_mean: float = 60.0
    u_trigger_late_task_count: int = 1
    u_default_budget_when_triggered: int = 1
    u_aggressive_trigger_arrival_slack_mean: float = 180.0
    u_aggressive_trigger_late_task_count: int = 24
    u_slack_repair_arrival_slack_mean_threshold: float = 120.0
    u_slack_repair_late_task_count_threshold: int = 6
    u_slack_repair_extra_budget: int = 1
    z_budget_after_streak_reject: int = 1
    z_hotspot_batch_size: int = 3
    z_destroy_fraction: float = 0.30
    z_min_budget: int = 2
    z_structural_eval_topk: int = 2
    z_global_eval_topk: int = 1
    z_eval_all_candidates: bool = False
    z_full_global_eval_experiment: bool = False
    z_all_global_eval_default: bool = True
    z_micro_safe_ops_only: bool = False
    z_strict_safe_operator_semantics: bool = False
    z_generation_route_guard: bool = False
    z_repeat_reject_cache: bool = False
    z_hotspot_require_distinct_signature: bool = False
    z_operator_allowlist_experiment: Tuple[str, ...] = ()
    z_dual_eval_gap_ratio: float = 0.05
    z_f0_topk: int = 4
    z_f1_topk: int = 2
    z_f2_topk: int = 2
    z_uncertainty_probe_period: int = 3
    z_f1_trip_cap: int = 4
    z_f1_force_full_replay_threshold: int = 6
    z_hotspot_topk: int = 3
    z_route_gap_soft_cap: float = 25.0
    z_route_gap_hard_cap: float = 90.0
    z_local_delta_task_cap: int = 2
    z_local_delta_stack_cap: int = 1
    z_arrival_shift_soft_cap: float = 140.0
    z_arrival_shift_hard_cap: float = 180.0
    z_wait_overflow_soft_cap: float = 180.0
    z_wait_overflow_hard_cap: float = 240.0
    z_route_tail_soft_cap: float = 90.0
    z_route_tail_hard_cap: float = 120.0
    z_operator_subtask_ban_after_failures: int = 2
    z_operator_failure_decay_rounds: int = 4
    z_route_pressure_weight: float = 1.0
    z_station_load_weight: float = 0.75
    z_processing_overflow_weight: float = 1.0
    z_false_positive_streak_threshold: int = 2
    z_throttle_rounds: int = 2
    target_runtime_sec: float = 50.0
    runtime_guard_mode: str = "soft"
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
    surrogate_warmup_samples: int = 8
    surrogate_buffer_size: int = 512
    surrogate_ensemble_size: int = 3
    surrogate_rank_alpha: float = 1e-4
    surrogate_residual_alpha: float = 1e-4
    surrogate_uncertainty_k: float = 1.25
    surrogate_min_improve_abs: float = 1.0
    surrogate_mapping_bonus_weight: float = 5.0
    surrogate_uncertainty_floor_ratio: float = 0.05
    surrogate_warmup_uncertainty_cap_ratio: float = 0.08
    x_surrogate_bootstrap_eval_budget: int = 4
    z_surrogate_bootstrap_eval_budget: int = 4
    x_surrogate_trust_valid_samples: int = 2
    z_surrogate_trust_valid_samples: int = 2
    xz_hard_bad_margin: float = 320.0
    xz_f1_rebalance_rank_window: int = 2
    xz_f1_rebalance_station_window: int = 2
    x_station_overload_penalty_scale: float = 0.01
    x_station_overload_disaster_cap: float = 360.0
    x_micro_move_order_cap: int = 1
    x_micro_move_group_cap: int = 1
    x_bom_destroy_ratio: float = 0.15
    x_bom_destroy_max_lines: int = 2
    x_same_tote_bonus_weight: float = 1.5
    x_adjacent_stack_bonus_weight: float = 1.0
    x_route_span_penalty_weight: float = 1.0
    x_trust_region_penalty_weight: float = 1.0
    x_y_hotspot_bonus_weight: float = 0.5
    x_adjacent_stack_distance_cap: float = 1.0
    recent_y_accept_window: int = 2
    z_high_hit_bonus_weight: float = 1.0
    z_stack_locality_bonus_weight: float = 0.5
    z_congestion_top_quantile: float = 0.75
    z_mode_toggle_cap: int = 1
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
class ZStructuralEvalResult:
    objective_value: float
    sorting_cost_proxy: float
    coverage_gap: float
    multi_stack_penalty: float
    noise_ratio: float
    route_pressure_proxy: float
    station_load_std: float
    processing_overflow: float
    used_joint_repair: bool


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
    - 用近似 LB + 扰动做跳跃
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
        self.commit_anchor: Optional[SolutionSnapshot] = None
        self.shadow: Optional[SolutionSnapshot] = None
        self.anchor_z: float = float("inf")
        self.commit_anchor_z: float = float("inf")
        self.shadow_depth: int = 0
        self.shadow_last_layer: str = ""
        self.shadow_chain_layers: List[str] = []
        self.shadow_chain_head_candidate: Optional[Dict[str, Any]] = None
        self.last_shadow_chain_reset_reason: str = ""
        self.anchor_reference: Dict[str, Any] = {}
        self._z_detour_cache: Dict[int, float] = {}
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
        self._resolved_log_dir: Optional[str] = None
        self.layer_runtime_sec_by_name: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.layer_trial_count_by_name: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.global_eval_count: int = 0
        self.shadow_chain_commit_count: int = 0
        self.shadow_chain_rollback_count: int = 0
        self.operator_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.stagnation_stats: Dict[str, Dict[str, float]] = {}
        self.layer_operator_catalog: Dict[str, List[str]] = {}
        self.y_recent_signatures: Deque[str] = deque(maxlen=max(1, int(getattr(self.cfg, "y_recent_signature_window", 3))))
        self.layer_reject_surrogate_streak: Dict[str, int] = {}
        self.layer_global_reject_streak: Dict[str, int] = {}
        self.layer_fast_gate_reject_streak: Dict[str, int] = {}
        self.last_layer_accept: str = ""
        self.z_false_positive_streak: int = 0
        self.z_throttle_rounds_remaining: int = 0
        self.x_surrogate_state: Optional[SurrogateLayerState] = None
        self.z_surrogate_state: Optional[SurrogateLayerState] = None
        self.anchor_version: int = 0
        self.z_operator_subtask_failures: Dict[Tuple[str, int], int] = {}
        self.z_operator_subtask_bans: Set[Tuple[str, int]] = set()
        self.z_operator_subtask_failure_iter: Dict[Tuple[str, int], int] = {}
        self.z_signature_reject_cache: Set[str] = set()
        self.x_signature_reject_until: Dict[str, int] = {}
        self.recent_y_accept_age: int = 999999
        self.current_iter: int = 0
        self.supervised_candidate_dataset: Dict[str, List[Dict[str, Any]]] = {"X": [], "Z": []}
        self._reset_runtime_caches()

    # ----------------------------
    # 基础设施
    # ----------------------------
    def _set_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _current_search_scheme(self) -> str:
        return str(getattr(self.cfg, "search_scheme", "resource_time_alns") or "resource_time_alns").strip().lower()

    def _ensure_log_dir(self):
        if self._resolved_log_dir:
            os.makedirs(self._resolved_log_dir, exist_ok=True)
            return self._resolved_log_dir

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_log_dir = str(getattr(self.cfg, "log_dir", "log") or "log")
        if self._current_search_scheme() == "resource_time_alns":
            if cfg_log_dir.strip().lower() == "log":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join(
                    root,
                    "result",
                    f"resource_time_alns_{str(getattr(self.cfg, 'scale', '')).upper()}_{int(getattr(self.cfg, 'seed', -1))}_{timestamp}",
                )
            else:
                log_dir = cfg_log_dir if os.path.isabs(cfg_log_dir) else os.path.join(root, cfg_log_dir)
        else:
            log_dir = cfg_log_dir if os.path.isabs(cfg_log_dir) else os.path.join(root, cfg_log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self._resolved_log_dir = str(log_dir)
        return self._resolved_log_dir

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
        self.surrogate_stats: Dict[str, float] = {
            "evaluated": 0.0,
            "promoted": 0.0,
            "signature_skip": 0.0,
            "pruned": 0.0,
        }
        self.snapshot_time_sec = 0.0
        self.restore_time_sec = 0.0
        self.shadow_restore_time_sec = 0.0
        self.global_eval_time_sec = 0.0
        self.x_f1_time_sec = 0.0
        self.z_f1_time_sec = 0.0
        self.shadow_append_count = 0
        self.x_surrogate_bootstrap_eval_count = 0
        self.z_surrogate_bootstrap_eval_count = 0
        self.x_surrogate_probe_attempt_count = 0
        self.z_surrogate_probe_attempt_count = 0
        self.x_surrogate_probe_train_success_count = 0
        self.z_surrogate_probe_train_success_count = 0
        self.x_f1_invalid_count = 0
        self.z_f1_invalid_count = 0
        self.z_positive_mining_verify_count = 0
        self.z_positive_mining_success_count = 0
        self.lightweight_snapshot_count = 0
        self.heavy_snapshot_count = 0
        self.z_operator_subtask_failures = {}
        self.z_operator_subtask_bans = set()
        self.z_operator_subtask_failure_iter = {}
        self.z_signature_reject_cache = set()

    def _shadow_chain_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_shadow_chain", False))

    def _shadow_chain_depth_limit(self) -> int:
        if self._shadow_chain_enabled():
            return max(1, int(getattr(self.cfg, "shadow_chain_max_depth", 3)))
        return max(1, int(getattr(self.cfg, "max_shadow_layers_without_global", 4)))

    def _shadow_chain_layers_payload(self) -> List[str]:
        return [str(layer).upper() for layer in (self.shadow_chain_layers or []) if str(layer).strip()]

    def _xz_evaluator_mode(self) -> str:
        mode = str(getattr(self.cfg, "xz_evaluator_mode", "neural") or "neural").strip().lower()
        if mode not in {"neural", "classic_soft"}:
            return "neural"
        return mode


    def _is_zrich_scale(self) -> bool:
        return "ZRICH" in str(getattr(self.cfg, "scale", "") or "").upper()

    def _z_positive_mining_enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_z_positive_mining_verify", False))

    def _z_positive_mining_budget_limit(self) -> int:
        if not self._z_positive_mining_enabled():
            return 0
        if self._is_zrich_scale():
            return max(0, int(getattr(self.cfg, "z_positive_mining_verify_budget_zrich", 2)))
        return max(0, int(getattr(self.cfg, "z_positive_mining_verify_budget_base", 1)))

    def _z_positive_mining_budget_remaining(self) -> int:
        return max(0, int(self._z_positive_mining_budget_limit()) - int(getattr(self, "z_positive_mining_verify_count", 0)))

    def _z_positive_mining_allowlist(self) -> Set[str]:
        return {
            "z_hotspot_destroy_repair",
            "range_shrink_expand",
            "tote_replace_within_stack",
            "stack_replace",
        }

    def _safe_z_operator_sequence(
        self,
        operator_sequence: List[str],
        budget: int,
        priority_subtask_ids: Optional[List[int]] = None,
    ) -> List[str]:
        if int(budget) <= 0:
            return []
        base_sequence = [str(op) for op in (operator_sequence or []) if str(op).strip()]
        catalog = [str(op) for op in (self.layer_operator_catalog.get("Z", []) or []) if str(op).strip()]
        hotspot_active = bool(priority_subtask_ids) and "z_hotspot_destroy_repair" in catalog
        prefix = [
            "range_shrink_expand",
            "tote_replace_within_stack",
            "z_hotspot_destroy_repair" if hotspot_active else "stack_replace",
            "stack_replace" if hotspot_active else "z_hotspot_destroy_repair",
        ]
        experiment_allowlist = {
            str(op).strip()
            for op in (getattr(self.cfg, "z_operator_allowlist_experiment", ()) or ())
            if str(op).strip()
        }
        if bool(getattr(self.cfg, "z_micro_safe_ops_only", False)):
            allowed_ops = set(self._z_safe_operator_allowlist())
            if experiment_allowlist:
                allowed_ops |= experiment_allowlist
        else:
            allowed_ops = set(catalog)
            if self._z_positive_mining_enabled():
                allowed_ops &= set(self._z_positive_mining_allowlist())
            if experiment_allowlist:
                allowed_ops &= experiment_allowlist
        allowlist_experiment = {
            str(op).strip() for op in experiment_allowlist
        }
        merged: List[str] = []
        for op in prefix + base_sequence + catalog:
            if op not in allowed_ops:
                continue
            if op not in catalog or op in merged:
                continue
            merged.append(op)
        return merged[: max(0, int(budget))]

    def _z_positive_mining_eligibility(
        self,
        operator: str,
        score: Dict[str, Any],
        raw_proxy_z: float,
        route_gap_ratio: float,
        arrival_ratio: float,
        wait_ratio: float,
        tail_ratio: float,
    ) -> Tuple[bool, str]:
        if not self._z_positive_mining_enabled():
            return False, "disabled"
        if not bool(score.get("proposal_pass_fast_gate", False)):
            return False, "fast_gate_reject"
        operator = str(operator or score.get("z_move_type", ""))
        if operator not in self._z_positive_mining_allowlist():
            return False, "operator_not_allowlist"
        if bool(score.get("z_operator_fallback_used", False)):
            return False, "fallback_used"
        if not math.isfinite(float(raw_proxy_z)):
            return False, "nonfinite_proxy"
        anchor_scale = max(1.0, float(self.anchor_z if math.isfinite(self.anchor_z) else raw_proxy_z))
        proxy_margin = max(
            float(getattr(self.cfg, "z_positive_mining_proxy_margin_abs", 10.0)),
            anchor_scale * float(getattr(self.cfg, "z_positive_mining_proxy_margin_ratio", 0.12)),
        )
        if float(raw_proxy_z) > float(self.anchor_z) + proxy_margin + 1e-9:
            return False, "proxy_margin"
        if float(route_gap_ratio) > float(getattr(self.cfg, "z_positive_mining_route_gap_cap_ratio", 0.16)) + 1e-9:
            return False, "route_gap_cap"
        if float(arrival_ratio) > float(getattr(self.cfg, "z_positive_mining_arrival_cap_ratio", 0.25)) + 1e-9:
            return False, "arrival_cap"
        if float(wait_ratio) > float(getattr(self.cfg, "z_positive_mining_wait_cap_ratio", 0.30)) + 1e-9:
            return False, "wait_cap"
        if float(tail_ratio) > float(getattr(self.cfg, "z_positive_mining_tail_cap_ratio", 0.18)) + 1e-9:
            return False, "tail_cap"
        if abs(float(score.get("z_candidate_mode_delta", 0.0))) > float(getattr(self.cfg, "z_positive_mining_max_mode_delta", 1.0)) + 1e-9:
            return False, "mode_delta"
        if float(score.get("hit_tote_preservation_ratio", 1.0)) + 1e-9 < float(getattr(self.cfg, "z_positive_mining_min_hit_preservation_ratio", 0.75)):
            return False, "hit_preservation"
        if float(score.get("noise_tote_delta", 0.0)) > float(getattr(self.cfg, "z_positive_mining_max_noise_tote_delta", 3.0)) + 1e-9:
            return False, "noise_tote_delta"
        return True, ""

    def _score_z_positive_mining_candidate(
        self,
        operator: str,
        score: Dict[str, Any],
        f1_result: F1EvalResult,
    ) -> Dict[str, Any]:
        anchor_scale = max(1.0, float(self.anchor_z if math.isfinite(self.anchor_z) else f1_result.proxy_z))
        raw_proxy_z = float((f1_result.extra or {}).get("raw_post_y_proxy_z", f1_result.proxy_z))
        route_gap_ratio = max(0.0, float(score.get("z_route_gap_penalty", 0.0))) / anchor_scale
        arrival_ratio = max(
            0.0,
            max(float(score.get("z_arrival_shift_estimate", 0.0)), float(f1_result.arrival_shift_total)),
        ) / anchor_scale
        wait_ratio = max(
            0.0,
            max(float(score.get("z_wait_overflow_estimate", 0.0)), float(f1_result.wait_overflow_total)),
        ) / anchor_scale
        tail_ratio = max(
            0.0,
            max(float(score.get("z_route_tail_delta_estimate", 0.0)), float(f1_result.route_tail_delta)),
        ) / anchor_scale
        capped_route_gap = min(1.0, route_gap_ratio / max(1e-9, float(getattr(self.cfg, "z_positive_mining_route_gap_cap_ratio", 0.16))))
        capped_arrival = min(1.0, arrival_ratio / max(1e-9, float(getattr(self.cfg, "z_positive_mining_arrival_cap_ratio", 0.25))))
        capped_wait = min(1.0, wait_ratio / max(1e-9, float(getattr(self.cfg, "z_positive_mining_wait_cap_ratio", 0.30))))
        capped_tail = min(1.0, tail_ratio / max(1e-9, float(getattr(self.cfg, "z_positive_mining_tail_cap_ratio", 0.18))))
        risk_penalty = anchor_scale * (
            float(getattr(self.cfg, "z_positive_mining_route_gap_weight", 0.08)) * capped_route_gap
            + float(getattr(self.cfg, "z_positive_mining_arrival_weight", 0.10)) * capped_arrival
            + float(getattr(self.cfg, "z_positive_mining_wait_weight", 0.12)) * capped_wait
            + float(getattr(self.cfg, "z_positive_mining_tail_weight", 0.08)) * capped_tail
        )
        hit_bonus = min(1.0, max(0.0, float(score.get("z_hit_frequency_bonus", 0.0))))
        stack_locality = min(1.0, max(0.0, float(score.get("stack_locality_score", score.get("z_stack_locality_score", 0.0)))))
        demand_ratio = min(1.0, max(0.0, float(score.get("z_demand_ratio", 0.0))))
        preservation_ratio = min(1.0, max(0.0, float(score.get("hit_tote_preservation_ratio", 1.0))))
        preservation_floor = float(getattr(self.cfg, "z_positive_mining_min_hit_preservation_ratio", 0.75))
        preservation_bonus = min(1.0, max(0.0, (preservation_ratio - preservation_floor) / max(0.05, 1.0 - preservation_floor)))
        reward_bonus = anchor_scale * (
            float(getattr(self.cfg, "z_positive_mining_hit_bonus_weight", 0.06)) * hit_bonus
            + float(getattr(self.cfg, "z_positive_mining_stack_locality_weight", 0.05)) * stack_locality
            + float(getattr(self.cfg, "z_positive_mining_demand_bonus_weight", 0.04)) * demand_ratio
            + float(getattr(self.cfg, "z_positive_mining_preservation_bonus_weight", 0.10)) * preservation_bonus
        )
        noise_ratio = min(
            1.0,
            max(0.0, float(score.get("noise_tote_delta", 0.0)))
            / max(1.0, float(getattr(self.cfg, "z_positive_mining_max_noise_tote_delta", 3.0))),
        )
        mode_ratio = min(
            1.0,
            abs(float(score.get("z_candidate_mode_delta", 0.0)))
            / max(1.0, float(getattr(self.cfg, "z_positive_mining_max_mode_delta", 1.0))),
        )
        fallback_ratio = 1.0 if bool(score.get("z_operator_fallback_used", False)) else 0.0
        structure_penalty = anchor_scale * (
            float(getattr(self.cfg, "z_positive_mining_noise_penalty_weight", 0.06)) * noise_ratio
            + float(getattr(self.cfg, "z_positive_mining_fallback_penalty_weight", 0.10)) * fallback_ratio
            + float(getattr(self.cfg, "z_positive_mining_mode_penalty_weight", 0.08)) * mode_ratio
        )
        eligible, eligible_reason = self._z_positive_mining_eligibility(
            operator=operator,
            score=score,
            raw_proxy_z=raw_proxy_z,
            route_gap_ratio=route_gap_ratio,
            arrival_ratio=arrival_ratio,
            wait_ratio=wait_ratio,
            tail_ratio=tail_ratio,
        )
        mining_score = float(raw_proxy_z + risk_penalty + structure_penalty - reward_bonus)
        return {
            "z_positive_mining_triggered": False,
            "z_positive_mining_score": float(mining_score),
            "z_positive_candidate_eligible": bool(eligible),
            "z_positive_candidate_eligibility_reason": str(eligible_reason),
            "z_positive_candidate_operator": str(operator),
            "z_positive_mining_raw_proxy_z": float(raw_proxy_z),
            "z_positive_mining_route_gap_ratio": float(route_gap_ratio),
            "z_positive_mining_arrival_ratio": float(arrival_ratio),
            "z_positive_mining_wait_ratio": float(wait_ratio),
            "z_positive_mining_tail_ratio": float(tail_ratio),
            "z_positive_mining_risk_penalty": float(risk_penalty),
            "z_positive_mining_reward_bonus": float(reward_bonus),
            "z_positive_mining_structure_penalty": float(structure_penalty),
            "z_positive_mining_hit_bonus": float(hit_bonus),
            "z_positive_mining_stack_locality": float(stack_locality),
            "z_positive_mining_demand_ratio": float(demand_ratio),
            "z_positive_mining_hit_preservation_ratio": float(preservation_ratio),
            "z_positive_mining_noise_tote_delta": float(score.get("noise_tote_delta", 0.0)),
            "z_positive_mining_mode_delta": float(score.get("z_candidate_mode_delta", 0.0)),
            "z_positive_mining_fallback_penalty": float(fallback_ratio * anchor_scale * float(getattr(self.cfg, "z_positive_mining_fallback_penalty_weight", 0.10))),
            "z_positive_mining_noise_penalty": float(noise_ratio * anchor_scale * float(getattr(self.cfg, "z_positive_mining_noise_penalty_weight", 0.06))),
            "z_positive_mining_mode_penalty": float(mode_ratio * anchor_scale * float(getattr(self.cfg, "z_positive_mining_mode_penalty_weight", 0.08))),
        }









    def _build_surrogate_probe_candidate(
        self,
        layer: str,
        best_snap: Optional["SolutionSnapshot"],
        best_score: Optional[Dict[str, Any]],
        runtime: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        layer = str(layer).upper()
        if layer not in {"X", "Z"} or best_snap is None or not self._shadow_chain_enabled():
            return None
        if not self._surrogate_warmup_active(layer):
            return None
        if self._surrogate_bootstrap_budget_remaining(layer) <= 0:
            return None
        score = dict(best_score or {})
        eval_items = list(score.get(f"_{layer.lower()}_eval_candidates", []) or [])
        if eval_items:
            probe = copy.deepcopy(eval_items[0])
        else:
            probe = {
                "snapshot": best_snap,
                "score": score,
                "operator": str(runtime.get("selected_operator", "")),
                "operator_rank": int(runtime.get("selected_operator_rank", 0.0)),
                "rank": int(runtime.get("selected_candidate_rank", 0.0)),
                "meta": {},
            }
        probe["origin_layer"] = layer
        if probe.get("snapshot") is None:
            probe["snapshot"] = best_snap
        probe["score"] = dict(probe.get("score", {}) or score)
        probe["operator"] = str(probe.get("operator", runtime.get("selected_operator", "")))
        probe["operator_rank"] = int(probe.get("operator_rank", runtime.get("selected_operator_rank", 0.0)))
        probe["rank"] = int(probe.get("rank", runtime.get("selected_candidate_rank", 0.0)))
        probe["meta"] = dict(probe.get("meta", {}) or {})
        return probe


    def _shadow_chain_reset(self, reason: str = ""):
        self.shadow_depth = 0
        self.shadow_last_layer = ""
        self.shadow_chain_layers = []
        self.shadow_chain_head_candidate = None
        if reason:
            self.last_shadow_chain_reset_reason = str(reason)

    def _set_committed_anchor(self, snap: SolutionSnapshot, z: float):
        self.commit_anchor = snap
        self.commit_anchor_z = float(z)
        self.anchor = snap
        self.anchor_z = float(z)

    def _append_shadow_head(self, layer: str, snap: SolutionSnapshot):
        layer = str(layer).upper()
        self.anchor = snap
        self.shadow = snap
        self.shadow_last_layer = layer
        self.shadow_chain_layers.append(layer)
        self.shadow_depth = int(len(self.shadow_chain_layers))
        self.shadow_append_count = int(getattr(self, "shadow_append_count", 0)) + 1
        self.x_signature_reject_until = {}

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

    def _refresh_runtime_cache(self, z: Optional[float] = None):
        self.cached_eval = None if z is None else float(z)
        self.cached_metrics = dict(self._collect_layer_metrics())
        self._clear_z_detour_cache()
        self._clear_dirty()

    def _clear_z_detour_cache(self) -> None:
        self._z_detour_cache = {}

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


    def _reset_xz_surrogate_states(self):
        self.anchor_version = 0
        self.x_surrogate_state = self._create_surrogate_layer_state("X")
        self.z_surrogate_state = self._create_surrogate_layer_state("Z")



    def _runtime_elapsed_sec(self) -> float:
        if self.run_start_time_sec <= 0.0:
            return 0.0
        return float(max(0.0, time.perf_counter() - self.run_start_time_sec))

    def _resource_time_run_stats_payload(self) -> Dict[str, Any]:
        iter_rows = list(getattr(self, "iter_log", []) or [])
        validation_trigger_counts: Dict[str, int] = {}
        layer_selected_count_by_name = {str(name): 0 for name in self.layer_names}
        layer_accepted_count_by_name = {str(name): 0 for name in self.layer_names}
        for row in iter_rows:
            selected_layer = str(row.get("selected_resource_layer", row.get("focus", "")) or "").upper()
            if selected_layer in layer_selected_count_by_name:
                layer_selected_count_by_name[selected_layer] = int(layer_selected_count_by_name[selected_layer]) + 1
                if bool(row.get("local_accept", False)):
                    layer_accepted_count_by_name[selected_layer] = int(layer_accepted_count_by_name[selected_layer]) + 1
            trigger = str(row.get("validation_trigger", "") or "")
            if not trigger:
                continue
            validation_trigger_counts[trigger] = int(validation_trigger_counts.get(trigger, 0)) + 1
        return {
            "search_scheme": "resource_time_alns",
            "result_root": self._ensure_log_dir(),
            "run_start_time_sec": float(self.run_start_time_sec),
            "run_total_time_sec": float(self.run_total_time_sec if self.run_total_time_sec > 0.0 else self._runtime_elapsed_sec()),
            "layer_runtime_sec_by_name": {str(k): float(v) for k, v in self.layer_runtime_sec_by_name.items()},
            "layer_trial_count_by_name": {str(k): float(v) for k, v in self.layer_trial_count_by_name.items()},
            "layer_selected_count_by_name": layer_selected_count_by_name,
            "layer_accepted_count_by_name": layer_accepted_count_by_name,
            "global_eval_count": int(self.global_eval_count),
            "best_validated_makespan": float(self.best.z) if self.best is not None else float("nan"),
            "lkh_call_count": int(max((int(row.get("lkh_call_count", 0) or 0) for row in iter_rows), default=0)),
            "lkh_budget_consumed_by_rollback": int(max((int(row.get("lkh_budget_consumed_by_rollback", 0) or 0) for row in iter_rows), default=0)),
            "fallback_count": int(sum(1 for row in iter_rows if bool(row.get("fallback_repair_used", False)))),
            "catastrophic_rollback_count": int(sum(1 for row in iter_rows if bool(row.get("catastrophic_rollback", False)))),
            "force_rotate_count": int(sum(1 for row in iter_rows if bool(row.get("force_rotate_used", False)))),
            "heavy_destroy_count": int(sum(1 for row in iter_rows if bool(row.get("heavy_destroy_active", False)))),
            "candidate_hard_reject_count": int(sum(1 for row in iter_rows if str(row.get("candidate_hard_reject_reason", "")).strip())),
            "empty_candidate_penalized_count": int(sum(1 for row in iter_rows if bool(row.get("empty_candidate_penalized", False)))),
            "coverage_hard_reject_count": int(getattr(self, "coverage_hard_reject_count", sum(1 for row in iter_rows if bool(row.get("coverage_hard_reject", False))))),
            "exact_eval_cache_hit_count": int(max((int(row.get("exact_eval_cache_hit_count", 0) or 0) for row in iter_rows), default=0)),
            "x_failure_decapitation_count": int(getattr(self, "x_failure_decapitation_count", 0)),
            "stop_reason": str(getattr(self, "stop_reason", "") or ""),
            "resource_real_eval_period": int(getattr(self.cfg, "resource_real_eval_period", 8)),
            "validation_trigger_counts": dict(sorted(validation_trigger_counts.items())),
            "operator_stats": copy.deepcopy(getattr(self, "operator_stats", {}) or {}),
            "timing_breakdown": self._timing_breakdown_payload(),
        }

    def _runtime_stats_payload(self) -> Dict[str, Any]:
        if self._current_search_scheme() == "resource_time_alns":
            return self._resource_time_run_stats_payload()

        reward_summary: Dict[str, Dict[str, float]] = {}
        for layer, rows in (self.operator_stats or {}).items():
            reward_summary[str(layer)] = {
                str(op): float(meta.get("reward_mean", 0.0)) for op, meta in rows.items()
            }
        iter_rows = list(getattr(self, "iter_log", []) or [])

        def _row_eval_origin(row: Dict[str, Any]) -> str:
            reason = str(row.get("global_eval_reason", ""))
            if reason.startswith("forced:"):
                return str(row.get("forced_eval_origin_layer", row.get("focus", ""))).upper()
            return str(row.get("focus", "")).upper()

        timing_breakdown = self._timing_breakdown_payload()
        z_positive_rows = [row for row in list(self.supervised_candidate_dataset.get("Z", []) or []) if int(row.get("win_label", 0) or 0) > 0]
        z_positive_operator_mix: Dict[str, int] = {}
        for row in z_positive_rows:
            operator = str(row.get("operator", "")).strip()
            if not operator:
                continue
            z_positive_operator_mix[operator] = int(z_positive_operator_mix.get(operator, 0)) + 1
        return {
            "run_start_time_sec": float(self.run_start_time_sec),
            "run_total_time_sec": float(self.run_total_time_sec if self.run_total_time_sec > 0.0 else self._runtime_elapsed_sec()),
            "xz_evaluator_mode": str(self._xz_evaluator_mode()),
            "layer_runtime_sec_by_name": {str(k): float(v) for k, v in self.layer_runtime_sec_by_name.items()},
            "layer_trial_count_by_name": {str(k): float(v) for k, v in self.layer_trial_count_by_name.items()},
            "global_eval_count": int(self.global_eval_count),
            "shadow_chain_enabled": bool(self._shadow_chain_enabled()),
            "shadow_chain_max_depth": int(self._shadow_chain_depth_limit()),
            "shadow_chain_commit_count": int(getattr(self, "shadow_chain_commit_count", 0)),
            "shadow_chain_rollback_count": int(getattr(self, "shadow_chain_rollback_count", 0)),
            "shadow_append_count": int(getattr(self, "shadow_append_count", 0)),
            "shadow_depth": int(getattr(self, "shadow_depth", 0)),
            "shadow_chain_layers": list(self._shadow_chain_layers_payload()),
            "x_probe_global_eval_count": int(sum(
                int(row.get("global_eval_candidate_count", 0) or 0)
                for row in iter_rows
                if bool(row.get("global_eval_triggered", False))
                and str(row.get("global_eval_reason", "")) == "surrogate_probe"
                and _row_eval_origin(row) == "X"
            )),
            "z_probe_global_eval_count": int(sum(
                int(row.get("global_eval_candidate_count", 0) or 0)
                for row in iter_rows
                if bool(row.get("global_eval_triggered", False))
                and str(row.get("global_eval_reason", "")) == "surrogate_probe"
                and _row_eval_origin(row) == "Z"
            )),
            "classic_verify_count_x": int(sum(
                int(row.get("global_eval_candidate_count", 0) or 0)
                for row in iter_rows
                if bool(row.get("global_eval_triggered", False))
                and str(row.get("global_eval_reason", "")) == "classic_soft_verify"
                and _row_eval_origin(row) == "X"
            )),
            "classic_verify_count_z": int(sum(
                int(row.get("global_eval_candidate_count", 0) or 0)
                for row in iter_rows
                if bool(row.get("global_eval_triggered", False))
                and str(row.get("global_eval_reason", "")) == "classic_soft_verify"
                and _row_eval_origin(row) == "Z"
            )),
            "z_positive_mining_verify_count": int(sum(
                int(row.get("global_eval_candidate_count", 0) or 0)
                for row in iter_rows
                if bool(row.get("global_eval_triggered", False))
                and str(row.get("global_eval_reason", "")) == "z_positive_mining_verify"
                and _row_eval_origin(row) == "Z"
            )),
            "x_surrogate_positive_count": int(sum(
                1 for row in iter_rows
                if bool(self._xz_uses_neural_evaluator())
                and str(row.get("focus", "")).upper() == "X"
                and bool(row.get("proposal_pass_surrogate", False))
            )),
            "z_surrogate_positive_count": int(sum(
                1 for row in iter_rows
                if bool(self._xz_uses_neural_evaluator())
                and str(row.get("focus", "")).upper() == "Z"
                and bool(row.get("proposal_pass_surrogate", False))
            )),
            "x_valid_surrogate_samples_seen": int(self._surrogate_valid_sample_count("X")),
            "z_valid_surrogate_samples_seen": int(self._surrogate_valid_sample_count("Z")),
            "x_surrogate_warmup_samples_seen": int(self._surrogate_state("X").warmup_count),
            "z_surrogate_warmup_samples_seen": int(self._surrogate_state("Z").warmup_count),
            "x_surrogate_trusted": bool(self._surrogate_trusted("X")),
            "z_surrogate_trusted": bool(self._surrogate_trusted("Z")),
            "x_surrogate_bootstrap_eval_count": int(getattr(self, "x_surrogate_bootstrap_eval_count", 0)),
            "z_surrogate_bootstrap_eval_count": int(getattr(self, "z_surrogate_bootstrap_eval_count", 0)),
            "x_surrogate_probe_attempt_count": int(getattr(self, "x_surrogate_probe_attempt_count", 0)),
            "z_surrogate_probe_attempt_count": int(getattr(self, "z_surrogate_probe_attempt_count", 0)),
            "x_surrogate_probe_train_success_count": int(getattr(self, "x_surrogate_probe_train_success_count", 0)),
            "z_surrogate_probe_train_success_count": int(getattr(self, "z_surrogate_probe_train_success_count", 0)),
            "x_f1_invalid_count": int(getattr(self, "x_f1_invalid_count", 0)),
            "z_f1_invalid_count": int(getattr(self, "z_f1_invalid_count", 0)),
            "z_positive_mining_success_count": int(getattr(self, "z_positive_mining_success_count", 0)),
            "z_positive_candidate_eligible_count": int(sum(
                int(float(row.get("z_positive_candidate_eligible_count", 0.0) or 0.0))
                for row in iter_rows
                if str(row.get("focus", "")).upper() == "Z"
            )),
            "z_row_count": int(len(list(self.supervised_candidate_dataset.get("Z", []) or []))),
            "z_positive_row_count": int(len(z_positive_rows)),
            "z_positive_operator_mix": dict(sorted(z_positive_operator_mix.items())),
            "top1_x_proxy_improve_but_not_verified_count": int(sum(
                1 for row in iter_rows
                if bool(row.get("top1_x_proxy_improve_but_not_verified", False))
            )),
            "simulate_call_count": int(self._simulate_call_count),
            "snapshot_time_sec": float(getattr(self, "snapshot_time_sec", 0.0)),
            "restore_time_sec": float(getattr(self, "restore_time_sec", 0.0)),
            "shadow_restore_time_sec": float(getattr(self, "shadow_restore_time_sec", 0.0)),
            "global_eval_time_sec": float(getattr(self, "global_eval_time_sec", 0.0)),
            "x_f1_time_sec": float(getattr(self, "x_f1_time_sec", 0.0)),
            "z_f1_time_sec": float(getattr(self, "z_f1_time_sec", 0.0)),
            "forced_global_eval_time_sec": float(timing_breakdown.get("forced_global_eval_time_sec", 0.0)),
            "lightweight_snapshot_count": int(getattr(self, "lightweight_snapshot_count", 0)),
            "heavy_snapshot_count": int(getattr(self, "heavy_snapshot_count", 0)),
            "layer_augmented_acceptance_mode": str(getattr(self.cfg, "acceptance_mode", "strict_global")),
            "operator_selection_mode": str(getattr(self.cfg, "operator_selection_mode", "ucb1")),
            "operator_reward_mean_by_layer": reward_summary,
            "timing_breakdown": timing_breakdown,
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

    def _timing_breakdown_payload(self) -> Dict[str, Any]:
        iter_rows = list(getattr(self, "iter_log", []) or [])
        wall_time_sec = float(self.run_total_time_sec if self.run_total_time_sec > 0.0 else self._runtime_elapsed_sec())
        layer_runtime_total = {str(k): float(v) for k, v in self.layer_runtime_sec_by_name.items()}
        local_vns_time_sec_by_layer: Dict[str, float] = {}
        for layer in self.layer_names:
            layer_rows = [row for row in iter_rows if str(row.get("focus", "")).upper() == str(layer).upper()]
            exclusive = 0.0
            for row in layer_rows:
                exclusive += max(
                    0.0,
                    float(row.get("iter_runtime_sec", 0.0) or 0.0)
                    - float(row.get("global_eval_time_sec", 0.0) or 0.0)
                    - float(row.get("x_f1_time_sec", 0.0) or 0.0)
                    - float(row.get("z_f1_time_sec", 0.0) or 0.0)
                    - float(row.get("shadow_restore_time_sec", 0.0) or 0.0),
                )
            local_vns_time_sec_by_layer[str(layer)] = float(exclusive)

        forced_global_eval_time_sec = float(sum(
            float(row.get("global_eval_time_sec", 0.0) or 0.0)
            for row in iter_rows
            if str(row.get("global_eval_reason", "")).startswith("forced:")
        ))
        global_eval_time_sec = float(getattr(self, "global_eval_time_sec", 0.0))
        x_f1_time_sec = float(getattr(self, "x_f1_time_sec", 0.0))
        z_f1_time_sec = float(getattr(self, "z_f1_time_sec", 0.0))
        snapshot_time_sec = float(getattr(self, "snapshot_time_sec", 0.0))
        restore_time_sec = float(getattr(self, "restore_time_sec", 0.0))
        shadow_restore_time_sec = float(getattr(self, "shadow_restore_time_sec", 0.0))
        exclusive_bucket_total_sec = float(sum(local_vns_time_sec_by_layer.values()) + x_f1_time_sec + z_f1_time_sec + global_eval_time_sec)
        reconciliation_gap_sec = float(wall_time_sec - exclusive_bucket_total_sec)

        return {
            "wall_time_sec": wall_time_sec,
            "layer_runtime_total_sec_by_name": layer_runtime_total,
            "local_vns_time_sec_by_layer": local_vns_time_sec_by_layer,
            "x_f1_time_sec": x_f1_time_sec,
            "z_f1_time_sec": z_f1_time_sec,
            "global_eval_time_sec": global_eval_time_sec,
            "forced_global_eval_time_sec": forced_global_eval_time_sec,
            "global_eval_sp2_time_sec": float(sum(float(row.get("global_eval_sp2_time_sec", 0.0) or 0.0) for row in iter_rows)),
            "global_eval_sp3_time_sec": float(sum(float(row.get("global_eval_sp3_time_sec", 0.0) or 0.0) for row in iter_rows)),
            "global_eval_sp4_time_sec": float(sum(float(row.get("global_eval_sp4_time_sec", 0.0) or 0.0) for row in iter_rows)),
            "snapshot_time_sec": snapshot_time_sec,
            "restore_time_sec": restore_time_sec,
            "shadow_restore_time_sec": shadow_restore_time_sec,
            "snapshot_restore_overhead_sec": float(snapshot_time_sec + restore_time_sec),
            "shadow_append_count": int(getattr(self, "shadow_append_count", 0)),
            "exclusive_bucket_total_sec": exclusive_bucket_total_sec,
            "reconciliation_gap_vs_wall_sec": reconciliation_gap_sec,
        }

    def _dataset_scale_id(self, scale_name: str) -> int:
        ordered = [
            "SMALL",
            "SMALL2",
            "SMALL_ZRICH",
            "SMALL2_ZRICH",
            "SMALL3",
            "SMALL_UNEVEN",
            "SMALL2_UNEVEN",
            "SMALL3_UNEVEN",
            "MEDIUM",
            "LARGE",
        ]
        scale_upper = str(scale_name).upper()
        if scale_upper in ordered:
            return int(ordered.index(scale_upper))
        return int(len(ordered))

    def _infer_x_dataset_subtask_id(self, proposal: Optional[XSplitProposal]) -> int:
        if proposal is None:
            return -1
        order_id = int(getattr(proposal, "x_destroy_order_id", -1))
        moved_sku_ids = [
            int(sku_id)
            for sku_id in (getattr(proposal, "x_directly_moved_sku_ids", []) or [])
            if int(sku_id) >= 0
        ]
        if order_id < 0 or not moved_sku_ids:
            return -1
        anchor_sku_profile = self.anchor_reference.get("anchor_sku_profile", {}) or {}
        subtask_hits: Dict[int, int] = defaultdict(int)
        for sku_id in moved_sku_ids:
            subtask_id = int(anchor_sku_profile.get((order_id, sku_id), {}).get("anchor_subtask_id", -1))
            if subtask_id >= 0:
                subtask_hits[subtask_id] += 1
        if not subtask_hits:
            return -1
        return int(min(subtask_hits.items(), key=lambda item: (-item[1], item[0]))[0])

    def _build_dataset_context_features(self, metrics: Optional[Dict[str, float]], global_z_before: float) -> Dict[str, float]:
        metrics = dict(metrics or {})
        task_count = float(len(self._collect_all_tasks())) if self.problem is not None else 0.0
        warehouse_scale = float(self._warehouse_distance_scale())
        anchor_time_scale = max(
            1.0,
            float(global_z_before) if math.isfinite(float(global_z_before)) else 0.0,
            float(metrics.get("global_makespan", 0.0)),
        )
        path_scale = max(1.0, warehouse_scale * max(1.0, task_count))
        avg_stack_span = float(metrics.get("avg_stack_span", 0.0))
        arrival_slack_mean = float(metrics.get("arrival_slack_mean", 0.0))
        robot_path_length_total = float(metrics.get("robot_path_length_total", 0.0))
        latest_robot_finish = float(metrics.get("latest_robot_finish", 0.0))
        return {
            "ctx_scale_id": float(self._dataset_scale_id(getattr(self.cfg, "scale", ""))),
            "ctx_anchor_z": float(global_z_before),
            "ctx_subtask_count": float(metrics.get("subtask_count", 0.0)),
            "ctx_task_count": float(task_count),
            "ctx_avg_sku_per_subtask": float(metrics.get("avg_sku_per_subtask", 0.0)),
            "ctx_station_load_std": float(metrics.get("station_load_std", 0.0)),
            "ctx_station_load_max_ratio": float(metrics.get("station_load_max_ratio", 0.0)),
            "ctx_noise_ratio": float(metrics.get("noise_ratio", 0.0)),
            "ctx_avg_stack_span": float(avg_stack_span),
            "ctx_arrival_slack_mean": float(arrival_slack_mean),
            "ctx_robot_path_length_total": float(robot_path_length_total),
            "ctx_latest_robot_finish": float(latest_robot_finish),
            "ctx_avg_stack_span_norm": float(avg_stack_span / max(1.0, warehouse_scale)),
            "ctx_arrival_slack_mean_norm": float(arrival_slack_mean / anchor_time_scale),
            "ctx_robot_path_length_total_norm": float(robot_path_length_total / path_scale),
            "ctx_latest_robot_finish_norm": float(latest_robot_finish / anchor_time_scale),
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
        runtime_guard = self._runtime_guard_level()
        metrics = self._collect_layer_metrics() if self.problem is not None else {}
        late_task_count = self._compute_late_task_count() if self.problem is not None else 0
        y_skew_ctx = self._y_load_skew_context(metrics) if self.problem is not None else {"enabled": False, "reason": "", "metrics": {}}
        u_slack_ctx = self._u_slack_repair_context(metrics, late_task_count) if self.problem is not None else {"enabled": False, "reason": "", "late_task_count": 0, "metrics": {}}
        z_throttled = self._is_z_throttled()
        if runtime_guard >= 2:
            if layer == "X":
                base_budget = min(base_budget, 1)
            elif layer == "Y":
                base_budget = min(base_budget, 3)
            elif layer == "Z":
                base_budget = min(base_budget, 1)
            elif layer == "U":
                base_budget = 0
        elif runtime_guard >= 1:
            if layer == "X":
                base_budget = min(base_budget, 2)
            elif layer == "Y":
                base_budget = min(base_budget, 4)
            elif layer == "Z":
                base_budget = min(base_budget, 2)
            elif layer == "U":
                base_budget = min(base_budget, 1)
        if layer == "Y":
            if bool(y_skew_ctx.get("enabled", False)):
                base_budget += 1
            if z_throttled:
                base_budget += 1
        if bool(getattr(self.cfg, "enable_triggered_zu_budget", True)):
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
                    "throttled": bool(z_throttled),
                }
                if not trigger_reasons:
                    return 0
                if z_throttled:
                    return 1
                if int(self.layer_reject_surrogate_streak.get("Z", 0)) >= 2:
                    return max(
                        int(getattr(self.cfg, "z_min_budget", 2)),
                        int(getattr(self.cfg, "z_budget_after_streak_reject", 1)),
                    )
                if float(self.stagnation_stats.get("Z", {}).get("restart_triggered", 0.0)) > 0.0:
                    return max(int(getattr(self.cfg, "z_min_budget", 2)), min(base_budget, 3))
            if layer == "U":
                trigger_reasons = []
                if float(metrics.get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "u_trigger_arrival_slack_mean", 60.0)):
                    trigger_reasons.append("arrival_slack_mean")
                if int(late_task_count) >= int(getattr(self.cfg, "u_trigger_late_task_count", 1)):
                    trigger_reasons.append("late_task_count")
                if self._recent_y_accept_active():
                    trigger_reasons.append("recent_y_accept")
                self.current_trigger_gate = {
                    "layer": "U",
                    "open": bool(trigger_reasons),
                    "reason": ",".join(trigger_reasons) if trigger_reasons else "below_threshold",
                    "late_task_count": int(late_task_count),
                    "slack_repair_mode": bool(u_slack_ctx.get("enabled", False)),
                    "slack_repair_reason": str(u_slack_ctx.get("reason", "")),
                    "budget_boosted": False,
                    "recent_y_accept_active": bool(self._recent_y_accept_active()),
                    "recent_y_accept_age": int(getattr(self, "recent_y_accept_age", 999999)),
                }
                if not trigger_reasons:
                    return 0
                strong_trigger = (
                    float(metrics.get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "u_aggressive_trigger_arrival_slack_mean", 180.0))
                    or int(late_task_count) >= int(getattr(self.cfg, "u_aggressive_trigger_late_task_count", 24))
                )
                self.current_trigger_gate["strong"] = bool(strong_trigger)
                if not self._recent_y_accept_active() and not strong_trigger and not bool(u_slack_ctx.get("enabled", False)):
                    return 0
                desired_budget = max(1, int(getattr(self.cfg, "u_default_budget_when_triggered", 1)))
                if bool(u_slack_ctx.get("enabled", False)):
                    desired_budget = max(desired_budget, 1)
                    if z_throttled and runtime_guard < 2:
                        desired_budget = max(
                            desired_budget,
                            1 + max(0, int(getattr(self.cfg, "u_slack_repair_extra_budget", 1))),
                        )
                        self.current_trigger_gate["budget_boosted"] = bool(desired_budget > int(getattr(self.cfg, "u_default_budget_when_triggered", 1)))
                budget_cap = max(0, int(base_budget))
                if bool(u_slack_ctx.get("enabled", False)):
                    budget_cap = max(budget_cap, 1)
                return min(budget_cap, desired_budget)
        return base_budget

    def _runtime_guard_level(self) -> int:
        if str(getattr(self.cfg, "runtime_guard_mode", "soft")).lower() != "soft":
            return 0
        target = float(getattr(self.cfg, "target_runtime_sec", 0.0) or 0.0)
        if target <= 0.0:
            return 0
        elapsed = float(self._runtime_elapsed_sec())
        if elapsed >= target:
            return 2
        if elapsed >= 0.7 * target:
            return 1
        return 0

    def _is_z_throttled(self) -> bool:
        return int(getattr(self, "z_throttle_rounds_remaining", 0)) > 0

    def _recent_y_accept_active(self) -> bool:
        return int(getattr(self, "recent_y_accept_age", 999999)) <= max(0, int(getattr(self.cfg, "recent_y_accept_window", 2)))

    def _current_station_subtask_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = defaultdict(int)
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid >= 0:
                counts[sid] += 1
        return dict(counts)

    def _anchor_subtask_station_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for st in self._iter_snapshot_subtasks(self.anchor):
            mapping[int(getattr(st, "id", -1))] = int(getattr(st, "assigned_station_id", -1))
        return mapping

    def _anchor_subtask_rank_map(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for st in self._iter_snapshot_subtasks(self.anchor):
            mapping[int(getattr(st, "id", -1))] = int(getattr(st, "station_sequence_rank", -1))
        return mapping

    def _y_load_skew_context(self, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        metrics = dict(metrics or self._collect_layer_metrics())
        reasons: List[str] = []
        if float(metrics.get("station_load_std", 0.0)) >= float(getattr(self.cfg, "y_load_skew_station_load_std_threshold", 2.0)):
            reasons.append("station_load_std")
        if float(metrics.get("station_load_max_ratio", 0.0)) >= float(getattr(self.cfg, "y_load_skew_station_load_ratio_threshold", 1.6)):
            reasons.append("station_load_max_ratio")
        if float(metrics.get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "y_load_skew_arrival_slack_mean_threshold", 180.0)):
            reasons.append("arrival_slack_mean")
        return {
            "enabled": bool(reasons),
            "reason": ",".join(reasons) if reasons else "",
            "metrics": metrics,
        }

    def _u_slack_repair_context(
        self,
        metrics: Optional[Dict[str, float]] = None,
        late_task_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        metrics = dict(metrics or self._collect_layer_metrics())
        late_rows = int(self._compute_late_task_count() if late_task_count is None else late_task_count)
        reasons: List[str] = []
        if float(metrics.get("arrival_slack_mean", 0.0)) >= float(getattr(self.cfg, "u_slack_repair_arrival_slack_mean_threshold", 120.0)):
            reasons.append("arrival_slack_mean")
        if int(late_rows) >= int(getattr(self.cfg, "u_slack_repair_late_task_count_threshold", 6)):
            reasons.append("late_task_count")
        return {
            "enabled": bool(reasons),
            "reason": ",".join(reasons) if reasons else "",
            "late_task_count": int(late_rows),
            "metrics": metrics,
        }

    def _effective_x_global_eval_topk(self) -> int:
        topk = max(1, int(getattr(self.cfg, "x_global_eval_topk", 2)))
        if self._runtime_guard_level() >= 2:
            return 1
        return min(2, topk)

    def _effective_y_route_eval_topk(self) -> int:
        topk = max(1, int(getattr(self.cfg, "y_route_eval_topk", 2)))
        if self._runtime_guard_level() >= 2:
            return 1
        return topk

    def _effective_y_global_eval_topk(self) -> int:
        topk = max(1, int(getattr(self.cfg, "y_global_eval_topk", 2)))
        if self._runtime_guard_level() >= 2:
            return 1
        return topk

    def _effective_z_structural_eval_topk(self) -> int:
        topk = max(1, int(getattr(self.cfg, "z_structural_eval_topk", 2)))
        if self._runtime_guard_level() >= 2:
            return 1
        return topk

    def _effective_z_global_eval_topk(self) -> int:
        topk = max(1, int(getattr(self.cfg, "z_global_eval_topk", 1)))
        if self._z_all_candidates_global_eval_enabled():
            return max(1, topk)
        if self._is_z_throttled():
            return 0
        if self._runtime_guard_level() >= 2:
            return 1
        return min(2, topk)

    def _z_all_candidates_global_eval_enabled(self) -> bool:
        return bool(
            getattr(self.cfg, "z_all_global_eval_default", True)
            or getattr(self.cfg, "z_full_global_eval_experiment", False)
            or getattr(self.cfg, "z_eval_all_candidates", False)
        )

    def _z_full_global_eval_experiment_enabled(self) -> bool:
        return bool(self._z_all_candidates_global_eval_enabled())

    def _z_safe_operator_allowlist(self) -> Set[str]:
        return {
            "range_shrink_expand",
            "tote_replace_within_stack",
            "stack_replace",
        }

    def _z_strict_safe_operator_semantics_enabled(self) -> bool:
        return bool(getattr(self.cfg, "z_strict_safe_operator_semantics", False))

    def _z_generation_route_guard_enabled(self) -> bool:
        return bool(getattr(self.cfg, "z_generation_route_guard", False))

    def _z_repeat_reject_cache_enabled(self) -> bool:
        return bool(getattr(self.cfg, "z_repeat_reject_cache", False))

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

    def _x_anchor_groups_for_order(self, order_id: int) -> List[List[int]]:
        groups: List[List[int]] = []
        for st in self._iter_snapshot_subtasks(self.anchor):
            if int(getattr(getattr(st, "parent_order", None), "order_id", -1)) != int(order_id):
                continue
            sku_ids = sorted({
                int(getattr(sku, "id", -1))
                for sku in getattr(st, "unique_sku_list", []) or []
                if int(getattr(sku, "id", -1)) >= 0
            })
            if sku_ids:
                groups.append(sku_ids)
        return groups

    def _x_group_badness_components(self, order_id: int, group: List[int], features: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, float]:
        normalized = [int(sku_id) for sku_id in group if int(sku_id) >= 0]
        if not normalized:
            return {
                "affinity_penalty": 0.0,
                "route_span": 0.0,
                "completion_span": 0.0,
                "robot_mix": 0.0,
                "station_mismatch": 0.0,
            }
        route_positions = [float(features.get((int(order_id), sku_id), {}).get("route_pos", 0.0)) for sku_id in normalized]
        completions = [float(features.get((int(order_id), sku_id), {}).get("completion", 0.0)) for sku_id in normalized]
        robot_ids = {
            int(features.get((int(order_id), sku_id), {}).get("robot_id", -1))
            for sku_id in normalized
            if int(features.get((int(order_id), sku_id), {}).get("robot_id", -1)) >= 0
        }
        anchor_sku_profile = self.anchor_reference.get("anchor_sku_profile", {}) or {}
        station_ids = {
            int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("station_id", -1))
            for sku_id in normalized
            if int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("station_id", -1)) >= 0
        }
        return {
            "affinity_penalty": float(max(0.0, 1.0 - self._proposal_group_mean_affinity(int(order_id), normalized))),
            "route_span": float((max(route_positions) - min(route_positions)) if route_positions else 0.0),
            "completion_span": float((max(completions) - min(completions)) if completions else 0.0),
            "robot_mix": float(max(0.0, len(robot_ids) - 1)),
            "station_mismatch": float(max(0.0, len(station_ids) - 1)),
        }

    def _x_group_badness_score(self, order_id: int, group: List[int], features: Dict[Tuple[int, int], Dict[str, float]]) -> float:
        comp = self._x_group_badness_components(order_id, group, features)
        return float(
            6.0 * comp["affinity_penalty"]
            + 0.2 * comp["route_span"]
            + 0.03 * comp["completion_span"]
            + 3.0 * comp["robot_mix"]
            + 2.0 * comp["station_mismatch"]
        )

    def _x_select_focus_order(self, proposal: XSplitProposal, features: Dict[Tuple[int, int], Dict[str, float]], rng: random.Random) -> Optional[int]:
        rows: List[Tuple[float, int]] = []
        for order_id, groups in (proposal.order_to_subtask_sku_sets or {}).items():
            badness = sum(self._x_group_badness_score(int(order_id), list(group), features) for group in groups if group)
            if badness > 0.0:
                rows.append((float(badness), int(order_id)))
        if not rows:
            return None
        rows.sort(key=lambda item: (-item[0], item[1]))
        if len(rows) == 1 or rng.random() > 0.2:
            return int(rows[0][1])
        k = min(len(rows), 3)
        return int(rows[rng.randrange(k)][1])

    def _x_find_group_index_containing(self, groups: List[List[int]], sku_id: int) -> int:
        for idx, group in enumerate(groups):
            if int(sku_id) in {int(x) for x in group}:
                return int(idx)
        return -1

    def _x_try_move_single_sku_to_sibling_group(self, proposal: XSplitProposal, order_id: int, rng: random.Random, features: Dict[Tuple[int, int], Dict[str, float]]) -> int:
        groups = proposal.order_to_subtask_sku_sets.get(int(order_id), [])
        if len(groups) < 2:
            return 0
        scored_groups = sorted(
            [(self._x_group_badness_score(int(order_id), list(group), features), idx) for idx, group in enumerate(groups) if group],
            key=lambda item: (-item[0], item[1]),
        )
        if not scored_groups:
            return 0
        src_idx = int(scored_groups[0][1])
        source = list(groups[src_idx])
        if not source:
            return 0
        anchor_groups = self._x_anchor_groups_for_order(int(order_id))
        candidate_targets: List[int] = []
        for other_idx, group in enumerate(groups):
            if other_idx == src_idx:
                continue
            if any(set(int(x) for x in group).issubset(set(anchor_group)) or set(anchor_group).issubset(set(int(x) for x in group)) for anchor_group in anchor_groups):
                candidate_targets.append(int(other_idx))
        if not candidate_targets:
            candidate_targets = [idx for idx in range(len(groups)) if idx != src_idx]
        if not candidate_targets:
            return 0
        sku_scores = []
        for sku_id in source:
            route_pos = float(features.get((int(order_id), int(sku_id)), {}).get("route_pos", 0.0))
            completion = float(features.get((int(order_id), int(sku_id)), {}).get("completion", 0.0))
            affinity_without = self._proposal_group_mean_affinity(int(order_id), [row for row in source if int(row) != int(sku_id)])
            sku_scores.append((float(route_pos + 0.05 * completion + max(0.0, 1.0 - affinity_without)), int(sku_id)))
        sku_scores.sort(key=lambda item: (-item[0], item[1]))
        moved_sku = int(sku_scores[0][1])
        best_target = None
        best_gain = -float("inf")
        for tgt_idx in candidate_targets:
            target = list(groups[tgt_idx])
            before = self._x_group_badness_score(int(order_id), source, features) + self._x_group_badness_score(int(order_id), target, features)
            source_after = [row for row in source if int(row) != moved_sku]
            target_after = sorted({int(row) for row in target + [moved_sku]})
            after = self._x_group_badness_score(int(order_id), source_after, features) + self._x_group_badness_score(int(order_id), target_after, features)
            gain = before - after
            if gain > best_gain:
                best_gain = gain
                best_target = int(tgt_idx)
        if best_target is None:
            return 0
        groups[src_idx] = [row for row in groups[src_idx] if int(row) != moved_sku]
        groups[best_target] = sorted({int(row) for row in list(groups[best_target]) + [moved_sku]})
        proposal.touched_orders.add(int(order_id))
        setattr(proposal, "x_move_type", "x_move_single_sku_to_sibling_group")
        setattr(proposal, "x_moved_sku_count", 1)
        setattr(proposal, "x_swap_pair_count", 0)
        setattr(proposal, "x_changed_assignment_pair_count_hint", 1)
        return 1

    def _x_try_split_one_order_group(self, proposal: XSplitProposal, order_id: int, rng: random.Random, features: Dict[Tuple[int, int], Dict[str, float]]) -> int:
        groups = proposal.order_to_subtask_sku_sets.get(int(order_id), [])
        candidates = [(self._x_group_badness_score(int(order_id), list(group), features), idx) for idx, group in enumerate(groups) if len(group) >= 2]
        if not candidates:
            return 0
        candidates.sort(key=lambda item: (-item[0], item[1]))
        group_idx = int(candidates[0][1])
        group = list(groups[group_idx])
        outliers: List[Tuple[float, int]] = []
        group_route = [float(features.get((int(order_id), int(sku_id)), {}).get("route_pos", 0.0)) for sku_id in group]
        group_comp = [float(features.get((int(order_id), int(sku_id)), {}).get("completion", 0.0)) for sku_id in group]
        mean_route = float(sum(group_route) / max(1, len(group_route))) if group_route else 0.0
        mean_comp = float(sum(group_comp) / max(1, len(group_comp))) if group_comp else 0.0
        anchor_sku_profile = self.anchor_reference.get("anchor_sku_profile", {}) or {}
        group_station_ids = {
            int(anchor_sku_profile.get((int(order_id), int(sku_id)), {}).get("station_id", -1))
            for sku_id in group
            if int(anchor_sku_profile.get((int(order_id), int(sku_id)), {}).get("station_id", -1)) >= 0
        }
        dominant_station = next(iter(group_station_ids)) if len(group_station_ids) == 1 else -1
        for sku_id in group:
            row = features.get((int(order_id), int(sku_id)), {})
            route_score = abs(float(row.get("route_pos", 0.0)) - mean_route)
            completion_score = 0.1 * abs(float(row.get("completion", 0.0)) - mean_comp)
            robot_penalty = 0.0
            station_penalty = 0.0
            profile = anchor_sku_profile.get((int(order_id), int(sku_id)), {})
            if dominant_station >= 0 and int(profile.get("station_id", -1)) != dominant_station:
                station_penalty = 2.0
            robot_ids = {
                int(features.get((int(order_id), int(other)), {}).get("robot_id", -1))
                for other in group if int(other) != int(sku_id)
            }
            if len({rid for rid in robot_ids if rid >= 0}) >= 1 and int(row.get("robot_id", -1)) not in robot_ids:
                robot_penalty = 1.5
            outliers.append((float(route_score + completion_score + robot_penalty + station_penalty), int(sku_id)))
        outliers.sort(key=lambda item: (-item[0], item[1]))
        moved_sku = int(outliers[0][1])
        groups[group_idx] = [row for row in groups[group_idx] if int(row) != moved_sku]
        groups.append([moved_sku])
        proposal.touched_orders.add(int(order_id))
        setattr(proposal, "x_move_type", "x_split_one_order_group")
        setattr(proposal, "x_moved_sku_count", 1)
        setattr(proposal, "x_swap_pair_count", 0)
        setattr(proposal, "x_changed_assignment_pair_count_hint", 1)
        return 1

    def _x_try_merge_adjacent_anchor_compatible_groups(self, proposal: XSplitProposal, order_id: int, rng: random.Random, features: Dict[Tuple[int, int], Dict[str, float]]) -> int:
        groups = proposal.order_to_subtask_sku_sets.get(int(order_id), [])
        if len(groups) < 2:
            return 0
        anchor_sku_profile = self.anchor_reference.get("anchor_sku_profile", {}) or {}
        best_pair = None
        best_gain = -float("inf")
        for idx in range(len(groups) - 1):
            left = [int(x) for x in groups[idx] if int(x) >= 0]
            right = [int(x) for x in groups[idx + 1] if int(x) >= 0]
            if not left or not right:
                continue
            left_station = {int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("station_id", -1)) for sku_id in left}
            right_station = {int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("station_id", -1)) for sku_id in right}
            left_robot = {int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("robot_id", -1)) for sku_id in left}
            right_robot = {int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("robot_id", -1)) for sku_id in right}
            if len({sid for sid in left_station | right_station if sid >= 0}) > 1:
                continue
            if len({rid for rid in left_robot | right_robot if rid >= 0}) > 1:
                continue
            moved_pair_hint = min(len(left), len(right))
            if moved_pair_hint > max(1, int(getattr(self.cfg, "x_micro_move_group_cap", 1))):
                continue
            merged = sorted({int(x) for x in left + right})
            capacity_limit = max(2, min(self._get_order_capacity_limit(int(order_id)), 3))
            if len(merged) > capacity_limit:
                continue
            before = self._x_group_badness_score(int(order_id), left, features) + self._x_group_badness_score(int(order_id), right, features)
            after = self._x_group_badness_score(int(order_id), merged, features)
            gain = before - after
            if gain > best_gain:
                best_gain = gain
                best_pair = (idx, idx + 1, merged, moved_pair_hint)
        if best_pair is None:
            return 0
        left_idx, right_idx, merged_group, moved_pair_hint = best_pair
        groups[left_idx] = merged_group
        del groups[right_idx]
        proposal.touched_orders.add(int(order_id))
        setattr(proposal, "x_move_type", "x_merge_adjacent_anchor_compatible_groups")
        setattr(proposal, "x_moved_sku_count", 0)
        setattr(proposal, "x_swap_pair_count", 0)
        setattr(proposal, "x_changed_assignment_pair_count_hint", int(moved_pair_hint))
        return 1

    def _x_try_swap_outlier_skus_between_two_groups(self, proposal: XSplitProposal, order_id: int, rng: random.Random, features: Dict[Tuple[int, int], Dict[str, float]]) -> int:
        groups = proposal.order_to_subtask_sku_sets.get(int(order_id), [])
        if len(groups) < 2:
            return 0
        scored = [(self._x_group_badness_score(int(order_id), list(group), features), idx) for idx, group in enumerate(groups) if group]
        if len(scored) < 2:
            return 0
        scored.sort(key=lambda item: (-item[0], item[1]))
        idx_a = int(scored[0][1])
        idx_b = int(scored[1][1])
        group_a = list(groups[idx_a])
        group_b = list(groups[idx_b])
        if not group_a or not group_b:
            return 0
        def _pick_outlier(group: List[int]) -> int:
            rows = []
            route_positions = [float(features.get((int(order_id), int(sku_id)), {}).get("route_pos", 0.0)) for sku_id in group]
            mean_route = float(sum(route_positions) / max(1, len(route_positions))) if route_positions else 0.0
            for sku_id in group:
                rows.append((abs(float(features.get((int(order_id), int(sku_id)), {}).get("route_pos", 0.0)) - mean_route), int(sku_id)))
            rows.sort(key=lambda item: (-item[0], item[1]))
            return int(rows[0][1])
        sku_a = _pick_outlier(group_a)
        sku_b = _pick_outlier(group_b)
        if sku_a == sku_b:
            return 0
        before = self._x_group_badness_score(int(order_id), group_a, features) + self._x_group_badness_score(int(order_id), group_b, features)
        new_a = sorted({int(x) for x in group_a if int(x) != sku_a} | {int(sku_b)})
        new_b = sorted({int(x) for x in group_b if int(x) != sku_b} | {int(sku_a)})
        after = self._x_group_badness_score(int(order_id), new_a, features) + self._x_group_badness_score(int(order_id), new_b, features)
        if after > before and rng.random() > 0.2:
            return 0
        groups[idx_a] = new_a
        groups[idx_b] = new_b
        proposal.touched_orders.add(int(order_id))
        setattr(proposal, "x_move_type", "x_swap_outlier_skus_between_two_groups")
        setattr(proposal, "x_moved_sku_count", 2)
        setattr(proposal, "x_swap_pair_count", 1)
        setattr(proposal, "x_changed_assignment_pair_count_hint", 2)
        return 2

    def _point_xy(self, point: Any) -> Optional[Tuple[float, float]]:
        if point is None:
            return None
        if not hasattr(point, "x") or not hasattr(point, "y"):
            return None
        return (float(point.x), float(point.y))

    def _stack_xy(self, stack_id: int) -> Optional[Tuple[float, float]]:
        if self.problem is None:
            return None
        stack = self.problem.point_to_stack.get(int(stack_id))
        if stack is None or getattr(stack, "store_point", None) is None:
            return None
        return self._point_xy(stack.store_point)

    def _xy_manhattan(self, left: Optional[Tuple[float, float]], right: Optional[Tuple[float, float]]) -> float:
        if left is None or right is None:
            return float("inf")
        return float(abs(float(left[0]) - float(right[0])) + abs(float(left[1]) - float(right[1])))

    def _warehouse_distance_scale(self) -> float:
        if self.problem is None:
            return 1.0
        pts = [
            self._point_xy(getattr(stack, "store_point", None))
            for stack in (getattr(self.problem, "point_to_stack", {}) or {}).values()
            if getattr(stack, "store_point", None) is not None
        ]
        pts = [pt for pt in pts if pt is not None]
        if len(pts) < 2:
            return 1.0
        xs = [float(pt[0]) for pt in pts]
        ys = [float(pt[1]) for pt in pts]
        return max(1.0, float((max(xs) - min(xs)) + (max(ys) - min(ys))))

    def _x_candidate_tote_ids_for_sku(self, sku_id: int) -> List[int]:
        if self.problem is None:
            return []
        sku_obj = self.problem.id_to_sku.get(int(sku_id))
        if sku_obj is None:
            return []
        return list(dict.fromkeys(int(tid) for tid in (getattr(sku_obj, "storeToteList", []) or []) if int(tid) >= 0))

    def _x_candidate_stack_ids_for_sku(self, sku_id: int) -> List[int]:
        if self.problem is None:
            return []
        stack_ids: List[int] = []
        for tote_id in self._x_candidate_tote_ids_for_sku(int(sku_id)):
            tote = self.problem.id_to_tote.get(int(tote_id))
            if tote is None or getattr(tote, "store_point", None) is None:
                continue
            stack_ids.append(int(getattr(tote.store_point, "idx", -1)))
        return list(dict.fromkeys(stack_id for stack_id in stack_ids if stack_id >= 0))

    def _x_candidate_stack_points_for_sku(self, sku_id: int) -> List[Tuple[float, float]]:
        rows = [self._stack_xy(stack_id) for stack_id in self._x_candidate_stack_ids_for_sku(int(sku_id))]
        return [pt for pt in rows if pt is not None]

    def _x_same_tote_bonus(self, sku_id: int, group: List[int]) -> float:
        if not group:
            return 0.0
        candidate_totes = set(self._x_candidate_tote_ids_for_sku(int(sku_id)))
        if not candidate_totes:
            return 0.0
        same = 0.0
        for other in group:
            if candidate_totes & set(self._x_candidate_tote_ids_for_sku(int(other))):
                same += 1.0
        return float(same / max(1, len(group)))

    def _x_adjacent_stack_bonus(self, sku_id: int, group: List[int]) -> float:
        if not group:
            return 0.0
        sku_stack_ids = self._x_candidate_stack_ids_for_sku(int(sku_id))
        if not sku_stack_ids:
            return 0.0
        distance_cap = float(getattr(self.cfg, "x_adjacent_stack_distance_cap", 1.0))
        matched = 0.0
        for other in group:
            other_stack_ids = self._x_candidate_stack_ids_for_sku(int(other))
            adjacent = False
            for left_sid in sku_stack_ids:
                left_xy = self._stack_xy(int(left_sid))
                for right_sid in other_stack_ids:
                    right_xy = self._stack_xy(int(right_sid))
                    if self._xy_manhattan(left_xy, right_xy) <= distance_cap + 1e-9:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                matched += 1.0
        return float(matched / max(1, len(group)))

    def _x_group_stack_points(self, sku_ids: List[int]) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        seen: Set[Tuple[float, float]] = set()
        for sku_id in sku_ids:
            for pt in self._x_candidate_stack_points_for_sku(int(sku_id)):
                if pt in seen:
                    continue
                seen.add(pt)
                points.append(pt)
        return points

    def _x_group_stack_span_normalized(self, sku_ids: List[int]) -> float:
        pts = self._x_group_stack_points(list(sku_ids))
        if len(pts) < 2:
            return 0.0
        span = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                span = max(span, self._xy_manhattan(pts[i], pts[j]))
        return float(span / self._warehouse_distance_scale())

    def _x_group_spatial_dispersion_score(self, group: List[int]) -> float:
        pts = self._x_group_stack_points(list(group))
        if len(pts) < 2:
            return 0.0
        pairwise = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                pairwise.append(self._xy_manhattan(pts[i], pts[j]))
        if not pairwise:
            return 0.0
        cx = float(sum(pt[0] for pt in pts) / len(pts))
        cy = float(sum(pt[1] for pt in pts) / len(pts))
        max_radius = max(self._xy_manhattan(pt, (cx, cy)) for pt in pts)
        scale = self._warehouse_distance_scale()
        return float(0.5 * (sum(pairwise) / len(pairwise)) / scale + 0.5 * max_radius / scale)

    def _x_pair_co_box_rate(self, left_sku_id: int, right_sku_id: int) -> float:
        left_totes = set(self._x_candidate_tote_ids_for_sku(int(left_sku_id)))
        right_totes = set(self._x_candidate_tote_ids_for_sku(int(right_sku_id)))
        union = left_totes | right_totes
        if not union:
            return 0.0
        return float(len(left_totes & right_totes) / max(1, len(union)))

    def _x_pair_adjacent_stack_rate(self, left_sku_id: int, right_sku_id: int) -> float:
        left_stacks = self._x_candidate_stack_ids_for_sku(int(left_sku_id))
        right_stacks = self._x_candidate_stack_ids_for_sku(int(right_sku_id))
        total_pairs = 0.0
        adjacent_pairs = 0.0
        distance_cap = float(getattr(self.cfg, "x_adjacent_stack_distance_cap", 1.0))
        for left_sid in left_stacks:
            left_xy = self._stack_xy(int(left_sid))
            for right_sid in right_stacks:
                right_xy = self._stack_xy(int(right_sid))
                if left_xy is None or right_xy is None:
                    continue
                total_pairs += 1.0
                if self._xy_manhattan(left_xy, right_xy) <= distance_cap + 1e-9:
                    adjacent_pairs += 1.0
        if total_pairs <= 0.0:
            return 0.0
        return float(adjacent_pairs / total_pairs)

    def _x_group_low_consolidation_score(self, group: List[int]) -> float:
        rows = [int(sku_id) for sku_id in group if int(sku_id) >= 0]
        if len(rows) < 2:
            return 0.0
        pair_scores: List[float] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                co_box = self._x_pair_co_box_rate(int(rows[i]), int(rows[j]))
                adjacent = self._x_pair_adjacent_stack_rate(int(rows[i]), int(rows[j]))
                pair_scores.append(float(1.0 - max(co_box, 0.5 * adjacent)))
        return float(sum(pair_scores) / len(pair_scores)) if pair_scores else 0.0

    def _x_order_bom_trust_region(self, order_id: int) -> int:
        bom_lines = max(1, len(self._get_order_unique_sku_ids(int(order_id))))
        ratio = max(0.01, float(getattr(self.cfg, "x_bom_destroy_ratio", 0.15)))
        cap = max(1, int(getattr(self.cfg, "x_bom_destroy_max_lines", 2)))
        return max(1, min(cap, int(math.ceil(ratio * bom_lines))))

    def _x_robot_hotspot_centroids(self) -> List[Tuple[float, float]]:
        cached = list(self.anchor_reference.get("path_hotspot_centroids", []) or [])
        if cached:
            return [(float(pt[0]), float(pt[1])) for pt in cached]
        grouped: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        for profile in (self.anchor_reference.get("anchor_task_profile", {}) or {}).values():
            stack_xy = self._stack_xy(int(profile.get("stack_id", -1)))
            rid = int(profile.get("robot_id", -1))
            if stack_xy is None or rid < 0:
                continue
            grouped[rid].append(stack_xy)
        centers: List[Tuple[float, float]] = []
        for rows in grouped.values():
            if not rows:
                continue
            centers.append((
                float(sum(pt[0] for pt in rows) / len(rows)),
                float(sum(pt[1] for pt in rows) / len(rows)),
            ))
        return centers

    def _x_group_hotspot_alignment(self, sku_ids: List[int]) -> float:
        pts = self._x_group_stack_points(list(sku_ids))
        centers = self._x_robot_hotspot_centroids()
        if not pts or not centers:
            return 0.0
        cx = float(sum(pt[0] for pt in pts) / len(pts))
        cy = float(sum(pt[1] for pt in pts) / len(pts))
        best_dist = min(self._xy_manhattan((cx, cy), center) for center in centers)
        return float(max(0.0, 1.0 - best_dist / self._warehouse_distance_scale()))

    def _x_rank_spatial_outliers(self, group: List[int]) -> List[int]:
        rows: List[Tuple[float, int]] = []
        for sku_id in group:
            pts = self._x_candidate_stack_points_for_sku(int(sku_id))
            if not pts:
                rows.append((0.0, int(sku_id)))
                continue
            mean_dist = 0.0
            cnt = 0.0
            for other in group:
                if int(other) == int(sku_id):
                    continue
                for left in pts:
                    for right in self._x_candidate_stack_points_for_sku(int(other)):
                        mean_dist += self._xy_manhattan(left, right)
                        cnt += 1.0
            rows.append(((mean_dist / cnt) if cnt > 0.0 else 0.0, int(sku_id)))
        rows.sort(key=lambda item: (-item[0], item[1]))
        return [sku_id for _, sku_id in rows]

    def _x_rank_low_consolidation_rows(self, group: List[int]) -> List[int]:
        rows: List[Tuple[float, int]] = []
        for sku_id in group:
            best_pair_score = 0.0
            for other in group:
                if int(other) == int(sku_id):
                    continue
                best_pair_score = max(
                    best_pair_score,
                    max(
                        self._x_pair_co_box_rate(int(sku_id), int(other)),
                        0.5 * self._x_pair_adjacent_stack_rate(int(sku_id), int(other)),
                    ),
                )
            rows.append((1.0 - best_pair_score, int(sku_id)))
        rows.sort(key=lambda item: (-item[0], item[1]))
        return [sku_id for _, sku_id in rows]

    def _x_select_destroy_group(self, proposal: XSplitProposal, operator: str) -> Optional[Tuple[int, int, float]]:
        best: Optional[Tuple[float, int, int]] = None
        for order_id, groups in (proposal.order_to_subtask_sku_sets or {}).items():
            for idx, group in enumerate(groups):
                normalized = sorted(int(x) for x in group if int(x) >= 0)
                if not normalized:
                    continue
                if operator == "x_destroy_spatial_dispersion":
                    score = self._x_group_spatial_dispersion_score(normalized)
                else:
                    score = self._x_group_low_consolidation_score(normalized)
                candidate = (float(score), int(order_id), int(idx))
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] <= 0.0:
            return None
        return int(best[1]), int(best[2]), float(best[0])

    def _x_repair_insertion_components(self, order_id: int, group: List[int], sku_id: int, use_y_hotspot: bool) -> Dict[str, float]:
        same_tote_bonus = self._x_same_tote_bonus(int(sku_id), list(group))
        adjacent_stack_bonus = self._x_adjacent_stack_bonus(int(sku_id), list(group))
        route_span_penalty = self._x_group_stack_span_normalized(list(group) + [int(sku_id)])
        capacity = max(1, self._get_order_capacity_limit(int(order_id)))
        trust_penalty = max(0.0, float(len(group) + 1 - capacity) / max(1.0, float(capacity)))
        y_hotspot_bonus = self._x_group_hotspot_alignment(list(group) + [int(sku_id)]) if use_y_hotspot else 0.0
        score = (
            float(getattr(self.cfg, "x_same_tote_bonus_weight", 1.5)) * same_tote_bonus
            + float(getattr(self.cfg, "x_adjacent_stack_bonus_weight", 1.0)) * adjacent_stack_bonus
            - float(getattr(self.cfg, "x_route_span_penalty_weight", 1.0)) * route_span_penalty
            - float(getattr(self.cfg, "x_trust_region_penalty_weight", 1.0)) * trust_penalty
            + float(getattr(self.cfg, "x_y_hotspot_bonus_weight", 0.5)) * y_hotspot_bonus
        )
        return {
            "score": float(score),
            "same_tote_bonus": float(same_tote_bonus),
            "adjacent_stack_bonus": float(adjacent_stack_bonus),
            "route_span_penalty": float(route_span_penalty),
            "trust_region_penalty": float(trust_penalty),
            "y_hotspot_bonus": float(y_hotspot_bonus),
        }

    def _apply_x_destroy_operator(self, proposal: XSplitProposal, operator: str, rng: random.Random, strength: int) -> int:
        self._normalize_x_split_proposal(proposal)
        setattr(proposal, "x_merge_split_fallback_used", False)
        setattr(proposal, "x_destroy_size_effective", 0)
        setattr(proposal, "x_move_type", str(operator))
        setattr(proposal, "x_moved_sku_count", 0)
        setattr(proposal, "x_swap_pair_count", 0)
        setattr(proposal, "x_changed_assignment_pair_count_hint", 0)
        setattr(proposal, "x_spatial_dispersion_score", 0.0)
        setattr(proposal, "x_low_consolidation_score", 0.0)
        setattr(proposal, "x_same_tote_gain", 0.0)
        setattr(proposal, "x_adjacent_stack_gain", 0.0)
        setattr(proposal, "x_y_hotspot_alignment", 0.0)
        setattr(proposal, "x_affected_bom_ratio_hint", 0.0)
        setattr(proposal, "x_destroy_order_id", -1)
        setattr(proposal, "x_destroy_source_group_idx", -1)
        setattr(proposal, "x_directly_moved_sku_ids", [])
        proposal.unassigned_skus = {}
        selected = self._x_select_destroy_group(proposal, str(operator))
        if selected is None:
            return 0
        order_id, group_idx, destroy_score = selected
        groups = proposal.order_to_subtask_sku_sets.get(int(order_id), [])
        if group_idx < 0 or group_idx >= len(groups):
            return 0
        source_group = [int(x) for x in groups[group_idx] if int(x) >= 0]
        if not source_group:
            return 0
        remove_n = min(len(source_group), self._x_order_bom_trust_region(int(order_id)))
        ranked = (
            self._x_rank_spatial_outliers(source_group)
            if str(operator) == "x_destroy_spatial_dispersion"
            else self._x_rank_low_consolidation_rows(source_group)
        )
        removed = [int(sku_id) for sku_id in ranked[: max(1, remove_n)]]
        if not removed:
            return 0
        groups[group_idx] = [int(sku_id) for sku_id in source_group if int(sku_id) not in set(removed)]
        proposal.unassigned_skus[int(order_id)] = list(removed)
        proposal.touched_orders = {int(order_id)}
        setattr(proposal, "x_destroy_order_id", int(order_id))
        setattr(proposal, "x_destroy_source_group_idx", int(group_idx))
        setattr(proposal, "x_destroy_size_effective", int(len(removed)))
        setattr(proposal, "x_moved_sku_count", int(len(removed)))
        setattr(proposal, "x_changed_assignment_pair_count_hint", int(len(removed)))
        setattr(proposal, "x_directly_moved_sku_ids", list(removed))
        setattr(proposal, "x_spatial_dispersion_score", float(destroy_score if str(operator) == "x_destroy_spatial_dispersion" else self._x_group_spatial_dispersion_score(source_group)))
        setattr(proposal, "x_low_consolidation_score", float(destroy_score if str(operator) == "x_destroy_low_consolidation" else self._x_group_low_consolidation_score(source_group)))
        setattr(
            proposal,
            "x_affected_bom_ratio_hint",
            float(len(removed) / max(1, len(self._get_order_unique_sku_ids(int(order_id))))),
        )
        self._normalize_x_split_proposal(proposal)
        return int(len(removed))

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
        changed = False
        use_y_hotspot = str(operator) == "x_repair_y_proxy_guided"
        same_tote_gain_sum = 0.0
        adjacent_stack_gain_sum = 0.0
        y_hotspot_alignment_sum = 0.0
        affected_bom_ratio = float(getattr(proposal, "x_affected_bom_ratio_hint", 0.0))
        repair_count = 0
        for order_id in sorted(list((proposal.unassigned_skus or {}).keys())):
            pending = list(sorted({int(sku_id) for sku_id in proposal.unassigned_skus.get(order_id, []) if int(sku_id) >= 0}))
            if not pending:
                continue
            groups = proposal.order_to_subtask_sku_sets.setdefault(int(order_id), [])
            capacity = max(1, self._get_order_capacity_limit(int(order_id)))
            source_group_idx = int(getattr(proposal, "x_destroy_source_group_idx", -1))
            if int(getattr(proposal, "x_destroy_order_id", -1)) != int(order_id):
                source_group_idx = -1
            candidate_rows: List[Tuple[float, int]] = []
            for group_idx, group in enumerate(groups):
                normalized = sorted({int(x) for x in group if int(x) >= 0})
                if not normalized or int(group_idx) == source_group_idx:
                    continue
                if len(normalized) + len(pending) > capacity:
                    continue
                simulated = list(normalized)
                total_score = 0.0
                for sku_id in pending:
                    comp = self._x_repair_insertion_components(int(order_id), list(simulated), int(sku_id), use_y_hotspot)
                    total_score += float(comp.get("score", 0.0))
                    simulated = sorted({int(x) for x in simulated + [int(sku_id)]})
                candidate_rows.append((float(total_score), int(group_idx)))
            candidate_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            locked_target_idx = int(candidate_rows[0][1]) if candidate_rows and float(candidate_rows[0][0]) > 0.0 else -1
            if locked_target_idx < 0:
                groups.append([])
                locked_target_idx = int(len(groups) - 1)
            target_group = groups[locked_target_idx]
            for sku_id in pending:
                normalized_target = sorted({int(x) for x in target_group if int(x) >= 0})
                comp = self._x_repair_insertion_components(int(order_id), normalized_target, int(sku_id), use_y_hotspot)
                target_group.append(int(sku_id))
                target_group[:] = sorted({int(row) for row in target_group})
                same_tote_gain_sum += float(comp.get("same_tote_bonus", 0.0))
                adjacent_stack_gain_sum += float(comp.get("adjacent_stack_bonus", 0.0))
                y_hotspot_alignment_sum += float(comp.get("y_hotspot_bonus", 0.0))
                repair_count += 1
                proposal.touched_orders.add(int(order_id))
                changed = True
            proposal.unassigned_skus[int(order_id)] = []
        setattr(proposal, "x_same_tote_gain", float(same_tote_gain_sum / max(1, repair_count)))
        setattr(proposal, "x_adjacent_stack_gain", float(adjacent_stack_gain_sum / max(1, repair_count)))
        setattr(proposal, "x_y_hotspot_alignment", float(y_hotspot_alignment_sum / max(1, repair_count)))
        setattr(proposal, "x_affected_bom_ratio_hint", float(affected_bom_ratio))
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

    def _compute_x_proposal_signature(self, proposal: XSplitProposal) -> str:
        rows: List[str] = []
        for order_id in sorted((proposal.order_to_subtask_sku_sets or {}).keys()):
            groups = [
                ".".join(str(int(sku_id)) for sku_id in sorted({int(sku_id) for sku_id in group if int(sku_id) >= 0}))
                for group in (proposal.order_to_subtask_sku_sets or {}).get(order_id, [])
                if group
            ]
            rows.append(f"{int(order_id)}:{'|'.join(sorted(groups))}")
        return ";".join(rows)

    def _compute_x_solution_signature_from_problem(self) -> str:
        return self._compute_x_proposal_signature(self._extract_x_split_solution())

    def _compute_z_solution_signature(self) -> str:
        rows: List[str] = []
        for task in sorted(self._collect_all_tasks(), key=lambda item: int(getattr(item, "task_id", -1))):
            rows.append(
                "|".join([
                    str(int(getattr(task, "task_id", -1))),
                    str(int(getattr(task, "sub_task_id", -1))),
                    str(int(getattr(task, "target_stack_id", -1))),
                    str(int(getattr(task, "target_station_id", -1))),
                    str(int(getattr(task, "robot_id", -1))),
                    str(int(getattr(task, "trip_id", 0))),
                    str(int(getattr(task, "robot_visit_sequence", 0))),
                    str(getattr(task, "operation_mode", "")),
                ])
            )
        return ";".join(rows)

    def _anchor_noise_ratio(self) -> float:
        noise_total = 0.0
        target_total = 0.0
        for task in self._iter_snapshot_tasks(self.anchor):
            noise_total += float(len(getattr(task, "noise_tote_ids", []) or []))
            target_total += float(len(getattr(task, "target_tote_ids", []) or []))
        return float(noise_total / target_total) if target_total > 0.0 else 0.0

    def _anchor_route_tail(self) -> float:
        rows = [
            max(float(getattr(task, "arrival_time_at_stack", 0.0)), float(getattr(task, "arrival_time_at_station", 0.0)))
            for task in self._iter_snapshot_tasks(self.anchor)
        ]
        return float(max(rows)) if rows else 0.0

    def _estimate_group_stack_span(self, group: List[int]) -> float:
        points: Dict[int, Tuple[float, float]] = {}
        for sku_id in group:
            sku_obj = self.problem.id_to_sku.get(int(sku_id)) if self.problem is not None else None
            if sku_obj is None:
                continue
            for tote_id in getattr(sku_obj, "storeToteList", []) or []:
                tote = self.problem.id_to_tote.get(int(tote_id)) if self.problem is not None else None
                if tote is None or getattr(tote, "store_point", None) is None:
                    continue
                pt = tote.store_point
                points[int(pt.idx)] = (float(pt.x), float(pt.y))
        coords = list(points.values())
        span = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                span = max(span, abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]))
        return float(span)

    def _extract_x_f0_features(self, proposal: XSplitProposal) -> Dict[str, float]:
        anchor_count = float(len(self._iter_snapshot_subtasks(self.anchor)))
        delta_subtask_count = float(proposal.subtask_count) - anchor_count
        diff_stats = self._x_assignment_diff_stats(proposal)
        return {
            "spatial_dispersion_score": float(getattr(proposal, "x_spatial_dispersion_score", 0.0)),
            "low_consolidation_score": float(getattr(proposal, "x_low_consolidation_score", 0.0)),
            "same_tote_gain": float(getattr(proposal, "x_same_tote_gain", 0.0)),
            "adjacent_stack_gain": float(getattr(proposal, "x_adjacent_stack_gain", 0.0)),
            "y_hotspot_alignment": float(getattr(proposal, "x_y_hotspot_alignment", 0.0)),
            "delta_subtask_count": float(delta_subtask_count),
            "changed_orders": float(len(diff_stats.get("changed_orders_set", set()))),
            "changed_assignment_pairs": float(diff_stats.get("changed_assignment_pair_count", 0)),
            "anchor_template_preservation_ratio": float(diff_stats.get("anchor_template_preservation_ratio", 0.0)),
            "affected_bom_ratio": float(diff_stats.get("affected_bom_ratio", getattr(proposal, "x_affected_bom_ratio_hint", 0.0))),
            "x_micro_move_size": float(diff_stats.get("x_micro_move_size", 0.0)),
        }

    def _score_x_f0_prior(self, features: Dict[str, float]) -> float:
        return float(
            10.0 * float(features.get("spatial_dispersion_score", 0.0))
            + 8.0 * float(features.get("low_consolidation_score", 0.0))
            + 2.0 * float(features.get("changed_assignment_pairs", 0.0))
            + 8.0 * float(features.get("affected_bom_ratio", 0.0))
            - 4.0 * float(features.get("same_tote_gain", 0.0))
            - 2.5 * float(features.get("adjacent_stack_gain", 0.0))
            - 2.0 * float(features.get("y_hotspot_alignment", 0.0))
        )

    def _extract_z_f0_features(self, move_meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        move_meta = move_meta or {}
        anchor_tasks = self._iter_snapshot_tasks(self.anchor)
        anchor_stack_ids = {int(getattr(task, "target_stack_id", -1)) for task in anchor_tasks if int(getattr(task, "target_stack_id", -1)) >= 0}
        current_stack_ids = {int(getattr(task, "target_stack_id", -1)) for task in self._collect_all_tasks() if int(getattr(task, "target_stack_id", -1)) >= 0}
        current_mode_set = {str(getattr(task, "operation_mode", "")) for task in self._collect_all_tasks()}
        anchor_mode_set = {str(getattr(task, "operation_mode", "")) for task in anchor_tasks}
        noise_now = 0.0
        noise_anchor = 0.0
        hit_now = 0.0
        hit_anchor = 0.0
        for st in getattr(self.problem, "subtask_list", []) or []:
            current_tasks = list(getattr(st, "execution_tasks", []) or [])
            anchor_task_rows = [
                profile for profile in (self.anchor_reference.get("anchor_task_profile", {}) or {}).values()
                if int(profile.get("subtask_id", -1)) == int(getattr(st, "id", -1))
            ]
            noise_now += float(sum(len(getattr(task, "noise_tote_ids", []) or []) for task in current_tasks))
            noise_anchor += float(sum(len(profile.get("noise_tote_ids", []) or []) for profile in anchor_task_rows))
            hit_now += float(sum(len(getattr(task, "hit_tote_ids", []) or []) for task in current_tasks))
            hit_anchor += float(sum(len(profile.get("hit_tote_ids", []) or []) for profile in anchor_task_rows))
        return {
            "route_insertion_detour": float(move_meta.get("z_route_insertion_detour", move_meta.get("route_insertion_detour", 0.0))),
            "hit_frequency_bonus": float(move_meta.get("z_hit_frequency_bonus", move_meta.get("hit_frequency_bonus", 0.0))),
            "stack_locality_score": float(move_meta.get("z_stack_locality_score", move_meta.get("stack_locality_score", 0.0))),
            "demand_ratio": float(move_meta.get("z_demand_ratio", move_meta.get("demand_ratio", 0.0))),
            "congestion_proxy": float(move_meta.get("z_congestion_proxy", move_meta.get("congestion_proxy", 0.0))),
            "noise_tote_delta": float(max(0.0, noise_now - noise_anchor)),
            "hit_tote_preservation_ratio": float(min(1.0, hit_now / max(1.0, hit_anchor))) if hit_anchor > 0.0 else 1.0,
            "changed_task_count": float(max(0.0, float(move_meta.get("task_delta", 0.0)))),
            "changed_subtask_count": float(move_meta.get("changed_subtask_count", move_meta.get("z_changed_subtask_count", 0.0))),
            "changed_stack_count": float(len(current_stack_ids.symmetric_difference(anchor_stack_ids))),
            "changed_mode_count": float(len(current_mode_set.symmetric_difference(anchor_mode_set))),
        }

    def _score_z_f0_prior(self, features: Dict[str, float]) -> float:
        return float(
            10.0 * float(features.get("route_insertion_detour", 0.0))
            - 6.0 * float(features.get("hit_frequency_bonus", 0.0))
            - 3.0 * float(features.get("stack_locality_score", 0.0))
            - 4.0 * float(features.get("demand_ratio", 0.0))
            + 2.0 * float(features.get("congestion_proxy", 0.0))
            + 4.0 * float(features.get("noise_tote_delta", 0.0))
            - 4.0 * float(features.get("hit_tote_preservation_ratio", 1.0))
        )



    def _prediction_interval_overlap(self, left: Dict[str, Any], right: Dict[str, Any], band_k: float) -> bool:
        left_pred = float((left.get("score", {}) or {}).get("augmented_obj", float("inf")))
        right_pred = float((right.get("score", {}) or {}).get("augmented_obj", float("inf")))
        left_unc = float((left.get("score", {}) or {}).get("prediction_uncertainty", 0.0))
        right_unc = float((right.get("score", {}) or {}).get("prediction_uncertainty", 0.0))
        if not math.isfinite(left_pred) or not math.isfinite(right_pred):
            return False
        left_lo = left_pred - float(band_k) * max(0.0, left_unc)
        left_hi = left_pred + float(band_k) * max(0.0, left_unc)
        right_lo = right_pred - float(band_k) * max(0.0, right_unc)
        right_hi = right_pred + float(band_k) * max(0.0, right_unc)
        return bool(max(left_lo, right_lo) <= min(left_hi, right_hi) + 1e-9)



    def _anchor_station_cmax(self) -> float:
        rows = list((self.anchor_reference.get("anchor_subtask_profile", {}) or {}).values())
        if not rows:
            return float(self.anchor_z if math.isfinite(self.anchor_z) else 0.0)
        return float(max(float(row.get("completion", 0.0)) for row in rows))

    def _penalty_excess(self, value: float, soft_cap: float, hard_cap: float, weight: float) -> float:
        value = float(value)
        soft_cap = float(soft_cap)
        hard_cap = max(soft_cap, float(hard_cap))
        if value <= soft_cap + 1e-9:
            return 0.0
        span = max(1.0, hard_cap - soft_cap)
        return float(weight * ((value - soft_cap) ** 2) / span)

    def _normalized_excess_ratio(self, value: float, baseline: float = 0.0, scale: float = 1.0) -> float:
        value = float(value)
        baseline = float(baseline)
        scale = max(1e-9, float(scale))
        return float(max(0.0, value - baseline) / scale)

    def _soft_cap_excess_ratio(self, value: float, soft_cap: float, hard_cap: float) -> float:
        value = float(max(0.0, value))
        soft_cap = float(max(0.0, soft_cap))
        hard_cap = max(soft_cap + 1e-9, float(hard_cap))
        if value <= soft_cap + 1e-9:
            return 0.0
        return float((value - soft_cap) / max(1e-9, hard_cap - soft_cap))



    def _score_x_candidate_classic_soft(
        self,
        proposal: XSplitProposal,
        score: Dict[str, Any],
        f1_result: F1EvalResult,
    ) -> Dict[str, Any]:
        anchor_z = float(self.anchor_z if math.isfinite(self.anchor_z) else f1_result.proxy_z)
        anchor_scale = max(1.0, anchor_z)
        soft_penalties = self._estimate_x_soft_penalties(proposal, f1_result)
        anchor_station_cmax = max(1.0, float(self._anchor_station_cmax()))
        changed_orders_scale = max(
            1.0,
            float(score.get("changed_orders", 0.0)),
            float(getattr(proposal, "touched_subtask_count", 0.0)),
        )
        template_change_total = float(score.get("station_template_change_count", 0.0)) + float(
            score.get("robot_trip_template_change_count", 0.0)
        )

        affinity_signal = self._normalized_excess_ratio(
            float(score.get("x_affinity_penalty", 0.0)),
            baseline=0.0,
            scale=changed_orders_scale,
        )
        template_signal = self._soft_cap_excess_ratio(
            template_change_total,
            float(getattr(self.cfg, "x_template_change_soft_cap", 2)),
            float(getattr(self.cfg, "x_template_change_hard_cap", 5)),
        )
        route_conflict_signal = self._normalized_excess_ratio(
            float(score.get("x_route_conflict_penalty", 0.0)),
            baseline=0.0,
            scale=anchor_scale,
        )
        route_span_signal = self._soft_cap_excess_ratio(
            float(score.get("group_route_span_delta", 0.0)),
            float(getattr(self.cfg, "x_group_route_span_cap", 8.0)),
            float(getattr(self.cfg, "x_group_route_span_cap", 8.0)) * 4.0,
        )
        route_tail_signal = self._normalized_excess_ratio(
            float(f1_result.route_tail_delta),
            baseline=0.0,
            scale=anchor_scale,
        )
        finish_dispersion_signal = self._normalized_excess_ratio(
            float(score.get("x_finish_time_dispersion_penalty", 0.0)),
            baseline=0.0,
            scale=anchor_scale,
        )
        completion_span_signal = self._soft_cap_excess_ratio(
            float(score.get("group_completion_span_delta", 0.0)),
            float(getattr(self.cfg, "x_group_completion_span_cap", 240.0)),
            float(getattr(self.cfg, "x_group_completion_span_cap", 240.0)) * 4.0,
        )
        station_overload_signal = self._normalized_excess_ratio(
            float(f1_result.station_cmax),
            baseline=anchor_station_cmax,
            scale=anchor_station_cmax,
        )

        p_x_affinity = float(affinity_signal + template_signal)
        p_x_route = float(route_conflict_signal + route_span_signal + route_tail_signal)
        p_x_time = float(finish_dispersion_signal + completion_span_signal + station_overload_signal)

        local_obj = float(f1_result.proxy_z)
        prox_penalty = float(max(0.0, float(getattr(proposal, "touched_subtask_count", 0.0)))) * float(
            getattr(self.cfg, "x_prox_weight", 0.25)
        )
        coupling_penalty = (
            anchor_scale * (
                float(getattr(self.cfg, "lambda_x_affinity", 0.5)) * p_x_affinity
                + float(getattr(self.cfg, "lambda_x_route", 0.5)) * p_x_route
                + float(getattr(self.cfg, "lambda_x_time", 0.5)) * p_x_time
            )
        )
        augmented_obj = float(
            local_obj
            + coupling_penalty
            + float(self._trust_region_tau("X")) * prox_penalty
        )
        return {
            "local_obj": float(local_obj),
            "coupling_penalty": float(coupling_penalty),
            "prox_penalty": float(prox_penalty),
            "augmented_obj": float(augmented_obj),
            "predicted_proxy_z": float(augmented_obj),
            "predicted_proxy_delta": float(anchor_z - augmented_obj),
            "win_prob": 0.5,
            "prediction_uncertainty": 0.0,
            "residual_hat": 0.0,
            "residual_std": 0.0,
            "couplings": {
                "x_affinity": float(anchor_scale * p_x_affinity),
                "x_route": float(anchor_scale * p_x_route),
                "x_time": float(anchor_scale * p_x_time),
            },
            "x_classic_cx": float(local_obj),
            "x_classic_p_affinity": float(p_x_affinity),
            "x_classic_p_route": float(p_x_route),
            "x_classic_p_time": float(p_x_time),
            "x_classic_dx": float(prox_penalty),
            **soft_penalties,
        }

    def _classic_z_prox_penalty(self) -> float:
        tasks = self._collect_all_tasks()
        used_stack_ids = {
            int(getattr(task, "target_stack_id", -1))
            for task in tasks
            if int(getattr(task, "target_stack_id", -1)) >= 0
        }
        anchor_sig = self.anchor_reference.get("anchor_stack_set", set())
        mode_set = {str(getattr(task, "operation_mode", "")) for task in tasks}
        prox_penalty = float(abs(len(tasks) - int(self.anchor_reference.get("anchor_task_count", 0))))
        prox_penalty += 0.25 * float(len(used_stack_ids.symmetric_difference(anchor_sig)))
        prox_penalty += 0.25 * float(len(mode_set.symmetric_difference(self.anchor_reference.get("anchor_mode_set", set()))))
        return float(prox_penalty)

    def _score_z_candidate_classic_soft(
        self,
        score: Dict[str, Any],
        f1_result: F1EvalResult,
    ) -> Dict[str, Any]:
        anchor_z = float(self.anchor_z if math.isfinite(self.anchor_z) else f1_result.proxy_z)
        anchor_scale = max(1.0, anchor_z)
        raw_proxy_z = float((f1_result.extra or {}).get("raw_post_y_proxy_z", f1_result.proxy_z))
        station_cmax = float(f1_result.station_cmax)
        soft_penalties = self._estimate_z_soft_penalties(score, f1_result)
        anchor_station_cmax = max(1.0, float(self._anchor_station_cmax()))
        local_obj = float(raw_proxy_z)

        changed_task_pressure = self._normalized_excess_ratio(
            float(score.get("changed_task_count", 0.0)),
            baseline=0.0,
            scale=max(1.0, float(getattr(self.cfg, "z_local_delta_task_cap", 2))),
        )
        changed_stack_pressure = self._normalized_excess_ratio(
            float(score.get("changed_stack_count", 0.0)),
            baseline=0.0,
            scale=max(1.0, float(getattr(self.cfg, "z_local_delta_stack_cap", 1))),
        )
        mode_pressure = self._normalized_excess_ratio(
            abs(float(score.get("z_candidate_mode_delta", 0.0))),
            baseline=0.0,
            scale=max(1.0, float(getattr(self.cfg, "z_mode_toggle_cap", 1))),
        )
        fallback_pressure = 1.0 if bool(score.get("z_operator_fallback_used", False)) else 0.0
        p_zx = float(changed_task_pressure + changed_stack_pressure + 0.5 * mode_pressure + 0.5 * fallback_pressure)

        station_overload_pressure = self._normalized_excess_ratio(
            station_cmax,
            baseline=anchor_station_cmax,
            scale=anchor_station_cmax,
        )
        wait_pressure = self._normalized_excess_ratio(
            max(float(score.get("z_wait_overflow_estimate", 0.0)), float(f1_result.wait_overflow_total)),
            baseline=0.0,
            scale=anchor_scale,
        )
        p_zy = float(station_overload_pressure + wait_pressure)

        route_gap_pressure = self._normalized_excess_ratio(
            float(score.get("z_route_gap_penalty", 0.0)),
            baseline=0.0,
            scale=anchor_scale,
        )
        arrival_pressure = self._normalized_excess_ratio(
            max(float(score.get("z_arrival_shift_estimate", 0.0)), float(f1_result.arrival_shift_total)),
            baseline=0.0,
            scale=anchor_scale,
        )
        tail_pressure = self._normalized_excess_ratio(
            max(float(score.get("z_route_tail_delta_estimate", 0.0)), float(f1_result.route_tail_delta)),
            baseline=0.0,
            scale=anchor_scale,
        )
        p_zu = float(route_gap_pressure + arrival_pressure + tail_pressure)

        coupling_penalty = (
            anchor_scale * (
                float(self.layer_lambda_weights.get("zx", float(getattr(self.cfg, "lambda_init", 1.0)))) * p_zx
                + float(self.layer_lambda_weights.get("zy", float(getattr(self.cfg, "lambda_init", 1.0)))) * p_zy
                + float(self.layer_lambda_weights.get("zu", float(getattr(self.cfg, "lambda_init", 1.0)))) * p_zu
            )
        )
        prox_penalty = float(self._classic_z_prox_penalty())
        augmented_obj = float(
            local_obj
            + coupling_penalty
            + float(self._trust_region_tau("Z")) * prox_penalty
        )
        return {
            "local_obj": float(local_obj),
            "coupling_penalty": float(coupling_penalty),
            "prox_penalty": float(prox_penalty),
            "augmented_obj": float(augmented_obj),
            "predicted_proxy_z": float(augmented_obj),
            "predicted_proxy_delta": float(anchor_z - augmented_obj),
            "win_prob": 0.5,
            "prediction_uncertainty": 0.0,
            "residual_hat": 0.0,
            "residual_std": 0.0,
            "couplings": {
                "zx": float(anchor_scale * p_zx),
                "zy": float(anchor_scale * p_zy),
                "zu": float(anchor_scale * p_zu),
            },
            "z_classic_cz": float(local_obj),
            "z_classic_p_zx": float(p_zx),
            "z_classic_p_zy": float(p_zy),
            "z_classic_p_zu": float(p_zu),
            "z_classic_dz": float(prox_penalty),
            **soft_penalties,
        }

    def _apply_restricted_y_rebalance(
        self,
        restrict_subtask_ids: Optional[Set[int]] = None,
        restrict_station_ids: Optional[Set[int]] = None,
    ) -> bool:
        if self.problem is None:
            return False
        context = self._build_sp2_layer_context()
        target_ids = {int(x) for x in (restrict_subtask_ids or set()) if int(x) >= 0}
        if not target_ids:
            return False
        all_subtasks = list(getattr(self.problem, "subtask_list", []) or [])
        target_subtasks = [st for st in all_subtasks if int(getattr(st, "id", -1)) in target_ids]
        if not target_subtasks:
            return False
        allowed_station_ids = {int(x) for x in (restrict_station_ids or set()) if int(x) >= 0}
        if not allowed_station_ids:
            for st in target_subtasks:
                sid = int(getattr(st, "assigned_station_id", -1))
                if sid >= 0:
                    allowed_station_ids.add(sid)
                anchor_sid = int(context.anchor_station_by_subtask.get(int(getattr(st, "id", -1)), -1))
                if anchor_sid >= 0:
                    allowed_station_ids.add(anchor_sid)
        if not allowed_station_ids:
            return False
        station_window = max(1, int(getattr(self.cfg, "xz_f1_rebalance_station_window", 2)))
        allowed_station_ids = set(sorted(allowed_station_ids)[: max(1, station_window)])
        station_ids = sorted(allowed_station_ids)
        station_finish: Dict[int, float] = {sid: 0.0 for sid in station_ids}
        station_loads: Dict[int, float] = {sid: 0.0 for sid in station_ids}
        unaffected_by_station: Dict[int, List[Any]] = defaultdict(list)
        for st in all_subtasks:
            sid = int(getattr(st, "assigned_station_id", -1))
            if sid not in station_finish:
                continue
            if int(getattr(st, "id", -1)) in target_ids:
                continue
            unaffected_by_station[sid].append(st)
        for sid, rows in unaffected_by_station.items():
            rows.sort(key=lambda item: (int(getattr(item, "station_sequence_rank", 10 ** 6)), int(getattr(item, "id", -1))))
            current = 0.0
            for st in rows:
                arrival = float(context.arrival_time_by_subtask.get(int(getattr(st, "id", -1)), self._estimate_subtask_arrival(st)))
                proc = float(context.processing_time_by_subtask.get(int(getattr(st, "id", -1)), self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)))
                current = max(current, arrival) + proc
                station_loads[sid] += proc
            station_finish[sid] = current
        ordered_targets = sorted(
            target_subtasks,
            key=lambda st: (
                float(context.arrival_time_by_subtask.get(int(getattr(st, "id", -1)), self._estimate_subtask_arrival(st))),
                -float(context.processing_time_by_subtask.get(int(getattr(st, "id", -1)), self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))),
                int(getattr(st, "id", -1)),
            ),
        )
        rank_window = max(0, int(getattr(self.cfg, "xz_f1_rebalance_rank_window", 2)))
        station_sequences: Dict[int, int] = {sid: len(unaffected_by_station.get(sid, [])) for sid in station_ids}
        changed = False
        for st in ordered_targets:
            sid0 = int(getattr(st, "assigned_station_id", -1))
            arrival = float(context.arrival_time_by_subtask.get(int(getattr(st, "id", -1)), self._estimate_subtask_arrival(st)))
            proc = float(context.processing_time_by_subtask.get(int(getattr(st, "id", -1)), self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)))
            best = None
            for sid in station_ids:
                rank = int(station_sequences[sid])
                start = max(float(station_finish[sid]), arrival)
                finish = start + proc
                candidate_loads = dict(station_loads)
                candidate_loads[sid] += proc
                load_gap = max(candidate_loads.values()) - min(candidate_loads.values()) if candidate_loads else 0.0
                pref = float(self.sp2._local_station_preference(st, sid, context))
                prox_station = float(self.sp2._local_anchor_station_penalty(st, sid, context))
                anchor_rank = int(context.anchor_rank_by_subtask.get(int(getattr(st, "id", -1)), rank))
                prox_rank = float(abs(rank - anchor_rank))
                objective = (
                    float(finish)
                    + float(context.lambda_yx) * pref
                    + 0.25 * max(0.0, start - arrival)
                    + 0.1 * load_gap
                    + 0.2 * prox_station
                    + 0.1 * max(0.0, prox_rank - rank_window)
                )
                candidate = (objective, finish, sid, rank, start)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                continue
            _, finish, best_sid, best_rank, start = best
            st.assigned_station_id = int(best_sid)
            st.station_sequence_rank = int(best_rank)
            st.estimated_process_start_time = float(start)
            station_finish[int(best_sid)] = float(finish)
            station_loads[int(best_sid)] += proc
            station_sequences[int(best_sid)] += 1
            changed = changed or (int(best_sid) != sid0)
        self._normalize_station_assignments()
        return bool(changed)

    def _subtask_arrival_from_tasks(self) -> Dict[int, float]:
        arrival_by_subtask: Dict[int, float] = defaultdict(float)
        for task in self._collect_all_tasks():
            sid = int(getattr(task, "sub_task_id", -1))
            arrival_by_subtask[sid] = max(arrival_by_subtask.get(sid, 0.0), float(getattr(task, "arrival_time_at_station", 0.0)))
        return {int(k): float(v) for k, v in arrival_by_subtask.items()}

    def _normalize_task_sequences(self):
        grouped: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
        for task in self._collect_all_tasks():
            rid = int(getattr(task, "robot_id", -1))
            trip_id = int(getattr(task, "trip_id", 0))
            grouped[(rid, trip_id)].append(task)
        for (_, _), rows in grouped.items():
            rows.sort(key=lambda task: (int(getattr(task, "robot_visit_sequence", 0)), int(getattr(task, "task_id", -1))))
            for seq, task in enumerate(rows):
                task.robot_visit_sequence = int(seq)

    def _assign_anchor_route_skeleton_for_x(self) -> float:
        anchor_subtasks = self.anchor_reference.get("anchor_subtask_profile", {}) or {}
        anchor_tasks = self.anchor_reference.get("anchor_task_profile", {}) or {}
        if not anchor_subtasks or not anchor_tasks:
            return 0.0
        used_anchor_task_ids: Set[int] = set()
        matched_subtasks = 0
        total_subtasks = 0
        for st in getattr(self.problem, "subtask_list", []) or []:
            total_subtasks += 1
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            sku_ids = {
                int(getattr(sku, "id", -1))
                for sku in getattr(st, "unique_sku_list", []) or []
                if int(getattr(sku, "id", -1)) >= 0
            }
            candidates = [
                (subtask_id, profile)
                for subtask_id, profile in anchor_subtasks.items()
                if int(profile.get("order_id", -1)) == order_id
            ]
            best_anchor_subtask_id = -1
            best_overlap = -1.0
            for subtask_id, profile in candidates:
                anchor_skus = set(int(x) for x in profile.get("sku_ids", []) or [])
                overlap = float(len(anchor_skus & sku_ids)) / max(1.0, float(len(anchor_skus | sku_ids)))
                if overlap > best_overlap + 1e-9:
                    best_overlap = overlap
                    best_anchor_subtask_id = int(subtask_id)
            if best_anchor_subtask_id >= 0:
                matched_subtasks += 1
            anchor_profile = anchor_subtasks.get(best_anchor_subtask_id, {})
            dominant_robot = int(anchor_profile.get("dominant_robot_id", -1))
            anchor_trip_ids = list(anchor_profile.get("trip_ids", []) or [])
            fallback_trip_id = int(anchor_trip_ids[0][1]) if anchor_trip_ids else 0
            anchor_task_candidates = [
                (task_id, profile)
                for task_id, profile in anchor_tasks.items()
                if int(profile.get("subtask_id", -1)) == int(best_anchor_subtask_id)
            ]
            anchor_task_candidates.sort(key=lambda item: (int(item[1].get("robot_visit_sequence", 0)), int(item[0])))
            for offset, task in enumerate(sorted(getattr(st, "execution_tasks", []) or [], key=lambda item: (int(getattr(item, "target_stack_id", -1)), int(getattr(item, "task_id", -1))))):
                best_task_id = -1
                best_match = -float("inf")
                for task_id, profile in anchor_task_candidates:
                    if int(task_id) in used_anchor_task_ids:
                        continue
                    match_score = 0.0
                    if int(profile.get("stack_id", -1)) == int(getattr(task, "target_stack_id", -1)):
                        match_score += 2.0
                    if str(profile.get("operation_mode", "")) == str(getattr(task, "operation_mode", "")):
                        match_score += 1.0
                    match_score -= 0.1 * abs(int(profile.get("robot_visit_sequence", 0)) - offset)
                    if match_score > best_match + 1e-9:
                        best_match = match_score
                        best_task_id = int(task_id)
                if best_task_id >= 0:
                    profile = anchor_tasks.get(best_task_id, {})
                    task.robot_id = int(profile.get("robot_id", dominant_robot))
                    task.trip_id = int(profile.get("trip_id", fallback_trip_id))
                    task.robot_visit_sequence = int(profile.get("robot_visit_sequence", offset))
                    used_anchor_task_ids.add(best_task_id)
                else:
                    task.robot_id = int(dominant_robot)
                    task.trip_id = int(fallback_trip_id)
                    task.robot_visit_sequence = int(offset)
        self._normalize_task_sequences()
        return float(matched_subtasks / max(1, total_subtasks))


    def _collect_z_changed_task_profiles(self) -> Tuple[int, Set[Tuple[int, int]]]:
        anchor_tasks = self.anchor_reference.get("anchor_task_profile", {}) or {}
        changed = 0
        affected: Set[Tuple[int, int]] = set()
        for task in self._collect_all_tasks():
            task_id = int(getattr(task, "task_id", -1))
            anchor_profile = anchor_tasks.get(task_id)
            curr_signature = (
                int(getattr(task, "sub_task_id", -1)),
                int(getattr(task, "target_stack_id", -1)),
                str(getattr(task, "operation_mode", "")),
                tuple(int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0),
                tuple(int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0),
                tuple(int(x) for x in (getattr(task, "sort_layer_range", ()) or ()) if int(x) >= 0),
                round(float(getattr(task, "robot_service_time", 0.0)), 6),
                round(float(getattr(task, "station_service_time", 0.0)), 6),
            )
            anchor_signature = None
            if anchor_profile is not None:
                anchor_signature = (
                    int(anchor_profile.get("subtask_id", -1)),
                    int(anchor_profile.get("stack_id", -1)),
                    str(anchor_profile.get("operation_mode", "")),
                    tuple(int(x) for x in (anchor_profile.get("target_tote_ids", []) or []) if int(x) >= 0),
                    tuple(int(x) for x in (anchor_profile.get("hit_tote_ids", []) or []) if int(x) >= 0),
                    tuple(int(x) for x in (anchor_profile.get("sort_layer_range", ()) or ()) if int(x) >= 0),
                    round(float(anchor_profile.get("robot_service_time", 0.0)), 6),
                    round(float(anchor_profile.get("station_service_time", 0.0)), 6),
                )
            if anchor_signature != curr_signature:
                changed += 1
                rid = int(getattr(task, "robot_id", -1))
                trip_id = int(getattr(task, "trip_id", 0))
                if rid >= 0:
                    affected.add((rid, trip_id))
        return int(changed), affected



    def _maybe_rank_hit_top1(self, layer: str, candidate: Dict[str, Any], actual_reduction: float):
        layer = str(layer).upper()
        if layer not in {"X", "Z"}:
            return
        state = self._surrogate_state(layer)
        if int(candidate.get("rank", 0)) == 1:
            state.rank_top1_total += 1
            if float(actual_reduction) > float(getattr(self.cfg, "acceptance_min_actual_improve", 1e-6)):
                state.rank_hit_top1_count += 1


    def _x_candidate_move_too_large(self, score: Dict[str, Any]) -> bool:
        changed_orders = int(round(float(score.get("changed_orders", 0.0) or 0.0)))
        changed_pairs = int(round(float(score.get("x_changed_assignment_pair_count", score.get("changed_assignment_pairs", 0.0)) or 0.0)))
        delta_subtask_count = abs(int(round(float(score.get("delta_subtask_count", 0.0) or 0.0))))
        moved_sku_count = int(round(float(score.get("x_moved_sku_count", 0.0) or 0.0)))
        destroy_size = int(round(float(score.get("x_destroy_size_effective", score.get("x_destroy_size", 0.0)) or 0.0)))
        effective_changed_pairs = changed_pairs
        if destroy_size > 0:
            effective_changed_pairs = min(effective_changed_pairs, destroy_size)
        if moved_sku_count > 0:
            effective_changed_pairs = min(effective_changed_pairs, moved_sku_count)
        return bool(
            changed_orders != 1
            or effective_changed_pairs > max(2, int(getattr(self.cfg, "x_bom_destroy_max_lines", 2)))
            or delta_subtask_count > 1
            or moved_sku_count > max(2, int(getattr(self.cfg, "x_bom_destroy_max_lines", 2)))
        )


    def _compute_current_z_local_proxy_baseline(self) -> float:
        if self.problem is None or self.sim is None:
            return float(self.anchor_z if math.isfinite(self.anchor_z) else self.work_z)
        replayed = bool(self._replay_u_routes())
        if not replayed:
            return float(self.anchor_z if math.isfinite(self.anchor_z) else self.work_z)
        arrival_by_subtask = self._subtask_arrival_from_tasks()
        proc_by_subtask = {
            int(getattr(st, "id", -1)): float(self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
            for st in getattr(self.problem, "subtask_list", []) or []
        }
        self._recompute_station_schedule(arrival_by_subtask=arrival_by_subtask, proc_by_subtask=proc_by_subtask)
        return float(self.sim.calculate_with_existing_arrivals())


    def _classic_soft_hard_gate(
        self,
        layer: str,
        f1_result: F1EvalResult,
        score: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        layer = str(layer).upper()
        if layer != "Z":
            return self._surrogate_hard_gate(layer, f1_result, score=score)
        if not bool(f1_result.replayed_route) or not math.isfinite(float(f1_result.proxy_z)):
            return False, "f1_replay_failed"
        return True, ""

    def _classic_soft_verify_signal(
        self,
        layer: str,
        score: Dict[str, Any],
        anchor_z: float,
        min_improve: float,
    ) -> Tuple[bool, str]:
        layer = str(layer).upper()
        anchor_z = float(anchor_z)
        min_improve = max(1e-9, float(min_improve))
        if layer == "X":
            proxy_z = float(score.get("x_f1_proxy_z", float("inf")))
            if math.isfinite(proxy_z) and proxy_z < anchor_z - min_improve:
                return True, "x_proxy_improve"
            if math.isfinite(proxy_z) and proxy_z <= anchor_z + min_improve:
                return True, "x_proxy_nonworse"
            return False, ""
        if layer == "Z":
            raw_proxy_z = float(score.get("z_classic_cz", score.get("z_f1_proxy_z", float("inf"))))
            if math.isfinite(raw_proxy_z) and raw_proxy_z < anchor_z - min_improve:
                return True, "z_raw_proxy_improve"
            return False, ""
        return False, ""


    def _process_xz_candidate_pool(
        self,
        layer: str,
        candidate_pool: List[Dict[str, Any]],
        baseline: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> Tuple[Optional[SolutionSnapshot], Optional[Dict[str, Any]]]:
        layer = str(layer).upper()
        evaluator_mode = self._xz_evaluator_mode()
        neural_mode = bool(evaluator_mode == "neural")
        state = self._surrogate_state(layer) if neural_mode else None
        warmup_limit = max(1, int(getattr(self.cfg, "surrogate_warmup_samples", 8)))
        use_warmup = bool(neural_mode and state is not None and int(state.warmup_count) < warmup_limit)
        anchor_z = float(self.anchor_z if math.isfinite(self.anchor_z) else baseline.get("augmented_obj", float("inf")))
        band_k = float(getattr(self.cfg, "surrogate_uncertainty_k", 1.25))
        min_improve = float(getattr(self.cfg, "surrogate_min_improve_abs", 1.0))
        runtime["xz_evaluator_mode"] = str(evaluator_mode)
        runtime["surrogate_warmup_active"] = bool(use_warmup) if neural_mode else False
        runtime[f"{layer.lower()}_surrogate_warmup_samples_seen"] = float(state.warmup_count) if neural_mode and state is not None else 0.0
        runtime[f"{layer.lower()}_valid_surrogate_samples_seen"] = float(self._surrogate_valid_sample_count(layer)) if neural_mode else 0.0
        runtime[f"{layer.lower()}_surrogate_trusted"] = bool(self._surrogate_trusted(layer)) if neural_mode else False
        runtime["classic_verify_triggered"] = False
        runtime["classic_verify_reason"] = ""
        runtime["z_positive_mining_triggered"] = False
        runtime["z_positive_mining_score"] = float("nan")
        runtime["z_positive_candidate_eligible"] = False
        runtime["z_positive_candidate_operator"] = ""
        runtime["z_positive_candidate_eligibility_reason"] = ""
        runtime["z_positive_candidate_eligible_count"] = 0.0

        for item in candidate_pool:
            score = dict(item.get("score", {}) or {})
            f0_features = dict(score.get("_f0_features", {}) or {})
            prior_score = float(score.get(f"{layer.lower()}_f0_rank_score", score.get("_prior_score", float("inf"))))
            if neural_mode and state is not None:
                f0_win_prob, f0_win_prob_std = self._rank_probability_from_state(state, f0_features)
            else:
                f0_win_prob, f0_win_prob_std = 0.5, 0.0
            score["_prior_score"] = float(prior_score)
            score["_f0_win_prob"] = float(f0_win_prob)
            score["_f0_win_prob_std"] = float(f0_win_prob_std)
            item["score"] = score

        if neural_mode:
            candidate_pool.sort(
                key=(
                    (lambda item: (
                        float((item.get("score", {}) or {}).get("_prior_score", float("inf"))),
                        -float((item.get("score", {}) or {}).get("_f0_win_prob", 0.5)),
                        int(item.get("operator_rank", 10 ** 9)),
                    ))
                    if use_warmup
                    else
                    (lambda item: (
                        -float((item.get("score", {}) or {}).get("_f0_win_prob", 0.5)),
                        float((item.get("score", {}) or {}).get("_prior_score", float("inf"))),
                        int(item.get("operator_rank", 10 ** 9)),
                    ))
                )
            )
        else:
            candidate_pool.sort(
                key=lambda item: (
                    float((item.get("score", {}) or {}).get("_prior_score", float("inf"))),
                    int(item.get("operator_rank", 10 ** 9)),
                )
            )
        runtime["candidate_count"] = float(len(candidate_pool))
        runtime["candidate_topk"] = float(len(candidate_pool))
        runtime["surrogate_buffer_size"] = float(len(state.training_buffer)) if neural_mode and state is not None else 0.0
        if layer == "X":
            runtime["x_candidates_generated_count"] = float(len(candidate_pool))
        for f0_rank, item in enumerate(candidate_pool, start=1):
            item["f0_rank"] = int(f0_rank)
        shortlisted = list(candidate_pool)
        if layer == "X":
            runtime["x_shortlist_count"] = float(len(shortlisted))
            runtime["x_unique_candidate_count"] = float(len({self._x_equivalent_candidate_signature((item.get("score", {}) or {}).get("_proposal", self._extract_x_split_solution())) for item in shortlisted}))

        scored_candidates: List[Dict[str, Any]] = []
        seen_equivalent_signatures: Set[str] = set()
        hard_reject_count = 0
        dedup_count = 0
        f1_candidates = list(shortlisted)
        for item in f1_candidates:
            snap = item.get("snapshot")
            if snap is None:
                continue
            self.restore_snapshot(snap)
            score = dict(item.get("score", {}) or {})
            signature = str(score.get("_candidate_signature", ""))
            f0_features = dict(score.get("_f0_features", {}) or {})
            if layer == "X":
                proposal = score.get("_proposal", self._extract_x_split_solution())
                equivalent_signature = self._x_equivalent_candidate_signature(proposal)
                score["x_equivalent_candidate_deduped"] = False
                if equivalent_signature in seen_equivalent_signatures:
                    score["x_equivalent_candidate_deduped"] = True
                    score["x_candidate_reject_stage"] = "pre_f1"
                    score["x_candidate_reject_reason"] = "equivalent_candidate_deduped"
                    item["score"] = score
                    dedup_count += 1
                    continue
                seen_equivalent_signatures.add(equivalent_signature)
                template_counts = self._x_template_change_counts(proposal)
                score.update(template_counts)
                t_f1 = time.perf_counter()
                f1_result = self._evaluate_x_candidate_with_local_replay(signature, proposal=proposal)
                runtime["x_f1_time_sec"] = float(runtime.get("x_f1_time_sec", 0.0)) + float(time.perf_counter() - t_f1)
                f1_features = self._build_x_f1_features(f1_result, f0_features)
            else:
                t_f1 = time.perf_counter()
                f1_result = self._evaluate_z_candidate_with_local_replay(signature, move_meta=dict(item.get("meta", {}) or {}))
                runtime["z_f1_time_sec"] = float(runtime.get("z_f1_time_sec", 0.0)) + float(time.perf_counter() - t_f1)
                f1_features = self._build_z_f1_features(f1_result, f0_features)
                guardrails = self._estimate_z_candidate_guardrails()
                score["z_route_gap_penalty"] = float((item.get("meta", {}) or {}).get("z_route_gap_penalty", 0.0))
                score["z_operator_fallback_used"] = bool((item.get("meta", {}) or {}).get("z_operator_fallback_used", False))
                score["z_arrival_shift_estimate"] = float(guardrails.get("arrival_shift_estimate", 0.0))
                score["z_wait_overflow_estimate"] = float(guardrails.get("wait_overflow_estimate", 0.0))
                score["z_route_tail_delta_estimate"] = float(guardrails.get("route_tail_delta_estimate", 0.0))
                score["changed_task_count"] = float(guardrails.get("changed_task_count", f0_features.get("changed_task_count", 0.0)))
                score["changed_stack_count"] = float(guardrails.get("changed_stack_count", f0_features.get("changed_stack_count", 0.0)))
            invalid_reason = self._f1_invalid_reason(f1_result)
            if invalid_reason:
                if layer == "X":
                    score["x_f1_proxy_z"] = float(f1_result.proxy_z)
                    score["x_f1_station_cmax"] = float(f1_result.station_cmax)
                    score["x_f1_route_tail"] = float(f1_result.route_tail)
                    score["x_f1_mapping_coverage"] = float(f1_result.mapping_coverage)
                    score["x_f1_used_sp2"] = bool(f1_result.used_sp2)
                    score["x_f1_used_sp3"] = bool(f1_result.used_sp3)
                    score["x_f1_replayed_route"] = bool(f1_result.replayed_route)
                    score["x_candidate_reject_stage"] = "pre_surrogate"
                    score["x_candidate_reject_reason"] = str(invalid_reason)
                    runtime["x_f1_invalid"] = float(runtime.get("x_f1_invalid", 0.0)) + 1.0
                    self.x_f1_invalid_count = int(getattr(self, "x_f1_invalid_count", 0)) + 1
                else:
                    score["z_f1_proxy_z"] = float(f1_result.proxy_z)
                    score["z_f1_arrival_shift_total"] = float(f1_result.arrival_shift_total)
                    score["z_f1_wait_overflow_total"] = float(f1_result.wait_overflow_total)
                    score["z_f1_route_tail_delta"] = float(f1_result.route_tail_delta)
                    score["z_f1_replayed_trip_count"] = float(f1_result.replayed_trip_count)
                    score["z_f1_used_full_replay"] = bool(f1_result.used_full_replay)
                    score["z_candidate_reject_stage"] = "pre_surrogate"
                    score["z_candidate_reject_reason"] = str(invalid_reason)
                    runtime["z_f1_invalid"] = float(runtime.get("z_f1_invalid", 0.0)) + 1.0
                    self.z_f1_invalid_count = int(getattr(self, "z_f1_invalid_count", 0)) + 1
                item["score"] = score
                continue
            hard_gate_ok, hard_gate_reason = (
                self._surrogate_hard_gate(layer, f1_result, score=score)
                if neural_mode
                else self._classic_soft_hard_gate(layer, f1_result, score=score)
            )
            if (not neural_mode) and (not hard_gate_ok):
                if layer == "X":
                    score["x_f1_proxy_z"] = float(f1_result.proxy_z)
                    score["x_f1_station_cmax"] = float(f1_result.station_cmax)
                    score["x_f1_route_tail"] = float(f1_result.route_tail)
                    score["x_f1_mapping_coverage"] = float(f1_result.mapping_coverage)
                    score["x_f1_used_sp2"] = bool(f1_result.used_sp2)
                    score["x_f1_used_sp3"] = bool(f1_result.used_sp3)
                    score["x_f1_replayed_route"] = bool(f1_result.replayed_route)
                    score["x_candidate_reject_stage"] = "pre_classic_soft"
                    score["x_candidate_reject_reason"] = str(hard_gate_reason)
                    runtime["x_candidate_hard_reject_count"] = float(runtime.get("x_candidate_hard_reject_count", 0.0)) + 1.0
                else:
                    score["z_f1_proxy_z"] = float(f1_result.proxy_z)
                    score["z_f1_arrival_shift_total"] = float(f1_result.arrival_shift_total)
                    score["z_f1_wait_overflow_total"] = float(f1_result.wait_overflow_total)
                    score["z_f1_route_tail_delta"] = float(f1_result.route_tail_delta)
                    score["z_f1_replayed_trip_count"] = float(f1_result.replayed_trip_count)
                    score["z_f1_used_full_replay"] = bool(f1_result.used_full_replay)
                    score["z_candidate_reject_stage"] = "pre_classic_soft"
                    score["z_candidate_reject_reason"] = str(hard_gate_reason)
                    runtime["z_candidate_hard_reject_count"] = float(runtime.get("z_candidate_hard_reject_count", 0.0)) + 1.0
                hard_reject_count += 1
                item["score"] = score
                continue
            if neural_mode and state is not None:
                prediction = self._surrogate_prediction_from_state(
                    state=state,
                    prior_score=float(score.get("_prior_score", float("inf"))),
                    f0_features=f0_features,
                    f1_features=f1_features,
                    proxy_z=float(f1_result.proxy_z),
                )
                soft_penalties = self._estimate_x_soft_penalties(proposal, f1_result) if layer == "X" else self._estimate_z_soft_penalties(score, f1_result)
                predicted_proxy_z = float(prediction.predicted_proxy_z + float(soft_penalties.get("x_soft_penalty_total", 0.0)) + float(soft_penalties.get("z_soft_penalty_total", 0.0)))
                pred_upper = float(predicted_proxy_z + band_k * max(0.0, prediction.uncertainty))
                surrogate_positive = bool(pred_upper < anchor_z - min_improve)
                score.update({
                    "layer": layer,
                    "local_obj": float(f1_result.proxy_z),
                    "coupling_penalty": 0.0,
                    "prox_penalty": 0.0,
                    "augmented_obj": float(predicted_proxy_z),
                    "couplings": {},
                    "proposal_pass_fast_gate": True,
                    "fast_gate_reason": "",
                    "x_fast_gate_band": "",
                    "fidelity_selected": "F1",
                    "predicted_proxy_z": float(predicted_proxy_z),
                    "predicted_proxy_delta": float(anchor_z - predicted_proxy_z),
                    "win_prob": float(prediction.win_prob),
                    "prediction_uncertainty": float(prediction.uncertainty),
                    "residual_hat": float(prediction.residual_hat),
                    "residual_std": float(prediction.residual_std),
                    "_prediction_upper": float(pred_upper),
                    "_prediction_lower": float(predicted_proxy_z - band_k * max(0.0, prediction.uncertainty)),
                    "_surrogate_positive": bool(surrogate_positive),
                    "_hard_gate_ok": bool(hard_gate_ok),
                    "_hard_gate_reason": str(hard_gate_reason),
                    "_f1_features": dict(f1_features),
                })
                score.update(soft_penalties)
            else:
                classic_score = (
                    self._score_x_candidate_classic_soft(proposal, score, f1_result)
                    if layer == "X"
                    else self._score_z_candidate_classic_soft(score, f1_result)
                )
                predicted_proxy_z = float(classic_score.get("predicted_proxy_z", float("inf")))
                surrogate_positive = bool(predicted_proxy_z < anchor_z - min_improve)
                score.update({
                    "layer": layer,
                    "proposal_pass_fast_gate": True,
                    "fast_gate_reason": "",
                    "x_fast_gate_band": "",
                    "fidelity_selected": "F1_CLASSIC",
                    "_prediction_upper": float(predicted_proxy_z),
                    "_prediction_lower": float(predicted_proxy_z),
                    "_surrogate_positive": bool(surrogate_positive),
                    "_hard_gate_ok": True,
                    "_hard_gate_reason": "",
                    "_f1_features": dict(f1_features),
                })
                score.update(classic_score)
            if layer == "X":
                score["x_f1_proxy_z"] = float(f1_result.proxy_z)
                score["x_f1_station_cmax"] = float(f1_result.station_cmax)
                score["x_f1_route_tail"] = float(f1_result.route_tail)
                score["x_f1_mapping_coverage"] = float(f1_result.mapping_coverage)
                score["x_f1_used_sp2"] = bool(f1_result.used_sp2)
                score["x_f1_used_sp3"] = bool(f1_result.used_sp3)
                score["x_f1_replayed_route"] = bool(f1_result.replayed_route)
                score["x_f1_pre_y_proxy_z"] = float((f1_result.extra or {}).get("pre_y_proxy_z", f1_result.proxy_z))
                score["x_f1_post_y_proxy_z"] = float((f1_result.extra or {}).get("post_y_proxy_z", f1_result.proxy_z))
                score["x_f1_rebalance_effective"] = bool((f1_result.extra or {}).get("x_f1_rebalance_effective", False))
                score["x_candidate_reject_stage"] = ""
                score["x_candidate_reject_reason"] = ""
            else:
                score["z_f1_proxy_z"] = float(f1_result.proxy_z)
                score["z_f1_arrival_shift_total"] = float(f1_result.arrival_shift_total)
                score["z_f1_wait_overflow_total"] = float(f1_result.wait_overflow_total)
                score["z_f1_route_tail_delta"] = float(f1_result.route_tail_delta)
                score["z_f1_replayed_trip_count"] = float(f1_result.replayed_trip_count)
                score["z_f1_used_full_replay"] = bool(f1_result.used_full_replay)
                score["z_f1_pre_y_proxy_z"] = float((f1_result.extra or {}).get("pre_y_proxy_z", f1_result.proxy_z))
                score["z_f1_post_y_proxy_z"] = float((f1_result.extra or {}).get("post_y_proxy_z", f1_result.proxy_z))
                score["z_candidate_reject_stage"] = ""
                score["z_candidate_reject_reason"] = ""
                score.update(self._score_z_positive_mining_candidate(str(item.get("operator", "")), score, f1_result))
            item["score"] = score
            scored_candidates.append(item)

        viable_candidates = list(scored_candidates)
        ranked_candidates = list(scored_candidates)
        if layer == "X":
            runtime["x_post_gate_candidate_count"] = float(len(viable_candidates))
        else:
            runtime["z_feasible_candidate_count"] = float(len(viable_candidates))
        ranked_candidates.sort(
            key=lambda item: (
                float((item.get("score", {}) or {}).get("_prediction_upper", float("inf"))),
                float((item.get("score", {}) or {}).get("augmented_obj", float("inf"))),
                -float((item.get("score", {}) or {}).get("win_prob", 0.0)),
                float((item.get("score", {}) or {}).get("_prior_score", float("inf"))),
                int(item.get("f0_rank", 10 ** 9)),
            )
        )
        for rank_idx, item in enumerate(ranked_candidates, start=1):
            item["rank"] = int(rank_idx)

        if layer == "X":
            runtime["x_f1_eval_count"] = float(len(scored_candidates))
            runtime["x_fast_gate_pass_count"] = float(len(viable_candidates))
            runtime["x_candidate_hard_reject_count"] = float(hard_reject_count)
            runtime["x_equivalent_dedup_count"] = float(dedup_count)
            candidate_details = []
            scored_map = {
                str((item.get("score", {}) or {}).get("_candidate_signature", "")): item
                for item in scored_candidates
            }
            for item in candidate_pool:
                score = dict(item.get("score", {}) or {})
                sig = str(score.get("_candidate_signature", ""))
                scored_item = scored_map.get(sig)
                scored_score = dict((scored_item.get("score", {}) or {})) if scored_item is not None else {}
                pruned_reason = ""
                if sig not in scored_map:
                    pruned_reason = str(score.get("x_candidate_reject_reason", "")) or "not_scored"
                elif not bool(scored_score.get("_surrogate_positive", False)):
                    pruned_reason = "surrogate_negative" if neural_mode else "classic_soft_negative"
                candidate_details.append({
                    "rank": int(item.get("f0_rank", 0)),
                    "destroy_operator": str(score.get("x_destroy_operator", "")),
                    "repair_operator": str(score.get("x_repair_operator", "")),
                    "combined_operator": str(item.get("operator", "")),
                    "destroy_size": float(score.get("x_destroy_size", 0.0)),
                    "local_obj": float(scored_score.get("local_obj", float("nan"))),
                    "augmented_obj": float(scored_score.get("augmented_obj", float("nan"))),
                    "x_f0_rank_score": float(score.get("x_f0_rank_score", float("nan"))),
                    "predicted_proxy_z": float(scored_score.get("predicted_proxy_z", float("nan"))),
                    "prediction_uncertainty": float(scored_score.get("prediction_uncertainty", float("nan"))),
                    "proposal_pass_fast_gate": bool(scored_score.get("proposal_pass_fast_gate", False)),
                    "pruned_reason": pruned_reason,
                    "would_shortlist": True,
                    "would_pass_fast_gate": bool(scored_score.get("proposal_pass_fast_gate", False)),
                    "x_candidate_reject_stage": str(scored_score.get("x_candidate_reject_stage", "")),
                    "x_candidate_reject_reason": str(scored_score.get("x_candidate_reject_reason", "")),
                    "x_equivalent_candidate_deduped": bool(scored_score.get("x_equivalent_candidate_deduped", False)),
                    "x_classic_cx": float(scored_score.get("x_classic_cx", float("nan"))),
                    "x_classic_p_affinity": float(scored_score.get("x_classic_p_affinity", float("nan"))),
                    "x_classic_p_route": float(scored_score.get("x_classic_p_route", float("nan"))),
                    "x_classic_p_time": float(scored_score.get("x_classic_p_time", float("nan"))),
                    "x_classic_dx": float(scored_score.get("x_classic_dx", float("nan"))),
                    "z_evaluated": None,
                })
            runtime["x_candidates_pruned_count"] = float(sum(1 for row in candidate_details if str(row.get("pruned_reason", "")).strip()))
            runtime["x_candidate_details"] = candidate_details
        else:
            runtime["z_f1_eval_count"] = float(len(scored_candidates))
            runtime["z_candidate_hard_reject_count"] = float(hard_reject_count)
            runtime["z_operator_ban_count"] = float(len(self.z_operator_subtask_bans))
            eligible_candidates = [
                item for item in ranked_candidates
                if bool((item.get("score", {}) or {}).get("z_positive_candidate_eligible", False))
            ]
            runtime["z_positive_candidate_eligible_count"] = float(len(eligible_candidates))
            if eligible_candidates:
                eligible_candidates.sort(
                    key=lambda item: (
                        float((item.get("score", {}) or {}).get("z_positive_mining_score", float("inf"))),
                        float((item.get("score", {}) or {}).get("z_positive_mining_raw_proxy_z", float("inf"))),
                        int(item.get("rank", 10 ** 9)),
                    )
                )
                mining_score = dict(eligible_candidates[0].get("score", {}) or {})
                runtime["z_positive_mining_score"] = float(mining_score.get("z_positive_mining_score", float("nan")))
                runtime["z_positive_candidate_eligible"] = bool(mining_score.get("z_positive_candidate_eligible", False))
                runtime["z_positive_candidate_operator"] = str(
                    eligible_candidates[0].get("operator", mining_score.get("z_positive_candidate_operator", ""))
                )
                runtime["z_positive_candidate_eligibility_reason"] = str(
                    mining_score.get("z_positive_candidate_eligibility_reason", "")
                )
            elif ranked_candidates:
                fallback_score = dict(ranked_candidates[0].get("score", {}) or {})
                runtime["z_positive_mining_score"] = float(fallback_score.get("z_positive_mining_score", float("nan")))
                runtime["z_positive_candidate_operator"] = str(
                    ranked_candidates[0].get("operator", fallback_score.get("z_positive_candidate_operator", ""))
                )
                runtime["z_positive_candidate_eligibility_reason"] = str(
                    fallback_score.get("z_positive_candidate_eligibility_reason", "")
                )

        chosen = ranked_candidates[0] if ranked_candidates else None
        best_snap = chosen.get("snapshot") if chosen is not None else None
        best_score = dict(chosen.get("score", {}) or {}) if chosen is not None else dict(baseline)
        if (not neural_mode) and chosen is not None:
            classic_verify_triggered, classic_verify_reason = self._classic_soft_verify_signal(
                layer=layer,
                score=best_score,
                anchor_z=anchor_z,
                min_improve=min_improve,
            )
            best_score["_classic_verify"] = bool(classic_verify_triggered)
            best_score["_classic_verify_reason"] = str(classic_verify_reason)
            runtime["classic_verify_triggered"] = bool(classic_verify_triggered)
            runtime["classic_verify_reason"] = str(classic_verify_reason)
        eval_candidates = self._select_xz_global_eval_candidates(layer, ranked_candidates, runtime) if ranked_candidates else []
        if layer == "X":
            best_score["_x_all_candidates"] = [
                {
                    "snapshot": item.get("snapshot"),
                    "score": dict(item.get("score", {}) or {}),
                    "operator": str(item.get("operator", "")),
                    "operator_rank": int(item.get("operator_rank", 0)),
                    "rank": int(item.get("rank", item.get("f0_rank", 0))),
                    "meta": dict(item.get("meta", {}) or {}),
                }
                for item in candidate_pool
            ]
            best_score["_x_eval_candidates"] = [
                {
                    "snapshot": item.get("snapshot"),
                    "score": dict(item.get("score", {}) or {}),
                    "operator": str(item.get("operator", "")),
                    "operator_rank": int(item.get("operator_rank", 0)),
                    "rank": int(item.get("rank", item.get("f0_rank", 0))),
                    "meta": dict(item.get("meta", {}) or {}),
                }
                for item in eval_candidates
            ]
            runtime["x_eval_pair_diversity"] = float(
                len({
                    (
                        str((item.get("score", {}) or {}).get("x_destroy_operator", "")),
                        str((item.get("score", {}) or {}).get("x_repair_operator", "")),
                    )
                    for item in eval_candidates
                })
            )
        else:
            best_score["_z_eval_candidates"] = [
                {
                    "snapshot": item.get("snapshot"),
                    "score": dict(item.get("score", {}) or {}),
                    "operator": str(item.get("operator", "")),
                    "operator_rank": int(item.get("operator_rank", 0)),
                    "rank": int(item.get("rank", item.get("f0_rank", 0))),
                    "meta": dict(item.get("meta", {}) or {}),
                }
                for item in eval_candidates
            ]
            best_score["_z_all_candidates"] = [
                {
                    "snapshot": item.get("snapshot"),
                    "score": dict(item.get("score", {}) or {}),
                    "operator": str(item.get("operator", "")),
                    "operator_rank": int(item.get("operator_rank", 0)),
                    "rank": int(item.get("rank", item.get("f0_rank", 0))),
                    "meta": dict(item.get("meta", {}) or {}),
                }
                    for item in candidate_pool
                ]
            best_score["_z_positive_mining_candidate"] = None
            eligible_candidates = [
                {
                    "snapshot": item.get("snapshot"),
                    "score": dict(item.get("score", {}) or {}),
                    "operator": str(item.get("operator", "")),
                    "operator_rank": int(item.get("operator_rank", 0)),
                    "rank": int(item.get("rank", item.get("f0_rank", 0))),
                    "meta": dict(item.get("meta", {}) or {}),
                }
                for item in ranked_candidates
                if bool((item.get("score", {}) or {}).get("z_positive_candidate_eligible", False))
            ]
            if eligible_candidates:
                eligible_candidates.sort(
                    key=lambda item: (
                        float((item.get("score", {}) or {}).get("z_positive_mining_score", float("inf"))),
                        float((item.get("score", {}) or {}).get("z_positive_mining_raw_proxy_z", float("inf"))),
                        int(item.get("rank", 10 ** 9)),
                    )
                )
                best_score["_z_positive_mining_candidate"] = eligible_candidates[0]
            all_candidates_payload = list(best_score.get("_z_all_candidates", []) or [])
            eval_candidates_payload = list(best_score.get("_z_eval_candidates", []) or [])
            runtime["z_all_candidate_count"] = float(len(all_candidates_payload))
            runtime["z_full_global_eval_experiment"] = bool(self._z_full_global_eval_experiment_enabled())
            if all_candidates_payload:
                hard_bad_margin = float(getattr(self.cfg, "xz_hard_bad_margin", 180.0))
                eval_key_set = {
                    (
                        str(item.get("operator", "")),
                        int(item.get("operator_rank", 0)),
                        int(item.get("rank", 0)),
                    )
                    for item in eval_candidates_payload
                }
                proxy_filtered = 0
                fast_gate_filtered = 0
                topk_filtered = 0
                for item in all_candidates_payload:
                    score = dict(item.get("score", {}) or {})
                    if float(score.get("predicted_proxy_z", float("inf"))) >= float(self.anchor_z) + hard_bad_margin:
                        proxy_filtered += 1
                        continue
                    if not bool(score.get("proposal_pass_fast_gate", False)):
                        fast_gate_filtered += 1
                        continue
                    key = (
                        str(item.get("operator", "")),
                        int(item.get("operator_rank", 0)),
                        int(item.get("rank", 0)),
                    )
                    if key not in eval_key_set:
                        topk_filtered += 1
                runtime["z_filtered_by_proxy_count_legacy"] = float(proxy_filtered)
                runtime["z_filtered_by_fast_gate_count_legacy"] = float(fast_gate_filtered)
                runtime["z_filtered_by_topk_count_legacy"] = float(topk_filtered)
        runtime["selected_candidate_rank"] = float(chosen.get("rank", chosen.get("f0_rank", 0))) if chosen is not None else 0.0
        runtime["selected_operator"] = str(chosen.get("operator", "")) if chosen is not None else ""
        runtime["selected_operator_rank"] = float(chosen.get("operator_rank", 0)) if chosen is not None else 0.0
        runtime["fidelity_selected"] = str(best_score.get("fidelity_selected", "F1"))
        runtime["predicted_proxy_z"] = float(best_score.get("predicted_proxy_z", best_score.get("augmented_obj", float("nan"))))
        runtime["predicted_proxy_delta"] = float(best_score.get("predicted_proxy_delta", float("nan")))
        runtime["win_prob"] = float(best_score.get("win_prob", float("nan")))
        runtime["prediction_uncertainty"] = float(best_score.get("prediction_uncertainty", float("nan")))
        runtime["residual_hat"] = float(best_score.get("residual_hat", float("nan")))
        runtime["residual_std"] = float(best_score.get("residual_std", float("nan")))
        runtime["proposal_pass_fast_gate"] = bool(best_score.get("proposal_pass_fast_gate", False))
        runtime["fast_gate_reason"] = str(best_score.get("fast_gate_reason", ""))
        runtime["surrogate_buffer_size"] = float(len(state.training_buffer)) if neural_mode and state is not None else 0.0
        if layer == "X":
            runtime["x_f0_rank_score"] = float(best_score.get("x_f0_rank_score", float("nan")))
            runtime["x_f1_proxy_z"] = float(best_score.get("x_f1_proxy_z", float("nan")))
            runtime["x_f1_station_cmax"] = float(best_score.get("x_f1_station_cmax", float("nan")))
            runtime["x_f1_route_tail"] = float(best_score.get("x_f1_route_tail", float("nan")))
            runtime["x_f1_mapping_coverage"] = float(best_score.get("x_f1_mapping_coverage", float("nan")))
            runtime["x_f1_used_sp2"] = bool(best_score.get("x_f1_used_sp2", False))
            runtime["x_f1_used_sp3"] = bool(best_score.get("x_f1_used_sp3", False))
            runtime["x_f1_replayed_route"] = bool(best_score.get("x_f1_replayed_route", False))
            runtime["x_f1_pre_y_proxy_z"] = float(best_score.get("x_f1_pre_y_proxy_z", float("nan")))
            runtime["x_f1_post_y_proxy_z"] = float(best_score.get("x_f1_post_y_proxy_z", float("nan")))
            runtime["x_f1_rebalance_effective"] = bool(best_score.get("x_f1_rebalance_effective", False))
            runtime["x_station_overload_penalty"] = float(best_score.get("x_station_overload_penalty", float("nan")))
            runtime["x_classic_cx"] = float(best_score.get("x_classic_cx", float("nan")))
            runtime["x_classic_p_affinity"] = float(best_score.get("x_classic_p_affinity", float("nan")))
            runtime["x_classic_p_route"] = float(best_score.get("x_classic_p_route", float("nan")))
            runtime["x_classic_p_time"] = float(best_score.get("x_classic_p_time", float("nan")))
            runtime["x_classic_dx"] = float(best_score.get("x_classic_dx", float("nan")))
            runtime["x_micro_move_size"] = float(best_score.get("x_micro_move_size", 0.0))
            runtime["x_anchor_template_preservation_ratio"] = float(best_score.get("anchor_template_preservation_ratio", float("nan")))
            runtime["x_changed_assignment_pair_count"] = float(best_score.get("x_changed_assignment_pair_count", 0.0))
            runtime["x_move_type"] = str(best_score.get("x_move_type", best_score.get("x_destroy_operator", "")))
            runtime["x_spatial_dispersion_score"] = float(best_score.get("x_spatial_dispersion_score", float("nan")))
            runtime["x_low_consolidation_score"] = float(best_score.get("x_low_consolidation_score", float("nan")))
            runtime["x_y_hotspot_alignment"] = float(best_score.get("x_y_hotspot_alignment", float("nan")))
            runtime["x_moved_sku_count"] = float(best_score.get("x_moved_sku_count", 0.0))
            runtime["x_swap_pair_count"] = float(best_score.get("x_swap_pair_count", 0.0))
            runtime["x_destroy_size_effective"] = float(best_score.get("x_destroy_size_effective", best_score.get("x_destroy_size", 0.0)))
            runtime["x_merge_split_fallback_used"] = bool(best_score.get("x_merge_split_fallback_used", False))
            runtime["x_changed_orders"] = float(best_score.get("changed_orders", 0.0))
            runtime["x_station_template_change_count"] = float(best_score.get("station_template_change_count", 0.0))
            runtime["x_robot_trip_template_change_count"] = float(best_score.get("robot_trip_template_change_count", 0.0))
            runtime["x_candidate_reject_stage"] = str(best_score.get("x_candidate_reject_stage", ""))
            runtime["x_candidate_reject_reason"] = str(best_score.get("x_candidate_reject_reason", ""))
            runtime["x_equivalent_candidate_deduped"] = bool(best_score.get("x_equivalent_candidate_deduped", False))
        else:
            runtime["z_move_type"] = str(best_score.get("z_move_type", ""))
            runtime["z_move_variant"] = str(best_score.get("z_move_variant", best_score.get("z_move_type", "")))
            runtime["z_changed_subtask_count"] = float(best_score.get("z_changed_subtask_count", 0.0))
            runtime["z_candidate_task_delta"] = float(best_score.get("z_candidate_task_delta", 0.0))
            runtime["z_candidate_stack_delta"] = float(best_score.get("z_candidate_stack_delta", 0.0))
            runtime["z_candidate_mode_delta"] = float(best_score.get("z_candidate_mode_delta", 0.0))
            runtime["z_f0_rank_score"] = float(best_score.get("z_f0_rank_score", float("nan")))
            runtime["z_f1_proxy_z"] = float(best_score.get("z_f1_proxy_z", float("nan")))
            runtime["z_f1_arrival_shift_total"] = float(best_score.get("z_f1_arrival_shift_total", float("nan")))
            runtime["z_f1_wait_overflow_total"] = float(best_score.get("z_f1_wait_overflow_total", float("nan")))
            runtime["z_f1_route_tail_delta"] = float(best_score.get("z_f1_route_tail_delta", float("nan")))
            runtime["z_f1_replayed_trip_count"] = float(best_score.get("z_f1_replayed_trip_count", 0.0))
            runtime["z_f1_used_full_replay"] = bool(best_score.get("z_f1_used_full_replay", False))
            runtime["z_f1_pre_y_proxy_z"] = float(best_score.get("z_f1_pre_y_proxy_z", float("nan")))
            runtime["z_f1_post_y_proxy_z"] = float(best_score.get("z_f1_post_y_proxy_z", float("nan")))
            runtime["z_route_gap_penalty"] = float(best_score.get("z_route_gap_penalty", float("nan")))
            runtime["z_route_gap_penalty_normalized"] = float(best_score.get("z_route_gap_penalty_normalized", float("nan")))
            runtime["z_route_gap_signal_raw"] = float(best_score.get("z_route_gap_signal_raw", float("nan")))
            runtime["z_route_gap_signal_normalized"] = float(best_score.get("z_route_gap_signal_normalized", float("nan")))
            runtime["z_classic_cz"] = float(best_score.get("z_classic_cz", float("nan")))
            runtime["z_classic_p_zx"] = float(best_score.get("z_classic_p_zx", float("nan")))
            runtime["z_classic_p_zy"] = float(best_score.get("z_classic_p_zy", float("nan")))
            runtime["z_classic_p_zu"] = float(best_score.get("z_classic_p_zu", float("nan")))
            runtime["z_classic_dz"] = float(best_score.get("z_classic_dz", float("nan")))
            runtime["z_arrival_soft_penalty"] = float(best_score.get("z_arrival_soft_penalty", float("nan")))
            runtime["z_arrival_soft_penalty_normalized"] = float(best_score.get("z_arrival_soft_penalty_normalized", float("nan")))
            runtime["z_arrival_shift_signal_raw"] = float(best_score.get("z_arrival_shift_signal_raw", float("nan")))
            runtime["z_arrival_shift_signal_normalized"] = float(best_score.get("z_arrival_shift_signal_normalized", float("nan")))
            runtime["z_wait_soft_penalty"] = float(best_score.get("z_wait_soft_penalty", float("nan")))
            runtime["z_wait_soft_penalty_normalized"] = float(best_score.get("z_wait_soft_penalty_normalized", float("nan")))
            runtime["z_wait_overflow_signal_raw"] = float(best_score.get("z_wait_overflow_signal_raw", float("nan")))
            runtime["z_wait_overflow_signal_normalized"] = float(best_score.get("z_wait_overflow_signal_normalized", float("nan")))
            runtime["z_route_tail_soft_penalty"] = float(best_score.get("z_route_tail_soft_penalty", float("nan")))
            runtime["z_route_tail_soft_penalty_normalized"] = float(best_score.get("z_route_tail_soft_penalty_normalized", float("nan")))
            runtime["z_route_tail_signal_raw"] = float(best_score.get("z_route_tail_signal_raw", float("nan")))
            runtime["z_route_tail_signal_normalized"] = float(best_score.get("z_route_tail_signal_normalized", float("nan")))
            runtime["z_soft_penalty_total"] = float(best_score.get("z_soft_penalty_total", float("nan")))
            runtime["z_soft_penalty_total_normalized"] = float(best_score.get("z_soft_penalty_total_normalized", float("nan")))
            runtime["z_noise_tote_delta"] = float(best_score.get("noise_tote_delta", float("nan")))
            runtime["z_hit_tote_preservation_ratio"] = float(best_score.get("hit_tote_preservation_ratio", float("nan")))
            runtime["z_stack_locality_score"] = float(best_score.get("stack_locality_score", float("nan")))
            runtime["z_route_insertion_detour"] = float(best_score.get("z_route_insertion_detour", float("nan")))
            runtime["z_hit_frequency_bonus"] = float(best_score.get("z_hit_frequency_bonus", float("nan")))
            runtime["z_demand_ratio"] = float(best_score.get("z_demand_ratio", float("nan")))
            runtime["z_congestion_proxy"] = float(best_score.get("z_congestion_proxy", float("nan")))
            runtime["z_hotspot_score"] = float(best_score.get("z_hotspot_score", 0.0))
            runtime["z_candidate_reject_stage"] = str(best_score.get("z_candidate_reject_stage", ""))
            runtime["z_candidate_reject_reason"] = str(best_score.get("z_candidate_reject_reason", ""))
            runtime["z_arrival_shift_estimate"] = float(best_score.get("z_arrival_shift_estimate", float("nan")))
            runtime["z_wait_overflow_estimate"] = float(best_score.get("z_wait_overflow_estimate", float("nan")))
            runtime["z_route_tail_delta_estimate"] = float(best_score.get("z_route_tail_delta_estimate", float("nan")))
            runtime["z_operator_banned"] = bool(best_score.get("z_operator_banned", False))
            runtime["z_operator_fallback_used"] = bool(best_score.get("z_operator_fallback_used", False))
            runtime["z_fallback_type"] = str(best_score.get("z_fallback_type", ""))
            runtime["z_generation_guard_reason"] = str(runtime.get("z_generation_guard_reason", best_score.get("z_generation_guard_reason", "")))
            runtime["z_candidate_signature_blocked"] = bool(runtime.get("z_candidate_signature_blocked", best_score.get("z_candidate_signature_blocked", False)))
            runtime["z_positive_mining_score"] = float(runtime.get("z_positive_mining_score", best_score.get("z_positive_mining_score", float("nan"))))
            runtime["z_positive_candidate_eligible"] = bool(runtime.get("z_positive_candidate_eligible", best_score.get("z_positive_candidate_eligible", False)))
            runtime["z_positive_candidate_operator"] = str(runtime.get("z_positive_candidate_operator", best_score.get("z_positive_candidate_operator", "")))
            runtime["z_positive_candidate_eligibility_reason"] = str(runtime.get("z_positive_candidate_eligibility_reason", best_score.get("z_positive_candidate_eligibility_reason", "")))
            runtime["z_positive_mining_raw_proxy_z"] = float(best_score.get("z_positive_mining_raw_proxy_z", float("nan")))
            runtime["z_positive_mining_risk_penalty"] = float(best_score.get("z_positive_mining_risk_penalty", float("nan")))
            runtime["z_positive_mining_reward_bonus"] = float(best_score.get("z_positive_mining_reward_bonus", float("nan")))
            runtime["z_positive_mining_structure_penalty"] = float(best_score.get("z_positive_mining_structure_penalty", float("nan")))
            runtime["z_positive_mining_route_gap_ratio"] = float(best_score.get("z_positive_mining_route_gap_ratio", float("nan")))
            runtime["z_positive_mining_arrival_ratio"] = float(best_score.get("z_positive_mining_arrival_ratio", float("nan")))
            runtime["z_positive_mining_wait_ratio"] = float(best_score.get("z_positive_mining_wait_ratio", float("nan")))
            runtime["z_positive_mining_tail_ratio"] = float(best_score.get("z_positive_mining_tail_ratio", float("nan")))
        return best_snap, best_score

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
        x_fast_eval = self._fast_x_rollout_eval()
        station_term = float(getattr(self.cfg, "x_surrogate_station_weight", 6.0)) * float(x_fast_eval.delta_station_load_drift)
        arrival_term = float(getattr(self.cfg, "x_surrogate_arrival_weight", 10.0)) * float(x_fast_eval.delta_arrival_shift)
        route_term = float(getattr(self.cfg, "x_surrogate_route_weight", 4.0)) * float(x_fast_eval.delta_route_pressure)
        surrogate_core_score = float(station_term + arrival_term + route_term)
        affinity_term = float(getattr(self.cfg, "x_surrogate_affinity_weight", 0.75)) * float(low_affinity_penalty)
        finish_term = float(getattr(self.cfg, "x_surrogate_finish_weight", 1.5)) * float(finish_time_dispersion_penalty)
        subtask_term = float(getattr(self.cfg, "x_surrogate_subtask_weight", 0.35)) * float(x_fast_eval.delta_subtask_count)
        route_conflict_term = (
            float(self.layer_lambda_weights.get("x_route", float(getattr(self.cfg, "lambda_x_route", 0.5))))
            * float(route_conflict_penalty)
        )
        structure_regularizer = float(affinity_term + finish_term + subtask_term + route_conflict_term)
        prox_penalty = float(current.touched_subtask_count) * float(getattr(self.cfg, "x_prox_weight", 0.25))
        local_obj = float(surrogate_core_score)
        coupling_penalty = float(structure_regularizer)
        augmented_obj = float(local_obj + coupling_penalty + float(self._trust_region_tau("X")) * prox_penalty)
        return {
            "local_obj": float(local_obj),
            "coupling_penalty": float(coupling_penalty),
            "prox_penalty": float(prox_penalty),
            "augmented_obj": float(augmented_obj),
            "x_surrogate_core_score": float(surrogate_core_score),
            "x_surrogate_station_term": float(station_term),
            "x_surrogate_arrival_term": float(arrival_term),
            "x_surrogate_route_term": float(route_term),
            "x_surrogate_affinity_term": float(affinity_term),
            "x_surrogate_finish_term": float(finish_term),
            "x_surrogate_subtask_term": float(subtask_term),
            "x_affinity_penalty": float(low_affinity_penalty),
            "x_route_conflict_penalty": float(route_conflict_penalty),
            "x_finish_time_dispersion_penalty": float(finish_time_dispersion_penalty),
            "x_subtask_count": float(current.subtask_count),
            "x_touched_subtask_count": float(current.touched_subtask_count),
            "x_fast_penalty": float(x_fast_eval.objective_value),
            "x_fast_delta_station_load_drift": float(x_fast_eval.delta_station_load_drift),
            "x_fast_delta_arrival_shift": float(x_fast_eval.delta_arrival_shift),
            "x_fast_delta_route_pressure": float(x_fast_eval.delta_route_pressure),
            "x_fast_delta_subtask_count": float(x_fast_eval.delta_subtask_count),
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
        self._clear_z_detour_cache()
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
            "anchor_subtask_profile": {},
            "anchor_task_profile": {},
            "anchor_sku_profile": {},
            "robot_stack_sequences": {},
            "robot_insertion_windows": {},
            "stack_visit_frequency": {},
            "store_point_visit_frequency": {},
            "congestion_visit_threshold": 0.0,
            "robot_hotspot_centroids": {},
            "path_hotspot_centroids": [],
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
            st_id = int(getattr(st, "id", -1))
            sku_ids = sorted({
                int(getattr(sku, "id", -1))
                for sku in getattr(st, "unique_sku_list", []) or []
                if int(getattr(sku, "id", -1)) >= 0
            })
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
            route_tail = 0.0
            robot_counter: Dict[int, int] = defaultdict(int)
            trip_set: Set[Tuple[int, int]] = set()
            stack_ids_for_subtask: Set[int] = set()
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
                    stack_ids_for_subtask.add(stack_id)
                    stack_route_cost[stack_id].append(
                        float(getattr(task, "arrival_time_at_stack", 0.0)) + float(getattr(task, "robot_service_time", 0.0))
                    )
                rid = int(getattr(task, "robot_id", -1))
                if rid >= 0:
                    robot_counter[rid] += 1
                    trip_set.add((rid, int(getattr(task, "trip_id", 0))))
                route_tail = max(
                    route_tail,
                    float(getattr(task, "arrival_time_at_stack", 0.0)),
                    float(getattr(task, "arrival_time_at_station", 0.0)),
                )
                route_val += float(getattr(task, "robot_service_time", 0.0))
                route_val += float(getattr(task, "arrival_time_at_stack", 0.0))
                ref["anchor_task_profile"][int(getattr(task, "task_id", -1))] = {
                    "task_signature": (
                        int(getattr(task, "sub_task_id", -1)),
                        int(getattr(task, "target_stack_id", -1)),
                        str(getattr(task, "operation_mode", "")),
                    ),
                    "subtask_id": st_id,
                    "order_id": oid,
                    "robot_id": rid,
                    "trip_id": int(getattr(task, "trip_id", 0)),
                    "robot_visit_sequence": int(getattr(task, "robot_visit_sequence", 0)),
                    "stack_id": int(getattr(task, "target_stack_id", -1)),
                    "station_id": int(getattr(task, "target_station_id", -1)),
                    "arrival_stack": float(getattr(task, "arrival_time_at_stack", 0.0)),
                    "arrival_station": float(getattr(task, "arrival_time_at_station", 0.0)),
                    "robot_service_time": float(getattr(task, "robot_service_time", 0.0)),
                    "station_service_time": float(getattr(task, "station_service_time", 0.0)),
                    "operation_mode": str(getattr(task, "operation_mode", "")),
                    "target_tote_ids": list(int(x) for x in (getattr(task, "target_tote_ids", []) or [])),
                    "hit_tote_ids": list(int(x) for x in (getattr(task, "hit_tote_ids", []) or [])),
                    "noise_tote_ids": list(int(x) for x in (getattr(task, "noise_tote_ids", []) or [])),
                    "sort_layer_range": copy.deepcopy(getattr(task, "sort_layer_range", None)),
                }

            if proc_val <= 1e-9:
                proc_val = len(getattr(st, "sku_list", []) or []) * float(OFSConfig.PICKING_TIME)
            slack_val = max(0.0, start_val - arr_val)
            dominant_robot_id = min(robot_counter.items(), key=lambda item: (-item[1], item[0]))[0] if robot_counter else -1
            completion_val = float(start_val + proc_val)
            ref["subtask_proc"][st_id] = float(proc_val)
            ref["subtask_arrival"][st_id] = float(arr_val)
            ref["subtask_start"][st_id] = float(start_val)
            ref["subtask_slack"][st_id] = float(slack_val)
            ref["anchor_subtask_profile"][st_id] = {
                "order_id": oid,
                "station_id": sid,
                "rank": int(getattr(st, "station_sequence_rank", -1)),
                "proc": float(proc_val),
                "arrival": float(arr_val),
                "start": float(start_val),
                "completion": float(completion_val),
                "stack_ids": sorted(int(x) for x in stack_ids_for_subtask if int(x) >= 0),
                "dominant_robot_id": int(dominant_robot_id),
                "trip_ids": sorted((int(r), int(t)) for r, t in trip_set),
                "route_service_sum": float(route_val),
                "route_tail": float(route_tail),
                "sku_ids": list(sku_ids),
            }
            if oid >= 0:
                for sku_id in sku_ids:
                    ref["anchor_sku_profile"][(int(oid), int(sku_id))] = {
                        "anchor_subtask_id": st_id,
                        "station_id": sid,
                        "rank": int(getattr(st, "station_sequence_rank", -1)),
                        "robot_id": int(dominant_robot_id),
                        "route_pos": float(getattr(st, "station_sequence_rank", -1)),
                        "completion": float(completion_val),
                        "stack_ids": sorted(int(x) for x in stack_ids_for_subtask if int(x) >= 0),
                        "proc_share": float(proc_val / max(1, len(sku_ids))),
                    }

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
        robot_stack_sequences: Dict[int, List[int]] = {}
        robot_insertion_windows: Dict[int, List[Tuple[Optional[int], Optional[int]]]] = {}
        stack_visit_frequency: Dict[int, int] = defaultdict(int)
        store_point_visit_frequency: Dict[int, int] = defaultdict(int)
        robot_hotspot_centroids: Dict[int, Tuple[float, float]] = {}
        robot_task_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for profile in (ref["anchor_task_profile"] or {}).values():
            rid = int(profile.get("robot_id", -1))
            sid = int(profile.get("stack_id", -1))
            if rid < 0 or sid < 0:
                continue
            robot_task_rows[rid].append(dict(profile))
            stack_visit_frequency[sid] += 1
            stack = self.problem.point_to_stack.get(sid) if self.problem is not None else None
            point_idx = int(getattr(getattr(stack, "store_point", None), "idx", sid)) if stack is not None else sid
            store_point_visit_frequency[point_idx] += 1
        for rid, rows in robot_task_rows.items():
            rows.sort(key=lambda item: (int(item.get("robot_visit_sequence", 0)), int(item.get("task_signature", (0, 0, ""))[0]), int(item.get("stack_id", -1))))
            seq: List[int] = []
            pts: List[Tuple[float, float]] = []
            for row in rows:
                sid = int(row.get("stack_id", -1))
                if sid < 0:
                    continue
                if not seq or seq[-1] != sid:
                    seq.append(sid)
                xy = self._stack_xy(sid)
                if xy is not None:
                    pts.append(xy)
            robot_stack_sequences[int(rid)] = list(seq)
            windows: List[Tuple[Optional[int], Optional[int]]] = []
            if seq:
                windows.append((None, int(seq[0])))
                for idx in range(len(seq) - 1):
                    windows.append((int(seq[idx]), int(seq[idx + 1])))
                windows.append((int(seq[-1]), None))
            robot_insertion_windows[int(rid)] = windows
            if pts:
                robot_hotspot_centroids[int(rid)] = (
                    float(sum(pt[0] for pt in pts) / len(pts)),
                    float(sum(pt[1] for pt in pts) / len(pts)),
                )
        freq_rows = sorted(int(val) for val in stack_visit_frequency.values())
        quantile = float(getattr(self.cfg, "z_congestion_top_quantile", 0.75))
        if freq_rows:
            idx = max(0, min(len(freq_rows) - 1, int(math.ceil(quantile * len(freq_rows))) - 1))
            ref["congestion_visit_threshold"] = float(freq_rows[idx])
        ref["robot_stack_sequences"] = {int(rid): list(seq) for rid, seq in robot_stack_sequences.items()}
        ref["robot_insertion_windows"] = {
            int(rid): [(None if prev is None else int(prev), None if nxt is None else int(nxt)) for prev, nxt in windows]
            for rid, windows in robot_insertion_windows.items()
        }
        ref["stack_visit_frequency"] = {int(sid): int(cnt) for sid, cnt in stack_visit_frequency.items()}
        ref["store_point_visit_frequency"] = {int(pid): int(cnt) for pid, cnt in store_point_visit_frequency.items()}
        ref["robot_hotspot_centroids"] = {
            int(rid): (float(center[0]), float(center[1])) for rid, center in robot_hotspot_centroids.items()
        }
        ref["path_hotspot_centroids"] = [
            (float(center[0]), float(center[1])) for _, center in sorted(robot_hotspot_centroids.items(), key=lambda item: item[0])
        ]
        ref["robot_path_length"] = float(self._compute_robot_path_length_from_tasks(anchor_tasks))
        local_proxy_baseline = float(self._compute_current_z_local_proxy_baseline())
        ref["x_local_proxy_baseline"] = float(local_proxy_baseline)
        ref["z_local_proxy_baseline"] = float(local_proxy_baseline)
        self.anchor_version = int(getattr(self, "anchor_version", -1)) + 1
        ref["anchor_version"] = int(self.anchor_version)
        for state in [self.x_surrogate_state, self.z_surrogate_state]:
            if state is None:
                continue
            state.anchor_version = int(self.anchor_version)
            state.f1_cache.clear()
        self.z_operator_subtask_failures = {}
        self.z_operator_subtask_bans = set()
        self.z_operator_subtask_failure_iter = {}
        self.z_signature_reject_cache = set()
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
            return {
                "layer": "X",
                "local_obj": float(x_score.get("local_obj", float("nan"))),
                "coupling_penalty": float(x_score.get("coupling_penalty", 0.0)),
                "prox_penalty": float(x_score.get("prox_penalty", 0.0)),
                "augmented_obj": float(x_score.get("augmented_obj", float("nan"))),
                "couplings": {str(k): float(v) for k, v in (x_score.get("couplings", {}) or {}).items()},
                "x_destroy_operator": "",
                "x_repair_operator": "",
                "x_destroy_size": float("nan"),
                "x_subtask_count_before": float(len(self._iter_snapshot_subtasks(self.anchor)) if self.anchor is not None else 0),
                "x_subtask_count_after": float(x_score.get("x_subtask_count", float("nan"))),
                "x_affinity_penalty": float(x_score.get("x_affinity_penalty", float("nan"))),
                "x_route_conflict_penalty": float(x_score.get("x_route_conflict_penalty", float("nan"))),
                "x_finish_time_dispersion_penalty": float(x_score.get("x_finish_time_dispersion_penalty", float("nan"))),
                "x_surrogate_core_score": float(x_score.get("x_surrogate_core_score", float("nan"))),
                "x_surrogate_station_term": float(x_score.get("x_surrogate_station_term", float("nan"))),
                "x_surrogate_arrival_term": float(x_score.get("x_surrogate_arrival_term", float("nan"))),
                "x_surrogate_route_term": float(x_score.get("x_surrogate_route_term", float("nan"))),
                "x_surrogate_affinity_term": float(x_score.get("x_surrogate_affinity_term", float("nan"))),
                "x_surrogate_finish_term": float(x_score.get("x_surrogate_finish_term", float("nan"))),
                "x_surrogate_subtask_term": float(x_score.get("x_surrogate_subtask_term", float("nan"))),
                "x_fast_penalty": float(x_score.get("x_fast_penalty", float("nan"))),
                "x_fast_delta_station_load_drift": float(x_score.get("x_fast_delta_station_load_drift", float("nan"))),
                "x_fast_delta_arrival_shift": float(x_score.get("x_fast_delta_arrival_shift", float("nan"))),
                "x_fast_delta_route_pressure": float(x_score.get("x_fast_delta_route_pressure", float("nan"))),
                "x_fast_delta_subtask_count": float(x_score.get("x_fast_delta_subtask_count", float("nan"))),
            }

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
        # 当前先复用现有 route replay，保持 task/robot/trip 不变，
        # 只按新的 station/rank 回放到站时间。
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

    def _evaluate_z_candidate_structurally(self, used_joint_repair: bool = False) -> ZStructuralEvalResult:
        self._sync_sp3_caches_from_problem()
        coverage = self._compute_solution_coverage()
        sorting_cost_proxy = float(sum((self.last_sp3_sorting_costs or {}).values()))
        coverage_gap = float(coverage.get("unmet_sku_total", 0))

        route_cost_rows = list((self.anchor_reference.get("stack_route_cost", {}) or {}).values())
        route_proxy_default = float(sum(route_cost_rows) / len(route_cost_rows)) if route_cost_rows else 0.0
        anchor_route_pressure_sum = max(1.0, float(sum((self.anchor_reference.get("order_route_pressure", {}) or {}).values())))
        anchor_makespan = max(1.0, float(self.anchor_z if math.isfinite(self.anchor_z) else 1.0))

        noise_total = 0.0
        target_total = 0.0
        multi_stack_penalty = 0.0
        route_upper_bound = 0.0
        baseline_route_pressure = 0.0
        processing_overflow_raw = 0.0
        proc_by_subtask: Dict[int, float] = {}

        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            subtask_stack_ids: Set[int] = set()
            for task in getattr(st, "execution_tasks", []) or []:
                noise_total += float(len(getattr(task, "noise_tote_ids", []) or []))
                target_total += float(len(getattr(task, "target_tote_ids", []) or []))
                stack_id = int(getattr(task, "target_stack_id", -1))
                if stack_id >= 0:
                    subtask_stack_ids.add(stack_id)
            multi_stack_penalty += max(0.0, float(len(subtask_stack_ids)) - 1.0)
            route_upper_bound += sum(
                float((self.anchor_reference.get("stack_route_cost", {}) or {}).get(stack_id, route_proxy_default))
                for stack_id in subtask_stack_ids
            )
            baseline_route_pressure += float((self.anchor_reference.get("order_route_pressure", {}) or {}).get(oid, 0.0))
            proc_curr = float(self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs))
            proc_by_subtask[sid] = proc_curr
            processing_overflow_raw += max(0.0, proc_curr - float(self._estimate_subtask_slack(st)))

        station_metrics = self._recompute_station_schedule(proc_by_subtask=proc_by_subtask)
        noise_ratio = float(noise_total / target_total) if target_total > 0.0 else 0.0
        route_pressure_proxy = max(0.0, float(route_upper_bound - baseline_route_pressure)) / anchor_route_pressure_sum
        station_load_std = float(station_metrics.get("station_load_std", 0.0))
        processing_overflow = float(processing_overflow_raw) / anchor_makespan
        objective_value = (
            sorting_cost_proxy
            + 25.0 * noise_ratio
            + coverage_gap
            + multi_stack_penalty
            + float(getattr(self.cfg, "z_route_pressure_weight", 1.0)) * route_pressure_proxy
            + float(getattr(self.cfg, "z_station_load_weight", 0.75)) * station_load_std
            + float(getattr(self.cfg, "z_processing_overflow_weight", 1.0)) * processing_overflow
        )
        return ZStructuralEvalResult(
            objective_value=float(objective_value),
            sorting_cost_proxy=float(sorting_cost_proxy),
            coverage_gap=float(coverage_gap),
            multi_stack_penalty=float(multi_stack_penalty),
            noise_ratio=float(noise_ratio),
            route_pressure_proxy=float(route_pressure_proxy),
            station_load_std=float(station_load_std),
            processing_overflow=float(processing_overflow),
            used_joint_repair=bool(used_joint_repair),
        )

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

    def _z_stack_visit_frequency(self, stack_id: int) -> int:
        stack_freq = int((self.anchor_reference.get("stack_visit_frequency", {}) or {}).get(int(stack_id), 0))
        stack = self.problem.point_to_stack.get(int(stack_id)) if self.problem is not None else None
        point_idx = int(getattr(getattr(stack, "store_point", None), "idx", -1)) if stack is not None else -1
        point_freq = int((self.anchor_reference.get("store_point_visit_frequency", {}) or {}).get(int(point_idx), 0))
        return max(stack_freq, point_freq)

    def _z_congestion_proxy(self, stack_id: int) -> float:
        freq_rows = list((self.anchor_reference.get("stack_visit_frequency", {}) or {}).values()) + list((self.anchor_reference.get("store_point_visit_frequency", {}) or {}).values())
        max_freq = max([float(val) for val in freq_rows] + [1.0])
        return float(self._z_stack_visit_frequency(int(stack_id)) / max_freq)

    def _z_is_congested_stack(self, stack_id: int) -> bool:
        threshold = float(self.anchor_reference.get("congestion_visit_threshold", 0.0))
        return bool(threshold > 0.0 and float(self._z_stack_visit_frequency(int(stack_id))) >= threshold - 1e-9)

    def _z_best_insertion_detour(self, stack_id: int) -> float:
        stack_id = int(stack_id)
        if stack_id in self._z_detour_cache:
            return float(self._z_detour_cache[stack_id])
        scale = self._warehouse_distance_scale()
        windows_map = self.anchor_reference.get("robot_insertion_windows", {}) or {}
        if not windows_map:
            self._z_detour_cache[stack_id] = 0.0
            return 0.0
        stack_xy = self._stack_xy(int(stack_id))
        if stack_xy is None:
            self._z_detour_cache[stack_id] = 0.0
            return 0.0
        best_raw: Optional[float] = None
        for windows in windows_map.values():
            for prev_sid, next_sid in windows:
                prev_xy = self._stack_xy(int(prev_sid)) if prev_sid is not None else None
                next_xy = self._stack_xy(int(next_sid)) if next_sid is not None else None
                if prev_xy is None and next_xy is None:
                    detour = 0.0
                elif prev_xy is None:
                    detour = self._xy_manhattan(stack_xy, next_xy)
                elif next_xy is None:
                    detour = self._xy_manhattan(prev_xy, stack_xy)
                else:
                    detour = self._xy_manhattan(prev_xy, stack_xy) + self._xy_manhattan(stack_xy, next_xy) - self._xy_manhattan(prev_xy, next_xy)
                if best_raw is None or float(detour) < float(best_raw):
                    best_raw = float(detour)
        value = float((best_raw or 0.0) / max(1.0, scale))
        self._z_detour_cache[stack_id] = float(value)
        return float(value)

    def _z_subtask_demand_counts(self, st: Any) -> Dict[int, int]:
        req: Dict[int, int] = defaultdict(int)
        for sku in getattr(st, "sku_list", []) or []:
            sid = int(getattr(sku, "id", -1))
            if sid >= 0:
                req[sid] += 1
        return dict(req)

    def _z_used_tote_ids(self, st: Any, exclude_task_ids: Optional[Set[int]] = None) -> Set[int]:
        excluded = {int(x) for x in (exclude_task_ids or set())}
        used: Set[int] = set()
        for task in getattr(st, "execution_tasks", []) or []:
            if int(getattr(task, "task_id", -1)) in excluded:
                continue
            used.update(int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0)
        return used

    def _z_remaining_demand(self, st: Any, exclude_task_ids: Optional[Set[int]] = None) -> Dict[int, int]:
        req = defaultdict(int, self._z_subtask_demand_counts(st))
        tote_map = getattr(self.problem, "id_to_tote", {}) if self.problem is not None else {}
        excluded = {int(x) for x in (exclude_task_ids or set())}
        for task in getattr(st, "execution_tasks", []) or []:
            if int(getattr(task, "task_id", -1)) in excluded:
                continue
            for tote_id in getattr(task, "hit_tote_ids", []) or []:
                tote = tote_map.get(int(tote_id))
                if tote is None:
                    continue
                for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
                    sid = int(sku_id)
                    if sid in req and req[sid] > 0:
                        req[sid] = max(0, int(req[sid]) - int(qty))
        return {int(sid): int(qty) for sid, qty in req.items()}

    def _z_available_stack_totes(self, st: Any, stack_id: int, exclude_task_ids: Optional[Set[int]] = None) -> List[Any]:
        stack = self.problem.point_to_stack.get(int(stack_id)) if self.problem is not None else None
        if stack is None or not getattr(stack, "totes", None):
            return []
        used_other = self._z_used_tote_ids(st, exclude_task_ids)
        return [tote for tote in getattr(stack, "totes", []) or [] if int(getattr(tote, "id", -1)) not in used_other]

    def _z_stack_summary(self, st: Any, stack_id: int, exclude_task_ids: Optional[Set[int]] = None) -> Dict[str, Any]:
        remaining = self._z_remaining_demand(st, exclude_task_ids)
        available_totes = self._z_available_stack_totes(st, int(stack_id), exclude_task_ids)
        demanded_qty = 0
        available_qty = 0
        remaining_local = dict(remaining)
        hit_tote_ids: List[int] = []
        hit_sku_ids: Set[int] = set()
        for sku_id, qty in remaining.items():
            if int(qty) <= 0:
                continue
            stack_qty = sum(int(getattr(tote, "sku_quantity_map", {}).get(int(sku_id), 0)) for tote in available_totes)
            if stack_qty > 0:
                demanded_qty += int(qty)
                available_qty += int(stack_qty)
        for tote in available_totes:
            tote_hit = False
            for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
                sid = int(sku_id)
                use = min(int(remaining_local.get(sid, 0)), int(qty))
                if use > 0:
                    remaining_local[sid] = int(remaining_local.get(sid, 0)) - int(use)
                    hit_sku_ids.add(sid)
                    tote_hit = True
            if tote_hit:
                hit_tote_ids.append(int(getattr(tote, "id", -1)))
        demand_ratio = float(demanded_qty / max(1, available_qty)) if demanded_qty > 0 else 0.0
        return {
            "available_totes": list(available_totes),
            "remaining_demand": dict(remaining),
            "hit_tote_ids": list(hit_tote_ids),
            "hit_sku_ids": set(hit_sku_ids),
            "demanded_qty": int(demanded_qty),
            "available_qty": int(available_qty),
            "demand_ratio": float(demand_ratio),
        }

    def _z_build_plan_from_hits(
        self,
        st: Any,
        task: Any,
        stack_id: int,
        hit_tote_ids: List[int],
        mode: str,
        exclude_task_ids: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        stack = self.problem.point_to_stack.get(int(stack_id)) if self.problem is not None else None
        if stack is None or not getattr(stack, "totes", None):
            return {"valid": False}
        hit_set = {int(tid) for tid in hit_tote_ids if int(tid) >= 0}
        ordered_hits = [int(getattr(tote, "id", -1)) for tote in getattr(stack, "totes", []) or [] if int(getattr(tote, "id", -1)) in hit_set]
        if str(mode).upper() == "FLIP":
            if not ordered_hits:
                return {"valid": False}
            target_tote_ids = list(ordered_hits)
            noise_tote_ids: List[int] = []
            sort_layer_range = None
        else:
            if not ordered_hits:
                return {"valid": False}
            layers = [int(stack.get_tote_layer(int(tid))) for tid in ordered_hits]
            layers = [layer for layer in layers if layer >= 0]
            if not layers:
                return {"valid": False}
            lo = min(layers)
            hi = max(layers)
            target_tote_ids = [int(getattr(tote, "id", -1)) for tote in (getattr(stack, "totes", []) or [])[lo:hi + 1]]
            if any(int(tid) in self._z_used_tote_ids(st, exclude_task_ids) for tid in target_tote_ids):
                return {"valid": False}
            noise_tote_ids = [int(tid) for tid in target_tote_ids if int(tid) not in set(ordered_hits)]
            sort_layer_range = (int(lo), int(hi))
        summary = self._z_stack_summary(st, int(stack_id), exclude_task_ids)
        return {
            "valid": True,
            "target_stack_id": int(stack_id),
            "target_tote_ids": list(target_tote_ids),
            "hit_tote_ids": list(ordered_hits),
            "noise_tote_ids": list(noise_tote_ids),
            "sort_layer_range": sort_layer_range,
            "operation_mode": str(mode).upper(),
            "station_service_time": float(len(noise_tote_ids)) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)),
            "demanded_qty": int(summary.get("demanded_qty", 0)),
            "available_qty": int(summary.get("available_qty", 0)),
            "demand_ratio": float(summary.get("demand_ratio", 0.0)),
        }

    def _z_build_noise_only_sort_plan(
        self,
        st: Any,
        stack_id: int,
        tote_ids: List[int],
        exclude_task_ids: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        stack = self.problem.point_to_stack.get(int(stack_id)) if self.problem is not None else None
        if stack is None or not getattr(stack, "totes", None):
            return {"valid": False}
        tote_set = {int(tid) for tid in tote_ids if int(tid) >= 0}
        if not tote_set:
            return {"valid": False}
        layers = [int(stack.get_tote_layer(int(tid))) for tid in tote_set]
        layers = [layer for layer in layers if layer >= 0]
        if not layers:
            return {"valid": False}
        lo = min(layers)
        hi = max(layers)
        target_tote_ids = [int(getattr(tote, "id", -1)) for tote in (getattr(stack, "totes", []) or [])[lo:hi + 1]]
        if set(target_tote_ids) != tote_set:
            return {"valid": False}
        if any(int(tid) in self._z_used_tote_ids(st, exclude_task_ids) for tid in target_tote_ids):
            return {"valid": False}
        return {
            "valid": True,
            "target_stack_id": int(stack_id),
            "target_tote_ids": list(target_tote_ids),
            "hit_tote_ids": [],
            "noise_tote_ids": list(target_tote_ids),
            "sort_layer_range": (int(lo), int(hi)),
            "operation_mode": "SORT",
            "station_service_time": float(len(target_tote_ids)) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)),
            "demanded_qty": 0,
            "available_qty": 0,
            "demand_ratio": 0.0,
        }

    def _z_build_stack_plan_for_task(
        self,
        st: Any,
        task: Any,
        stack_id: int,
        mode: Optional[str] = None,
        exclude_task_ids: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        excluded = {int(x) for x in (exclude_task_ids or set())}
        excluded.add(int(getattr(task, "task_id", -1)))
        summary = self._z_stack_summary(st, int(stack_id), excluded)
        chosen_mode = str(mode or getattr(task, "operation_mode", "FLIP")).upper()
        plan = self._z_build_plan_from_hits(st, task, int(stack_id), list(summary.get("hit_tote_ids", [])), chosen_mode, excluded)
        if not bool(plan.get("valid", False)):
            return {"valid": False}
        plan["demand_ratio"] = float(summary.get("demand_ratio", 0.0))
        plan["demanded_qty"] = int(summary.get("demanded_qty", 0))
        plan["available_qty"] = int(summary.get("available_qty", 0))
        plan["hit_sku_ids"] = set(summary.get("hit_sku_ids", set()) or set())
        return plan

    def _z_apply_plan(self, task: Any, plan: Dict[str, Any]) -> None:
        task.target_stack_id = int(plan.get("target_stack_id", getattr(task, "target_stack_id", -1)))
        task.operation_mode = str(plan.get("operation_mode", getattr(task, "operation_mode", "FLIP"))).upper()
        task.target_tote_ids = list(int(x) for x in (plan.get("target_tote_ids", []) or []))
        task.hit_tote_ids = list(int(x) for x in (plan.get("hit_tote_ids", []) or []))
        task.noise_tote_ids = list(int(x) for x in (plan.get("noise_tote_ids", []) or []))
        task.sort_layer_range = copy.deepcopy(plan.get("sort_layer_range", None))
        task.station_service_time = float(plan.get("station_service_time", 0.0))

    def _z_active_hit_profile(self, stack_id: int) -> Tuple[float, float]:
        stack = self.problem.point_to_stack.get(int(stack_id)) if self.problem is not None else None
        if stack is None or not getattr(stack, "totes", None):
            return 0.0, 0.0
        stack_skus = {
            int(sku_id)
            for tote in getattr(stack, "totes", []) or []
            for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items()
            if int(qty) > 0
        }
        subtask_hits = 0.0
        sku_hits: Set[int] = set()
        for row in getattr(self.problem, "subtask_list", []) or []:
            req = self._z_subtask_demand_counts(row)
            matched = {int(sku_id) for sku_id, qty in req.items() if int(qty) > 0 and int(sku_id) in stack_skus}
            if matched:
                subtask_hits += 1.0
                sku_hits.update(matched)
        return float(subtask_hits), float(len(sku_hits))

    def _z_stack_locality_bonus(self, subtask_tasks: List[Any], candidate_stack_id: int, source_stack_id: int) -> float:
        other_stack_ids = {
            int(getattr(row, "target_stack_id", -1))
            for row in subtask_tasks
            if int(getattr(row, "target_stack_id", -1)) >= 0 and int(getattr(row, "target_stack_id", -1)) != int(source_stack_id)
        }
        if not other_stack_ids:
            return 0.0
        candidate_xy = self._stack_xy(int(candidate_stack_id))
        if candidate_xy is None:
            return 0.0
        dist_rows = [self._xy_manhattan(candidate_xy, self._stack_xy(int(stack_id))) for stack_id in other_stack_ids if self._stack_xy(int(stack_id)) is not None]
        if not dist_rows:
            return 0.0
        avg_dist = float(sum(dist_rows) / len(dist_rows))
        return float(max(0.0, 1.0 - avg_dist / max(1.0, self._warehouse_distance_scale())))

    def _z_mode_from_demand(self, current_mode: str, stack_id: int, demand_ratio: float, hit_count: int) -> str:
        normalized_mode = str(current_mode or "FLIP").upper()
        if self._z_is_congested_stack(int(stack_id)) or float(demand_ratio) >= 0.8 - 1e-9:
            return "SORT"
        if float(demand_ratio) <= 0.2 + 1e-9 or int(hit_count) <= 2:
            return "FLIP"
        return normalized_mode if normalized_mode in {"SORT", "FLIP"} else "FLIP"

    def _build_z_candidate_from_subtasks(self, rng: random.Random, priority_subtask_ids: Optional[List[int]] = None, forced_move_type: Optional[str] = None) -> Dict[str, Any]:
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "execution_tasks", None)]
        if priority_subtask_ids:
            preferred_ids = {int(x) for x in priority_subtask_ids}
            preferred = [st for st in subtasks if int(getattr(st, "id", -1)) in preferred_ids]
            if preferred:
                subtasks = preferred
        if not subtasks:
            return {
                "feasible": False,
                "move_type": "none",
                "changed_subtask_count": 0,
                "task_delta": 0,
                "stack_delta": 0,
                "mode_delta": 0,
                "z_route_insertion_detour": 0.0,
                "z_hit_frequency_bonus": 0.0,
                "z_stack_locality_score": 0.0,
                "z_demand_ratio": 0.0,
                "z_congestion_proxy": 0.0,
            }
        st = subtasks[0] if priority_subtask_ids else rng.choice(subtasks)
        subtask_id = int(getattr(st, "id", -1))
        tasks = list(getattr(st, "execution_tasks", []) or [])
        if not tasks:
            return {
                "feasible": False,
                "move_type": "none",
                "changed_subtask_count": 0,
                "task_delta": 0,
                "stack_delta": 0,
                "mode_delta": 0,
                "z_route_insertion_detour": 0.0,
                "z_hit_frequency_bonus": 0.0,
                "z_stack_locality_score": 0.0,
                "z_demand_ratio": 0.0,
                "z_congestion_proxy": 0.0,
            }
        before_task_count = len(tasks)
        before_stacks = {int(getattr(t, "target_stack_id", -1)) for t in tasks}
        before_modes = {str(getattr(t, "operation_mode", "")).upper() for t in tasks}
        move_type = str(forced_move_type or rng.choice(["stack_replace", "tote_replace_within_stack", "mode_flip_sort_toggle", "range_shrink_expand", "task_merge_split"]))
        strict_safe_ops = bool(self._z_strict_safe_operator_semantics_enabled())
        move_variant = str(move_type)
        if self._is_z_operator_banned(move_type, subtask_id):
            return {
                "feasible": False,
                "move_type": move_type,
                "changed_subtask_count": 0,
                "task_delta": 0,
                "stack_delta": 0,
                "mode_delta": 0,
                "reject_reason": "operator_banned",
                "subtask_id": subtask_id,
                "z_route_insertion_detour": 0.0,
                "z_hit_frequency_bonus": 0.0,
                "z_stack_locality_score": 0.0,
                "z_demand_ratio": 0.0,
                "z_congestion_proxy": 0.0,
            }
        hotspot_score = float(dict(self._z_hotspot_rows(limit=max(1, len(subtasks)))).get(subtask_id, 0.0))
        fallback_used = False
        fallback_type = ""
        move_metrics: Dict[str, float] = {
            "z_route_insertion_detour": 0.0,
            "z_hit_frequency_bonus": 0.0,
            "z_stack_locality_score": 0.0,
            "z_demand_ratio": 0.0,
            "z_congestion_proxy": 0.0,
        }

        def _capture_plan_metrics(plan: Dict[str, Any], hit_frequency_bonus: float = 0.0, stack_locality_score: float = 0.0) -> None:
            move_metrics["z_route_insertion_detour"] = float(self._z_best_insertion_detour(int(plan.get("target_stack_id", -1))))
            move_metrics["z_hit_frequency_bonus"] = float(hit_frequency_bonus)
            move_metrics["z_stack_locality_score"] = float(stack_locality_score)
            move_metrics["z_demand_ratio"] = float(plan.get("demand_ratio", 0.0))
            move_metrics["z_congestion_proxy"] = float(self._z_congestion_proxy(int(plan.get("target_stack_id", -1))))

        def _apply_local_shrink(task: Any, fallback_name: str) -> bool:
            nonlocal fallback_used, fallback_type
            stack_id = int(getattr(task, "target_stack_id", -1))
            hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0]
            target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0]
            plan: Dict[str, Any]
            if hit_ids:
                plan = self._z_build_plan_from_hits(st, task, stack_id, hit_ids, "FLIP", {int(getattr(task, "task_id", -1))})
                if not bool(plan.get("valid", False)):
                    plan = self._z_build_plan_from_hits(st, task, stack_id, hit_ids, "SORT", {int(getattr(task, "task_id", -1))})
            elif target_ids:
                plan = self._z_build_noise_only_sort_plan(st, stack_id, [target_ids[0]], {int(getattr(task, "task_id", -1))})
            else:
                return False
            if not bool(plan.get("valid", False)):
                return False
            if list(plan.get("target_tote_ids", [])) == target_ids and str(plan.get("operation_mode", "")).upper() == str(getattr(task, "operation_mode", "")).upper():
                return False
            self._z_apply_plan(task, plan)
            _capture_plan_metrics(plan)
            fallback_used = True
            fallback_type = str(fallback_name)
            return True

        def _drop_one_noise_tote(task: Any, fallback_name: str) -> bool:
            nonlocal fallback_used, fallback_type
            target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0]
            hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0]
            noise_ids = [tid for tid in target_ids if tid not in set(hit_ids)]
            if not noise_ids:
                return False
            current_mode = str(getattr(task, "operation_mode", "FLIP")).upper()
            if current_mode == "SORT" and target_ids:
                if int(target_ids[0]) in set(noise_ids):
                    kept_target = list(target_ids[1:])
                elif int(target_ids[-1]) in set(noise_ids):
                    kept_target = list(target_ids[:-1])
                else:
                    return False
                if not kept_target:
                    return False
                kept_hits = [int(x) for x in hit_ids if int(x) in set(kept_target)]
                if kept_hits:
                    plan = self._z_build_plan_from_hits(st, task, int(getattr(task, "target_stack_id", -1)), kept_hits, "SORT", {int(getattr(task, "task_id", -1))})
                else:
                    plan = self._z_build_noise_only_sort_plan(st, int(getattr(task, "target_stack_id", -1)), kept_target, {int(getattr(task, "task_id", -1))})
            else:
                plan = self._z_build_plan_from_hits(st, task, int(getattr(task, "target_stack_id", -1)), hit_ids, "FLIP", {int(getattr(task, "task_id", -1))}) if hit_ids else {"valid": False}
            if not bool(plan.get("valid", False)) or len(plan.get("target_tote_ids", [])) >= len(target_ids):
                return False
            self._z_apply_plan(task, plan)
            _capture_plan_metrics(plan)
            fallback_used = True
            fallback_type = str(fallback_name)
            return True

        focus_task = max(
            tasks,
            key=lambda row: (
                len(getattr(row, "noise_tote_ids", []) or []),
                len(getattr(row, "target_tote_ids", []) or []),
                -int(getattr(row, "task_id", -1)),
            ),
        )

        if move_type == "stack_replace":
            source_stack = int(getattr(focus_task, "target_stack_id", -1))
            current_hit_count = len([int(x) for x in (getattr(focus_task, "hit_tote_ids", []) or []) if int(x) >= 0])
            current_noise_count = len([int(x) for x in (getattr(focus_task, "noise_tote_ids", []) or []) if int(x) >= 0])
            candidate_stack_ids = sorted({
                int(stack_id)
                for sku in getattr(st, "unique_sku_list", []) or []
                for stack_id in self._x_candidate_stack_ids_for_sku(int(getattr(sku, "id", -1)))
                if int(stack_id) >= 0 and int(stack_id) != source_stack
            })
            candidate_profiles = {int(stack_id): self._z_active_hit_profile(int(stack_id)) for stack_id in candidate_stack_ids}
            max_subtask_hits = max([float(rows[0]) for rows in candidate_profiles.values()] + [1.0])
            max_unique_hits = max([float(rows[1]) for rows in candidate_profiles.values()] + [1.0])
            candidate_rows: List[Tuple[float, int, Dict[str, Any], float, float]] = []
            current_mode = str(getattr(focus_task, "operation_mode", "FLIP")).upper()
            for stack_id in candidate_stack_ids:
                plan = self._z_build_stack_plan_for_task(st, focus_task, int(stack_id), current_mode, {int(getattr(focus_task, "task_id", -1))})
                if not bool(plan.get("valid", False)) or not plan.get("hit_tote_ids"):
                    continue
                subtask_hits, unique_hits = candidate_profiles.get(int(stack_id), (0.0, 0.0))
                hit_bonus = 0.6 * float(subtask_hits / max_subtask_hits) + 0.4 * float(unique_hits / max_unique_hits)
                stack_locality = self._z_stack_locality_bonus(tasks, int(stack_id), source_stack)
                detour = self._z_best_insertion_detour(int(stack_id))
                if strict_safe_ops:
                    if len(list(plan.get("hit_tote_ids", []) or [])) < current_hit_count:
                        continue
                    if len(list(plan.get("noise_tote_ids", []) or [])) > current_noise_count:
                        continue
                    if float(detour) > float(getattr(self.cfg, "z_route_gap_soft_cap", 25.0)) + 1e-9:
                        continue
                candidate_rows.append((float(detour - hit_bonus - 0.5 * stack_locality), int(stack_id), plan, float(hit_bonus), float(stack_locality)))
            candidate_rows.sort(key=lambda item: (item[0], item[1]))
            if candidate_rows:
                _, _, chosen_plan, hit_bonus, stack_locality = candidate_rows[0]
                self._z_apply_plan(focus_task, chosen_plan)
                _capture_plan_metrics(chosen_plan, hit_frequency_bonus=hit_bonus, stack_locality_score=stack_locality)
                if strict_safe_ops:
                    move_variant = "detour_bounded_restack"
            elif strict_safe_ops:
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            elif not _apply_local_shrink(focus_task, "stack_replace_to_shrink"):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        elif move_type == "tote_replace_within_stack":
            task = focus_task
            stack = self.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1))) if self.problem is not None else None
            if stack is None or not getattr(stack, "totes", None):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            if str(getattr(task, "operation_mode", "FLIP")).upper() == "FLIP":
                summary = self._z_stack_summary(st, int(getattr(task, "target_stack_id", -1)), {int(getattr(task, "task_id", -1))})
                current_hits = [int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0]
                current_target_ids = {int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0}
                best_row: Optional[Tuple[float, int, Dict[str, Any]]] = None
                for tote in self._z_available_stack_totes(st, int(getattr(task, "target_stack_id", -1)), {int(getattr(task, "task_id", -1))}):
                    tote_id = int(getattr(tote, "id", -1))
                    if tote_id in current_target_ids:
                        continue
                    gain = 0.0
                    for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
                        gain += float(min(int(summary.get("remaining_demand", {}).get(int(sku_id), 0)), int(qty)))
                    if gain <= 0.0:
                        continue
                    plan = self._z_build_plan_from_hits(st, task, int(getattr(task, "target_stack_id", -1)), current_hits + [tote_id], "FLIP", {int(getattr(task, "task_id", -1))})
                    if not bool(plan.get("valid", False)):
                        continue
                    row = (-float(gain), int(tote_id), plan)
                    if best_row is None or row < best_row:
                        best_row = row
                if best_row is not None:
                    _, _, plan = best_row
                    self._z_apply_plan(task, plan)
                    _capture_plan_metrics(plan)
                    if strict_safe_ops:
                        move_variant = "single_tote_gain"
                elif strict_safe_ops:
                    if _drop_one_noise_tote(task, "tote_replace_drop_noise"):
                        move_variant = "single_tote_noise_drop"
                    else:
                        return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
                elif not _drop_one_noise_tote(task, "tote_replace_drop_noise") and not _apply_local_shrink(task, "tote_replace_drop_noise"):
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            elif strict_safe_ops:
                if _drop_one_noise_tote(task, "tote_replace_drop_noise"):
                    move_variant = "single_tote_noise_drop"
                else:
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            elif not _drop_one_noise_tote(task, "tote_replace_drop_noise") and not _apply_local_shrink(task, "tote_replace_drop_noise"):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        elif move_type == "mode_flip_sort_toggle":
            task = focus_task
            stack_id = int(getattr(task, "target_stack_id", -1))
            stack = self.problem.point_to_stack.get(stack_id) if self.problem is not None else None
            if stack is None or not getattr(stack, "totes", None):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            excluded = {int(getattr(task, "task_id", -1))}
            summary = self._z_stack_summary(st, stack_id, excluded)
            current_mode = str(getattr(task, "operation_mode", "FLIP")).upper()
            desired_mode = self._z_mode_from_demand(
                current_mode,
                stack_id,
                float(summary.get("demand_ratio", 0.0)),
                len(summary.get("hit_tote_ids", []) or []),
            )
            if desired_mode != current_mode:
                plan = self._z_build_stack_plan_for_task(st, task, stack_id, desired_mode, excluded)
                if bool(plan.get("valid", False)) and (
                    str(plan.get("operation_mode", "")).upper() != current_mode
                    or list(plan.get("target_tote_ids", [])) != list(getattr(task, "target_tote_ids", []) or [])
                ):
                    self._z_apply_plan(task, plan)
                    _capture_plan_metrics(plan)
                elif not _apply_local_shrink(task, "mode_flip_to_shrink"):
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            elif not _apply_local_shrink(task, "mode_flip_to_shrink"):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        elif move_type == "range_shrink_expand":
            task = focus_task
            stack_id = int(getattr(task, "target_stack_id", -1))
            stack = self.problem.point_to_stack.get(stack_id) if self.problem is not None else None
            if stack is None or not getattr(stack, "totes", None):
                return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            task_id = int(getattr(task, "task_id", -1))
            current_mode = str(getattr(task, "operation_mode", "FLIP")).upper()
            current_target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0]
            current_hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0]
            shrink_plan = {"valid": False}
            if current_hit_ids:
                shrink_plan = self._z_build_plan_from_hits(st, task, stack_id, current_hit_ids, current_mode, {task_id})
                if not bool(shrink_plan.get("valid", False)) and current_mode == "SORT":
                    shrink_plan = self._z_build_plan_from_hits(st, task, stack_id, current_hit_ids, "FLIP", {task_id})
            shrink_changed = bool(shrink_plan.get("valid", False)) and (
                list(shrink_plan.get("target_tote_ids", [])) != current_target_ids
                or str(shrink_plan.get("operation_mode", "")).upper() != current_mode
            )
            if strict_safe_ops:
                shrink_keeps_hits = set(int(x) for x in (shrink_plan.get("hit_tote_ids", []) or [])) == set(current_hit_ids)
                shrink_reduces_targets = len(list(shrink_plan.get("target_tote_ids", []) or [])) < len(current_target_ids)
                if shrink_changed and shrink_keeps_hits and shrink_reduces_targets:
                    self._z_apply_plan(task, shrink_plan)
                    _capture_plan_metrics(shrink_plan)
                    move_variant = "shrink_only"
                elif not current_hit_ids and _drop_one_noise_tote(task, "range_shrink_only"):
                    move_variant = "shrink_only"
                else:
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
            elif shrink_changed and len(list(shrink_plan.get("target_tote_ids", [])) or []) <= len(current_target_ids):
                self._z_apply_plan(task, shrink_plan)
                _capture_plan_metrics(shrink_plan)
            else:
                remaining = self._z_remaining_demand(st, {task_id})
                used_other = self._z_used_tote_ids(st, {task_id})
                current_layers = [int(stack.get_tote_layer(int(tid))) for tid in current_target_ids]
                current_layers = [layer for layer in current_layers if layer >= 0]
                best_expand: Optional[Tuple[float, int, Dict[str, Any]]] = None
                if current_layers:
                    lo = min(current_layers)
                    hi = max(current_layers)
                    for layer_idx in [lo - 1, hi + 1]:
                        if layer_idx < 0 or layer_idx >= len(getattr(stack, "totes", []) or []):
                            continue
                        tote = getattr(stack, "totes", [])[layer_idx]
                        tote_id = int(getattr(tote, "id", -1))
                        if tote_id in used_other:
                            continue
                        gain = 0.0
                        for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
                            gain += float(min(int(remaining.get(int(sku_id), 0)), int(qty)))
                        if gain <= 0.0:
                            continue
                        expanded_hits = list(dict.fromkeys(list(current_hit_ids) + [tote_id]))
                        plan = self._z_build_plan_from_hits(st, task, stack_id, expanded_hits, "SORT", {task_id})
                        if not bool(plan.get("valid", False)):
                            continue
                        row = (-float(gain), len(plan.get("target_tote_ids", []) or []), int(tote_id), plan)
                        if best_expand is None or row < best_expand:
                            best_expand = row
                if best_expand is not None:
                    _, _, _, plan = best_expand
                    self._z_apply_plan(task, plan)
                    _capture_plan_metrics(plan)
                elif not _apply_local_shrink(task, "range_expand_to_shrink"):
                    return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        else:
            def _task_layer_bounds(task_obj: Any) -> Optional[Tuple[int, int]]:
                stack_obj = self.problem.point_to_stack.get(int(getattr(task_obj, "target_stack_id", -1))) if self.problem is not None else None
                if stack_obj is None or not getattr(stack_obj, "totes", None):
                    return None
                layers = [int(stack_obj.get_tote_layer(int(tid))) for tid in (getattr(task_obj, "target_tote_ids", []) or [])]
                layers = [layer for layer in layers if layer >= 0]
                if not layers:
                    return None
                return int(min(layers)), int(max(layers))

            merge_choice: Optional[Tuple[float, int, int, Dict[str, Any]]] = None
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    left = tasks[i]
                    right = tasks[j]
                    left_stack = int(getattr(left, "target_stack_id", -1))
                    right_stack = int(getattr(right, "target_stack_id", -1))
                    if left_stack < 0 or left_stack != right_stack:
                        continue
                    if int(getattr(left, "target_station_id", -1)) != int(getattr(right, "target_station_id", -1)):
                        continue
                    left_bounds = _task_layer_bounds(left)
                    right_bounds = _task_layer_bounds(right)
                    if left_bounds is None or right_bounds is None:
                        continue
                    if left_bounds[1] + 1 < right_bounds[0] and right_bounds[1] + 1 < left_bounds[0]:
                        continue
                    excluded = {int(getattr(left, "task_id", -1)), int(getattr(right, "task_id", -1))}
                    merged_hits = list(dict.fromkeys(
                        [int(x) for x in (getattr(left, "hit_tote_ids", []) or []) if int(x) >= 0]
                        + [int(x) for x in (getattr(right, "hit_tote_ids", []) or []) if int(x) >= 0]
                    ))
                    summary = self._z_stack_summary(st, left_stack, excluded)
                    merged_mode = self._z_mode_from_demand(
                        str(getattr(left, "operation_mode", "FLIP")).upper(),
                        left_stack,
                        float(summary.get("demand_ratio", 0.0)),
                        len(merged_hits),
                    )
                    plan = self._z_build_plan_from_hits(st, left, left_stack, merged_hits, merged_mode, excluded)
                    if not bool(plan.get("valid", False)):
                        continue
                    current_ratio = float(
                        (len(getattr(left, "hit_tote_ids", []) or []) + len(getattr(right, "hit_tote_ids", []) or []))
                        / max(1, len(getattr(left, "target_tote_ids", []) or []) + len(getattr(right, "target_tote_ids", []) or []))
                    )
                    merged_ratio = float(len(plan.get("hit_tote_ids", []) or []) / max(1, len(plan.get("target_tote_ids", []) or [])))
                    if merged_ratio + 1e-9 < current_ratio:
                        continue
                    row = (-float(merged_ratio - current_ratio), len(plan.get("target_tote_ids", []) or []), i, j, plan)
                    if merge_choice is None or row < merge_choice:
                        merge_choice = row
            if merge_choice is not None:
                _, _, i, j, plan = merge_choice
                left = tasks[i]
                right = tasks[j]
                self._z_apply_plan(left, plan)
                left.target_station_id = int(getattr(right, "target_station_id", getattr(left, "target_station_id", -1)))
                st.execution_tasks.remove(right)
                _capture_plan_metrics(plan)
            else:
                task = focus_task
                stack_id = int(getattr(task, "target_stack_id", -1))
                stack = self.problem.point_to_stack.get(stack_id) if self.problem is not None else None
                target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0]
                hit_set = {int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0}
                if stack is None or not getattr(stack, "totes", None) or not target_ids:
                    if not _drop_one_noise_tote(task, "task_split_drop_noise") and not _apply_local_shrink(task, "task_split_noop_after_shrink"):
                        return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
                else:
                    left_noise: List[int] = []
                    for tote_id in target_ids:
                        if tote_id in hit_set:
                            break
                        left_noise.append(int(tote_id))
                    right_noise: List[int] = []
                    for tote_id in reversed(target_ids):
                        if tote_id in hit_set:
                            break
                        right_noise.append(int(tote_id))
                    right_noise.reverse()
                    boundary_noise = left_noise if len(left_noise) >= len(right_noise) else right_noise
                    current_mode = str(getattr(task, "operation_mode", "FLIP")).upper()
                    remaining_hits = [tid for tid in target_ids if tid in hit_set]
                    split_done = False
                    if boundary_noise and remaining_hits:
                        hit_plan = self._z_build_plan_from_hits(st, task, stack_id, remaining_hits, current_mode, {int(getattr(task, "task_id", -1))})
                        noise_plan = self._z_build_noise_only_sort_plan(st, stack_id, boundary_noise, {int(getattr(task, "task_id", -1))})
                        if bool(hit_plan.get("valid", False)) and bool(noise_plan.get("valid", False)):
                            new_task = copy.deepcopy(task)
                            new_task.task_id = int(self._next_task_id())
                            new_task.target_station_id = int(getattr(task, "target_station_id", -1))
                            self._z_apply_plan(task, hit_plan)
                            self._z_apply_plan(new_task, noise_plan)
                            st.execution_tasks.append(new_task)
                            _capture_plan_metrics(hit_plan)
                            split_done = True
                    if not split_done and not _drop_one_noise_tote(task, "task_split_drop_noise") and not _apply_local_shrink(task, "task_split_noop_after_shrink"):
                        return {"feasible": False, "move_type": move_type, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}

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
        after_modes = {str(getattr(t, "operation_mode", "")).upper() for t in after_tasks}
        return {
            "feasible": True,
            "move_type": move_type,
            "changed_subtask_count": 1,
            "task_delta": int(len(after_tasks) - before_task_count),
            "stack_delta": int(len(after_stacks.symmetric_difference(before_stacks))),
            "mode_delta": int(len(after_modes.symmetric_difference(before_modes))),
            "move_variant": str(move_variant),
            "reject_reason": "",
            "used_stack_count": int(subtask_info.get("used_stack_count", len(after_stacks))),
            "subtask_id": int(subtask_id),
            "z_hotspot_score": float(hotspot_score),
            "z_route_gap_penalty": float(move_metrics.get("z_route_insertion_detour", 0.0)),
            "z_operator_fallback_used": bool(fallback_used),
            "z_fallback_type": str(fallback_type),
            **move_metrics,
        }

    def _select_priority_z_subtasks(self, limit: int = 2) -> List[int]:
        return [sid for _, sid in self._z_hotspot_rows(limit=limit)]

    def _is_micro_scale_case(self) -> bool:
        scale = str(getattr(self.cfg, "scale", "")).upper()
        return scale in {"SMALL", "SMALL2", "SMALL_ZRICH", "SMALL2_ZRICH", "SMALL_UNEVEN", "SMALL2_UNEVEN"}

    def _z_hotspot_rows(self, limit: Optional[int] = None) -> List[Tuple[float, int]]:
        safe_rows: List[Tuple[float, float, int]] = []
        all_rows: List[Tuple[float, float, int]] = []
        limit = max(1, int(limit or getattr(self.cfg, "z_hotspot_topk", 3)))
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            anchor_arrival = float(self.anchor_reference.get("subtask_arrival", {}).get(sid, self._estimate_subtask_arrival(st)))
            anchor_start = float(self.anchor_reference.get("subtask_start", {}).get(sid, self._estimate_subtask_start(st)))
            arrival = float(self._estimate_subtask_arrival(st))
            start = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            wait_overflow = max(0.0, start - anchor_start)
            arrival_shift = abs(arrival - anchor_arrival)
            stack_ids = {
                int(getattr(task, "target_stack_id", -1))
                for task in getattr(st, "execution_tasks", []) or []
                if int(getattr(task, "target_stack_id", -1)) >= 0
            }
            multi_stack_pen = max(0.0, float(len(stack_ids)) - 1.0)
            noise_tote_count = sum(float(len(getattr(task, "noise_tote_ids", []) or [])) for task in getattr(st, "execution_tasks", []) or [])
            proc = self._estimate_subtask_processing_time(st, self.last_sp3_sorting_costs)
            proc_overflow = max(0.0, proc - self._estimate_subtask_slack(st))
            route_ok = bool(
                arrival_shift <= float(getattr(self.cfg, "z_arrival_shift_soft_cap", 140.0)) + 1e-9
                and wait_overflow <= float(getattr(self.cfg, "z_wait_overflow_soft_cap", 180.0)) + 1e-9
            )
            structural_score = 3.0 * noise_tote_count + 2.0 * multi_stack_pen + 2.0 * proc_overflow
            route_risk = float(arrival_shift + wait_overflow)
            row = (float(structural_score), float(route_risk), int(sid))
            all_rows.append(row)
            if route_ok:
                safe_rows.append(row)
        ranked_rows = safe_rows if safe_rows else all_rows
        ranked_rows.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(float(score), int(sid)) for score, _, sid in ranked_rows[:limit]]

    def _z_operator_ban_key(self, operator: str, subtask_id: int) -> Tuple[str, int]:
        return (str(operator), int(subtask_id))

    def _is_z_operator_banned(self, operator: str, subtask_id: int) -> bool:
        key = self._z_operator_ban_key(operator, subtask_id)
        decay_rounds = max(0, int(getattr(self.cfg, "z_operator_failure_decay_rounds", 4)))
        last_iter = int(self.z_operator_subtask_failure_iter.get(key, -1))
        if decay_rounds > 0 and last_iter >= 0 and abs(int(self.current_iter) - last_iter) > decay_rounds:
            self.z_operator_subtask_failures.pop(key, None)
            self.z_operator_subtask_failure_iter.pop(key, None)
            self.z_operator_subtask_bans.discard(key)
        return key in self.z_operator_subtask_bans

    def _record_z_operator_failure(self, operator: str, subtask_id: int, catastrophic: bool = False) -> None:
        key = self._z_operator_ban_key(operator, subtask_id)
        if not bool(catastrophic):
            return
        self.z_operator_subtask_failures[key] = int(self.z_operator_subtask_failures.get(key, 0)) + 1
        self.z_operator_subtask_failure_iter[key] = int(getattr(self, "current_iter", 0))
        if int(self.z_operator_subtask_failures.get(key, 0)) >= max(1, int(getattr(self.cfg, "z_operator_subtask_ban_after_failures", 2))):
            self.z_operator_subtask_bans.add(key)

    def _clear_z_operator_failure(self, operator: str, subtask_id: int) -> None:
        key = self._z_operator_ban_key(operator, subtask_id)
        self.z_operator_subtask_failures.pop(key, None)
        self.z_operator_subtask_bans.discard(key)

    def _is_x_signature_temporarily_blocked(self, signature: str) -> bool:
        signature = str(signature or "").strip()
        if not signature:
            return False
        until_iter = int(self.x_signature_reject_until.get(signature, -1))
        if until_iter < int(getattr(self, "current_iter", 0)):
            self.x_signature_reject_until.pop(signature, None)
            return False
        return True

    def _record_x_signature_reject(self, signature: str, catastrophic: bool = False) -> None:
        signature = str(signature or "").strip()
        if not signature:
            return
        curr_iter = int(getattr(self, "current_iter", 0))
        if catastrophic:
            horizon = max(2, int(getattr(self.cfg, "layer_restart_patience", 2)) * 2)
        else:
            horizon = 1
        self.x_signature_reject_until[signature] = max(
            int(self.x_signature_reject_until.get(signature, -1)),
            curr_iter + horizon,
        )

    def _z_signature_cache_key(self, signature: str) -> str:
        return f"{int(getattr(self, 'anchor_version', 0))}::{str(signature or '').strip()}"

    def _is_z_signature_reject_blocked(self, signature: str) -> bool:
        signature = str(signature or "").strip()
        if not signature:
            return False
        return str(self._z_signature_cache_key(signature)) in self.z_signature_reject_cache

    def _record_z_signature_reject(self, signature: str) -> None:
        signature = str(signature or "").strip()
        if not signature:
            return
        self.z_signature_reject_cache.add(str(self._z_signature_cache_key(signature)))

    def _z_generation_guard_reason_from_metrics(
        self,
        guardrails: Dict[str, float],
        move_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        move_meta = move_meta or {}
        arrival_shift = float(guardrails.get("arrival_shift_estimate", 0.0))
        wait_overflow = float(guardrails.get("wait_overflow_estimate", 0.0))
        route_tail_delta = float(guardrails.get("route_tail_delta_estimate", 0.0))
        route_detour = float(move_meta.get("z_route_insertion_detour", move_meta.get("z_route_gap_penalty", 0.0)))
        if arrival_shift > float(getattr(self.cfg, "z_arrival_shift_soft_cap", 140.0)) + 1e-9:
            return "z_arrival_shift_soft_cap"
        if wait_overflow > float(getattr(self.cfg, "z_wait_overflow_soft_cap", 180.0)) + 1e-9:
            return "z_wait_overflow_soft_cap"
        if route_tail_delta > float(getattr(self.cfg, "z_route_tail_soft_cap", 90.0)) + 1e-9:
            return "z_route_tail_soft_cap"
        if route_detour > float(getattr(self.cfg, "z_route_gap_soft_cap", 25.0)) + 1e-9:
            return "z_route_gap_soft_cap"
        return ""

    def _estimate_z_candidate_guardrails(self) -> Dict[str, float]:
        arrival_shift_total = 0.0
        wait_overflow_total = 0.0
        route_tail_delta = 0.0
        changed_task_count = 0.0
        changed_stack_count = 0.0
        anchor_tasks = self.anchor_reference.get("anchor_task_profile", {}) or {}
        current_stack_ids: Set[int] = set()
        anchor_stack_ids: Set[int] = set()
        for task in self._collect_all_tasks():
            task_id = int(getattr(task, "task_id", -1))
            current_stack_id = int(getattr(task, "target_stack_id", -1))
            if current_stack_id >= 0:
                current_stack_ids.add(current_stack_id)
            anchor_profile = anchor_tasks.get(task_id, {})
            anchor_stack_id = int(anchor_profile.get("stack_id", -1))
            if anchor_stack_id >= 0:
                anchor_stack_ids.add(anchor_stack_id)
            anchor_arrival_station = float(anchor_profile.get("arrival_station", float(getattr(task, "arrival_time_at_station", 0.0))))
            arrival_station = float(getattr(task, "arrival_time_at_station", 0.0))
            arrival_shift_total += abs(arrival_station - anchor_arrival_station)
            anchor_arrival_stack = float(anchor_profile.get("arrival_stack", float(getattr(task, "arrival_time_at_stack", 0.0))))
            route_tail_delta = max(
                route_tail_delta,
                max(0.0, float(getattr(task, "arrival_time_at_stack", 0.0)) - anchor_arrival_stack),
                max(0.0, arrival_station - anchor_arrival_station),
            )
            if (
                int(anchor_profile.get("subtask_id", int(getattr(task, "sub_task_id", -1)))) != int(getattr(task, "sub_task_id", -1))
                or int(anchor_profile.get("stack_id", current_stack_id)) != current_stack_id
                or str(anchor_profile.get("operation_mode", str(getattr(task, "operation_mode", "")))) != str(getattr(task, "operation_mode", ""))
            ):
                changed_task_count += 1.0
        for st in getattr(self.problem, "subtask_list", []) or []:
            sid = int(getattr(st, "id", -1))
            start_now = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
            anchor_start = float(self.anchor_reference.get("subtask_start", {}).get(sid, start_now))
            wait_overflow_total += max(0.0, start_now - anchor_start)
        changed_stack_count = float(len(current_stack_ids.symmetric_difference(anchor_stack_ids)))
        return {
            "arrival_shift_estimate": float(arrival_shift_total),
            "wait_overflow_estimate": float(wait_overflow_total),
            "route_tail_delta_estimate": float(route_tail_delta),
            "changed_task_count": float(changed_task_count),
            "changed_stack_count": float(changed_stack_count),
        }

    def _x_assignment_diff_stats(self, proposal: XSplitProposal) -> Dict[str, Any]:
        anchor_sku_profile = self.anchor_reference.get("anchor_sku_profile", {}) or {}
        anchor_group_by_order: Dict[int, Set[Tuple[int, ...]]] = defaultdict(set)
        anchor_group_by_pair: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        for key, profile in anchor_sku_profile.items():
            order_id = int(key[0])
            anchor_subtask_id = int(profile.get("anchor_subtask_id", -1))
            if anchor_subtask_id < 0:
                continue
            group = tuple(sorted(
                int(k[1]) for k, p in anchor_sku_profile.items()
                if int(k[0]) == order_id and int(p.get("anchor_subtask_id", -1)) == anchor_subtask_id
            ))
            if not group:
                continue
            anchor_group_by_order[order_id].add(group)
            anchor_group_by_pair[(order_id, int(key[1]))] = group

        proposal_group_by_pair: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        changed_orders: Set[int] = set()
        proposal_group_by_order: Dict[int, Set[Tuple[int, ...]]] = defaultdict(set)
        preserved_groups = 0
        touched_orders = {int(x) for x in (proposal.touched_orders or set()) if int(x) >= 0}
        for order_id, groups in (proposal.order_to_subtask_sku_sets or {}).items():
            normalized_groups = [tuple(sorted(int(x) for x in group if int(x) >= 0)) for group in groups if group]
            for group in normalized_groups:
                proposal_group_by_order[int(order_id)].add(group)
                for sku_id in group:
                    proposal_group_by_pair[(int(order_id), int(sku_id))] = group
        total_anchor_groups = 0
        for order_id, anchor_groups in anchor_group_by_order.items():
            total_anchor_groups += len(anchor_groups)
            if int(order_id) in touched_orders:
                continue
            proposal_groups = proposal_group_by_order.get(int(order_id), set())
            for group in anchor_groups:
                if group in proposal_groups and all(anchor_group_by_pair.get((int(order_id), int(sku_id))) == group for sku_id in group):
                    preserved_groups += 1
        for order_id, groups in proposal_group_by_order.items():
            if set(groups) != anchor_group_by_order.get(int(order_id), set()):
                changed_orders.add(int(order_id))

        moved_rows = {
            int(sku_id)
            for sku_id in (getattr(proposal, "x_directly_moved_sku_ids", []) or [])
            if int(sku_id) >= 0
        }
        destroy_order_id = int(getattr(proposal, "x_destroy_order_id", -1))
        if moved_rows and destroy_order_id >= 0:
            changed_assignment_pair_count = int(len(moved_rows))
            changed_orders.add(int(destroy_order_id))
        else:
            changed_assignment_pair_count = 0
            all_pairs = set(anchor_group_by_pair.keys()) | set(proposal_group_by_pair.keys())
            for pair in all_pairs:
                if proposal_group_by_pair.get(pair) != anchor_group_by_pair.get(pair):
                    changed_assignment_pair_count += 1
                    changed_orders.add(int(pair[0]))

        delta_subtask_count = abs(int(getattr(proposal, "subtask_count", 0)) - int(len(self._iter_snapshot_subtasks(self.anchor))))
        x_micro_move_size = int(changed_assignment_pair_count + delta_subtask_count)
        changed_order_rows = [len(self._get_order_unique_sku_ids(int(order_id))) for order_id in changed_orders if int(order_id) >= 0]
        affected_bom_ratio = float(getattr(proposal, "x_affected_bom_ratio_hint", 0.0))
        if changed_order_rows and changed_assignment_pair_count > 0:
            affected_bom_ratio = float(changed_assignment_pair_count / max(1, max(changed_order_rows)))
        return {
            "changed_orders_set": changed_orders,
            "changed_assignment_pair_count": int(changed_assignment_pair_count),
            "anchor_template_preservation_ratio": float(preserved_groups / max(1, total_anchor_groups)),
            "x_micro_move_size": int(x_micro_move_size),
            "affected_bom_ratio": float(max(0.0, affected_bom_ratio)),
        }

    def _x_template_change_counts(self, proposal: XSplitProposal) -> Dict[str, Any]:
        anchor_sku_profile = self.anchor_reference.get("anchor_sku_profile", {}) or {}
        diff_stats = self._x_assignment_diff_stats(proposal)
        features = self._x_sku_feature_map()
        anchor_subtask_count = int(len(self._iter_snapshot_subtasks(self.anchor)))
        station_template_change_count = 0
        robot_trip_template_change_count = 0
        route_spans: List[float] = []
        completion_spans: List[float] = []
        current_order_groups = proposal.order_to_subtask_sku_sets or {}
        for order_id, groups in current_order_groups.items():
            for group in groups:
                normalized = sorted(int(x) for x in group if int(x) >= 0)
                if not normalized:
                    continue
                station_ids = {int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("station_id", -1)) for sku_id in normalized}
                robot_ids = {int(anchor_sku_profile.get((int(order_id), sku_id), {}).get("robot_id", -1)) for sku_id in normalized}
                if len([sid for sid in station_ids if sid >= 0]) > 1:
                    station_template_change_count += 1
                if len([rid for rid in robot_ids if rid >= 0]) > 1:
                    robot_trip_template_change_count += 1
                route_positions = [float(features.get((int(order_id), sku_id), {}).get("route_pos", 0.0)) for sku_id in normalized]
                completions = [float(features.get((int(order_id), sku_id), {}).get("completion", 0.0)) for sku_id in normalized]
                route_spans.append((max(route_positions) - min(route_positions)) if route_positions else 0.0)
                completion_spans.append((max(completions) - min(completions)) if completions else 0.0)
        return {
            "changed_orders": int(len(diff_stats.get("changed_orders_set", set()))),
            "x_changed_assignment_pair_count": int(diff_stats.get("changed_assignment_pair_count", 0)),
            "delta_subtask_count": int(abs(int(getattr(proposal, "subtask_count", 0)) - anchor_subtask_count)),
            "x_micro_move_size": int(diff_stats.get("x_micro_move_size", 0)),
            "anchor_template_preservation_ratio": float(diff_stats.get("anchor_template_preservation_ratio", 0.0)),
            "affected_bom_ratio": float(diff_stats.get("affected_bom_ratio", 0.0)),
            "station_template_change_count": int(station_template_change_count),
            "robot_trip_template_change_count": int(robot_trip_template_change_count),
            "group_route_span_delta": float(max(route_spans) if route_spans else 0.0),
            "group_completion_span_delta": float(max(completion_spans) if completion_spans else 0.0),
        }

    def _x_equivalent_candidate_signature(self, proposal: XSplitProposal) -> str:
        order_part = tuple(
            (
                int(order_id),
                tuple(tuple(sorted(int(x) for x in group if int(x) >= 0)) for group in (proposal.order_to_subtask_sku_sets or {}).get(int(order_id), []))
            )
            for order_id in sorted((proposal.order_to_subtask_sku_sets or {}).keys())
        )
        return repr(order_part)

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
        if self._shadow_chain_enabled() and int(getattr(self, "current_iter", 0)) >= int(getattr(self.cfg, "max_iters", 0)):
            return True, "final_iteration"
        if self.shadow_depth >= int(self._shadow_chain_depth_limit()):
            return True, "shadow_depth_limit"
        if self._runtime_guard_level() >= 1:
            return False, "runtime_guard"
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
            "global_eval_sp2_time_sec": 0.0,
            "global_eval_sp3_time_sec": 0.0,
            "global_eval_sp4_time_sec": 0.0,
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
            elapsed = float(time.perf_counter() - t0)
            runtime["sp2_time_sec"] += elapsed
            runtime["global_eval_sp2_time_sec"] += elapsed
        if last_layer in {"X", "Y"}:
            t0 = time.perf_counter()
            self._run_sp3()
            runtime["sp3_called"] += 1.0
            elapsed = float(time.perf_counter() - t0)
            runtime["sp3_time_sec"] += elapsed
            runtime["global_eval_sp3_time_sec"] += elapsed
        if last_layer in {"X", "Y"}:
            t0 = time.perf_counter()
            self._run_sp4_augmented()
            runtime["sp4_called"] += 1.0
            elapsed = float(time.perf_counter() - t0)
            runtime["sp4_time_sec"] += elapsed
            runtime["global_eval_sp4_time_sec"] += elapsed
        elif last_layer == "Z":
            if bool(getattr(self.cfg, "u_global_sp4_polish", False)):
                t0 = time.perf_counter()
                self._run_sp4_augmented()
                runtime["sp4_called"] += 1.0
                elapsed = float(time.perf_counter() - t0)
                runtime["sp4_time_sec"] += elapsed
                runtime["global_eval_sp4_time_sec"] += elapsed
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





    # ----------------------------
    # 评估函数与下界
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
        # 工作站下界：总工作量/站数 与 单站最大工作量 取 max
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
        # sp3 改变选箱会同时影响运输与工作量结构，因此取 max 更稳
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

        # 恢复 SP2 的站点与顺位，再重建 SP3->SP4->仿真
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

        # 回填 subtask 的 assigned_robot_id，供日志/仿真输出使用
        st_map = {st.id: st for st in self.problem.subtask_list}
        for st_id, robot_id in (robot_assign or {}).items():
            if st_id in st_map:
                st_map[st_id].assigned_robot_id = int(robot_id)

    # ----------------------------
    # 主入口
    # ----------------------------
    def initialize(self):
        self._reset_runtime_caches()
        self._resolved_log_dir = None
        self.run_start_time_sec = float(time.perf_counter())
        self.run_total_time_sec = 0.0
        self.layer_runtime_sec_by_name = {name: 0.0 for name in self.layer_names}
        self.layer_trial_count_by_name = {name: 0.0 for name in self.layer_names}
        self.global_eval_count = 0
        self._set_seed(self.cfg.seed)
        self.problem = CreateOFSProblem.generate_problem_by_scale(self.cfg.scale, seed=self.cfg.seed)
        setattr(self.problem, "runtime_result_dir", self._ensure_log_dir())
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
        self.initial_makespan_raw = float(z0)
        initial_coverage = self._compute_solution_coverage()
        initial_coverage_ok = bool(initial_coverage.get("coverage_ok", False))
        incumbent_z0 = float(z0) if bool(initial_coverage_ok) else float("inf")
        # harvest soft-coupling caches
        self._harvest_station_start_times()
        self._update_beta_from_station()

        self.best = self.snapshot(incumbent_z0, iter_id=0, lightweight=True)
        self.work = self.snapshot(z0, iter_id=0, lightweight=True)
        self.work_z = float(z0)
        init_resource_time_runtime_state(self, z0)
        self._refresh_runtime_cache(z0)
        self._append_iter_log(0, focus="init", z=incumbent_z0, improved=bool(initial_coverage_ok), skipped=False, lb=None)

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

    def _y_subtask_priority_score(
        self,
        st: Any,
        station_counts: Optional[Dict[int, int]] = None,
        anchor_station_map: Optional[Dict[int, int]] = None,
        load_skew_mode: bool = False,
    ) -> float:
        station_counts = station_counts or self._current_station_subtask_counts()
        anchor_station_map = anchor_station_map or self._anchor_subtask_station_map()
        subtask_id = int(getattr(st, "id", -1))
        arrival = float(self._estimate_subtask_arrival(st))
        start = float(getattr(st, "estimated_process_start_time", self._estimate_subtask_start(st)))
        wait = max(0.0, start - arrival)
        anchor_slack = float(self.anchor_reference.get("subtask_slack", {}).get(subtask_id, self._estimate_subtask_slack(st)))
        anchor_start = float(self.anchor_reference.get("subtask_start", {}).get(subtask_id, self._estimate_subtask_start(st)))
        mismatch = max(0.0, start - anchor_start)
        station_id = int(getattr(st, "assigned_station_id", -1))
        station_load = float(station_counts.get(station_id, 0))
        station_mean = float(sum(station_counts.values()) / len(station_counts)) if station_counts else 0.0
        congestion = max(0.0, station_load - station_mean)
        anchor_drift = 1.0 if int(anchor_station_map.get(subtask_id, station_id)) != station_id else 0.0
        base_score = wait + max(0.0, wait - anchor_slack) + mismatch
        if load_skew_mode:
            base_score += 2.0 * congestion + 0.75 * anchor_drift
        else:
            base_score += 0.5 * congestion + 0.25 * anchor_drift
        return float(base_score)

    def _y_station_target_score(
        self,
        order_id: int,
        station_id: int,
        context: SP2LayerContext,
        load_now: Dict[int, int],
        prefer_light: bool = False,
    ) -> float:
        penalty = float(context.order_station_penalty.get((int(order_id), int(station_id)), 0.0))
        load_term = float(load_now.get(int(station_id), 0))
        if prefer_light:
            penalty += 1.25 * load_term
        else:
            penalty += 0.35 * load_term
        return float(penalty)

    def _y_priority_subtasks(self, limit: int = 3) -> List[Any]:
        load_skew_mode = bool(self._y_load_skew_context().get("enabled", False))
        station_counts = self._current_station_subtask_counts()
        anchor_station_map = self._anchor_subtask_station_map()
        rows: List[Tuple[float, Any]] = []
        for st in getattr(self.problem, "subtask_list", []) or []:
            rows.append((
                self._y_subtask_priority_score(
                    st,
                    station_counts=station_counts,
                    anchor_station_map=anchor_station_map,
                    load_skew_mode=load_skew_mode,
                ),
                st,
            ))
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

    def _filter_y_operator_sequence(self, operator_sequence: List[str], search_mode: str, budget: int, load_skew_mode: bool = False) -> List[str]:
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
        if load_skew_mode:
            prioritized = [
                "station_block_relocate",
                "cross_station_reinsert",
                "rank_reinsert_within_station",
                "congested_station_destroy_repair",
            ]
            ordered: List[str] = []
            for op in prioritized + list(operator_sequence) + local_ops + destroy_ops:
                if op in local_ops + destroy_ops and op not in ordered:
                    ordered.append(op)
                if len(ordered) >= int(budget):
                    break
            return ordered[:max(1, int(budget))]
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
        if bool(self._y_load_skew_context().get("enabled", False)):
            station_limit += 1
            rank_window += 1
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
        load_skew_mode = bool(self._y_load_skew_context().get("enabled", False))
        target_rows = self._y_priority_subtasks(limit=max(2, strength + 1))
        block_size = max(2, int(getattr(self.cfg, "y_block_move_size", 3)))
        destroy_fraction = self._current_y_destroy_fraction()
        order_groups: Dict[int, List[Any]] = defaultdict(list)
        station_groups: Dict[int, List[Any]] = defaultdict(list)
        for st in subtasks:
            oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            order_groups[oid].append(st)
            station_groups[int(getattr(st, "assigned_station_id", -1))].append(st)
        load_now = {sid: len(station_groups.get(sid, [])) for sid in station_ids}
        anchor_rank_map = self._anchor_subtask_rank_map()
        if operator == "station_reassign_single":
            st = rng.choice(target_rows or subtasks)
            curr_sid = int(getattr(st, "assigned_station_id", -1))
            candidate_station_ids = [sid for sid in station_ids if sid != curr_sid]
            if not candidate_station_ids:
                return False
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            best_sid = min(
                candidate_station_ids,
                key=lambda sid: self._y_station_target_score(
                    order_id,
                    sid,
                    context,
                    load_now,
                    prefer_light=load_skew_mode,
                ),
            )
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
            sid = max(crowded, key=lambda sid_i: (len(station_groups.get(sid_i, [])), sid_i)) if load_skew_mode else rng.choice(crowded)
            rows = sorted(station_groups[sid], key=lambda st: (int(getattr(st, "station_sequence_rank", -1)), int(getattr(st, "id", -1))))
            if load_skew_mode:
                chosen = max(
                    rows,
                    key=lambda row: (
                        self._y_subtask_priority_score(row, station_counts=load_now, load_skew_mode=True),
                        -int(getattr(row, "station_sequence_rank", -1)),
                        -int(getattr(row, "id", -1)),
                    ),
                )
                current_pos = rows.index(chosen)
                if current_pos <= 0 and len(rows) >= 2:
                    chosen = rows[1]
                    current_pos = 1
                rows.remove(chosen)
                anchor_pos = int(anchor_rank_map.get(int(getattr(chosen, "id", -1)), 0))
                insert_pos = max(0, min(len(rows), min(current_pos - 1, max(0, anchor_pos))))
                if insert_pos == current_pos:
                    insert_pos = 0
            else:
                chosen = rng.choice(rows)
                rows.remove(chosen)
                insert_pos = rng.randrange(len(rows) + 1)
            rows.insert(insert_pos, chosen)
            for rank, st in enumerate(rows):
                st.station_sequence_rank = int(rank)
            changed = True
        elif operator == "cross_station_reinsert":
            st = max(
                target_rows or subtasks,
                key=lambda row: self._y_subtask_priority_score(row, station_counts=load_now, load_skew_mode=load_skew_mode),
            ) if load_skew_mode else rng.choice(target_rows or subtasks)
            curr_sid = int(getattr(st, "assigned_station_id", -1))
            alt = [sid for sid in station_ids if sid != curr_sid]
            if not alt:
                return False
            order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
            new_sid = min(
                alt,
                key=lambda sid: self._y_station_target_score(
                    order_id,
                    sid,
                    context,
                    load_now,
                    prefer_light=True if load_skew_mode else False,
                ),
            ) if load_skew_mode else rng.choice(alt)
            st.assigned_station_id = int(new_sid)
            st.station_sequence_rank = int(load_now.get(new_sid, len(station_groups.get(new_sid, []))))
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
                    key=lambda sid: self._y_station_target_score(
                        oid,
                        sid,
                        context,
                        load_now,
                        prefer_light=load_skew_mode,
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
                key=lambda sid: self._y_station_target_score(
                    oid,
                    sid,
                    context,
                    load_now,
                    prefer_light=load_skew_mode,
                ),
            )
            base_rank = len([x for x in subtasks if int(getattr(x, "assigned_station_id", -1)) == target_sid])
            for offset, st in enumerate(chosen_rows):
                st.assigned_station_id = int(target_sid)
                st.station_sequence_rank = int(base_rank + offset)
            _ = self.sp2.solve_local_layer(subtasks, context, use_mip=False, time_limit_sec=float(self.cfg.sp2_time_limit_sec))
            changed = True
        elif operator == "congested_station_destroy_repair":
            if not load_skew_mode and search_mode != "diversify" and float(self.stagnation_stats.get("Y", {}).get("restart_triggered", 0.0)) <= 0.0:
                return False
            heavy_sid = max(station_groups.keys(), key=lambda sid: len(station_groups.get(sid, []))) if station_groups else -1
            rows = list(station_groups.get(heavy_sid, []))
            if not rows:
                return False
            destroy_n = max(1, min(len(rows), int(math.ceil(len(subtasks) * destroy_fraction))))
            removed = rows[-destroy_n:]
            for st in removed:
                oid = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
                target_sid = min(
                    station_ids,
                    key=lambda sid: self._y_station_target_score(
                        oid,
                        sid,
                        context,
                        load_now,
                        prefer_light=True,
                    ),
                )
                st.assigned_station_id = int(target_sid)
                st.station_sequence_rank = int(load_now.get(target_sid, 0))
                load_now[target_sid] = int(load_now.get(target_sid, 0)) + 1
            _ = self.sp2.solve_local_layer(subtasks, context, use_mip=False, time_limit_sec=float(self.cfg.sp2_time_limit_sec))
            changed = True
        elif operator == "order_cohesion_destroy_repair":
            if not load_skew_mode and search_mode != "diversify" and float(self.stagnation_stats.get("Y", {}).get("restart_triggered", 0.0)) <= 0.0:
                return False
            bad_orders = [oid for oid, rows in order_groups.items() if oid >= 0 and len({int(getattr(st, "assigned_station_id", -1)) for st in rows}) >= 2]
            if not bad_orders:
                return False
            oid = rng.choice(bad_orders)
            rows = list(order_groups[oid])
            target_sid = min(
                station_ids,
                key=lambda sid: self._y_station_target_score(
                    oid,
                    sid,
                    context,
                    load_now,
                    prefer_light=load_skew_mode,
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
        ordered_ops = (
            [
                "range_shrink_expand",
                "tote_replace_within_stack",
                "stack_replace",
            ]
            if self._z_positive_mining_enabled() or bool(getattr(self.cfg, "z_micro_safe_ops_only", False))
            else [
                "range_shrink_expand",
                "tote_replace_within_stack",
                "stack_replace",
                "mode_flip_sort_toggle",
                "task_merge_split",
            ]
        )
        if operator != "z_hotspot_destroy_repair":
            restore_z = float(self.anchor_z if math.isfinite(self.anchor_z) else self.work_z)
            base_snapshot = self.snapshot(restore_z, iter_id=int(getattr(self, "current_iter", 0)), lightweight=True)
            preferred_ids = [int(x) for x in (priority_subtask_ids or []) if int(x) >= 0]
            all_subtask_ids = [
                int(getattr(st, "id", -1))
                for st in (getattr(self.problem, "subtask_list", []) or [])
                if int(getattr(st, "id", -1)) >= 0 and getattr(st, "execution_tasks", None)
            ]
            candidate_batches: List[Optional[List[int]]] = [[sid] for sid in preferred_ids]
            for sid in all_subtask_ids:
                if sid in preferred_ids:
                    continue
                candidate_batches.append([sid])
            candidate_batches.append(None)
            for batch_ids in candidate_batches:
                self.restore_snapshot(base_snapshot)
                self._normalize_station_assignments()
                candidate = self._build_z_candidate_from_subtasks(
                    rng,
                    priority_subtask_ids=batch_ids,
                    forced_move_type=str(operator),
                )
                if bool(candidate.get("feasible", False)):
                    return candidate
            return {
                "feasible": False,
                "move_type": str(operator),
                "changed_subtask_count": 0,
                "task_delta": 0,
                "stack_delta": 0,
                "mode_delta": 0,
            }
        subtasks = [st for st in getattr(self.problem, "subtask_list", []) or [] if getattr(st, "execution_tasks", None)]
        if not subtasks:
            return {"feasible": False, "move_type": operator, "changed_subtask_count": 0, "task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        batch_size = max(1, int(getattr(self.cfg, "z_hotspot_batch_size", 3)))
        if self._is_micro_scale_case():
            batch_size = 1
        destroy_fraction = max(0.05, float(getattr(self.cfg, "z_destroy_fraction", 0.30)))
        selected_ids = self._select_priority_z_subtasks(limit=max(batch_size, int(getattr(self.cfg, "z_hotspot_topk", max(batch_size, strength)))))
        selected_ids = list(selected_ids[:batch_size])
        changed = 0
        agg = {"task_delta": 0, "stack_delta": 0, "mode_delta": 0}
        hotspot_score = 0.0
        banned = False
        metric_keys = [
            "z_route_insertion_detour",
            "z_hit_frequency_bonus",
            "z_stack_locality_score",
            "z_demand_ratio",
            "z_congestion_proxy",
        ]
        metric_totals = {key: 0.0 for key in metric_keys}
        metric_count = 0
        fallback_used = False
        fallback_type = ""
        main_sid = int(selected_ids[0]) if selected_ids else -1
        for idx, sid in enumerate(selected_ids):
            sid = int(sid)
            if self._is_z_operator_banned(operator, sid):
                banned = True
                continue
            restore_z = float(self.anchor_z if math.isfinite(self.anchor_z) else self.work_z)
            base_snapshot = self.snapshot(restore_z, iter_id=int(getattr(self, "current_iter", 0)), lightweight=True)
            candidate = {"feasible": False}
            move_types = list(ordered_ops) if idx == 0 and sid == main_sid else ["range_shrink_expand"]
            for move_type in move_types:
                self.restore_snapshot(base_snapshot)
                self._normalize_station_assignments()
                st = next((item for item in subtasks if int(getattr(item, "id", -1)) == sid), None)
                if st is not None and idx == 0:
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
                candidate = self._build_z_candidate_from_subtasks(rng, priority_subtask_ids=[sid], forced_move_type=move_type)
                if bool(candidate.get("feasible", False)):
                    break
            if not bool(candidate.get("feasible", False)):
                continue
            hotspot_score = max(hotspot_score, float(candidate.get("z_hotspot_score", 0.0)))
            changed += int(candidate.get("changed_subtask_count", 0))
            for key in agg:
                agg[key] += int(candidate.get(key, 0))
            for key in metric_keys:
                metric_totals[key] += float(candidate.get(key, 0.0) or 0.0)
            metric_count += 1
            fallback_used = bool(fallback_used or candidate.get("z_operator_fallback_used", False))
            if not fallback_type and str(candidate.get("z_fallback_type", "")):
                fallback_type = str(candidate.get("z_fallback_type", ""))
        if changed <= 0:
            return {"feasible": False, "move_type": operator, "changed_subtask_count": 0, "z_operator_banned": bool(banned), **agg}
        result = {
            "feasible": True,
            "move_type": operator,
            "move_variant": str(operator),
            "changed_subtask_count": changed,
            "z_hotspot_score": float(hotspot_score),
            "z_operator_banned": bool(banned),
            "z_operator_fallback_used": bool(fallback_used),
            "z_fallback_type": str(fallback_type),
            **agg,
        }
        for key in metric_keys:
            result[key] = float(metric_totals[key] / max(1, metric_count))
        result["z_route_gap_penalty"] = float(result.get("z_route_insertion_detour", 0.0))
        return result

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

        self.cfg.enable_role_vns = False
        scheme = self._current_search_scheme()
        if scheme == "layer_augmented":
            raise NotImplementedError("layer_augmented has been retired; use resource_time_alns")

        engine = ResourceTimeALNSEngine(self)
        z_final = float(engine.run())
        self.run_total_time_sec = float(self._runtime_elapsed_sec())
        if self.best is not None:
            self.export_best()
        return float(z_final)


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
        scheme = self._current_search_scheme()
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

        if scheme == "resource_time_alns":
            write_resource_time_iters_csv(log_dir, self.iter_log)
            if bool(getattr(self.cfg, "resource_candidate_pool_log", True)):
                write_resource_time_candidates_csv(log_dir, getattr(self, "candidate_iter_log", []))
            write_resource_time_best_runtime_txt(log_dir, self, run_stats)
        else:
            supervised_candidates_path = self._log_path("xz_supervised_candidates.json")
            with open(supervised_candidates_path, "w", encoding="utf-8") as f:
                json.dump({
                    "config": {
                        "scale": str(getattr(self.cfg, "scale", "")).upper(),
                        "seed": int(getattr(self.cfg, "seed", -1)),
                        "search_scheme": str(getattr(self.cfg, "search_scheme", "")),
                    },
                    "x_row_count": int(len(self.supervised_candidate_dataset.get("X", []) or [])),
                    "z_row_count": int(len(self.supervised_candidate_dataset.get("Z", []) or [])),
                    "x_rows": list(self.supervised_candidate_dataset.get("X", []) or []),
                    "z_rows": list(self.supervised_candidate_dataset.get("Z", []) or []),
                }, f, ensure_ascii=False, indent=2)

        txt_path = self._log_path("tra_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== Resource-Time ALNS Summary ===\n" if scheme == "resource_time_alns" else "=== TRA Rotating Outer Loop Summary ===\n")
            f.write(f"scale={self.cfg.scale}, seed={self.cfg.seed}\n")
            f.write(f"total_runtime_sec={run_stats['run_total_time_sec']:.6f}\n")
            f.write(f"best_z={self.best.z:.3f}s @ iter={self.best.iter_id}\n\n")
            if scheme == "resource_time_alns":
                f.write(f"stop_reason={str(run_stats.get('stop_reason', '') or '')}\n\n")
                f.write(f"exact_eval_cache_hit_count={int(run_stats.get('exact_eval_cache_hit_count', 0))}\n")
                f.write(f"empty_candidate_penalized_count={int(run_stats.get('empty_candidate_penalized_count', 0))}\n\n")
                f.write(f"coverage_hard_reject_count={int(run_stats.get('coverage_hard_reject_count', 0))}\n")
                f.write(f"x_failure_decapitation_count={int(run_stats.get('x_failure_decapitation_count', 0))}\n\n")
            if scheme == "resource_time_alns":
                f.write(
                    f"{'iter':>4} | {'focus':>4} | {'z':>10} | {'best':>10} | "
                    f"{'F_raw':>12} | {'F_cal':>12} | {'tier':>6} | {'gval':>4} | {'emp':>3} | {'cd':>2} | {'cache':>5} | {'imp':>3} | {'skip':>4} | {'lb':>10}\n"
                )
                f.write("-" * 148 + "\n")
            else:
                f.write(f"{'iter':>4} | {'focus':>4} | {'z':>10} | {'best':>10} | {'imp':>3} | {'skip':>4} | {'lb':>10}\n")
                f.write("-" * 72 + "\n")
            for row in self.iter_log:
                z = row["z"]
                z_str = "SKIP" if (isinstance(z, float) and math.isnan(z)) else f"{z:10.3f}"
                lb = row["lb"]
                lb_str = "   -   " if lb is None else f"{lb:10.3f}"
                if scheme == "resource_time_alns":
                    f_raw = row.get("F_raw", None)
                    f_cal = row.get("F_cal", None)
                    destroy_tier = str(row.get("destroy_tier", "") or "")
                    global_eval_flag = "Y" if bool(row.get("global_eval_triggered", False)) else "N"
                    empty_flag = "Y" if bool(row.get("empty_candidate_penalized", False)) else "N"
                    cooldown_remaining = int(row.get("layer_cooldown_remaining", 0) or 0)
                    cache_flag = "Y" if bool(row.get("used_exact_eval_cache", False)) else "N"
                    f_raw_str = "     -     " if not isinstance(f_raw, (int, float)) or math.isnan(float(f_raw)) else f"{float(f_raw):12.3f}"
                    f_cal_str = "     -     " if not isinstance(f_cal, (int, float)) or math.isnan(float(f_cal)) else f"{float(f_cal):12.3f}"
                    f.write(
                        f"{row['iter']:4d} | {row['focus']:>4} | {z_str} | {row['best_z']:10.3f} | "
                        f"{f_raw_str} | {f_cal_str} | {destroy_tier:>6} | {global_eval_flag:>4} | "
                        f"{empty_flag:>3} | {cooldown_remaining:>2d} | {cache_flag:>5} | "
                        f"{'Y' if row['improved'] else 'N':>3} | {('Y' if row['skipped'] else 'N'):>4} | {lb_str}\n"
                    )
                else:
                    f.write(
                        f"{row['iter']:4d} | {row['focus']:>4} | {z_str} | {row['best_z']:10.3f} | "
                        f"{'Y' if row['improved'] else 'N':>3} | {('Y' if row['skipped'] else 'N'):>4} | {lb_str}\n"
                    )

        print(f"  >>> [TRA] Logs written to {log_dir}")

        if self.cfg.enable_sp1_feedback_analysis:
            self._write_sp1_feedback_suggestions()

    def _write_sp1_feedback_suggestions(self):
        """
        生成“更外层（SP1）”的软耦合反馈建议文件。
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
                # 简单策略：每触发一次建议就把 cap-1（下限 1）
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
        out_dir = self._log_path("best_solution_export") if self._current_search_scheme() == "resource_time_alns" else self._log_path("tra_best_export")
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

        calc = self.sim if self.sim is not None else RankAwareGlobalTimeCalculator(self.problem)
        z = float(calc.calculate())
        calc.calculate_and_export(out_dir)
        verification_result = self._verify_makespan_breakdown(out_dir)
        self._write_best_solution_summary(out_dir, z)
        self._write_best_solution_dump(out_dir, z)
        self._write_best_solution_audit(out_dir, z, verification_result=verification_result)

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

            f.write("\n[SP4 Trips By Robot]\n")
            trip_rows: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
            for t in all_tasks:
                robot_id = int(getattr(t, "robot_id", -1))
                trip_id = int(getattr(t, "trip_id", 0))
                if robot_id < 0:
                    continue
                trip_rows[(robot_id, trip_id)].append(t)
            for (robot_id, trip_id), rows in sorted(trip_rows.items(), key=lambda item: (item[0][0], item[0][1])):
                ordered = sorted(
                    rows,
                    key=lambda t: (
                        float(getattr(t, "arrival_time_at_stack", 0.0)),
                        float(getattr(t, "arrival_time_at_station", 0.0)),
                        int(getattr(t, "task_id", -1)),
                    ),
                )
                f.write(
                    f"robot_id={robot_id}, trip_id={trip_id}, "
                    f"task_ids={[int(getattr(t, 'task_id', -1)) for t in ordered]}, "
                    f"station_ids={[int(getattr(t, 'target_station_id', -1)) for t in ordered]}, "
                    f"stack_ids={[int(getattr(t, 'target_stack_id', -1)) for t in ordered]}\n"
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

    def _build_best_solution_audit(self, z: float, verification_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        coverage = self._compute_solution_coverage()
        best_z = float(self.best.z) if self.best is not None else float("nan")
        recomputed_z = float(z)
        global_makespan = float(getattr(self.problem, "global_makespan", 0.0))
        makespan_consistent = bool(
            math.isfinite(best_z)
            and abs(best_z - recomputed_z) <= 1e-6
            and abs(recomputed_z - global_makespan) <= 1e-6
        )

        invalid_station_rows: List[Dict[str, Any]] = []
        invalid_rank_rows: List[Dict[str, Any]] = []
        invalid_z_rows: List[Dict[str, Any]] = []
        unassigned_robot_rows: List[Dict[str, Any]] = []
        station_rank_rows: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        tote_to_task_rows: Dict[int, List[int]] = defaultdict(list)
        all_tasks: List[Any] = []

        for st in getattr(self.problem, "subtask_list", []) or []:
            subtask_id = int(getattr(st, "id", -1))
            station_id = int(getattr(st, "assigned_station_id", -1))
            rank = int(getattr(st, "station_sequence_rank", -1))
            if station_id < 0:
                invalid_station_rows.append({"entity": "subtask", "subtask_id": subtask_id, "station_id": station_id})
            if rank < 0:
                invalid_rank_rows.append({"entity": "subtask", "subtask_id": subtask_id, "station_id": station_id, "rank": rank, "reason": "negative_rank"})
            if station_id >= 0 and rank >= 0:
                station_rank_rows[(station_id, rank)].append(subtask_id)
            all_tasks.extend(getattr(st, "execution_tasks", []) or [])

        for task in all_tasks:
            task_id = int(getattr(task, "task_id", -1))
            station_id = int(getattr(task, "target_station_id", -1))
            rank = int(getattr(task, "station_sequence_rank", -1))
            if station_id < 0:
                invalid_station_rows.append({"entity": "task", "task_id": task_id, "station_id": station_id})
            if rank < 0:
                invalid_rank_rows.append({"entity": "task", "task_id": task_id, "station_id": station_id, "rank": rank, "reason": "negative_rank"})

            target_ids = [int(x) for x in (getattr(task, "target_tote_ids", []) or []) if int(x) >= 0]
            hit_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0]
            noise_ids = [int(x) for x in (getattr(task, "noise_tote_ids", []) or []) if int(x) >= 0]
            target_set = set(target_ids)
            hit_set = set(hit_ids)
            noise_set = set(noise_ids)
            if not target_ids:
                invalid_z_rows.append({"task_id": task_id, "reason": "empty_target_totes"})
            if not hit_set.issubset(target_set):
                invalid_z_rows.append({"task_id": task_id, "reason": "hit_not_subset_of_target", "hit_tote_ids": hit_ids, "target_tote_ids": target_ids})
            if noise_set.intersection(hit_set):
                invalid_z_rows.append({"task_id": task_id, "reason": "noise_hit_overlap", "hit_tote_ids": hit_ids, "noise_tote_ids": noise_ids})
            if not noise_set.issubset(target_set):
                invalid_z_rows.append({"task_id": task_id, "reason": "noise_not_subset_of_target", "noise_tote_ids": noise_ids, "target_tote_ids": target_ids})
            for tote_id in target_ids:
                tote_to_task_rows[int(tote_id)].append(task_id)

            if int(getattr(task, "robot_id", -1)) < 0:
                unassigned_robot_rows.append({"task_id": task_id, "subtask_id": int(getattr(task, "sub_task_id", -1))})

        duplicate_rank_rows = [
            {"station_id": int(station_id), "rank": int(rank), "subtask_ids": list(ids)}
            for (station_id, rank), ids in station_rank_rows.items()
            if len(ids) > 1
        ]
        duplicate_tote_rows = {
            int(tote_id): list(task_ids)
            for tote_id, task_ids in tote_to_task_rows.items()
            if len(task_ids) > 1
        }

        verification_failures = list((verification_result or {}).get("failures", []) or [])
        has_unreasonable_solution = bool(
            not bool(coverage.get("coverage_ok", False))
            or not makespan_consistent
            or invalid_station_rows
            or invalid_rank_rows
            or duplicate_rank_rows
            or invalid_z_rows
            or duplicate_tote_rows
            or unassigned_robot_rows
            or verification_failures
        )
        issue_summary: List[str] = []
        if int(coverage.get("unmet_sku_total", 0)) > 0:
            issue_summary.append(f"SKU coverage unmet: {int(coverage.get('unmet_sku_total', 0))} units")
        if not makespan_consistent:
            issue_summary.append(
                f"Makespan inconsistent: best_z={best_z:.6f}, recomputed_z={recomputed_z:.6f}, global_makespan={global_makespan:.6f}"
            )
        if invalid_station_rows:
            issue_summary.append(f"Invalid station assignments: {len(invalid_station_rows)}")
        if invalid_rank_rows or duplicate_rank_rows:
            issue_summary.append(f"Invalid station ranks: {len(invalid_rank_rows) + len(duplicate_rank_rows)}")
        if invalid_z_rows:
            issue_summary.append(f"Invalid Z task rows: {len(invalid_z_rows)}")
        if duplicate_tote_rows:
            issue_summary.append(f"Duplicate tote use count: {sum(max(0, len(rows) - 1) for rows in duplicate_tote_rows.values())}")
        if unassigned_robot_rows:
            issue_summary.append(f"Unassigned robot task count: {len(unassigned_robot_rows)}")
        issue_summary.extend(str(item) for item in verification_failures)

        return {
            "coverage_ok": bool(coverage.get("coverage_ok", False)),
            "missing_sku_hit": bool(int(coverage.get("unmet_sku_total", 0)) > 0),
            "unmet_sku_total": int(coverage.get("unmet_sku_total", 0)),
            "unmet_subtask_count": int(coverage.get("unmet_subtask_count", 0)),
            "coverage_subtasks": list(coverage.get("subtasks", []) or []),
            "best_z": best_z,
            "recomputed_z": recomputed_z,
            "global_makespan": global_makespan,
            "makespan_consistent": makespan_consistent,
            "invalid_station_assignment_count": int(len(invalid_station_rows)),
            "invalid_station_assignments": invalid_station_rows,
            "invalid_rank_count": int(len(invalid_rank_rows) + len(duplicate_rank_rows)),
            "invalid_rank_rows": invalid_rank_rows,
            "duplicate_station_ranks": duplicate_rank_rows,
            "invalid_z_task_count": int(len(invalid_z_rows)),
            "invalid_z_tasks": invalid_z_rows,
            "duplicate_tote_use_count": int(sum(max(0, len(rows) - 1) for rows in duplicate_tote_rows.values())),
            "duplicate_tote_uses": duplicate_tote_rows,
            "empty_robot_trip_count": 0,
            "unassigned_robot_task_count": int(len(unassigned_robot_rows)),
            "unassigned_robot_tasks": unassigned_robot_rows,
            "verification_failures": verification_failures,
            "has_unreasonable_solution": has_unreasonable_solution,
            "issues": issue_summary,
        }

    def _write_best_solution_audit(self, out_dir: str, z: float, verification_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        audit = self._build_best_solution_audit(z, verification_result=verification_result)
        json_path = os.path.join(out_dir, "best_solution_audit.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)

        txt_path = os.path.join(out_dir, "best_solution_audit.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("[Best Solution Audit]\n")
            f.write(f"coverage_ok={bool(audit.get('coverage_ok', False))}\n")
            f.write(f"missing_sku_hit={bool(audit.get('missing_sku_hit', False))}\n")
            f.write(f"unmet_sku_total={int(audit.get('unmet_sku_total', 0))}\n")
            f.write(f"unmet_subtask_count={int(audit.get('unmet_subtask_count', 0))}\n")
            f.write(f"makespan_consistent={bool(audit.get('makespan_consistent', False))}\n")
            f.write(f"invalid_station_assignment_count={int(audit.get('invalid_station_assignment_count', 0))}\n")
            f.write(f"invalid_rank_count={int(audit.get('invalid_rank_count', 0))}\n")
            f.write(f"invalid_z_task_count={int(audit.get('invalid_z_task_count', 0))}\n")
            f.write(f"duplicate_tote_use_count={int(audit.get('duplicate_tote_use_count', 0))}\n")
            f.write(f"unassigned_robot_task_count={int(audit.get('unassigned_robot_task_count', 0))}\n")
            f.write(f"has_unreasonable_solution={bool(audit.get('has_unreasonable_solution', False))}\n")
            f.write(
                "sku_hit_check="
                + ("PASS" if not bool(audit.get("missing_sku_hit", False)) else "FAIL")
                + "\n"
            )
            f.write(
                "unreasonable_solution_check="
                + ("PASS" if not bool(audit.get("has_unreasonable_solution", False)) else "FAIL")
                + "\n"
            )
            if audit.get("issues"):
                f.write("issues:\n")
                for item in list(audit.get("issues", []) or []):
                    f.write(f"- {item}\n")
        return audit

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
        return result


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



