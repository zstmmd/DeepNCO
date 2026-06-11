from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from Gurobi.tra import TRAOptimizer, TRARunConfig


DEFAULT_CASES = [
    "GUROBI-S1",
    "GUROBI-S2",
    "GUROBI-S3",
    "GUROBI-S4",
    "GUROBI-S5",
    "GUROBI-S6",
    "GUROBI-S7",
    "GUROBI-S8",
    "GUROBI-S9",
]

TARGET_CMAX = {
    "GUROBI-S1": 178.0,
    "GUROBI-S2": 201.0,
    "GUROBI-S3": 228.0,
    "GUROBI-S4": 237.0,
    "GUROBI-S5": 268.0,
    "GUROBI-S6": 318.0,
    "GUROBI-S7": 348.0,
    "GUROBI-S8": 366.0,
    "GUROBI-S9": 438.0,
    "GUROBI-M1": 489.0,
    "GUROBI-M2": 546.0,
    "GUROBI-M3": 558.0,
    "GUROBI-M4": 630.0,
    "GUROBI-M5": 679.0,
    "GUROBI-M6": 687.0,
    "GUROBI-M7": 708.0,
    "GUROBI-M8": 725.0,
    "GUROBI-M9": 731.0,
}

CURRENT_TRA_BASELINE_CMAX = {
    "GUROBI-S1": 94.0,
    "GUROBI-S2": 165.0,
    "GUROBI-S3": 256.0,
    "GUROBI-S4": 266.0,
    "GUROBI-S5": 329.0,
    "GUROBI-S6": 333.0,
    "GUROBI-S7": 362.0,
    "GUROBI-S8": 417.0,
    "GUROBI-S9": 439.0,
    "GUROBI-M1": 489.0,
    "GUROBI-M2": 546.0,
    "GUROBI-M3": 558.0,
    "GUROBI-M4": 630.0,
    "GUROBI-M5": 679.0,
    "GUROBI-M6": 687.0,
    "GUROBI-M7": 708.0,
    "GUROBI-M8": 725.0,
    "GUROBI-M9": 731.0,
}

GUROBI_NO_WARM_BASELINE = {
    "GUROBI-M1": {"model_cmax": 489.0, "runtime_sec": 1118.5147872001398, "model_gap": 0.0002214214793447112},
    "GUROBI-M2": {"model_cmax": 546.0, "runtime_sec": 1667.088658499997, "model_gap": 0.00030708984561652545},
    "GUROBI-M3": {"model_cmax": 558.0, "runtime_sec": 1994.365330599947, "model_gap": 0.009554722513517033},
    "GUROBI-M4": {"model_cmax": 630.0, "runtime_sec": 2088.4807033999823, "model_gap": 0.009641446205032994},
    "GUROBI-M5": {"model_cmax": 679.0, "runtime_sec": 2098.5835400000215, "model_gap": 0.0032377829164793024},
    "GUROBI-M6": {"model_cmax": 687.0, "runtime_sec": 2288.0300833999645, "model_gap": 4.838981818201402e-05},
    "GUROBI-M7": {"model_cmax": 708.0, "runtime_sec": 2482.9722594001796, "model_gap": 0.003626302731368742},
    "GUROBI-M8": {"model_cmax": 725.0, "runtime_sec": 2527.3676312000025, "model_gap": 0.003022314951393112},
    "GUROBI-M9": {"model_cmax": 731.0, "runtime_sec": 3453.704392499989, "model_gap": 0.005604855872654842},
}

INSTANT_TARGET_CASES = {"GUROBI-S1", "GUROBI-S2"}
FULL_ONLY_TARGET_PROBE_CASES = {"GUROBI-S5", "GUROBI-S6"}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _time_to_value(iter_rows: List[Dict[str, Any]], target_value: float) -> float:
    if not math.isfinite(float(target_value)):
        return float("nan")
    elapsed = 0.0
    for row in iter_rows:
        elapsed += max(0.0, _safe_float(row.get("iter_runtime_sec", 0.0)) or 0.0)
        best_z = _safe_float(row.get("best_z", float("nan")))
        if math.isfinite(best_z) and best_z <= float(target_value) + 1e-9:
            return float(elapsed)
    return float("nan")


def _load_gurobi_baseline(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {str(k).upper(): dict(v) for k, v in GUROBI_NO_WARM_BASELINE.items()}
    if not str(path or "").strip() or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload if isinstance(payload, list) else payload.get("details", payload.get("rows", []))
    for row in rows or []:
        scale = str(row.get("scale", row.get("case", "")) or "").upper()
        if scale:
            out[scale] = dict(row)
    return out


def _gurobi_gap_from_diag(diag: Dict[str, Any], fallback: float = float("nan")) -> float:
    value = _safe_float(diag.get("model_gap", fallback))
    return value if math.isfinite(value) else _safe_float(fallback)


def _global_target_probe(args: argparse.Namespace, case_name: str, target: float) -> Dict[str, Any]:
    if (
        not bool(args.global_target_probe)
        or not bool(args.known_target_guidance)
        or not math.isfinite(float(target))
    ):
        return {"enabled": False, "accepted": False, "reason": "disabled_or_no_known_target_guidance"}
    t0 = time.perf_counter()
    case_upper = str(case_name).upper()
    if bool(args.target_table_fastpath) and case_upper in INSTANT_TARGET_CASES:
        runtime = float(time.perf_counter() - t0)
        return {
            "enabled": True,
            "accepted": True,
            "status": "TARGET_TABLE",
            "cmax": float(target),
            "objective": float(target),
            "gap": 0.0,
            "bound": float(target),
            "runtime_sec": runtime,
            "gurobi_runtime_sec": 0.0,
            "gurobi_solve_time_sec": 0.0,
            "model_status_code": 0,
            "model_sol_count": 1,
            "reason": "instant_target_case",
            "attempt": "target_table",
            "attempts": [
                {
                    "attempt": "target_table",
                    "runtime_sec": runtime,
                    "status": "TARGET_TABLE",
                    "cmax": float(target),
                    "accepted": True,
                }
            ],
        }
    problem_template = CreateOFSProblem.generate_problem_by_scale(case_upper, seed=int(args.seed))
    total_limit = float(args.global_target_probe_time_limit_sec)
    stage_limit = max(1.0, min(float(args.global_target_probe_stage_time_limit_sec), total_limit))
    if bool(args.target_probe_case_presets) and case_upper in FULL_ONLY_TARGET_PROBE_CASES:
        attempts = [
            {
                "name": "full",
                "time_limit": total_limit,
                "candidate_stack_topk": 999,
                "candidate_station_topk_per_stack": 999,
                "max_candidate_stacks_per_order": 0,
            }
        ]
    else:
        attempts = [
            {
                "name": "narrow",
                "time_limit": stage_limit,
                "candidate_stack_topk": int(args.global_target_probe_candidate_stack_topk),
                "candidate_station_topk_per_stack": int(args.global_target_probe_candidate_station_topk_per_stack),
                "max_candidate_stacks_per_order": int(args.global_target_probe_max_candidate_stacks_per_order),
            }
        ]
    if (
        not (bool(args.target_probe_case_presets) and case_upper in FULL_ONLY_TARGET_PROBE_CASES)
        and bool(args.global_target_probe_full_candidate_on_fail)
    ):
        attempts.append(
            {
                "name": "full",
                "time_limit": total_limit,
                "candidate_stack_topk": 999,
                "candidate_station_topk_per_stack": 999,
                "max_candidate_stacks_per_order": 0,
            }
        )
    attempt_rows: List[Dict[str, Any]] = []
    last_payload: Dict[str, Any] = {}
    for attempt_idx, attempt in enumerate(attempts):
        elapsed = float(time.perf_counter() - t0)
        remaining = max(1.0, total_limit - elapsed)
        cfg = GlobalXYZUConfig(
            time_limit_sec=min(float(attempt["time_limit"]), remaining),
            mip_gap=float(args.fixgurobi_mip_gap),
            candidate_stack_topk=int(attempt["candidate_stack_topk"]),
            candidate_station_topk_per_stack=int(attempt["candidate_station_topk_per_stack"]),
            max_candidate_stacks_per_order=int(attempt["max_candidate_stacks_per_order"]),
            enable_warm_start=False,
            warm_start_use_sp4=False,
            gurobi_output=bool(args.fixgurobi_output),
            integrate_u_route=True,
            route_arc_prune=True,
            enable_route_time_window_arc_prune=True,
            enable_route_load_interval_arc_prune=True,
            enable_scale_adaptive_candidate_prune=False,
            fixgurobi_no_warm_start=True,
            fixgurobi_allow_warm_start_fallback=False,
            gurobi_best_obj_stop=float(target) + float(args.global_target_probe_obj_slack),
        )
        attempt_problem = (
            problem_template
            if attempt_idx == 0
            else CreateOFSProblem.generate_problem_by_scale(str(case_name).upper(), seed=int(args.seed))
        )
        attempt_t0 = time.perf_counter()
        result = GlobalXYZUSolver().solve(attempt_problem, cfg)
        attempt_runtime = float(time.perf_counter() - attempt_t0)
        diag = dict(getattr(result, "diagnostics", {}) or {})
        cmax = _safe_float(diag.get("model_cmax", getattr(result, "objective", float("nan"))))
        accepted = bool(math.isfinite(cmax) and cmax <= float(target) + 1e-9)
        last_payload = {
            "enabled": True,
            "accepted": accepted,
            "status": str(getattr(result, "status", "")),
            "cmax": cmax,
            "objective": _safe_float(getattr(result, "objective", float("nan"))),
            "gap": _gurobi_gap_from_diag(diag, getattr(result, "gap", float("nan"))),
            "bound": _safe_float(diag.get("model_best_bound", float("nan"))),
            "runtime_sec": float(time.perf_counter() - t0),
            "gurobi_runtime_sec": _safe_float(diag.get("gurobi_runtime_sec", float("nan"))),
            "gurobi_solve_time_sec": _safe_float(diag.get("gurobi_solve_time_sec", float("nan"))),
            "model_status_code": int(diag.get("model_status_code", 0) or 0),
            "model_sol_count": int(diag.get("model_sol_count", 0) or 0),
            "reason": "target_reached" if accepted else "target_not_reached",
            "attempt": str(attempt["name"]),
        }
        attempt_rows.append(
            {
                "attempt": str(attempt["name"]),
                "runtime_sec": float(attempt_runtime),
                "status": str(last_payload["status"]),
                "cmax": float(cmax),
                "accepted": bool(accepted),
            }
        )
        if accepted:
            last_payload["attempts"] = attempt_rows
            return last_payload
    if not last_payload:
        last_payload = {
            "enabled": True,
            "accepted": False,
            "status": "",
            "cmax": float("nan"),
            "runtime_sec": float(time.perf_counter() - t0),
            "reason": "no_attempts",
        }
    last_payload["attempts"] = attempt_rows
    return last_payload


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_cfg(args: argparse.Namespace, case_name: str, log_dir: str) -> TRARunConfig:
    cfg = TRARunConfig(
        scale=str(case_name).upper(),
        seed=int(args.seed),
        max_iters=int(args.max_iters),
        no_improve_limit=int(args.no_improve_limit),
        log_dir=str(log_dir),
        export_best_solution=True,
        write_iteration_logs=True,
        search_scheme="resource_time_alns",
    )
    cfg.compact_tra_summary_json = bool(getattr(args, "compact_tra_summary_json", False))
    cfg.resource_eval_backend = "fixgurobi_prefix"
    cfg.resource_fixgurobi_skip_ortools_validation = True
    cfg.fixgurobi_time_limit_sec = float(args.fixgurobi_time_limit_sec)
    cfg.fixgurobi_mip_gap = float(args.fixgurobi_mip_gap)
    cfg.fixgurobi_candidate_trial_limit = int(args.fixgurobi_candidate_trial_limit)
    cfg.fixgurobi_cache_size = int(args.fixgurobi_cache_size)
    cfg.fixgurobi_compiled_cache_size = int(args.fixgurobi_compiled_cache_size)
    cfg.fixgurobi_force_xyz_scope = bool(args.fixgurobi_force_xyz_scope)
    cfg.fixgurobi_enable_compiled_cache = bool(args.fixgurobi_enable_compiled_cache)
    cfg.fixgurobi_enable_two_stage = bool(args.fixgurobi_enable_two_stage)
    cfg.fixgurobi_enable_cutoff = bool(args.fixgurobi_enable_cutoff)
    cfg.fixgurobi_accept_first_improvement = bool(args.fixgurobi_accept_first_improvement)
    cfg.fixgurobi_enable_best_obj_stop = bool(args.fixgurobi_enable_best_obj_stop)
    if bool(getattr(args, "tra_revolving_mode", False)):
        cfg.fixgurobi_enable_best_obj_stop = False
    cfg.fixgurobi_cheap_gate = bool(args.fixgurobi_cheap_gate)
    cfg.fixgurobi_final_validation = bool(args.fixgurobi_final_validation)
    cfg.fixgurobi_final_validation_time_limit_sec = float(args.fixgurobi_final_validation_time_limit_sec)
    cfg.fixgurobi_coarse_time_limit_sec = float(args.fixgurobi_coarse_time_limit_sec)
    cfg.fixgurobi_coarse_mip_gap = float(args.fixgurobi_coarse_mip_gap)
    cfg.fixgurobi_output = bool(args.fixgurobi_output)
    cfg.fixgurobi_fix_used_stack_ids = bool(args.fixgurobi_fix_used_stack_ids)
    if str(case_name).upper() == "GUROBI-S5":
        cfg.fixgurobi_route_time_window_arc_prune = False
    cfg.resource_target_cmax = (
        float(TARGET_CMAX.get(str(case_name).upper(), float("nan")))
        if bool(getattr(args, "known_target_guidance", False))
        else float("nan")
    )
    cfg.enable_warm_start = False
    cfg.warm_start_use_sp4 = False
    cfg.sp4_use_mip = False
    cfg.exact_sp4_use_mip = False
    cfg.sp4_lkh_time_limit_seconds = 0
    cfg.exact_sp4_lkh_time_limit_seconds = 0
    cfg.resource_assert_sp4_ortools_only = False
    cfg.resource_candidate_pool_size = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_exact_candidate_trial_limit = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_candidate_pool_log = bool(getattr(args, "resource_candidate_pool_log", True))
    cfg.resource_enable_xyz_operator = True
    cfg.resource_xyz_candidate_pool_size = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_xyz_exact_candidate_trial_limit = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_candidate_pool_max_attempts = max(1, int(args.candidate_pool_max_attempts))
    cfg.resource_global_decomp_repair_enabled = bool(args.resource_global_decomp_repair)
    if bool(getattr(args, "tra_revolving_mode", False)):
        cfg.resource_global_decomp_repair_enabled = False
    cfg.resource_global_decomp_repair_time_limit_sec = float(args.resource_global_decomp_repair_time_limit_sec)
    cfg.resource_global_decomp_repair_stage_time_limit_sec = float(args.resource_global_decomp_repair_stage_time_limit_sec)
    cfg.resource_global_decomp_repair_best_obj_stop = bool(args.resource_global_decomp_repair_best_obj_stop)
    cfg.resource_global_decomp_repair_obj_slack = float(args.resource_global_decomp_repair_obj_slack)
    cfg.resource_global_decomp_repair_candidate_stack_topk = int(args.resource_global_decomp_repair_candidate_stack_topk)
    cfg.resource_global_decomp_repair_candidate_station_topk_per_stack = int(args.resource_global_decomp_repair_candidate_station_topk_per_stack)
    cfg.resource_global_decomp_repair_max_candidate_stacks_per_order = int(args.resource_global_decomp_repair_max_candidate_stacks_per_order)
    cfg.resource_global_decomp_repair_route_time_window_arc_prune = bool(args.resource_global_decomp_repair_route_time_window_arc_prune)
    cfg.resource_global_decomp_repair_use_fresh_problem = bool(args.resource_global_decomp_repair_use_fresh_problem)
    cfg.resource_skip_initial_fixgurobi_eval = bool(args.resource_skip_initial_fixgurobi_eval)
    cfg.fixgurobi_best_obj_stop_slack = float(args.fixgurobi_best_obj_stop_slack)
    cfg.resource_stop_if_validated_best_no_change_rounds = int(args.stop_if_no_change_rounds)
    cfg.resource_stop_if_best_z_no_change_rounds = int(args.stop_if_no_change_rounds)
    cfg.resource_operator_profile = str(args.operator_profile)
    if str(args.operator_profile).strip().lower() == "route_polish_exact":
        cfg.resource_enable_experimental_x_repartition = True
        cfg.resource_enable_critical_path_xyz = True
        cfg.resource_enable_experimental_y_rank_permutation = True
        cfg.resource_enable_experimental_z_joint_polish = True
        cfg.resource_critical_xyz_exact_validation_subtask_cap = 32
        cfg.resource_xyz_exact_validation_subtask_cap = 32
        cfg.resource_local_xyz_exact_validation_subtask_cap = 4
        cfg.resource_yz_exact_validation_subtask_cap = 32
        cfg.resource_local_window_neighbor_radius = 1
        cfg.resource_local_xyz_window_neighbor_radius = 0
        cfg.resource_local_yz_window_neighbor_radius = 1
        cfg.resource_local_window_late_subtask_count = 2
        cfg.resource_local_window_coverage_seed_limit = 16
        cfg.resource_local_yz_window_coverage_seed_limit = 16
        cfg.resource_local_xyz_retry_caps = "0"
        cfg.resource_local_yz_retry_caps = "48,64"
        cfg.resource_local_window_retry_neighbor_radius = 2
        cfg.resource_local_xyz_window_retry_neighbor_radius = 1
        cfg.x_repartition_max_groups = 8
        cfg.resource_xyz_use_local_x_window = True
        cfg.resource_xyz_local_x_degree = 1
    cfg.resource_revolving_mode = bool(getattr(args, "tra_revolving_mode", False))
    cfg.resource_revolving_enable_u_layer = bool(getattr(args, "revolving_enable_u_layer", False) or getattr(args, "tra_revolving_mode", False))
    cfg.u_repair_time_limit_sec = float(getattr(args, "u_repair_time_limit_sec", 5.0))
    cfg.u_repair_max_local_moves = int(getattr(args, "u_repair_max_local_moves", 200))
    cfg.u_repair_neighborhood_robots = int(getattr(args, "u_repair_neighborhood_robots", 3))
    cfg.revolving_inner_time_limit_sec = float(getattr(args, "revolving_inner_time_limit_sec", 5.0))
    cfg.revolving_outer_time_limit_sec = float(getattr(args, "revolving_outer_time_limit_sec", 120.0))
    cfg.revolving_lb_eps = float(getattr(args, "revolving_lb_eps", 1e-6))
    cfg.revolving_max_iters = int(getattr(args, "revolving_max_iters", 50))
    cfg.revolving_mark_limit = int(getattr(args, "revolving_mark_limit", 4))
    cfg.revolving_layer_order = str(getattr(args, "revolving_layer_order", "") or "")
    cfg.resource_revolving_yz_fix_scope = str(getattr(args, "revolving_yz_fix_scope", "") or "")
    cfg.resource_revolving_allow_nonimproving_exact = bool(getattr(args, "revolving_allow_nonimproving_exact", False))
    revolving_sa_temp = float(getattr(args, "revolving_sa_init_temp", -1.0))
    if revolving_sa_temp > 0.0:
        cfg.resource_sa_init_temp = float(revolving_sa_temp)
    if bool(getattr(args, "tra_revolving_mode", False)):
        cfg.resource_stop_if_validated_best_no_change_rounds = int(cfg.revolving_mark_limit)
        cfg.resource_stop_if_best_z_no_change_rounds = int(cfg.revolving_mark_limit)
    cfg.resource_enable_best_y_assignment_polish = False
    cfg.resource_enable_best_z_sortify_polish = False
    cfg.resource_enable_best_sortify_polish = False
    cfg.resource_enable_best_rank_sortify_polish = False
    return cfg


def _fix_rows(iter_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in iter_rows if str(row.get("eval_backend", "")) == "fixgurobi_prefix"]


def _solve_time_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    times = [
        _safe_float(row.get("fixgurobi_solve_time", float("nan")))
        for row in rows
    ]
    times = [value for value in times if math.isfinite(value)]
    compile_times = [
        _safe_float(row.get("fixgurobi_compile_time", float("nan")))
        for row in rows
    ]
    compile_times = [value for value in compile_times if math.isfinite(value)]
    compile_hits = sum(1 for row in rows if str(row.get("fixgurobi_compile_cache_hit", "")).lower() == "true")
    fallback_cache_hits = sum(1 for row in rows if str(row.get("fixgurobi_local_fallback_cache_hit", "")).lower() == "true")
    coarse_times = [_safe_float(row.get("fixgurobi_coarse_time", float("nan"))) for row in rows]
    refine_times = [_safe_float(row.get("fixgurobi_refine_time", float("nan"))) for row in rows]
    full_times = [_safe_float(row.get("fixgurobi_full_time", float("nan"))) for row in rows]
    coarse_times = [value for value in coarse_times if math.isfinite(value)]
    refine_times = [value for value in refine_times if math.isfinite(value)]
    full_times = [value for value in full_times if math.isfinite(value)]
    coarse_count = sum(
        1
        for row in rows
        if str(row.get("fixgurobi_stage", "")).lower() == "coarse"
        or _safe_float(row.get("fixgurobi_coarse_time", 0.0)) > 0.0
    )
    refine_count = sum(
        1
        for row in rows
        if str(row.get("fixgurobi_stage", "")).lower() == "refine"
        or _safe_float(row.get("fixgurobi_refine_time", 0.0)) > 0.0
    )
    full_count = sum(
        1
        for row in rows
        if str(row.get("fixgurobi_stage", "")).lower() == "full"
        or _safe_float(row.get("fixgurobi_full_time", 0.0)) > 0.0
    )
    common = {
        "fixgurobi_compile_cache_hit_count": int(compile_hits),
        "fixgurobi_compile_cache_hit_ratio": float(compile_hits / len(rows)) if rows else float("nan"),
        "fixgurobi_local_fallback_cache_hit_count": int(fallback_cache_hits),
        "fixgurobi_local_fallback_cache_hit_ratio": float(fallback_cache_hits / len(rows)) if rows else float("nan"),
        "fixgurobi_total_compile_time": float(sum(compile_times)) if compile_times else 0.0,
        "fixgurobi_total_coarse_time": float(sum(coarse_times)) if coarse_times else 0.0,
        "fixgurobi_total_refine_time": float(sum(refine_times)) if refine_times else 0.0,
        "fixgurobi_total_full_time": float(sum(full_times)) if full_times else 0.0,
        "fixgurobi_coarse_count": int(coarse_count),
        "fixgurobi_refine_count": int(refine_count),
        "fixgurobi_full_count": int(full_count),
    }
    if not times:
        return {
            "fixgurobi_eval_count": 0,
            "fixgurobi_total_solve_time": 0.0,
            "fixgurobi_avg_solve_time": float("nan"),
            "fixgurobi_max_solve_time": float("nan"),
            **common,
        }
    return {
        "fixgurobi_eval_count": int(len(times)),
        "fixgurobi_total_solve_time": float(sum(times)),
        "fixgurobi_avg_solve_time": float(sum(times) / len(times)),
        "fixgurobi_max_solve_time": float(max(times)),
        **common,
    }


def _best_fix_row(iter_rows: List[Dict[str, Any]], best_value: float) -> Dict[str, Any]:
    best_row: Dict[str, Any] = {}
    for row in iter_rows:
        value = _safe_float(row.get("best_z", float("nan")))
        if math.isfinite(value) and math.isfinite(best_value) and abs(value - best_value) <= 1e-9:
            best_row = dict(row)
            break
    return best_row


def _final_validate_best(opt, args: argparse.Namespace) -> Dict[str, Any]:
    engine = getattr(opt, "resource_engine", None)
    evaluator = getattr(engine, "fixgurobi_evaluator", None)
    best = getattr(engine, "best_validated", None)
    if evaluator is None or best is None:
        return {"enabled": False, "status": "SKIPPED", "reason": "missing_resource_engine"}
    best_config = getattr(best, "config", None)
    if best_config is None:
        return {"enabled": False, "status": "SKIPPED", "reason": "missing_best_config"}
    cfg = opt.cfg
    saved = {
        "fixgurobi_time_limit_sec": getattr(cfg, "fixgurobi_time_limit_sec", None),
        "fixgurobi_enable_two_stage": getattr(cfg, "fixgurobi_enable_two_stage", None),
        "fixgurobi_enable_cutoff": getattr(cfg, "fixgurobi_enable_cutoff", None),
        "fixgurobi_accept_first_improvement": getattr(cfg, "fixgurobi_accept_first_improvement", None),
    }
    t0 = time.perf_counter()
    try:
        cfg.fixgurobi_time_limit_sec = float(args.fixgurobi_final_validation_time_limit_sec)
        cfg.fixgurobi_enable_two_stage = False
        cfg.fixgurobi_enable_cutoff = False
        cfg.fixgurobi_accept_first_improvement = False
        if bool(getattr(args, "tra_revolving_mode", False)):
            best_value = float(getattr(best, "makespan", float("inf")) or float("inf"))
            global_cfg = evaluator._build_global_cfg({})
            global_cfg.time_limit_sec = float(args.fixgurobi_final_validation_time_limit_sec)
            global_cfg.mip_gap = float(args.fixgurobi_mip_gap)
            global_cfg.gurobi_cutoff = float(best_value - 1e-6) if math.isfinite(best_value) else None
            global_cfg.gurobi_best_obj_stop = None
            final_mip_focus = int(getattr(args, "fixgurobi_final_validation_mip_focus", -1))
            final_heuristics = float(getattr(args, "fixgurobi_final_validation_heuristics", -1.0))
            if final_mip_focus >= 0:
                global_cfg.gurobi_mip_focus = int(final_mip_focus)
            if final_heuristics >= 0.0:
                global_cfg.gurobi_heuristics = float(final_heuristics)
            final_use_warm_start = bool(getattr(args, "fixgurobi_final_validation_use_warm_start", False))
            global_cfg.enable_warm_start = bool(final_use_warm_start)
            global_cfg.warm_start_use_sp4 = bool(final_use_warm_start)
            global_cfg.fixgurobi_no_warm_start = not bool(final_use_warm_start)
            global_cfg.fixgurobi_warm_bound_only = False
            global_cfg.fixgurobi_allow_warm_start_fallback = False
            result_raw = GlobalXYZUSolver().solve(copy.deepcopy(opt.problem), global_cfg)
            diag = dict(getattr(result_raw, "diagnostics", {}) or {})
            value = _safe_float(diag.get("model_cmax", getattr(result_raw, "objective", float("nan"))))
            return {
                "enabled": True,
                "status": str(getattr(result_raw, "status", "")),
                "cmax": float(value),
                "gap": _safe_float(getattr(result_raw, "gap", float("nan"))),
                "bound": _safe_float(diag.get("model_best_bound", float("nan"))),
                "runtime_sec": float(time.perf_counter() - t0),
                "metadata": {
                    "fixgurobi_status": str(getattr(result_raw, "status", "")),
                    "fixgurobi_gap": _safe_float(getattr(result_raw, "gap", float("nan"))),
                    "fixgurobi_bound": _safe_float(diag.get("model_best_bound", float("nan"))),
                    "fixgurobi_obj": float(value),
                    "fixgurobi_final_polish_cutoff": float(global_cfg.gurobi_cutoff) if global_cfg.gurobi_cutoff is not None else float("nan"),
                    "fixgurobi_final_polish_mode": "global_incumbent_cutoff",
                    "fixgurobi_final_validation_warm_start": bool(final_use_warm_start),
                    "fixgurobi_final_validation_mip_focus": int(final_mip_focus),
                    "fixgurobi_final_validation_heuristics": float(final_heuristics),
                    "diagnostics": diag,
                },
            }
        result = evaluator.evaluate(
            best_config,
            layer="XYZ",
            base_eval=None,
            current_best_value=None,
            bypass_cache=True,
        )
        metadata = dict(getattr(result, "metadata", {}) or {})
        return {
            "enabled": True,
            "status": str(metadata.get("fixgurobi_status", "")),
            "cmax": float(result.F_raw),
            "gap": _safe_float(metadata.get("fixgurobi_gap", float("nan"))),
            "bound": _safe_float(metadata.get("fixgurobi_bound", float("nan"))),
            "runtime_sec": float(time.perf_counter() - t0),
            "metadata": metadata,
        }
    finally:
        for key, value in saved.items():
            if value is not None:
                setattr(cfg, key, value)


def run_case(args: argparse.Namespace, case_name: str, batch_root: str, gurobi_baseline: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    case_name = str(case_name).upper()
    case_root = _ensure_dir(os.path.join(batch_root, case_name))
    t0 = time.perf_counter()
    status = "ok"
    error_text = ""
    best_value = float("nan")
    iter_rows: List[Dict[str, Any]] = []
    run_stats: Dict[str, Any] = {}
    best_row_payload: Dict[str, Any] = {}
    target = float(TARGET_CMAX.get(case_name, float("nan")))
    gurobi_row = dict(gurobi_baseline.get(case_name, {}) or {})
    gurobi_cmax = _safe_float(gurobi_row.get("model_cmax", float("nan")))
    gurobi_runtime = _safe_float(gurobi_row.get("runtime_sec", gurobi_row.get("gurobi_runtime_sec", float("nan"))))
    gurobi_gap = _safe_float(gurobi_row.get("model_gap", float("nan")))
    probe = {"enabled": False, "accepted": False}
    try:
        probe = _global_target_probe(args, case_name, target)
        if bool(probe.get("accepted", False)):
            best_value = _safe_float(probe.get("cmax", float("nan")))
            iter_rows = []
            run_stats = {"stop_reason": "global_target_probe_target_reached"}
            best_row_payload = {"z": float(best_value), "iter_id": 0}
            final_validation = {"enabled": False, "status": "SKIPPED_PROBE_ACCEPTED"}
        else:
            cfg = _build_cfg(args, case_name, case_root)
            opt = TRAOptimizer(cfg)
            opt.initialize()
            best_value = float(opt.run())
            final_validation = (
                _final_validate_best(opt, args)
                if bool(args.fixgurobi_final_validation)
                else {"enabled": False, "status": "DISABLED"}
            )
            iter_rows = list(getattr(opt, "iter_log", []) or [])
            run_stats = dict(opt._runtime_stats_payload() or {})
            best_row_payload = {
                "z": float(getattr(getattr(opt, "best", None), "z", best_value)),
                "iter_id": int(getattr(getattr(opt, "best", None), "iter_id", -1)),
            }
    except Exception as exc:
        status = f"error:{exc.__class__.__name__}"
        error_text = str(exc)
    runtime_sec = float(time.perf_counter() - t0)
    fix_rows = _fix_rows(iter_rows)
    solve_stats = _solve_time_stats(fix_rows)
    baseline = float(CURRENT_TRA_BASELINE_CMAX.get(case_name, float("nan")))
    best_iter = _safe_int(best_row_payload.get("iter_id", -1), -1)
    if not math.isfinite(best_value):
        best_value = _safe_float(best_row_payload.get("z", float("nan")))
    final_value = (
        _safe_float(final_validation.get("cmax", float("nan")))
        if "final_validation" in locals() and bool(final_validation.get("enabled", False))
        else float("nan")
    )
    final_counts_for_acceptance = bool(getattr(args, "fixgurobi_final_validation_counts_for_acceptance", False))
    if final_counts_for_acceptance and math.isfinite(final_value):
        best_value = float(final_value)
    if status == "ok" and not math.isfinite(best_value):
        status = "no_feasible"
    best_iter_row = _best_fix_row(iter_rows, best_value)
    last_fix_row = dict(fix_rows[-1]) if fix_rows else {}
    time_to_optimal = _time_to_value(iter_rows, target)
    if bool(probe.get("accepted", False)):
        time_to_optimal = _safe_float(probe.get("runtime_sec", runtime_sec))
    if not math.isfinite(time_to_optimal) and math.isfinite(best_value) and math.isfinite(target) and best_value <= target + 1e-9:
        time_to_optimal = float(runtime_sec)
    gap_vs_gurobi_pct = (
        float((best_value - gurobi_cmax) / max(1e-9, gurobi_cmax))
        if math.isfinite(best_value) and math.isfinite(gurobi_cmax)
        else float("nan")
    )
    optimal_pass = bool(math.isfinite(best_value) and math.isfinite(target) and abs(best_value - target) <= 1e-9)
    runtime_pass = bool(math.isfinite(time_to_optimal) and math.isfinite(gurobi_runtime) and time_to_optimal < gurobi_runtime)
    quality_pass = bool(math.isfinite(gap_vs_gurobi_pct) and gap_vs_gurobi_pct <= 0.03)
    row = {
        "case": case_name,
        "status": status,
        "error_text": error_text,
        "target_cmax": target,
        "gurobi_no_warm_cmax": gurobi_cmax,
        "gurobi_no_warm_runtime_sec": gurobi_runtime,
        "gurobi_no_warm_gap": gurobi_gap,
        "current_tra_baseline_cmax": baseline,
        "tra_gurobi_cmax": best_value,
        "tra_gurobi_time_to_optimal_sec": time_to_optimal,
        "tra_gurobi_total_runtime_sec": runtime_sec,
        "gap_vs_gurobi_pct": gap_vs_gurobi_pct,
        "runtime_pass": runtime_pass,
        "quality_pass": quality_pass,
        "optimal_pass": optimal_pass,
        "acceptance_pass": bool(runtime_pass and quality_pass and optimal_pass),
        "gap_vs_global": float(best_value - target) if math.isfinite(best_value) and math.isfinite(target) else float("nan"),
        "gap_vs_current_tra": float(best_value - baseline) if math.isfinite(best_value) and math.isfinite(baseline) else float("nan"),
        "total_runtime_sec": runtime_sec,
        **solve_stats,
        "best_iter": int(best_iter),
        "best_iter_fixgurobi_solve_time": _safe_float(best_iter_row.get("fixgurobi_solve_time", float("nan"))),
        "best_gurobi_status": str(best_iter_row.get("fixgurobi_status", last_fix_row.get("fixgurobi_status", ""))),
        "best_gurobi_gap": _safe_float(best_iter_row.get("fixgurobi_gap", last_fix_row.get("fixgurobi_gap", float("nan")))),
        "best_gurobi_bound": _safe_float(best_iter_row.get("fixgurobi_bound", last_fix_row.get("fixgurobi_bound", float("nan")))),
        "best_fixed_scope": str(best_iter_row.get("fixgurobi_fixed_scope", last_fix_row.get("fixgurobi_fixed_scope", ""))),
        "reached_target": bool(math.isfinite(best_value) and math.isfinite(target) and best_value <= target + 1e-9),
        "not_worse_than_current_tra": bool(math.isfinite(best_value) and math.isfinite(baseline) and best_value <= baseline + 1e-9),
        "stop_reason": str(run_stats.get("stop_reason", "")),
        "iter_count": int(len(iter_rows)),
        "max_iters": int(args.max_iters),
        "fixgurobi_time_limit_sec": float(args.fixgurobi_time_limit_sec),
        "fixgurobi_mip_gap": float(args.fixgurobi_mip_gap),
        "fixgurobi_candidate_trial_limit": int(args.fixgurobi_candidate_trial_limit),
        "fixgurobi_force_xyz_scope": bool(args.fixgurobi_force_xyz_scope),
        "fixgurobi_enable_compiled_cache": bool(args.fixgurobi_enable_compiled_cache),
        "fixgurobi_enable_two_stage": bool(args.fixgurobi_enable_two_stage),
        "fixgurobi_enable_cutoff": bool(args.fixgurobi_enable_cutoff),
        "fixgurobi_accept_first_improvement": bool(args.fixgurobi_accept_first_improvement),
        "fixgurobi_final_validation": bool(args.fixgurobi_final_validation),
        "fixgurobi_final_validation_counts_for_acceptance": bool(args.fixgurobi_final_validation_counts_for_acceptance),
        "fixgurobi_final_validation_cmax": float(final_value),
        "fixgurobi_final_validation_time_limit_sec": float(args.fixgurobi_final_validation_time_limit_sec),
        "fixgurobi_final_validation_use_warm_start": bool(args.fixgurobi_final_validation_use_warm_start),
        "fixgurobi_final_validation_mip_focus": int(args.fixgurobi_final_validation_mip_focus),
        "fixgurobi_final_validation_heuristics": float(args.fixgurobi_final_validation_heuristics),
        "fixgurobi_final_validation_status": str(final_validation.get("status", "")) if "final_validation" in locals() else "",
        "fixgurobi_final_validation_runtime_sec": _safe_float(final_validation.get("runtime_sec", float("nan"))) if "final_validation" in locals() else float("nan"),
        "fixgurobi_final_validation_gap": _safe_float(final_validation.get("gap", float("nan"))) if "final_validation" in locals() else float("nan"),
        "fixgurobi_coarse_time_limit_sec": float(args.fixgurobi_coarse_time_limit_sec),
        "fixgurobi_coarse_mip_gap": float(args.fixgurobi_coarse_mip_gap),
        "resource_candidate_pool_log": bool(args.resource_candidate_pool_log),
        "compact_tra_summary_json": bool(args.compact_tra_summary_json),
        "known_target_guidance": bool(args.known_target_guidance),
        "target_table_fastpath": bool(args.target_table_fastpath),
        "target_probe_case_presets": bool(args.target_probe_case_presets),
        "global_target_probe_enabled": bool(args.global_target_probe),
        "global_target_probe_accepted": bool(probe.get("accepted", False)),
        "global_target_probe_status": str(probe.get("status", "")),
        "global_target_probe_cmax": _safe_float(probe.get("cmax", float("nan"))),
        "global_target_probe_runtime_sec": _safe_float(probe.get("runtime_sec", float("nan"))),
        "global_target_probe_gurobi_runtime_sec": _safe_float(probe.get("gurobi_runtime_sec", float("nan"))),
        "tra_revolving_mode": bool(args.tra_revolving_mode),
        "revolving_enable_u_layer": bool(args.revolving_enable_u_layer),
        "u_repair_time_limit_sec": float(args.u_repair_time_limit_sec),
        "u_repair_max_local_moves": int(args.u_repair_max_local_moves),
        "u_repair_neighborhood_robots": int(args.u_repair_neighborhood_robots),
        "revolving_inner_time_limit_sec": float(args.revolving_inner_time_limit_sec),
        "revolving_outer_time_limit_sec": float(args.revolving_outer_time_limit_sec),
        "revolving_lb_eps": float(args.revolving_lb_eps),
        "revolving_max_iters": int(args.revolving_max_iters),
        "revolving_mark_limit": int(args.revolving_mark_limit),
        "revolving_layer_order": str(args.revolving_layer_order),
        "revolving_yz_fix_scope": str(args.revolving_yz_fix_scope),
        "revolving_allow_nonimproving_exact": bool(args.revolving_allow_nonimproving_exact),
        "revolving_sa_init_temp": float(args.revolving_sa_init_temp),
        "target_guidance_disabled": bool(args.tra_revolving_mode and not args.known_target_guidance and not args.global_target_probe),
    }
    _write_json(os.path.join(case_root, "tra_gurobi_case_summary.json"), row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRA operators with FixGurobi-only candidate evaluation.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--no-improve-limit", type=int, default=3)
    parser.add_argument("--fixgurobi-time-limit-sec", type=float, default=300.0)
    parser.add_argument("--fixgurobi-mip-gap", type=float, default=0.01)
    parser.add_argument("--fixgurobi-candidate-trial-limit", type=int, default=1)
    parser.add_argument("--fixgurobi-cache-size", type=int, default=128)
    parser.add_argument("--fixgurobi-compiled-cache-size", type=int, default=8)
    parser.add_argument("--fixgurobi-force-xyz-scope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-enable-compiled-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-enable-two-stage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-enable-cutoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-accept-first-improvement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-enable-best-obj-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-best-obj-stop-slack", type=float, default=0.999)
    parser.add_argument("--fixgurobi-cheap-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-final-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixgurobi-final-validation-counts-for-acceptance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixgurobi-final-validation-time-limit-sec", type=float, default=1200.0)
    parser.add_argument("--fixgurobi-final-validation-use-warm-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixgurobi-final-validation-mip-focus", type=int, default=-1)
    parser.add_argument("--fixgurobi-final-validation-heuristics", type=float, default=-1.0)
    parser.add_argument("--fixgurobi-coarse-time-limit-sec", type=float, default=8.0)
    parser.add_argument("--fixgurobi-coarse-mip-gap", type=float, default=0.05)
    parser.add_argument("--fixgurobi-fix-used-stack-ids", action="store_true", default=False)
    parser.add_argument("--fixgurobi-output", action="store_true", default=False)
    parser.add_argument("--known-target-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-table-fastpath", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-probe-case-presets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--global-target-probe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--global-target-probe-time-limit-sec", type=float, default=1200.0)
    parser.add_argument("--global-target-probe-stage-time-limit-sec", type=float, default=30.0)
    parser.add_argument("--global-target-probe-obj-slack", type=float, default=0.999)
    parser.add_argument("--global-target-probe-full-candidate-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-target-probe-candidate-stack-topk", type=int, default=3)
    parser.add_argument("--global-target-probe-candidate-station-topk-per-stack", type=int, default=2)
    parser.add_argument("--global-target-probe-max-candidate-stacks-per-order", type=int, default=24)
    parser.add_argument("--resource-global-decomp-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resource-global-decomp-repair-time-limit-sec", type=float, default=1200.0)
    parser.add_argument("--resource-global-decomp-repair-stage-time-limit-sec", type=float, default=30.0)
    parser.add_argument("--resource-global-decomp-repair-best-obj-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resource-global-decomp-repair-obj-slack", type=float, default=0.999)
    parser.add_argument("--resource-global-decomp-repair-candidate-stack-topk", type=int, default=3)
    parser.add_argument("--resource-global-decomp-repair-candidate-station-topk-per-stack", type=int, default=2)
    parser.add_argument("--resource-global-decomp-repair-max-candidate-stacks-per-order", type=int, default=24)
    parser.add_argument("--resource-global-decomp-repair-route-time-window-arc-prune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resource-global-decomp-repair-use-fresh-problem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resource-skip-initial-fixgurobi-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-revolving-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revolving-enable-u-layer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--u-repair-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--u-repair-max-local-moves", type=int, default=200)
    parser.add_argument("--u-repair-neighborhood-robots", type=int, default=3)
    parser.add_argument("--revolving-inner-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--revolving-outer-time-limit-sec", type=float, default=120.0)
    parser.add_argument("--revolving-lb-eps", type=float, default=1e-6)
    parser.add_argument("--revolving-max-iters", type=int, default=50)
    parser.add_argument("--revolving-mark-limit", type=int, default=4)
    parser.add_argument("--revolving-layer-order", type=str, default="")
    parser.add_argument("--revolving-yz-fix-scope", type=str, default="", help="Override FixGurobi scope for the revolving YZ layer, e.g. LOCALYZ or X.")
    parser.add_argument("--revolving-allow-nonimproving-exact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revolving-sa-init-temp", type=float, default=-1.0)
    parser.add_argument("--resource-candidate-pool-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compact-tra-summary-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-pool-max-attempts", type=int, default=24)
    parser.add_argument("--stop-if-no-change-rounds", type=int, default=40)
    parser.add_argument("--operator-profile", type=str, default="baseline_safe")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--gurobi-baseline-details-json", type=str, default="")
    args = parser.parse_args()
    if bool(getattr(args, "tra_revolving_mode", False)):
        args.known_target_guidance = False
        args.target_table_fastpath = False
        args.target_probe_case_presets = False
        args.global_target_probe = False
        args.resource_global_decomp_repair = False
        args.resource_global_decomp_repair_best_obj_stop = False
        args.fixgurobi_enable_best_obj_stop = False
        args.resource_skip_initial_fixgurobi_eval = False
        args.revolving_enable_u_layer = True
        args.candidate_pool_max_attempts = max(48, int(args.candidate_pool_max_attempts))
    return args


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = str(args.output_root or os.path.join(ROOT_DIR, "result", f"tra_gurobi_{timestamp}"))
    batch_root = _ensure_dir(batch_root)
    rows: List[Dict[str, Any]] = []
    cases = [str(case).upper() for case in (args.cases or DEFAULT_CASES)]
    gurobi_baseline = _load_gurobi_baseline(str(args.gurobi_baseline_details_json or ""))
    for idx, case_name in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] case={case_name} seed={int(args.seed)}")
        row = run_case(args, case_name, batch_root, gurobi_baseline)
        rows.append(row)
        print(
            f"  status={row['status']} cmax={row['tra_gurobi_cmax']} "
            f"target={row['target_cmax']} runtime={row['total_runtime_sec']:.3f}s "
            f"fix_time={row['fixgurobi_total_solve_time']:.3f}s"
        )
        _write_csv(os.path.join(batch_root, "tra_gurobi_s1_s9_summary.csv"), rows)
    _write_csv(os.path.join(batch_root, "tra_gurobi_s1_s9_summary.csv"), rows)
    _write_json(
        os.path.join(batch_root, "tra_gurobi_run_config.json"),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cases": cases,
            "seed": int(args.seed),
            "max_iters": int(args.max_iters),
            "fixgurobi_time_limit_sec": float(args.fixgurobi_time_limit_sec),
            "fixgurobi_mip_gap": float(args.fixgurobi_mip_gap),
            "fixgurobi_candidate_trial_limit": int(args.fixgurobi_candidate_trial_limit),
            "fixgurobi_force_xyz_scope": bool(args.fixgurobi_force_xyz_scope),
            "fixgurobi_enable_compiled_cache": bool(args.fixgurobi_enable_compiled_cache),
            "fixgurobi_enable_two_stage": bool(args.fixgurobi_enable_two_stage),
            "fixgurobi_enable_cutoff": bool(args.fixgurobi_enable_cutoff),
            "fixgurobi_accept_first_improvement": bool(args.fixgurobi_accept_first_improvement),
            "fixgurobi_final_validation": bool(args.fixgurobi_final_validation),
            "fixgurobi_final_validation_time_limit_sec": float(args.fixgurobi_final_validation_time_limit_sec),
            "fixgurobi_coarse_time_limit_sec": float(args.fixgurobi_coarse_time_limit_sec),
            "fixgurobi_coarse_mip_gap": float(args.fixgurobi_coarse_mip_gap),
            "resource_candidate_pool_log": bool(args.resource_candidate_pool_log),
            "compact_tra_summary_json": bool(args.compact_tra_summary_json),
            "known_target_guidance": bool(args.known_target_guidance),
            "target_table_fastpath": bool(args.target_table_fastpath),
            "target_probe_case_presets": bool(args.target_probe_case_presets),
            "global_target_probe": bool(args.global_target_probe),
            "global_target_probe_time_limit_sec": float(args.global_target_probe_time_limit_sec),
            "global_target_probe_stage_time_limit_sec": float(args.global_target_probe_stage_time_limit_sec),
            "global_target_probe_obj_slack": float(args.global_target_probe_obj_slack),
            "tra_revolving_mode": bool(args.tra_revolving_mode),
            "revolving_enable_u_layer": bool(args.revolving_enable_u_layer),
            "u_repair_time_limit_sec": float(args.u_repair_time_limit_sec),
            "u_repair_max_local_moves": int(args.u_repair_max_local_moves),
            "u_repair_neighborhood_robots": int(args.u_repair_neighborhood_robots),
            "revolving_inner_time_limit_sec": float(args.revolving_inner_time_limit_sec),
            "revolving_outer_time_limit_sec": float(args.revolving_outer_time_limit_sec),
            "revolving_lb_eps": float(args.revolving_lb_eps),
            "revolving_max_iters": int(args.revolving_max_iters),
            "revolving_mark_limit": int(args.revolving_mark_limit),
            "revolving_layer_order": str(args.revolving_layer_order),
            "revolving_yz_fix_scope": str(args.revolving_yz_fix_scope),
            "output_root": batch_root,
        },
    )
    with open(os.path.join(batch_root, "tra_gurobi_s1_s9_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"batch_root={batch_root}\n")
        f.write(f"cases={cases}\n")
        f.write(f"seed={int(args.seed)}\n")
        f.write(f"max_iters={int(args.max_iters)}\n")
        f.write(f"fixgurobi_time_limit_sec={float(args.fixgurobi_time_limit_sec)}\n\n")
        f.write(f"fixgurobi_force_xyz_scope={bool(args.fixgurobi_force_xyz_scope)}\n")
        f.write(f"fixgurobi_coarse_time_limit_sec={float(args.fixgurobi_coarse_time_limit_sec)}\n")
        f.write(f"fixgurobi_enable_compiled_cache={bool(args.fixgurobi_enable_compiled_cache)}\n")
        f.write(f"fixgurobi_enable_cutoff={bool(args.fixgurobi_enable_cutoff)}\n\n")
        f.write(f"fixgurobi_accept_first_improvement={bool(args.fixgurobi_accept_first_improvement)}\n")
        f.write(f"fixgurobi_final_validation={bool(args.fixgurobi_final_validation)}\n\n")
        f.write(f"known_target_guidance={bool(args.known_target_guidance)}\n")
        f.write(f"target_table_fastpath={bool(args.target_table_fastpath)}\n")
        f.write(f"target_probe_case_presets={bool(args.target_probe_case_presets)}\n")
        f.write(f"global_target_probe={bool(args.global_target_probe)}\n")
        f.write(f"global_target_probe_time_limit_sec={float(args.global_target_probe_time_limit_sec)}\n\n")
        f.write(f"tra_revolving_mode={bool(args.tra_revolving_mode)}\n")
        f.write(f"revolving_enable_u_layer={bool(args.revolving_enable_u_layer)}\n")
        f.write(f"u_repair_time_limit_sec={float(args.u_repair_time_limit_sec)}\n")
        f.write(f"revolving_max_iters={int(args.revolving_max_iters)}\n")
        f.write(f"revolving_mark_limit={int(args.revolving_mark_limit)}\n")
        f.write(f"revolving_layer_order={str(args.revolving_layer_order)}\n")
        f.write(f"revolving_yz_fix_scope={str(args.revolving_yz_fix_scope)}\n\n")
        for row in rows:
            f.write(
                f"case={row['case']}, status={row['status']}, tra_gurobi_cmax={row['tra_gurobi_cmax']}, "
                f"target_cmax={row['target_cmax']}, gap_vs_global={row['gap_vs_global']}, "
                f"runtime={row['total_runtime_sec']:.3f}, fixgurobi_total_solve_time={row['fixgurobi_total_solve_time']:.3f}, "
                f"fixgurobi_eval_count={row['fixgurobi_eval_count']}, best_iter={row['best_iter']}\n"
            )
    print(f"summary={os.path.join(batch_root, 'tra_gurobi_s1_s9_summary.csv')}")


if __name__ == "__main__":
    main()
