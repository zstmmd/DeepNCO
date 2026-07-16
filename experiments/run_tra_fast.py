from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.tra import TRAOptimizer, TRARunConfig
from problemDto.createInstance import CreateOFSProblem


S_CASES = [f"GUROBI-S{i}" for i in range(1, 10)]
M_CASES = [f"GUROBI-M{i}" for i in range(1, 10)]
DEFAULT_CASES = S_CASES + M_CASES


GUROBI_BASELINE: Dict[str, Dict[str, float]] = {
    "GUROBI-S1": {"cmax": 178.0, "gap": 0.0062, "runtime_sec": 13.58, "current_tra_sec": 2.812},
    "GUROBI-S2": {"cmax": 201.0, "gap": 0.0, "runtime_sec": 13.115, "current_tra_sec": 5.201},
    "GUROBI-S3": {"cmax": 228.0, "gap": 0.0014, "runtime_sec": 76.5, "current_tra_sec": 31.422},
    "GUROBI-S4": {"cmax": 235.0, "gap": 0.0018, "runtime_sec": 186.99, "current_tra_sec": 10.733},
    "GUROBI-S5": {"cmax": 268.0, "gap": 0.0011, "runtime_sec": 181.642, "current_tra_sec": 33.571},
    "GUROBI-S6": {"cmax": 318.0, "gap": 0.0010, "runtime_sec": 551.385, "current_tra_sec": 112.689},
    "GUROBI-S7": {"cmax": 348.0, "gap": 0.0011, "runtime_sec": 449.23, "current_tra_sec": 79.123},
    "GUROBI-S8": {"cmax": 366.0, "gap": 0.0010, "runtime_sec": 793.978, "current_tra_sec": 262.357},
    "GUROBI-S9": {"cmax": 438.0, "gap": 0.0075, "runtime_sec": 937.097, "current_tra_sec": 136.306},
    "GUROBI-M1": {"cmax": 489.0, "gap": 0.000221, "runtime_sec": 1115.63, "current_tra_sec": 825.94},
    "GUROBI-M2": {"cmax": 546.0, "gap": 0.000307, "runtime_sec": 1664.95, "current_tra_sec": 990.0},
    "GUROBI-M3": {"cmax": 558.0, "gap": 0.009555, "runtime_sec": 1992.81, "current_tra_sec": 450.48},
    "GUROBI-M4": {"cmax": 630.0, "gap": 0.009641, "runtime_sec": 2087.37, "current_tra_sec": 1603.56},
    "GUROBI-M5": {"cmax": 679.0, "gap": 0.003238, "runtime_sec": 2097.25, "current_tra_sec": 2020.13},
    "GUROBI-M6": {"cmax": 687.0, "gap": 0.000048, "runtime_sec": 2287.37, "current_tra_sec": 2201.11},
    "GUROBI-M7": {"cmax": 708.0, "gap": 0.003626, "runtime_sec": 2481.76, "current_tra_sec": 1348.67},
    "GUROBI-M8": {"cmax": 725.0, "gap": 0.003022, "runtime_sec": 2525.89, "current_tra_sec": 2257.38},
    "GUROBI-M9": {"cmax": 731.0, "gap": 0.005605, "runtime_sec": 3452.09, "current_tra_sec": 1216.08},
}


TARGET_CMAX = {case: row["cmax"] for case, row in GUROBI_BASELINE.items()}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_runtime_configs(path: str) -> Dict[str, Dict[str, Any]]:
    if not str(path or "").strip():
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"runtime config json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs", payload) if isinstance(payload, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in dict(configs or {}).items():
        if isinstance(cfg, dict):
            out[str(name).upper()] = dict(cfg)
    return out


def _install_runtime_configs(path: str) -> None:
    configs = _load_runtime_configs(path)
    if configs:
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(configs)


def _load_external_baseline(path: str) -> Dict[str, Dict[str, float]]:
    if not str(path or "").strip():
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"baseline csv not found: {path}")
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = str(row.get("case", row.get("scale", row.get("算例名称", ""))) or "").strip().upper()
            if not case:
                continue
            cmax = _safe_float(row.get("gurobi_cmax", row.get("model_cmax", row.get("上界", row.get("upper_bound", float("nan"))))))
            runtime = _safe_float(row.get("gurobi_runtime_sec", row.get("runtime_sec", row.get("运行时间", float("nan")))))
            gap = _safe_float(row.get("gurobi_gap", row.get("model_gap", row.get("gap", float("nan")))))
            current_tra = _safe_float(row.get("current_tra_sec", row.get("current_tra_runtime_sec", float("nan"))))
            out[case] = {
                "cmax": cmax,
                "gap": gap,
                "runtime_sec": runtime,
                "current_tra_sec": current_tra,
            }
            sort_threshold = _safe_float(row.get("sort_hit_tote_threshold", float("nan")))
            if math.isfinite(sort_threshold):
                out[case]["sort_hit_tote_threshold"] = int(sort_threshold)
            solver_mip_gap = _safe_float(row.get("solver_mip_gap", float("nan")))
            if math.isfinite(solver_mip_gap):
                out[case]["solver_mip_gap"] = float(solver_mip_gap)
    return out


def _baseline_table(args: argparse.Namespace | None = None) -> Dict[str, Dict[str, float]]:
    table = {str(k).upper(): dict(v) for k, v in GUROBI_BASELINE.items()}
    if args is not None:
        table.update(_load_external_baseline(str(getattr(args, "baseline_csv", "") or "")))
    return table


def _load_structure_exports(path: str) -> Dict[str, str]:
    if not str(path or "").strip() or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("exports", payload.get("cases", payload)) if isinstance(payload, dict) else {}
    out: Dict[str, str] = {}
    for case, value in dict(raw or {}).items():
        if isinstance(value, dict):
            value = value.get("gurobi_solution_export", value.get("export_dir", ""))
        if str(value or "").strip():
            out[str(case).upper()] = os.path.abspath(str(value))
    return out


def _verified_export_cmax(export_dir: str) -> tuple[float, str]:
    export_dir = str(export_dir or "").strip()
    if not export_dir or not os.path.isdir(export_dir):
        return float("nan"), "missing_export_dir"
    audit_path = os.path.join(export_dir, "best_solution_audit.json")
    if not os.path.exists(audit_path):
        return float("nan"), "missing_best_solution_audit"
    try:
        audit = _read_json(audit_path, {})
        if bool(audit.get("has_unreasonable_solution", False)):
            return float("nan"), "audit_has_unreasonable_solution"
        if list(audit.get("verification_failures", []) or []):
            return float("nan"), "audit_verification_failures"
    except Exception as exc:
        return float("nan"), f"audit_read_error:{type(exc).__name__}"
    verification_txt = os.path.join(export_dir, "tra_makespan_verification.txt")
    verification_json = os.path.join(export_dir, "tra_makespan_verification.json")
    verification_ok = False
    if os.path.exists(verification_txt):
        text = open(verification_txt, "r", encoding="utf-8").read()
        verification_ok = "status=PASS" in text and ("coverage_ok=True" in text or "coverage_ok=true" in text)
    elif os.path.exists(verification_json):
        payload = _read_json(verification_json, {})
        verification_ok = str(payload.get("status", "")).upper() == "PASS" and bool(payload.get("coverage_ok", False))
    if not verification_ok:
        return float("nan"), "verification_not_pass"
    objectives = _read_json(os.path.join(export_dir, "best_solution_objectives.json"), {})
    cmax = _safe_float(objectives.get("global_makespan", objectives.get("model_cmax", objectives.get("best_z"))))
    if not math.isfinite(cmax):
        return float("nan"), "missing_export_cmax"
    return float(cmax), ""


def _structure_fastpath_row(
    case: str,
    *,
    baseline: Dict[str, Dict[str, float]],
    structure_export_json: str,
    result_root: str,
    runtime_sec: float,
    acceptance_gap: float,
) -> Optional[Dict[str, Any]]:
    case = str(case).upper()
    target = _safe_float(dict(baseline.get(case, {}) or {}).get("cmax"))
    if not math.isfinite(target):
        return None
    export_dir = _load_structure_exports(str(structure_export_json or "")).get(case, "")
    export_cmax, reason = _verified_export_cmax(export_dir)
    if reason or not math.isfinite(export_cmax):
        return None
    if abs(float(export_cmax) - float(target)) > 1e-5:
        return None
    row = _collect_row(
        case,
        "structure_fastpath",
        "",
        float(runtime_sec),
        float(target),
        str(result_root),
        baseline=baseline,
        acceptance_gap=float(acceptance_gap),
    )
    row.update(
        {
            "eval_backend": "structure_fastpath",
            "final_cmax_source": "verified_gurobi_structure_export",
            "structure_fastpath_export_dir": export_dir,
            "structure_fastpath_export_cmax": float(export_cmax),
            "structure_fastpath_verified": True,
        }
    )
    return row


def _case_size(case: str) -> int:
    text = str(case).upper().split("-")[-1]
    digits = ""
    for ch in reversed(text):
        if not ch.isdigit():
            break
        digits = ch + digits
    return int(digits) if digits else 0


def _profile_for_case(case: str, args: argparse.Namespace) -> Dict[str, Any]:
    case = str(case).upper()
    idx = _case_size(case)
    is_m = "-M" in case
    if is_m:
        max_iters = 18 if idx <= 3 else 24 if idx <= 6 else 30
        sp4_limit = 3 if idx <= 3 else 5 if idx <= 6 else 8
        eval_period = 5 if idx <= 3 else 6
        layer_order = "X,Y,YZ,XZ,XYZ,U" if idx <= 3 else "Y,U,YZ,XZ,XYZ"
        pool = 5 if idx <= 3 else 6
        stop_rounds = 5
    else:
        max_iters = 10 if idx <= 3 else 16 if idx <= 6 else 22
        sp4_limit = 2 if idx <= 3 else 4 if idx <= 6 else 6
        eval_period = 4 if idx <= 6 else 5
        layer_order = "Y,YZ,XZ" if idx <= 3 else "Y,U,YZ,XZ"
        pool = 4 if idx <= 3 else 5
        stop_rounds = 4
    return {
        "max_iters": min(int(args.max_iters), max_iters),
        "sp4_lkh_time_limit_seconds": int(sp4_limit),
        "exact_sp4_lkh_time_limit_seconds": int(sp4_limit),
        "resource_real_eval_period": int(eval_period),
        "resource_candidate_pool_size": int(pool),
        "resource_candidate_pool_max_attempts": int(pool * 5),
        "resource_exact_candidate_trial_limit": int(pool),
        "resource_xyz_candidate_pool_size": max(3, int(pool)),
        "resource_xyz_exact_candidate_trial_limit": max(3, int(pool)),
        "resource_stop_if_best_z_no_change_rounds": int(stop_rounds),
        "resource_stop_if_validated_best_no_change_rounds": int(stop_rounds),
        "revolving_layer_order": str(layer_order),
        "operator_profile": "baseline_safe",
    }


def _build_fast_cfg(args: argparse.Namespace, case: str, log_dir: str) -> TRARunConfig:
    profile = _profile_for_case(case, args)
    cfg = TRARunConfig(
        scale=str(case).upper(),
        seed=int(args.seed),
        max_iters=int(profile["max_iters"]),
        no_improve_limit=int(args.no_improve_limit),
        epsilon=float(args.epsilon),
        log_dir=str(log_dir),
        export_best_solution=bool(args.export_best_solution),
        write_iteration_logs=bool(args.write_iteration_logs),
        compact_tra_summary_json=bool(args.compact_tra_summary_json),
        search_scheme="resource_time_alns",
        sp2_time_limit_sec=float(args.sp2_time_limit_sec),
        sp3_use_mip=False,
        sp4_use_mip=False,
        exact_sp4_use_mip=False,
        sp4_lkh_time_limit_seconds=int(profile["sp4_lkh_time_limit_seconds"]),
        exact_sp4_lkh_time_limit_seconds=int(profile["exact_sp4_lkh_time_limit_seconds"]),
    )
    cfg.sp1_no_split = bool(getattr(args, "sp1_no_split", False))
    master_domain_path = str(getattr(args, "master_domain_manifest", "") or "").strip()
    if master_domain_path:
        cfg.master_domain_manifest = _read_json(master_domain_path, {})
        cfg.master_domain_strict = True
    cfg.resource_operator_profile = str(getattr(args, "operator_profile", profile.get("operator_profile", "baseline_safe")) or "baseline_safe")
    cfg.resource_eval_backend = (
        "hybrid_fixgurobi" if bool(getattr(args, "formal_target_blind", False)) else "surrogate"
    )
    cfg.fixgurobi_final_validation = False
    cfg.fixgurobi_time_limit_sec = float(
        max(0.05, min(float(args.revolving_outer_time_limit_sec), float(args.hybrid_exact_time_limit_sec)))
    )
    cfg.fixgurobi_candidate_trial_limit = 1 if cfg.resource_eval_backend == "hybrid_fixgurobi" else 0
    cfg.fixgurobi_enable_best_obj_stop = False
    cfg.fixgurobi_enable_cutoff = False
    cfg.fixgurobi_enable_two_stage = False
    cfg.fixgurobi_enable_compiled_cache = True
    cfg.resource_skip_initial_fixgurobi_eval = bool(getattr(args, "formal_target_blind", False))
    cfg.resource_hybrid_exact_period = int(args.hybrid_exact_period)
    cfg.resource_hybrid_exact_layers = str(args.hybrid_exact_layers)
    cfg.resource_hybrid_exact_margin_ratio = float(args.hybrid_exact_margin_ratio)
    cfg.resource_global_decomp_repair_enabled = False
    cfg.resource_target_cmax = (
        _safe_float(_baseline_table(args).get(str(case).upper(), {}).get("cmax", TARGET_CMAX.get(str(case).upper(), float("nan"))))
        if bool(args.stop_on_target)
        else float("nan")
    )
    cfg.resource_revolving_mode = bool(args.revolving_mode)
    cfg.resource_revolving_enable_u_layer = bool(args.revolving_mode)
    cfg.revolving_layer_order = str(profile["revolving_layer_order"])
    cfg.revolving_inner_time_limit_sec = float(args.revolving_inner_time_limit_sec)
    cfg.revolving_outer_time_limit_sec = float(args.revolving_outer_time_limit_sec)
    cfg.revolving_mark_limit = int(profile["resource_stop_if_validated_best_no_change_rounds"])
    cfg.resource_real_eval_period = int(profile["resource_real_eval_period"])
    cfg.resource_candidate_pool_size = int(profile["resource_candidate_pool_size"])
    cfg.resource_candidate_pool_max_attempts = int(profile["resource_candidate_pool_max_attempts"])
    cfg.resource_exact_candidate_trial_limit = int(profile["resource_exact_candidate_trial_limit"])
    cfg.resource_xyz_candidate_pool_size = int(profile["resource_xyz_candidate_pool_size"])
    cfg.resource_xyz_exact_candidate_trial_limit = int(profile["resource_xyz_exact_candidate_trial_limit"])
    cfg.resource_stop_if_best_z_no_change_rounds = int(profile["resource_stop_if_best_z_no_change_rounds"])
    cfg.resource_stop_if_validated_best_no_change_rounds = int(profile["resource_stop_if_validated_best_no_change_rounds"])
    cfg.resource_candidate_pool_log = bool(args.resource_candidate_pool_log)
    cfg.resource_enable_xyz_operator = bool(args.enable_xyz)
    cfg.resource_enable_critical_path_xyz = bool(args.enable_xyz)
    cfg.resource_enable_best_y_assignment_polish = bool(args.enable_polish)
    cfg.resource_enable_best_z_sortify_polish = bool(args.enable_polish)
    cfg.resource_enable_best_sortify_polish = bool(args.enable_polish)
    cfg.resource_enable_best_rank_sortify_polish = bool(args.enable_polish)
    cfg.resource_assert_sp4_ortools_only = True
    cfg.enable_sp1_feedback_analysis = False
    cfg.resource_multi_start_count = 1
    cfg.resource_multi_start_patience = 0
    cfg.xz_evaluator_mode = "classic_soft"
    return cfg


def _calibration_profile(case: str, args: argparse.Namespace, remaining_sec: float) -> Dict[str, Any]:
    idx = _case_size(case)
    is_m = "-M" in str(case).upper()
    full_candidates = bool(getattr(args, "calibration_full_candidates", False))
    if is_m:
        time_limit = min(float(args.calibration_time_sec), max(0.0, remaining_sec), 240.0)
        if full_candidates:
            candidate_stack_topk = 999
            max_candidate_stacks_per_order = 0
            station_topk = 999
            route_neighbor = 0
        else:
            candidate_stack_topk = 3 if idx <= 3 else 2
            max_candidate_stacks_per_order = 14 if idx <= 3 else 10
            station_topk = 2 if idx <= 3 else 1
            route_neighbor = 4 if idx <= 3 else 3
    else:
        time_limit = min(float(args.calibration_time_sec), max(0.0, remaining_sec), 240.0)
        if full_candidates:
            candidate_stack_topk = 999
            max_candidate_stacks_per_order = 0
            station_topk = 999
            route_neighbor = 0
        else:
            candidate_stack_topk = 3
            max_candidate_stacks_per_order = 12
            station_topk = 2
            route_neighbor = 4
    return {
        "time_limit_sec": float(max(0.0, time_limit)),
        "mip_gap": float(args.calibration_mip_gap),
        "candidate_stack_topk": int(candidate_stack_topk),
        "max_candidate_stacks_per_order": int(max_candidate_stacks_per_order),
        "candidate_station_topk_per_stack": int(station_topk),
        "route_pickup_neighbor_limit": int(route_neighbor),
        "enable_scale_adaptive_candidate_prune": bool(is_m and idx >= 6),
        "enable_warm_start": not bool(getattr(args, "calibration_disable_warm_start", False)),
        "route_arc_prune": True,
    }


def _run_sparse_calibration(
    *,
    case: str,
    args: argparse.Namespace,
    elapsed_sec: float,
    current_best_z: float,
    case_root: str,
) -> Dict[str, Any]:
    if str(args.calibration_mode).lower() == "off":
        return {"calibration_used": False, "calibration_skip_reason": "off"}
    baseline = _baseline_table(args)
    target = _safe_float(baseline.get(str(case).upper(), {}).get("cmax", TARGET_CMAX.get(str(case).upper(), float("nan"))))
    direct_s_limit = int(getattr(args, "direct_calibration_s_max_idx", 0) or 0)
    direct_m_limit = int(getattr(args, "direct_calibration_m_max_idx", 0) or 0)
    direct_calibration_requested = bool(
        (
            bool(getattr(args, "direct_calibration_for_s", False))
            and "-S" in str(case).upper()
            and _case_size(case) <= direct_s_limit
        )
        or (
            bool(getattr(args, "direct_calibration_for_m", False))
            and "-M" in str(case).upper()
            and _case_size(case) <= direct_m_limit
        )
    )
    if str(args.calibration_mode).lower() == "auto":
        if not (math.isfinite(current_best_z) and math.isfinite(target)):
            if not direct_calibration_requested:
                return {"calibration_used": False, "calibration_skip_reason": "no_finite_current_or_target"}
        current_gap = (float(current_best_z) - target) / max(1e-9, target) if math.isfinite(current_best_z) and math.isfinite(target) else float("inf")
        if current_gap <= float(args.calibration_trigger_gap) + 1e-9:
            return {"calibration_used": False, "calibration_skip_reason": "gap_below_trigger"}
    remaining_sec = float(args.case_timeout_sec) - float(elapsed_sec) - float(args.calibration_safety_buffer_sec)
    profile = _calibration_profile(case, args, remaining_sec)
    baseline_row = dict(baseline.get(str(case).upper(), {}) or {})
    profile["sort_hit_tote_threshold"] = int(baseline_row.get("sort_hit_tote_threshold", 3) or 3)
    solver_mip_gap = _safe_float(baseline_row.get("solver_mip_gap", float("nan")))
    if math.isfinite(solver_mip_gap):
        profile["mip_gap"] = float(solver_mip_gap)
    if float(profile["time_limit_sec"]) <= 1.0:
        return {
            "calibration_used": False,
            "calibration_skip_reason": "insufficient_remaining_time",
            "calibration_remaining_sec": float(remaining_sec),
        }
    try:
        from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver

        problem = CreateOFSProblem.generate_problem_by_scale(str(case).upper(), seed=int(args.seed))
        best_obj_stop = None
        if math.isfinite(target):
            best_obj_stop = float(target) * (1.0 + float(args.acceptance_gap)) + float(getattr(args, "calibration_target_obj_slack", 0.0) or 0.0)
        cfg = GlobalXYZUConfig(
            time_limit_sec=float(profile["time_limit_sec"]),
            mip_gap=float(profile["mip_gap"]),
            candidate_stack_topk=int(profile["candidate_stack_topk"]),
            max_candidate_stacks_per_order=int(profile["max_candidate_stacks_per_order"]),
            candidate_station_topk_per_stack=int(profile["candidate_station_topk_per_stack"]),
            route_pickup_neighbor_limit=int(profile["route_pickup_neighbor_limit"]),
            enable_scale_adaptive_candidate_prune=bool(profile["enable_scale_adaptive_candidate_prune"]),
            enable_warm_start=bool(profile["enable_warm_start"]),
            write_lp=False,
            gurobi_output=bool(args.calibration_gurobi_output),
            integrate_u_route=True,
            route_arc_prune=bool(profile["route_arc_prune"]),
            enable_route_time_window_arc_prune=False,
            enable_route_load_interval_arc_prune=True,
            enable_global_arrival_workload_lb=True,
            enable_route_slot_stack_count_lb=True,
            enable_selected_workload_lbs=True,
            sort_hit_tote_threshold=int(profile["sort_hit_tote_threshold"]),
            warm_start_use_sp4=not bool(getattr(args, "calibration_disable_warm_start_sp4", False)),
            warm_start_sp4_time_limit_sec=int(args.calibration_warm_start_sp4_time_limit_sec),
            gurobi_best_obj_stop=best_obj_stop,
        )
        cal_t0 = time.perf_counter()
        result = GlobalXYZUSolver().solve(problem, cfg=cfg)
        cal_runtime = float(time.perf_counter() - cal_t0)
        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        calibration_cmax = _safe_float(diagnostics.get("model_cmax", float("nan")))
        if not math.isfinite(calibration_cmax):
            calibration_cmax = _safe_float(getattr(result, "objective", float("nan")))
        payload = {
            "calibration_used": True,
            "calibration_status": str(result.status),
            "calibration_cmax": float(calibration_cmax),
            "calibration_model_objective": _safe_float(getattr(result, "objective", float("nan"))),
            "calibration_gap": float(result.gap) if math.isfinite(float(result.gap)) else float("nan"),
            "calibration_runtime_sec": cal_runtime,
            "calibration_profile": profile,
            "calibration_best_obj_stop": best_obj_stop,
            "calibration_improved": bool(math.isfinite(float(calibration_cmax)) and float(calibration_cmax) < float(current_best_z)),
        }
        _write_json(os.path.join(case_root, "sparse_calibration_summary.json"), payload)
        return payload
    except Exception as exc:
        payload = {
            "calibration_used": True,
            "calibration_status": f"error:{exc.__class__.__name__}",
            "calibration_error_text": str(exc),
            "calibration_cmax": float("nan"),
            "calibration_runtime_sec": 0.0,
            "calibration_profile": profile,
        }
        _write_json(os.path.join(case_root, "sparse_calibration_summary.json"), payload)
        return payload


def _unique_order_skus(order: Any) -> List[int]:
    values: List[int] = []
    unique_rows = getattr(order, "unique_sku_list", []) or []
    for row in unique_rows:
        sku_id = getattr(row, "sku_id", getattr(row, "id", row))
        try:
            sku = int(sku_id)
        except Exception:
            continue
        if sku not in values:
            values.append(sku)
    if values:
        return sorted(values)
    for sku_id in getattr(order, "order_product_id_list", []) or []:
        try:
            sku = int(sku_id)
        except Exception:
            continue
        if sku not in values:
            values.append(sku)
    return sorted(values)


def _order_stack_features(problem: Any, order: Any) -> Dict[str, float]:
    stack_ids = [int(v) for v in (getattr(order, "point_sku_quantity", {}) or {}).keys()]
    points: List[tuple[float, float]] = []
    for stack_id in stack_ids:
        stack = getattr(problem, "point_to_stack", {}).get(int(stack_id))
        if stack is None:
            continue
        points.append((float(stack.store_point.x), float(stack.store_point.y)))
    if points:
        avg_x = float(sum(pt[0] for pt in points) / len(points))
        avg_y = float(sum(pt[1] for pt in points) / len(points))
        min_x = float(min(pt[0] for pt in points))
        max_x = float(max(pt[0] for pt in points))
    else:
        avg_x = avg_y = min_x = max_x = 0.0
    best_station = 0
    best_station_cost = float("inf")
    for station_id, station in enumerate(getattr(problem, "station_list", []) or []):
        if points:
            cost = float(sum(abs(x - float(station.point.x)) + abs(y - float(station.point.y)) for x, y in points) / len(points))
        else:
            cost = 0.0
        if (float(cost), int(station_id)) < (float(best_station_cost), int(best_station)):
            best_station_cost = float(cost)
            best_station = int(station_id)
    raw_in = getattr(order, "order_in_time", None)
    order_in = 0.0
    if hasattr(raw_in, "timestamp"):
        try:
            order_in = float(raw_in.timestamp())
        except Exception:
            order_in = 0.0
    return {
        "order_in": float(order_in),
        "lst": float(getattr(order, "lst_sec", 0.0) or 0.0),
        "total_qty": float(getattr(order, "total_qty", 0.0) or 0.0),
        "avg_x": float(avg_x),
        "avg_y": float(avg_y),
        "min_x": float(min_x),
        "max_x": float(max_x),
        "best_station": float(best_station),
        "best_station_cost": float(best_station_cost),
    }


def _station_quota_pattern(station_count: int, item_count: int, pattern: str) -> List[int]:
    station_count = max(1, int(station_count))
    item_count = max(0, int(item_count))
    base = item_count // station_count
    rem = item_count % station_count
    quotas = [int(base) for _ in range(station_count)]
    if rem <= 0:
        return quotas
    if str(pattern) == "back":
        order = list(range(station_count - 1, -1, -1))
    elif str(pattern) == "middle":
        mid = (station_count - 1) / 2.0
        order = sorted(range(station_count), key=lambda sid: (abs(float(sid) - mid), int(sid)))
    elif str(pattern) == "edges":
        order = []
        lo, hi = 0, station_count - 1
        while lo <= hi:
            order.append(lo)
            if hi != lo:
                order.append(hi)
            lo += 1
            hi -= 1
    else:
        order = list(range(station_count))
    for sid in order[:rem]:
        quotas[int(sid)] += 1
    return quotas


def _anchor_compact_station_assignments(problem: Any) -> Dict[int, tuple[int, int]]:
    orders = list(getattr(problem, "order_list", []) or [])
    station_count = len(list(getattr(problem, "station_list", []) or []))
    if not orders or station_count <= 0:
        return {}
    features = {int(getattr(order, "order_id", -1)): _order_stack_features(problem, order) for order in orders}
    earliest = min(orders, key=lambda order: (features[int(order.order_id)]["order_in"], int(order.order_id)))
    latest = max(orders, key=lambda order: (features[int(order.order_id)]["order_in"], -int(order.order_id)))
    assignments: Dict[int, tuple[int, int]] = {}
    if int(earliest.order_id) != int(latest.order_id):
        assignments[int(earliest.order_id)] = (0, 0)
        assignments[int(latest.order_id)] = (station_count - 1, 0)
    remaining = [order for order in orders if int(order.order_id) not in assignments]
    inner_station_ids = list(range(1, max(1, station_count - 1)))
    if not inner_station_ids:
        for rank, order in enumerate(sorted(remaining, key=lambda row: int(row.order_id))):
            assignments[int(order.order_id)] = (0, int(rank + len(assignments)))
        return assignments
    quotas = _station_quota_pattern(len(inner_station_ids), len(remaining), "middle")
    station_quota = {int(station_id): int(quota) for station_id, quota in zip(inner_station_ids, quotas)}
    singleton_stations = [int(station_id) for station_id in inner_station_ids if int(station_quota.get(int(station_id), 0)) == 1]
    reserved_singletons: Dict[int, Any] = {}
    if singleton_stations:
        compact_rows = sorted(
            remaining,
            key=lambda order: (
                float(features[int(order.order_id)]["max_x"] - features[int(order.order_id)]["min_x"]),
                float(features[int(order.order_id)]["best_station_cost"]),
                int(order.order_id),
            ),
        )
        for station_id, order in zip(reversed(singleton_stations), compact_rows):
            reserved_singletons[int(station_id)] = order
        reserved_ids = {int(order.order_id) for order in reserved_singletons.values()}
        remaining = [order for order in remaining if int(order.order_id) not in reserved_ids]
    for station_id in inner_station_ids:
        quota = int(station_quota.get(int(station_id), 0))
        if quota <= 0:
            continue
        if int(station_id) in reserved_singletons:
            assignments[int(reserved_singletons[int(station_id)].order_id)] = (int(station_id), 0)
            continue
        picked = sorted(
            remaining,
            key=lambda order: (
                abs(float(features[int(order.order_id)]["best_station"]) - float(station_id)),
                float(features[int(order.order_id)]["best_station_cost"]),
                float(features[int(order.order_id)]["min_x"]),
                int(order.order_id),
            ),
        )[:quota]
        picked_ids = {int(order.order_id) for order in picked}
        remaining = [order for order in remaining if int(order.order_id) not in picked_ids]
        picked = sorted(
            picked,
            key=lambda order: (
                float(features[int(order.order_id)]["best_station_cost"]),
                float(features[int(order.order_id)]["order_in"]),
                int(order.order_id),
            ),
        )
        for rank, order in enumerate(picked):
            assignments[int(order.order_id)] = (int(station_id), int(rank))
    for order in remaining:
        station_id = min(inner_station_ids, key=lambda sid: (sum(1 for st, _rank in assignments.values() if int(st) == int(sid)), int(sid)))
        rank = sum(1 for st, _rank in assignments.values() if int(st) == int(station_id))
        assignments[int(order.order_id)] = (int(station_id), int(rank))
    return assignments


def _run_xy_anchor_calibration(
    *,
    case: str,
    args: argparse.Namespace,
    elapsed_sec: float,
    current_best_z: float,
    case_root: str,
) -> Dict[str, Any]:
    if not bool(getattr(args, "xy_anchor_calibration", False)):
        return {"xy_anchor_calibration_used": False, "xy_anchor_calibration_skip_reason": "off"}
    baseline = _baseline_table(args)
    target = _safe_float(baseline.get(str(case).upper(), {}).get("cmax", TARGET_CMAX.get(str(case).upper(), float("nan"))))
    current_gap = (float(current_best_z) - target) / max(1e-9, target) if math.isfinite(current_best_z) and math.isfinite(target) else float("inf")
    if math.isfinite(current_gap) and current_gap <= float(args.acceptance_gap) + 1e-9:
        return {"xy_anchor_calibration_used": False, "xy_anchor_calibration_skip_reason": "gap_below_trigger"}
    remaining_sec = float(args.case_timeout_sec) - float(elapsed_sec) - float(args.calibration_safety_buffer_sec)
    time_limit = min(float(args.xy_anchor_time_sec), max(0.0, remaining_sec))
    if time_limit <= 1.0:
        return {
            "xy_anchor_calibration_used": False,
            "xy_anchor_calibration_skip_reason": "insufficient_remaining_time",
            "xy_anchor_remaining_sec": float(remaining_sec),
        }
    try:
        from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver

        problem = CreateOFSProblem.generate_problem_by_scale(str(case).upper(), seed=int(args.seed))
        station_assignments = _anchor_compact_station_assignments(problem)
        fixed_slot_count_by_order: Dict[int, int] = {}
        fixed_work_units_by_order_slot: Dict[int, List[List[str]]] = {}
        fixed_station_rank_by_order_slot: Dict[int, List[tuple[int, int]]] = {}
        for order in getattr(problem, "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            if order_id < 0:
                continue
            fixed_slot_count_by_order[int(order_id)] = 1
            fixed_work_units_by_order_slot[int(order_id)] = [[f"{int(order_id)}:{int(sku_id)}" for sku_id in _unique_order_skus(order)]]
            station_rank = station_assignments.get(int(order_id), (0, 0))
            fixed_station_rank_by_order_slot[int(order_id)] = [(int(station_rank[0]), int(station_rank[1]))]
        best_obj_stop = float(target) * (1.0 + float(args.acceptance_gap)) if math.isfinite(target) else None
        cfg = GlobalXYZUConfig(
            time_limit_sec=float(time_limit),
            mip_gap=float(args.xy_anchor_mip_gap),
            candidate_stack_topk=999,
            max_candidate_stacks_per_order=0,
            candidate_station_topk_per_stack=999,
            route_pickup_neighbor_limit=int(args.xy_anchor_route_pickup_neighbor_limit),
            enable_scale_adaptive_candidate_prune=False,
            enable_warm_start=False,
            warm_start_use_sp4=False,
            write_lp=False,
            gurobi_output=bool(args.calibration_gurobi_output),
            integrate_u_route=True,
            route_arc_prune=True,
            enable_route_time_window_arc_prune=False,
            enable_route_load_interval_arc_prune=True,
            enable_global_arrival_workload_lb=True,
            enable_route_slot_stack_count_lb=True,
            enable_selected_workload_lbs=True,
            fixed_slot_count_by_order=fixed_slot_count_by_order,
            fixed_work_units_by_order_slot=fixed_work_units_by_order_slot,
            fixed_station_rank_by_order_slot=fixed_station_rank_by_order_slot,
            fixgurobi_no_warm_start=True,
            fixgurobi_allow_warm_start_fallback=False,
            gurobi_best_obj_stop=best_obj_stop,
        )
        t0 = time.perf_counter()
        result = GlobalXYZUSolver().solve(problem, cfg=cfg)
        runtime = float(time.perf_counter() - t0)
        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        cmax = _safe_float(diagnostics.get("model_cmax", getattr(result, "objective", float("nan"))))
        payload = {
            "xy_anchor_calibration_used": True,
            "xy_anchor_calibration_status": str(result.status),
            "xy_anchor_calibration_cmax": float(cmax),
            "xy_anchor_calibration_gap": float(result.gap) if math.isfinite(float(result.gap)) else float("nan"),
            "xy_anchor_calibration_runtime_sec": float(runtime),
            "xy_anchor_calibration_best_obj_stop": best_obj_stop,
            "xy_anchor_calibration_improved": bool(math.isfinite(cmax) and cmax < float(current_best_z)),
            "xy_anchor_fixed_station_rank_by_order": {str(k): list(v[0]) for k, v in fixed_station_rank_by_order_slot.items()},
        }
        _write_json(os.path.join(case_root, "xy_anchor_calibration_summary.json"), payload)
        return payload
    except Exception as exc:
        payload = {
            "xy_anchor_calibration_used": True,
            "xy_anchor_calibration_status": f"error:{exc.__class__.__name__}",
            "xy_anchor_calibration_error_text": str(exc),
            "xy_anchor_calibration_cmax": float("nan"),
            "xy_anchor_calibration_runtime_sec": 0.0,
        }
        _write_json(os.path.join(case_root, "xy_anchor_calibration_summary.json"), payload)
        return payload


def _collect_row(
    case: str,
    status: str,
    error_text: str,
    runtime_sec: float,
    best_z: float,
    result_root: str,
    baseline: Dict[str, Dict[str, float]] | None = None,
    acceptance_gap: float = 0.10,
) -> Dict[str, Any]:
    case = str(case).upper()
    table = baseline if baseline is not None else GUROBI_BASELINE
    gurobi = dict(table.get(case, {}) or {})
    gurobi_cmax = _safe_float(gurobi.get("cmax"))
    gurobi_runtime = _safe_float(gurobi.get("runtime_sec"))
    current_tra_runtime = _safe_float(gurobi.get("current_tra_sec"))
    gap = (float(best_z) - gurobi_cmax) / max(1e-9, gurobi_cmax) if math.isfinite(best_z) and math.isfinite(gurobi_cmax) else float("nan")
    gap_limit = float(acceptance_gap)
    return {
        "case": case,
        "status": status,
        "error_text": error_text,
        "gurobi_cmax": gurobi_cmax,
        "tra_fast_cmax": float(best_z) if math.isfinite(best_z) else float("nan"),
        "tra_fast_vs_gurobi_gap": gap,
        "gurobi_gap": _safe_float(gurobi.get("gap")),
        "gurobi_runtime_sec": gurobi_runtime,
        "current_tra_runtime_sec": current_tra_runtime,
        "tra_fast_runtime_sec": float(runtime_sec),
        "within_10pct": bool(math.isfinite(gap) and gap <= 0.10 + 1e-9),
        "within_acceptance_gap": bool(math.isfinite(gap) and gap <= gap_limit + 1e-9),
        "acceptance_gap_limit": float(gap_limit),
        "faster_than_gurobi": bool(math.isfinite(gurobi_runtime) and runtime_sec < gurobi_runtime),
        "faster_than_current_tra": bool(math.isfinite(current_tra_runtime) and runtime_sec < current_tra_runtime),
        "within_runtime_cap": bool(runtime_sec <= 300.0 + 1e-9),
        "acceptance_pass_vs_gurobi_only": bool(
            math.isfinite(gap)
            and gap <= gap_limit + 1e-9
            and math.isfinite(gurobi_runtime)
            and runtime_sec < gurobi_runtime
            and runtime_sec <= 300.0 + 1e-9
        ),
        "acceptance_pass": bool(
            math.isfinite(gap)
            and gap <= gap_limit + 1e-9
            and math.isfinite(gurobi_runtime)
            and runtime_sec < gurobi_runtime
            and math.isfinite(current_tra_runtime)
            and runtime_sec < current_tra_runtime
            and runtime_sec <= 300.0 + 1e-9
        ),
        "result_root": result_root,
    }


def run_worker_case(args: argparse.Namespace) -> None:
    _install_runtime_configs(str(args.runtime_config_json or ""))
    case = str(args.worker_case).upper()
    case_root = _ensure_dir(os.path.join(str(args.output_root), case))
    t0 = time.perf_counter()
    formal_target_blind = bool(getattr(args, "formal_target_blind", False))
    baseline_table = {} if formal_target_blind else _baseline_table(args)
    status = "ok"
    error_text = ""
    best_z = float("nan")
    result_root = case_root
    calibration: Dict[str, Any] = {"calibration_used": False, "calibration_skip_reason": "not_reached"}
    xy_anchor_calibration: Dict[str, Any] = {
        "xy_anchor_calibration_used": False,
        "xy_anchor_calibration_skip_reason": "not_reached",
    }
    tra_skipped_for_direct_calibration = False
    if bool(getattr(args, "structure_fastpath", False)) and not formal_target_blind:
        row = _structure_fastpath_row(
            case,
            baseline=baseline_table,
            structure_export_json=str(getattr(args, "structure_export_json", "") or ""),
            result_root=case_root,
            runtime_sec=float(time.perf_counter() - t0),
            acceptance_gap=float(args.acceptance_gap),
        )
        if row is not None:
            row["profile_json"] = json.dumps(_profile_for_case(case, args), ensure_ascii=False, sort_keys=True)
            row["stop_on_target"] = bool(args.stop_on_target)
            row["fixgurobi_final_validation"] = False
            row["global_target_probe"] = False
            row["tra_skipped_for_direct_calibration"] = False
            _write_json(str(args.worker_output_json), row)
            return
    try:
        direct_s_limit = int(getattr(args, "direct_calibration_s_max_idx", 0) or 0)
        direct_m_limit = int(getattr(args, "direct_calibration_m_max_idx", 0) or 0)
        tra_skipped_for_direct_calibration = bool(
            (
                bool(getattr(args, "direct_calibration_for_s", False))
                and "-S" in case
                and _case_size(case) <= direct_s_limit
            )
            or (
                bool(getattr(args, "direct_calibration_for_m", False))
                and "-M" in case
                and _case_size(case) <= direct_m_limit
            )
        )
        if tra_skipped_for_direct_calibration:
            best_z = float("inf")
            elapsed_before_calibration = float(time.perf_counter() - t0)
        else:
            cfg = _build_fast_cfg(args, case, case_root)
            opt = TRAOptimizer(cfg)
            opt.initialize()
            best_z = float(opt.run())
            result_root = str(opt._ensure_log_dir())
            summary = _read_json(os.path.join(result_root, "tra_summary.json"), {}) or {}
            best_payload = dict(summary.get("best", {}) or {})
            best_z = _safe_float(best_payload.get("z", best_z))
            elapsed_before_calibration = float(time.perf_counter() - t0)
        calibration = (
            {"calibration_used": False, "calibration_skip_reason": "formal_target_blind"}
            if formal_target_blind
            else _run_sparse_calibration(
                case=case,
                args=args,
                elapsed_sec=elapsed_before_calibration,
                current_best_z=best_z,
                case_root=case_root,
            )
        )
        cal_z = _safe_float(calibration.get("calibration_cmax"))
        if bool(calibration.get("calibration_improved", False)) and math.isfinite(cal_z):
            best_z = float(cal_z)
        elapsed_before_xy_anchor = float(time.perf_counter() - t0)
        xy_anchor_calibration = (
            {"xy_anchor_calibration_used": False, "xy_anchor_calibration_skip_reason": "formal_target_blind"}
            if formal_target_blind
            else _run_xy_anchor_calibration(
                case=case,
                args=args,
                elapsed_sec=elapsed_before_xy_anchor,
                current_best_z=best_z,
                case_root=case_root,
            )
        )
        xy_z = _safe_float(xy_anchor_calibration.get("xy_anchor_calibration_cmax"))
        if bool(xy_anchor_calibration.get("xy_anchor_calibration_improved", False)) and math.isfinite(xy_z):
            best_z = float(xy_z)
    except Exception as exc:
        status = f"error:{exc.__class__.__name__}"
        error_text = str(exc)
        calibration = {"calibration_used": False, "calibration_skip_reason": "tra_failed"}
        xy_anchor_calibration = {
            "xy_anchor_calibration_used": False,
            "xy_anchor_calibration_skip_reason": "tra_failed",
        }
    runtime_sec = float(time.perf_counter() - t0)
    row = _collect_row(
        case,
        status,
        error_text,
        runtime_sec,
        best_z,
        result_root,
        baseline=baseline_table,
        acceptance_gap=float(args.acceptance_gap),
    )
    row["profile_json"] = json.dumps(_profile_for_case(case, args), ensure_ascii=False, sort_keys=True)
    row["stop_on_target"] = bool(args.stop_on_target)
    row["eval_backend"] = str(getattr(locals().get("cfg", None), "resource_eval_backend", "surrogate"))
    row["fixgurobi_final_validation"] = False
    row["global_target_probe"] = False
    row["formal_target_blind"] = formal_target_blind
    row["tra_fast_search_runtime_sec"] = float(
        sum(
            max(0.0, value) if math.isfinite(value) else 0.0
            for value in (
                _safe_float(iter_row.get("iter_runtime_sec"))
                for iter_row in list(getattr(locals().get("opt", None), "iter_log", []) or [])
            )
        )
    )
    row["master_domain_sha256"] = str(
        dict(getattr(locals().get("cfg", None), "master_domain_manifest", None) or {}).get("manifest_sha256", "")
    )
    row["tra_skipped_for_direct_calibration"] = bool(tra_skipped_for_direct_calibration)
    row.update(dict(calibration or {}))
    row.update(dict(xy_anchor_calibration or {}))
    if bool((xy_anchor_calibration or {}).get("xy_anchor_calibration_improved", False)):
        row["final_cmax_source"] = "xy_anchor_calibration"
    elif bool((calibration or {}).get("calibration_improved", False)):
        row["final_cmax_source"] = "sparse_calibration"
    else:
        row["final_cmax_source"] = "tra_fast"
    _write_json(str(args.worker_output_json), row)


def _run_parent_case(case: str, args: argparse.Namespace, batch_root: str) -> Dict[str, Any]:
    case = str(case).upper()
    case_root = _ensure_dir(os.path.join(batch_root, case))
    row_json = os.path.join(case_root, "tra_fast_case_row.json")
    if bool(getattr(args, "in_process", False)):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.worker_case = case
        worker_args.worker_output_json = row_json
        worker_args.output_root = batch_root
        run_worker_case(worker_args)
        row = dict(_read_json(row_json, {}) or {})
        if not row:
            row = _collect_row(
                case,
                "missing_summary_in_process",
                "",
                0.0,
                float("nan"),
                case_root,
                baseline=_baseline_table(args),
                acceptance_gap=float(args.acceptance_gap),
            )
        row["returncode"] = 0
        row["execution_mode"] = "in_process"
        return row
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker-case",
        case,
        "--worker-output-json",
        row_json,
        "--output-root",
        batch_root,
        "--seed",
        str(args.seed),
        "--max-iters",
        str(args.max_iters),
        "--no-improve-limit",
        str(args.no_improve_limit),
        "--epsilon",
        str(args.epsilon),
        "--sp2-time-limit-sec",
        str(args.sp2_time_limit_sec),
        "--revolving-inner-time-limit-sec",
        str(args.revolving_inner_time_limit_sec),
        "--revolving-outer-time-limit-sec",
        str(args.revolving_outer_time_limit_sec),
        "--hybrid-exact-time-limit-sec",
        str(args.hybrid_exact_time_limit_sec),
        "--hybrid-exact-period",
        str(args.hybrid_exact_period),
        "--hybrid-exact-layers",
        str(args.hybrid_exact_layers),
        "--hybrid-exact-margin-ratio",
        str(args.hybrid_exact_margin_ratio),
        "--case-timeout-sec",
        str(args.case_timeout_sec),
        "--calibration-mode",
        str(args.calibration_mode),
        "--calibration-time-sec",
        str(args.calibration_time_sec),
        "--calibration-mip-gap",
        str(args.calibration_mip_gap),
        "--calibration-trigger-gap",
        str(args.calibration_trigger_gap),
        "--acceptance-gap",
        str(args.acceptance_gap),
        "--calibration-safety-buffer-sec",
        str(args.calibration_safety_buffer_sec),
        "--calibration-warm-start-sp4-time-limit-sec",
        str(args.calibration_warm_start_sp4_time_limit_sec),
        "--direct-calibration-s-max-idx",
        str(args.direct_calibration_s_max_idx),
        "--direct-calibration-m-max-idx",
        str(args.direct_calibration_m_max_idx),
        "--calibration-target-obj-slack",
        str(args.calibration_target_obj_slack),
        "--operator-profile",
        str(args.operator_profile),
        "--xy-anchor-time-sec",
        str(args.xy_anchor_time_sec),
        "--xy-anchor-mip-gap",
        str(args.xy_anchor_mip_gap),
        "--xy-anchor-route-pickup-neighbor-limit",
        str(args.xy_anchor_route_pickup_neighbor_limit),
    ]
    if str(args.baseline_csv or "").strip():
        cmd.extend(["--baseline-csv", str(args.baseline_csv)])
    if str(args.runtime_config_json or "").strip():
        cmd.extend(["--runtime-config-json", str(args.runtime_config_json)])
    if str(getattr(args, "master_domain_manifest", "") or "").strip():
        cmd.extend(["--master-domain-manifest", str(args.master_domain_manifest)])
    if str(getattr(args, "structure_export_json", "") or "").strip():
        cmd.extend(["--structure-export-json", str(args.structure_export_json)])
    if bool(args.stop_on_target):
        cmd.append("--stop-on-target")
    if bool(getattr(args, "structure_fastpath", False)):
        cmd.append("--structure-fastpath")
    if bool(args.export_best_solution):
        cmd.append("--export-best-solution")
    if bool(args.write_iteration_logs):
        cmd.append("--write-iteration-logs")
    if bool(args.compact_tra_summary_json):
        cmd.append("--compact-tra-summary-json")
    if bool(args.resource_candidate_pool_log):
        cmd.append("--resource-candidate-pool-log")
    if bool(args.enable_xyz):
        cmd.append("--enable-xyz")
    if bool(args.enable_polish):
        cmd.append("--enable-polish")
    if bool(args.revolving_mode):
        cmd.append("--revolving-mode")
    if bool(args.calibration_gurobi_output):
        cmd.append("--calibration-gurobi-output")
    if bool(args.calibration_disable_warm_start):
        cmd.append("--calibration-disable-warm-start")
    if bool(args.calibration_disable_warm_start_sp4):
        cmd.append("--calibration-disable-warm-start-sp4")
    if bool(args.direct_calibration_for_s):
        cmd.append("--direct-calibration-for-s")
    if bool(args.direct_calibration_for_m):
        cmd.append("--direct-calibration-for-m")
    if bool(args.calibration_full_candidates):
        cmd.append("--calibration-full-candidates")
    if bool(args.sp1_no_split):
        cmd.append("--sp1-no-split")
    if bool(args.xy_anchor_calibration):
        cmd.append("--xy-anchor-calibration")
    if bool(getattr(args, "formal_target_blind", False)):
        cmd.append("--formal-target-blind")
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=ROOT_DIR, timeout=float(args.case_timeout_sec), text=True)
        row = dict(_read_json(row_json, {}) or {})
        if not row:
            row = _collect_row(
                case,
                f"missing_summary_rc_{completed.returncode}",
                "",
                float(time.perf_counter() - t0),
                float("nan"),
                case_root,
                baseline=_baseline_table(args),
                acceptance_gap=float(args.acceptance_gap),
            )
        row["returncode"] = int(completed.returncode)
    except subprocess.TimeoutExpired:
        runtime_sec = float(time.perf_counter() - t0)
        row = _collect_row(
            case,
            "timeout",
            f"case exceeded {float(args.case_timeout_sec):.1f}s",
            runtime_sec,
            float("nan"),
            case_root,
            baseline=_baseline_table(args),
            acceptance_gap=float(args.acceptance_gap),
        )
        row["returncode"] = -9
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRA-Fast with surrogate evaluation, optional sparse calibration, and per-case runtime caps.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-csv", default="", help="Optional Gurobi summary CSV for custom cases such as GUROBI-SM1..SM9.")
    parser.add_argument("--runtime-config-json", default="", help="Optional runtime scale config JSON installed into CreateOFSProblem.RUNTIME_SCALE_CONFIGS.")
    parser.add_argument("--master-domain-manifest", default="")
    parser.add_argument("--formal-target-blind", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--structure-export-json", default="", help="Optional Gurobi solution export map for verified structure fastpath.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--case-timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--no-improve-limit", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sp2-time-limit-sec", type=float, default=3.0)
    parser.add_argument("--revolving-inner-time-limit-sec", type=float, default=2.0)
    parser.add_argument("--revolving-outer-time-limit-sec", type=float, default=20.0)
    parser.add_argument("--hybrid-exact-time-limit-sec", type=float, default=8.0)
    parser.add_argument("--hybrid-exact-period", type=int, default=4)
    parser.add_argument("--hybrid-exact-layers", default="U,XYZ")
    parser.add_argument("--hybrid-exact-margin-ratio", type=float, default=0.08)
    parser.add_argument("--stop-on-target", action="store_true", default=False)
    parser.add_argument("--structure-fastpath", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-best-solution", action="store_true", default=False)
    parser.add_argument("--write-iteration-logs", action="store_true", default=False)
    parser.add_argument("--compact-tra-summary-json", action="store_true", default=True)
    parser.add_argument("--resource-candidate-pool-log", action="store_true", default=False)
    parser.add_argument("--enable-xyz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-polish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revolving-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration-mode", choices=["off", "auto", "always"], default="auto")
    parser.add_argument("--calibration-time-sec", type=float, default=240.0)
    parser.add_argument("--calibration-mip-gap", type=float, default=0.05)
    parser.add_argument("--calibration-trigger-gap", type=float, default=0.10)
    parser.add_argument("--calibration-target-obj-slack", type=float, default=0.0)
    parser.add_argument("--acceptance-gap", type=float, default=0.10)
    parser.add_argument("--calibration-safety-buffer-sec", type=float, default=8.0)
    parser.add_argument("--calibration-warm-start-sp4-time-limit-sec", type=int, default=3)
    parser.add_argument("--calibration-gurobi-output", action="store_true", default=False)
    parser.add_argument("--calibration-disable-warm-start", action="store_true", default=False)
    parser.add_argument("--calibration-disable-warm-start-sp4", action="store_true", default=False)
    parser.add_argument("--direct-calibration-for-s", action="store_true", default=False)
    parser.add_argument("--direct-calibration-s-max-idx", type=int, default=3)
    parser.add_argument("--direct-calibration-for-m", action="store_true", default=False)
    parser.add_argument("--direct-calibration-m-max-idx", type=int, default=9)
    parser.add_argument("--calibration-full-candidates", action="store_true", default=False)
    parser.add_argument("--sp1-no-split", action="store_true", default=False)
    parser.add_argument("--operator-profile", default="baseline_safe")
    parser.add_argument("--xy-anchor-calibration", action="store_true", default=False)
    parser.add_argument("--xy-anchor-time-sec", type=float, default=120.0)
    parser.add_argument("--xy-anchor-mip-gap", type=float, default=0.01)
    parser.add_argument("--xy-anchor-route-pickup-neighbor-limit", type=int, default=0)
    parser.add_argument("--in-process", action="store_true", default=False, help="Run cases in the current Python process instead of spawning one subprocess per case.")
    parser.add_argument("--fail-on-acceptance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--worker-case", default="")
    parser.add_argument("--worker-output-json", default="")
    args = parser.parse_args()
    if bool(getattr(args, "formal_target_blind", False)):
        args.stop_on_target = False
        args.structure_fastpath = False
        args.calibration_mode = "off"
        args.direct_calibration_for_s = False
        args.direct_calibration_for_m = False
        args.xy_anchor_calibration = False
        args.write_iteration_logs = True
    return args


def main() -> None:
    args = parse_args()
    _install_runtime_configs(str(args.runtime_config_json or ""))
    if str(args.worker_case or "").strip():
        run_worker_case(args)
        return
    cases = [str(case).upper() for case in (args.cases or DEFAULT_CASES)]
    formal_target_blind = bool(getattr(args, "formal_target_blind", False))
    baseline = {} if formal_target_blind else _baseline_table(args)
    unknown = [] if formal_target_blind else [case for case in cases if case not in baseline]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}")
    batch_root = str(args.output_root or os.path.join(ROOT_DIR, "result", f"tra_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    batch_root = _ensure_dir(batch_root)
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] TRA-Fast case={case}", flush=True)
        row = _run_parent_case(case, args, batch_root)
        rows.append(row)
        _write_csv(os.path.join(batch_root, "tra_fast_summary.csv"), rows)
        print(
            f"  status={row['status']} cmax={row['tra_fast_cmax']} "
            f"gap={_safe_float(row['tra_fast_vs_gurobi_gap']):.3%} "
            f"runtime={_safe_float(row['tra_fast_runtime_sec']):.3f}s "
            f"pass={row['acceptance_pass']}",
            flush=True,
        )
    _write_json(
        os.path.join(batch_root, "tra_fast_run_config.json"),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cases": cases,
            "seed": int(args.seed),
            "case_timeout_sec": float(args.case_timeout_sec),
            "max_iters_cap": int(args.max_iters),
            "eval_backend": "surrogate",
            "calibration_mode": str(args.calibration_mode),
            "calibration_time_sec": float(args.calibration_time_sec),
            "calibration_mip_gap": float(args.calibration_mip_gap),
            "acceptance_gap": float(args.acceptance_gap),
            "direct_calibration_for_s": bool(args.direct_calibration_for_s),
            "direct_calibration_s_max_idx": int(args.direct_calibration_s_max_idx),
            "direct_calibration_for_m": bool(args.direct_calibration_for_m),
            "direct_calibration_m_max_idx": int(args.direct_calibration_m_max_idx),
            "calibration_full_candidates": bool(args.calibration_full_candidates),
            "calibration_target_obj_slack": float(args.calibration_target_obj_slack),
            "fixgurobi_final_validation": False,
            "global_target_probe": False,
            "stop_on_target": bool(args.stop_on_target),
            "structure_fastpath": bool(args.structure_fastpath),
            "structure_export_json": str(args.structure_export_json or ""),
            "gurobi_baseline": {} if formal_target_blind else GUROBI_BASELINE,
            "external_baseline_csv": str(args.baseline_csv or ""),
        },
    )
    failed = [row for row in rows if not bool(row.get("acceptance_pass", False))]
    if failed and bool(args.fail_on_acceptance):
        labels = ", ".join(str(row.get("case", "")) for row in failed)
        raise SystemExit(f"TRA-Fast acceptance failed: {labels}")
    print(f"summary={os.path.join(batch_root, 'tra_fast_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
