import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.tra import TRAOptimizer, TRARunConfig
from Gurobi.resource_time_alns.diagnostics import compare_solution_signatures, write_structure_compare_csv


DEFAULT_ALL_CASES = [
    "GUROBI-S1",
    "GUROBI-S2",
    "GUROBI-S3",
    "GUROBI-S4",
    "GUROBI-S5",
    "GUROBI-S6",
    "GUROBI-S7",
    "GUROBI-S8",
    "GUROBI-S9",
    #  "SMALL",
    # "SMALL2",
    # "SMALL_ZRICH",
    # "SMALL2_ZRICH",
    # "SMALL3",
    # "SMALL_UNEVEN",
    # "SMALL2_UNEVEN",
]

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
}

KNOWN_GUROBI_EXPORT_DIRS = {
    "GUROBI-S1": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s1_20260520_231530", "gurobi_solution_export"),
    "GUROBI-S2": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s2_20260520_231546", "gurobi_solution_export"),
    "GUROBI-S3": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s3_20260520_231605", "gurobi_solution_export"),
    "GUROBI-S4": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s4_20260520_231645", "gurobi_solution_export"),
    "GUROBI-S5": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s5_20260520_231825", "gurobi_solution_export"),
    "GUROBI-S6": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s6_20260520_231843", "gurobi_solution_export"),
    "GUROBI-S7": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s7_20260521_000507", "gurobi_solution_export"),
    "GUROBI-S8": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s8_20260520_232208", "gurobi_solution_export"),
    "GUROBI-S9": os.path.join(ROOT_DIR, "result", "gurobi_gurobi-s9_20260520_225718", "gurobi_solution_export"),
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_txt(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")


def _case_target_cmax(scale: str) -> float:
    targets = {
        "GUROBI-S1": 90.0,
        "GUROBI-S2": 164.0,
        "GUROBI-S3": 222.0,
        "GUROBI-S4": 237.0,
        "GUROBI-S5": 275.0,
        "GUROBI-S6": 299.0,
        "GUROBI-S7": 361.0,
        "GUROBI-S8": 366.0,
        "GUROBI-S9": 438.0,
    }
    return float(targets.get(str(scale).upper(), float("nan")))


def _case_current_baseline_cmax(scale: str) -> float:
    return float(CURRENT_TRA_BASELINE_CMAX.get(str(scale).upper(), float("nan")))


def _gurobi_export_dir(args, scale: str) -> str:
    case_name = str(scale).upper()
    known = KNOWN_GUROBI_EXPORT_DIRS.get(case_name)
    if known and os.path.exists(known):
        return known
    root = str(args.gurobi_s6_reference_root if case_name == "GUROBI-S6" else args.gurobi_reference_root)
    candidates = [
        os.path.join(root, case_name, "gurobi_solution_export"),
        os.path.join(root, "gurobi_solution_export"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    return value


def _collapse_values(values: List[Any]) -> Any:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    encoded = [json.dumps(_normalize_jsonable(value), ensure_ascii=False, sort_keys=True) for value in cleaned]
    if len(set(encoded)) == 1:
        return cleaned[0]
    return json.dumps([_normalize_jsonable(value) for value in cleaned], ensure_ascii=False)


def _per_robot_path_lengths(opt: TRAOptimizer) -> Dict[str, float]:
    tasks = list(opt._collect_all_tasks() or [])
    robots = list(getattr(opt.problem, "robot_list", []) or [])
    robot_map = {int(getattr(r, "id", -1)): r for r in robots}
    events_by_robot: Dict[int, List[Any]] = {int(getattr(r, "id", idx)): [] for idx, r in enumerate(robots)}
    for task in tasks:
        rid = int(getattr(task, "robot_id", -1))
        if rid < 0:
            continue
        stack_obj = opt.problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1)))
        if stack_obj is not None and getattr(stack_obj, "store_point", None) is not None:
            events_by_robot.setdefault(rid, []).append((
                float(getattr(task, "arrival_time_at_stack", 0.0)),
                int(stack_obj.store_point.x),
                int(stack_obj.store_point.y),
            ))
        sid = int(getattr(task, "target_station_id", -1))
        stations = list(getattr(opt.problem, "station_list", []) or [])
        if 0 <= sid < len(stations):
            pt = stations[sid].point
            events_by_robot.setdefault(rid, []).append((
                float(getattr(task, "arrival_time_at_station", 0.0)),
                int(pt.x),
                int(pt.y),
            ))
    out: Dict[str, float] = {}
    for rid, robot in sorted(robot_map.items(), key=lambda item: item[0]):
        events = list(events_by_robot.get(rid, []) or [])
        if getattr(robot, "start_point", None) is None:
            out[str(rid)] = 0.0
            continue
        events.sort(key=lambda x: x[0])
        x0 = int(robot.start_point.x)
        y0 = int(robot.start_point.y)
        last_x, last_y = x0, y0
        total = 0.0
        for _, x, y in events:
            total += abs(x - last_x) + abs(y - last_y)
            last_x, last_y = x, y
        total += abs(last_x - x0) + abs(last_y - y0)
        out[str(rid)] = float(total)
    return out


def _collect_init_metrics(opt: TRAOptimizer) -> Dict[str, Any]:
    metrics = dict(opt._collect_layer_metrics() or {})
    station_loads = {str(idx): 0 for idx, _ in enumerate(getattr(opt.problem, "station_list", []) or [])}
    for sid, cnt in dict(opt._current_station_subtask_counts() or {}).items():
        station_loads[str(int(sid))] = int(cnt)
    order_unique_sku_counts = [
        int(len(getattr(order, "unique_sku_list", []) or []))
        for order in (getattr(opt.problem, "order_list", []) or [])
    ]
    robot_path_lengths = _per_robot_path_lengths(opt)
    return {
        "initial_makespan": float(opt.best.z if opt.best is not None else float("nan")),
        "initial_task_count": int(len(getattr(opt.problem, "task_list", []) or [])),
        "initial_subtask_count": int(len(getattr(opt.problem, "subtask_list", []) or [])),
        "initial_station_loads": station_loads,
        "initial_robot_path_lengths": robot_path_lengths,
        "initial_robot_path_length_total": float(sum(robot_path_lengths.values())),
        "bom_unique_sku_counts": order_unique_sku_counts,
        "bom_unique_sku_total": int(sum(order_unique_sku_counts)),
        "bom_unique_sku_avg_per_order": float(sum(order_unique_sku_counts) / len(order_unique_sku_counts)) if order_unique_sku_counts else float("nan"),
        "initial_station_load_max": float(metrics.get("station_load_max", 0.0)),
        "initial_station_load_std": float(metrics.get("station_load_std", 0.0)),
    }


def _build_cfg(args, scale: str, seed: int, run_log_dir: str) -> TRARunConfig:
    scale_idx = 0
    try:
        scale_idx = int(str(scale).upper().split("-S")[-1])
    except Exception:
        scale_idx = 0
    sp4_limit = int(args.sp4_lkh_time_limit_seconds)
    if scale_idx >= 5 and sp4_limit <= 5:
        sp4_limit = 30
    cfg = TRARunConfig(
        scale=str(scale).upper(),
        seed=int(seed),
        max_iters=int(args.max_iters),
        no_improve_limit=int(args.no_improve_limit),
        epsilon=float(args.epsilon),
        bom_arrival_window_sec=float(args.bom_arrival_window_sec),
        enable_order_time_windows=not bool(args.disable_order_time_windows),
        kitting_span_penalty_weight=float(args.kitting_span_penalty_weight),
        deadline_penalty_weight=float(args.deadline_penalty_weight),
        sp2_time_limit_sec=float(args.sp2_time_limit_sec),
        sp3_use_mip=bool(args.sp3_use_mip),
        sp4_lkh_time_limit_seconds=int(sp4_limit),
        sp4_first_solution_strategies=tuple(str(x).strip().upper() for x in str(args.sp4_first_solution_strategies).split(",") if str(x).strip()),
        sp4_first_solution_slice_seconds=int(args.sp4_first_solution_slice_seconds),
        sp4_enable_guided_local_search=bool(args.sp4_enable_guided_local_search),
        sp4_same_subtask_vehicle_mode=str(args.sp4_same_subtask_vehicle_mode),
        sp4_same_subtask_vehicle_threshold=int(args.sp4_same_subtask_vehicle_threshold),
        sp4_enable_greedy_fallback=bool(args.sp4_enable_greedy_fallback),
        sp4_raise_on_no_solution=bool(args.sp4_raise_on_no_solution),
        export_best_solution=bool(args.export_best_solution),
        write_iteration_logs=bool(args.write_iteration_logs),
        enable_sp1_feedback_analysis=False,
        log_dir=run_log_dir,
        xz_evaluator_mode="classic_soft",
        search_scheme="resource_time_alns",
        resource_real_eval_period=int(args.resource_real_eval_period),
    )
    cfg.resource_candidate_pool_size = int(args.resource_candidate_pool_size)
    cfg.resource_candidate_pool_max_attempts = int(args.resource_candidate_pool_max_attempts)
    cfg.resource_exact_candidate_trial_limit = int(args.resource_exact_candidate_trial_limit)
    cfg.resource_stop_if_best_z_no_change_rounds = int(args.resource_stop_if_best_z_no_change_rounds)
    cfg.resource_stop_if_validated_best_no_change_rounds = int(args.resource_stop_if_validated_best_no_change_rounds)
    cfg.z_f0_topk = int(args.z_f0_topk)
    cfg.z_f1_topk = int(args.z_f1_topk)
    cfg.z_f2_topk = int(args.z_f2_topk)
    cfg.z_eval_all_candidates = bool(args.z_eval_all_candidates)
    cfg.enable_z_positive_mining_verify = bool(args.enable_z_positive_mining_verify)
    cfg.resource_multi_start_count = int(args.resource_multi_start_count)
    cfg.resource_multi_start_patience = int(args.resource_multi_start_patience)
    cfg.resource_enable_xyz_operator = bool(args.resource_enable_xyz_operator)
    cfg.resource_enable_critical_path_xyz = bool(args.resource_enable_critical_path_xyz)
    cfg.resource_enable_best_y_assignment_polish = bool(args.resource_enable_best_y_assignment_polish)
    cfg.resource_enable_best_z_sortify_polish = bool(args.resource_enable_best_z_sortify_polish)
    cfg.resource_y_polish_candidate_limit = int(args.resource_y_polish_candidate_limit)
    cfg.resource_xyz_stagnation_gate = not bool(args.disable_resource_xyz_stagnation_gate)
    cfg.resource_assert_sp4_ortools_only = not bool(args.allow_sp4_mip)
    if bool(cfg.resource_assert_sp4_ortools_only):
        cfg.sp4_use_mip = False
        cfg.exact_sp4_use_mip = False
    cfg.exact_sp4_lkh_time_limit_seconds = int(args.exact_sp4_lkh_time_limit_seconds)
    cfg.resource_eval_backend = str(args.resource_eval_backend)
    cfg.fixgurobi_time_limit_sec = float(args.fixgurobi_time_limit_sec)
    cfg.fixgurobi_mip_gap = float(args.fixgurobi_mip_gap)
    cfg.fixgurobi_candidate_trial_limit = int(args.fixgurobi_candidate_trial_limit)
    cfg.fixgurobi_cache_size = int(args.fixgurobi_cache_size)
    cfg.fixgurobi_fix_used_stack_ids = bool(args.fixgurobi_fix_used_stack_ids)
    cfg.fixgurobi_output = bool(args.fixgurobi_output)
    profile = str(args.operator_profile or "baseline_safe").strip().lower()
    cfg.resource_operator_profile = profile
    if profile == "critical_xyz_expanded":
        cfg.resource_enable_xyz_operator = True
        cfg.resource_enable_critical_path_xyz = True
        cfg.resource_enable_experimental_z_joint_polish = True
        cfg.resource_enable_single_flip_sortify = True
        cfg.resource_enable_best_sortify_polish = True
        cfg.resource_enable_best_rank_sortify_polish = True
        cfg.resource_candidate_pool_size = max(int(cfg.resource_candidate_pool_size), 10)
        cfg.resource_exact_candidate_trial_limit = max(int(cfg.resource_exact_candidate_trial_limit), 10)
        cfg.resource_xyz_candidate_pool_size = max(int(cfg.resource_xyz_candidate_pool_size), 6)
        cfg.resource_xyz_exact_candidate_trial_limit = max(int(cfg.resource_xyz_exact_candidate_trial_limit), 6)
        cfg.resource_stop_if_best_z_no_change_rounds = max(int(cfg.resource_stop_if_best_z_no_change_rounds), 80)
        cfg.resource_stop_if_validated_best_no_change_rounds = max(int(cfg.resource_stop_if_validated_best_no_change_rounds), 80)
        cfg.x_repartition_beam_width = max(int(cfg.x_repartition_beam_width), 10)
        cfg.resource_z_candidate_stack_topk = max(int(cfg.resource_z_candidate_stack_topk), 8)
    if profile == "route_polish_exact":
        cfg.resource_enable_xyz_operator = False
        cfg.resource_enable_critical_path_xyz = False
        cfg.resource_enable_experimental_z_joint_polish = False
        cfg.resource_enable_best_y_assignment_polish = True
        cfg.resource_enable_best_z_sortify_polish = True
        cfg.resource_enable_best_sortify_polish = True
        cfg.resource_enable_best_rank_sortify_polish = True
        cfg.sp4_lkh_time_limit_seconds = max(int(cfg.sp4_lkh_time_limit_seconds), int(args.route_polish_sp4_lkh_time_limit_seconds))
        cfg.exact_sp4_lkh_time_limit_seconds = max(int(cfg.exact_sp4_lkh_time_limit_seconds), int(args.route_polish_sp4_lkh_time_limit_seconds))
        cfg.resource_real_eval_period = max(1, min(int(cfg.resource_real_eval_period), 4))
    if str(cfg.resource_eval_backend).strip().lower() == "fixgurobi_prefix":
        limit = max(1, int(cfg.fixgurobi_candidate_trial_limit))
        cfg.resource_candidate_pool_size = min(int(cfg.resource_candidate_pool_size), limit)
        cfg.resource_exact_candidate_trial_limit = min(int(cfg.resource_exact_candidate_trial_limit), limit)
        cfg.resource_xyz_candidate_pool_size = min(int(cfg.resource_xyz_candidate_pool_size), limit)
        cfg.resource_xyz_exact_candidate_trial_limit = min(int(cfg.resource_xyz_exact_candidate_trial_limit), limit)
    return cfg


def _run_variant(
    args,
    scale: str,
    run_idx: int,
    seed: int,
    batch_root: str,
    variant_name: str,
    force_sp3_mip: bool | None = None,
    emit_artifacts: bool = True,
) -> Dict[str, Any]:
    case_root = _ensure_dir(os.path.join(batch_root, str(scale).upper()))
    run_root = os.path.join(case_root, f"run_{run_idx:03d}_seed_{seed}", str(variant_name))
    t0 = time.perf_counter()
    status = "ok"
    best_z = float("nan")
    result_root = run_root
    audit = {}
    summary = {}
    init_metrics: Dict[str, Any] = {}
    error_text = ""

    try:
        cfg = _build_cfg(args, scale=scale, seed=seed, run_log_dir=run_root)
        if force_sp3_mip is not None:
            cfg.sp3_use_mip = bool(force_sp3_mip)
        if not bool(emit_artifacts):
            cfg.export_best_solution = False
            cfg.write_iteration_logs = False
        opt = TRAOptimizer(cfg)
        opt.initialize()
        init_metrics = _collect_init_metrics(opt)
        best_z = float(opt.run())
        result_root = opt._ensure_log_dir()
        summary = _read_json(os.path.join(result_root, "tra_summary.json"), {}) or {}
        audit = _read_json(os.path.join(result_root, "best_solution_export", "best_solution_audit.json"), {}) or {}
    except Exception as exc:
        status = f"error:{exc.__class__.__name__}"
        error_text = str(exc)

    runtime_sec = float(time.perf_counter() - t0)
    run_stats = dict((summary or {}).get("run_stats", {}) or {})
    best_row = dict((summary or {}).get("best", {}) or {})
    config_row = dict((summary or {}).get("config", {}) or {})
    iter_rows = list((summary or {}).get("iters", []) or [])
    layer_selected = {name: 0 for name in ["X", "Y", "Z", "XYZ"]}
    layer_accepted = {name: 0 for name in ["X", "Y", "Z", "XYZ"]}
    for iter_row in iter_rows:
        layer_name = str(iter_row.get("selected_resource_layer", iter_row.get("focus", "")) or "").upper()
        if layer_name in layer_selected:
            layer_selected[layer_name] = int(layer_selected[layer_name]) + 1
            if bool(iter_row.get("local_accept", False)):
                layer_accepted[layer_name] = int(layer_accepted[layer_name]) + 1
    fix_rows = [row for row in iter_rows if str(row.get("eval_backend", "")) == "fixgurobi_prefix"]
    last_fix_row = dict(fix_rows[-1]) if fix_rows else {}
    initial_makespan = _safe_float(init_metrics.get("initial_makespan", float("nan")))
    summary_best_z = _safe_float(best_row.get("z", float("nan")))
    run_best_z = _safe_float(best_z)
    best_z_value = summary_best_z if math.isfinite(summary_best_z) else run_best_z
    best_iter_raw = best_row.get("iter_id", -1)
    best_iter = -1 if best_iter_raw is None else int(best_iter_raw)
    if status == "ok" and not math.isfinite(best_z_value):
        status = "no_feasible"
        if not error_text:
            error_text = "no finite feasible incumbent"
    improvement_ratio = float("nan")
    if math.isfinite(initial_makespan) and initial_makespan > 0.0 and math.isfinite(best_z_value):
        improvement_ratio = float((initial_makespan - best_z_value) / initial_makespan)

    return {
        "scale": str(scale).upper(),
        "run_idx": int(run_idx),
        "seed": int(seed),
        "status": status,
        "error_text": error_text,
        "runtime_sec": runtime_sec,
        "best_z": best_z_value,
        "true_makespan": _safe_float(run_stats.get("best_validated_true_makespan", summary.get("best", {}).get("true_global_makespan", float("nan")))),
        "total_fulfillment_time": _safe_float(summary.get("best", {}).get("total_fulfillment_time", float("nan"))),
        "best_iter": int(best_iter),
        "improvement_ratio": improvement_ratio,
        "global_eval_count": int(run_stats.get("global_eval_count", 0) or 0),
        "lkh_call_count": int(run_stats.get("lkh_call_count", 0) or 0),
        "fallback_count": int(run_stats.get("fallback_count", 0) or 0),
        "catastrophic_rollback_count": int(run_stats.get("catastrophic_rollback_count", 0) or 0),
        "coverage_hard_reject_count": int(run_stats.get("coverage_hard_reject_count", 0) or 0),
        "exact_eval_cache_hit_count": int(run_stats.get("exact_eval_cache_hit_count", 0) or 0),
        "x_failure_decapitation_count": int(run_stats.get("x_failure_decapitation_count", 0) or 0),
        "stop_reason": str(run_stats.get("stop_reason", "") or ""),
        "resource_real_eval_period": int(config_row.get("resource_real_eval_period", run_stats.get("resource_real_eval_period", 0)) or 0),
        "resource_eval_backend": str(args.resource_eval_backend),
        "fixgurobi_eval_count": int(len(fix_rows)),
        "fixgurobi_last_status": str(last_fix_row.get("fixgurobi_status", "")),
        "fixgurobi_last_obj": _safe_float(last_fix_row.get("fixgurobi_obj", float("nan"))),
        "fixgurobi_last_gap": _safe_float(last_fix_row.get("fixgurobi_gap", float("nan"))),
        "fixgurobi_last_solve_time": _safe_float(last_fix_row.get("fixgurobi_solve_time", float("nan"))),
        "fixgurobi_last_fixed_scope": str(last_fix_row.get("fixgurobi_fixed_scope", "")),
        "layer_selected_x": int(layer_selected["X"]),
        "layer_selected_y": int(layer_selected["Y"]),
        "layer_selected_z": int(layer_selected["Z"]),
        "layer_selected_xyz": int(layer_selected["XYZ"]),
        "layer_accepted_x": int(layer_accepted["X"]),
        "layer_accepted_y": int(layer_accepted["Y"]),
        "layer_accepted_z": int(layer_accepted["Z"]),
        "layer_accepted_xyz": int(layer_accepted["XYZ"]),
        "coverage_ok": bool(audit.get("coverage_ok", False)),
        "unmet_sku_total": int(audit.get("unmet_sku_total", 0) or 0),
        "makespan_consistent": bool(audit.get("makespan_consistent", False)),
        "has_unreasonable_solution": bool(audit.get("has_unreasonable_solution", False)),
        "result_root": result_root,
        "variant_name": str(variant_name),
        "variant_sp3_use_mip": bool(force_sp3_mip if force_sp3_mip is not None else bool(args.sp3_use_mip)),
        **init_metrics,
    }


def _run_one(args, scale: str, run_idx: int, seed: int, batch_root: str) -> Dict[str, Any]:
    portfolio_modes = ["heuristic"]
    if bool(args.enable_sp3_init_portfolio):
        portfolio_modes.append("mip")
    variant_rows: List[Dict[str, Any]] = []
    for variant_name in portfolio_modes:
        force_sp3_mip = None
        if str(variant_name) == "heuristic":
            force_sp3_mip = False
        elif str(variant_name) == "mip":
            force_sp3_mip = True
        variant_rows.append(
            _run_variant(
                args,
                scale,
                run_idx,
                seed,
                batch_root,
                variant_name,
                force_sp3_mip=force_sp3_mip,
                emit_artifacts=False,
            )
        )

    ok_rows = [row for row in variant_rows if str(row.get("status", "")).lower() == "ok" and math.isfinite(float(row.get("best_z", float("nan"))))]
    if ok_rows:
        best_row = min(ok_rows, key=lambda row: (float(row.get("best_z", float("inf"))), float(row.get("runtime_sec", float("inf")))))
    else:
        best_row = variant_rows[0]
    best_variant_name = str(best_row.get("variant_name", "heuristic"))
    best_force_sp3_mip = False if best_variant_name == "heuristic" else True if best_variant_name == "mip" else None
    final_row = _run_variant(
        args,
        scale,
        run_idx,
        seed,
        batch_root,
        best_variant_name,
        force_sp3_mip=best_force_sp3_mip,
        emit_artifacts=True,
    )
    best_row = dict(final_row)
    best_row["portfolio_variant_count"] = int(len(variant_rows))
    best_row["portfolio_variants"] = json.dumps(
        [
            {
                "variant_name": str(row.get("variant_name", "")),
                "status": str(row.get("status", "")),
                "best_z": _safe_float(row.get("best_z", float("nan"))),
                "runtime_sec": _safe_float(row.get("runtime_sec", float("nan"))),
            }
            for row in variant_rows
        ],
        ensure_ascii=False,
    )
    target_cmax = _case_target_cmax(scale)
    current_baseline = _case_current_baseline_cmax(scale)
    best_value = float(best_row.get("best_z", float("nan")))
    best_row["target_cmax"] = target_cmax
    best_row["delta_to_target"] = float(best_value - target_cmax) if math.isfinite(target_cmax) and math.isfinite(best_value) else float("nan")
    best_row["current_baseline_cmax"] = current_baseline
    best_row["delta_to_current_baseline"] = float(best_value - current_baseline) if math.isfinite(current_baseline) and math.isfinite(best_value) else float("nan")
    best_row["not_worse_than_current_baseline"] = bool(
        (not bool(args.disable_current_baseline_gate))
        and math.isfinite(current_baseline)
        and math.isfinite(best_value)
        and best_value <= current_baseline + 1e-9
    )
    if bool(args.disable_current_baseline_gate):
        best_row["not_worse_than_current_baseline"] = ""
    return best_row


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scale in sorted({str(row.get("scale", "")).upper() for row in rows}):
        scale_rows = [row for row in rows if str(row.get("scale", "")).upper() == scale]
        ok_rows = [row for row in scale_rows if str(row.get("status", "")).lower() == "ok"]
        best_values = [float(row["best_z"]) for row in ok_rows if math.isfinite(float(row.get("best_z", float("nan"))))]
        runtime_values = [float(row["runtime_sec"]) for row in ok_rows]
        init_makespan_values = [float(row["initial_makespan"]) for row in ok_rows if math.isfinite(float(row.get("initial_makespan", float("nan"))))]
        improvement_values = [float(row["improvement_ratio"]) for row in ok_rows if math.isfinite(float(row.get("improvement_ratio", float("nan"))))]
        init_task_values = [int(row.get("initial_task_count", 0) or 0) for row in ok_rows]
        init_subtask_values = [int(row.get("initial_subtask_count", 0) or 0) for row in ok_rows]
        init_station_loads = [row.get("initial_station_loads") for row in ok_rows if row.get("initial_station_loads") is not None]
        init_robot_paths = [row.get("initial_robot_path_lengths") for row in ok_rows if row.get("initial_robot_path_lengths") is not None]
        init_robot_path_total_values = [float(row["initial_robot_path_length_total"]) for row in ok_rows if math.isfinite(float(row.get("initial_robot_path_length_total", float("nan"))))]
        bom_unique_counts = [row.get("bom_unique_sku_counts") for row in ok_rows if row.get("bom_unique_sku_counts") is not None]
        bom_unique_total_values = [int(row.get("bom_unique_sku_total", 0) or 0) for row in ok_rows]
        resource_real_eval_period_values = [int(row.get("resource_real_eval_period", 0) or 0) for row in ok_rows]
        resource_eval_backend_values = [row.get("resource_eval_backend") for row in ok_rows if row.get("resource_eval_backend") not in (None, "")]
        fixgurobi_eval_count_values = [int(row.get("fixgurobi_eval_count", 0) or 0) for row in ok_rows]
        fixgurobi_last_status_values = [row.get("fixgurobi_last_status") for row in ok_rows if row.get("fixgurobi_last_status") not in (None, "")]
        coverage_hard_reject_values = [int(row.get("coverage_hard_reject_count", 0) or 0) for row in ok_rows]
        exact_cache_hit_values = [int(row.get("exact_eval_cache_hit_count", 0) or 0) for row in ok_rows]
        x_decap_values = [int(row.get("x_failure_decapitation_count", 0) or 0) for row in ok_rows]
        stop_reason_values = [row.get("stop_reason") for row in ok_rows if row.get("stop_reason") not in (None, "")]
        layer_selected_x_values = [int(row.get("layer_selected_x", 0) or 0) for row in ok_rows]
        layer_selected_y_values = [int(row.get("layer_selected_y", 0) or 0) for row in ok_rows]
        layer_selected_z_values = [int(row.get("layer_selected_z", 0) or 0) for row in ok_rows]
        layer_selected_xyz_values = [int(row.get("layer_selected_xyz", 0) or 0) for row in ok_rows]
        layer_accepted_x_values = [int(row.get("layer_accepted_x", 0) or 0) for row in ok_rows]
        layer_accepted_y_values = [int(row.get("layer_accepted_y", 0) or 0) for row in ok_rows]
        layer_accepted_z_values = [int(row.get("layer_accepted_z", 0) or 0) for row in ok_rows]
        layer_accepted_xyz_values = [int(row.get("layer_accepted_xyz", 0) or 0) for row in ok_rows]
        current_baseline = _case_current_baseline_cmax(scale)
        best_of_best_z = min(best_values) if best_values else float("nan")
        out.append({
            "scale": scale,
            "run_count": int(len(scale_rows)),
            "ok_count": int(len(ok_rows)),
            "error_count": int(len(scale_rows) - len(ok_rows)),
            "best_of_best_z": best_of_best_z,
            "mean_best_z": (sum(best_values) / len(best_values)) if best_values else float("nan"),
            "target_cmax": _case_target_cmax(scale),
            "delta_to_target": float(best_of_best_z - _case_target_cmax(scale)) if math.isfinite(best_of_best_z) and math.isfinite(_case_target_cmax(scale)) else float("nan"),
            "current_baseline_cmax": current_baseline,
            "delta_to_current_baseline": float(best_of_best_z - current_baseline) if math.isfinite(best_of_best_z) and math.isfinite(current_baseline) else float("nan"),
            "not_worse_than_current_baseline": bool(math.isfinite(best_of_best_z) and math.isfinite(current_baseline) and best_of_best_z <= current_baseline + 1e-9),
            "total_runtime_sec": float(sum(runtime_values)) if runtime_values else float("nan"),
            "initial_makespan": _collapse_values(init_makespan_values),
            "improvement_ratio": _collapse_values(improvement_values),
            "bom_unique_sku_counts": _collapse_values(bom_unique_counts),
            "bom_unique_sku_total": _collapse_values(bom_unique_total_values),
            "initial_task_count": _collapse_values(init_task_values),
            "initial_subtask_count": _collapse_values(init_subtask_values),
            "initial_station_loads": _collapse_values(init_station_loads),
            "initial_robot_path_lengths": _collapse_values(init_robot_paths),
            "initial_robot_path_length_total": _collapse_values(init_robot_path_total_values),
            "resource_real_eval_period": _collapse_values(resource_real_eval_period_values),
            "resource_eval_backend": _collapse_values(resource_eval_backend_values),
            "fixgurobi_eval_count": _collapse_values(fixgurobi_eval_count_values),
            "fixgurobi_last_status": _collapse_values(fixgurobi_last_status_values),
            "coverage_hard_reject_count": _collapse_values(coverage_hard_reject_values),
            "exact_eval_cache_hit_count": _collapse_values(exact_cache_hit_values),
            "x_failure_decapitation_count": _collapse_values(x_decap_values),
            "stop_reason": _collapse_values(stop_reason_values),
            "layer_selected_x": _collapse_values(layer_selected_x_values),
            "layer_selected_y": _collapse_values(layer_selected_y_values),
            "layer_selected_z": _collapse_values(layer_selected_z_values),
            "layer_selected_xyz": _collapse_values(layer_selected_xyz_values),
            "layer_accepted_x": _collapse_values(layer_accepted_x_values),
            "layer_accepted_y": _collapse_values(layer_accepted_y_values),
            "layer_accepted_z": _collapse_values(layer_accepted_z_values),
            "layer_accepted_xyz": _collapse_values(layer_accepted_xyz_values),
            "coverage_ok_count": int(sum(1 for row in ok_rows if bool(row.get("coverage_ok", False)))),
            "makespan_consistent_count": int(sum(1 for row in ok_rows if bool(row.get("makespan_consistent", False)))),
            "unreasonable_solution_count": int(sum(1 for row in ok_rows if bool(row.get("has_unreasonable_solution", False)))),
        })
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run resource_time_alns on all built-in scales with ALNS max_iters=200 by default.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_ALL_CASES), help="Case list")
    parser.add_argument("--runs", type=int, default=1, help="Runs per case")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed")
    parser.add_argument("--same-seed", action="store_true", help="Reuse the same seed for every run")
    parser.add_argument("--max-iters", type=int, default=300, help="ALNS max_iters")
    parser.add_argument("--no-improve-limit", type=int, default=3, help="TRA no_improve_limit")
    parser.add_argument("--epsilon", type=float, default=0.05, help="TRA epsilon")
    parser.add_argument("--bom-arrival-window-sec", type=float, default=60.0, help="BOM arrival window hard constraint; disable with <= 0")
    parser.add_argument("--disable-order-time-windows", action="store_true", help="Disable order time windows in TRA objective")
    parser.add_argument("--kitting-span-penalty-weight", type=float, default=5.0, help="Soft penalty weight for kitting span overruns")
    parser.add_argument("--deadline-penalty-weight", type=float, default=1000.0, help="Soft penalty weight for deadline overruns")
    parser.add_argument("--sp2-time-limit-sec", type=float, default=10.0, help="SP2 time limit")
    parser.add_argument("--sp3-use-mip", action="store_true", help="Use SP3 MIP instead of heuristic")
    parser.add_argument("--sp4-lkh-time-limit-seconds", type=int, default=5, help="SP4 LKH time limit")
    parser.add_argument("--exact-sp4-lkh-time-limit-seconds", type=int, default=120, help="Exact validation SP4 LKH time limit")
    parser.add_argument("--route-polish-sp4-lkh-time-limit-seconds", type=int, default=180, help="SP4 LKH time limit used by route_polish_exact profile")
    parser.add_argument("--sp4-first-solution-strategies", type=str, default="PATH_CHEAPEST_ARC,SAVINGS,PARALLEL_CHEAPEST_INSERTION", help="Comma-separated OR-Tools first solution strategies")
    parser.add_argument("--sp4-first-solution-slice-seconds", type=int, default=10, help="Time slice for each SP4 first solution strategy")
    parser.add_argument("--sp4-enable-guided-local-search", action="store_true", help="Enable guided local search as final SP4 pass")
    parser.add_argument("--sp4-same-subtask-vehicle-mode", type=str, default="conditional", choices=["strict", "conditional", "relaxed"], help="How strongly to bind same-subtask tasks to one robot")
    parser.add_argument("--sp4-same-subtask-vehicle-threshold", type=int, default=2, help="Conditional same-subtask robot threshold")
    parser.add_argument("--sp4-enable-greedy-fallback", action="store_true", help="Enable greedy fallback if SP4 exact routing fails")
    parser.add_argument("--sp4-raise-on-no-solution", action="store_true", help="Raise structured SP4 error on no-solution before fallback handling")
    parser.add_argument("--resource-real-eval-period", type=int, default=8, help="Validator period")
    parser.add_argument("--resource-candidate-pool-size", type=int, default=8)
    parser.add_argument("--resource-candidate-pool-max-attempts", type=int, default=40)
    parser.add_argument("--resource-exact-candidate-trial-limit", type=int, default=8)
    parser.add_argument("--resource-stop-if-best-z-no-change-rounds", type=int, default=40)
    parser.add_argument("--resource-stop-if-validated-best-no-change-rounds", type=int, default=40)
    parser.add_argument("--z-f0-topk", type=int, default=4)
    parser.add_argument("--z-f1-topk", type=int, default=2)
    parser.add_argument("--z-f2-topk", type=int, default=2)
    parser.add_argument("--z-eval-all-candidates", action="store_true")
    parser.add_argument("--enable-z-positive-mining-verify", action="store_true")
    parser.add_argument("--resource-multi-start-count", type=int, default=1)
    parser.add_argument("--resource-multi-start-patience", type=int, default=0)
    parser.add_argument("--resource-enable-xyz-operator", action="store_true", default=False)
    parser.add_argument("--resource-enable-critical-path-xyz", action="store_true", default=False)
    parser.add_argument("--resource-enable-best-y-assignment-polish", action="store_true", default=False)
    parser.add_argument("--resource-enable-best-z-sortify-polish", action="store_true", default=False)
    parser.add_argument("--resource-y-polish-candidate-limit", type=int, default=64)
    parser.add_argument("--disable-resource-xyz-stagnation-gate", action="store_true", default=False)
    parser.add_argument("--enable-sp3-init-portfolio", action="store_true", default=False, help="Run both SP3 heuristic and SP3 MIP initializations, then keep the better result")
    parser.add_argument("--allow-sp4-mip", action="store_true", default=False, help="Unsafe escape hatch; default forbids SP4 MIP and forces OR-Tools.")
    parser.add_argument("--operator-profile", type=str, default="baseline_safe", choices=["baseline_safe", "critical_xyz_expanded", "route_polish_exact"], help="Gated TRA operator profile")
    parser.add_argument("--resource-eval-backend", type=str, default="surrogate", choices=["surrogate", "fixgurobi_prefix"], help="Candidate evaluation backend")
    parser.add_argument("--fixgurobi-time-limit-sec", type=float, default=20.0, help="Time limit for each fixed-Gurobi candidate evaluation")
    parser.add_argument("--fixgurobi-mip-gap", type=float, default=0.01, help="MIP gap for each fixed-Gurobi candidate evaluation")
    parser.add_argument("--fixgurobi-candidate-trial-limit", type=int, default=1, help="Max exact candidates per ALNS round evaluated by fixed Gurobi")
    parser.add_argument("--fixgurobi-cache-size", type=int, default=128, help="Fixed-Gurobi evaluator cache size")
    parser.add_argument("--fixgurobi-fix-used-stack-ids", action="store_true", default=False, help="Also force the exact used stack id set for each order")
    parser.add_argument("--fixgurobi-output", action="store_true", default=False, help="Enable Gurobi log output inside fixed evaluator")
    parser.add_argument("--disable-current-baseline-gate", action="store_true", default=False, help="Do not mark runs against the locked current TRA baseline")
    parser.add_argument("--fail-on-current-baseline-regression", action="store_true", default=False, help="Exit nonzero if any case is worse than the locked current TRA baseline")
    parser.add_argument("--gurobi-reference-root", type=str, default=os.path.join(ROOT_DIR, "result", "gurobi_s1_s9_no_scale_adapt_300s_001_20260517"))
    parser.add_argument("--gurobi-s6-reference-root", type=str, default=os.path.join(ROOT_DIR, "result", "gurobi_s6_tune_20260517_try3"))
    parser.add_argument("--write-structure-compare", action="store_true", default=False)
    parser.add_argument("--export-best-solution", action="store_true", help="Keep explicit export_best_solution=True")
    parser.add_argument("--write-iteration-logs", action="store_true", help="Keep explicit write_iteration_logs=True")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"tra_alns_{timestamp}"))

    all_rows: List[Dict[str, Any]] = []
    total_jobs = max(1, int(args.runs)) * max(1, len(args.cases))
    done = 0

    for scale in [str(case).upper() for case in (args.cases or ["SMALL", "SMALL2"])]:
        for run_idx in range(int(args.runs)):
            seed = int(args.seed_base) if bool(args.same_seed) else int(args.seed_base) + int(run_idx)
            done += 1
            print(f"[{done}/{total_jobs}] scale={scale} run={run_idx + 1}/{int(args.runs)} seed={seed}")
            row = _run_one(args, scale=scale, run_idx=run_idx, seed=seed, batch_root=batch_root)
            all_rows.append(row)

    summary_rows = _summarize(all_rows)
    structure_rows: List[Dict[str, Any]] = []
    if bool(args.write_structure_compare):
        for row in all_rows:
            scale = str(row.get("scale", "")).upper()
            tra_export_dir = os.path.join(str(row.get("result_root", "")), "best_solution_export")
            gurobi_export_dir = _gurobi_export_dir(args, scale)
            cmp_row = compare_solution_signatures(
                case_name=scale,
                tra_export_dir=tra_export_dir,
                gurobi_export_dir=gurobi_export_dir,
                target_cmax=_case_target_cmax(scale),
            )
            cmp_row["runtime_sec"] = _safe_float(row.get("runtime_sec", float("nan")))
            cmp_row["status"] = str(row.get("status", ""))
            cmp_row["variant_name"] = str(row.get("variant_name", ""))
            cmp_row["current_baseline_cmax"] = _case_current_baseline_cmax(scale)
            cmp_row["not_worse_than_current_baseline"] = bool(row.get("not_worse_than_current_baseline", False))
            cmp_row["reached_by_operator"] = str(row.get("stop_reason", ""))
            structure_rows.append(cmp_row)
    _write_csv(os.path.join(batch_root, "batch_runs.csv"), all_rows)
    _write_csv(os.path.join(batch_root, "batch_summary.csv"), summary_rows)
    if structure_rows:
        write_structure_compare_csv(os.path.join(batch_root, "structure_compare.csv"), structure_rows)
        write_structure_compare_csv(os.path.join(batch_root, "route_compare.csv"), structure_rows)
    _write_json(
        os.path.join(batch_root, "batch_meta.json"),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cases": [str(case).upper() for case in (args.cases or [])],
            "runs_per_case": int(args.runs),
            "seed_base": int(args.seed_base),
            "same_seed": bool(args.same_seed),
            "alns_max_iters": int(args.max_iters),
            "operator_profile": str(args.operator_profile),
            "resource_eval_backend": str(args.resource_eval_backend),
            "current_baseline_cmax": CURRENT_TRA_BASELINE_CMAX,
            "batch_root": batch_root,
        },
    )
    _write_txt(
        os.path.join(batch_root, "batch_summary.txt"),
        [
            f"batch_root={batch_root}",
            f"cases={[str(case).upper() for case in (args.cases or [])]}",
            f"runs_per_case={int(args.runs)}",
            f"alns_max_iters={int(args.max_iters)}",
            f"seed_base={int(args.seed_base)}",
            f"same_seed={bool(args.same_seed)}",
            "",
            *[
                (
                    f"scale={row['scale']}, run_count={row['run_count']}, ok_count={row['ok_count']}, "
                    f"error_count={row['error_count']}, best_of_best_z={row['best_of_best_z']}, "
                    f"mean_best_z={row['mean_best_z']}, total_runtime_sec={row['total_runtime_sec']}, "
                    f"target_cmax={row['target_cmax']}, delta_to_target={row['delta_to_target']}, "
                    f"current_baseline_cmax={row['current_baseline_cmax']}, "
                    f"delta_to_current_baseline={row['delta_to_current_baseline']}, "
                    f"not_worse_than_current_baseline={row['not_worse_than_current_baseline']}, "
                    f"initial_makespan={row['initial_makespan']}, "
                    f"bom_unique_sku_counts={row['bom_unique_sku_counts']}, "
                    f"bom_unique_sku_total={row['bom_unique_sku_total']}, "
                    f"initial_task_count={row['initial_task_count']}, "
                    f"initial_subtask_count={row['initial_subtask_count']}, "
                    f"initial_station_loads={row['initial_station_loads']}, "
                    f"initial_robot_path_lengths={row['initial_robot_path_lengths']}, "
                    f"coverage_ok_count={row['coverage_ok_count']}, "
                    f"makespan_consistent_count={row['makespan_consistent_count']}, "
                    f"unreasonable_solution_count={row['unreasonable_solution_count']}"
                )
                for row in summary_rows
            ],
        ],
    )
    if bool(args.fail_on_current_baseline_regression):
        regressions = [
            row for row in summary_rows
            if not bool(row.get("not_worse_than_current_baseline", False))
        ]
        if regressions:
            labels = ", ".join(
                f"{row['scale']}: {row['best_of_best_z']} > {row['current_baseline_cmax']}"
                for row in regressions
            )
            raise SystemExit(f"current baseline regression: {labels}")
    print(f"[DONE] batch_root={batch_root}")


if __name__ == "__main__":
    main()
