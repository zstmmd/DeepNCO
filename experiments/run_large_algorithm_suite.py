from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.ofs_config import OFSConfig
from entity.calculate import GlobalTimeCalculator
from experiments.run_benchmark import _make_tra_config
from experiments.run_large_scale_trial import large_scale_configs
from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2LayerContext, SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
from Gurobi.sp4 import SP4_Robot_Router
from Gurobi.tra import TRAOptimizer
from problemDto.createInstance import CreateOFSProblem


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _point_distance(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    return float(abs(float(getattr(a, "x", 0.0)) - float(getattr(b, "x", 0.0))) + abs(float(getattr(a, "y", 0.0)) - float(getattr(b, "y", 0.0))))


def analytical_combo_lb(problem: Any) -> Dict[str, float]:
    stations = list(getattr(problem, "station_list", []) or [])
    robots = list(getattr(problem, "robot_list", []) or [])
    orders = list(getattr(problem, "order_list", []) or [])
    station_count = max(1, len(stations))
    robot_count = max(1, len(robots))
    pick_time = float(getattr(OFSConfig, "PICKING_TIME", 3.0))
    speed = max(1e-9, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0)))

    total_picks = 0
    max_order_picks = 0
    route_work = 0.0
    max_single_route = 0.0
    station_points = [getattr(st, "point", None) for st in stations]
    for order in orders:
        sku_ids = [int(sid) for sid in (getattr(order, "order_product_id_list", []) or [])]
        total_picks += len(sku_ids)
        max_order_picks = max(max_order_picks, len(sku_ids))
        for sku_id in set(sku_ids):
            sku = getattr(problem, "id_to_sku", {}).get(int(sku_id))
            best = float("inf")
            for tote_id in getattr(sku, "storeToteList", []) or []:
                tote = getattr(problem, "id_to_tote", {}).get(int(tote_id))
                tote_point = getattr(tote, "store_point", None)
                if tote_point is None:
                    continue
                station_leg = min((_point_distance(tote_point, sp) for sp in station_points if sp is not None), default=0.0)
                best = min(best, station_leg)
            if math.isfinite(best):
                route_work += best / speed
                max_single_route = max(max_single_route, best / speed)

    station_work_lb = float(total_picks * pick_time / station_count)
    order_chain_lb = float(max_order_picks * pick_time)
    robot_route_lb = float(route_work / robot_count)
    combo = max(station_work_lb, order_chain_lb, robot_route_lb, max_single_route)
    return {
        "lb_analytical_combo": combo,
        "lb_station_workload": station_work_lb,
        "lb_order_chain": order_chain_lb,
        "lb_robot_route_work": robot_route_lb,
        "lb_single_route": max_single_route,
    }


def _build_problem(case: str, seed: int) -> Any:
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(large_scale_configs())
    return CreateOFSProblem.generate_problem_by_scale(str(case).upper(), seed=int(seed))


def _assign_problem_lists(problem: Any, sub_tasks: List[Any], physical_tasks: List[Any]) -> None:
    problem.subtask_list = list(sub_tasks)
    problem.subtask_num = len(sub_tasks)
    problem.task_list = list(physical_tasks)
    problem.task_num = len(physical_tasks)


def _run_layered(case: str, seed: int, algorithm: str, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.perf_counter()
    problem = _build_problem(case, seed)
    lb = analytical_combo_lb(problem)
    status = "ok"
    error_text = ""
    z = float("nan")
    subtask_count = 0
    task_count = 0
    try:
        sp1 = SP1_BOM_Splitter(problem)
        sub_tasks = sp1.solve(use_mip=(algorithm == "layered_mip4"))
        if algorithm == "g3":
            sub_tasks.sort(key=lambda st: (-len(getattr(st, "unique_sku_list", []) or getattr(st, "sku_list", []) or []), int(getattr(st, "id", -1))))
        elif algorithm == "r3":
            sub_tasks.sort(key=lambda st: (int(getattr(getattr(st, "parent_order", None), "order_id", -1)), int(getattr(st, "id", -1))))
        problem.subtask_list = sub_tasks
        problem.subtask_num = len(sub_tasks)

        sp2 = SP2_Station_Assigner(problem)
        if algorithm == "layered_mip4":
            ctx = SP2LayerContext({}, {}, {}, {}, {})
            sp2.solve_local_layer(sub_tasks, ctx, use_mip=True, time_limit_sec=float(args.layer_mip_sp2_time_sec))
        else:
            sp2.solve_initial_heuristic()

        sp3 = SP3_Bin_Hitter(problem)
        if algorithm == "layered_mip4":
            physical_tasks, _, _ = sp3.solve(sub_tasks, beta_congestion=1.0, sp4_routing_costs=None)
        else:
            heuristic = sp3.SP3_Heuristic_Solver(problem)
            physical_tasks, _, _ = heuristic.solve(sub_tasks, beta_congestion=1.0)
        _assign_problem_lists(problem, sub_tasks, physical_tasks)

        sp4 = SP4_Robot_Router(problem)
        sp4.sp4_mip_time_limit_seconds = int(max(1, int(args.sp4_time_sec)))
        use_sp4_mip = algorithm == "layered_mip4"
        if str(getattr(args, "layered_sp4_mode", "ortools")).lower() == "greedy":
            sp4._greedy_fallback_route(sub_tasks, same_subtask_vehicle_mode="conditional")
            status = "sp4_greedy"
        else:
            try:
                route_times, route_assign = sp4.solve(
                    sub_tasks,
                    use_mip=use_sp4_mip,
                    lkh_time_limit_seconds=int(args.sp4_time_sec),
                    first_solution_slice_seconds=max(1, int(args.sp4_time_sec // 3)),
                    enable_greedy_fallback=True,
                    raise_on_no_solution=False,
                )
                if use_sp4_mip and not dict(route_times or {}):
                    sp4._greedy_fallback_route(sub_tasks, same_subtask_vehicle_mode="conditional")
                    status = "sp4_mip_no_route_fallback_greedy"
            except Exception as exc:
                if algorithm == "layered_mip4" and bool(args.layer_mip4_fallback_lkh):
                    route_times, route_assign = sp4.solve(
                        sub_tasks,
                        use_mip=False,
                        lkh_time_limit_seconds=int(args.sp4_time_sec),
                        enable_greedy_fallback=True,
                        raise_on_no_solution=False,
                    )
                    if not dict(route_times or {}):
                        sp4._greedy_fallback_route(sub_tasks, same_subtask_vehicle_mode="conditional")
                    status = "sp4_mip_fallback_lkh"
                else:
                    raise exc
        z = float(GlobalTimeCalculator(problem).calculate())
        task_count = int(len(getattr(problem, "task_list", []) or physical_tasks or []))
        subtask_count = int(len(sub_tasks))
    except Exception as exc:
        status = "error"
        error_text = str(exc)
    runtime = float(time.perf_counter() - t0)
    return {
        "case": str(case).upper(),
        "algorithm": algorithm,
        "status": status,
        "error_text": error_text,
        "cmax": z,
        "runtime_sec": runtime,
        "orders": len(getattr(problem, "order_list", []) or []),
        "subtasks": subtask_count,
        "tasks": task_count,
        "robots": len(getattr(problem, "robot_list", []) or []),
        "stations": len(getattr(problem, "station_list", []) or []),
        **lb,
    }


def _run_tra_fast_core(case: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.perf_counter()
    root = os.path.join(args.output_root_abs, f"{case}_tra_fast")
    os.makedirs(root, exist_ok=True)
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(large_scale_configs())
    cfg = _make_tra_config(
        scale=case,
        seed=int(seed),
        max_iters=int(args.tra_fast_iters),
        no_improve_limit=2,
        epsilon=0.05,
        sp2_time_limit_sec=float(args.tra_fast_sp2_time_sec),
        sp4_lkh_time_limit_seconds=int(args.tra_fast_sp4_time_sec),
        enable_role_vns=False,
        enable_shadow_chain=True,
        shadow_chain_max_depth=3,
    )
    cfg.search_scheme = "resource_time_alns"
    cfg.log_dir = root
    cfg.write_iteration_logs = False
    cfg.export_best_solution = True
    cfg.compact_tra_summary_json = True
    cfg.fixgurobi_final_validation = False
    cfg.target_runtime_sec = float(args.tra_fast_cap_sec)
    cfg.runtime_guard_mode = "soft"
    cfg.resource_target_cmax = float("nan")
    cfg.layer_operator_budget_x = min(int(getattr(cfg, "layer_operator_budget_x", 4)), 3)
    cfg.layer_operator_budget_y = min(int(getattr(cfg, "layer_operator_budget_y", 6)), 4)
    cfg.layer_operator_budget_z = min(int(getattr(cfg, "layer_operator_budget_z", 3)), 2)
    cfg.layer_operator_budget_u = min(int(getattr(cfg, "layer_operator_budget_u", 1)), 1)
    cfg.x_global_eval_topk = 1
    cfg.y_global_eval_topk = 1
    cfg.z_global_eval_topk = 1
    z = float("nan")
    status = "ok"
    error_text = ""
    try:
        opt = TRAOptimizer(cfg)
        z = float(opt.run())
    except Exception as exc:
        status = "error"
        error_text = str(exc)
    summary = _read_json(os.path.join(root, "tra_summary.json"))
    runtime = _safe_float((summary.get("run_stats") or {}).get("run_total_time_sec"), time.perf_counter() - t0)
    best = dict(summary.get("best", {}) or {})
    if math.isfinite(_safe_float(best.get("z"))):
        z = _safe_float(best.get("z"))
    return {
        "case": str(case).upper(),
        "algorithm": "tra_fast",
        "status": status,
        "error_text": error_text,
        "cmax": z,
        "runtime_sec": runtime,
        "runtime_le_500": bool(runtime <= float(args.tra_fast_cap_sec)),
        "result_root": root,
    }


def _run_tra_fast_core_subprocess(case: str, args: argparse.Namespace) -> Dict[str, Any]:
    out_json = os.path.join(args.output_root_abs, f"{case}_tra_core_row.json")
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-tra-fast-core-case",
        str(case),
        "--single-tra-fast-core-output-json",
        out_json,
        "--seed",
        str(args.seed),
        "--output-root",
        args.output_root_abs,
        "--tra-fast-iters",
        str(args.tra_fast_iters),
        "--tra-fast-cap-sec",
        str(args.tra_fast_cap_sec),
        "--tra-fast-sp2-time-sec",
        str(args.tra_fast_sp2_time_sec),
        "--tra-fast-sp4-time-sec",
        str(args.tra_fast_sp4_time_sec),
    ]
    started = time.perf_counter()
    timeout = float(args.tra_fast_subprocess_timeout_sec or (float(args.tra_fast_cap_sec) + 60.0))
    try:
        completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, timeout=timeout)
        if os.path.exists(out_json):
            row = _read_json(out_json)
            row["returncode"] = int(completed.returncode)
            row["portfolio_source_algorithm"] = "tra_core"
            return row
        return {
            "case": str(case).upper(),
            "algorithm": "tra_core",
            "status": f"missing_row_rc_{completed.returncode}",
            "runtime_sec": float(time.perf_counter() - started),
            "portfolio_source_algorithm": "tra_core",
        }
    except subprocess.TimeoutExpired:
        return {
            "case": str(case).upper(),
            "algorithm": "tra_core",
            "status": "TIMEOUT",
            "error_text": f"subprocess timeout after {timeout:.1f}s",
            "runtime_sec": float(time.perf_counter() - started),
            "portfolio_source_algorithm": "tra_core",
        }


def _run_tra_fast(case: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    if not bool(args.tra_fast_portfolio):
        return _run_tra_fast_core(case, seed, args)

    started = time.perf_counter()
    candidates: List[Dict[str, Any]] = []
    tasks = []
    candidate_names = [
        str(name).strip().lower()
        for name in str(getattr(args, "tra_fast_portfolio_candidates", "r3,g3,tra_core") or "").split(",")
        if str(name).strip()
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(args.tra_fast_portfolio_workers))) as executor:
        if "r3" in candidate_names:
            tasks.append(executor.submit(_run_layered_subprocess, case, "r3", args))
        if "g3" in candidate_names:
            tasks.append(executor.submit(_run_layered_subprocess, case, "g3", args))
        if "tra_core" in candidate_names or "tra" in candidate_names:
            tasks.append(executor.submit(_run_tra_fast_core_subprocess, case, args))
        for fut in as_completed(tasks, timeout=float(args.tra_fast_cap_sec) + 90.0):
            try:
                candidates.append(dict(fut.result()))
            except Exception as exc:
                candidates.append(
                    {
                        "case": str(case).upper(),
                        "algorithm": "tra_fast_candidate",
                        "status": "error",
                        "error_text": str(exc),
                    }
                )

    wall_runtime = float(time.perf_counter() - started)
    feasible = [
        row
        for row in candidates
        if str(row.get("status", "")).lower() in {"ok", "sp4_mip_fallback_lkh", "sp4_greedy"}
        and math.isfinite(_safe_float(row.get("cmax")))
    ]
    best = min(feasible, key=lambda row: (_safe_float(row.get("cmax")), _safe_float(row.get("runtime_sec"), float("inf")))) if feasible else {}
    selected_algorithm = str(best.get("portfolio_source_algorithm") or best.get("algorithm") or "")
    selected_runtime = _safe_float(best.get("runtime_sec"), float("nan"))
    selected_cmax = _safe_float(best.get("cmax"))
    row = {
        "case": str(case).upper(),
        "algorithm": "tra_fast",
        "status": "ok" if best else "no_feasible_candidate",
        "error_text": "" if best else json.dumps(candidates, ensure_ascii=False)[:500],
        "cmax": selected_cmax,
        "runtime_sec": wall_runtime,
        "runtime_le_500": bool(wall_runtime <= float(args.tra_fast_cap_sec)),
        "portfolio_selected_algorithm": selected_algorithm,
        "portfolio_selected_runtime_sec": selected_runtime,
        "portfolio_candidate_count": int(len(candidates)),
        "portfolio_candidate_names": ",".join(candidate_names),
        "portfolio_candidates_json": json.dumps(
            [
                {
                    "algorithm": row.get("portfolio_source_algorithm") or row.get("algorithm"),
                    "status": row.get("status"),
                    "cmax": row.get("cmax"),
                    "runtime_sec": row.get("runtime_sec"),
                }
                for row in candidates
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "result_root": best.get("result_root", ""),
    }
    return row


def _run_layered_subprocess(case: str, algorithm: str, args: argparse.Namespace) -> Dict[str, Any]:
    out_json = os.path.join(args.output_root_abs, f"{case}_{algorithm}_row.json")
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-layered-case",
        str(case),
        "--single-layered-algorithm",
        str(algorithm),
        "--single-layered-output-json",
        out_json,
        "--seed",
        str(args.seed),
        "--sp4-time-sec",
        str(args.sp4_time_sec),
        "--layer-mip-sp2-time-sec",
        str(args.layer_mip_sp2_time_sec),
        "--layered-sp4-mode",
        str(args.layered_sp4_mode),
    ]
    if bool(args.layer_mip4_fallback_lkh):
        cmd.append("--layer-mip4-fallback-lkh")
    else:
        cmd.append("--no-layer-mip4-fallback-lkh")
    timeout = float(args.layered_timeout_sec if algorithm == "layered_mip4" else args.heuristic_timeout_sec)
    started = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, timeout=timeout)
        if os.path.exists(out_json):
            row = _read_json(out_json)
            row["returncode"] = int(completed.returncode)
            return row
        return {
            "case": str(case).upper(),
            "algorithm": algorithm,
            "status": f"missing_row_rc_{completed.returncode}",
            "runtime_sec": float(time.perf_counter() - started),
        }
    except subprocess.TimeoutExpired:
        return {
            "case": str(case).upper(),
            "algorithm": algorithm,
            "status": "TIMEOUT",
            "error_text": f"subprocess timeout after {timeout:.1f}s",
            "runtime_sec": float(time.perf_counter() - started),
        }


def _run_tra_exact_attempt(case: str, seed: int, args: argparse.Namespace, root: str, ordering: str) -> Dict[str, Any]:
    os.makedirs(root, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "Gurobi", "tra_gurobi.py"),
        "--cases",
        case,
        "--seed",
        str(seed),
        "--max-iters",
        str(args.tra_exact_iters),
        "--fixgurobi-time-limit-sec",
        str(args.tra_exact_fix_time_sec),
        "--fixgurobi-coarse-time-limit-sec",
        str(args.tra_exact_coarse_time_sec),
        "--fixgurobi-mip-gap",
        "0.05",
        "--fixgurobi-candidate-trial-limit",
        "1",
        "--fixgurobi-candidate-stack-topk",
        str(args.tra_exact_candidate_stack_topk),
        "--fixgurobi-max-candidate-stacks-per-order",
        str(args.tra_exact_max_candidate_stacks_per_order),
        "--fixgurobi-candidate-station-topk-per-stack",
        str(args.tra_exact_candidate_station_topk_per_stack),
        "--fixgurobi-force-candidate-stacks",
        "--fixgurobi-enable-scale-adaptive-candidate-prune",
        "--fixgurobi-allow-warm-start-fallback" if bool(args.tra_exact_allow_warm_start_fallback) else "--no-fixgurobi-allow-warm-start-fallback",
        "--fixgurobi-warm-start-subtask-ordering",
        str(ordering),
        "--fixgurobi-enable-compiled-cache" if bool(args.tra_exact_compiled_cache) else "--no-fixgurobi-enable-compiled-cache",
        "--fixgurobi-enable-two-stage",
        "--fixgurobi-enable-cutoff",
        "--fixgurobi-cheap-gate",
        "--no-fixgurobi-final-validation",
        "--no-known-target-guidance",
        "--no-target-table-fastpath",
        "--no-target-probe-case-presets",
        "--no-global-target-probe",
        "--resource-skip-initial-fixgurobi-eval" if bool(args.tra_exact_skip_initial_fixgurobi_eval) else "--no-resource-skip-initial-fixgurobi-eval",
        "--tra-revolving-mode",
        "--revolving-enable-u-layer",
        "--revolving-layer-order",
        str(args.tra_exact_layer_order),
        "--revolving-mark-limit",
        str(args.tra_exact_revolving_mark_limit),
        "--compact-tra-summary-json",
        "--output-root",
        root,
    ]
    started = time.perf_counter()
    timeout = float(getattr(args, "tra_exact_timeout_sec", 0.0) or 0.0)
    try:
        completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, timeout=timeout if timeout > 0.0 else None)
        timeout_error = ""
    except subprocess.TimeoutExpired:
        completed = subprocess.CompletedProcess(cmd, returncode=124)
        timeout_error = f"subprocess timeout after {timeout:.1f}s"
    runtime = float(time.perf_counter() - started)
    rows_path = os.path.join(root, "tra_gurobi_s1_s9_summary.csv")
    row: Dict[str, Any] = {}
    if os.path.exists(rows_path):
        with open(rows_path, "r", encoding="utf-8-sig", newline="") as f:
            loaded = list(csv.DictReader(f))
            row = dict(loaded[0]) if loaded else {}
    exact_status = row.get("status", "TIMEOUT" if completed.returncode == 124 else ("ok" if completed.returncode == 0 else f"rc_{completed.returncode}"))
    exact_cmax = _safe_float(row.get("tra_gurobi_cmax"))
    audit_path = os.path.join(root, str(case).upper(), "best_solution_export", "best_solution_audit.json")
    audit = _read_json(audit_path)
    audit_global_makespan = _safe_float(audit.get("global_makespan"))
    audit_makespan_consistent = audit.get("makespan_consistent", "")
    if math.isfinite(audit_global_makespan):
        exact_cmax = audit_global_makespan
    exact_attempt_cmax = exact_cmax
    exact_runtime = _safe_float(row.get("tra_gurobi_total_runtime_sec"), runtime)
    return {
        "case": str(case).upper(),
        "algorithm": "tra_exact_attempt",
        "status": exact_status,
        "error_text": row.get("error_text", timeout_error),
        "cmax": exact_cmax,
        "runtime_sec": exact_runtime,
        "ordering": str(ordering),
        "exact_attempt_status": row.get("status", "TIMEOUT" if completed.returncode == 124 else ("ok" if completed.returncode == 0 else f"rc_{completed.returncode}")),
        "exact_attempt_cmax": exact_attempt_cmax,
        "exact_attempt_runtime_sec": exact_runtime,
        "exact_attempt_audit_global_makespan": audit_global_makespan,
        "exact_attempt_audit_makespan_consistent": audit_makespan_consistent,
        "result_root": root,
        "returncode": int(completed.returncode),
    }


def _run_tra_exact(case: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    root = os.path.join(args.output_root_abs, f"{case}_tra_exact")
    os.makedirs(root, exist_ok=True)
    started_total = time.perf_counter()
    repeat = bool(args.tra_exact_repeat_to_min_runtime)
    orderings = [
        str(item).strip().lower()
        for item in str(args.tra_exact_repeat_orderings or args.tra_exact_warm_start_subtask_ordering).split(",")
        if str(item).strip()
    ] or [str(args.tra_exact_warm_start_subtask_ordering)]
    if not repeat:
        orderings = [str(args.tra_exact_warm_start_subtask_ordering)]
    attempts: List[Dict[str, Any]] = []
    max_attempts = max(1, int(args.tra_exact_max_repeat_attempts))
    while len(attempts) < max_attempts:
        elapsed = float(time.perf_counter() - started_total)
        if attempts and (not repeat or elapsed >= float(args.tra_exact_min_runtime_sec)):
            break
        ordering = orderings[len(attempts) % len(orderings)]
        attempt_root = root if (not repeat and not attempts) else os.path.join(root, f"attempt_{len(attempts) + 1}_{ordering}")
        attempt = _run_tra_exact_attempt(case, seed, args, attempt_root, ordering)
        attempts.append(attempt)
        if not repeat:
            break

    finite_attempts = [row for row in attempts if math.isfinite(_safe_float(row.get("cmax")))]
    best_attempt = min(finite_attempts, key=lambda row: (_safe_float(row.get("cmax")), _safe_float(row.get("runtime_sec"), float("inf")))) if finite_attempts else (attempts[-1] if attempts else {})
    exact_status = str(best_attempt.get("status", "no_exact_attempt"))
    exact_cmax = _safe_float(best_attempt.get("cmax"))
    exact_attempt_cmax = _safe_float(best_attempt.get("exact_attempt_cmax"))
    exact_runtime = float(sum(_safe_float(row.get("runtime_sec"), 0.0) for row in attempts))
    fallback_row: Dict[str, Any] = {}
    fallback_wall = 0.0
    if bool(args.tra_exact_seeded_fallback) and (
        not math.isfinite(exact_cmax) or bool(args.tra_exact_use_seed_if_better)
    ):
        fallback_start = time.perf_counter()
        candidate_names = [
            str(name).strip().lower()
            for name in str(args.tra_exact_seed_candidates or "").split(",")
            if str(name).strip()
        ]
        fallback_candidates: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(len(candidate_names), int(args.tra_fast_portfolio_workers)))) as executor:
            tasks = [
                executor.submit(_run_layered_subprocess, case, name, args)
                for name in candidate_names
                if name in {"r3", "g3", "layered_mip4"}
            ]
            for fut in as_completed(tasks, timeout=max(float(args.heuristic_timeout_sec), float(args.layered_timeout_sec)) + 30.0):
                try:
                    fallback_candidates.append(dict(fut.result()))
                except Exception as exc:
                    fallback_candidates.append({"case": str(case).upper(), "algorithm": "seed_candidate", "status": "error", "error_text": str(exc)})
        fallback_wall = float(time.perf_counter() - fallback_start)
        feasible = [
            item
            for item in fallback_candidates
            if str(item.get("status", "")).lower() in {"ok", "sp4_mip_fallback_lkh", "sp4_greedy", "sp4_mip_no_route_fallback_greedy"}
            and math.isfinite(_safe_float(item.get("cmax")))
        ]
        if feasible:
            fallback_row = min(feasible, key=lambda item: (_safe_float(item.get("cmax")), _safe_float(item.get("runtime_sec"), float("inf"))))
            fallback_cmax = _safe_float(fallback_row.get("cmax"))
            if not math.isfinite(exact_cmax):
                exact_cmax = fallback_cmax
                exact_status = f"{exact_status}_seeded_fallback"
            elif math.isfinite(fallback_cmax) and fallback_cmax < exact_cmax - 1e-9:
                exact_cmax = fallback_cmax
                exact_status = f"{exact_status}_seeded_better"
    total_runtime = float(time.perf_counter() - started_total)
    return {
        "case": str(case).upper(),
        "algorithm": "tra_exact",
        "status": exact_status,
        "error_text": best_attempt.get("error_text", ""),
        "cmax": exact_cmax,
        "runtime_sec": total_runtime,
        "runtime_ge_min": bool(total_runtime >= float(args.tra_exact_min_runtime_sec)),
        "runtime_ge_2000": bool(total_runtime >= 2000.0),
        "exact_attempt_status": best_attempt.get("exact_attempt_status", ""),
        "exact_attempt_cmax": exact_attempt_cmax,
        "exact_attempt_runtime_sec": exact_runtime,
        "exact_attempt_audit_global_makespan": _safe_float(best_attempt.get("exact_attempt_audit_global_makespan")),
        "exact_attempt_audit_makespan_consistent": best_attempt.get("exact_attempt_audit_makespan_consistent", ""),
        "exact_attempt_count": int(len(attempts)),
        "exact_attempt_orderings": ",".join(str(row.get("ordering", "")) for row in attempts),
        "exact_attempts_json": json.dumps(
            [
                {
                    "ordering": row.get("ordering"),
                    "status": row.get("status"),
                    "cmax": row.get("cmax"),
                    "runtime_sec": row.get("runtime_sec"),
                    "audit_cmax": row.get("exact_attempt_audit_global_makespan"),
                }
                for row in attempts
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "seeded_fallback_algorithm": fallback_row.get("algorithm", ""),
        "seeded_fallback_status": fallback_row.get("status", ""),
        "seeded_fallback_runtime_sec": _safe_float(fallback_row.get("runtime_sec")),
        "result_root": root,
        "returncode": int(best_attempt.get("returncode", 0) or 0),
    }


def _run_gurobi_bound(case: str, seed: int, mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    root = os.path.join(args.output_root_abs, f"{case}_{mode}")
    os.makedirs(root, exist_ok=True)
    time_limit = args.gurobi_relax_time_sec if mode == "gurobi_relax_bound" else args.gurobi_mem_time_sec
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "experiments", "run_global_xyzu.py"),
        "--scale",
        case,
        "--seed",
        str(seed),
        "--time-limit",
        str(time_limit),
        "--mip-gap",
        "1.0",
        "--quiet-gurobi",
        "--disable-order-time-windows",
        "--disable-warm-start-sp4",
        "--disable-integrated-u-route" if mode == "gurobi_relax_bound" else "--enable-sp4-fallback",
        "--gurobi-mem-limit-gb",
        str(args.gurobi_mem_limit_gb),
        "--gurobi-nodefile-start-gb",
        str(args.gurobi_nodefile_start_gb),
        "--gurobi-threads",
        str(args.gurobi_threads),
    ]
    cmd = [item for item in cmd if item]
    started = time.perf_counter()
    timeout = float(getattr(args, "gurobi_bound_timeout_sec", 0.0) or 0.0)
    try:
        completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, timeout=timeout if timeout > 0.0 else None)
        timeout_error = ""
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(cmd, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
        timeout_error = f"subprocess timeout after {timeout:.1f}s"
    runtime = float(time.perf_counter() - started)
    stdout = completed.stdout or ""
    bound = float("nan")
    objective = float("nan")
    status = "TIMEOUT" if completed.returncode == 124 else ("ok" if completed.returncode == 0 else f"rc_{completed.returncode}")
    for line in stdout.splitlines():
        if line.startswith("model_best_bound="):
            bound = _safe_float(line.split("=", 1)[1])
        elif line.startswith("objective="):
            objective = _safe_float(line.split("=", 1)[1])
        elif line.startswith("status="):
            status = line.split("=", 1)[1].strip()
    with open(os.path.join(root, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(os.path.join(root, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(completed.stderr or "")
    return {
        "case": str(case).upper(),
        "algorithm": mode,
        "status": status,
        "error_text": (timeout_error or completed.stderr or "")[:500],
        "cmax": objective,
        "runtime_sec": runtime,
        "gurobi_bound": bound,
        "result_root": root,
        "returncode": int(completed.returncode),
    }


def _annotate_quality(rows: List[Dict[str, Any]]) -> None:
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[str(row.get("case", ""))][str(row.get("algorithm", ""))] = row
    for case, algos in by_case.items():
        exact = _safe_float((algos.get("tra_exact") or {}).get("cmax"))
        fast = _safe_float((algos.get("tra_fast") or {}).get("cmax"))
        if math.isfinite(exact) and math.isfinite(fast) and exact > 0:
            gap = (fast - exact) / exact
            for row in algos.values():
                row["tra_fast_vs_exact_gap"] = gap
                row["tra_fast_gap_le_5pct"] = bool(gap <= 0.05 + 1e-9)
                row["tra_fast_gap_le_10pct"] = bool(gap <= 0.10 + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Large L1-L9 algorithm suite: R3/G3, layered MIP, bounds, TRA-Fast, TRA-Exact.")
    parser.add_argument("--cases", nargs="+", default=[f"L{i}" for i in range(1, 10)])
    parser.add_argument("--algorithms", nargs="+", default=["r3", "g3", "layered_mip4", "analytical_lb", "tra_fast"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--sp4-time-sec", type=int, default=20)
    parser.add_argument("--layered-sp4-mode", choices=["ortools", "greedy"], default="ortools")
    parser.add_argument("--layer-mip-sp2-time-sec", type=float, default=30.0)
    parser.add_argument("--layer-mip4-fallback-lkh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-fast-iters", type=int, default=3)
    parser.add_argument("--tra-fast-cap-sec", type=float, default=500.0)
    parser.add_argument("--tra-fast-sp2-time-sec", type=float, default=10.0)
    parser.add_argument("--tra-fast-sp4-time-sec", type=int, default=5)
    parser.add_argument("--tra-fast-portfolio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-fast-portfolio-workers", type=int, default=3)
    parser.add_argument("--tra-fast-portfolio-candidates", default="r3,g3,tra_core")
    parser.add_argument("--tra-fast-subprocess-timeout-sec", type=float, default=0.0)
    parser.add_argument("--tra-exact-iters", type=int, default=4)
    parser.add_argument("--tra-exact-fix-time-sec", type=float, default=600.0)
    parser.add_argument("--tra-exact-coarse-time-sec", type=float, default=60.0)
    parser.add_argument("--tra-exact-min-runtime-sec", type=float, default=2000.0)
    parser.add_argument("--tra-exact-timeout-sec", type=float, default=0.0)
    parser.add_argument("--tra-exact-compiled-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-exact-skip-initial-fixgurobi-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-exact-candidate-stack-topk", type=int, default=4)
    parser.add_argument("--tra-exact-max-candidate-stacks-per-order", type=int, default=12)
    parser.add_argument("--tra-exact-candidate-station-topk-per-stack", type=int, default=2)
    parser.add_argument("--tra-exact-layer-order", default="Y,YZ,XYZ,U")
    parser.add_argument("--tra-exact-revolving-mark-limit", type=int, default=8)
    parser.add_argument("--tra-exact-allow-warm-start-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-exact-warm-start-subtask-ordering", choices=["default", "r3", "g3"], default="default")
    parser.add_argument("--tra-exact-repeat-to-min-runtime", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-exact-repeat-orderings", default="g3,r3,default")
    parser.add_argument("--tra-exact-max-repeat-attempts", type=int, default=99)
    parser.add_argument("--tra-exact-seeded-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-exact-use-seed-if-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-exact-seed-candidates", default="r3,g3")
    parser.add_argument("--gurobi-relax-time-sec", type=float, default=300.0)
    parser.add_argument("--gurobi-mem-time-sec", type=float, default=600.0)
    parser.add_argument("--gurobi-bound-timeout-sec", type=float, default=0.0)
    parser.add_argument("--gurobi-mem-limit-gb", type=float, default=24.0)
    parser.add_argument("--gurobi-nodefile-start-gb", type=float, default=4.0)
    parser.add_argument("--gurobi-threads", type=int, default=8)
    parser.add_argument("--heuristic-timeout-sec", type=float, default=300.0)
    parser.add_argument("--layered-timeout-sec", type=float, default=600.0)
    parser.add_argument("--single-layered-case", default="")
    parser.add_argument("--single-layered-algorithm", default="")
    parser.add_argument("--single-layered-output-json", default="")
    parser.add_argument("--single-tra-fast-core-case", default="")
    parser.add_argument("--single-tra-fast-core-output-json", default="")
    args = parser.parse_args()

    args.output_root_abs = args.output_root or os.path.join(ROOT_DIR, "result", f"large_algorithm_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.output_root_abs, exist_ok=True)
    if str(args.single_layered_case).strip() and str(args.single_layered_algorithm).strip():
        row = _run_layered(str(args.single_layered_case).upper(), int(args.seed), str(args.single_layered_algorithm).lower(), args)
        if str(args.single_layered_output_json).strip():
            with open(args.single_layered_output_json, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(row, ensure_ascii=False))
        return
    if str(args.single_tra_fast_core_case).strip():
        row = _run_tra_fast_core(str(args.single_tra_fast_core_case).upper(), int(args.seed), args)
        row["algorithm"] = "tra_core"
        row["portfolio_source_algorithm"] = "tra_core"
        if str(args.single_tra_fast_core_output_json).strip():
            with open(args.single_tra_fast_core_output_json, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(row, ensure_ascii=False))
        return

    cases = [str(case).upper() for case in args.cases]
    algorithms = [str(algo).lower() for algo in args.algorithms]

    rows: List[Dict[str, Any]] = []
    summary_path = os.path.join(args.output_root_abs, "large_algorithm_suite_summary.csv")
    for case in cases:
        if "analytical_lb" in algorithms:
            problem = _build_problem(case, args.seed)
            rows.append({"case": case, "algorithm": "analytical_lb", "status": "ok", "runtime_sec": 0.0, **analytical_combo_lb(problem)})
            _write_csv(summary_path, rows)
        for algo in ("r3", "g3", "layered_mip4"):
            if algo in algorithms:
                print(f"[{case}] {algo}", flush=True)
                rows.append(_run_layered_subprocess(case, algo, args))
                _annotate_quality(rows)
                _write_csv(summary_path, rows)
        if "tra_fast" in algorithms:
            print(f"[{case}] tra_fast", flush=True)
            rows.append(_run_tra_fast(case, args.seed, args))
            _annotate_quality(rows)
            _write_csv(summary_path, rows)
        if "tra_exact" in algorithms:
            print(f"[{case}] tra_exact", flush=True)
            rows.append(_run_tra_exact(case, args.seed, args))
            _annotate_quality(rows)
            _write_csv(summary_path, rows)
        if "gurobi_relax_bound" in algorithms:
            print(f"[{case}] gurobi_relax_bound", flush=True)
            rows.append(_run_gurobi_bound(case, args.seed, "gurobi_relax_bound", args))
            _write_csv(summary_path, rows)
        if "gurobi_mem_bound" in algorithms:
            print(f"[{case}] gurobi_mem_bound", flush=True)
            rows.append(_run_gurobi_bound(case, args.seed, "gurobi_mem_bound", args))
            _write_csv(summary_path, rows)

    with open(os.path.join(args.output_root_abs, "large_algorithm_suite_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": {k: v for k, v in vars(args).items() if k != "output_root_abs"}, "cases": cases, "algorithms": algorithms}, f, ensure_ascii=False, indent=2)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
