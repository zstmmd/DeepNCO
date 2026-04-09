import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
from Gurobi.sp4 import SP4_Robot_Router
from Gurobi.tra import TRAOptimizer, TRARunConfig
from Gurobi.alns_relax_decomp import ALNSRelaxDecompConfig, ALNSRelaxDecompOptimizer

from entity.calculate import GlobalTimeCalculator
from config.ofs_config import OFSConfig


ALL_SCALES = ["SMALL", "SMALL2", "SMALL3", "SMALL_UNEVEN", "SMALL2_UNEVEN", "SMALL3_UNEVEN", "MEDIUM", "LARGE"]
GPU_DATASET_SCALES = ["SMALL", "SMALL2", "SMALL3", "SMALL_UNEVEN", "SMALL2_UNEVEN", "SMALL3_UNEVEN", "MEDIUM"]
EXPLICIT_ZRICH_SCALES = ["SMALL_ZRICH", "SMALL2_ZRICH"]
GPU_DATASET_SPLIT_SEEDS = {
    "train": [11, 22, 33, 44, 55],
    "val": [66],
    "test": [77],
}
GPU_DATASET_REPLAY_SEEDS = [42]


def _safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    w = idx - lo
    return float(ordered[lo] * (1 - w) + ordered[hi] * w)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _collect_all_tasks(problem) -> List[Any]:
    tasks = []
    for st in getattr(problem, "subtask_list", []) or []:
        tasks.extend(getattr(st, "execution_tasks", []) or [])
    return tasks


def _instance_stats(problem) -> Dict[str, Any]:
    subtasks = getattr(problem, "subtask_list", []) or []
    tasks = _collect_all_tasks(problem)
    need_points = getattr(problem, "need_points", []) or []

    return {
        "node_num": int(getattr(problem, "node_num", len(need_points))),
        "need_points": int(len(need_points)),
        "subtask_num": int(len(subtasks)),
        "task_num": int(len(tasks)),
        "robot_num": int(len(getattr(problem, "robot_list", []) or [])),
        "station_num": int(len(getattr(problem, "station_list", []) or [])),
        "order_num": int(len(getattr(problem, "order_list", []) or [])),
        "sku_num": int(len(getattr(problem, "skus_list", []) or [])),
        "tote_num": int(len(getattr(problem, "tote_list", []) or [])),
    }


def _coverage_metrics(problem) -> Dict[str, Any]:
    subtasks = getattr(problem, "subtask_list", []) or []
    unmet_by_subtask = [int(getattr(st, "sp3_unmet_sku_total", 0) or 0) for st in subtasks]
    unmet_total = int(sum(v for v in unmet_by_subtask if v > 0))
    unmet_subtask_count = int(sum(1 for v in unmet_by_subtask if v > 0))
    return {
        "unmet_sku_total": unmet_total,
        "unmet_subtask_count": unmet_subtask_count,
    }


def _sp1_metrics(problem) -> Dict[str, float]:
    subtasks = getattr(problem, "subtask_list", []) or []
    uniq_sku_counts = [len(getattr(st, "unique_sku_list", []) or []) for st in subtasks]
    sku_total_counts = [len(getattr(st, "sku_list", []) or []) for st in subtasks]
    return {
        "sp1_subtask_num": float(len(subtasks)),
        "sp1_avg_unique_sku_per_subtask": _safe_mean([float(v) for v in uniq_sku_counts]),
        "sp1_avg_sku_units_per_subtask": _safe_mean([float(v) for v in sku_total_counts]),
    }


def _sp2_metrics(problem) -> Dict[str, float]:
    stations = getattr(problem, "station_list", []) or []
    station_num = max(1, len(stations))
    loads = [0.0 for _ in range(station_num)]
    for st in getattr(problem, "subtask_list", []) or []:
        sid = int(getattr(st, "assigned_station_id", -1))
        if 0 <= sid < station_num:
            loads[sid] += 1.0

    if len(loads) <= 1:
        variance = 0.0
    else:
        variance = float(statistics.pvariance(loads))

    return {
        "sp2_max_station_load": max(loads) if loads else 0.0,
        "sp2_station_load_variance": variance,
    }


def _sp3_metrics(problem, sorting_costs: Dict[int, float]) -> Dict[str, float]:
    tasks = _collect_all_tasks(problem)
    noise_cnt = 0
    for t in tasks:
        noise_cnt += len(getattr(t, "noise_tote_ids", []) or [])

    return {
        "sp3_task_num": float(len(tasks)),
        "sp3_sorting_cost_total": float(sum((sorting_costs or {}).values())),
        "sp3_noise_tote_total": float(noise_cnt),
    }


def _sp4_metrics(problem) -> Dict[str, float]:
    tasks = _collect_all_tasks(problem)
    if not tasks:
        return {
            "sp4_max_arrival_stack": 0.0,
            "sp4_max_arrival_station": 0.0,
            "sp4_assigned_robot_num": 0.0,
        }

    robot_ids = {int(getattr(t, "robot_id", -1)) for t in tasks if int(getattr(t, "robot_id", -1)) >= 0}
    max_stack = max(float(getattr(t, "arrival_time_at_stack", 0.0)) for t in tasks)
    max_station = max(float(getattr(t, "arrival_time_at_station", 0.0)) for t in tasks)

    return {
        "sp4_max_arrival_stack": float(max_stack),
        "sp4_max_arrival_station": float(max_station),
        "sp4_assigned_robot_num": float(len(robot_ids)),
    }


def _compute_robot_path_length(problem) -> float:
    robots = getattr(problem, "robot_list", []) or []
    robot_map = {int(getattr(r, "id", -1)): r for r in robots}
    events_by_robot: Dict[int, List[Tuple[float, int, int]]] = {}

    for st in getattr(problem, "subtask_list", []) or []:
        for task in getattr(st, "execution_tasks", []) or []:
            rid = int(getattr(task, "robot_id", -1))
            if rid < 0:
                continue
            stack_obj = problem.point_to_stack.get(int(getattr(task, "target_stack_id", -1)))
            station_id = int(getattr(task, "target_station_id", -1))
            station_point = None
            if 0 <= station_id < len(getattr(problem, "station_list", []) or []):
                station_point = problem.station_list[station_id].point
            if stack_obj is not None and stack_obj.store_point is not None:
                events_by_robot.setdefault(rid, []).append((
                    float(getattr(task, "arrival_time_at_stack", 0.0)),
                    int(stack_obj.store_point.x),
                    int(stack_obj.store_point.y),
                ))
            if station_point is not None:
                events_by_robot.setdefault(rid, []).append((
                    float(getattr(task, "arrival_time_at_station", 0.0)),
                    int(station_point.x),
                    int(station_point.y),
                ))

    total_length = 0.0
    for rid, events in events_by_robot.items():
        robot = robot_map.get(rid)
        if robot is None or robot.start_point is None:
            continue
        events.sort(key=lambda x: x[0])
        last_x = int(robot.start_point.x)
        last_y = int(robot.start_point.y)
        dedup_events = []
        for tm, x, y in events:
            if dedup_events and dedup_events[-1][1] == x and dedup_events[-1][2] == y and abs(dedup_events[-1][0] - tm) <= 1e-9:
                continue
            dedup_events.append((tm, x, y))
        for _, x, y in dedup_events:
            total_length += abs(x - last_x) + abs(y - last_y)
            last_x, last_y = x, y
        total_length += abs(last_x - int(robot.start_point.x)) + abs(last_y - int(robot.start_point.y))

    return float(total_length)


def _extract_layer_objective_values(problem, sorting_costs: Dict[int, float] = None) -> Dict[str, float]:
    tasks = _collect_all_tasks(problem)
    hit_stack_ids = sorted({
        int(getattr(t, "target_stack_id", -1))
        for t in tasks
        if len(getattr(t, "hit_tote_ids", []) or []) > 0 and int(getattr(t, "target_stack_id", -1)) >= 0
    })
    station_idle_total = sum(float(getattr(s, "total_idle_time", 0.0)) for s in (getattr(problem, "station_list", []) or []))
    return {
        "sub_obj1_subtask_count": float(len(getattr(problem, "subtask_list", []) or [])),
        "sub_obj2_station_idle_total": float(station_idle_total),
        "sub_obj3_hit_stack_count": float(len(hit_stack_ids)),
        "sub_obj4_robot_path_length": float(_compute_robot_path_length(problem)),
    }


def _materialize_best_tra_solution(opt: TRAOptimizer) -> Dict[str, float]:
    if opt is None or getattr(opt, "best", None) is None or getattr(opt, "problem", None) is None:
        return {
            "sub_obj1_subtask_count": float("nan"),
            "sub_obj2_station_idle_total": float("nan"),
            "sub_obj3_hit_stack_count": float("nan"),
            "sub_obj4_robot_path_length": float("nan"),
        }
    opt._set_seed(opt.best.seed)
    opt.restore_snapshot(opt.best)
    _ = opt.evaluate()
    return _extract_layer_objective_values(opt.problem, getattr(opt, "last_sp3_sorting_costs", {}) or {})


def _utilization_metrics(problem, makespan: float) -> Dict[str, float]:
    tasks = _collect_all_tasks(problem)
    stations = getattr(problem, "station_list", []) or []
    robots = getattr(problem, "robot_list", []) or []

    if makespan <= 1e-9:
        return {
            "station_utilization_mean": 0.0,
            "station_idle_total": 0.0,
            "robot_busy_time_total": 0.0,
            "robot_utilization_mean": 0.0,
            "noise_ratio": 0.0,
            "task_per_subtask": 0.0,
        }

    station_utils = []
    station_idle_total = 0.0
    for s in stations:
        seq = getattr(s, "processed_tasks", []) or []
        busy = sum(float(getattr(t, "total_process_duration", 0.0)) for t in seq)
        station_utils.append(min(1.0, busy / makespan))
        station_idle_total += float(getattr(s, "total_idle_time", 0.0))

    robot_busy_map: Dict[int, float] = {int(getattr(r, "id", idx)): 0.0 for idx, r in enumerate(robots)}
    for t in tasks:
        rid = int(getattr(t, "robot_id", -1))
        if rid in robot_busy_map:
            robot_busy_map[rid] += float(getattr(t, "robot_service_time", 0.0))

    robot_utils = [min(1.0, v / makespan) for v in robot_busy_map.values()] if robot_busy_map else []

    noise_total = sum(len(getattr(t, "noise_tote_ids", []) or []) for t in tasks)
    target_total = sum(len(getattr(t, "target_tote_ids", []) or []) for t in tasks)
    noise_ratio = (float(noise_total) / float(target_total)) if target_total > 0 else 0.0

    subtasks = getattr(problem, "subtask_list", []) or []
    task_per_subtask = (float(len(tasks)) / float(len(subtasks))) if subtasks else 0.0

    return {
        "station_utilization_mean": _safe_mean(station_utils),
        "station_idle_total": float(station_idle_total),
        "robot_busy_time_total": float(sum(robot_busy_map.values())) if robot_busy_map else 0.0,
        "robot_utilization_mean": _safe_mean(robot_utils),
        "noise_ratio": float(noise_ratio),
        "task_per_subtask": float(task_per_subtask),
    }


def run_baseline_once(scale: str, seed: int, sp4_lkh_time_limit_seconds: int) -> Dict[str, Any]:
    t0 = time.perf_counter()

    problem = CreateOFSProblem.generate_problem_by_scale(scale, seed=seed)

    sp1 = SP1_BOM_Splitter(problem)
    sp2 = SP2_Station_Assigner(problem)
    sp3 = SP3_Bin_Hitter(problem)
    sp4 = SP4_Robot_Router(problem)
    sim = GlobalTimeCalculator(problem)

    stage_t0 = time.perf_counter()
    sub_tasks = sp1.solve(use_mip=False)
    problem.subtask_list = sub_tasks
    problem.subtask_num = len(sub_tasks)
    t_sp1 = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    sp2.solve_initial_heuristic()
    t_sp2 = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    heuristic = sp3.SP3_Heuristic_Solver(problem)
    physical_tasks, tote_selection, sorting_costs = heuristic.solve(problem.subtask_list, beta_congestion=1.0)
    problem.task_list = physical_tasks
    problem.task_num = len(physical_tasks)
    t_sp3 = time.perf_counter() - stage_t0

    # IMPORTANT: instance stats should reflect solved state (after SP1/SP3), not raw init state.
    instance_info = _instance_stats(problem)

    stage_t0 = time.perf_counter()
    arrival_times, robot_assign = sp4.solve(
        problem.subtask_list,
        use_mip=False,
        lkh_time_limit_seconds=int(sp4_lkh_time_limit_seconds),
    )
    _ = arrival_times, robot_assign, tote_selection
    t_sp4 = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    makespan = float(sim.calculate())
    t_sim = time.perf_counter() - stage_t0

    total_runtime = time.perf_counter() - t0

    row: Dict[str, Any] = {
        "algorithm": "baseline",
        "scale": scale,
        "seed": int(seed),
        **instance_info,
        **_sp1_metrics(problem),
        **_sp2_metrics(problem),
        **_sp3_metrics(problem, sorting_costs),
        **_sp4_metrics(problem),
        **_extract_layer_objective_values(problem, sorting_costs),
        **_utilization_metrics(problem, makespan),
        **_coverage_metrics(problem),
        "global_makespan": makespan,
        "runtime_total_sec": float(total_runtime),
        "runtime_sp1_sec": float(t_sp1),
        "runtime_sp2_sec": float(t_sp2),
        "runtime_sp3_sec": float(t_sp3),
        "runtime_sp4_sec": float(t_sp4),
        "runtime_sim_sec": float(t_sim),
    }
    return row


def _make_tra_config(scale: str, seed: int, max_iters: int, no_improve_limit: int, epsilon: float,
                     sp2_time_limit_sec: float, sp4_lkh_time_limit_seconds: int,
                     enable_sp3_precheck: bool = True, precheck_fail_action: str = "log",
                     enable_soft_mu: bool = False, enable_soft_pi: bool = False, enable_soft_beta: bool = False,
                     enable_sku_affinity: bool = False, mu_value: float = 1.0, pi_scale: float = 1.0,
                     pi_clip: float = 120.0, d0_threshold: float = 20.0, beta_base: float = 1.0,
                     beta_gain: float = 1.0, beta_min: float = 0.5, beta_max: float = 3.0,
                     sp2_shadow_weight: float = 1.0, enable_role_vns: bool = False,
                     eps_skip: float = 0.05, eps_light: float = 0.15,
                     weak_accept_eta: float = 0.02, vns_max_trials: int = 10,
                     mode_fail_limit: int = 3,
                     enable_shadow_chain: bool = True,
                     shadow_chain_max_depth: int = 3) -> TRARunConfig:
    cfg = TRARunConfig(
        scale=scale,
        seed=seed,
        max_iters=max_iters,
        no_improve_limit=no_improve_limit,
        epsilon=epsilon,
        sp2_use_mip=True,
        sp3_use_mip=False,
        sp4_use_mip=False,
        sp2_time_limit_sec=sp2_time_limit_sec,
        sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
        export_best_solution=False,
        write_iteration_logs=False,
        enable_sp1_feedback_analysis=False,
        enable_sp3_precheck=enable_sp3_precheck,
        sp3_precheck_fail_action=precheck_fail_action,
        enable_soft_mu=enable_soft_mu,
        enable_soft_pi=enable_soft_pi,
        enable_soft_beta=enable_soft_beta,
        enable_sku_affinity=enable_sku_affinity,
        mu_value=mu_value,
        pi_scale=pi_scale,
        pi_clip=pi_clip,
        d0_threshold=d0_threshold,
        beta_base=beta_base,
        beta_gain=beta_gain,
        beta_min=beta_min,
        beta_max=beta_max,
        sp2_shadow_weight=sp2_shadow_weight,
        enable_role_vns=False,
        eps_skip=eps_skip,
        eps_light=eps_light,
        weak_accept_eta=weak_accept_eta,
        vns_max_trials=vns_max_trials,
        mode_fail_limit=mode_fail_limit,
    )
    cfg.search_scheme = "layer_augmented"
    cfg.enable_shadow_chain = bool(enable_shadow_chain)
    cfg.shadow_chain_max_depth = max(1, int(shadow_chain_max_depth))
    return cfg


def _make_alns_relax_config(scale: str, seed: int, max_iters: int, no_improve_limit: int, epsilon: float,
                            sp2_time_limit_sec: float, sp4_lkh_time_limit_seconds: int,
                            enable_sp3_precheck: bool = True, precheck_fail_action: str = "log",
                            enable_soft_mu: bool = False, enable_soft_pi: bool = False, enable_soft_beta: bool = False,
                            enable_sku_affinity: bool = False, mu_value: float = 1.0, pi_scale: float = 1.0,
                            pi_clip: float = 120.0, d0_threshold: float = 20.0, beta_base: float = 1.0,
                            beta_gain: float = 1.0, beta_min: float = 0.5, beta_max: float = 3.0,
                            sp2_shadow_weight: float = 1.0, weak_accept_eta: float = 0.02,
                            alns_init_iters: int = 10) -> ALNSRelaxDecompConfig:
    return ALNSRelaxDecompConfig(
        scale=scale,
        seed=seed,
        max_iters=max_iters,
        no_improve_limit=no_improve_limit,
        epsilon=epsilon,
        sp2_use_mip=True,
        sp3_use_mip=True,
        sp4_use_mip=False,
        sp2_time_limit_sec=sp2_time_limit_sec,
        sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
        export_best_solution=False,
        write_iteration_logs=False,
        enable_sp1_feedback_analysis=False,
        enable_sp3_precheck=enable_sp3_precheck,
        sp3_precheck_fail_action=precheck_fail_action,
        enable_soft_mu=enable_soft_mu,
        enable_soft_pi=enable_soft_pi,
        enable_soft_beta=enable_soft_beta,
        enable_sku_affinity=enable_sku_affinity,
        mu_value=mu_value,
        pi_scale=pi_scale,
        pi_clip=pi_clip,
        d0_threshold=d0_threshold,
        beta_base=beta_base,
        beta_gain=beta_gain,
        beta_min=beta_min,
        beta_max=beta_max,
        sp2_shadow_weight=sp2_shadow_weight,
        enable_role_vns=False,
        weak_accept_eta=weak_accept_eta,
        alns_init_iters=alns_init_iters,
    )


def _tra_layer_row(scale: str, run_id: int, seed: int, iter_log: List[Dict[str, Any]], best_z: float,
                   runtime_sec: float, status: str, instance_info: Dict[str, Any],
                   unmet_sku_total: int, unmet_subtask_count: int,
                   precheck_unmet_sku_total: int = 0, precheck_unmet_subtask_count: int = 0,
                   precheck_status: str = "") -> Dict[str, Any]:
    focus_to_z: Dict[str, List[float]] = {"sp2": [], "sp3": [], "sp4": []}
    focus_to_lb: Dict[str, List[float]] = {"sp2": [], "sp3": [], "sp4": []}

    skipped = 0
    for row in iter_log:
        focus = str(row.get("focus", ""))
        if bool(row.get("skipped", False)):
            skipped += 1
            continue
        z = row.get("z", None)
        lb = row.get("lb", None)
        if focus in focus_to_z and isinstance(z, (int, float)) and not math.isnan(float(z)):
            focus_to_z[focus].append(float(z))
        if focus in focus_to_lb and isinstance(lb, (int, float)):
            focus_to_lb[focus].append(float(lb))

    def _min_or_nan(vals: List[float]) -> float:
        return min(vals) if vals else float("nan")

    return {
        "algorithm": "tra",
        "scale": scale,
        "run_id": int(run_id),
        "seed": int(seed),
        **instance_info,
        "best_z": float(best_z),
        "runtime_sec": float(runtime_sec),
        "iter_count": int(len(iter_log)),
        "skip_ratio": (float(skipped) / float(len(iter_log))) if iter_log else 0.0,
        "sp2_best_z": _min_or_nan(focus_to_z["sp2"]),
        "sp3_best_z": _min_or_nan(focus_to_z["sp3"]),
        "sp4_best_z": _min_or_nan(focus_to_z["sp4"]),
        "sp2_lb_mean": _safe_mean(focus_to_lb["sp2"]),
        "sp3_lb_mean": _safe_mean(focus_to_lb["sp3"]),
        "sp4_lb_mean": _safe_mean(focus_to_lb["sp4"]),
        "unmet_sku_total": int(unmet_sku_total),
        "unmet_subtask_count": int(unmet_subtask_count),
        "precheck_unmet_sku_total": int(precheck_unmet_sku_total),
        "precheck_unmet_subtask_count": int(precheck_unmet_subtask_count),
        "precheck_status": precheck_status,
        "status": status,
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_dict(prefix: str, obj: Any, out: Dict[str, Any]):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_dict(next_prefix, value, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = obj


def _flatten_row(obj: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    _flatten_dict("", dict(obj or {}), flat)
    return flat


def _write_timing_breakdown_files(result_root: str, case: str, seed: int, timing_breakdown: Dict[str, Any]):
    payload = {
        "case": str(case).upper(),
        "seed": int(seed),
        **dict(timing_breakdown or {}),
    }
    _write_json(os.path.join(result_root, "timing_breakdown.json"), payload)
    _write_csv(os.path.join(result_root, "timing_breakdown.csv"), [_flatten_row(payload)])


def _write_generator_summary_files(result_root: str, case: str, seed: int, problem: Any):
    generator_summary = dict(getattr(problem, "generator_summary", {}) or {})
    generator_summary.setdefault("case", str(case).upper())
    generator_summary.setdefault("seed", int(seed))
    generator_summary.setdefault("scale_name", str(getattr(problem, "scale_name", case)).upper())
    generator_summary.setdefault("generator_profile", str(getattr(problem, "generator_profile", "")))
    generator_summary.setdefault("instance_stats", _instance_stats(problem))
    redundancy_summary = dict(getattr(problem, "redundancy_summary", {}) or {})
    _write_json(os.path.join(result_root, "generator_summary.json"), generator_summary)
    _write_json(os.path.join(result_root, "redundancy_summary.json"), redundancy_summary)


def _case_z_layer_activity(iter_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    z_rows = [row for row in (iter_rows or []) if str(row.get("focus", "")).upper() == "Z"]
    global_eval_rows = [row for row in (iter_rows or []) if bool(row.get("global_eval_triggered", False))]
    z_global_eval_rows = [
        row for row in global_eval_rows
        if str(row.get("forced_eval_origin_layer", row.get("focus", ""))).upper() == "Z"
    ]
    return {
        "z_iter_count": int(len(z_rows)),
        "z_candidate_total": int(sum(int(row.get("candidate_count", 0) or 0) for row in z_rows)),
        "z_global_eval_count": int(len(z_global_eval_rows)),
        "z_global_eval_candidate_total": int(sum(int(row.get("z_global_eval_candidate_count", 0) or 0) for row in z_rows)),
        "z_f1_eval_count": int(sum(int(row.get("z_f1_eval_count", 0) or 0) for row in z_rows)),
        "z_append_shadow_count": int(sum(1 for row in z_rows if str(row.get("shadow_chain_event", "")) == "append_shadow")),
    }


def _base_scale_for_zrich(scale: str) -> str:
    scale_upper = str(scale).upper()
    if scale_upper == "SMALL_ZRICH":
        return "SMALL"
    if scale_upper == "SMALL2_ZRICH":
        return "SMALL2"
    return scale_upper


def _timing_report_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Shadow-Chain Timing Report",
        "",
        "This report summarizes wall time allocation for the seed-42 layer-augmented shadow-chain runs.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['case']}")
        lines.append("")
        lines.append(f"- best_z: {float(row.get('best_z', 0.0)):.6f}")
        lines.append(f"- wall_time_sec: {float(row.get('wall_time_sec', 0.0)):.6f}")
        lines.append(f"- local_vns_total_sec: {float(row.get('local_vns_total_sec', 0.0)):.6f}")
        lines.append(f"- x_f1_time_sec: {float(row.get('x_f1_time_sec', 0.0)):.6f}")
        lines.append(f"- z_f1_time_sec: {float(row.get('z_f1_time_sec', 0.0)):.6f}")
        lines.append(f"- forced_global_eval_time_sec: {float(row.get('forced_global_eval_time_sec', 0.0)):.6f}")
        lines.append(f"- global_eval_sp_total_sec: {float(row.get('global_eval_sp_total_sec', 0.0)):.6f}")
        lines.append(f"- snapshot_restore_overhead_sec: {float(row.get('snapshot_restore_overhead_sec', 0.0)):.6f}")
        delta = float(row.get("forced_global_eval_time_sec", 0.0)) - float(row.get("global_eval_time_sec", 0.0))
        chain_note = "reduced" if delta < -1e-9 else "increased" if delta > 1e-9 else "left unchanged"
        lines.append(
            f"- shadow-chain forced-global cost {chain_note} relative to aggregate global-eval time by {abs(delta):.6f} sec"
        )
        lines.append(
            f"- wall-time reconciliation gap: {float(row.get('reconciliation_gap_vs_wall_sec', 0.0)):.6f} sec"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _dataset_scale_vocab(scales: Optional[List[str]] = None) -> Dict[str, int]:
    ordered = [str(scale).upper() for scale in (scales or GPU_DATASET_SCALES)]
    for extra_scale in EXPLICIT_ZRICH_SCALES:
        if extra_scale not in ordered:
            ordered.append(extra_scale)
    if "LARGE" not in ordered:
        ordered.append("LARGE")
    return {scale: idx for idx, scale in enumerate(ordered)}


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _build_z_fallback_vocab(raw_rows: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, int]:
    names = {""}
    for split_rows in raw_rows.get("Z", {}).values():
        for row in split_rows:
            f1_features = dict(row.get("f1_features", {}) or {})
            names.add(str(f1_features.get("z_fallback_type", "")).strip())
    ordered = sorted(names)
    return {name: idx for idx, name in enumerate(ordered)}


def _flatten_supervised_candidate_row(
    raw_row: Dict[str, Any],
    layer: str,
    scale_vocab: Dict[str, int],
    fallback_vocab: Dict[str, int],
) -> Dict[str, Any]:
    layer = str(layer).upper()
    scale_name = str(raw_row.get("scale", raw_row.get("case", ""))).upper()
    f0_features = dict(raw_row.get("f0_features", {}) or {})
    f1_features = dict(raw_row.get("f1_features", {}) or {})
    ctx_features = dict(raw_row.get("ctx_features", {}) or {})
    flat = {
        "case": str(raw_row.get("case", scale_name)).upper(),
        "scale": scale_name,
        "seed": _safe_int(raw_row.get("seed", -1), -1),
        "iter": _safe_int(raw_row.get("iter", -1), -1),
        "layer": layer,
        "operator": _safe_str(raw_row.get("operator", "")),
        "operator_rank": _safe_int(raw_row.get("operator_rank", 0), 0),
        "candidate_signature": _safe_str(raw_row.get("candidate_signature", "")),
        "subtask_id": _safe_int(raw_row.get("subtask_id", -1), -1),
        "win_label": _safe_int(raw_row.get("win_label", 0), 0),
        "risk_label": _safe_int(raw_row.get("risk_label", 0), 0),
        "actual_reduction": _safe_float_or_none(raw_row.get("actual_reduction", 0.0)),
        "global_z_before": _safe_float_or_none(raw_row.get("global_z_before", 0.0)),
        "global_z_after": _safe_float_or_none(raw_row.get("global_z_after", 0.0)),
        "global_eval_triggered": bool(raw_row.get("global_eval_triggered", False)),
        "proposal_pass_fast_gate": bool(raw_row.get("proposal_pass_fast_gate", False)),
        "commit_decision": _safe_str(raw_row.get("commit_decision", "")),
        "accepted_type": _safe_str(raw_row.get("accepted_type", "")),
    }
    for key, value in f0_features.items():
        flat[f"f0_{key}"] = _safe_float_or_none(value)
    for key, value in f1_features.items():
        if layer == "Z" and str(key) == "z_fallback_type":
            continue
        flat[f"f1_{key}"] = _safe_float_or_none(value)
    if layer == "Z":
        flat["f1_z_fallback_type_code"] = float(fallback_vocab.get(str(f1_features.get("z_fallback_type", "")).strip(), 0))
    ctx_features["ctx_scale_id"] = float(scale_vocab.get(scale_name, len(scale_vocab)))
    for key, value in ctx_features.items():
        if str(key).startswith("ctx_"):
            flat[str(key)] = _safe_float_or_none(value)
    return flat


def _dataset_column_groups(flat_rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    id_columns = [
        "case",
        "scale",
        "seed",
        "iter",
        "layer",
        "operator",
        "operator_rank",
        "candidate_signature",
        "subtask_id",
    ]
    label_columns = [
        "win_label",
        "risk_label",
        "actual_reduction",
        "global_z_before",
        "global_z_after",
    ]
    diagnostic_columns = [
        "global_eval_triggered",
        "proposal_pass_fast_gate",
        "commit_decision",
        "accepted_type",
    ]
    feature_columns = sorted(
        key
        for key in {
            col
            for row in flat_rows
            for col in row.keys()
            if str(col).startswith(("f0_", "f1_", "ctx_"))
        }
    )
    return {
        "id": id_columns,
        "feature": feature_columns,
        "label": label_columns,
        "diagnostic": diagnostic_columns,
        "all": id_columns + feature_columns + label_columns + diagnostic_columns,
    }


def _dataset_missing_defaults(columns: List[str]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    string_columns = {"case", "scale", "layer", "operator", "candidate_signature", "commit_decision", "accepted_type"}
    int_columns = {"seed", "iter", "operator_rank", "subtask_id", "win_label", "risk_label"}
    bool_columns = {"global_eval_triggered", "proposal_pass_fast_gate"}
    for column in columns:
        if column in string_columns:
            defaults[column] = ""
        elif column in int_columns:
            defaults[column] = -1 if column not in {"operator_rank", "win_label", "risk_label"} else 0
        elif column in bool_columns:
            defaults[column] = False
        else:
            defaults[column] = 0.0
    return defaults


def _build_arrow_schema(columns: List[str]):
    import pyarrow as pa

    string_columns = {"case", "scale", "layer", "operator", "candidate_signature", "commit_decision", "accepted_type"}
    int_columns = {"seed", "iter", "operator_rank", "subtask_id", "win_label", "risk_label"}
    bool_columns = {"global_eval_triggered", "proposal_pass_fast_gate"}
    fields = []
    for column in columns:
        if column in string_columns:
            fields.append(pa.field(column, pa.string()))
        elif column in int_columns:
            fields.append(pa.field(column, pa.int64()))
        elif column in bool_columns:
            fields.append(pa.field(column, pa.bool_()))
        else:
            fields.append(pa.field(column, pa.float64()))
    return pa.schema(fields)


def _write_parquet_rows(path: str, rows: List[Dict[str, Any]], columns: List[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    defaults = _dataset_missing_defaults(columns)
    normalized_rows = []
    for row in rows:
        normalized_rows.append({column: row.get(column, defaults[column]) for column in columns})
    schema = _build_arrow_schema(columns)
    table = pa.Table.from_pylist(normalized_rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _write_xz_dataset_readme(
    path: str,
    scales: List[str],
    split_seeds: Dict[str, List[int]],
    replay_seeds: List[int],
    feature_order_x: List[str],
    feature_order_z: List[str],
    harvest_mode: str,
    distribution_note: str,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# X/Z GPU Dataset\n\n")
        f.write("This package is for remote PyTorch training only. It does not include local training scripts or model artifacts.\n\n")
        f.write("## Harvest Mode\n")
        f.write(f"- harvest_mode: `{harvest_mode}`\n")
        f.write(f"- distribution_note: `{distribution_note}`\n\n")
        f.write("## Scales\n")
        for scale in scales:
            f.write(f"- {str(scale).upper()}\n")
        f.write("\n## Split Seeds\n")
        for split_name, seeds in split_seeds.items():
            f.write(f"- {split_name}: {', '.join(str(seed) for seed in seeds)}\n")
        f.write(f"- replay: {', '.join(str(seed) for seed in replay_seeds)}\n")
        f.write("\n## Remote Training Assumptions\n")
        f.write("- Framework: PyTorch\n")
        f.write("- Input: tabular tensor built from numeric parquet columns\n")
        f.write("- Output heads: `p_win`, `p_risk`\n")
        f.write("- Suggested setup: two separate models, one for `X`, one for `Z`\n")
        f.write("\n## Feature Order\n")
        f.write(f"- X feature count: {len(feature_order_x)}\n")
        f.write(f"- Z feature count: {len(feature_order_z)}\n")
        f.write("- Exact feature order is recorded in `manifest.json`, `schema_x.json`, and `schema_z.json`.\n")


def _dedup_signature_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("layer", "")).upper(),
            str(row.get("scale", "")).upper(),
            int(_safe_int(row.get("seed", -1), -1)),
            str(row.get("candidate_signature", "")),
        )
        groups[key].append(row)

    kept_rows: List[Dict[str, Any]] = []
    for group_rows in groups.values():
        selected: List[Dict[str, Any]] = []
        if not group_rows:
            continue
        best_win = max(
            group_rows,
            key=lambda item: (
                float(_safe_float_or_none(item.get("actual_reduction", 0.0)) or 0.0),
                -int(_safe_int(item.get("iter", 10 ** 9), 10 ** 9)),
            ),
        )
        worst_risk = min(
            group_rows,
            key=lambda item: (
                float(_safe_float_or_none(item.get("actual_reduction", 0.0)) or 0.0),
                int(_safe_int(item.get("iter", 10 ** 9), 10 ** 9)),
            ),
        )
        earliest_seen = min(
            group_rows,
            key=lambda item: (
                int(_safe_int(item.get("iter", 10 ** 9), 10 ** 9)),
                -float(_safe_float_or_none(item.get("actual_reduction", 0.0)) or 0.0),
            ),
        )
        for picked in [best_win, worst_risk, earliest_seen]:
            if all(str(existing.get("candidate_signature", "")) != str(picked.get("candidate_signature", "")) or int(existing.get("iter", -1)) != int(picked.get("iter", -1)) or str(existing.get("operator", "")) != str(picked.get("operator", "")) for existing in selected):
                selected.append(picked)
        kept_rows.extend(selected)
    return kept_rows


def _layer_split_stats(rows: List[Dict[str, Any]], deduped_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    unique_signatures = len({str(row.get("candidate_signature", "")) for row in rows if str(row.get("candidate_signature", ""))})
    dedup_unique_signatures = len({str(row.get("candidate_signature", "")) for row in deduped_rows if str(row.get("candidate_signature", ""))})
    row_count = int(len(rows))
    dedup_row_count = int(len(deduped_rows))
    win_count = int(sum(int(_safe_int(row.get("win_label", 0), 0)) for row in deduped_rows))
    risk_count = int(sum(int(_safe_int(row.get("risk_label", 0), 0)) for row in deduped_rows))
    per_scale_counts: Dict[str, int] = defaultdict(int)
    per_scale_unique: Dict[str, set] = defaultdict(set)
    for row in deduped_rows:
        scale_name = str(row.get("scale", "")).upper()
        per_scale_counts[scale_name] += 1
        if str(row.get("candidate_signature", "")):
            per_scale_unique[scale_name].add(str(row.get("candidate_signature", "")))
    return {
        "rows_pre_dedup": row_count,
        "rows_post_dedup": dedup_row_count,
        "win_label_count": win_count,
        "risk_label_count": risk_count,
        "unique_candidate_signature_count": int(unique_signatures),
        "unique_candidate_signature_count_post_dedup": int(dedup_unique_signatures),
        "unique_signature_ratio": float(unique_signatures / max(1, row_count)),
        "signature_duplicate_ratio": float(1.0 - unique_signatures / max(1, row_count)),
        "positive_rate": float(win_count / max(1, dedup_row_count)),
        "risk_rate": float(risk_count / max(1, dedup_row_count)),
        "scale_coverage": dict(sorted(per_scale_counts.items())),
        "scale_unique_signature_count": {key: int(len(val)) for key, val in sorted(per_scale_unique.items())},
    }


def _build_dataset_report(sample_counts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"layers": {}}
    for layer in ["X", "Z"]:
        layer_rows_pre = 0
        layer_rows_post = 0
        layer_win = 0
        layer_risk = 0
        layer_unique = 0
        scale_coverage: Dict[str, int] = defaultdict(int)
        for split_name, stats in (sample_counts.get(layer, {}) or {}).items():
            layer_rows_pre += int(stats.get("rows_pre_dedup", 0))
            layer_rows_post += int(stats.get("rows_post_dedup", 0))
            layer_win += int(stats.get("win_label_count", 0))
            layer_risk += int(stats.get("risk_label_count", 0))
            layer_unique += int(stats.get("unique_candidate_signature_count", 0))
            for scale_name, count in (stats.get("scale_coverage", {}) or {}).items():
                scale_coverage[str(scale_name).upper()] += int(count)
        report["layers"][layer] = {
            "rows_pre_dedup": int(layer_rows_pre),
            "rows_post_dedup": int(layer_rows_post),
            "positive_rate": float(layer_win / max(1, layer_rows_post)),
            "risk_rate": float(layer_risk / max(1, layer_rows_post)),
            "signature_duplicate_rate": float(1.0 - layer_unique / max(1, layer_rows_pre)),
            "scale_coverage": dict(sorted(scale_coverage.items())),
        }
    return report


def _write_layer_augmented_proxy_csv(path: str, iter_rows: List[Dict[str, Any]]):
    proxy_rows: List[Dict[str, Any]] = []
    for row in iter_rows:
        focus = str(row.get("focus", "")).upper()
        out = {
            "iter": int(row.get("iter", -1)),
            "focus": focus,
            "x_proxy": "",
            "y_proxy": "",
            "z_proxy": "",
            "u_proxy": "",
            "focused_local_obj": row.get("local_obj", ""),
            "focused_augmented_obj": row.get("augmented_obj", ""),
            "baseline_augmented_obj": row.get("baseline_augmented_obj", ""),
            "commit_decision": row.get("commit_decision", row.get("accepted_type", "")),
            "best_z": row.get("best_z", ""),
        }
        if focus == "X":
            out["x_proxy"] = row.get("augmented_obj", "")
        elif focus == "Y":
            out["y_proxy"] = row.get("augmented_obj", "")
        elif focus == "Z":
            out["z_proxy"] = row.get("augmented_obj", "")
        elif focus == "U":
            out["u_proxy"] = row.get("augmented_obj", "")
        proxy_rows.append(out)
    _write_csv(path, proxy_rows)


def _write_layer_solution_audit(path: str, opt) -> None:
    problem = getattr(opt, "problem", None)
    with open(path, "w", encoding="utf-8") as f:
        if problem is None:
            f.write("No problem state available.\n")
            return

        def _sku_count_dict_from_ids(sku_ids):
            counts: Dict[int, int] = {}
            for sku_id in sku_ids or []:
                sid = int(getattr(sku_id, "id", sku_id))
                counts[sid] = counts.get(sid, 0) + 1
            return dict(sorted(counts.items()))

        def _sku_count_dict_from_objs(skus):
            counts: Dict[int, int] = {}
            for sku in skus or []:
                sid = int(getattr(sku, "id", -1))
                counts[sid] = counts.get(sid, 0) + 1
            return dict(sorted(counts.items()))

        all_tasks = _collect_all_tasks(problem)
        robot_to_tasks: Dict[int, List[Any]] = {}
        for task in all_tasks:
            rid = int(getattr(task, "robot_id", -1))
            robot_to_tasks.setdefault(rid, []).append(task)

        f.write("[ALNS Relax Solution Audit]\n")
        f.write(f"global_makespan={float(getattr(problem, 'global_makespan', 0.0)):.6f}\n")
        f.write(f"robot_capacity={int(getattr(OFSConfig, 'ROBOT_CAPACITY', 0))}\n")
        f.write(f"picking_time={float(getattr(OFSConfig, 'PICKING_TIME', 0.0)):.6f}\n")
        f.write(f"move_extra_tote_time={float(getattr(OFSConfig, 'MOVE_EXTRA_TOTE_TIME', 0.0)):.6f}\n")
        f.write(f"task_count={int(len(all_tasks))}\n")

        f.write("\n[BOM Requirements]\n")
        for order in sorted(getattr(problem, "order_list", []) or [], key=lambda x: int(getattr(x, "order_id", -1))):
            order_skus = getattr(order, "order_product_id_list", []) or []
            f.write(
                f"order_id={int(order.order_id)}, sku_unit_count={int(len(order_skus))}, "
                f"unique_sku_count={int(len(set(order_skus)))}, sku_qty={_sku_count_dict_from_ids(order_skus)}\n"
            )

        f.write("\n[SP1 Allocation]\n")
        for st in sorted(getattr(problem, "subtask_list", []) or [], key=lambda x: int(getattr(x, "id", -1))):
            unique_skus = sorted(int(getattr(s, "id", -1)) for s in (getattr(st, "unique_sku_list", []) or []))
            f.write(
                f"subtask_id={int(st.id)}, order_id={int(getattr(st.parent_order, 'order_id', -1))}, "
                f"sku_unit_count={int(len(getattr(st, 'sku_list', []) or []))}, "
                f"unique_sku_count={int(len(unique_skus))}, unique_skus={unique_skus}, "
                f"sku_qty={_sku_count_dict_from_objs(getattr(st, 'sku_list', []) or [])}\n"
            )

        f.write("\n[SP2 Assignment And Sequence]\n")
        for st in sorted(
            getattr(problem, "subtask_list", []) or [],
            key=lambda x: (
                int(getattr(x, "assigned_station_id", -1)),
                int(getattr(x, "station_sequence_rank", -1)),
                int(getattr(x, "id", -1)),
            ),
        ):
            f.write(
                f"subtask_id={int(st.id)}, station_id={int(getattr(st, 'assigned_station_id', -1))}, "
                f"rank={int(getattr(st, 'station_sequence_rank', -1))}, "
                f"estimated_process_start_time={float(getattr(st, 'estimated_process_start_time', 0.0)):.6f}, "
                f"completion_time={float(getattr(st, 'completion_time', 0.0)):.6f}\n"
            )

        f.write("\n[SP3 Task Fields]\n")
        for task in sorted(all_tasks, key=lambda t: int(getattr(t, "task_id", -1))):
            f.write(f"task_id={int(getattr(task, 'task_id', -1))}\n")
            for key, value in sorted(getattr(task, "__dict__", {}).items(), key=lambda kv: kv[0]):
                f.write(f"  {key}={value}\n")

        f.write("\n[SP4 Robot Paths]\n")
        for rid in sorted(robot_to_tasks.keys()):
            f.write(f"robot_id={rid}\n")
            seq = sorted(
                robot_to_tasks[rid],
                key=lambda t: (
                    int(getattr(t, "trip_id", 0)),
                    float(getattr(t, "arrival_time_at_stack", 0.0)),
                    int(getattr(t, "task_id", -1)),
                ),
            )
            for task in seq:
                f.write(
                    f"  task_id={int(getattr(task, 'task_id', -1))}, subtask_id={int(getattr(task, 'sub_task_id', -1))}, "
                    f"trip_id={int(getattr(task, 'trip_id', 0))}, target_stack_id={int(getattr(task, 'target_stack_id', -1))}, "
                    f"target_station_id={int(getattr(task, 'target_station_id', -1))}, "
                    f"arrival_stack={float(getattr(task, 'arrival_time_at_stack', 0.0)):.6f}, "
                    f"arrival_station={float(getattr(task, 'arrival_time_at_station', 0.0)):.6f}, "
                    f"robot_service_time={float(getattr(task, 'robot_service_time', 0.0)):.6f}, "
                    f"load={int(getattr(task, 'total_load_count', 0))}\n"
                )

        f.write("\n[Quick Checks]\n")
        issues = []
        robot_capacity = int(getattr(OFSConfig, "ROBOT_CAPACITY", 0))
        for task in all_tasks:
            load = int(getattr(task, "total_load_count", 0))
            if load > robot_capacity:
                issues.append(f"task {int(task.task_id)} load {load} exceeds robot capacity {robot_capacity}")
            if float(getattr(task, "end_process_time", 0.0)) + 1e-6 < float(getattr(task, "start_process_time", 0.0)):
                issues.append(f"task {int(task.task_id)} end before start")
        if not issues:
            f.write("No obvious hard-constraint violations found in quick checks.\n")
        else:
            for item in issues:
                f.write(f"{item}\n")


def _save_line_plot_png(path: str, title: str, xlabel: str, ylabel: str,
                        series: List[Tuple[List[float], List[float], str]]) -> str:
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # pragma: no cover
        return f"png fallback skipped: {e}"

    valid_series = []
    for xs, ys, label in series:
        pts = []
        for x, y in zip(xs, ys):
            if x is None or y is None:
                continue
            xf = float(x)
            yf = float(y)
            if math.isfinite(xf) and math.isfinite(yf):
                pts.append((xf, yf))
        if pts:
            valid_series.append((pts, label))
    if not valid_series:
        return "png fallback skipped: no valid data"

    width, height = 1200, 700
    left, right, top, bottom = 90, 30, 70, 90
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    all_x = [x for pts, _ in valid_series for x, _ in pts]
    all_y = [y for pts, _ in valid_series for _, y in pts]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if abs(max_x - min_x) < 1e-9:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-9:
        pad = max(1.0, abs(max_y) * 0.05)
        min_y -= pad
        max_y += pad

    def map_x(x: float) -> float:
        return left + (x - min_x) * plot_w / (max_x - min_x)

    def map_y(y: float) -> float:
        return top + plot_h - (y - min_y) * plot_h / (max_y - min_y)

    grid_color = (220, 220, 220)
    axis_color = (0, 0, 0)
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
    ]

    draw.text((left, 20), title, fill=axis_color)
    draw.line((left, top, left, top + plot_h), fill=axis_color, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis_color, width=2)

    tick_count = 5
    for i in range(tick_count + 1):
        ratio = i / tick_count
        x = left + ratio * plot_w
        y = top + ratio * plot_h
        draw.line((x, top, x, top + plot_h), fill=grid_color, width=1)
        draw.line((left, y, left + plot_w, y), fill=grid_color, width=1)
        xv = min_x + ratio * (max_x - min_x)
        yv = max_y - ratio * (max_y - min_y)
        draw.text((x - 10, top + plot_h + 8), f"{xv:.0f}", fill=axis_color)
        draw.text((10, y - 7), f"{yv:.1f}", fill=axis_color)

    for idx, (pts, label) in enumerate(valid_series):
        color = colors[idx % len(colors)]
        mapped = [(map_x(x), map_y(y)) for x, y in pts]
        if len(mapped) == 1:
            x, y = mapped[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=color)
        else:
            draw.line(mapped, fill=color, width=3)
            for x, y in mapped:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=color)
        legend_x = left + 15 + idx * 220
        legend_y = height - 35
        draw.line((legend_x, legend_y + 8, legend_x + 24, legend_y + 8), fill=color, width=3)
        draw.text((legend_x + 32, legend_y), label, fill=axis_color)

    draw.text((width // 2 - 20, height - 20), xlabel, fill=axis_color)
    draw.text((10, top - 20), ylabel, fill=axis_color)
    image.save(path, format="PNG")
    return ""


def _build_summary_rows(baseline_rows: List[Dict[str, Any]], tra_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    scales = sorted({r["scale"] for r in baseline_rows} | {r["scale"] for r in tra_rows})
    for scale in scales:
        b = [r for r in baseline_rows if r.get("scale") == scale]
        t = [r for r in tra_rows if r.get("scale") == scale and r.get("status") == "ok"]

        b_makespan = [float(r.get("global_makespan", 0.0)) for r in b]
        b_runtime = [float(r.get("runtime_total_sec", 0.0)) for r in b]
        t_makespan = [float(r.get("best_z", 0.0)) for r in t]
        t_runtime = [float(r.get("runtime_sec", 0.0)) for r in t]

        row = {
            "scale": scale,
            "baseline_count": len(b),
            "tra_count": len(t),
            "baseline_makespan_mean": _safe_mean(b_makespan),
            "baseline_runtime_mean": _safe_mean(b_runtime),
            "tra_makespan_mean": _safe_mean(t_makespan),
            "tra_makespan_best": min(t_makespan) if t_makespan else 0.0,
            "tra_makespan_worst": max(t_makespan) if t_makespan else 0.0,
            "tra_makespan_std": _safe_std(t_makespan),
            "tra_cv": (_safe_std(t_makespan) / _safe_mean(t_makespan)) if _safe_mean(t_makespan) > 1e-9 else 0.0,
            "tra_p95": _percentile(t_makespan, 0.95),
            "tra_p99": _percentile(t_makespan, 0.99),
            "tra_runtime_mean": _safe_mean(t_runtime),
            "tra_runtime_total": float(sum(t_runtime)),
            "tra_failure_count": len([r for r in tra_rows if r.get("scale") == scale and r.get("status") != "ok"]),
        }
        rows.append(row)
    return rows


def _try_plot(scale_dir: str, scale: str, tra_rows: List[Dict[str, Any]], iter_rows: List[Dict[str, Any]]) -> List[str]:
    notes = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        plt = None
        notes.append(f"plot fallback: {e}")

    this_scale = sorted([r for r in tra_rows if r.get("scale") == scale and r.get("status") == "ok"],
                        key=lambda x: int(x.get("run_id", 0)))
    if not this_scale:
        notes.append("plot skipped: no successful TRA rows")
        return notes

    run_ids = [int(r["run_id"]) for r in this_scale]
    best_z = [float(r["best_z"]) for r in this_scale]

    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(run_ids, best_z, linewidth=1.0)
        plt.title(f"TRA best_z per run - {scale}")
        plt.xlabel("run_id")
        plt.ylabel("best_z")
        plt.tight_layout()
        plt.savefig(os.path.join(scale_dir, "tra_bestz_vs_run.png"), dpi=140)
        plt.close()
    else:
        note = _save_line_plot_png(
            os.path.join(scale_dir, "tra_bestz_vs_run.png"),
            f"TRA best_z per run - {scale}",
            "run_id",
            "best_z",
            [(run_ids, best_z, "best_z")],
        )
        if note:
            notes.append(note)

    window = max(5, min(50, len(best_z) // 10 if len(best_z) >= 10 else 5))
    rolling = []
    for i in range(len(best_z)):
        left = max(0, i - window + 1)
        rolling.append(_safe_mean(best_z[left:i + 1]))

    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(run_ids, rolling, linewidth=1.2)
        plt.title(f"TRA rolling mean best_z (window={window}) - {scale}")
        plt.xlabel("run_id")
        plt.ylabel("rolling_mean_best_z")
        plt.tight_layout()
        plt.savefig(os.path.join(scale_dir, "tra_bestz_rolling_mean.png"), dpi=140)
        plt.close()
    else:
        note = _save_line_plot_png(
            os.path.join(scale_dir, "tra_bestz_rolling_mean.png"),
            f"TRA rolling mean best_z (window={window}) - {scale}",
            "run_id",
            "rolling_mean_best_z",
            [(run_ids, rolling, "rolling_mean")],
        )
        if note:
            notes.append(note)

    # Iter profile sample: run 0, median, best
    candidate_run_ids = []
    candidate_run_ids.append(run_ids[0])
    candidate_run_ids.append(run_ids[len(run_ids) // 2])
    best_row = min(this_scale, key=lambda x: float(x.get("best_z", float("inf"))))
    candidate_run_ids.append(int(best_row["run_id"]))
    candidate_run_ids = sorted(set(candidate_run_ids))

    sample_series = []
    for rid in candidate_run_ids:
        seq = [r for r in iter_rows if r.get("scale") == scale and int(r.get("run_id", -1)) == rid]
        seq = sorted(seq, key=lambda x: int(x.get("iter", 0)))
        xs = [int(r["iter"]) for r in seq]
        ys = [float(r["best_z"]) for r in seq]
        if xs and ys:
            sample_series.append((xs, ys, f"run={rid}"))

    if plt is not None:
        plt.figure(figsize=(10, 4))
        for xs, ys, label in sample_series:
            plt.plot(xs, ys, linewidth=1.2, label=label)
        plt.title(f"TRA iteration profile sample - {scale}")
        plt.xlabel("iter")
        plt.ylabel("best_z")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(scale_dir, "tra_iter_profile_sample.png"), dpi=140)
        plt.close()
    else:
        note = _save_line_plot_png(
            os.path.join(scale_dir, "tra_iter_profile_sample.png"),
            f"TRA iteration profile sample - {scale}",
            "iter",
            "best_z",
            sample_series,
        )
        if note:
            notes.append(note)

    return notes


def _write_best_iter_plot(scale_dir: str, scale: str, iter_rows: List[Dict[str, Any]], best_run_id: int) -> List[str]:
    notes = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        plt = None
        notes.append(f"best-iter-plot fallback: {e}")

    seq = [r for r in iter_rows if r.get("scale") == scale and int(r.get("run_id", -1)) == int(best_run_id)]
    seq = sorted(seq, key=lambda x: int(x.get("iter", 0)))
    xs = [int(r["iter"]) for r in seq]
    ys = [float(r["best_z"]) for r in seq]
    if not xs or not ys:
        notes.append("best-iter-plot skipped: no best-run iter rows")
        return notes

    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.4)
        plt.title(f"TRA best_z iteration curve - {scale} - run {best_run_id}")
        plt.xlabel("iter")
        plt.ylabel("best_z")
        plt.tight_layout()
        plt.savefig(os.path.join(scale_dir, "tra_best_run_iter_curve.png"), dpi=140)
        plt.close()
    else:
        note = _save_line_plot_png(
            os.path.join(scale_dir, "tra_best_run_iter_curve.png"),
            f"TRA best_z iteration curve - {scale} - run {best_run_id}",
            "iter",
            "best_z",
            [(xs, ys, f"run={best_run_id}")],
        )
        if note:
            notes.append(note)
    return notes


def _write_report(path: str, summary_rows: List[Dict[str, Any]], scales: List[str], tra_runs: int):
    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- scales: {', '.join(scales)}")
    lines.append(f"- tra_runs_per_scale: {tra_runs}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for row in summary_rows:
        lines.append(
            f"- {row['scale']}: baseline_mean={row['baseline_makespan_mean']:.3f}, "
            f"tra_mean={row['tra_makespan_mean']:.3f}, tra_best={row['tra_makespan_best']:.3f}, "
            f"tra_std={row['tra_makespan_std']:.3f}, tra_runtime_total={row['tra_runtime_total']:.2f}s"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")




def run_soft_coupling_ablation_table(
        scales: List[str],
        iter_limit: int = 1000,
        seed: int = 42,
        sp2_time_limit_sec: float = 10.0,
        sp4_lkh_time_limit_seconds: int = 5,
        enable_sp3_precheck: bool = True,
        precheck_fail_action: str = "log",
        result_root: str = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = result_root or _ensure_dir(os.path.join(ROOT_DIR, "result", f"soft_coupling_table_{timestamp}"))
    scales = [s.upper() for s in scales[:5]]

    algorithm_specs = [
        ("baseline", {}),
        ("tra_none", {}),
        ("tra_all", {
            "enable_soft_mu": True,
            "enable_soft_pi": True,
            "enable_soft_beta": True,
            "enable_sku_affinity": True,
        }),
        ("tra_mu", {"enable_soft_mu": True}),
        ("tra_pi", {"enable_soft_pi": True}),
        ("tra_beta", {"enable_soft_beta": True}),
        ("tra_affinity", {"enable_sku_affinity": True}),
    ]

    rows: List[Dict[str, Any]] = []
    total_jobs = len(scales) * len(algorithm_specs)
    done_jobs = 0
    current_iter = 0
    current_iter_total = max(1, int(iter_limit))
    current_focus = ""

    def _progress_lines(scale_name: str, algo_name: str):
        outer_width = 24
        inner_width = 24
        outer_filled = int(outer_width * done_jobs / max(1, total_jobs))
        inner_filled = int(inner_width * current_iter / max(1, current_iter_total))
        outer_bar = "#" * outer_filled + "-" * (outer_width - outer_filled)
        inner_bar = "#" * inner_filled + "-" * (inner_width - inner_filled)
        line1 = f"\r[Overall] [{outer_bar}] {done_jobs}/{total_jobs} | scale={scale_name} | algorithm={algo_name}"
        line2 = f"\n[Inner  ] [{inner_bar}] {current_iter}/{current_iter_total} | focus={current_focus or '-'}"
        return line1 + line2

    def _render_progress(scale_name: str, algo_name: str):
        sys.__stdout__.write(_progress_lines(scale_name, algo_name))
        sys.__stdout__.flush()

    def _run_quiet(fn):
        old_flag = os.environ.get("OFS_BATCH_SILENT")
        os.environ["OFS_BATCH_SILENT"] = "1"
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                result = fn()
        if old_flag is None:
            os.environ.pop("OFS_BATCH_SILENT", None)
        else:
            os.environ["OFS_BATCH_SILENT"] = old_flag
        return result

    for scale in scales:
        for algorithm_name, overrides in algorithm_specs:
            current_iter = 0
            current_iter_total = 1 if algorithm_name == "baseline" else max(1, int(iter_limit))
            current_focus = "init"
            _render_progress(scale, algorithm_name)
            t0 = time.perf_counter()
            if algorithm_name == "baseline":
                baseline_row = _run_quiet(lambda: run_baseline_once(
                    scale=scale,
                    seed=seed,
                    sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
                ))
                current_iter = 1
                current_focus = "done"
                _render_progress(scale, algorithm_name)
                row = {
                    "scale": scale,
                    "algorithm": algorithm_name,
                    "iter_count": 1,
                    "best_z": float(baseline_row["global_makespan"]),
                    "sub_obj1_subtask_count": float(baseline_row.get("sub_obj1_subtask_count", 0.0)),
                    "sub_obj2_station_idle_total": float(baseline_row.get("sub_obj2_station_idle_total", 0.0)),
                    "sub_obj3_hit_stack_count": float(baseline_row.get("sub_obj3_hit_stack_count", 0.0)),
                    "sub_obj4_robot_path_length": float(baseline_row.get("sub_obj4_robot_path_length", 0.0)),
                    "solve_time_sec": float(time.perf_counter() - t0),
                    "status": "ok",
                }
            else:
                def _progress_cb(payload: Dict[str, Any]):
                    nonlocal current_iter, current_iter_total, current_focus
                    current_iter = int(payload.get("iter", 0))
                    current_iter_total = max(1, int(payload.get("total_iters", current_iter_total)))
                    current_focus = str(payload.get("focus", current_focus))
                    _render_progress(scale, algorithm_name)

                cfg = _make_tra_config(
                    scale=scale,
                    seed=seed,
                    max_iters=int(iter_limit),
                    no_improve_limit=int(iter_limit),
                    epsilon=0.05,
                    sp2_time_limit_sec=sp2_time_limit_sec,
                    sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
                    enable_sp3_precheck=enable_sp3_precheck,
                    precheck_fail_action=precheck_fail_action,
                    enable_soft_mu=bool(overrides.get("enable_soft_mu", False)),
                    enable_soft_pi=bool(overrides.get("enable_soft_pi", False)),
                    enable_soft_beta=bool(overrides.get("enable_soft_beta", False)),
                    enable_sku_affinity=bool(overrides.get("enable_sku_affinity", False)),
                )
                cfg.progress_callback = _progress_cb
                status = "ok"
                best_z = float("nan")
                layer_obj = {
                    "sub_obj1_subtask_count": float("nan"),
                    "sub_obj2_station_idle_total": float("nan"),
                    "sub_obj3_hit_stack_count": float("nan"),
                    "sub_obj4_robot_path_length": float("nan"),
                }
                try:
                    opt = _run_quiet(lambda: TRAOptimizer(cfg))
                    best_z = float(_run_quiet(lambda: opt.run()))
                    if getattr(opt, "precheck_status", None):
                        status = str(opt.precheck_status)
                    elif math.isnan(best_z):
                        status = "nan"
                    else:
                        layer_obj = _run_quiet(lambda: _materialize_best_tra_solution(opt))
                except Exception as e:
                    status = f"error:{e}"

                row = {
                    "scale": scale,
                    "algorithm": algorithm_name,
                    "iter_count": int(iter_limit),
                    "best_z": float(best_z),
                    "sub_obj1_subtask_count": float(layer_obj["sub_obj1_subtask_count"]),
                    "sub_obj2_station_idle_total": float(layer_obj["sub_obj2_station_idle_total"]),
                    "sub_obj3_hit_stack_count": float(layer_obj["sub_obj3_hit_stack_count"]),
                    "sub_obj4_robot_path_length": float(layer_obj["sub_obj4_robot_path_length"]),
                    "solve_time_sec": float(time.perf_counter() - t0),
                    "status": status,
                }
            rows.append(row)
            done_jobs += 1
            current_focus = "done"
            _render_progress(scale, algorithm_name)

    sys.__stdout__.write("\n")
    sys.__stdout__.flush()
    _write_csv(os.path.join(result_root, "soft_coupling_ablation_table.csv"), rows)
    with open(os.path.join(result_root, "soft_coupling_ablation_table.txt"), "w", encoding="utf-8") as f:
        headers = [
            "scale",
            "algorithm",
            "iter_count",
            "best_z",
            "sub_obj1_subtask_count",
            "sub_obj2_station_idle_total",
            "sub_obj3_hit_stack_count",
            "sub_obj4_robot_path_length",
            "solve_time_sec",
            "status",
        ]
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
    _write_json(os.path.join(result_root, "soft_coupling_ablation_meta.json"), {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scales": scales,
        "iter_limit": int(iter_limit),
        "seed": int(seed),
        "algorithms": [name for name, _ in algorithm_specs],
    })
    print(f"[SoftCouplingTable] done. result_root={result_root}")
    return result_root


def _run_tra_vns_case_export(scale: str, args, tra_max_iters: int = 10, vns_max_trials: int = 10,
                             result_tag: str = "tra_vns_case") -> str:
    scale = str(scale).upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"{result_tag}_{scale}_{timestamp}"))
    cfg = _make_tra_config(
        scale=scale,
        seed=int(args.base_seed),
        max_iters=int(tra_max_iters),
        no_improve_limit=int(tra_max_iters),
        epsilon=float(args.epsilon),
        sp2_time_limit_sec=float(args.sp2_time_limit_sec),
        sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
        enable_sp3_precheck=bool(args.precheck_sp3),
        precheck_fail_action=str(args.precheck_fail),
        enable_soft_mu=bool(args.enable_soft_mu),
        enable_soft_pi=bool(args.enable_soft_pi),
        enable_soft_beta=bool(args.enable_soft_beta),
        enable_sku_affinity=bool(args.enable_sku_affinity),
        mu_value=float(args.mu_value),
        pi_scale=float(args.pi_scale),
        pi_clip=float(args.pi_clip),
        d0_threshold=float(args.d0_threshold),
        beta_base=float(args.beta_base),
        beta_gain=float(args.beta_gain),
        beta_min=float(args.beta_min),
        beta_max=float(args.beta_max),
        sp2_shadow_weight=float(args.sp2_shadow_weight),
        enable_role_vns=False,
        eps_skip=float(args.eps_skip),
        eps_light=float(args.eps_light),
        weak_accept_eta=float(args.weak_accept_eta),
        vns_max_trials=int(vns_max_trials),
        mode_fail_limit=int(args.mode_fail_limit),
    )
    cfg.export_best_solution = True
    cfg.write_iteration_logs = True
    cfg.enable_sp1_feedback_analysis = False

    t0 = time.perf_counter()
    opt = TRAOptimizer(cfg)
    best_z = float(opt.run())
    total_runtime_sec = float(time.perf_counter() - t0)

    iter_rows = list(opt.iter_log)
    best_export_dir = os.path.join(ROOT_DIR, "log", "tra_best_export")
    best_solution_objectives = {}
    best_verification = {}
    if os.path.exists(os.path.join(best_export_dir, "best_solution_objectives.json")):
        with open(os.path.join(best_export_dir, "best_solution_objectives.json"), "r", encoding="utf-8") as f:
            best_solution_objectives = json.load(f)
    if os.path.exists(os.path.join(best_export_dir, "tra_makespan_verification.json")):
        with open(os.path.join(best_export_dir, "tra_makespan_verification.json"), "r", encoding="utf-8") as f:
            best_verification = json.load(f)
    _write_csv(os.path.join(result_root, "tra_vns_small_iter_log.csv"), iter_rows)
    _write_json(os.path.join(result_root, "tra_vns_small_iter_log.json"), iter_rows)

    summary = {
        "scale": scale,
        "seed": int(args.base_seed),
        "tra_max_iters": int(tra_max_iters),
        "vns_max_trials": int(vns_max_trials),
        "best_z": float(best_z),
        "total_runtime_sec": float(total_runtime_sec),
        "iter_count": int(len(iter_rows)),
        "mode_stats": getattr(opt, "mode_stats", {}),
        "final_metrics": (iter_rows[-1] if iter_rows else {}),
        "best_solution_objectives": best_solution_objectives,
        "best_verification": best_verification,
    }
    _write_json(os.path.join(result_root, "tra_vns_small_summary.json"), summary)
    with open(os.path.join(result_root, "tra_vns_small_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"scale={scale}\nseed={int(args.base_seed)}\n")
        f.write(f"tra_max_iters={int(tra_max_iters)}\nvns_max_trials={int(vns_max_trials)}\n")
        f.write(f"best_z={float(best_z):.6f}\n")
        f.write(f"total_runtime_sec={float(total_runtime_sec):.6f}\n")
        f.write(f"iter_count={int(len(iter_rows))}\n")
    print(f"[TRA-VNS-CASE] done. scale={scale}, result_root={result_root}")
    return result_root


def run_tra_vns_small_test(args) -> str:
    return _run_tra_vns_case_export(
        scale="SMALL",
        args=args,
        tra_max_iters=10,
        vns_max_trials=10,
        result_tag="tra_vns_small",
    )


def _layer_commit_count(iter_rows: List[Dict[str, Any]], layer: str) -> int:
    target = str(layer).upper()
    return int(sum(
        1
        for row in (iter_rows or [])
        if str(row.get("focus", "")).upper() == target
        and str(row.get("commit_decision", row.get("accepted_type", ""))).lower() in {"accept", "append_shadow"}
    ))


def _nonfinite_iter_row_count(iter_rows: List[Dict[str, Any]]) -> int:
    count = 0
    for row in (iter_rows or []):
        if int(row.get("iter", -1) or -1) <= 0:
            continue
        bad = False
        for key in ["local_obj", "augmented_obj"]:
            value = row.get(key, None)
            if value is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if not math.isfinite(numeric):
                bad = True
                break
        if bad:
            count += 1
    return int(count)


def run_small_layer_augmented_case_export(
    seed: int = 42,
    max_iters: int = 10,
    case: str = "SMALL",
    no_improve_limit: int = 3,
    epsilon: float = 0.05,
    sp2_time_limit_sec: float = 10.0,
    sp4_lkh_time_limit_seconds: int = 5,
    export_best_solution: bool = True,
    silent: bool = True,
    xz_evaluator_mode: str = "neural",
    result_tag_suffix: str = "",
    config_hook: Optional[Callable[[TRARunConfig], None]] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = str(result_tag_suffix or "").strip().strip("_")
    result_tag: str = f"{case}_{seed}_layer_augmented"
    if suffix:
        result_tag = f"{result_tag}_{suffix}"
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"{result_tag}_{timestamp}"))

    cfg = _make_tra_config(
        scale=case,
        seed=int(seed),
        max_iters=int(max_iters),
        no_improve_limit=int(no_improve_limit),
        epsilon=float(epsilon),
        sp2_time_limit_sec=float(sp2_time_limit_sec),
        sp4_lkh_time_limit_seconds=int(sp4_lkh_time_limit_seconds),
        enable_role_vns=False,
        enable_shadow_chain=True,
        shadow_chain_max_depth=3,
    )
    cfg.search_scheme = "layer_augmented"
    cfg.xz_evaluator_mode = str(xz_evaluator_mode).strip().lower() or "neural"
    cfg.log_dir = result_root
    cfg.write_iteration_logs = True
    cfg.export_best_solution = bool(export_best_solution)
    cfg.enable_sp1_feedback_analysis = False
    cfg.target_runtime_sec = 85.0
    cfg.runtime_guard_mode = "soft"
    cfg.x_eval_all_candidates = False
    cfg.x_global_eval_topk = 2
    cfg.x_f2_topk = 3
    cfg.x_dual_eval_gap_ratio = 0.04
    cfg.x_operator_pair_budget = 8
    cfg.x_destroy_size_max = 4
    cfg.x_micro_move_order_cap = 2
    cfg.x_micro_move_group_cap = 2
    cfg.z_structural_eval_topk = 2
    cfg.z_global_eval_topk = 1
    cfg.z_f2_topk = 2
    cfg.z_dual_eval_gap_ratio = 0.05
    cfg.z_route_pressure_weight = 1.0
    cfg.z_station_load_weight = 0.75
    cfg.z_processing_overflow_weight = 1.0
    cfg.z_hotspot_batch_size = 5
    cfg.z_min_budget = 3
    cfg.z_f1_trip_cap = 6
    cfg.x_surrogate_bootstrap_eval_budget = 3
    cfg.z_surrogate_bootstrap_eval_budget = 3
    cfg.z_local_delta_task_cap = 3
    cfg.z_local_delta_stack_cap = 2
    cfg.layer_operator_budget_x = 4
    cfg.layer_operator_budget_y = 6
    cfg.layer_operator_budget_z = 3
    cfg.layer_operator_budget_u = 1
    cfg.y_route_eval_topk = 2
    cfg.y_global_eval_topk = 2
    cfg.u_default_budget_when_triggered = 1
    if callable(config_hook):
        config_hook(cfg)

    opt = TRAOptimizer(cfg)

    def _runner():
        t0 = time.perf_counter()
        best_z_val = float(opt.run())
        runtime_sec = float(time.perf_counter() - t0)
        return best_z_val, runtime_sec

    if silent:
        old_flag = os.environ.get("OFS_BATCH_SILENT")
        os.environ["OFS_BATCH_SILENT"] = "1"
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                best_z, total_runtime_sec = _runner()
        if old_flag is None:
            os.environ.pop("OFS_BATCH_SILENT", None)
        else:
            os.environ["OFS_BATCH_SILENT"] = old_flag
    else:
        best_z, total_runtime_sec = _runner()

    iter_rows = list(getattr(opt, "iter_log", []) or [])
    tra_summary = _read_json(os.path.join(result_root, "tra_summary.json"), {}) or {}
    run_stats = dict(tra_summary.get("run_stats", {}) or {})
    timing_breakdown = dict(
        run_stats.get("timing_breakdown", {})
        or (opt._timing_breakdown_payload() if hasattr(opt, "_timing_breakdown_payload") else {})
    )
    _write_layer_augmented_proxy_csv(os.path.join(result_root, "iter_xyzu_proxy_values.csv"), iter_rows)
    _write_timing_breakdown_files(result_root, case, seed, timing_breakdown)
    _write_generator_summary_files(result_root, case, seed, getattr(opt, "problem", None))

    best_iter = int(getattr(opt.best, "iter_id", -1)) if getattr(opt, "best", None) is not None else -1
    with open(os.path.join(result_root, "run_brief.txt"), "w", encoding="utf-8") as f:
        f.write(f"scale={str(case).upper()}\n")
        f.write(f"seed={int(seed)}\n")
        f.write(f"xz_evaluator_mode={str(getattr(cfg, 'xz_evaluator_mode', 'neural'))}\n")
        f.write(f"total_runtime_sec={float(total_runtime_sec):.6f}\n")
        f.write(f"best_z={float(best_z):.3f}s @ iter={best_iter}\n")
        if timing_breakdown:
            f.write(f"forced_global_eval_time_sec={float(timing_breakdown.get('forced_global_eval_time_sec', 0.0)):.6f}\n")
            f.write(f"x_f1_time_sec={float(timing_breakdown.get('x_f1_time_sec', 0.0)):.6f}\n")
            f.write(f"z_f1_time_sec={float(timing_breakdown.get('z_f1_time_sec', 0.0)):.6f}\n")

    return result_root


def _classic_vs_neural_case_row(case_root: str, case: str, mode: str) -> Dict[str, Any]:
    summary = _read_json(os.path.join(case_root, "tra_summary.json"), {}) or {}
    run_stats = dict(summary.get("run_stats", {}) or {})
    iter_rows = list(summary.get("iters", []) or [])
    classic_verify_count_x = int(run_stats.get("classic_verify_count_x", 0) or 0)
    classic_verify_count_z = int(run_stats.get("classic_verify_count_z", 0) or 0)
    z_feasible_candidate_count = int(sum(
        int(float(row.get("z_feasible_candidate_count", 0.0) or 0.0))
        for row in iter_rows
        if str(row.get("focus", "")).upper() == "Z"
    ))
    top1_x_proxy_improve_but_not_verified_count = int(
        run_stats.get("top1_x_proxy_improve_but_not_verified_count", 0)
        or sum(1 for row in iter_rows if bool(row.get("top1_x_proxy_improve_but_not_verified", False)))
    )
    return {
        "case": str(case).upper(),
        "seed": int((summary.get("config", {}) or {}).get("seed", 42) or 42),
        "xz_evaluator_mode": str(mode),
        "result_root": case_root,
        "best_z": float((summary.get("best", {}) or {}).get("z", 0.0) or 0.0),
        "total_runtime_sec": float(run_stats.get("run_total_time_sec", 0.0) or 0.0),
        "global_eval_count": int(run_stats.get("global_eval_count", 0) or 0),
        "x_f1_time_sec": float(run_stats.get("x_f1_time_sec", 0.0) or 0.0),
        "z_f1_time_sec": float(run_stats.get("z_f1_time_sec", 0.0) or 0.0),
        "inf_rows": int(_nonfinite_iter_row_count(iter_rows)),
        "accepted_or_committed_x_count": int(_layer_commit_count(iter_rows, "X")),
        "accepted_or_committed_z_count": int(_layer_commit_count(iter_rows, "Z")),
        "x_f1_invalid_count": int(run_stats.get("x_f1_invalid_count", 0) or 0),
        "z_f1_invalid_count": int(run_stats.get("z_f1_invalid_count", 0) or 0),
        "classic_verify_count_x": int(classic_verify_count_x),
        "classic_verify_count_z": int(classic_verify_count_z),
        "z_feasible_candidate_count": int(z_feasible_candidate_count),
        "top1_x_proxy_improve_but_not_verified_count": int(top1_x_proxy_improve_but_not_verified_count),
    }


def _classic_vs_neural_report_markdown(mode_rows: List[Dict[str, Any]], compare_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Classic vs Neural X/Z Evaluator Suite",
        "",
        "This report compares `xz_evaluator_mode=classic_soft` against `xz_evaluator_mode=neural` on `SMALL` and `SMALL2` with `seed=42`.",
        "",
        "## Per-mode rows",
        "",
    ]
    for row in mode_rows:
        lines.append(
            f"- {row['case']} | {row['xz_evaluator_mode']}: "
            f"best_z={float(row['best_z']):.3f}, runtime={float(row['total_runtime_sec']):.3f}s, "
            f"global_eval_count={int(row['global_eval_count'])}, inf_rows={int(row['inf_rows'])}, "
            f"x_accept={int(row['accepted_or_committed_x_count'])}, z_accept={int(row['accepted_or_committed_z_count'])}, "
            f"classic_verify_x={int(row.get('classic_verify_count_x', 0))}, "
            f"classic_verify_z={int(row.get('classic_verify_count_z', 0))}, "
            f"z_feasible={int(row.get('z_feasible_candidate_count', 0))}, "
            f"x_proxy_improve_not_verified={int(row.get('top1_x_proxy_improve_but_not_verified_count', 0))}"
        )
    lines.extend(["", "## Delta (classic_soft - neural)", ""])
    for row in compare_rows:
        lines.append(
            f"- {row['case']}: best_z_delta={float(row['best_z_delta_abs']):.3f} "
            f"({float(row['best_z_delta_pct']):.2f}%), runtime_delta={float(row['total_runtime_sec_delta_abs']):.3f}s, "
            f"global_eval_delta={int(row['global_eval_count_delta'])}, "
            f"x_accept_delta={int(row['accepted_or_committed_x_count_delta'])}, "
            f"z_accept_delta={int(row['accepted_or_committed_z_count_delta'])}, "
            f"classic_verify_x_delta={int(row.get('classic_verify_count_x_delta', 0))}, "
            f"classic_verify_z_delta={int(row.get('classic_verify_count_z_delta', 0))}, "
            f"z_feasible_delta={int(row.get('z_feasible_candidate_count_delta', 0))}, "
            f"x_proxy_improve_not_verified_delta={int(row.get('top1_x_proxy_improve_but_not_verified_count_delta', 0))}"
        )
    return "\n".join(lines) + "\n"


def run_classic_vs_neural_suite(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"classic_vs_neural_suite_{timestamp}"))
    mode_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    for case in ["SMALL", "SMALL2"]:
        case_rows: Dict[str, Dict[str, Any]] = {}
        for mode in ["classic_soft", "neural"]:
            case_root = run_small_layer_augmented_case_export(
                seed=42,
                max_iters=int(args.tra_max_iters),
                case=case,
                no_improve_limit=int(args.no_improve_limit),
                epsilon=float(args.epsilon),
                sp2_time_limit_sec=float(args.sp2_time_limit_sec),
                sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
                export_best_solution=True,
                silent=True,
                xz_evaluator_mode=mode,
                result_tag_suffix=mode,
            )
            case_name = os.path.basename(case_root.rstrip("\\/"))
            dest_root = os.path.join(result_root, case_name)
            if os.path.abspath(case_root) != os.path.abspath(dest_root):
                if os.path.exists(dest_root):
                    shutil.rmtree(dest_root)
                shutil.move(case_root, dest_root)
            row = _classic_vs_neural_case_row(dest_root, case, mode)
            mode_rows.append(row)
            case_rows[mode] = row
        if "classic_soft" in case_rows and "neural" in case_rows:
            classic_row = case_rows["classic_soft"]
            neural_row = case_rows["neural"]
            neural_best = max(1e-9, float(neural_row.get("best_z", 0.0) or 0.0))
            compare_rows.append({
                "case": str(case).upper(),
                "seed": 42,
                "classic_result_root": str(classic_row.get("result_root", "")),
                "neural_result_root": str(neural_row.get("result_root", "")),
                "best_z_classic": float(classic_row.get("best_z", 0.0) or 0.0),
                "best_z_neural": float(neural_row.get("best_z", 0.0) or 0.0),
                "best_z_delta_abs": float(classic_row.get("best_z", 0.0) or 0.0) - float(neural_row.get("best_z", 0.0) or 0.0),
                "best_z_delta_pct": 100.0 * (
                    (float(classic_row.get("best_z", 0.0) or 0.0) - float(neural_row.get("best_z", 0.0) or 0.0))
                    / neural_best
                ),
                "total_runtime_sec_classic": float(classic_row.get("total_runtime_sec", 0.0) or 0.0),
                "total_runtime_sec_neural": float(neural_row.get("total_runtime_sec", 0.0) or 0.0),
                "total_runtime_sec_delta_abs": float(classic_row.get("total_runtime_sec", 0.0) or 0.0) - float(neural_row.get("total_runtime_sec", 0.0) or 0.0),
                "global_eval_count_classic": int(classic_row.get("global_eval_count", 0) or 0),
                "global_eval_count_neural": int(neural_row.get("global_eval_count", 0) or 0),
                "global_eval_count_delta": int(classic_row.get("global_eval_count", 0) or 0) - int(neural_row.get("global_eval_count", 0) or 0),
                "accepted_or_committed_x_count_classic": int(classic_row.get("accepted_or_committed_x_count", 0) or 0),
                "accepted_or_committed_x_count_neural": int(neural_row.get("accepted_or_committed_x_count", 0) or 0),
                "accepted_or_committed_x_count_delta": int(classic_row.get("accepted_or_committed_x_count", 0) or 0) - int(neural_row.get("accepted_or_committed_x_count", 0) or 0),
                "accepted_or_committed_z_count_classic": int(classic_row.get("accepted_or_committed_z_count", 0) or 0),
                "accepted_or_committed_z_count_neural": int(neural_row.get("accepted_or_committed_z_count", 0) or 0),
                "accepted_or_committed_z_count_delta": int(classic_row.get("accepted_or_committed_z_count", 0) or 0) - int(neural_row.get("accepted_or_committed_z_count", 0) or 0),
                "classic_verify_count_x_classic": int(classic_row.get("classic_verify_count_x", 0) or 0),
                "classic_verify_count_x_neural": int(neural_row.get("classic_verify_count_x", 0) or 0),
                "classic_verify_count_x_delta": int(classic_row.get("classic_verify_count_x", 0) or 0) - int(neural_row.get("classic_verify_count_x", 0) or 0),
                "classic_verify_count_z_classic": int(classic_row.get("classic_verify_count_z", 0) or 0),
                "classic_verify_count_z_neural": int(neural_row.get("classic_verify_count_z", 0) or 0),
                "classic_verify_count_z_delta": int(classic_row.get("classic_verify_count_z", 0) or 0) - int(neural_row.get("classic_verify_count_z", 0) or 0),
                "z_feasible_candidate_count_classic": int(classic_row.get("z_feasible_candidate_count", 0) or 0),
                "z_feasible_candidate_count_neural": int(neural_row.get("z_feasible_candidate_count", 0) or 0),
                "z_feasible_candidate_count_delta": int(classic_row.get("z_feasible_candidate_count", 0) or 0) - int(neural_row.get("z_feasible_candidate_count", 0) or 0),
                "top1_x_proxy_improve_but_not_verified_count_classic": int(classic_row.get("top1_x_proxy_improve_but_not_verified_count", 0) or 0),
                "top1_x_proxy_improve_but_not_verified_count_neural": int(neural_row.get("top1_x_proxy_improve_but_not_verified_count", 0) or 0),
                "top1_x_proxy_improve_but_not_verified_count_delta": int(classic_row.get("top1_x_proxy_improve_but_not_verified_count", 0) or 0) - int(neural_row.get("top1_x_proxy_improve_but_not_verified_count", 0) or 0),
            })

    _write_json(
        os.path.join(result_root, "classic_vs_neural_summary.json"),
        {"seed": 42, "mode_rows": mode_rows, "compare_rows": compare_rows},
    )
    _write_csv(os.path.join(result_root, "classic_vs_neural_mode_rows.csv"), mode_rows)
    _write_csv(os.path.join(result_root, "classic_vs_neural_compare_rows.csv"), compare_rows)
    with open(os.path.join(result_root, "classic_vs_neural_report.md"), "w", encoding="utf-8") as f:
        f.write(_classic_vs_neural_report_markdown(mode_rows, compare_rows))
    return result_root


def _is_zrich_case(case: str) -> bool:
    return "ZRICH" in str(case).upper()


def _configure_z_positive_tuning(cfg: TRARunConfig, case: str, tuned: bool) -> None:
    cfg.xz_evaluator_mode = "neural"
    cfg.enable_z_positive_mining_verify = bool(tuned)
    cfg.z_positive_mining_verify_budget_base = 1
    cfg.z_positive_mining_verify_budget_zrich = 2
    if not tuned:
        return
    cfg.layer_operator_budget_z = max(int(getattr(cfg, "layer_operator_budget_z", 3)), 3)
    if _is_zrich_case(case):
        cfg.layer_operator_budget_z = max(int(getattr(cfg, "layer_operator_budget_z", 3)), 5)
        cfg.z_hotspot_batch_size = max(int(getattr(cfg, "z_hotspot_batch_size", 3)), 6)
        cfg.z_f1_topk = max(int(getattr(cfg, "z_f1_topk", 2)), 3)
        cfg.z_f2_topk = max(int(getattr(cfg, "z_f2_topk", 2)), 2)
        cfg.z_global_eval_topk = max(int(getattr(cfg, "z_global_eval_topk", 1)), 1)


def _z_positive_tuning_case_row(case_root: str, case: str, variant: str) -> Dict[str, Any]:
    summary = _read_json(os.path.join(case_root, "tra_summary.json"), {}) or {}
    run_stats = dict(summary.get("run_stats", {}) or {})
    iter_rows = list(summary.get("iters", []) or [])
    candidate_payload = _read_json(os.path.join(case_root, "xz_supervised_candidates.json"), {}) or {}
    z_rows = list(candidate_payload.get("z_rows", []) or [])
    z_positive_rows = [row for row in z_rows if int(row.get("win_label", 0) or 0) > 0]
    z_positive_operator_mix: Dict[str, int] = {}
    z_operator_mix: Dict[str, int] = {}
    z_all_big_negative = bool(z_rows)
    for row in z_rows:
        operator = str(row.get("operator", "")).strip()
        if operator:
            z_operator_mix[operator] = int(z_operator_mix.get(operator, 0)) + 1
        if int(row.get("win_label", 0) or 0) > 0 and operator:
            z_positive_operator_mix[operator] = int(z_positive_operator_mix.get(operator, 0)) + 1
        actual = float(row.get("actual_reduction", 0.0) or 0.0)
        before = max(1.0, float(row.get("global_z_before", 1.0) or 1.0))
        if actual > -max(50.0, 0.1 * before) + 1e-9:
            z_all_big_negative = False
    z_positive_rate = float(len(z_positive_rows) / max(1, len(z_rows))) if z_rows else 0.0
    return {
        "case": str(case).upper(),
        "seed": int((summary.get("config", {}) or {}).get("seed", 42) or 42),
        "variant": str(variant),
        "result_root": case_root,
        "best_z": float((summary.get("best", {}) or {}).get("z", 0.0) or 0.0),
        "total_runtime_sec": float(run_stats.get("run_total_time_sec", 0.0) or 0.0),
        "global_eval_count": int(run_stats.get("global_eval_count", 0) or 0),
        "z_global_eval_count": int(len(z_rows)),
        "z_positive_mining_verify_count": int(run_stats.get("z_positive_mining_verify_count", 0) or 0),
        "z_positive_mining_success_count": int(run_stats.get("z_positive_mining_success_count", 0) or 0),
        "z_positive_candidate_eligible_count": int(run_stats.get("z_positive_candidate_eligible_count", 0) or 0),
        "z_row_count": int(len(z_rows)),
        "z_positive_row_count": int(len(z_positive_rows)),
        "z_positive_rate": float(z_positive_rate),
        "z_positive_operator_mix": dict(sorted(z_positive_operator_mix.items())),
        "z_operator_mix": dict(sorted(z_operator_mix.items())),
        "z_all_big_negative": bool(z_all_big_negative),
        "inf_rows": int(_nonfinite_iter_row_count(iter_rows)),
        "accepted_or_committed_x_count": int(_layer_commit_count(iter_rows, "X")),
        "accepted_or_committed_z_count": int(_layer_commit_count(iter_rows, "Z")),
        "x_f1_invalid_count": int(run_stats.get("x_f1_invalid_count", 0) or 0),
        "z_f1_invalid_count": int(run_stats.get("z_f1_invalid_count", 0) or 0),
        "x_f1_time_sec": float(run_stats.get("x_f1_time_sec", 0.0) or 0.0),
        "z_f1_time_sec": float(run_stats.get("z_f1_time_sec", 0.0) or 0.0),
    }


def _z_positive_tuning_next_actions(
    mode_rows: List[Dict[str, Any]],
    compare_rows: List[Dict[str, Any]],
) -> List[str]:
    tuned_rows = [row for row in mode_rows if str(row.get("variant", "")) == "tuned"]
    z_global_eval_total = int(sum(int(row.get("z_global_eval_count", 0) or 0) for row in tuned_rows))
    z_positive_total = int(sum(int(row.get("z_positive_row_count", 0) or 0) for row in tuned_rows))
    base_positive_total = int(sum(
        int(row.get("z_positive_row_count", 0) or 0)
        for row in tuned_rows
        if not _is_zrich_case(str(row.get("case", "")))
    ))
    zrich_positive_total = int(sum(
        int(row.get("z_positive_row_count", 0) or 0)
        for row in tuned_rows
        if _is_zrich_case(str(row.get("case", "")))
    ))
    regression_rows = [row for row in compare_rows if not _is_zrich_case(str(row.get("case", "")))]
    actions: List[str] = []
    if z_global_eval_total <= 0:
        actions.append("rule1: z_global_eval_count==0 -> relax verify eligibility caps or widen safe operator coverage; do not change Phi_z_main first")
        return actions
    if z_positive_total <= 0 and all(bool(row.get("z_all_big_negative", False)) for row in tuned_rows if int(row.get("z_global_eval_count", 0) or 0) > 0):
        actions.append("rule2: Z rows are all big negatives -> tighten allowlist/noise/mode/fallback tolerance and keep mode_flip_sort_toggle at zero verify probability")
    if z_positive_total > 0 and any(float(row.get("best_z_delta_pct", 0.0) or 0.0) > 5.0 for row in regression_rows):
        actions.append("rule3: positives exist but best_z regressed >5% on regression cases -> reduce z_positive_mining_verify budget before changing operators")
    if z_positive_total > 0 and any(float(row.get("runtime_delta_pct", 0.0) or 0.0) > 5.0 for row in regression_rows):
        actions.append("rule4: positives exist but runtime regressed >5% -> first reduce verify budget, then operator budget, then z_f1_topk")
    if zrich_positive_total > 0 and base_positive_total <= 0:
        actions.append("rule5: positives appear only in ZRICH -> acceptable; keep SMALL/SMALL2 stable and do not force base-case Z positives")
    if not actions:
        actions.append("hold: no rule escalation triggered; keep current mining objective/caps and inspect operator mix for the next pass")
    return actions


def _z_positive_tuning_report_markdown(
    mode_rows: List[Dict[str, Any]],
    compare_rows: List[Dict[str, Any]],
    next_actions: List[str],
) -> str:
    lines = [
        "# Z Positive Mining Tuning Suite",
        "",
        "This report compares the current neural baseline against the shared mainline `z_positive_mining_verify` variant on `SMALL`, `SMALL2`, `SMALL_ZRICH`, and `SMALL2_ZRICH` with `seed=42`.",
        "",
        "## Per-variant rows",
        "",
    ]
    for row in mode_rows:
        lines.append(
            f"- {row['case']} | {row['variant']}: best_z={float(row['best_z']):.3f}, runtime={float(row['total_runtime_sec']):.3f}s, "
            f"global_eval_count={int(row['global_eval_count'])}, z_global_eval_count={int(row['z_global_eval_count'])}, "
            f"z_positive_verify={int(row['z_positive_mining_verify_count'])}, z_positive_rows={int(row['z_positive_row_count'])}, "
            f"z_positive_rate={100.0 * float(row['z_positive_rate']):.2f}%, inf_rows={int(row['inf_rows'])}, "
            f"z_positive_ops={json.dumps(dict(row.get('z_positive_operator_mix', {})), ensure_ascii=False, sort_keys=True)}"
        )
    lines.extend(["", "## Delta (tuned - baseline)", ""])
    for row in compare_rows:
        lines.append(
            f"- {row['case']}: best_z_delta={float(row['best_z_delta_abs']):.3f} ({float(row['best_z_delta_pct']):.2f}%), "
            f"runtime_delta={float(row['runtime_delta_abs']):.3f}s ({float(row['runtime_delta_pct']):.2f}%), "
            f"global_eval_delta={int(row['global_eval_count_delta'])}, z_global_eval_delta={int(row['z_global_eval_count_delta'])}, "
            f"z_positive_verify_delta={int(row['z_positive_mining_verify_count_delta'])}, "
            f"z_positive_row_delta={int(row['z_positive_row_count_delta'])}"
        )
    lines.extend(["", "## Next Actions", ""])
    for item in next_actions:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def run_z_positive_tuning_suite(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"z_positive_tuning_suite_{timestamp}"))
    mode_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    cases = ["SMALL", "SMALL2", "SMALL_ZRICH", "SMALL2_ZRICH"]
    for case in cases:
        case_rows: Dict[str, Dict[str, Any]] = {}
        for variant, tuned in [("baseline", False), ("tuned", True)]:
            case_root = run_small_layer_augmented_case_export(
                seed=42,
                max_iters=int(args.tra_max_iters),
                case=case,
                no_improve_limit=int(args.no_improve_limit),
                epsilon=float(args.epsilon),
                sp2_time_limit_sec=float(args.sp2_time_limit_sec),
                sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
                export_best_solution=True,
                silent=True,
                xz_evaluator_mode="neural",
                result_tag_suffix=f"zpos_{variant}",
                config_hook=lambda cfg, _case=case, _tuned=tuned: _configure_z_positive_tuning(cfg, _case, _tuned),
            )
            case_name = os.path.basename(case_root.rstrip("\\/"))
            dest_root = os.path.join(result_root, case_name)
            if os.path.abspath(case_root) != os.path.abspath(dest_root):
                if os.path.exists(dest_root):
                    shutil.rmtree(dest_root)
                shutil.move(case_root, dest_root)
            row = _z_positive_tuning_case_row(dest_root, case, variant)
            mode_rows.append(row)
            case_rows[variant] = row
        if "baseline" in case_rows and "tuned" in case_rows:
            baseline_row = case_rows["baseline"]
            tuned_row = case_rows["tuned"]
            baseline_best = max(1e-9, float(baseline_row.get("best_z", 0.0) or 0.0))
            baseline_runtime = max(1e-9, float(baseline_row.get("total_runtime_sec", 0.0) or 0.0))
            compare_rows.append({
                "case": str(case).upper(),
                "seed": 42,
                "baseline_result_root": str(baseline_row.get("result_root", "")),
                "tuned_result_root": str(tuned_row.get("result_root", "")),
                "best_z_baseline": float(baseline_row.get("best_z", 0.0) or 0.0),
                "best_z_tuned": float(tuned_row.get("best_z", 0.0) or 0.0),
                "best_z_delta_abs": float(tuned_row.get("best_z", 0.0) or 0.0) - float(baseline_row.get("best_z", 0.0) or 0.0),
                "best_z_delta_pct": 100.0 * (
                    (float(tuned_row.get("best_z", 0.0) or 0.0) - float(baseline_row.get("best_z", 0.0) or 0.0))
                    / baseline_best
                ),
                "runtime_delta_abs": float(tuned_row.get("total_runtime_sec", 0.0) or 0.0) - float(baseline_row.get("total_runtime_sec", 0.0) or 0.0),
                "runtime_delta_pct": 100.0 * (
                    (float(tuned_row.get("total_runtime_sec", 0.0) or 0.0) - float(baseline_row.get("total_runtime_sec", 0.0) or 0.0))
                    / baseline_runtime
                ),
                "global_eval_count_delta": int(tuned_row.get("global_eval_count", 0) or 0) - int(baseline_row.get("global_eval_count", 0) or 0),
                "z_global_eval_count_delta": int(tuned_row.get("z_global_eval_count", 0) or 0) - int(baseline_row.get("z_global_eval_count", 0) or 0),
                "z_positive_mining_verify_count_delta": int(tuned_row.get("z_positive_mining_verify_count", 0) or 0) - int(baseline_row.get("z_positive_mining_verify_count", 0) or 0),
                "z_positive_row_count_delta": int(tuned_row.get("z_positive_row_count", 0) or 0) - int(baseline_row.get("z_positive_row_count", 0) or 0),
                "z_all_big_negative": bool(tuned_row.get("z_all_big_negative", False)),
            })
    next_actions = _z_positive_tuning_next_actions(mode_rows, compare_rows)
    _write_json(
        os.path.join(result_root, "z_positive_tuning_summary.json"),
        {"seed": 42, "mode_rows": mode_rows, "compare_rows": compare_rows, "next_actions": next_actions},
    )
    _write_csv(
        os.path.join(result_root, "z_positive_tuning_mode_rows.csv"),
        [
            {
                **dict(row),
                "z_positive_operator_mix": json.dumps(dict(row.get("z_positive_operator_mix", {})), ensure_ascii=False, sort_keys=True),
                "z_operator_mix": json.dumps(dict(row.get("z_operator_mix", {})), ensure_ascii=False, sort_keys=True),
            }
            for row in mode_rows
        ],
    )
    _write_csv(os.path.join(result_root, "z_positive_tuning_compare_rows.csv"), compare_rows)
    with open(os.path.join(result_root, "z_positive_tuning_report.md"), "w", encoding="utf-8") as f:
        f.write(_z_positive_tuning_report_markdown(mode_rows, compare_rows, next_actions))
    return result_root


def _configure_z_full_global_eval_variant(cfg: TRARunConfig, variant: str) -> None:
    variant = str(variant).strip().lower()
    cfg.xz_evaluator_mode = "neural"
    cfg.enable_z_positive_mining_verify = False
    cfg.z_all_global_eval_default = True
    cfg.z_eval_all_candidates = False
    cfg.z_full_global_eval_experiment = False
    cfg.z_micro_safe_ops_only = False
    cfg.z_strict_safe_operator_semantics = False
    cfg.z_generation_route_guard = False
    cfg.z_repeat_reject_cache = False
    cfg.z_hotspot_require_distinct_signature = False
    cfg.z_operator_allowlist_experiment = ()
    if variant == "baseline_current":
        cfg.z_all_global_eval_default = False
        return
    if variant == "full_global_default":
        return
    if variant in {
        "micro_safe_ops",
        "micro_safe_ops_route_guarded",
        "micro_safe_ops_route_guarded_norepeat",
        "reenable_mode_flip_after_positive",
        "reenable_task_merge_after_positive",
        "reenable_hotspot_after_positive",
    }:
        cfg.z_micro_safe_ops_only = True
        cfg.z_strict_safe_operator_semantics = True
        cfg.z_hotspot_batch_size = 1
        cfg.z_local_delta_task_cap = min(int(getattr(cfg, "z_local_delta_task_cap", 2)), 2)
        cfg.z_local_delta_stack_cap = min(int(getattr(cfg, "z_local_delta_stack_cap", 1)), 1)
    if variant in {"micro_safe_ops_route_guarded", "micro_safe_ops_route_guarded_norepeat", "reenable_mode_flip_after_positive", "reenable_task_merge_after_positive", "reenable_hotspot_after_positive"}:
        cfg.z_generation_route_guard = True
    if variant in {"micro_safe_ops_route_guarded_norepeat", "reenable_mode_flip_after_positive", "reenable_task_merge_after_positive", "reenable_hotspot_after_positive"}:
        cfg.z_repeat_reject_cache = True
    if variant == "reenable_mode_flip_after_positive":
        cfg.z_operator_allowlist_experiment = ("mode_flip_sort_toggle",)
    elif variant == "reenable_task_merge_after_positive":
        cfg.z_operator_allowlist_experiment = ("task_merge_split",)
    elif variant == "reenable_hotspot_after_positive":
        cfg.z_operator_allowlist_experiment = ("z_hotspot_destroy_repair",)
        cfg.z_hotspot_require_distinct_signature = True


def _z_full_global_eval_case_row(case_root: str, case: str, variant: str) -> Dict[str, Any]:
    summary = _read_json(os.path.join(case_root, "tra_summary.json"), {}) or {}
    run_stats = dict(summary.get("run_stats", {}) or {})
    iter_rows = list(summary.get("iters", []) or [])
    z_iter_rows = [row for row in iter_rows if str(row.get("focus", "")).upper() == "Z"]
    candidate_payload = _read_json(os.path.join(case_root, "xz_supervised_candidates.json"), {}) or {}
    z_rows = list(candidate_payload.get("z_rows", []) or [])
    positive_rows = [row for row in z_rows if int(row.get("win_label", 0) or 0) > 0]
    filtered_positive_rows = [
        row for row in positive_rows
        if bool(row.get("legacy_filtered_by_proxy", False))
        or bool(row.get("legacy_filtered_by_fast_gate", False))
        or bool(row.get("legacy_filtered_by_topk", False))
    ]
    first_positive = None
    if positive_rows:
        first_positive = min(
            positive_rows,
            key=lambda row: (
                int(row.get("iter", 10 ** 9) or 10 ** 9),
                int(row.get("candidate_rank", 10 ** 9) or 10 ** 9),
                str(row.get("operator", "")),
            ),
        )
    return {
        "case": str(case).upper(),
        "seed": int((summary.get("config", {}) or {}).get("seed", 42) or 42),
        "variant": str(variant),
        "result_root": case_root,
        "best_z": float((summary.get("best", {}) or {}).get("z", 0.0) or 0.0),
        "total_runtime_sec": float(run_stats.get("run_total_time_sec", 0.0) or 0.0),
        "global_eval_count": int(run_stats.get("global_eval_count", 0) or 0),
        "accepted_or_committed_z_count": int(_layer_commit_count(iter_rows, "Z")),
        "z_iter_count": int(len(z_iter_rows)),
        "z_row_count": int(len(z_rows)),
        "positive_z_eval_count": int(len(positive_rows)),
        "positive_would_have_been_filtered_count": int(len(filtered_positive_rows)),
        "proxy_hypothesis_supported": bool(len(filtered_positive_rows) > 0),
        "z_all_candidate_count_max": int(max([int(float(row.get("z_all_candidate_count", 0.0) or 0.0)) for row in z_iter_rows] + [0])),
        "z_global_eval_full_count_max": int(max([int(float(row.get("z_global_eval_full_count", 0.0) or 0.0)) for row in z_iter_rows] + [0])),
        "z_filtered_by_proxy_count_legacy_max": int(max([int(float(row.get("z_filtered_by_proxy_count_legacy", 0.0) or 0.0)) for row in z_iter_rows] + [0])),
        "z_filtered_by_fast_gate_count_legacy_max": int(max([int(float(row.get("z_filtered_by_fast_gate_count_legacy", 0.0) or 0.0)) for row in z_iter_rows] + [0])),
        "z_filtered_by_topk_count_legacy_max": int(max([int(float(row.get("z_filtered_by_topk_count_legacy", 0.0) or 0.0)) for row in z_iter_rows] + [0])),
        "repeat_signature_block_count": int(sum(int(float(row.get("z_repeat_reject_blocked_count", 0.0) or 0.0)) for row in z_iter_rows)),
        "route_guard_reject_count": int(sum(int(float(row.get("z_route_guard_reject_count", 0.0) or 0.0)) for row in z_iter_rows)),
        "first_positive_iter": int(first_positive.get("iter", -1)) if first_positive else -1,
        "first_positive_operator": str(first_positive.get("operator", "")) if first_positive else "",
        "first_positive_actual_reduction": float(first_positive.get("actual_reduction", 0.0) or 0.0) if first_positive else 0.0,
        "first_positive_global_z_before": float(first_positive.get("global_z_before", 0.0) or 0.0) if first_positive else 0.0,
        "first_positive_global_z_after": float(first_positive.get("global_z_after", 0.0) or 0.0) if first_positive else 0.0,
        "first_positive_legacy_filtered_by_proxy": bool(first_positive.get("legacy_filtered_by_proxy", False)) if first_positive else False,
        "first_positive_legacy_filtered_by_fast_gate": bool(first_positive.get("legacy_filtered_by_fast_gate", False)) if first_positive else False,
        "first_positive_legacy_filtered_by_topk": bool(first_positive.get("legacy_filtered_by_topk", False)) if first_positive else False,
        "z_full_global_eval_experiment_seen": bool(any(bool(row.get("z_full_global_eval_experiment", False)) for row in z_iter_rows)),
    }


def _z_full_global_eval_report_markdown(
    mode_rows: List[Dict[str, Any]],
    compare_rows: List[Dict[str, Any]],
    stop_reason: str,
    first_positive_variant: str,
    first_positive_case: str,
) -> str:
    lines = [
        "# Z Iterative Optimization Suite",
        "",
        "This report tests default all-candidate global evaluation plus staged Z-operator tightening on SMALL and SMALL2.",
        "",
        f"stop_reason={stop_reason}",
        f"first_positive_variant={first_positive_variant}",
        f"first_positive_case={first_positive_case}",
        "",
        "## Per-variant rows",
        "",
    ]
    for row in mode_rows:
        lines.append(
            f"- {row['case']} | {row['variant']}: best_z={float(row['best_z']):.3f}, runtime={float(row['total_runtime_sec']):.3f}s, "
            f"z_accept={int(row['accepted_or_committed_z_count'])}, z_rows={int(row['z_row_count'])}, "
            f"positive_z={int(row['positive_z_eval_count'])}, filtered_positive={int(row['positive_would_have_been_filtered_count'])}, "
            f"all_cands_max={int(row['z_all_candidate_count_max'])}, full_eval_max={int(row['z_global_eval_full_count_max'])}, "
            f"repeat_block={int(row['repeat_signature_block_count'])}, route_guard={int(row['route_guard_reject_count'])}, "
            f"legacy_filtered(proxy/fast/topk)=({int(row['z_filtered_by_proxy_count_legacy_max'])}/{int(row['z_filtered_by_fast_gate_count_legacy_max'])}/{int(row['z_filtered_by_topk_count_legacy_max'])}), "
            f"first_positive=({int(row['first_positive_iter'])}, {row['first_positive_operator']}, {float(row['first_positive_actual_reduction']):.3f})"
        )
    lines.extend(["", "## Delta vs Baseline", ""])
    for row in compare_rows:
        lines.append(
            f"- {row['case']} | {row['variant']}: best_z_delta={float(row['best_z_delta_abs']):.3f}, "
            f"runtime_delta={float(row['runtime_delta_abs']):.3f}s, "
            f"z_accept_delta={int(row['accepted_or_committed_z_count_delta'])}, "
            f"positive_z_delta={int(row['positive_z_eval_count_delta'])}, "
            f"filtered_positive_delta={int(row['positive_would_have_been_filtered_count_delta'])}, "
            f"repeat_block_delta={int(row['repeat_signature_block_count_delta'])}, "
            f"route_guard_delta={int(row['route_guard_reject_count_delta'])}, "
            f"proxy_hypothesis_supported={bool(row['proxy_hypothesis_supported'])}"
        )
    return "\n".join(lines) + "\n"


def run_z_full_global_eval_suite(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"z_full_global_eval_suite_{timestamp}"))
    base_variants = [
        "baseline_current",
        "full_global_default",
        "micro_safe_ops",
        "micro_safe_ops_route_guarded",
        "micro_safe_ops_route_guarded_norepeat",
    ]
    reenable_variants = [
        "reenable_mode_flip_after_positive",
        "reenable_task_merge_after_positive",
        "reenable_hotspot_after_positive",
    ]
    cases = ["SMALL", "SMALL2"]
    mode_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    baseline_rows: Dict[str, Dict[str, Any]] = {}
    stop_reason = "exhausted_variants_without_positive"
    first_positive_variant = ""
    first_positive_case = ""

    def _run_variant(variant: str) -> Dict[str, Dict[str, Any]]:
        variant_rows: Dict[str, Dict[str, Any]] = {}
        for case in cases:
            case_root = run_small_layer_augmented_case_export(
                seed=int(args.base_seed),
                max_iters=int(args.tra_max_iters),
                case=case,
                no_improve_limit=int(args.no_improve_limit),
                epsilon=float(args.epsilon),
                sp2_time_limit_sec=float(args.sp2_time_limit_sec),
                sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
                export_best_solution=True,
                silent=True,
                xz_evaluator_mode="neural",
                result_tag_suffix=f"zfull_{variant}",
                config_hook=lambda cfg, _variant=variant: _configure_z_full_global_eval_variant(cfg, _variant),
            )
            case_name = os.path.basename(case_root.rstrip("\\/"))
            dest_root = os.path.join(result_root, case_name)
            if os.path.abspath(case_root) != os.path.abspath(dest_root):
                if os.path.exists(dest_root):
                    shutil.rmtree(dest_root)
                shutil.move(case_root, dest_root)
            row = _z_full_global_eval_case_row(dest_root, case, variant)
            mode_rows.append(row)
            variant_rows[case] = row
            if variant == "baseline_current":
                baseline_rows[case] = row
        if variant != "baseline_current":
            for case in cases:
                if case not in baseline_rows or case not in variant_rows:
                    continue
                baseline_row = baseline_rows[case]
                row = variant_rows[case]
                compare_rows.append({
                    "case": str(case).upper(),
                    "variant": str(variant),
                    "best_z_delta_abs": float(row.get("best_z", 0.0) or 0.0) - float(baseline_row.get("best_z", 0.0) or 0.0),
                    "runtime_delta_abs": float(row.get("total_runtime_sec", 0.0) or 0.0) - float(baseline_row.get("total_runtime_sec", 0.0) or 0.0),
                    "accepted_or_committed_z_count_delta": int(row.get("accepted_or_committed_z_count", 0) or 0) - int(baseline_row.get("accepted_or_committed_z_count", 0) or 0),
                    "positive_z_eval_count_delta": int(row.get("positive_z_eval_count", 0) or 0) - int(baseline_row.get("positive_z_eval_count", 0) or 0),
                    "positive_would_have_been_filtered_count_delta": int(row.get("positive_would_have_been_filtered_count", 0) or 0) - int(baseline_row.get("positive_would_have_been_filtered_count", 0) or 0),
                    "repeat_signature_block_count_delta": int(row.get("repeat_signature_block_count", 0) or 0) - int(baseline_row.get("repeat_signature_block_count", 0) or 0),
                    "route_guard_reject_count_delta": int(row.get("route_guard_reject_count", 0) or 0) - int(baseline_row.get("route_guard_reject_count", 0) or 0),
                    "proxy_hypothesis_supported": bool(row.get("proxy_hypothesis_supported", False)),
                })
        return variant_rows

    for variant in base_variants:
        variant_rows = _run_variant(variant)
        positive_cases = [case for case, row in variant_rows.items() if int(row.get("accepted_or_committed_z_count", 0) or 0) > 0]
        if variant != "baseline_current" and positive_cases:
            first_positive_variant = str(variant)
            first_positive_case = str(positive_cases[0]).upper()
            stop_reason = f"positive_found_at_variant={variant}"
            break

    if first_positive_variant:
        for variant in reenable_variants:
            _run_variant(variant)

    _write_json(
        os.path.join(result_root, "z_full_global_eval_summary.json"),
        {
            "mode_rows": mode_rows,
            "compare_rows": compare_rows,
            "stop_reason": stop_reason,
            "first_positive_variant": str(first_positive_variant),
            "first_positive_case": str(first_positive_case),
        },
    )
    _write_csv(os.path.join(result_root, "z_full_global_eval_mode_rows.csv"), mode_rows)
    _write_csv(os.path.join(result_root, "z_full_global_eval_compare_rows.csv"), compare_rows)
    with open(os.path.join(result_root, "z_full_global_eval_report.md"), "w", encoding="utf-8") as f:
        f.write(_z_full_global_eval_report_markdown(mode_rows, compare_rows, stop_reason, first_positive_variant, first_positive_case))
    return result_root


def run_xz_zpositive_case_export(
    seed: int = 42,
    max_iters: int = 10,
    case: str = "SMALL",
    no_improve_limit: int = 3,
    epsilon: float = 0.05,
    sp2_time_limit_sec: float = 10.0,
    sp4_lkh_time_limit_seconds: int = 5,
    export_best_solution: bool = False,
    silent: bool = True,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_tag: str = f"{case}_{seed}_xz_zpositive"
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"{result_tag}_{timestamp}"))

    cfg = _make_tra_config(
        scale=case,
        seed=int(seed),
        max_iters=int(max_iters),
        no_improve_limit=int(no_improve_limit),
        epsilon=float(epsilon),
        sp2_time_limit_sec=float(sp2_time_limit_sec),
        sp4_lkh_time_limit_seconds=int(sp4_lkh_time_limit_seconds),
        enable_role_vns=False,
        enable_shadow_chain=True,
        shadow_chain_max_depth=3,
    )
    cfg.search_scheme = "layer_augmented"
    cfg.xz_evaluator_mode = "neural"
    cfg.log_dir = result_root
    cfg.write_iteration_logs = True
    cfg.export_best_solution = bool(export_best_solution)
    cfg.enable_sp1_feedback_analysis = False
    cfg.target_runtime_sec = 85.0
    cfg.runtime_guard_mode = "soft"
    cfg.x_eval_all_candidates = True
    cfg.z_eval_all_candidates = True
    cfg.x_global_eval_topk = 999999
    cfg.z_global_eval_topk = 999999
    cfg.x_dual_eval_gap_ratio = 1.0
    cfg.z_dual_eval_gap_ratio = 1.0
    cfg.layer_operator_budget_x = 4
    cfg.layer_operator_budget_y = 0
    cfg.layer_operator_budget_z = 5
    cfg.layer_operator_budget_u = 0
    cfg.z_hotspot_batch_size = 6
    cfg.z_f0_topk = 8
    cfg.z_f1_topk = 6
    cfg.z_f2_topk = 6
    cfg.z_f1_trip_cap = 8
    cfg.z_f1_force_full_replay_threshold = 12
    cfg.z_local_delta_task_cap = 4
    cfg.z_local_delta_stack_cap = 2
    cfg.z_arrival_shift_hard_cap = 480.0
    cfg.z_wait_overflow_hard_cap = 600.0
    cfg.z_route_tail_hard_cap = 300.0
    cfg.surrogate_min_improve_abs = 0.0
    cfg.z_false_positive_streak_threshold = 999999
    cfg.z_throttle_rounds = 0

    opt = TRAOptimizer(cfg)
    opt.layer_names = ["X", "Z"]

    def _runner():
        t0 = time.perf_counter()
        best_z_val = float(opt.run())
        runtime_sec = float(time.perf_counter() - t0)
        return best_z_val, runtime_sec

    if silent:
        old_flag = os.environ.get("OFS_BATCH_SILENT")
        os.environ["OFS_BATCH_SILENT"] = "1"
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                best_z, total_runtime_sec = _runner()
        if old_flag is None:
            os.environ.pop("OFS_BATCH_SILENT", None)
        else:
            os.environ["OFS_BATCH_SILENT"] = old_flag
    else:
        best_z, total_runtime_sec = _runner()

    iter_rows = list(getattr(opt, "iter_log", []) or [])
    tra_summary = _read_json(os.path.join(result_root, "tra_summary.json"), {})
    run_stats = dict(tra_summary.get("run_stats", {}) or {})
    timing_breakdown = dict(
        run_stats.get("timing_breakdown", {})
        or (opt._timing_breakdown_payload() if hasattr(opt, "_timing_breakdown_payload") else {})
    )
    _write_layer_augmented_proxy_csv(os.path.join(result_root, "iter_xyzu_proxy_values.csv"), iter_rows)
    _write_timing_breakdown_files(result_root, case, seed, timing_breakdown)
    _write_generator_summary_files(result_root, case, seed, getattr(opt, "problem", None))

    best_iter = int(getattr(opt.best, "iter_id", -1)) if getattr(opt, "best", None) is not None else -1
    with open(os.path.join(result_root, "run_brief.txt"), "w", encoding="utf-8") as f:
        f.write(f"scale={str(case).upper()}\n")
        f.write(f"seed={int(seed)}\n")
        f.write("rotation_layers=X,Z\n")
        f.write("global_eval_mode=all_x_and_all_z_candidates\n")
        f.write("harvest_mode=xz_zpositive\n")
        f.write(f"total_runtime_sec={float(total_runtime_sec):.6f}\n")
        f.write(f"best_z={float(best_z):.3f}s @ iter={best_iter}\n")
        if timing_breakdown:
            f.write(f"forced_global_eval_time_sec={float(timing_breakdown.get('forced_global_eval_time_sec', 0.0)):.6f}\n")
            f.write(f"x_f1_time_sec={float(timing_breakdown.get('x_f1_time_sec', 0.0)):.6f}\n")
            f.write(f"z_f1_time_sec={float(timing_breakdown.get('z_f1_time_sec', 0.0)):.6f}\n")

    return result_root


def _extract_layer_augmented_baseline_z(summary: Dict[str, Any]) -> Dict[str, Any]:
    iter_rows = list(summary.get("iters", []) or [])
    best_info = dict(summary.get("best", {}) or {})

    def _finite_float(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if math.isfinite(out) else float("nan")

    for row in iter_rows:
        if str(row.get("focus", "")).upper() != "INIT":
            continue
        baseline_z = _finite_float(row.get("z", float("nan")))
        if math.isfinite(baseline_z):
            return {
                "baseline_z": float(baseline_z),
                "baseline_source": "init_z",
                "baseline_iter": int(row.get("iter", -1)),
            }

    for row in iter_rows:
        if str(row.get("focus", "")).upper() == "INIT":
            continue
        baseline_z = _finite_float(row.get("global_z_before", float("nan")))
        if math.isfinite(baseline_z):
            return {
                "baseline_z": float(baseline_z),
                "baseline_source": "first_global_z_before",
                "baseline_iter": int(row.get("iter", -1)),
            }

    if iter_rows:
        baseline_z = _finite_float(iter_rows[0].get("z", float("nan")))
        if math.isfinite(baseline_z):
            return {
                "baseline_z": float(baseline_z),
                "baseline_source": "first_row_z",
                "baseline_iter": int(iter_rows[0].get("iter", -1)),
            }

    return {
        "baseline_z": float(_finite_float(best_info.get("z", float("nan")))),
        "baseline_source": "best_fallback",
        "baseline_iter": int(best_info.get("iter_id", -1)),
    }


def run_layer_augmented_case_suite_export(
    cases: List[str],
    seed: int = 42,
    max_iters: int = 80,
    no_improve_limit: int = 20,
    epsilon: float = 0.05,
    sp2_time_limit_sec: float = 10.0,
    sp4_lkh_time_limit_seconds: int = 5,
    export_best_solution: bool = False,
    silent: bool = True,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"layer_augmented_suite_{seed}_{timestamp}"))
    summary_rows: List[Dict[str, Any]] = []

    for case in [str(c).upper() for c in cases]:
        case_root = run_small_layer_augmented_case_export(
            seed=seed,
            max_iters=max_iters,
            case=case,
            no_improve_limit=no_improve_limit,
            epsilon=epsilon,
            sp2_time_limit_sec=sp2_time_limit_sec,
            sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
            export_best_solution=export_best_solution,
            silent=silent,
        )
        summary_path = os.path.join(case_root, "tra_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary_cfg = dict(summary.get("config", {}) or {})
        iter_rows = list(summary.get("iters", []) or [])
        run_stats = dict(summary.get("run_stats", {}) or {})
        best_info = dict(summary.get("best", {}) or {})
        accept_by_layer = {layer: 0 for layer in ["X", "Y", "Z", "U"]}
        global_eval_by_layer = {layer: 0 for layer in ["X", "Y", "Z", "U"]}
        runtime_by_layer = dict(run_stats.get("layer_runtime_sec_by_name", {}) or {})
        x_eval_candidate_counts: List[float] = []
        x_surrogate_core_scores: List[float] = []
        x_surrogate_regularizer_scores: List[float] = []
        x_f1_eval_counts: List[float] = []
        x_uncertainty_probe_counts: List[float] = []
        x_candidate_hard_reject_counts: List[float] = []
        x_equivalent_dedup_counts: List[float] = []
        x_unique_candidate_counts: List[float] = []
        x_post_gate_candidate_counts: List[float] = []
        x_f1_post_y_gaps: List[float] = []
        x_anchor_template_preservation_ratios: List[float] = []
        x_spatial_dispersion_scores: List[float] = []
        x_low_consolidation_scores: List[float] = []
        x_surrogate_fp_count = 0
        x_rank_top1_total = 0
        x_rank_top1_hit = 0
        z_structural_eval_counts: List[float] = []
        z_global_eval_candidate_counts: List[float] = []
        z_f1_eval_counts: List[float] = []
        z_uncertainty_probe_counts: List[float] = []
        z_candidate_hard_reject_counts: List[float] = []
        z_operator_ban_counts: List[float] = []
        z_feasible_candidate_counts: List[float] = []
        z_f1_post_y_gaps: List[float] = []
        z_hit_tote_preservation_ratios: List[float] = []
        z_route_insertion_detours: List[float] = []
        z_hit_frequency_bonuses: List[float] = []
        z_operator_fallback_used_count = 0
        z_false_positive_reject_count = 0
        z_surrogate_fp_count = 0
        z_rank_top1_total = 0
        z_rank_top1_hit = 0
        y_load_skew_mode_count = 0
        u_slack_repair_mode_count = 0
        u_recent_y_trigger_count = 0
        z_throttled_round_count = 0
        y_budget_shifted = False
        u_budget_shifted = False
        for row in iter_rows:
            focus = str(row.get("focus", "")).upper()
            if focus in accept_by_layer and str(row.get("commit_decision", "")) == "accept":
                accept_by_layer[focus] += 1
            eval_origin = str(row.get("forced_eval_origin_layer", "") or focus).upper()
            if bool(row.get("global_eval_triggered", False)) and eval_origin in global_eval_by_layer:
                global_eval_by_layer[eval_origin] += 1
            if focus == "X":
                x_eval_candidate_counts.append(float(row.get("x_global_eval_candidate_count", 0.0) or 0.0))
                x_surrogate_core_scores.append(float(row.get("x_surrogate_core_score", 0.0) or 0.0))
                x_surrogate_regularizer_scores.append(
                    float(row.get("x_surrogate_affinity_term", 0.0) or 0.0)
                    + float(row.get("x_surrogate_finish_term", 0.0) or 0.0)
                    + float(row.get("x_surrogate_subtask_term", 0.0) or 0.0)
                    + float(row.get("x_route_conflict_penalty", 0.0) or 0.0)
                )
                x_f1_eval_counts.append(float(row.get("x_f1_eval_count", 0.0) or 0.0))
                x_uncertainty_probe_counts.append(float(row.get("x_uncertainty_probe_count", 0.0) or 0.0))
                x_candidate_hard_reject_counts.append(float(row.get("x_candidate_hard_reject_count", 0.0) or 0.0))
                x_equivalent_dedup_counts.append(float(row.get("x_equivalent_dedup_count", 0.0) or 0.0))
                x_unique_candidate_counts.append(float(row.get("x_unique_candidate_count", 0.0) or 0.0))
                x_post_gate_candidate_counts.append(float(row.get("x_post_gate_candidate_count", 0.0) or 0.0))
                x_f1_post_y_gaps.append(
                    float(row.get("x_f1_pre_y_proxy_z", 0.0) or 0.0) - float(row.get("x_f1_post_y_proxy_z", 0.0) or 0.0)
                )
                x_anchor_template_preservation_ratios.append(float(row.get("x_anchor_template_preservation_ratio", 0.0) or 0.0))
                x_spatial_dispersion_scores.append(float(row.get("x_spatial_dispersion_score", 0.0) or 0.0))
                x_low_consolidation_scores.append(float(row.get("x_low_consolidation_score", 0.0) or 0.0))
                if bool(row.get("surrogate_false_positive", False)):
                    x_surrogate_fp_count += 1
                if bool(row.get("x_rank_top1_considered", False)):
                    x_rank_top1_total += 1
                if bool(row.get("x_rank_top1_hit", False)):
                    x_rank_top1_hit += 1
            elif focus == "Z":
                z_structural_eval_counts.append(float(row.get("z_structural_eval_count", 0.0) or 0.0))
                z_global_eval_candidate_counts.append(float(row.get("z_global_eval_candidate_count", 0.0) or 0.0))
                z_f1_eval_counts.append(float(row.get("z_f1_eval_count", 0.0) or 0.0))
                z_uncertainty_probe_counts.append(float(row.get("z_uncertainty_probe_count", 0.0) or 0.0))
                z_candidate_hard_reject_counts.append(float(row.get("z_candidate_hard_reject_count", 0.0) or 0.0))
                z_operator_ban_counts.append(float(row.get("z_operator_ban_count", 0.0) or 0.0))
                z_feasible_candidate_counts.append(float(row.get("z_feasible_candidate_count", 0.0) or 0.0))
                z_f1_post_y_gaps.append(
                    float(row.get("z_f1_pre_y_proxy_z", 0.0) or 0.0) - float(row.get("z_f1_post_y_proxy_z", 0.0) or 0.0)
                )
                z_hit_tote_preservation_ratios.append(float(row.get("z_hit_tote_preservation_ratio", 0.0) or 0.0))
                z_route_insertion_detours.append(float(row.get("z_route_insertion_detour", 0.0) or 0.0))
                z_hit_frequency_bonuses.append(float(row.get("z_hit_frequency_bonus", 0.0) or 0.0))
                if bool(row.get("z_operator_fallback_used", False)):
                    z_operator_fallback_used_count += 1
                if (
                    str(row.get("global_eval_reason", "")) == "surrogate_pass"
                    and str(row.get("commit_decision", "")) == "reject_global"
                ):
                    z_false_positive_reject_count += 1
                if bool(row.get("surrogate_false_positive", False)):
                    z_surrogate_fp_count += 1
                if bool(row.get("z_rank_top1_considered", False)):
                    z_rank_top1_total += 1
                if bool(row.get("z_rank_top1_hit", False)):
                    z_rank_top1_hit += 1
                if bool(row.get("z_throttled_mode", False)):
                    z_throttled_round_count += 1
            elif focus == "Y":
                if bool(row.get("y_load_skew_mode", False)):
                    y_load_skew_mode_count += 1
                if float(row.get("operator_budget", 0.0) or 0.0) > float(summary_cfg.get("layer_operator_budget_y", 6) or 6):
                    y_budget_shifted = True
            elif focus == "U":
                if bool(row.get("u_slack_repair_mode", False)):
                    u_slack_repair_mode_count += 1
                if bool(row.get("u_budget_boosted", False)):
                    u_budget_shifted = True
                if bool(row.get("recent_y_accept_active", False)):
                    u_recent_y_trigger_count += 1
        x_dual_eval_seen = any(val >= 2.0 for val in x_eval_candidate_counts)
        z_structural_to_global_seen = any(
            float(row.get("z_structural_eval_count", 0.0) or 0.0) > 0.0
            and float(row.get("z_global_eval_candidate_count", 0.0) or 0.0) > 0.0
            for row in iter_rows
            if str(row.get("focus", "")).upper() == "Z"
        )
        runtime_total = float(run_stats.get("run_total_time_sec", float("inf")))
        runtime_vs_target = "near_target" if runtime_total <= 85.0 else "over_target"
        baseline_info = _extract_layer_augmented_baseline_z(summary)
        baseline_z = float(baseline_info.get("baseline_z", float("nan")))
        baseline_source = str(baseline_info.get("baseline_source", "best_fallback"))
        baseline_iter = int(baseline_info.get("baseline_iter", -1))
        best_z = float(best_info.get("z", float("nan")))
        best_iter = int(best_info.get("iter_id", -1))
        improve_ratio = (
            float((baseline_z - best_z) / baseline_z)
            if math.isfinite(baseline_z) and abs(baseline_z) > 1e-9 and math.isfinite(best_z)
            else 0.0
        )
        x_route_sensitive_dominant = _safe_mean(x_surrogate_core_scores) > _safe_mean(x_surrogate_regularizer_scores)
        yu_budget_shifted = bool(y_budget_shifted or u_budget_shifted)
        summary_rows.append({
            "case": case,
            "result_root": case_root,
            "baseline_z": float(baseline_z),
            "baseline_source": baseline_source,
            "baseline_iter": int(baseline_iter),
            "best_z": best_z,
            "best_iter": int(best_iter),
            "improve_vs_baseline_pct": float(improve_ratio * 100.0),
            "total_runtime_sec": float(runtime_total),
            "accepted_count": int(sum(accept_by_layer.values())),
            "global_eval_count": int(run_stats.get("global_eval_count", 0)),
            "accept_x": int(accept_by_layer["X"]),
            "accept_y": int(accept_by_layer["Y"]),
            "accept_z": int(accept_by_layer["Z"]),
            "accept_u": int(accept_by_layer["U"]),
            "eval_x": int(global_eval_by_layer["X"]),
            "eval_y": int(global_eval_by_layer["Y"]),
            "eval_z": int(global_eval_by_layer["Z"]),
            "eval_u": int(global_eval_by_layer["U"]),
            "runtime_x": float(runtime_by_layer.get("X", 0.0)),
            "runtime_y": float(runtime_by_layer.get("Y", 0.0)),
            "runtime_z": float(runtime_by_layer.get("Z", 0.0)),
            "runtime_u": float(runtime_by_layer.get("U", 0.0)),
            "x_global_eval_candidate_count_mean": _safe_mean(x_eval_candidate_counts),
            "x_surrogate_core_score_mean": _safe_mean(x_surrogate_core_scores),
            "x_f1_eval_count_mean": _safe_mean(x_f1_eval_counts),
            "x_uncertainty_probe_count": int(round(sum(x_uncertainty_probe_counts))),
            "x_candidate_hard_reject_count": int(round(sum(x_candidate_hard_reject_counts))),
            "x_equivalent_dedup_count": int(round(sum(x_equivalent_dedup_counts))),
            "x_unique_candidate_count_mean": _safe_mean(x_unique_candidate_counts),
            "x_post_gate_candidate_count_mean": _safe_mean(x_post_gate_candidate_counts),
            "x_f1_post_y_gap_mean": _safe_mean(x_f1_post_y_gaps),
            "x_anchor_template_preservation_ratio_mean": _safe_mean(x_anchor_template_preservation_ratios),
            "x_spatial_dispersion_score_mean": _safe_mean(x_spatial_dispersion_scores),
            "x_low_consolidation_score_mean": _safe_mean(x_low_consolidation_scores),
            "x_surrogate_fp_count": int(x_surrogate_fp_count),
            "x_rank_hit_rate_top1": float(x_rank_top1_hit / max(1, x_rank_top1_total)),
            "z_structural_eval_count": float(sum(z_structural_eval_counts)),
            "z_global_eval_candidate_count_mean": _safe_mean(z_global_eval_candidate_counts),
            "z_false_positive_reject_count": int(z_false_positive_reject_count),
            "z_f1_eval_count_mean": _safe_mean(z_f1_eval_counts),
            "z_uncertainty_probe_count": int(round(sum(z_uncertainty_probe_counts))),
            "z_candidate_hard_reject_count": int(round(sum(z_candidate_hard_reject_counts))),
            "z_operator_ban_count": int(round(max(z_operator_ban_counts) if z_operator_ban_counts else 0.0)),
            "z_feasible_candidate_count_mean": _safe_mean(z_feasible_candidate_counts),
            "z_f1_post_y_gap_mean": _safe_mean(z_f1_post_y_gaps),
            "z_hit_tote_preservation_ratio_mean": _safe_mean(z_hit_tote_preservation_ratios),
            "z_route_insertion_detour_mean": _safe_mean(z_route_insertion_detours),
            "z_hit_frequency_bonus_mean": _safe_mean(z_hit_frequency_bonuses),
            "z_operator_fallback_used_count": int(z_operator_fallback_used_count),
            "z_surrogate_fp_count": int(z_surrogate_fp_count),
            "z_rank_hit_rate_top1": float(z_rank_top1_hit / max(1, z_rank_top1_total)),
            "y_load_skew_mode_count": int(y_load_skew_mode_count),
            "u_slack_repair_mode_count": int(u_slack_repair_mode_count),
            "u_recent_y_trigger_count": int(u_recent_y_trigger_count),
            "z_throttled_round_count": int(z_throttled_round_count),
            "accept_count_by_layer": json.dumps(accept_by_layer, ensure_ascii=False, sort_keys=True),
            "x_route_sensitive_dominant": bool(x_route_sensitive_dominant),
            "x_dual_eval_seen": bool(x_dual_eval_seen),
            "z_throttled_seen": bool(z_throttled_round_count > 0),
            "yu_budget_shifted": bool(yu_budget_shifted),
            "z_structural_to_global_seen": bool(z_structural_to_global_seen),
            "runtime_vs_target_85s": runtime_vs_target,
            "within_100s": bool(runtime_total < 100.0),
            "meet_10pct_target": bool(improve_ratio >= 0.10 - 1e-9),
        })

    csv_path = os.path.join(suite_root, "suite_summary.csv")
    _write_csv(csv_path, summary_rows)
    _write_json(os.path.join(suite_root, "suite_summary.json"), summary_rows)
    with open(os.path.join(suite_root, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Layer-Augmented Suite Report\n\n")
        for row in summary_rows:
            warning_suffix = ""
            if str(row.get("baseline_source", "")) == "best_fallback":
                warning_suffix = (
                    f", warning=baseline_fallback, baseline_source={row['baseline_source']}, "
                    f"baseline_iter={row['baseline_iter']}"
                )
            f.write(
                f"- {row['case']}: baseline={row['baseline_z']:.3f}, best_z={row['best_z']:.3f}, improve={row['improve_vs_baseline_pct']:.2f}%, runtime={row['total_runtime_sec']:.3f}s, "
                f"accept(X/Y/Z/U)=({row['accept_x']}/{row['accept_y']}/{row['accept_z']}/{row['accept_u']}), "
                f"eval(X/Y/Z/U)=({row['eval_x']}/{row['eval_y']}/{row['eval_z']}/{row['eval_u']}), "
                f"x_route_sensitive={'Y' if row['x_route_sensitive_dominant'] else 'N'}, "
                f"x_dual_eval={'Y' if row['x_dual_eval_seen'] else 'N'}, "
                f"z_throttled={'Y' if row['z_throttled_seen'] else 'N'}, "
                f"z_structural_to_global={'Y' if row['z_structural_to_global_seen'] else 'N'}, "
                f"yu_budget_shifted={'Y' if row['yu_budget_shifted'] else 'N'}, "
                f"within_100s={'Y' if row['within_100s'] else 'N'}, "
                f"meet_10pct={'Y' if row['meet_10pct_target'] else 'N'}, "
                f"runtime_vs_85s={row['runtime_vs_target_85s']}{warning_suffix}\n"
            )
    return suite_root


def _run_xz_gpu_dataset_harvest_impl(
    cases: Optional[List[str]] = None,
    split_seeds: Optional[Dict[str, List[int]]] = None,
    replay_seeds: Optional[List[int]] = None,
    max_iters: int = 80,
    no_improve_limit: int = 20,
    epsilon: float = 0.05,
    sp2_time_limit_sec: float = 10.0,
    sp4_lkh_time_limit_seconds: int = 5,
    silent: bool = True,
    case_export_fn=run_small_layer_augmented_case_export,
    dataset_tag: str = "xz_gpu_dataset",
    harvest_mode: str = "xz_standard",
    distribution_note: str = "benchmark-like",
) -> str:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for X/Z GPU dataset parquet export") from exc
    cases = [str(case).upper() for case in (cases or GPU_DATASET_SCALES)]
    split_seeds = {str(name): [int(seed) for seed in seeds] for name, seeds in (split_seeds or GPU_DATASET_SPLIT_SEEDS).items()}
    replay_seeds = [int(seed) for seed in (replay_seeds or GPU_DATASET_REPLAY_SEEDS)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = _ensure_dir(os.path.join(ROOT_DIR, "dataset", f"{dataset_tag}_{timestamp}"))
    _ensure_dir(os.path.join(dataset_root, "splits"))
    _ensure_dir(os.path.join(dataset_root, "x"))
    _ensure_dir(os.path.join(dataset_root, "z"))

    for split_name, seeds in split_seeds.items():
        _write_json(os.path.join(dataset_root, "splits", f"{split_name}_seeds.json"), {"split": split_name, "seeds": list(seeds)})
    _write_json(os.path.join(dataset_root, "splits", "replay_seeds.json"), {"split": "replay", "seeds": list(replay_seeds)})

    raw_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "X": {split_name: [] for split_name in list(split_seeds.keys()) + ["replay"]},
        "Z": {split_name: [] for split_name in list(split_seeds.keys()) + ["replay"]},
    }
    source_runs: List[Dict[str, Any]] = []

    def _harvest_one(split_name: str, case: str, seed: int):
        case_root = case_export_fn(
            seed=int(seed),
            max_iters=int(max_iters),
            case=case,
            no_improve_limit=int(no_improve_limit),
            epsilon=float(epsilon),
            sp2_time_limit_sec=float(sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(sp4_lkh_time_limit_seconds),
            export_best_solution=False,
            silent=silent,
        )
        candidate_path = os.path.join(case_root, "xz_supervised_candidates.json")
        payload = {"x_rows": [], "z_rows": []}
        if os.path.exists(candidate_path):
            with open(candidate_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        x_rows = list(payload.get("x_rows", []) or [])
        z_rows = list(payload.get("z_rows", []) or [])
        raw_rows["X"][split_name].extend(x_rows)
        raw_rows["Z"][split_name].extend(z_rows)
        source_runs.append({
            "split": split_name,
            "case": case,
            "seed": int(seed),
            "result_root": case_root,
            "x_rows": int(len(x_rows)),
            "z_rows": int(len(z_rows)),
        })

    for split_name, seeds in split_seeds.items():
        for seed in seeds:
            for case in cases:
                _harvest_one(split_name, case, int(seed))
    for seed in replay_seeds:
        for case in cases:
            _harvest_one("replay", case, int(seed))

    scale_vocab = _dataset_scale_vocab(cases)
    fallback_vocab = _build_z_fallback_vocab(raw_rows)
    flattened_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "X": {split_name: [] for split_name in raw_rows["X"].keys()},
        "Z": {split_name: [] for split_name in raw_rows["Z"].keys()},
    }
    all_flat_rows_by_layer: Dict[str, List[Dict[str, Any]]] = {"X": [], "Z": []}
    for layer in ["X", "Z"]:
        for split_name, rows in raw_rows[layer].items():
            flat_rows = [
                _flatten_supervised_candidate_row(row, layer=layer, scale_vocab=scale_vocab, fallback_vocab=fallback_vocab)
                for row in rows
            ]
            flattened_rows[layer][split_name] = flat_rows
            all_flat_rows_by_layer[layer].extend(flat_rows)

    sample_counts: Dict[str, Dict[str, Any]] = {"X": {}, "Z": {}}
    feature_order_by_layer: Dict[str, List[str]] = {}
    files_written: List[str] = []
    source_result_roots = sorted({str(row.get("result_root", "")) for row in source_runs if str(row.get("result_root", ""))})
    for layer in ["X", "Z"]:
        column_groups = _dataset_column_groups(all_flat_rows_by_layer[layer])
        feature_order_by_layer[layer] = list(column_groups["feature"])
        schema_payload = {
            "layer": layer,
            "id_columns": list(column_groups["id"]),
            "feature_columns": list(column_groups["feature"]),
            "label_columns": list(column_groups["label"]),
            "diagnostic_columns": list(column_groups["diagnostic"]),
            "feature_order": list(column_groups["feature"]),
            "missing_value_defaults": _dataset_missing_defaults(column_groups["all"]),
            "categorical_vocab": {
                "scale_id": dict(scale_vocab),
                "z_fallback_type_code": dict(fallback_vocab) if layer == "Z" else {},
            },
        }
        _write_json(os.path.join(dataset_root, f"schema_{layer.lower()}.json"), schema_payload)
        ordered_columns = list(column_groups["all"])
        for split_name, rows in flattened_rows[layer].items():
            deduped_rows = _dedup_signature_rows(rows)
            parquet_path = os.path.join(dataset_root, layer.lower(), f"{split_name}-000.parquet")
            _write_parquet_rows(parquet_path, deduped_rows, ordered_columns)
            files_written.append(parquet_path)
            sample_counts[layer][split_name] = _layer_split_stats(rows, deduped_rows)

    dataset_report = _build_dataset_report(sample_counts)
    _write_json(os.path.join(dataset_root, "dataset_report.json"), dataset_report)

    manifest = {
        "created_at": timestamp,
        "dataset_root": dataset_root,
        "harvest_mode": str(harvest_mode),
        "distribution_note": str(distribution_note),
        "scales": list(cases),
        "split_seeds": dict(split_seeds),
        "replay_seeds": list(replay_seeds),
        "source_result_roots": source_result_roots,
        "label_rules": {
            "win_label": "actual_reduction > acceptance_min_actual_improve",
            "risk_label": "actual_reduction <= -max(50, 0.1 * global_z_before)",
        },
        "dedup_enabled": True,
        "dedup_key": ["layer", "scale", "seed", "candidate_signature"],
        "dedup_slots": ["best_win", "worst_risk", "earliest_seen"],
        "feature_order_x": list(feature_order_by_layer.get("X", [])),
        "feature_order_z": list(feature_order_by_layer.get("Z", [])),
        "categorical_vocab": {
            "scale_id": dict(scale_vocab),
            "z_fallback_type_code": dict(fallback_vocab),
        },
        "missing_value_defaults": {
            "numeric": 0.0,
            "integer": -1,
            "string": "",
            "bool": False,
        },
        "normalization_hint": {
            "ctx_avg_stack_span_norm": "ctx_avg_stack_span / warehouse_distance_scale",
            "ctx_arrival_slack_mean_norm": "ctx_arrival_slack_mean / max(anchor_z, global_makespan, 1)",
            "ctx_robot_path_length_total_norm": "ctx_robot_path_length_total / max(warehouse_distance_scale * task_count, 1)",
            "ctx_latest_robot_finish_norm": "ctx_latest_robot_finish / max(anchor_z, global_makespan, 1)",
        },
        "sample_counts": sample_counts,
        "dataset_report": dataset_report,
        "source_runs": source_runs,
        "files": files_written,
    }
    _write_json(os.path.join(dataset_root, "manifest.json"), manifest)
    _write_xz_dataset_readme(
        os.path.join(dataset_root, "README_dataset.md"),
        scales=cases,
        split_seeds=split_seeds,
        replay_seeds=replay_seeds,
        feature_order_x=feature_order_by_layer.get("X", []),
        feature_order_z=feature_order_by_layer.get("Z", []),
        harvest_mode=str(harvest_mode),
        distribution_note=str(distribution_note),
    )
    return dataset_root


def run_xz_gpu_dataset_harvest(
    cases: Optional[List[str]] = None,
    split_seeds: Optional[Dict[str, List[int]]] = None,
    replay_seeds: Optional[List[int]] = None,
    max_iters: int = 80,
    no_improve_limit: int = 20,
    epsilon: float = 0.05,
    sp2_time_limit_sec: float = 10.0,
    sp4_lkh_time_limit_seconds: int = 5,
    silent: bool = True,
) -> str:
    return _run_xz_gpu_dataset_harvest_impl(
        cases=cases,
        split_seeds=split_seeds,
        replay_seeds=replay_seeds,
        max_iters=max_iters,
        no_improve_limit=no_improve_limit,
        epsilon=epsilon,
        sp2_time_limit_sec=sp2_time_limit_sec,
        sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
        silent=silent,
        case_export_fn=run_small_layer_augmented_case_export,
        dataset_tag="xz_gpu_dataset",
        harvest_mode="xz_standard",
        distribution_note="benchmark-like",
    )


def run_xz_zpositive_dataset_harvest(
    cases: Optional[List[str]] = None,
    split_seeds: Optional[Dict[str, List[int]]] = None,
    replay_seeds: Optional[List[int]] = None,
    max_iters: int = 80,
    no_improve_limit: int = 20,
    epsilon: float = 0.05,
    sp2_time_limit_sec: float = 10.0,
    sp4_lkh_time_limit_seconds: int = 5,
    silent: bool = True,
) -> str:
    return _run_xz_gpu_dataset_harvest_impl(
        cases=cases,
        split_seeds=split_seeds,
        replay_seeds=replay_seeds,
        max_iters=max_iters,
        no_improve_limit=no_improve_limit,
        epsilon=epsilon,
        sp2_time_limit_sec=sp2_time_limit_sec,
        sp4_lkh_time_limit_seconds=sp4_lkh_time_limit_seconds,
        silent=silent,
        case_export_fn=run_xz_zpositive_case_export,
        dataset_tag="xz_zpositive_gpu_dataset",
        harvest_mode="xz_zpositive",
        distribution_note="positive-mining, not benchmark-faithful",
    )


def _run_alns_relax_case_export(scale: str, args, tra_max_iters: int = 10, vns_max_trials: int = 10,
                                result_tag: str = "alns_relax_small") -> str:
    scale = str(scale).upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"{result_tag}_{scale}_{timestamp}"))
    cfg = _make_alns_relax_config(
        scale=scale,
        seed=int(args.base_seed),
        max_iters=int(tra_max_iters),
        no_improve_limit=int(tra_max_iters),
        epsilon=float(args.epsilon),
        sp2_time_limit_sec=float(args.sp2_time_limit_sec),
        sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
        enable_sp3_precheck=bool(args.precheck_sp3),
        precheck_fail_action=str(args.precheck_fail),
        enable_soft_mu=bool(args.enable_soft_mu),
        enable_soft_pi=bool(args.enable_soft_pi),
        enable_soft_beta=bool(args.enable_soft_beta),
        enable_sku_affinity=bool(args.enable_sku_affinity),
        mu_value=float(args.mu_value),
        pi_scale=float(args.pi_scale),
        pi_clip=float(args.pi_clip),
        d0_threshold=float(args.d0_threshold),
        beta_base=float(args.beta_base),
        beta_gain=float(args.beta_gain),
        beta_min=float(args.beta_min),
        beta_max=float(args.beta_max),
        sp2_shadow_weight=float(args.sp2_shadow_weight),
        weak_accept_eta=float(args.weak_accept_eta),
        alns_init_iters=int(vns_max_trials),
    )
    cfg.export_best_solution = True
    cfg.write_iteration_logs = True
    cfg.enable_sp1_feedback_analysis = False

    t0 = time.perf_counter()
    opt = ALNSRelaxDecompOptimizer(cfg)
    best_z = float(opt.run())
    total_runtime_sec = float(time.perf_counter() - t0)

    iter_rows = list(opt.iter_log)
    best_export_dir = os.path.join(ROOT_DIR, "log", "tra_best_export")
    best_solution_objectives = {}
    best_verification = {}
    if os.path.exists(os.path.join(best_export_dir, "best_solution_objectives.json")):
        with open(os.path.join(best_export_dir, "best_solution_objectives.json"), "r", encoding="utf-8") as f:
            best_solution_objectives = json.load(f)
    if os.path.exists(os.path.join(best_export_dir, "tra_makespan_verification.json")):
        with open(os.path.join(best_export_dir, "tra_makespan_verification.json"), "r", encoding="utf-8") as f:
            best_verification = json.load(f)
    _write_csv(os.path.join(result_root, "tra_vns_small_iter_log.csv"), iter_rows)
    _write_json(os.path.join(result_root, "tra_vns_small_iter_log.json"), iter_rows)

    summary = {
        "scale": scale,
        "seed": int(args.base_seed),
        "tra_max_iters": int(tra_max_iters),
        "vns_max_trials": int(vns_max_trials),
        "best_z": float(best_z),
        "total_runtime_sec": float(total_runtime_sec),
        "iter_count": int(len(iter_rows)),
        "mode_stats": getattr(opt, "mode_stats", {}),
        "final_metrics": (iter_rows[-1] if iter_rows else {}),
        "best_solution_objectives": best_solution_objectives,
        "best_verification": best_verification,
    }
    _write_json(os.path.join(result_root, "tra_vns_small_summary.json"), summary)
    with open(os.path.join(result_root, "tra_vns_small_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"scale={scale}\nseed={int(args.base_seed)}\n")
        f.write(f"tra_max_iters={int(tra_max_iters)}\nvns_max_trials={int(vns_max_trials)}\n")
        f.write(f"best_z={float(best_z):.6f}\n")
        f.write(f"total_runtime_sec={float(total_runtime_sec):.6f}\n")
        f.write(f"iter_count={int(len(iter_rows))}\n")

    best_run_export_dir = os.path.join(result_root, "best_run_export")
    if os.path.exists(best_export_dir):
        if os.path.exists(best_run_export_dir):
            shutil.rmtree(best_run_export_dir)
        shutil.copytree(best_export_dir, best_run_export_dir)
    _write_layer_solution_audit(os.path.join(result_root, "layer_solution_audit.txt"), opt)
    print(f"[ALNS-RELAX-CASE] done. scale={scale}, result_root={result_root}")
    return result_root


def run_alns_relax_small_test(args) -> str:
    return _run_alns_relax_case_export(
        scale="SMALL",
        args=args,
        tra_max_iters=10,
        vns_max_trials=10,
        result_tag="alns_relax_small",
    )



def run_tra_vns_case_exports(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"tra_vns_cases_{timestamp}"))
    rows: List[Dict[str, Any]] = []
    for scale in [str(s).upper() for s in args.scales]:
        case_root = _run_tra_vns_case_export(
            scale=scale,
            args=args,
            tra_max_iters=int(args.tra_max_iters),
            vns_max_trials=10,
            result_tag="tra_vns_case",
        )
        summary_path = os.path.join(case_root, "tra_vns_small_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            rows.append({
                "scale": scale,
                "result_root": case_root,
                "best_z": float(summary.get("best_z", float("nan"))),
                "total_runtime_sec": float(summary.get("total_runtime_sec", 0.0)),
                "iter_count": int(summary.get("iter_count", 0)),
                "coverage_ok": bool(((summary.get("best_verification", {}) or {}).get("coverage", {}) or {}).get("coverage_ok", False)),
                "verification_status": str((summary.get("best_verification", {}) or {}).get("status", "")),
            })
    _write_csv(os.path.join(batch_root, "case_export_summary.csv"), rows)
    _write_json(os.path.join(batch_root, "case_export_summary.json"), rows)
    print(f"[TRA-VNS-CASES] done. batch_root={batch_root}")
    return batch_root


def run_shadow_chain_timing_suite(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"shadow_chain_timing_suite_{timestamp}"))
    rows: List[Dict[str, Any]] = []
    for case in ["SMALL", "SMALL2"]:
        case_root = run_small_layer_augmented_case_export(
            seed=42,
            max_iters=int(args.tra_max_iters),
            case=case,
            no_improve_limit=int(args.no_improve_limit),
            epsilon=float(args.epsilon),
            sp2_time_limit_sec=float(args.sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
            export_best_solution=True,
            silent=True,
        )
        case_name = os.path.basename(case_root.rstrip("\\/"))
        dest_root = os.path.join(result_root, case_name)
        if os.path.abspath(case_root) != os.path.abspath(dest_root):
            if os.path.exists(dest_root):
                shutil.rmtree(dest_root)
            shutil.move(case_root, dest_root)
        summary = _read_json(os.path.join(dest_root, "tra_summary.json"), {}) or {}
        timing = _read_json(os.path.join(dest_root, "timing_breakdown.json"), {}) or {}
        if not timing:
            timing = dict((summary.get("run_stats", {}) or {}).get("timing_breakdown", {}) or {})
        rows.append({
            "case": str(case).upper(),
            "seed": 42,
            "result_root": dest_root,
            "best_z": float((summary.get("best", {}) or {}).get("z", 0.0) or 0.0),
            "iter_count": int(len(summary.get("iters", []) or [])),
            "wall_time_sec": float(timing.get("wall_time_sec", 0.0) or 0.0),
            "local_vns_total_sec": float(sum((timing.get("local_vns_time_sec_by_layer", {}) or {}).values())),
            "x_f1_time_sec": float(timing.get("x_f1_time_sec", 0.0) or 0.0),
            "z_f1_time_sec": float(timing.get("z_f1_time_sec", 0.0) or 0.0),
            "global_eval_time_sec": float(timing.get("global_eval_time_sec", 0.0) or 0.0),
            "forced_global_eval_time_sec": float(timing.get("forced_global_eval_time_sec", 0.0) or 0.0),
            "global_eval_sp_total_sec": float(timing.get("global_eval_sp2_time_sec", 0.0) or 0.0)
            + float(timing.get("global_eval_sp3_time_sec", 0.0) or 0.0)
            + float(timing.get("global_eval_sp4_time_sec", 0.0) or 0.0),
            "snapshot_restore_overhead_sec": float(timing.get("snapshot_restore_overhead_sec", 0.0) or 0.0),
            "reconciliation_gap_vs_wall_sec": float(timing.get("reconciliation_gap_vs_wall_sec", 0.0) or 0.0),
        })

    _write_json(os.path.join(result_root, "suite_summary.json"), {"seed": 42, "rows": rows})
    _write_csv(os.path.join(result_root, "suite_summary.csv"), rows)
    with open(os.path.join(result_root, "timing_report.md"), "w", encoding="utf-8") as f:
        f.write(_timing_report_markdown(rows))
    return result_root


def run_zrich_smoke(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", f"zrich_smoke_{timestamp}"))
    rows: List[Dict[str, Any]] = []
    max_iters = min(int(args.tra_max_iters), 12)
    no_improve_limit = min(int(args.no_improve_limit), 4)

    for case in EXPLICIT_ZRICH_SCALES:
        base_case = _base_scale_for_zrich(case)
        base_problem = CreateOFSProblem.generate_problem_by_scale(scale=base_case, seed=42)
        zrich_problem = CreateOFSProblem.generate_problem_by_scale(scale=case, seed=42)
        case_root = run_small_layer_augmented_case_export(
            seed=42,
            max_iters=max_iters,
            case=case,
            no_improve_limit=no_improve_limit,
            epsilon=float(args.epsilon),
            sp2_time_limit_sec=float(args.sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
            export_best_solution=False,
            silent=True,
        )
        case_name = os.path.basename(case_root.rstrip("\\/"))
        dest_root = os.path.join(result_root, case_name)
        if os.path.abspath(case_root) != os.path.abspath(dest_root):
            if os.path.exists(dest_root):
                shutil.rmtree(dest_root)
            shutil.move(case_root, dest_root)
        summary = _read_json(os.path.join(dest_root, "tra_summary.json"), {}) or {}
        z_activity = _case_z_layer_activity(list(summary.get("iters", []) or []))
        redundancy_summary = dict(getattr(zrich_problem, "redundancy_summary", {}) or {})
        row = {
            "case": str(case).upper(),
            "base_case": base_case,
            "seed": 42,
            "result_root": dest_root,
            "shape_matches_base": bool(_instance_stats(zrich_problem) == _instance_stats(base_problem)),
            "demanded_sku_count": int(redundancy_summary.get("demanded_sku_count", 0) or 0),
            "min_distinct_stacks": int(redundancy_summary.get("min_distinct_stacks_per_demanded_sku", 0) or 0),
            "avg_distinct_stacks": float(redundancy_summary.get("avg_distinct_stacks_per_demanded_sku", 0.0) or 0.0),
            "max_distinct_stacks": int(redundancy_summary.get("max_distinct_stacks_per_demanded_sku", 0) or 0),
            "count_ge_4": int(redundancy_summary.get("demanded_sku_ge_target_count", 0) or 0),
            "share_ge_4": float(redundancy_summary.get("demanded_sku_ge_target_share", 0.0) or 0.0),
            **z_activity,
        }
        rows.append(row)

    _write_json(os.path.join(result_root, "zrich_smoke_summary.json"), {"seed": 42, "rows": rows})
    _write_csv(os.path.join(result_root, "zrich_smoke_summary.csv"), rows)
    return result_root


def run_experiments(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = _ensure_dir(os.path.join(ROOT_DIR, "result", timestamp))

    baseline_rows: List[Dict[str, Any]] = []
    baseline_layer_rows: List[Dict[str, Any]] = []
    tra_rows: List[Dict[str, Any]] = []
    tra_iter_rows: List[Dict[str, Any]] = []
    plot_notes: Dict[str, List[str]] = {}

    scales = [s.upper() for s in args.scales]

    for scale in scales:
        scale_dir = _ensure_dir(os.path.join(result_root, scale))
        baseline_dir = _ensure_dir(os.path.join(scale_dir, "baseline"))
        tra_dir = _ensure_dir(os.path.join(scale_dir, "tra"))

        # baseline
        b_row = run_baseline_once(
            scale=scale,
            seed=args.base_seed,
            sp4_lkh_time_limit_seconds=args.sp4_lkh_time_limit_seconds,
        )
        baseline_rows.append(b_row)

        baseline_layer_rows.append({
            "scale": scale,
            "seed": int(args.base_seed),
            "sp1_subtask_num": b_row["sp1_subtask_num"],
            "sp1_avg_unique_sku_per_subtask": b_row["sp1_avg_unique_sku_per_subtask"],
            "sp2_max_station_load": b_row["sp2_max_station_load"],
            "sp2_station_load_variance": b_row["sp2_station_load_variance"],
            "sp3_task_num": b_row["sp3_task_num"],
            "sp3_sorting_cost_total": b_row["sp3_sorting_cost_total"],
            "sp3_noise_tote_total": b_row["sp3_noise_tote_total"],
            "sp4_max_arrival_stack": b_row["sp4_max_arrival_stack"],
            "sp4_max_arrival_station": b_row["sp4_max_arrival_station"],
            "global_makespan": b_row["global_makespan"],
            "runtime_total_sec": b_row["runtime_total_sec"],
            "runtime_sp1_sec": b_row["runtime_sp1_sec"],
            "runtime_sp2_sec": b_row["runtime_sp2_sec"],
            "runtime_sp3_sec": b_row["runtime_sp3_sec"],
            "runtime_sp4_sec": b_row["runtime_sp4_sec"],
        })

        _write_json(os.path.join(baseline_dir, "baseline.json"), b_row)

        # TRA repeated runs
        best_ok_row = None
        best_ok_opt = None
        for run_id in range(args.tra_runs):
            # Keep the instance seed fixed across repeated TRA runs for fair comparison/reproducibility.
            seed = int(args.base_seed)
            cfg = _make_tra_config(
                scale=scale,
                seed=seed,
                max_iters=args.tra_max_iters,
                no_improve_limit=args.no_improve_limit,
                epsilon=args.epsilon,
                sp2_time_limit_sec=args.sp2_time_limit_sec,
                sp4_lkh_time_limit_seconds=args.sp4_lkh_time_limit_seconds,
                enable_sp3_precheck=bool(args.precheck_sp3),
                precheck_fail_action=str(args.precheck_fail),
                enable_soft_mu=bool(args.enable_soft_mu),
                enable_soft_pi=bool(args.enable_soft_pi),
                enable_soft_beta=bool(args.enable_soft_beta),
                enable_sku_affinity=bool(args.enable_sku_affinity),
                mu_value=float(args.mu_value),
                pi_scale=float(args.pi_scale),
                pi_clip=float(args.pi_clip),
                d0_threshold=float(args.d0_threshold),
                beta_base=float(args.beta_base),
                beta_gain=float(args.beta_gain),
                beta_min=float(args.beta_min),
                beta_max=float(args.beta_max),
                sp2_shadow_weight=float(args.sp2_shadow_weight),
                enable_role_vns=False,
                eps_skip=float(args.eps_skip),
                eps_light=float(args.eps_light),
                weak_accept_eta=float(args.weak_accept_eta),
                vns_max_trials=10,
                mode_fail_limit=int(args.mode_fail_limit),
            )

            t0 = time.perf_counter()
            status = "ok"
            best_z = float("nan")
            iter_log: List[Dict[str, Any]] = []
            try:
                opt = TRAOptimizer(cfg)
                best_z = float(opt.run())
                iter_log = list(opt.iter_log)
            except Exception as e:
                status = f"error:{e}"

            runtime_sec = time.perf_counter() - t0

            tra_instance_info: Dict[str, Any] = {
                "node_num": 0,
                "need_points": 0,
                "subtask_num": 0,
                "task_num": 0,
                "robot_num": 0,
                "station_num": 0,
                "order_num": 0,
                "sku_num": 0,
                "tote_num": 0,
            }
            unmet_sku_total = 0
            unmet_subtask_count = 0
            precheck_unmet_sku_total = 0
            precheck_unmet_subtask_count = 0
            precheck_status = ""
            if opt is not None and getattr(opt, "precheck_result", None):
                r = opt.precheck_result or {}
                precheck_unmet_sku_total = int(r.get("unmet_sku_total", 0))
                precheck_unmet_subtask_count = int(r.get("unmet_subtask_count", 0))
                if getattr(opt, "precheck_status", None):
                    precheck_status = str(opt.precheck_status)
                    status = precheck_status
            if status == "ok" and opt is not None and getattr(opt, "problem", None) is not None:
                tra_instance_info = _instance_stats(opt.problem)
                cov = _coverage_metrics(opt.problem)
                unmet_sku_total = int(cov["unmet_sku_total"])
                unmet_subtask_count = int(cov["unmet_subtask_count"])
                if unmet_sku_total > 0 and status == "ok":
                    status = f"partial_unmet:{unmet_sku_total}"

            tra_row = _tra_layer_row(
                scale=scale,
                run_id=run_id,
                seed=seed,
                iter_log=iter_log,
                best_z=best_z if not str(status).startswith("error:") else float("nan"),
                runtime_sec=runtime_sec,
                status=status,
                instance_info=tra_instance_info,
                unmet_sku_total=unmet_sku_total,
                unmet_subtask_count=unmet_subtask_count,
                precheck_unmet_sku_total=precheck_unmet_sku_total,
                precheck_unmet_subtask_count=precheck_unmet_subtask_count,
                precheck_status=precheck_status,
            )
            tra_rows.append(tra_row)

            if status == "ok":
                if best_ok_row is None or float(tra_row["best_z"]) < float(best_ok_row["best_z"]):
                    best_ok_row = dict(tra_row)
                    best_ok_opt = opt

            for item in iter_log:
                tra_iter_rows.append({
                    "scale": scale,
                    "run_id": int(run_id),
                    "seed": int(seed),
                    "iter": int(item.get("iter", 0)),
                    "focus": item.get("focus", ""),
                    "z": item.get("z", float("nan")),
                    "best_z": item.get("best_z", float("nan")),
                    "lb": item.get("lb", float("nan")),
                    "improved": bool(item.get("improved", False)),
                    "skipped": bool(item.get("skipped", False)),
                    "epsilon": item.get("epsilon", float("nan")),
                })

        # per-scale writes
        this_tra = [r for r in tra_rows if r.get("scale") == scale]
        this_iter = [r for r in tra_iter_rows if r.get("scale") == scale]
        _write_csv(os.path.join(tra_dir, "tra_runs.csv"), this_tra)
        _write_csv(os.path.join(tra_dir, "tra_iter_curve.csv"), this_iter)

        if best_ok_row is not None and best_ok_opt is not None:
            best_export_dir = _ensure_dir(os.path.join(tra_dir, "best_run_export"))
            best_ok_opt.export_best_to(best_export_dir)
            _write_json(os.path.join(best_export_dir, "best_run_row.json"), best_ok_row)
            if not args.skip_plots:
                notes = _write_best_iter_plot(tra_dir, scale, this_iter, int(best_ok_row["run_id"]))
                if notes:
                    plot_notes.setdefault(scale, []).extend(notes)

        if not args.skip_plots:
            plot_notes.setdefault(scale, [])
            plot_notes[scale].extend(_try_plot(tra_dir, scale, this_tra, this_iter))

    summary_rows = _build_summary_rows(baseline_rows, tra_rows)

    _write_csv(os.path.join(result_root, "summary.csv"), summary_rows)
    _write_csv(os.path.join(result_root, "baseline_layers.csv"), baseline_layer_rows)
    _write_csv(os.path.join(result_root, "baseline_runs.csv"), baseline_rows)
    _write_csv(os.path.join(result_root, "tra_runs.csv"), tra_rows)
    _write_csv(os.path.join(result_root, "tra_iter_curve.csv"), tra_iter_rows)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "scales": scales,
        "tra_runs": int(args.tra_runs),
        "tra_config_template": asdict(_make_tra_config(
            scale=scales[0] if scales else "SMALL",
            seed=args.base_seed,
            max_iters=args.tra_max_iters,
            no_improve_limit=args.no_improve_limit,
            epsilon=args.epsilon,
            sp2_time_limit_sec=args.sp2_time_limit_sec,
            sp4_lkh_time_limit_seconds=args.sp4_lkh_time_limit_seconds,
        )),
        "plot_notes": plot_notes,
    }
    _write_json(os.path.join(result_root, "meta.json"), meta)
    _write_report(os.path.join(result_root, "report.md"), summary_rows, scales, args.tra_runs)

    print(f"[Benchmark] done. result_root={result_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline + TRA repeated benchmark")
    parser.add_argument("--scales", nargs="+", default=ALL_SCALES, help="Scale list")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--tra-runs", type=int, default=1000, help="TRA repeated run count per scale")
    parser.add_argument("--tra-max-iters", type=int, default=50, help="TRA max_iters")
    parser.add_argument("--no-improve-limit", type=int, default=3, help="TRA no_improve_limit")
    parser.add_argument("--epsilon", type=float, default=0.05, help="TRA epsilon")
    parser.add_argument("--sp2-time-limit-sec", type=float, default=10.0, help="SP2 time limit in TRA")
    parser.add_argument("--sp4-lkh-time-limit-seconds", type=int, default=5, help="SP4 LKH time limit in TRA")
    parser.add_argument("--skip-plots", action="store_true", help="Disable plotting PNG")
    # precheck & soft-coupling flags
    parser.add_argument("--precheck-sp3", action="store_true", help="Enable SP3 coverage precheck before rotation")
    parser.add_argument("--precheck-fail", type=str, default="log", choices=["log","abort"], help="Precheck fail action")
    parser.add_argument("--enable-soft-mu", action="store_true", help="Enable soft time-window (mu)")
    parser.add_argument("--enable-soft-pi", action="store_true", help="Enable shadow assignment (pi_os)")
    parser.add_argument("--enable-soft-beta", action="store_true", help="Enable dynamic beta for SP3")
    parser.add_argument("--enable-sku-affinity", action="store_true", help="Enable SKU incompatibility feedback")
    parser.add_argument("--mu-value", type=float, default=1.0, help="mu penalty per second")
    parser.add_argument("--pi-scale", type=float, default=1.0, help="pi_os scale")
    parser.add_argument("--pi-clip", type=float, default=120.0, help="pi_os clip upper bound")
    parser.add_argument("--d0-threshold", type=float, default=20.0, help="distance threshold for pi_os")
    parser.add_argument("--beta-base", type=float, default=1.0, help="beta base")
    parser.add_argument("--beta-gain", type=float, default=1.0, help="beta gain by congestion")
    parser.add_argument("--beta-min", type=float, default=0.5, help="beta min")
    parser.add_argument("--beta-max", type=float, default=3.0, help="beta max")
    parser.add_argument("--sp2-shadow-weight", type=float, default=1.0, help="weight for shadow penalty in SP2 objective")
    parser.add_argument("--eps-skip", type=float, default=0.05, help="Skip threshold for role-VNS TRA")
    parser.add_argument("--eps-light", type=float, default=0.15, help="Light/Full threshold for role-VNS TRA")
    parser.add_argument("--weak-accept-eta", type=float, default=0.02, help="Weak accept tolerance for role-VNS TRA")
    parser.add_argument("--mode-fail-limit", type=int, default=3, help="Mode fail limit for role-VNS TRA")
    parser.add_argument("--run-tra-vns-small-test", action="store_true",
                        help="Run fixed SMALL test with TRA outer 10 iters and VNS inner 10 trials")
    parser.add_argument("--run-alns-relax-small-test", action="store_true",
                        help="Run fixed SMALL test with ALNS init + four-layer decomposition")
    parser.add_argument("--run-two-layer-tra-small-test", action="store_true",
                        help="Run fixed SMALL test with layered revolving coupled coordination")
    parser.add_argument("--run-small-layer-augmented-case", action="store_true",
                        help="Run one SMALL layer_augmented case export with proxy CSV and brief summary")
    parser.add_argument("--run-classic-vs-neural-suite", action="store_true",
                        help="Run SMALL/SMALL2 classic_soft vs neural X/Z evaluator comparison suite")
    parser.add_argument("--run-shadow-chain-timing-suite", action="store_true",
                        help="Run seed-42 shadow-chain timing bundle on SMALL and SMALL2")
    parser.add_argument("--run-zrich-smoke", action="store_true",
                        help="Run short seed-42 smoke on SMALL_ZRICH and SMALL2_ZRICH")
    parser.add_argument("--run-z-positive-tuning-suite", action="store_true",
                        help="Run seed-42 baseline vs tuned Z positive-mining suite on SMALL/SMALL2 and ZRICH cases")
    parser.add_argument("--run-z-full-global-eval-suite", action="store_true",
                        help="Run SMALL/SMALL2 baseline + Z full-global-eval repair ladder until a Z positive appears")
    parser.add_argument("--run-xz-zpositive-dataset-harvest", action="store_true",
                        help="Run X/Z-only positive-mining harvest and export supervised parquet dataset package")
    parser.add_argument("--xz-dataset-scales", nargs="+", default=GPU_DATASET_SCALES,
                        help="Scale list for X/Z GPU dataset harvest")
    parser.add_argument("--run-tra-vns-case-exports", action="store_true",
                        help="Run TRA-VNS case export for each scale in --scales, one result folder per scale")
    parser.add_argument("--run-soft-coupling-table", action="store_true",
                        help="Run 5-scale ablation table: baseline + 6 TRA variants")
    parser.add_argument("--table-iter-limit", type=int, default=1000,
                        help="TRA max_iters/no_improve_limit for soft-coupling table")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_tra_vns_small_test:
        run_tra_vns_small_test(args)
    elif args.run_alns_relax_small_test:
        run_alns_relax_small_test(args)
    elif args.run_small_layer_augmented_case:
        run_small_layer_augmented_case_export(
            seed=int(args.base_seed),
            max_iters=int(args.tra_max_iters),
            no_improve_limit=int(args.no_improve_limit),
            epsilon=float(args.epsilon),
            sp2_time_limit_sec=float(args.sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
            export_best_solution=False,
            silent=True,
        )
    elif args.run_classic_vs_neural_suite:
        run_classic_vs_neural_suite(args)
    elif args.run_shadow_chain_timing_suite:
        run_shadow_chain_timing_suite(args)
    elif args.run_zrich_smoke:
        run_zrich_smoke(args)
    elif args.run_z_positive_tuning_suite:
        run_z_positive_tuning_suite(args)
    elif args.run_z_full_global_eval_suite:
        run_z_full_global_eval_suite(args)
    elif args.run_xz_zpositive_dataset_harvest:
        run_xz_zpositive_dataset_harvest(
            cases=[str(scale).upper() for scale in (args.xz_dataset_scales or GPU_DATASET_SCALES)],
            max_iters=int(args.tra_max_iters),
            no_improve_limit=int(args.no_improve_limit),
            epsilon=float(args.epsilon),
            sp2_time_limit_sec=float(args.sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
            silent=True,
        )
    elif args.run_tra_vns_case_exports:
        run_tra_vns_case_exports(args)
    elif args.run_soft_coupling_table:
        run_soft_coupling_ablation_table(
            scales=args.scales,
            iter_limit=int(args.table_iter_limit),
            seed=int(args.base_seed),
            sp2_time_limit_sec=float(args.sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
            enable_sp3_precheck=bool(args.precheck_sp3),
            precheck_fail_action=str(args.precheck_fail),
        )
    else:
        run_experiments(args)
