import argparse
import contextlib
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
from Gurobi.sp4 import SP4_Robot_Router
from Gurobi.tra import TRAOptimizer, TRARunConfig
from entity.calculate import GlobalTimeCalculator
from config.ofs_config import OFSConfig


ALL_SCALES = ["SMALL", "SMALL2", "SMALL3", "MEDIUM", "LARGE"]


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
                     sp2_shadow_weight: float = 1.0, enable_role_vns: bool = True,
                     eps_skip: float = 0.05, eps_light: float = 0.15,
                     weak_accept_eta: float = 0.02, vns_max_trials: int = 10,
                     mode_fail_limit: int = 3) -> TRARunConfig:
    return TRARunConfig(
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
        enable_role_vns=enable_role_vns,
        eps_skip=eps_skip,
        eps_light=eps_light,
        weak_accept_eta=weak_accept_eta,
        vns_max_trials=vns_max_trials,
        mode_fail_limit=mode_fail_limit,
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
        enable_role_vns=True,
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
                enable_role_vns=True,
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
