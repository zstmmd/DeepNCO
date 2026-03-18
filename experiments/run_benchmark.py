import argparse
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
                     sp2_time_limit_sec: float, sp4_lkh_time_limit_seconds: int) -> TRARunConfig:
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
    )


def _tra_layer_row(scale: str, run_id: int, seed: int, iter_log: List[Dict[str, Any]], best_z: float,
                   runtime_sec: float, status: str, instance_info: Dict[str, Any],
                   unmet_sku_total: int, unmet_subtask_count: int) -> Dict[str, Any]:
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
        notes.append(f"plot skipped: {e}")
        return notes

    this_scale = sorted([r for r in tra_rows if r.get("scale") == scale and r.get("status") == "ok"],
                        key=lambda x: int(x.get("run_id", 0)))
    if not this_scale:
        notes.append("plot skipped: no successful TRA rows")
        return notes

    run_ids = [int(r["run_id"]) for r in this_scale]
    best_z = [float(r["best_z"]) for r in this_scale]

    plt.figure(figsize=(10, 4))
    plt.plot(run_ids, best_z, linewidth=1.0)
    plt.title(f"TRA best_z per run - {scale}")
    plt.xlabel("run_id")
    plt.ylabel("best_z")
    plt.tight_layout()
    plt.savefig(os.path.join(scale_dir, "tra_bestz_vs_run.png"), dpi=140)
    plt.close()

    window = max(5, min(50, len(best_z) // 10 if len(best_z) >= 10 else 5))
    rolling = []
    for i in range(len(best_z)):
        left = max(0, i - window + 1)
        rolling.append(_safe_mean(best_z[left:i + 1]))

    plt.figure(figsize=(10, 4))
    plt.plot(run_ids, rolling, linewidth=1.2)
    plt.title(f"TRA rolling mean best_z (window={window}) - {scale}")
    plt.xlabel("run_id")
    plt.ylabel("rolling_mean_best_z")
    plt.tight_layout()
    plt.savefig(os.path.join(scale_dir, "tra_bestz_rolling_mean.png"), dpi=140)
    plt.close()

    # Iter profile sample: run 0, median, best
    candidate_run_ids = []
    candidate_run_ids.append(run_ids[0])
    candidate_run_ids.append(run_ids[len(run_ids) // 2])
    best_row = min(this_scale, key=lambda x: float(x.get("best_z", float("inf"))))
    candidate_run_ids.append(int(best_row["run_id"]))
    candidate_run_ids = sorted(set(candidate_run_ids))

    plt.figure(figsize=(10, 4))
    for rid in candidate_run_ids:
        seq = [r for r in iter_rows if r.get("scale") == scale and int(r.get("run_id", -1)) == rid]
        seq = sorted(seq, key=lambda x: int(x.get("iter", 0)))
        xs = [int(r["iter"]) for r in seq]
        ys = [float(r["best_z"]) for r in seq]
        if xs and ys:
            plt.plot(xs, ys, linewidth=1.2, label=f"run={rid}")

    plt.title(f"TRA iteration profile sample - {scale}")
    plt.xlabel("iter")
    plt.ylabel("best_z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(scale_dir, "tra_iter_profile_sample.png"), dpi=140)
    plt.close()

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
        for run_id in range(args.tra_runs):
            seed = int(args.base_seed + run_id)
            cfg = _make_tra_config(
                scale=scale,
                seed=seed,
                max_iters=args.tra_max_iters,
                no_improve_limit=args.no_improve_limit,
                epsilon=args.epsilon,
                sp2_time_limit_sec=args.sp2_time_limit_sec,
                sp4_lkh_time_limit_seconds=args.sp4_lkh_time_limit_seconds,
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
            )
            tra_rows.append(tra_row)

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

        if not args.skip_plots:
            plot_notes[scale] = _try_plot(tra_dir, scale, this_tra, this_iter)

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
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
