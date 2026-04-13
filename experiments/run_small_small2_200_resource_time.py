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


DEFAULT_ALL_CASES = [
    "Gurobi-s1"
    #  "SMALL",
    # "SMALL2",
    # "SMALL_ZRICH",
    # "SMALL2_ZRICH",
    # "SMALL3",
    # "SMALL_UNEVEN",
    # "SMALL2_UNEVEN",
]


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
    return TRARunConfig(
        scale=str(scale).upper(),
        seed=int(seed),
        max_iters=int(args.max_iters),
        no_improve_limit=int(args.no_improve_limit),
        epsilon=float(args.epsilon),
        sp2_time_limit_sec=float(args.sp2_time_limit_sec),
        sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
        export_best_solution=bool(args.export_best_solution),
        write_iteration_logs=bool(args.write_iteration_logs),
        enable_sp1_feedback_analysis=False,
        log_dir=run_log_dir,
        xz_evaluator_mode="classic_soft",
        search_scheme="resource_time_alns",
        resource_real_eval_period=int(args.resource_real_eval_period),
    )


def _run_one(args, scale: str, run_idx: int, seed: int, batch_root: str) -> Dict[str, Any]:
    case_root = _ensure_dir(os.path.join(batch_root, str(scale).upper()))
    run_root = os.path.join(case_root, f"run_{run_idx:03d}_seed_{seed}")
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
    layer_selected = {name: 0 for name in ["X", "Y", "Z"]}
    layer_accepted = {name: 0 for name in ["X", "Y", "Z"]}
    for iter_row in iter_rows:
        layer_name = str(iter_row.get("selected_resource_layer", iter_row.get("focus", "")) or "").upper()
        if layer_name in layer_selected:
            layer_selected[layer_name] = int(layer_selected[layer_name]) + 1
            if bool(iter_row.get("local_accept", False)):
                layer_accepted[layer_name] = int(layer_accepted[layer_name]) + 1
    initial_makespan = _safe_float(init_metrics.get("initial_makespan", float("nan")))
    best_z_value = _safe_float(best_z if not math.isnan(best_z) else best_row.get("z", float("nan")))
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
        "best_iter": int(best_row.get("iter_id", -1) or -1),
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
        "layer_selected_x": int(layer_selected["X"]),
        "layer_selected_y": int(layer_selected["Y"]),
        "layer_selected_z": int(layer_selected["Z"]),
        "layer_accepted_x": int(layer_accepted["X"]),
        "layer_accepted_y": int(layer_accepted["Y"]),
        "layer_accepted_z": int(layer_accepted["Z"]),
        "coverage_ok": bool(audit.get("coverage_ok", False)),
        "unmet_sku_total": int(audit.get("unmet_sku_total", 0) or 0),
        "makespan_consistent": bool(audit.get("makespan_consistent", False)),
        "has_unreasonable_solution": bool(audit.get("has_unreasonable_solution", False)),
        "result_root": result_root,
        **init_metrics,
    }


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
        coverage_hard_reject_values = [int(row.get("coverage_hard_reject_count", 0) or 0) for row in ok_rows]
        exact_cache_hit_values = [int(row.get("exact_eval_cache_hit_count", 0) or 0) for row in ok_rows]
        x_decap_values = [int(row.get("x_failure_decapitation_count", 0) or 0) for row in ok_rows]
        stop_reason_values = [row.get("stop_reason") for row in ok_rows if row.get("stop_reason") not in (None, "")]
        layer_selected_x_values = [int(row.get("layer_selected_x", 0) or 0) for row in ok_rows]
        layer_selected_y_values = [int(row.get("layer_selected_y", 0) or 0) for row in ok_rows]
        layer_selected_z_values = [int(row.get("layer_selected_z", 0) or 0) for row in ok_rows]
        layer_accepted_x_values = [int(row.get("layer_accepted_x", 0) or 0) for row in ok_rows]
        layer_accepted_y_values = [int(row.get("layer_accepted_y", 0) or 0) for row in ok_rows]
        layer_accepted_z_values = [int(row.get("layer_accepted_z", 0) or 0) for row in ok_rows]
        out.append({
            "scale": scale,
            "run_count": int(len(scale_rows)),
            "ok_count": int(len(ok_rows)),
            "error_count": int(len(scale_rows) - len(ok_rows)),
            "best_of_best_z": min(best_values) if best_values else float("nan"),
            "mean_best_z": (sum(best_values) / len(best_values)) if best_values else float("nan"),
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
            "coverage_hard_reject_count": _collapse_values(coverage_hard_reject_values),
            "exact_eval_cache_hit_count": _collapse_values(exact_cache_hit_values),
            "x_failure_decapitation_count": _collapse_values(x_decap_values),
            "stop_reason": _collapse_values(stop_reason_values),
            "layer_selected_x": _collapse_values(layer_selected_x_values),
            "layer_selected_y": _collapse_values(layer_selected_y_values),
            "layer_selected_z": _collapse_values(layer_selected_z_values),
            "layer_accepted_x": _collapse_values(layer_accepted_x_values),
            "layer_accepted_y": _collapse_values(layer_accepted_y_values),
            "layer_accepted_z": _collapse_values(layer_accepted_z_values),
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
    parser.add_argument("--max-iters", type=int, default=200, help="ALNS max_iters")
    parser.add_argument("--no-improve-limit", type=int, default=3, help="TRA no_improve_limit")
    parser.add_argument("--epsilon", type=float, default=0.05, help="TRA epsilon")
    parser.add_argument("--sp2-time-limit-sec", type=float, default=10.0, help="SP2 time limit")
    parser.add_argument("--sp4-lkh-time-limit-seconds", type=int, default=5, help="SP4 LKH time limit")
    parser.add_argument("--resource-real-eval-period", type=int, default=8, help="Validator period")
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
    _write_csv(os.path.join(batch_root, "batch_runs.csv"), all_rows)
    _write_csv(os.path.join(batch_root, "batch_summary.csv"), summary_rows)
    _write_json(
        os.path.join(batch_root, "batch_meta.json"),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cases": [str(case).upper() for case in (args.cases or [])],
            "runs_per_case": int(args.runs),
            "seed_base": int(args.seed_base),
            "same_seed": bool(args.same_seed),
            "alns_max_iters": int(args.max_iters),
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
    print(f"[DONE] batch_root={batch_root}")


if __name__ == "__main__":
    main()
