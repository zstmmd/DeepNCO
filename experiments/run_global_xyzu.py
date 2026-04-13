import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from config.ofs_config import OFSConfig
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _task_rows(problem: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in getattr(problem, "task_list", []) or []:
        rows.append(
            {
                "task_id": int(getattr(task, "task_id", -1)),
                "subtask_id": int(getattr(task, "sub_task_id", -1)),
                "assigned_robot_id": int(getattr(task, "robot_id", -1)),
                "target_stack_id": int(getattr(task, "target_stack_id", -1)),
                "target_station_id": int(getattr(task, "target_station_id", -1)),
                "operation_mode": str(getattr(task, "operation_mode", "")),
                "target_tote_ids": list(getattr(task, "target_tote_ids", []) or []),
                "hit_tote_ids": list(getattr(task, "hit_tote_ids", []) or []),
                "noise_tote_ids": list(getattr(task, "noise_tote_ids", []) or []),
                "sort_layer_range": list(getattr(task, "sort_layer_range", []) or []),
                "arrival_time_at_stack": float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0),
                "arrival_time_at_station": float(getattr(task, "arrival_time_at_station", 0.0) or 0.0),
                "start_process_time": float(getattr(task, "start_process_time", 0.0) or 0.0),
                "end_process_time": float(getattr(task, "end_process_time", 0.0) or 0.0),
            }
        )
    return rows


def _subtask_rows(problem: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for st in getattr(problem, "subtask_list", []) or []:
        rows.append(
            {
                "subtask_id": int(getattr(st, "id", -1)),
                "order_id": int(getattr(getattr(st, "parent_order", None), "order_id", -1)),
                "sku_list": list(getattr(st, "sku_list", []) or []),
                "assigned_station_id": int(getattr(st, "assigned_station_id", -1)),
                "station_sequence_rank": int(getattr(st, "station_sequence_rank", -1)),
                "assigned_robot_id": int(getattr(st, "assigned_robot_id", -1)),
                "execution_task_ids": [
                    int(getattr(task, "task_id", -1))
                    for task in (getattr(st, "execution_tasks", []) or [])
                ],
            }
        )
    return rows


def _compute_solution_coverage(problem: Any) -> Dict[str, Any]:
    required: Dict[int, int] = defaultdict(int)
    covered: Dict[int, int] = defaultdict(int)
    coverage_subtasks: List[Dict[str, Any]] = []
    for order in getattr(problem, "order_list", []) or []:
        for sku_id in getattr(order, "order_product_id_list", []) or []:
            required[int(sku_id)] += 1
    for st in getattr(problem, "subtask_list", []) or []:
        for task in getattr(st, "execution_tasks", []) or []:
            for tote_id in (getattr(task, "hit_tote_ids", []) or []):
                tote = getattr(problem, "id_to_tote", {}).get(int(tote_id))
                for sku_id, qty in (getattr(tote, "sku_quantity_map", {}) or {}).items():
                    covered[int(sku_id)] += int(qty)
    unmet_total = 0
    for sku_id, demand in required.items():
        unmet_total += max(0, int(demand) - int(covered.get(int(sku_id), 0)))
    for st in getattr(problem, "subtask_list", []) or []:
        required_sub: Dict[int, int] = defaultdict(int)
        covered_sub: Dict[int, int] = defaultdict(int)
        for sku in getattr(st, "sku_list", []) or []:
            required_sub[int(getattr(sku, "id", -1))] += 1
        for task in getattr(st, "execution_tasks", []) or []:
            for tote_id in (getattr(task, "hit_tote_ids", []) or []):
                tote = getattr(problem, "id_to_tote", {}).get(int(tote_id))
                for sku_id, qty in (getattr(tote, "sku_quantity_map", {}) or {}).items():
                    covered_sub[int(sku_id)] += int(qty)
        unmet_sub_map: Dict[int, int] = {}
        unmet_sub_total = 0
        for sku_id, demand in required_sub.items():
            unmet = max(0, int(demand) - int(covered_sub.get(int(sku_id), 0)))
            if unmet > 0:
                unmet_sub_map[int(sku_id)] = int(unmet)
            unmet_sub_total += int(unmet)
        coverage_subtasks.append(
            {
                "subtask_id": int(getattr(st, "id", -1)),
                "order_id": int(getattr(getattr(st, "parent_order", None), "order_id", -1)),
                "required_sku_units": int(sum(required_sub.values())),
                "provided_sku_units": int(sum(min(required_sub.get(k, 0), covered_sub.get(k, 0)) for k in required_sub)),
                "unmet_sku_units": int(unmet_sub_total),
                "unmet_skus": {str(k): int(v) for k, v in unmet_sub_map.items()},
                "coverage_ok": bool(unmet_sub_total == 0),
            }
        )
    return {
        "coverage_ok": bool(unmet_total == 0),
        "unmet_sku_total": int(unmet_total),
        "unmet_subtask_count": int(sum(1 for row in coverage_subtasks if not bool(row.get("coverage_ok", False)))),
        "subtasks": coverage_subtasks,
    }


def _normalized_robot_service_time(problem: Any, task: Any) -> float:
    current = float(getattr(task, "robot_service_time", 0.0) or 0.0)
    if current > 0.0:
        return current
    mode = str(getattr(task, "operation_mode", "") or "").upper()
    hit_tote_ids = [int(x) for x in (getattr(task, "hit_tote_ids", []) or []) if int(x) >= 0]
    if mode == "FLIP":
        total = 0.0
        for tote_id in hit_tote_ids:
            tote = getattr(problem, "id_to_tote", {}).get(int(tote_id))
            stack_id = int(getattr(task, "target_stack_id", -1))
            stack = getattr(problem, "point_to_stack", {}).get(int(stack_id))
            totes = list(getattr(stack, "totes", []) or []) if stack is not None else []
            top_index = max(0, len(totes) - 1)
            tote_index = next((idx for idx, row in enumerate(totes) if int(getattr(row, "id", -1)) == int(tote_id)), top_index)
            total += float(getattr(OFSConfig, "PACKING_TIME", 0.0))
            if tote_index < top_index:
                total += float(getattr(OFSConfig, "LIFTING_TIME", 0.0))
        return float(total)
    if mode == "SORT":
        layer_range = getattr(task, "sort_layer_range", None) or []
        stack_id = int(getattr(task, "target_stack_id", -1))
        stack = getattr(problem, "point_to_stack", {}).get(int(stack_id))
        totes = list(getattr(stack, "totes", []) or []) if stack is not None else []
        if len(layer_range) == 2 and totes:
            high = int(layer_range[1])
            top_included = bool(high >= len(totes) - 1)
            return float(getattr(OFSConfig, "PACKING_TIME", 0.0) + (0.0 if top_included else float(getattr(OFSConfig, "LIFTING_TIME", 0.0))))
        return float(getattr(OFSConfig, "PACKING_TIME", 0.0))
    return current


def _verify_makespan_breakdown(problem: Any, out_dir: str) -> Dict[str, Any]:
    all_tasks: List[Any] = []
    for st in getattr(problem, "subtask_list", []) or []:
        all_tasks.extend(getattr(st, "execution_tasks", []) or [])
    failures: List[str] = []
    station_task_rows: List[Dict[str, Any]] = []
    for station in getattr(problem, "station_list", []) or []:
        seq = sorted(getattr(station, "processed_tasks", []) or [], key=lambda t: float(getattr(t, "start_process_time", 0.0)))
        prev_end = 0.0
        for t in seq:
            extra = float(getattr(t, "extra_service_used", 0.0) or 0.0)
            expected_end = float(getattr(t, "start_process_time", 0.0) or 0.0) + float(getattr(t, "picking_duration", 0.0) or 0.0) + extra
            actual_end = float(getattr(t, "end_process_time", 0.0) or 0.0)
            if abs(expected_end - actual_end) > 1e-6:
                failures.append(f"Task {int(getattr(t, 'task_id', -1))}: end mismatch expected={expected_end:.6f}, actual={actual_end:.6f}")
            if float(getattr(t, "start_process_time", 0.0) or 0.0) + 1e-6 < prev_end:
                failures.append(f"Station {int(getattr(station, 'id', -1))}: FCFS violation at task {int(getattr(t, 'task_id', -1))}")
            prev_end = actual_end
            station_task_rows.append(
                {
                    "station_id": int(getattr(station, "id", -1)),
                    "task_id": int(getattr(t, "task_id", -1)),
                    "start": float(getattr(t, "start_process_time", 0.0) or 0.0),
                    "end": actual_end,
                    "wait": float(getattr(t, "tote_wait_time", 0.0) or 0.0),
                    "pick": float(getattr(t, "picking_duration", 0.0) or 0.0),
                    "extra": extra,
                }
            )
    max_end = max((float(getattr(t, "end_process_time", 0.0) or 0.0) for t in all_tasks), default=0.0)
    global_makespan = float(getattr(problem, "global_makespan", 0.0) or 0.0)
    if abs(max_end - global_makespan) > 1e-6:
        failures.append(f"Global makespan mismatch: max_task_end={max_end:.6f}, global_makespan={global_makespan:.6f}")
    coverage = _compute_solution_coverage(problem)
    if int(coverage.get("unmet_sku_total", 0)) > 0:
        failures.append(f"SKU coverage unmet: unmet_sku_total={int(coverage.get('unmet_sku_total', 0))}, unmet_subtask_count={int(coverage.get('unmet_subtask_count', 0))}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "task_count": int(len(all_tasks)),
        "max_task_end": float(max_end),
        "global_makespan": float(global_makespan),
        "coverage": coverage,
        "failures": failures,
        "station_task_rows": station_task_rows,
    }
    with open(os.path.join(out_dir, "tra_makespan_verification.json"), "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(result), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "tra_makespan_verification.txt"), "w", encoding="utf-8") as f:
        f.write("[TRA Makespan Verification]\n")
        f.write(f"status={result['status']}\n")
        f.write(f"task_count={result['task_count']}\n")
        f.write(f"max_task_end={float(max_end):.6f}\n")
        f.write(f"global_makespan={float(global_makespan):.6f}\n")
        f.write(f"coverage_ok={bool(coverage.get('coverage_ok', False))}\n")
        f.write(f"unmet_sku_total={int(coverage.get('unmet_sku_total', 0))}\n")
        f.write(f"unmet_subtask_count={int(coverage.get('unmet_subtask_count', 0))}\n")
        if failures:
            f.write("failures:\n")
            for item in failures:
                f.write(f"- {item}\n")
    return result


def _build_solution_audit(problem: Any, best_z: float, verification_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    coverage = _compute_solution_coverage(problem)
    recomputed_z = float(getattr(problem, "global_makespan", 0.0) or 0.0)
    global_makespan = float(getattr(problem, "global_makespan", 0.0) or 0.0)
    makespan_consistent = bool(math.isfinite(float(best_z)) and abs(float(best_z) - recomputed_z) <= 1e-6 and abs(recomputed_z - global_makespan) <= 1e-6)
    invalid_station_rows: List[Dict[str, Any]] = []
    invalid_rank_rows: List[Dict[str, Any]] = []
    invalid_z_rows: List[Dict[str, Any]] = []
    unassigned_robot_rows: List[Dict[str, Any]] = []
    station_rank_rows: Dict[tuple[int, int], List[int]] = defaultdict(list)
    tote_to_task_rows: Dict[int, List[int]] = defaultdict(list)
    all_tasks: List[Any] = []
    for st in getattr(problem, "subtask_list", []) or []:
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
    duplicate_rank_rows = [{"station_id": int(station_id), "rank": int(rank), "subtask_ids": list(ids)} for (station_id, rank), ids in station_rank_rows.items() if len(ids) > 1]
    duplicate_tote_rows = {int(tote_id): list(task_ids) for tote_id, task_ids in tote_to_task_rows.items() if len(task_ids) > 1}
    verification_failures = list((verification_result or {}).get("failures", []) or [])
    has_unreasonable_solution = bool(not bool(coverage.get("coverage_ok", False)) or not makespan_consistent or invalid_station_rows or invalid_rank_rows or duplicate_rank_rows or invalid_z_rows or duplicate_tote_rows or unassigned_robot_rows or verification_failures)
    issue_summary: List[str] = []
    if int(coverage.get("unmet_sku_total", 0)) > 0:
        issue_summary.append(f"SKU coverage unmet: {int(coverage.get('unmet_sku_total', 0))} units")
    if not makespan_consistent:
        issue_summary.append(f"Makespan inconsistent: best_z={float(best_z):.6f}, recomputed_z={recomputed_z:.6f}, global_makespan={global_makespan:.6f}")
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
        "best_z": float(best_z),
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


def _write_warm_start_export(result_root: str, solver: Any, scale: str, seed: int) -> str:
    return ""

    warm = getattr(solver, "_warm_start", None)
    problem = getattr(solver, "_warm_start_problem_snapshot", None)
    if warm is None or problem is None:
        return ""

    out_dir = os.path.join(result_root, "warm_start_export")
    os.makedirs(out_dir, exist_ok=True)

    all_tasks: List[Any] = []
    for st in getattr(problem, "subtask_list", []) or []:
        all_tasks.extend(getattr(st, "execution_tasks", []) or [])
    for task in all_tasks:
        task.robot_service_time = float(_normalized_robot_service_time(problem, task))
    all_tasks.sort(key=lambda t: (int(getattr(t, "target_station_id", -1)), float(getattr(t, "start_process_time", 0.0)), int(getattr(t, "task_id", -1))))

    station_loads: Dict[int, float] = defaultdict(float)
    station_idle_total = 0.0
    for station in getattr(problem, "station_list", []) or []:
        station_idle_total += float(getattr(station, "total_idle_time", 0.0) or 0.0)
    for task in all_tasks:
        station_loads[int(getattr(task, "target_station_id", -1))] += float(getattr(task, "total_process_duration", 0.0) or 0.0)

    robot_finish: Dict[int, float] = defaultdict(float)
    robot_path_total = 0.0
    arrival_slack_rows: List[float] = []
    hit_stack_ids = set()
    total_noise = 0
    total_targets = 0
    stack_spans: List[float] = []
    sorting_cost_proxy = 0.0
    for task in all_tasks:
        rid = int(getattr(task, "robot_id", -1))
        robot_finish[rid] = max(robot_finish[rid], float(getattr(task, "arrival_time_at_station", 0.0) or 0.0))
        arr_stack = float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0)
        arr_station = float(getattr(task, "arrival_time_at_station", 0.0) or 0.0)
        if arr_station > 0.0 and arr_stack > 0.0:
            robot_path_total += max(0.0, arr_station - arr_stack)
        if float(getattr(task, "start_process_time", 0.0) or 0.0) > 0.0:
            arrival_slack_rows.append(max(0.0, float(getattr(task, "start_process_time", 0.0) or 0.0) - arr_station))
        hit_stack_ids.add(int(getattr(task, "target_stack_id", -1)))
        total_noise += len(getattr(task, "noise_tote_ids", []) or [])
        total_targets += len(getattr(task, "target_tote_ids", []) or [])
        sort_range = getattr(task, "sort_layer_range", None) or []
        if len(sort_range) == 2:
            stack_spans.append(float(int(sort_range[1]) - int(sort_range[0]) + 1))
        sorting_cost_proxy += float(getattr(task, "station_service_time", 0.0) or 0.0)

    coverage = _compute_solution_coverage(problem)
    station_load_values = list(station_loads.values()) or [0.0]
    avg_sku_per_subtask = 0.0
    max_sku_per_subtask = 0.0
    if getattr(problem, "subtask_list", None):
        sku_counts = [len(getattr(st, "sku_list", []) or []) for st in problem.subtask_list]
        avg_sku_per_subtask = float(sum(sku_counts) / max(1, len(sku_counts)))
        max_sku_per_subtask = float(max(sku_counts or [0]))

    summary = {
        "best_iter": -1,
        "best_z": float(getattr(warm, "makespan", 0.0) or 0.0),
        "recomputed_z": float(getattr(problem, "global_makespan", 0.0) or 0.0),
        "global_makespan": float(getattr(problem, "global_makespan", 0.0) or 0.0),
        "run_total_time_sec": float(getattr(warm, "sp4_runtime_sec", 0.0) or 0.0),
        "scale": str(scale),
        "seed": int(seed),
        "sp1": {
            "subtask_count": int(len(getattr(problem, "subtask_list", []) or [])),
            "avg_sku_per_subtask": float(avg_sku_per_subtask),
            "max_sku_per_subtask": float(max_sku_per_subtask),
        },
        "sp2": {
            "station_idle_total": float(station_idle_total),
            "station_load_max": float(max(station_load_values)),
            "station_load_std": float(0.0 if len(station_load_values) <= 1 else statistics.pstdev(station_load_values)),
        },
        "sp3": {
            "hit_stack_count": float(len(hit_stack_ids)),
            "noise_ratio": float(total_noise / max(1, total_targets)),
            "avg_stack_span": float(sum(stack_spans) / max(1, len(stack_spans))),
            "sorting_cost_proxy": float(sorting_cost_proxy),
        },
        "sp4": {
            "robot_path_length_total": float(robot_path_total),
            "latest_robot_finish": float(max(robot_finish.values() or [0.0])),
            "arrival_slack_mean": float(sum(arrival_slack_rows) / max(1, len(arrival_slack_rows))),
        },
        "coverage": coverage,
        "warm_start": {
            "sp2_mode": str(getattr(warm, "sp2_mode", "")),
            "sp4_mode": str(getattr(warm, "sp4_mode", "")),
            "sp4_error": str(getattr(warm, "sp4_error", "")),
            "sp4_runtime_sec": float(getattr(warm, "sp4_runtime_sec", 0.0) or 0.0),
        },
    }

    with open(os.path.join(out_dir, "best_solution_objectives.json"), "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(summary), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "best_solution_objectives.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_iter={summary['best_iter']}\n")
        f.write(f"best_z={summary['best_z']:.6f}\n")
        f.write(f"global_makespan={summary['global_makespan']:.6f}\n")
        f.write(f"run_total_time_sec={summary['run_total_time_sec']:.6f}\n")
        f.write(f"sp1_subtask_count={summary['sp1']['subtask_count']}, sp1_avg_sku_per_subtask={summary['sp1']['avg_sku_per_subtask']:.6f}, sp1_max_sku_per_subtask={summary['sp1']['max_sku_per_subtask']:.6f}\n")
        f.write(f"sp2_station_idle_total={summary['sp2']['station_idle_total']:.6f}, sp2_station_load_max={summary['sp2']['station_load_max']:.6f}, sp2_station_load_std={summary['sp2']['station_load_std']:.6f}\n")
        f.write(f"sp3_hit_stack_count={summary['sp3']['hit_stack_count']:.6f}, sp3_noise_ratio={summary['sp3']['noise_ratio']:.6f}, sp3_avg_stack_span={summary['sp3']['avg_stack_span']:.6f}, sp3_sorting_cost_proxy={summary['sp3']['sorting_cost_proxy']:.6f}\n")
        f.write(f"sp4_robot_path_length_total={summary['sp4']['robot_path_length_total']:.6f}, sp4_latest_robot_finish={summary['sp4']['latest_robot_finish']:.6f}, sp4_arrival_slack_mean={summary['sp4']['arrival_slack_mean']:.6f}\n")
        f.write(f"coverage_ok={summary['coverage']['coverage_ok']}, unmet_sku_total={summary['coverage']['unmet_sku_total']}, unmet_subtask_count={summary['coverage']['unmet_subtask_count']}\n")
        f.write(f"warm_sp2_mode={summary['warm_start']['sp2_mode']}, warm_sp4_mode={summary['warm_start']['sp4_mode']}, warm_sp4_runtime_sec={summary['warm_start']['sp4_runtime_sec']:.6f}\n")

    dump_path = os.path.join(out_dir, "best_solution_full_dump.txt")
    with open(dump_path, "w", encoding="utf-8") as f:
        f.write("[Warm Start Best Solution Dump]\n")
        f.write(f"seed={int(seed)}\n")
        f.write(f"best_z={float(getattr(warm, 'makespan', 0.0) or 0.0):.6f}\n")
        f.write(f"recomputed_z={float(getattr(problem, 'global_makespan', 0.0) or 0.0):.6f}\n")
        f.write(f"global_makespan={float(getattr(problem, 'global_makespan', 0.0) or 0.0):.6f}\n")
        f.write(f"warm_sp2_mode={str(getattr(warm, 'sp2_mode', ''))}\n")
        f.write(f"warm_sp4_mode={str(getattr(warm, 'sp4_mode', ''))}\n")
        f.write("\n[SP1 Decisions]\n")
        for st in sorted(getattr(problem, "subtask_list", []) or [], key=lambda x: int(getattr(x, "id", -1))):
            sku_ids = [int(getattr(s, "id", -1)) for s in (getattr(st, "sku_list", []) or [])]
            f.write(f"subtask_id={int(getattr(st, 'id', -1))}, order_id={int(getattr(getattr(st, 'parent_order', None), 'order_id', -1))}, sku_units={len(sku_ids)}, sku_list={sku_ids}\n")
        f.write("\n[SP2 Decisions]\n")
        for st in sorted(getattr(problem, "subtask_list", []) or [], key=lambda x: (int(getattr(x, "assigned_station_id", -1)), int(getattr(x, "station_sequence_rank", -1)), int(getattr(x, "id", -1)))):
            f.write(f"subtask_id={int(getattr(st, 'id', -1))}, station_id={int(getattr(st, 'assigned_station_id', -1))}, rank={int(getattr(st, 'station_sequence_rank', -1))}\n")
        f.write("\n[SP3 Decisions]\n")
        for t in sorted(all_tasks, key=lambda x: int(getattr(x, "task_id", -1))):
            f.write(
                f"task_id={int(getattr(t, 'task_id', -1))}, subtask_id={int(getattr(t, 'sub_task_id', -1))}, stack_id={int(getattr(t, 'target_stack_id', -1))}, "
                f"station_id={int(getattr(t, 'target_station_id', -1))}, mode={getattr(t, 'operation_mode', '')}, "
                f"target_totes={list(getattr(t, 'target_tote_ids', []) or [])}, hit_totes={list(getattr(t, 'hit_tote_ids', []) or [])}, "
                f"noise_totes={list(getattr(t, 'noise_tote_ids', []) or [])}, sort_range={getattr(t, 'sort_layer_range', None)}, "
                f"robot_service_time={float(getattr(t, 'robot_service_time', 0.0) or 0.0):.6f}, station_service_time={float(getattr(t, 'station_service_time', 0.0) or 0.0):.6f}\n"
            )
        f.write("\n[SP4 Decisions]\n")
        for t in sorted(all_tasks, key=lambda x: int(getattr(x, "task_id", -1))):
            f.write(
                f"task_id={int(getattr(t, 'task_id', -1))}, robot_id={int(getattr(t, 'robot_id', -1))}, trip_id={int(getattr(t, 'trip_id', 0))}, "
                f"arrival_stack={float(getattr(t, 'arrival_time_at_stack', 0.0) or 0.0):.6f}, arrival_station={float(getattr(t, 'arrival_time_at_station', 0.0) or 0.0):.6f}, "
                f"start_process={float(getattr(t, 'start_process_time', 0.0) or 0.0):.6f}, end_process={float(getattr(t, 'end_process_time', 0.0) or 0.0):.6f}\n"
            )

    verification_result = _verify_makespan_breakdown(problem, out_dir)
    audit = _build_solution_audit(problem, best_z=float(getattr(warm, "makespan", 0.0) or 0.0), verification_result=verification_result)
    with open(os.path.join(out_dir, "best_solution_audit.json"), "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(audit), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "best_solution_audit.txt"), "w", encoding="utf-8") as f:
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
        f.write("sku_hit_check=" + ("PASS" if not bool(audit.get("missing_sku_hit", False)) else "FAIL") + "\n")
        f.write("unreasonable_solution_check=" + ("PASS" if not bool(audit.get("has_unreasonable_solution", False)) else "FAIL") + "\n")
        if audit.get("issues"):
            f.write("issues:\n")
            for item in list(audit.get("issues", []) or []):
                f.write(f"- {item}\n")
    return out_dir


def _write_result_files(problem: Any, result: Any, scale: str, seed: int, cfg: GlobalXYZUConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = os.path.join(ROOT_DIR, "result", f"gurobi_{str(scale).lower()}_{timestamp}")
    os.makedirs(result_root, exist_ok=True)

    payload: Dict[str, Any] = {
        "scale": str(scale),
        "seed": int(seed),
        "status": str(result.status),
        "objective": float(result.objective),
        "gap": float(result.gap),
        "runtime_sec": float(result.runtime_sec),
        "gurobi_solve_time_sec": float(result.diagnostics.get("gurobi_solve_time_sec", 0.0) or 0.0),
        "gurobi_runtime_sec": float(result.diagnostics.get("gurobi_runtime_sec", 0.0) or 0.0),
        "subtask_count": int(result.subtask_count),
        "task_count": int(result.task_count),
        "global_makespan": float(getattr(problem, "global_makespan", 0.0) or 0.0),
        "station_schedule": result.station_schedule,
        "robot_routes": result.robot_routes,
        "diagnostics": result.diagnostics,
        "config": _normalize_jsonable(cfg.__dict__),
        "subtasks": _subtask_rows(problem),
        "tasks": _task_rows(problem),
    }
    json_path = os.path.join(result_root, "gurobi_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(payload), f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(result_root, "gurobi_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Global XYZU Gurobi Summary ===\n")
        f.write(f"scale={scale}\n")
        f.write(f"seed={int(seed)}\n")
        f.write(f"status={result.status}\n")
        f.write(f"objective={float(result.objective):.6f}\n")
        f.write(f"global_makespan={float(getattr(problem, 'global_makespan', 0.0) or 0.0):.6f}\n")
        f.write(f"runtime_sec={float(result.runtime_sec):.6f}\n")
        f.write(f"gurobi_solve_time_sec={float(result.diagnostics.get('gurobi_solve_time_sec', 0.0) or 0.0):.6f}\n")
        f.write(f"gurobi_runtime_sec={float(result.diagnostics.get('gurobi_runtime_sec', 0.0) or 0.0):.6f}\n")
        f.write(f"slot_time_ub={float(result.diagnostics.get('slot_time_ub', 0.0) or 0.0):.6f}\n")
        f.write(f"route_big_m={float(result.diagnostics.get('route_big_m', 0.0) or 0.0):.6f}\n")
        f.write(f"warm_makespan={float(result.diagnostics.get('warm_makespan', 0.0) or 0.0):.6f}\n")
        f.write(f"warm_route_end={float(result.diagnostics.get('warm_route_end', 0.0) or 0.0):.6f}\n")
        f.write(f"route_big_m_source={result.diagnostics.get('route_big_m_source', '')}\n")
        f.write(f"subtask_count={int(result.subtask_count)}\n")
        f.write(f"task_count={int(result.task_count)}\n")
        f.write(f"station_schedule={result.station_schedule}\n")
        f.write(f"robot_ids={sorted(result.robot_routes.keys())}\n")
        f.write(f"result_root={result_root}\n")

    warm_diag_payload = {
        "warm_start_route_steps": result.diagnostics.get("warm_start_route_steps", {}),
        "warm_start_slot_times": result.diagnostics.get("warm_start_slot_times", []),
        "warm_start_time_violations": result.diagnostics.get("warm_start_time_violations", []),
        "warm_start_model_cmax": result.diagnostics.get("warm_start_model_cmax", 0.0),
        "warm_start_route_end_max": result.diagnostics.get("warm_start_route_end_max", 0.0),
        "warm_start_route_end_gap": result.diagnostics.get("warm_start_route_end_gap", 0.0),
    }
    warm_diag_json = os.path.join(result_root, "warm_start_injection_timeline.json")
    with open(warm_diag_json, "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(warm_diag_payload), f, ensure_ascii=False, indent=2)

    warm_diag_txt = os.path.join(result_root, "warm_start_injection_timeline.txt")
    with open(warm_diag_txt, "w", encoding="utf-8") as f:
        f.write("[Warm Start Injection Timeline]\n")
        f.write(f"warm_start_model_cmax={float(result.diagnostics.get('warm_start_model_cmax', 0.0) or 0.0):.6f}\n")
        f.write(f"warm_start_route_end_max={float(result.diagnostics.get('warm_start_route_end_max', 0.0) or 0.0):.6f}\n")
        f.write(f"warm_start_route_end_gap={float(result.diagnostics.get('warm_start_route_end_gap', 0.0) or 0.0):.6f}\n")
        f.write("time_violations:\n")
        for row in list(result.diagnostics.get("warm_start_time_violations", []) or []):
            f.write(f"- {row}\n")
        f.write("slot_times:\n")
        for row in list(result.diagnostics.get("warm_start_slot_times", []) or []):
            f.write(f"- {row}\n")
        f.write("route_steps:\n")
        for robot_id, rows in dict(result.diagnostics.get("warm_start_route_steps", {}) or {}).items():
            f.write(f"robot_id={robot_id}\n")
            for row in list(rows or []):
                f.write(f"  {row}\n")
    return result_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone Global XYZU solver.")
    parser.add_argument("--scale", type=str, default="Gurobi-s1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--candidate-stack-topk", type=int, default=3)
    parser.add_argument("--max-rank", type=int, default=0)
    parser.add_argument("--write-lp", action="store_true")
    parser.add_argument("--quiet-gurobi", action="store_true", help="Disable Gurobi solver log output.")
    parser.add_argument("--disable-warm-start", action="store_true")
    parser.add_argument("--disable-integrated-u-route", action="store_true")
    parser.add_argument("--disable-route-arc-prune", action="store_true")
    parser.add_argument("--allow-multi-robot-slot", action="store_true")
    parser.add_argument("--warm-start-sp4", action="store_true", help="Deprecated: SP4/LKH warm start is now enabled by default.")
    parser.add_argument("--disable-warm-start-sp4", action="store_true", help="Disable SP4/LKH during warm-start construction and use greedy routing.")
    parser.add_argument("--enable-sp4-fallback", action="store_true", help="Allow SP4/ortools if the integrated MIP path falls back.")
    parser.add_argument("--max-candidate-stacks-per-order", type=int, default=24)
    parser.add_argument("--u-route-lkh", action="store_true", help="Use non-MIP routing in the U stage when SP4 is available.")
    parser.add_argument("--bom-arrival-window-sec", type=float, default=60.0)
    args = parser.parse_args()

    problem = CreateOFSProblem.generate_problem_by_scale(args.scale, seed=args.seed)
    cfg = GlobalXYZUConfig(
        time_limit_sec=float(args.time_limit),
        mip_gap=float(args.mip_gap),
        candidate_stack_topk=int(args.candidate_stack_topk),
        max_rank=int(args.max_rank),
        enable_warm_start=not bool(args.disable_warm_start),
        write_lp=bool(args.write_lp),
        gurobi_output=not bool(args.quiet_gurobi),
        max_candidate_stacks_per_order=int(args.max_candidate_stacks_per_order),
        u_route_use_mip=not bool(args.u_route_lkh),
        integrate_u_route=not bool(args.disable_integrated_u_route),
        route_arc_prune=False,
        u_same_slot_same_robot=not bool(args.allow_multi_robot_slot),
        bom_arrival_window_sec=float(args.bom_arrival_window_sec),
        warm_start_use_sp4=(not bool(args.disable_warm_start_sp4)) or bool(args.warm_start_sp4),
        enable_sp4_fallback=bool(args.enable_sp4_fallback),
    )
    solver = GlobalXYZUSolver()
    result = solver.solve(problem, cfg=cfg)
    result_root = _write_result_files(problem, result, scale=args.scale, seed=args.seed, cfg=cfg)
    warm_start_root = _write_warm_start_export(result_root=result_root, solver=solver, scale=args.scale, seed=args.seed)

    print("=== Global XYZU Result ===")
    print(f"status={result.status}")
    print(f"objective={result.objective:.6f}")
    print(f"gap={result.gap}")
    print(f"runtime_sec={result.runtime_sec:.6f}")
    print(f"gurobi_solve_time_sec={float(result.diagnostics.get('gurobi_solve_time_sec', 0.0)):.6f}")
    print(f"gurobi_runtime_sec={float(result.diagnostics.get('gurobi_runtime_sec', 0.0)):.6f}")
    print(f"model_cmax={float(result.diagnostics.get('model_cmax', 0.0) or 0.0):.6f}")
    print(f"validated_global_makespan={float(result.diagnostics.get('validated_global_makespan', result.objective) or result.objective):.6f}")
    if 'time_verify_cmax_diff' in result.diagnostics:
        print(f"time_verify_cmax_diff={float(result.diagnostics.get('time_verify_cmax_diff', 0.0) or 0.0):.6f}")
    print("=== Time Bounds ===")
    print(f"slot_time_ub={float(result.diagnostics.get('slot_time_ub', 0.0) or 0.0):.6f}")
    print(f"route_big_m={float(result.diagnostics.get('route_big_m', 0.0) or 0.0):.6f}")
    print(f"route_node_time_ub_max={float(result.diagnostics.get('route_node_time_ub_max', 0.0) or 0.0):.6f}")
    print(f"route_arc_time_m_max={float(result.diagnostics.get('route_arc_time_m_max', 0.0) or 0.0):.6f}")
    print(f"warm_makespan={float(result.diagnostics.get('warm_makespan', 0.0) or 0.0):.6f}")
    print(f"warm_route_end={float(result.diagnostics.get('warm_route_end', 0.0) or 0.0):.6f}")
    print(f"route_big_m_source={result.diagnostics.get('route_big_m_source', '')}")
    print(f"bom_arrival_window_sec={float(cfg.bom_arrival_window_sec):.6f}")
    print(f"subtask_count={result.subtask_count}")
    print(f"task_count={result.task_count}")
    print(f"station_schedule={result.station_schedule}")
    print(f"robot_ids={sorted(result.robot_routes.keys())}")
    print(f"result_root={result_root}")
    if warm_start_root:
        print(f"warm_start_root={warm_start_root}")
    print("=== Warm Start Injection ===")
    print(f"warm_start_model_cmax={float(result.diagnostics.get('warm_start_model_cmax', 0.0) or 0.0):.6f}")
    print(f"warm_start_route_end_max={float(result.diagnostics.get('warm_start_route_end_max', 0.0) or 0.0):.6f}")
    print(f"warm_start_route_end_gap={float(result.diagnostics.get('warm_start_route_end_gap', 0.0) or 0.0):.6f}")
    print(f"warm_start_time_violations={len(list(result.diagnostics.get('warm_start_time_violations', []) or []))}")
    for key in sorted(result.diagnostics.keys()):
        print(f"diag.{key}={result.diagnostics[key]}")


if __name__ == "__main__":
    main()
