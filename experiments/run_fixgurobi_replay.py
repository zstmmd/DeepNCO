import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver


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


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_int_list(text: str) -> List[int]:
    text = str(text or "").strip()
    if not text or text == "[]":
        return []
    return [int(v) for v in re.findall(r"-?\d+", text)]


def _parse_sort_range(text: str) -> Optional[List[int]]:
    values = _parse_int_list(text)
    if len(values) >= 2:
        return [int(values[0]), int(values[1])]
    return None


def parse_gurobi_export(export_dir: str) -> Dict[str, Any]:
    obj_path = os.path.join(export_dir, "best_solution_objectives.json")
    dump_path = os.path.join(export_dir, "best_solution_full_dump.txt")
    if not os.path.exists(dump_path):
        raise FileNotFoundError(f"missing dump: {dump_path}")
    objectives: Dict[str, Any] = {}
    if os.path.exists(obj_path):
        with open(obj_path, "r", encoding="utf-8") as f:
            objectives = json.load(f)
    header: Dict[str, Any] = {}
    subtasks: Dict[int, Dict[str, Any]] = {}
    tasks: Dict[int, Dict[str, Any]] = {}
    route_rows: List[Dict[str, Any]] = []
    route_order_by_robot: Dict[int, Dict[int, int]] = {}
    section = ""
    with open(dump_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]")
                continue
            if "=" in line and section in {"Gurobi Best Solution Dump", "TRA Best Solution Dump"}:
                key, value = line.split("=", 1)
                header[str(key).strip()] = value.strip()
                continue
            if section == "SP1 Decisions":
                m = re.search(r"subtask_id=(\d+), order_id=(\d+), sku_units=(\d+), sku_list=(\[.*\])", line)
                if m:
                    subtask_id = int(m.group(1))
                    subtasks[subtask_id] = {
                        "subtask_id": subtask_id,
                        "order_id": int(m.group(2)),
                        "sku_units": int(m.group(3)),
                        "sku_list": _parse_int_list(m.group(4)),
                    }
                    continue
                m = re.search(
                    r"subtask_id=(\d+), order_id=(\d+), sku_units=(\d+), unique_skus=(\[.*?\]), sku_list=(\[.*?\])",
                    line,
                )
                if m:
                    # TRA exports the expanded quantity list; GlobalXYZU fixed X expects one work unit per unique SKU.
                    subtask_id = int(m.group(1))
                    subtasks[subtask_id] = {
                        "subtask_id": subtask_id,
                        "order_id": int(m.group(2)),
                        "sku_units": int(m.group(3)),
                        "sku_list": _parse_int_list(m.group(4)),
                        "expanded_sku_list": _parse_int_list(m.group(5)),
                    }
            elif section == "SP2 Decisions":
                m = re.search(r"subtask_id=(\d+), station_id=(-?\d+), rank=(-?\d+)", line)
                if m:
                    subtask_id = int(m.group(1))
                    subtasks.setdefault(subtask_id, {"subtask_id": subtask_id})
                    subtasks[subtask_id].update({"station_id": int(m.group(2)), "rank": int(m.group(3))})
            elif section == "SP3 Decisions":
                m = re.search(
                    r"task_id=(\d+), subtask_id=(\d+), stack_id=(-?\d+), station_id=(-?\d+), mode=([^,]+), "
                    r"target_totes=(\[.*?\]), hit_totes=(\[.*?\]), noise_totes=(\[.*?\]), sort_range=([^,]+(?:, \d+\))?|None), "
                    r"(?:load=-?\d+, sku_pick_count=-?\d+, )?robot_service_time=([0-9.\-]+), station_service_time=([0-9.\-]+)",
                    line,
                )
                if m:
                    task_id = int(m.group(1))
                    tasks[task_id] = {
                        "task_id": task_id,
                        "subtask_id": int(m.group(2)),
                        "stack_id": int(m.group(3)),
                        "station_id": int(m.group(4)),
                        "mode": str(m.group(5)).strip().upper(),
                        "target_tote_ids": _parse_int_list(m.group(6)),
                        "hit_tote_ids": _parse_int_list(m.group(7)),
                        "noise_tote_ids": _parse_int_list(m.group(8)),
                        "sort_layer_range": _parse_sort_range(m.group(9)),
                        "robot_service_time": float(m.group(10)),
                        "station_service_time": float(m.group(11)),
                    }
            elif section == "SP4 Decisions":
                m = re.search(
                    r"task_id=(\d+), robot_id=(-?\d+), trip_id=(-?\d+), arrival_stack=([0-9.\-]+), "
                    r"arrival_station=([0-9.\-]+)(?:, start_process=([0-9.\-]+), end_process=([0-9.\-]+))?",
                    line,
                )
                if m:
                    route_rows.append(
                        {
                            "task_id": int(m.group(1)),
                            "robot_id": int(m.group(2)),
                            "trip_id": int(m.group(3)),
                            "arrival_stack": float(m.group(4)),
                            "arrival_station": float(m.group(5)),
                            "start_process": float(m.group(6)) if m.group(6) is not None else float("nan"),
                            "end_process": float(m.group(7)) if m.group(7) is not None else float("nan"),
                        }
                    )
            elif section == "SP4 Trips By Robot":
                m = re.search(r"robot_id=(-?\d+), trip_id=(-?\d+), task_ids=(\[.*?\])", line)
                if m:
                    robot_id = int(m.group(1))
                    for pos, task_id in enumerate(_parse_int_list(m.group(3))):
                        route_order_by_robot.setdefault(robot_id, {})[int(task_id)] = int(pos)
    for task in tasks.values():
        subtask_id = int(task["subtask_id"])
        subtasks.setdefault(subtask_id, {"subtask_id": subtask_id})
        subtasks[subtask_id].setdefault("tasks", []).append(task)
    for route in route_rows:
        task = tasks.get(int(route["task_id"]))
        if task is not None:
            route.update(task)
        route["_route_order"] = int(route_order_by_robot.get(int(route.get("robot_id", -1)), {}).get(int(route.get("task_id", -1)), int(route.get("task_id", -1))))
    route_rows.sort(key=lambda row: (int(row.get("robot_id", -1)), int(row.get("trip_id", 0)), int(row.get("_route_order", row.get("task_id", -1)))))
    return {
        "objectives": objectives,
        "header": header,
        "subtasks": subtasks,
        "tasks": tasks,
        "routes": route_rows,
    }


def build_fixed_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    subtasks = dict(parsed.get("subtasks", {}) or {})
    rows_by_order: Dict[int, List[Dict[str, Any]]] = {}
    for subtask in subtasks.values():
        order_id = int(subtask.get("order_id", -1))
        rows_by_order.setdefault(order_id, []).append(dict(subtask))
    for rows in rows_by_order.values():
        rows.sort(
            key=lambda row: (
                -int(len(row.get("sku_list", []) or [])),
                int(row.get("station_id", 10**9)),
                int(row.get("rank", 10**9)),
                int(row.get("subtask_id", 10**9)),
            )
        )

    fixed_slot_count_by_order: Dict[int, int] = {}
    fixed_work_units_by_order_slot: Dict[int, List[List[str]]] = {}
    fixed_station_rank_by_order_slot: Dict[int, List[Tuple[int, int]]] = {}
    fixed_z_descriptors_by_order_slot: Dict[int, List[List[Dict[str, Any]]]] = {}
    used_stack_ids_by_order: Dict[int, List[int]] = {}
    subtask_to_order_local: Dict[int, Tuple[int, int]] = {}

    for order_id, rows in rows_by_order.items():
        fixed_slot_count_by_order[int(order_id)] = int(len(rows))
        unit_rows: List[List[str]] = []
        y_rows: List[Tuple[int, int]] = []
        z_rows: List[List[Dict[str, Any]]] = []
        used_stack_ids = set()
        for local_idx, row in enumerate(rows):
            subtask_id = int(row.get("subtask_id", -1))
            subtask_to_order_local[subtask_id] = (int(order_id), int(local_idx))
            unit_rows.append([f"{int(order_id)}:{int(sku_id)}" for sku_id in sorted(int(v) for v in row.get("sku_list", []) or [])])
            y_rows.append((int(row.get("station_id", -1)), int(row.get("rank", -1))))
            descriptors = []
            for task in sorted(row.get("tasks", []) or [], key=lambda item: int(item.get("task_id", -1))):
                stack_id = int(task.get("stack_id", -1))
                if stack_id >= 0:
                    used_stack_ids.add(stack_id)
                descriptors.append(
                    {
                        "stack_id": stack_id,
                        "mode": str(task.get("mode", "FLIP")).upper(),
                        "target_tote_ids": [int(v) for v in task.get("target_tote_ids", []) or []],
                        "hit_tote_ids": [int(v) for v in task.get("hit_tote_ids", []) or []],
                        "noise_tote_ids": [int(v) for v in task.get("noise_tote_ids", []) or []],
                        "sort_layer_range": task.get("sort_layer_range", None),
                    }
                )
            z_rows.append(descriptors)
        fixed_work_units_by_order_slot[int(order_id)] = unit_rows
        fixed_station_rank_by_order_slot[int(order_id)] = y_rows
        fixed_z_descriptors_by_order_slot[int(order_id)] = z_rows
        used_stack_ids_by_order[int(order_id)] = sorted(used_stack_ids)

    route_sequence_by_robot: Dict[int, List[Dict[str, Any]]] = {}
    for route in parsed.get("routes", []) or []:
        subtask_id = int(route.get("subtask_id", -1))
        order_id, local_idx = subtask_to_order_local.get(subtask_id, (-1, -1))
        route_sequence_by_robot.setdefault(int(route.get("robot_id", -1)), []).append(
            {
                "task_id": int(route.get("task_id", -1)),
                "order_id": int(order_id),
                "local_slot_index": int(local_idx),
                "subtask_id": int(subtask_id),
                "stack_id": int(route.get("stack_id", -1)),
                "station_id": int(route.get("station_id", -1)),
                "trip_id": int(route.get("trip_id", 0)),
                "arrival_stack": float(route.get("arrival_stack", 0.0)),
                "arrival_station": float(route.get("arrival_station", 0.0)),
            }
        )
    return {
        "fixed_slot_count_by_order": fixed_slot_count_by_order,
        "fixed_work_units_by_order_slot": fixed_work_units_by_order_slot,
        "fixed_station_rank_by_order_slot": fixed_station_rank_by_order_slot,
        "fixed_z_descriptors_by_order_slot": fixed_z_descriptors_by_order_slot,
        "fixed_used_stack_ids_by_order": used_stack_ids_by_order,
        "forced_candidate_stacks_by_order": used_stack_ids_by_order,
        "fixed_route_task_sequence_by_robot": route_sequence_by_robot,
    }


def _cfg_for_phase(args, phase: str, payload: Dict[str, Any]) -> GlobalXYZUConfig:
    phase = str(phase).upper()
    cfg = GlobalXYZUConfig(
        time_limit_sec=float(args.time_limit),
        mip_gap=float(args.mip_gap),
        candidate_stack_topk=999,
        max_candidate_stacks_per_order=0,
        enable_warm_candidate_stack_prune=False,
        candidate_station_topk_per_stack=999,
        route_pickup_neighbor_limit=0,
        enable_scale_adaptive_candidate_prune=False,
        enable_warm_start=False,
        warm_start_use_sp4=False,
        fixgurobi_no_warm_start=True,
        fixgurobi_allow_warm_start_fallback=False,
        integrate_u_route=True,
        gurobi_output=bool(args.gurobi_output),
        forced_candidate_stacks_by_order=payload.get("forced_candidate_stacks_by_order"),
        fixed_slot_count_by_order=payload.get("fixed_slot_count_by_order"),
        fixed_work_units_by_order_slot=payload.get("fixed_work_units_by_order_slot"),
    )
    if phase in {"XY", "XYZ", "XYZ_USED_STACK", "XYZ_USED_STACK_ROUTE"}:
        cfg.fixed_station_rank_by_order_slot = payload.get("fixed_station_rank_by_order_slot")
    if phase in {"XYZ", "XYZ_USED_STACK", "XYZ_USED_STACK_ROUTE"}:
        cfg.fixed_z_descriptors_by_order_slot = payload.get("fixed_z_descriptors_by_order_slot")
    if phase in {"XYZ_USED_STACK", "XYZ_USED_STACK_ROUTE"}:
        cfg.fixed_used_stack_ids_by_order = payload.get("fixed_used_stack_ids_by_order")
    if phase == "XYZ_USED_STACK_ROUTE":
        cfg.fixed_route_task_sequence_by_robot = payload.get("fixed_route_task_sequence_by_robot")
    return cfg


def _mismatch_reason(row: Dict[str, Any], target_cmax: float) -> str:
    if str(row.get("status", "")) in {"FIXGUROBI_FAILED", "INFEASIBLE"}:
        return str(row.get("fallback_reason", row.get("status", "")))
    model_cmax = _safe_float(row.get("model_cmax"))
    if math.isfinite(target_cmax) and math.isfinite(model_cmax) and abs(model_cmax - target_cmax) > 1e-6:
        return f"model_cmax_mismatch:{model_cmax:.6f}!={target_cmax:.6f}"
    if bool(row.get("warm_start_applied", False)):
        return "warm_start_applied"
    return ""


def run_case(args, case_name: str, out_dir: str) -> List[Dict[str, Any]]:
    case_name = str(case_name).upper()
    export_dir = str(args.export_dir or KNOWN_GUROBI_EXPORT_DIRS.get(case_name, ""))
    if not export_dir or not os.path.exists(export_dir):
        raise FileNotFoundError(f"missing export dir for {case_name}: {export_dir}")
    parsed = parse_gurobi_export(export_dir)
    payload = build_fixed_payload(parsed)
    target_cmax = _safe_float(
        parsed.get("objectives", {}).get(
            "model_cmax",
            parsed.get("header", {}).get("model_cmax", parsed.get("header", {}).get("best_z", float("nan"))),
        )
    )
    target_obj = _safe_float(
        parsed.get("objectives", {}).get(
            "model_objective",
            parsed.get("header", {}).get("model_objective", parsed.get("header", {}).get("best_z", float("nan"))),
        )
    )
    phases = [str(v).upper() for v in (getattr(args, "phases", None) or ["X", "XY", "XYZ", "XYZ_USED_STACK", "XYZ_USED_STACK_ROUTE"])]
    rows: List[Dict[str, Any]] = []
    for phase in phases:
        problem = CreateOFSProblem.generate_problem_by_scale(case_name, seed=int(args.seed))
        cfg = _cfg_for_phase(args, phase, payload)
        if case_name == "GUROBI-S7":
            cfg.route_arc_prune = False
            cfg.enable_route_time_window_arc_prune = False
            cfg.enable_route_load_interval_arc_prune = False
        t0 = time.perf_counter()
        result = GlobalXYZUSolver().solve(problem, cfg=cfg)
        runtime = float(time.perf_counter() - t0)
        diag = dict(getattr(result, "diagnostics", {}) or {})
        model_cmax = _safe_float(diag.get("model_cmax", diag.get("validated_global_makespan", float("nan"))))
        row = {
            "case": case_name,
            "phase": phase,
            "status": str(getattr(result, "status", "")),
            "target_cmax": float(target_cmax),
            "target_objective": float(target_obj),
            "model_cmax": float(model_cmax),
            "model_objective": _safe_float(diag.get("model_objective", getattr(result, "objective", float("nan")))),
            "gap": _safe_float(getattr(result, "gap", float("nan"))),
            "bound": _safe_float(diag.get("model_best_bound", float("nan"))),
            "runtime": float(runtime),
            "solver_runtime": _safe_float(diag.get("gurobi_runtime_sec", float("nan"))),
            "warm_start_disabled": bool(diag.get("fixgurobi_warm_start_disabled", False)),
            "warm_start_applied": bool(diag.get("fixgurobi_warm_start_applied", False)),
            "fixed_constraint_count": int(diag.get("fixgurobi_fixed_constraint_count", 0) or 0),
            "invalid_fix_count": int(diag.get("fixgurobi_invalid_fix_count", 0) or 0),
            "route_sequence_missing_count": int(diag.get("fixgurobi_fixed_route_sequence_missing_count", 0) or 0),
            "tra_baseline_cmax": float(CURRENT_TRA_BASELINE_CMAX.get(case_name, float("nan"))),
            "gap_vs_global": float(model_cmax - target_cmax) if math.isfinite(model_cmax) and math.isfinite(target_cmax) else float("nan"),
            "gap_vs_tra": float(model_cmax - CURRENT_TRA_BASELINE_CMAX.get(case_name, float("nan"))) if math.isfinite(model_cmax) else float("nan"),
            "fallback_reason": str(diag.get("fallback_reason", "")),
        }
        row["mismatch_reason"] = _mismatch_reason(row, target_cmax)
        rows.append(row)
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exported Gurobi decisions through the FixGurobi fixed model path.")
    parser.add_argument("--cases", nargs="+", default=["GUROBI-S1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--phases", nargs="+", default=["X", "XY", "XYZ", "XYZ_USED_STACK", "XYZ_USED_STACK_ROUTE"])
    parser.add_argument("--summary-phase", type=str, default="XYZ_USED_STACK")
    parser.add_argument("--export-dir", type=str, default="", help="Single-case override for gurobi_solution_export dir")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--gurobi-output", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = _ensure_dir(args.output_dir or os.path.join(ROOT_DIR, "result", f"fixgurobi_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    all_rows: List[Dict[str, Any]] = []
    for case_name in [str(v).upper() for v in args.cases]:
        print(f"[FixGurobiReplay] case={case_name}")
        rows = run_case(args, case_name, out_dir)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"  phase={row['phase']} status={row['status']} model_cmax={row['model_cmax']} "
                f"target={row['target_cmax']} runtime={row['runtime']:.3f}s mismatch={row['mismatch_reason']}"
            )
    _write_csv(os.path.join(out_dir, "fixgurobi_replay_report.csv"), all_rows)
    with open(os.path.join(out_dir, "fixgurobi_replay_report.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "output_dir": out_dir}, f, ensure_ascii=False, indent=2)
    summary_phase = str(args.summary_phase).upper()
    _write_csv(os.path.join(out_dir, "fixgurobi_s1_s9_summary.csv"), [row for row in all_rows if str(row.get("phase", "")).upper() == summary_phase])
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
