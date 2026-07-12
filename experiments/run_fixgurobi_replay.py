import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from Gurobi.resource_time_alns.route_edge_audit import allowed_route_edges_from_global_payload, audit_fixed_route_edges


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



def _install_runtime_configs(path: str) -> None:
    if not str(path or "").strip():
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"runtime config json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs", payload) if isinstance(payload, dict) else {}
    if not isinstance(configs, dict):
        return
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    for name, cfg in configs.items():
        if isinstance(cfg, dict):
            CreateOFSProblem.RUNTIME_SCALE_CONFIGS[str(name).upper()] = dict(cfg)
def _cmd_int(command: str, flag: str, default: int) -> int:
    parts = str(command or "").split()
    for idx, token in enumerate(parts[:-1]):
        if token == flag:
            try:
                return int(parts[idx + 1])
            except Exception:
                return int(default)
    return int(default)


def _case_run_params(path: str, case_name: str) -> Dict[str, Any]:
    path = str(path or "").strip()
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs", payload) if isinstance(payload, dict) else {}
    if not isinstance(configs, dict):
        return {}
    case_key = str(case_name or "").upper()
    for key, value in configs.items():
        if str(key).upper() == case_key and isinstance(value, dict):
            return dict(value)
    return {}

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
    route_node_sequence_by_robot: Dict[int, List[Dict[str, Any]]] = {}
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
            elif section == "SP4 Full Node Sequence By Robot":
                m = re.search(r"robot_id=(-?\d+), sequence=(.*)", line)
                if m:
                    robot_id = int(m.group(1))
                    nodes: List[Dict[str, Any]] = []
                    for token in str(m.group(2)).split(" -> "):
                        text = token.strip()
                        if text.startswith("start("):
                            nodes.append({"kind": "start"})
                            continue
                        if text.startswith("end("):
                            nodes.append({"kind": "end"})
                            continue
                        nm = re.search(r"(pickup|delivery)\(task=(-?\d+),(?:stack|station)=(-?\d+),t=([0-9.\-]+)\)", text)
                        if nm:
                            nodes.append({
                                "kind": str(nm.group(1)),
                                "task_id": int(nm.group(2)),
                                "location_id": int(nm.group(3)),
                                "time": float(nm.group(4)),
                            })
                    if nodes:
                        route_node_sequence_by_robot[robot_id] = nodes
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
        "route_node_sequence_by_robot": route_node_sequence_by_robot,
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
                        "task_id": int(task.get("task_id", -1)),
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
    route_node_sequence_by_robot: Dict[int, List[Dict[str, Any]]] = {}
    task_lookup = {int(task_id): dict(task) for task_id, task in dict(parsed.get("tasks", {}) or {}).items()}
    parsed_node_sequences = dict(parsed.get("route_node_sequence_by_robot", {}) or {})
    if not parsed_node_sequences:
        rebuilt: Dict[int, List[Dict[str, Any]]] = {}
        by_robot: Dict[int, List[Dict[str, Any]]] = {}
        for route in parsed.get("routes", []) or []:
            by_robot.setdefault(int(route.get("robot_id", -1)), []).append(dict(route))
        for robot_id, rows in by_robot.items():
            events: List[Dict[str, Any]] = [{"kind": "start", "time": 0.0, "_rank": -1, "task_id": -1}]
            for row in rows:
                task_id = int(row.get("task_id", -1))
                events.append({"kind": "pickup", "task_id": task_id, "time": float(row.get("arrival_stack", 0.0)), "_rank": 0})
                events.append({"kind": "delivery", "task_id": task_id, "time": float(row.get("arrival_station", 0.0)), "_rank": 1})
            events.sort(key=lambda item: (float(item.get("time", 0.0)), int(item.get("_rank", 0)), int(item.get("task_id", -1))))
            events.append({"kind": "end", "time": max([float(v.get("time", 0.0)) for v in events] + [0.0]), "_rank": 2, "task_id": 10**9})
            rebuilt[int(robot_id)] = events
        parsed_node_sequences = rebuilt
    for robot_id, nodes in parsed_node_sequences.items():
        out_nodes: List[Dict[str, Any]] = []
        for node in nodes or []:
            kind = str(node.get("kind", "")).lower()
            if kind in {"start", "end"}:
                out_nodes.append({"kind": kind})
                continue
            task_id = int(node.get("task_id", -1))
            task = task_lookup.get(task_id, {})
            subtask_id = int(task.get("subtask_id", -1))
            order_id, local_idx = subtask_to_order_local.get(subtask_id, (-1, -1))
            out_nodes.append({
                "kind": kind,
                "task_id": task_id,
                "order_id": int(order_id),
                "local_slot_index": int(local_idx),
                "subtask_id": int(subtask_id),
                "stack_id": int(task.get("stack_id", -1)),
                "station_id": int(task.get("station_id", -1)),
                "time": float(node.get("time", 0.0)),
            })
        if out_nodes:
            route_node_sequence_by_robot[int(robot_id)] = out_nodes
    return {
        "fixed_slot_count_by_order": fixed_slot_count_by_order,
        "fixed_work_units_by_order_slot": fixed_work_units_by_order_slot,
        "fixed_station_rank_by_order_slot": fixed_station_rank_by_order_slot,
        "fixed_z_descriptors_by_order_slot": fixed_z_descriptors_by_order_slot,
        "fixed_used_stack_ids_by_order": used_stack_ids_by_order,
        "forced_candidate_stacks_by_order": used_stack_ids_by_order,
        "fixed_route_task_sequence_by_robot": route_sequence_by_robot,
        "fixed_route_node_sequence_by_robot": route_node_sequence_by_robot,
    }


def _cfg_for_phase(args, phase: str, payload: Dict[str, Any], case_name: str = "") -> GlobalXYZUConfig:
    phase = str(phase).upper()
    case_run = _case_run_params(str(getattr(args, "runtime_config_json", "") or ""), str(case_name).upper())
    use_case_prune = bool(getattr(args, "use_case_run_prune", False))
    disable_all_prune = bool(case_run.get("disable_all_prune", False)) if use_case_prune else False
    command_text = str(case_run.get("command", "") or "")
    candidate_stack_topk = int(case_run.get("candidate_stack_topk", _cmd_int(command_text, "--candidate-stack-topk", 999)) or 999) if use_case_prune else 999
    candidate_station_topk = int(case_run.get("gurobi_station_topk", 999) or 999) if use_case_prune else 999
    if phase == "XYZ_USED_STACK_ROUTE":
        candidate_station_topk = 999
    route_neighbor = int(case_run.get("route_pickup_neighbor_limit", _cmd_int(command_text, "--route-pickup-neighbor-limit", 0)) or 0) if use_case_prune else 0
    max_candidate = 0 if (not use_case_prune or disable_all_prune) else int(getattr(args, "case_run_max_candidate_stacks_per_order", 8) or 8)
    cfg = GlobalXYZUConfig(
        time_limit_sec=float(args.time_limit),
        mip_gap=float(args.mip_gap),
        candidate_stack_topk=int(candidate_stack_topk),
        max_candidate_stacks_per_order=int(max_candidate),
        enable_warm_candidate_stack_prune=False,
        candidate_station_topk_per_stack=int(candidate_station_topk),
        route_pickup_neighbor_limit=int(route_neighbor),
        enable_scale_adaptive_candidate_prune=False,
        enable_warm_start=False,
        warm_start_use_sp4=False,
        fixgurobi_no_warm_start=True,
        fixgurobi_allow_warm_start_fallback=False,
        integrate_u_route=True,
        route_arc_prune=not bool(disable_all_prune),
        enable_route_time_window_arc_prune=not bool(disable_all_prune),
        enable_route_load_interval_arc_prune=not bool(disable_all_prune),
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
        node_sequence = payload.get("fixed_route_node_sequence_by_robot")
        cfg.fixed_route_node_sequence_by_robot = node_sequence
        cfg.fixed_route_task_sequence_by_robot = None if node_sequence else payload.get("fixed_route_task_sequence_by_robot")
        cfg.enable_resource_lex_symmetry = False
        cfg.enable_robot_finish_lex_symmetry = False
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


def _fixed_route_edge_audit(problem: Any, cfg: GlobalXYZUConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not (payload.get("fixed_route_node_sequence_by_robot") or payload.get("fixed_route_task_sequence_by_robot")):
        return {"ok": True, "enabled": False, "reason": "empty_route_sequence"}
    try:
        compiled = GlobalXYZUSolver().compile_model(problem, cfg)
        allowed_edges = allowed_route_edges_from_global_payload(getattr(compiled, "vars_payload", {}) or {})
        audit = audit_fixed_route_edges(
            allowed_edges,
            route_task_sequence=payload.get("fixed_route_task_sequence_by_robot"),
            route_node_sequence=payload.get("fixed_route_node_sequence_by_robot"),
        )
        audit.update({"enabled": True, "source": "fixed_replay_global_compile"})
        return audit
    except Exception as exc:
        return {"ok": False, "enabled": True, "reason": "audit_exception", "error": str(exc)}


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
        cfg = _cfg_for_phase(args, phase, payload, case_name=case_name)
        if case_name == "GUROBI-S7":
            cfg.route_arc_prune = False
            cfg.enable_route_time_window_arc_prune = False
            cfg.enable_route_load_interval_arc_prune = False
        route_edge_audit = (
            _fixed_route_edge_audit(problem, cfg, payload)
            if phase == "XYZ_USED_STACK_ROUTE"
            else {"ok": True, "enabled": False}
        )
        t0 = time.perf_counter()
        if not bool(route_edge_audit.get("ok", True)):
            result = SimpleNamespace(
                status="ROUTE_EDGE_AUDIT_FAILED",
                diagnostics={"fallback_reason": "full_global_route_edge_missing"},
                gap=float("nan"),
                objective=float("nan"),
            )
        else:
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
            "route_edge_audit_ok": bool(route_edge_audit.get("ok", True)),
            "route_edge_audit_missing_count": int(route_edge_audit.get("missing_edge_count", 0) or 0),
            "route_edge_audit": json.dumps(route_edge_audit, ensure_ascii=False, default=str),
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
    parser.add_argument("--runtime-config-json", type=str, default="")
    parser.add_argument("--use-case-run-prune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--case-run-max-candidate-stacks-per-order", type=int, default=8)
    args = parser.parse_args()
    _install_runtime_configs(str(args.runtime_config_json or ""))

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
