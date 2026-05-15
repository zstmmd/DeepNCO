import argparse
import ast
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problemDto.createInstance import CreateOFSProblem
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


def _split_top_level_csv(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                parts.append(token)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return ""
    if raw == "None":
        return None
    if raw in {"True", "False"}:
        return raw == "True"
    if raw.startswith("[") or raw.startswith("(") or raw.startswith("{"):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def _parse_kv_line(line: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for token in _split_top_level_csv(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key.strip()] = _parse_value(value)
    return result


def _parse_alns_export(export_dir: str) -> Dict[str, Any]:
    dump_path = os.path.join(export_dir, "best_solution_full_dump.txt")
    if not os.path.exists(dump_path):
        raise FileNotFoundError(f"ALNS dump not found: {dump_path}")

    sections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    section = "header"
    with open(dump_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                continue
            if section == "header":
                sections[section].append(_parse_kv_line(line))
                continue
            row = _parse_kv_line(line)
            if row:
                sections[section].append(row)

    header: Dict[str, Any] = {}
    for row in sections.get("header", []):
        header.update(row)
    if not header:
        for section_name, rows in sections.items():
            if section_name == "header":
                continue
            merged: Dict[str, Any] = {}
            for row in rows:
                merged.update(row)
            if any(key in merged for key in ("best_z", "recomputed_z", "global_makespan", "seed")):
                header = merged
                break

    subtasks: Dict[int, Dict[str, Any]] = {}
    for row in sections.get("SP1 Decisions", []):
        subtasks[int(row["subtask_id"])] = {
            "subtask_id": int(row["subtask_id"]),
            "order_id": int(row["order_id"]),
            "sku_list": [int(v) for v in list(row.get("sku_list", []) or [])],
        }

    for row in sections.get("SP2 Decisions", []):
        subtask_id = int(row["subtask_id"])
        subtasks.setdefault(subtask_id, {"subtask_id": subtask_id})
        subtasks[subtask_id]["station_id"] = int(row["station_id"])
        subtasks[subtask_id]["rank"] = int(row["rank"])

    subtask_robot: Dict[int, int] = {}
    task_robot: Dict[int, int] = {}
    task_trip: Dict[int, int] = {}
    task_arrival_stack: Dict[int, float] = {}
    task_arrival_station: Dict[int, float] = {}
    for row in sections.get("SP4 Decisions", []):
        if "task_id" in row:
            task_id = int(row["task_id"])
            task_robot[task_id] = int(row["robot_id"])
            task_trip[task_id] = int(row.get("trip_id", 0) or 0)
            task_arrival_stack[task_id] = float(row.get("arrival_stack", 0.0) or 0.0)
            task_arrival_station[task_id] = float(row.get("arrival_station", 0.0) or 0.0)
        elif "subtask_id" in row and "assigned_robot_id" in row:
            subtask_robot[int(row["subtask_id"])] = int(row["assigned_robot_id"])

    tasks: Dict[int, Dict[str, Any]] = {}
    for row in sections.get("SP3 Decisions", []):
        task_id = int(row["task_id"])
        tasks[task_id] = {
            "task_id": task_id,
            "subtask_id": int(row["subtask_id"]),
            "stack_id": int(row["stack_id"]),
            "station_id": int(row["station_id"]),
            "mode": str(row["mode"]),
            "target_totes": [int(v) for v in list(row.get("target_totes", []) or [])],
            "hit_totes": [int(v) for v in list(row.get("hit_totes", []) or [])],
            "noise_totes": [int(v) for v in list(row.get("noise_totes", []) or [])],
            "sort_range": None if row.get("sort_range") is None else [int(v) for v in list(row.get("sort_range", []) or [])],
            "robot_service_time": float(row.get("robot_service_time", 0.0) or 0.0),
            "station_service_time": float(row.get("station_service_time", 0.0) or 0.0),
            "robot_id": int(task_robot.get(task_id, subtask_robot.get(int(row["subtask_id"]), -1))),
            "trip_id": int(task_trip.get(task_id, 0)),
            "arrival_stack": float(task_arrival_stack.get(task_id, 0.0)),
            "arrival_station": float(task_arrival_station.get(task_id, 0.0)),
        }

    for row in sections.get("Z Reproduction Fields", []):
        if "task_id" not in row:
            continue
        task_id = int(row["task_id"])
        if task_id not in tasks:
            continue
        tasks[task_id]["start_process_time"] = float(row.get("start_process_time", 0.0) or 0.0)
        tasks[task_id]["end_process_time"] = float(row.get("end_process_time", 0.0) or 0.0)
        tasks[task_id]["picking_duration"] = float(row.get("picking_duration", 0.0) or 0.0)
        tasks[task_id]["total_process_duration"] = float(row.get("total_process_duration", 0.0) or 0.0)

    trips_by_robot: Dict[int, List[List[int]]] = defaultdict(list)
    for row in sections.get("SP4 Trips By Robot", []):
        robot_id = int(row["robot_id"])
        task_ids = [int(v) for v in list(row.get("task_ids", []) or [])]
        if task_ids:
            trips_by_robot[robot_id].append(task_ids)
    if not trips_by_robot:
        by_robot_trip: Dict[Tuple[int, int], List[Tuple[float, int]]] = defaultdict(list)
        for task_id, row in tasks.items():
            by_robot_trip[(int(row["robot_id"]), int(row.get("trip_id", 0)))].append((float(row.get("arrival_stack", 0.0) or 0.0), task_id))
        for (robot_id, trip_id), pairs in sorted(by_robot_trip.items(), key=lambda item: (item[0][0], item[0][1])):
            del trip_id
            trips_by_robot[int(robot_id)].append([int(task_id) for _, task_id in sorted(pairs, key=lambda item: (item[0], item[1]))])

    return {
        "header": header,
        "subtasks": subtasks,
        "tasks": tasks,
        "subtask_robot": subtask_robot,
        "trips_by_robot": {int(k): list(v) for k, v in trips_by_robot.items()},
    }


def _augment_prepared_with_alns_solution(prepared: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(prepared)
    candidate_stacks_by_order = {
        int(order_id): list(stack_ids)
        for order_id, stack_ids in dict(prepared.get("candidate_stacks_by_order", {}) or {}).items()
    }
    support_totes_by_order = {
        int(order_id): list(tote_ids)
        for order_id, tote_ids in dict(prepared.get("support_totes_by_order", {}) or {}).items()
    }
    demand_hit_totes_by_order = {
        int(order_id): list(tote_ids)
        for order_id, tote_ids in dict(prepared.get("demand_hit_totes_by_order", {}) or {}).items()
    }
    tote_ids_by_order = {
        int(order_id): list(tote_ids)
        for order_id, tote_ids in dict(prepared.get("tote_ids_by_order", {}) or {}).items()
    }
    problem = prepared["problem"]

    for subtask_id, subtask_row in parsed["subtasks"].items():
        order_id = int(subtask_row["order_id"])
        del subtask_id
        candidate_stacks_by_order.setdefault(order_id, [])
        support_totes_by_order.setdefault(order_id, [])
        demand_hit_totes_by_order.setdefault(order_id, [])
        tote_ids_by_order.setdefault(order_id, [])

    for task in parsed["tasks"].values():
        order_id = int(parsed["subtasks"][int(task["subtask_id"])]["order_id"])
        stack_id = int(task["stack_id"])
        if stack_id not in candidate_stacks_by_order[order_id]:
            candidate_stacks_by_order[order_id].append(stack_id)
        support_set = set(int(v) for v in support_totes_by_order[order_id])
        demand_set = set(int(v) for v in demand_hit_totes_by_order[order_id])
        tote_set = set(int(v) for v in tote_ids_by_order[order_id])
        stack = getattr(problem, "point_to_stack", {}).get(stack_id)
        for tote in getattr(stack, "totes", []) or []:
            tote_id = int(getattr(tote, "id", -1))
            if tote_id >= 0:
                support_set.add(tote_id)
                tote_set.add(tote_id)
        for tote_id in list(task.get("hit_totes", []) or []):
            demand_set.add(int(tote_id))
            support_set.add(int(tote_id))
            tote_set.add(int(tote_id))
        for tote_id in list(task.get("target_totes", []) or []):
            support_set.add(int(tote_id))
            tote_set.add(int(tote_id))
        for tote_id in list(task.get("noise_totes", []) or []):
            support_set.add(int(tote_id))
            tote_set.add(int(tote_id))
        support_totes_by_order[order_id] = sorted(support_set)
        demand_hit_totes_by_order[order_id] = sorted(demand_set)
        tote_ids_by_order[order_id] = sorted(tote_set)

    prepared["candidate_stacks_by_order"] = {
        int(order_id): sorted(dict.fromkeys(int(stack_id) for stack_id in stack_ids))
        for order_id, stack_ids in candidate_stacks_by_order.items()
    }
    prepared["support_totes_by_order"] = support_totes_by_order
    prepared["demand_hit_totes_by_order"] = demand_hit_totes_by_order
    prepared["tote_ids_by_order"] = tote_ids_by_order
    return prepared


def _make_output_dir(output_dir: Optional[str]) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT_DIR, "result", f"alns_global_xyzu_iis_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _add_fix(model: gp.Model, var: gp.Var, value: float, name: str) -> None:
    model.addConstr(var == float(value), name=name)


def _selected_sort_keys_for_slot(task_rows: Sequence[Dict[str, Any]]) -> Set[Tuple[int, int, int]]:
    result: Set[Tuple[int, int, int]] = set()
    for task in task_rows:
        sort_range = task.get("sort_range")
        if str(task.get("mode", "")).upper() != "SORT" or not sort_range or len(sort_range) != 2:
            continue
        result.add((int(task["stack_id"]), int(sort_range[0]), int(sort_range[1])))
    return result


def _add_alns_fix_constraints(
    model: gp.Model,
    payload: Dict[str, Any],
    prepared: Dict[str, Any],
    parsed: Dict[str, Any],
    phase: str,
    fix_cmax: bool = False,
) -> Dict[str, Any]:
    slots = list(prepared["slots"])
    work_units = list(prepared["work_units"])
    candidate_stacks_by_order = dict(prepared["candidate_stacks_by_order"])
    station_ids = [int(v) for v in list(payload["station_ids"])]
    max_rank = int(payload["max_rank"])
    robot_ids = [int(v) for v in list(payload["robot_ids"])]
    route_task_by_tuple = dict(payload.get("route_task_by_tuple", {}) or {})
    route_tasks = dict(payload.get("route_tasks", {}) or {})
    route_start_node = int(payload.get("route_start_node", 0))
    route_end_node = int(payload.get("route_end_node", 1))

    x = payload["x"]
    a = payload["a"]
    sku_use = payload["sku_use"]
    y = payload["y"]
    flip = payload["flip"]
    sort_var = payload["sort"]
    sort_index = list(payload["sort_index"])
    carry = payload["carry"]
    hit = payload["hit"]
    noise = payload["noise"]
    flip_hit = payload["flip_hit"]
    pair_activate = payload["pair_activate"]
    arrival = payload["arrival"]
    start = payload["start"]
    finish = payload["finish"]
    cmax = payload["cmax"]
    order_arrival_lb = payload.get("order_arrival_lb")
    order_arrival_ub = payload.get("order_arrival_ub")
    order_span_overrun = payload.get("order_span_overrun")
    order_deadline_overrun = payload.get("order_deadline_overrun")
    slot_robot = payload.get("slot_robot")
    route_visit = payload.get("route_visit")
    route_arc = payload.get("route_arc")
    route_time = payload.get("route_time")

    subtasks = dict(parsed["subtasks"])
    tasks = dict(parsed["tasks"])
    subtask_task_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for task_row in tasks.values():
        subtask_task_rows[int(task_row["subtask_id"])].append(task_row)

    fixed_counts: Dict[str, int] = defaultdict(int)
    missing_route_tuples: List[Dict[str, Any]] = []

    for slot in slots:
        sid = int(slot.slot_id)
        order_id = int(slot.order_id)
        active = sid in subtasks
        _add_fix(model, a[sid], 1.0 if active else 0.0, f"FixA_{phase}_{sid}")
        fixed_counts["a"] += 1

        assigned_skus = set(int(v) for v in list(subtasks.get(sid, {}).get("sku_list", []) or []))
        for unit in work_units:
            if int(unit.order_id) != order_id:
                continue
            take = 1.0 if active and int(unit.sku_id) in assigned_skus else 0.0
            _add_fix(model, x[str(unit.unit_id), sid], take, f"FixX_{phase}_{str(unit.unit_id).replace(':', '_')}_{sid}")
            fixed_counts["x"] += 1
        for sku_id in prepared.get("unique_skus_by_order", {}).get(order_id, []):
            take = 1.0 if active and int(sku_id) in assigned_skus else 0.0
            if (order_id, int(sku_id), sid) in sku_use:
                _add_fix(model, sku_use[order_id, int(sku_id), sid], take, f"FixSkuUse_{phase}_{order_id}_{int(sku_id)}_{sid}")
                fixed_counts["sku_use"] += 1

        chosen_station = int(subtasks.get(sid, {}).get("station_id", -1))
        chosen_rank = int(subtasks.get(sid, {}).get("rank", -1))
        for station_id in station_ids:
            for rank in range(max_rank):
                val = 1.0 if active and int(station_id) == chosen_station and int(rank) == chosen_rank else 0.0
                _add_fix(model, y[sid, int(station_id), int(rank)], val, f"FixY_{phase}_{sid}_{station_id}_{rank}")
                fixed_counts["y"] += 1

        task_rows = list(subtask_task_rows.get(sid, []) or [])
        selected_pairs = {(sid, int(row["stack_id"]), int(row["station_id"])) for row in task_rows}
        selected_flip_stacks = {int(row["stack_id"]) for row in task_rows if str(row.get("mode", "")).upper() == "FLIP"}
        selected_sort_keys = _selected_sort_keys_for_slot(task_rows)
        carry_set: Set[int] = set()
        hit_set: Set[int] = set()
        noise_set: Set[int] = set()
        flip_hit_set: Set[int] = set()
        for row in task_rows:
            if str(row.get("mode", "")).upper() == "FLIP":
                flip_hit_set.update(int(v) for v in list(row.get("hit_totes", []) or []))
                hit_set.update(int(v) for v in list(row.get("hit_totes", []) or []))
                carry_set.update(int(v) for v in list(row.get("target_totes", []) or []))
            else:
                carry_set.update(int(v) for v in list(row.get("target_totes", []) or []))
                hit_set.update(int(v) for v in list(row.get("hit_totes", []) or []))
                noise_set.update(int(v) for v in list(row.get("noise_totes", []) or []))

        for stack_id in candidate_stacks_by_order.get(order_id, []):
            flip_val = 1.0 if int(stack_id) in selected_flip_stacks else 0.0
            if (sid, int(stack_id)) in flip:
                _add_fix(model, flip[sid, int(stack_id)], flip_val, f"FixFlip_{phase}_{sid}_{int(stack_id)}")
                fixed_counts["flip"] += 1
            for station_id in station_ids:
                val = 1.0 if (sid, int(stack_id), int(station_id)) in selected_pairs else 0.0
                if (sid, int(stack_id), int(station_id)) in pair_activate:
                    _add_fix(model, pair_activate[sid, int(stack_id), int(station_id)], val, f"FixPair_{phase}_{sid}_{int(stack_id)}_{int(station_id)}")
                    fixed_counts["pair"] += 1
                if val > 0.5 and (sid, int(stack_id), int(station_id)) not in route_task_by_tuple:
                    missing_route_tuples.append(
                        {"slot_id": sid, "stack_id": int(stack_id), "station_id": int(station_id), "reason": "missing_route_task"}
                    )

        for key in sort_index:
            if int(key[0]) != sid:
                continue
            sort_val = 1.0 if (int(key[1]), int(key[2]), int(key[3])) in selected_sort_keys else 0.0
            _add_fix(model, sort_var[key], sort_val, f"FixSort_{phase}_{sid}_{int(key[1])}_{int(key[2])}_{int(key[3])}")
            fixed_counts["sort"] += 1

        for key in list(carry.keys()):
            if int(key[0]) == sid:
                _add_fix(model, carry[key], 1.0 if int(key[1]) in carry_set else 0.0, f"FixCarry_{phase}_{sid}_{int(key[1])}")
                fixed_counts["carry"] += 1
        for key in list(hit.keys()):
            if int(key[0]) == sid:
                _add_fix(model, hit[key], 1.0 if int(key[1]) in hit_set else 0.0, f"FixHit_{phase}_{sid}_{int(key[1])}")
                fixed_counts["hit"] += 1
        for key in list(noise.keys()):
            if int(key[0]) == sid:
                _add_fix(model, noise[key], 1.0 if int(key[1]) in noise_set else 0.0, f"FixNoise_{phase}_{sid}_{int(key[1])}")
                fixed_counts["noise"] += 1
        for key in list(flip_hit.keys()):
            if int(key[0]) == sid:
                _add_fix(model, flip_hit[key], 1.0 if int(key[1]) in flip_hit_set else 0.0, f"FixFlipHit_{phase}_{sid}_{int(key[1])}")
                fixed_counts["flip_hit"] += 1

        if slot_robot is not None:
            selected_robot = -1
            for row in task_rows:
                if int(row.get("robot_id", -1)) >= 0:
                    selected_robot = int(row["robot_id"])
                    break
            for robot_id in robot_ids:
                val = 1.0 if active and int(robot_id) == selected_robot else 0.0
                _add_fix(model, slot_robot[sid, int(robot_id)], val, f"FixSlotRobot_{phase}_{sid}_{int(robot_id)}")
                fixed_counts["slot_robot"] += 1

        if phase == "full":
            slot_finish = 0.0
            slot_start = 0.0
            slot_arrival = 0.0
            if task_rows:
                slot_finish = max(float(row.get("end_process_time", 0.0) or 0.0) for row in task_rows)
                slot_start = min(float(row.get("start_process_time", 0.0) or 0.0) for row in task_rows)
                slot_arrival = max(float(row.get("arrival_station", 0.0) or 0.0) for row in task_rows)
            _add_fix(model, arrival[sid], slot_arrival, f"FixArrival_{phase}_{sid}")
            _add_fix(model, start[sid], slot_start, f"FixStart_{phase}_{sid}")
            _add_fix(model, finish[sid], slot_finish, f"FixFinish_{phase}_{sid}")
            fixed_counts["arrival"] += 1
            fixed_counts["start"] += 1
            fixed_counts["finish"] += 1

    if route_visit is not None and route_arc is not None:
        pass

    if payload.get("pass_x") is not None:
        pass_x = payload["pass_x"]
        route_node_robot: Dict[int, int] = {}
        selected_route_rows: List[Dict[str, Any]] = []
        for row in tasks.values():
            sid = int(row["subtask_id"])
            station_id = int(row["station_id"])
            stack_id = int(row["stack_id"])
            route_key = int(route_task_by_tuple.get((sid, stack_id, station_id), -1))
            route_spec = route_tasks.get(route_key)
            robot_id = int(row.get("robot_id", -1))
            if route_spec is None or robot_id < 0:
                continue
            route_node_robot[int(route_spec.pickup_node)] = int(robot_id)
            route_node_robot[int(route_spec.delivery_node)] = int(robot_id)
            selected_route_rows.append(
                {
                    "task_id": int(row["task_id"]),
                    "route_key": int(route_key),
                    "robot_id": int(robot_id),
                    "trip_id": int(row.get("trip_id", 0) or 0),
                    "arrival_stack": float(row.get("arrival_stack", 0.0) or 0.0),
                    "arrival_station": float(row.get("arrival_station", 0.0) or 0.0),
                    "robot_service_time": float(row.get("robot_service_time", 0.0) or 0.0),
                    "load": max(1, int(len(list(row.get("target_totes", []) or row.get("hit_totes", []) or [])))),
                }
            )
        for robot_id in robot_ids:
            start_node = int(payload.get("route_start_nodes", {}).get(int(robot_id), -1))
            end_node = int(payload.get("route_end_nodes", {}).get(int(robot_id), -1))
            if start_node >= 0:
                route_node_robot[start_node] = int(robot_id)
            if end_node >= 0:
                route_node_robot[end_node] = int(robot_id)
        for key in list(pass_x.keys()):
            node_id, robot_id = int(key[0]), int(key[1])
            if node_id not in route_node_robot:
                continue
            _add_fix(model, pass_x[key], 1.0 if int(route_node_robot[node_id]) == robot_id else 0.0, f"FixPassX_{phase}_{node_id}_{robot_id}")
            fixed_counts["pass_x"] += 1

        if route_arc is not None:
            selected_arcs: Set[Tuple[int, int]] = set()
            route_tau: Dict[Tuple[int, int], float] = payload.get("route_tau", {})
            rows_by_robot_trip: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
            for row in selected_route_rows:
                rows_by_robot_trip[(int(row["robot_id"]), int(row["trip_id"]))].append(dict(row))
            for (robot_id, _trip_id), rows in rows_by_robot_trip.items():
                start_node = int(payload.get("route_start_nodes", {}).get(int(robot_id), -1))
                end_node = int(payload.get("route_end_nodes", {}).get(int(robot_id), -1))
                prev_node = start_node
                for row in sorted(rows, key=lambda item: (float(item.get("arrival_stack", 0.0) or 0.0), int(item.get("task_id", -1)))):
                    spec = route_tasks.get(int(row["route_key"]))
                    if spec is None:
                        continue
                    pickup_node = int(spec.pickup_node)
                    delivery_node = int(spec.delivery_node)
                    if prev_node >= 0:
                        selected_arcs.add((int(prev_node), int(pickup_node)))
                    selected_arcs.add((int(pickup_node), int(delivery_node)))
                    prev_node = delivery_node
                if prev_node >= 0 and end_node >= 0:
                    selected_arcs.add((int(prev_node), int(end_node)))
            for key in list(route_arc.keys()):
                arc = (int(key[0]), int(key[1]))
                _add_fix(model, route_arc[key], 1.0 if arc in selected_arcs else 0.0, f"FixRouteArc_{phase}_{arc[0]}_{arc[1]}")
                fixed_counts["route_arc"] += 1
            if route_time is not None:
                for row in selected_route_rows:
                    spec = route_tasks.get(int(row["route_key"]))
                    if spec is None:
                        continue
                    _add_fix(model, route_time[int(spec.pickup_node)], float(row.get("arrival_stack", 0.0) or 0.0), f"FixRouteTimePickup_{phase}_{int(spec.pickup_node)}")
                    _add_fix(model, route_time[int(spec.delivery_node)], float(row.get("arrival_station", 0.0) or 0.0), f"FixRouteTimeDelivery_{phase}_{int(spec.delivery_node)}")
                    fixed_counts["route_time"] += 2
                for robot_id in robot_ids:
                    start_node = int(payload.get("route_start_nodes", {}).get(int(robot_id), -1))
                    end_node = int(payload.get("route_end_nodes", {}).get(int(robot_id), -1))
                    if start_node >= 0:
                        _add_fix(model, route_time[start_node], 0.0, f"FixRouteTimeStart_{phase}_{start_node}")
                        fixed_counts["route_time"] += 1
                    incoming = [arc for arc in selected_arcs if int(arc[1]) == end_node]
                    if end_node >= 0 and incoming:
                        prev_node = int(incoming[0][0])
                        prev_time = None
                        for row in selected_route_rows:
                            spec = route_tasks.get(int(row["route_key"]))
                            if spec is not None and int(spec.delivery_node) == prev_node:
                                prev_time = float(row.get("arrival_station", 0.0) or 0.0)
                                break
                        if prev_time is not None:
                            _add_fix(model, route_time[end_node], float(prev_time) + float(route_tau.get((prev_node, end_node), 0.0) or 0.0), f"FixRouteTimeEnd_{phase}_{end_node}")
                            fixed_counts["route_time"] += 1

    alns_cmax = float(parsed["header"].get("global_makespan", parsed["header"].get("best_z", 0.0)) or 0.0)
    if phase == "full":
        if bool(fix_cmax):
            _add_fix(model, cmax, alns_cmax, f"FixCmax_{phase}")
            fixed_counts["cmax"] += 1
        for subtask_row in subtasks.values():
            subtask_row["task_rows"] = list(subtask_task_rows.get(int(subtask_row["subtask_id"]), []) or [])
        order_rows: Dict[int, Dict[str, float]] = {}
        for order in getattr(prepared.get("problem", None), "order_list", []) or []:
            order_id = int(getattr(order, "order_id", -1))
            task_rows = [task for task in tasks.values() if int(parsed["subtasks"][int(task["subtask_id"])]["order_id"]) == order_id]
            arrivals = [float(task.get("arrival_station", 0.0) or 0.0) for task in task_rows]
            finishes = [float(task.get("end_process_time", 0.0) or 0.0) for task in task_rows]
            est_sec = float(getattr(order, "est_sec", 0.0) or 0.0)
            lst_sec = float(getattr(order, "lst_sec", 0.0) or 0.0)
            span_limit_sec = float(getattr(order, "kitting_span_limit_sec", 0.0) or 0.0)
            arrival_lb_val = float(min(arrivals)) if arrivals else 0.0
            arrival_ub_val = float(max(arrivals)) if arrivals else 0.0
            completion_val = float(max(finishes)) if finishes else 0.0
            order_rows[order_id] = {
                "arrival_lb": arrival_lb_val,
                "arrival_ub": arrival_ub_val,
                "span_overrun": max(0.0, arrival_ub_val - arrival_lb_val - span_limit_sec),
                "deadline_overrun": max(0.0, arrival_ub_val - lst_sec),
            }
        del order_rows, order_arrival_lb, order_arrival_ub, order_span_overrun, order_deadline_overrun

    return {
        "fixed_counts": dict(fixed_counts),
        "missing_route_tuples": missing_route_tuples,
        "alns_cmax": float(alns_cmax),
        "fix_cmax": bool(fix_cmax and phase == "full"),
    }


def _collect_iis_summary(model: gp.Model) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for constr in model.getConstrs():
        if int(getattr(constr, "IISConstr", 0)) != 1:
            continue
        rows.append({"kind": "linear", "name": constr.ConstrName})
    for constr in model.getGenConstrs():
        if int(getattr(constr, "IISGenConstr", 0)) != 1:
            continue
        rows.append({"kind": "general", "name": constr.GenConstrName})
    prefix_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        name = str(row["name"])
        prefix = name.split("_", 1)[0] if "_" in name else name
        prefix_counts[prefix] += 1
    return {
        "rows": rows,
        "count": len(rows),
        "prefix_counts": dict(sorted(prefix_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _write_iis_file(model: gp.Model, out_dir: str, phase: str) -> str:
    preferred = os.path.join(out_dir, f"{phase}_iis.ilp")
    try:
        model.write(preferred)
        return preferred
    except gp.GurobiError:
        fallback = os.path.join(ROOT_DIR, f"{phase}_iis.ilp")
        try:
            model.write(fallback)
            return fallback
        except gp.GurobiError:
            return ""


def _run_phase(
    solver: GlobalXYZUSolver,
    prepared: Dict[str, Any],
    cfg: GlobalXYZUConfig,
    parsed: Dict[str, Any],
    phase: str,
    out_dir: str,
    output_flag: bool,
    fix_cmax: bool,
) -> Dict[str, Any]:
    model = gp.Model(f"alns_xyzu_{phase}")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = float(cfg.time_limit_sec)
    model.Params.MIPGap = float(cfg.mip_gap)
    payload = solver._build_model(model, prepared, cfg)
    fix_diag = _add_alns_fix_constraints(
        model=model,
        payload=payload,
        prepared=prepared,
        parsed=parsed,
        phase=phase,
        fix_cmax=bool(fix_cmax),
    )
    model.optimize()

    status_name = {
        int(GRB.OPTIMAL): "OPTIMAL",
        int(GRB.INFEASIBLE): "INFEASIBLE",
        int(GRB.TIME_LIMIT): "TIME_LIMIT",
        int(GRB.SUBOPTIMAL): "SUBOPTIMAL",
    }.get(int(model.Status), str(model.Status))

    result: Dict[str, Any] = {
        "phase": phase,
        "status_code": int(model.Status),
        "status": status_name,
        "sol_count": int(model.SolCount),
        "obj_val": float(model.ObjVal) if model.SolCount > 0 else None,
        "obj_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else None,
        "fixed_counts": fix_diag["fixed_counts"],
        "missing_route_tuples": fix_diag["missing_route_tuples"],
        "alns_cmax": float(fix_diag["alns_cmax"]),
        "fix_cmax": bool(fix_diag["fix_cmax"]),
        "model_cmax": float(payload["cmax"].X) if model.SolCount > 0 else None,
        "iis": None,
    }
    if result["model_cmax"] is not None:
        result["cmax_gap_vs_alns"] = float(result["model_cmax"]) - float(result["alns_cmax"])
    if int(model.Status) == int(GRB.INFEASIBLE):
        model.computeIIS()
        ilp_path = _write_iis_file(model=model, out_dir=out_dir, phase=phase)
        result["iis_ilp_path"] = ilp_path
        result["iis"] = _collect_iis_summary(model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an ALNS best solution into global_xyzu and run feasibility/IIS checks.")
    parser.add_argument("--scale", type=str, default="Gurobi-s1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alns-export-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--time-limit-sec", type=float, default=2000.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--bom-arrival-window-sec", type=float, default=60.0)
    parser.add_argument("--disable-order-time-windows", action="store_true")
    parser.add_argument("--kitting-span-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--deadline-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--fix-cmax", action="store_true")
    parser.add_argument("--gurobi-output", action="store_true")
    args = parser.parse_args()

    out_dir = _make_output_dir(args.output_dir or None)
    parsed = _parse_alns_export(args.alns_export_dir)

    problem = CreateOFSProblem.generate_problem_by_scale(args.scale, seed=args.seed)
    solver = GlobalXYZUSolver()
    cfg = GlobalXYZUConfig(
        time_limit_sec=float(args.time_limit_sec),
        mip_gap=float(args.mip_gap),
        integrate_u_route=True,
        warm_start_use_sp4=False,
        bom_arrival_window_sec=float(args.bom_arrival_window_sec),
        gurobi_output=bool(args.gurobi_output),
        enable_order_time_windows=not bool(args.disable_order_time_windows),
        kitting_span_penalty_weight=float(args.kitting_span_penalty_weight),
        deadline_penalty_weight=float(args.deadline_penalty_weight),
    )
    warm = solver._build_warm_start(problem, cfg)
    prepared = solver._prepare(problem, cfg, warm)
    prepared = _augment_prepared_with_alns_solution(prepared, parsed)

    phases = ["structure", "full"]
    phase_results = [
        _run_phase(
            solver=solver,
            prepared=prepared,
            cfg=cfg,
            parsed=parsed,
            phase=phase,
            out_dir=out_dir,
            output_flag=bool(args.gurobi_output),
            fix_cmax=bool(args.fix_cmax),
        )
        for phase in phases
    ]

    report = {
        "scale": str(args.scale),
        "seed": int(args.seed),
        "alns_export_dir": os.path.abspath(args.alns_export_dir),
        "output_dir": os.path.abspath(out_dir),
        "alns_header": parsed["header"],
        "phase_results": phase_results,
    }
    json_path = os.path.join(out_dir, "alns_global_xyzu_iis_report.json")
    txt_path = os.path.join(out_dir, "alns_global_xyzu_iis_report.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_normalize_jsonable(report), f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[ALNS -> Global XYZU Feasibility / IIS]\n")
        f.write(f"scale={args.scale}\n")
        f.write(f"seed={int(args.seed)}\n")
        f.write(f"alns_export_dir={os.path.abspath(args.alns_export_dir)}\n")
        f.write(f"output_dir={os.path.abspath(out_dir)}\n")
        f.write(f"alns_best_z={float(parsed['header'].get('best_z', 0.0) or 0.0):.6f}\n")
        f.write(f"alns_global_makespan={float(parsed['header'].get('global_makespan', 0.0) or 0.0):.6f}\n")
        for phase_result in phase_results:
            f.write("\n")
            f.write(f"[phase={phase_result['phase']}]\n")
            f.write(f"status={phase_result['status']}\n")
            f.write(f"sol_count={int(phase_result['sol_count'])}\n")
            if phase_result.get("obj_val") is not None:
                f.write(f"obj_val={float(phase_result['obj_val']):.6f}\n")
            if phase_result.get("obj_bound") is not None:
                f.write(f"obj_bound={float(phase_result['obj_bound']):.6f}\n")
            f.write(f"fix_cmax={bool(phase_result.get('fix_cmax', False))}\n")
            f.write(f"alns_cmax={float(phase_result.get('alns_cmax', 0.0) or 0.0):.6f}\n")
            if phase_result.get("model_cmax") is not None:
                f.write(f"model_cmax={float(phase_result['model_cmax']):.6f}\n")
            if phase_result.get("cmax_gap_vs_alns") is not None:
                f.write(f"cmax_gap_vs_alns={float(phase_result['cmax_gap_vs_alns']):.6f}\n")
            f.write(f"fixed_counts={phase_result['fixed_counts']}\n")
            f.write(f"missing_route_tuples={phase_result['missing_route_tuples']}\n")
            if phase_result.get("iis"):
                f.write(f"iis_count={int(phase_result['iis']['count'])}\n")
                f.write(f"iis_prefix_counts={phase_result['iis']['prefix_counts']}\n")
                for row in list(phase_result["iis"]["rows"][:100]):
                    f.write(f"iis::{row['kind']}::{row['name']}\n")
                if phase_result.get("iis_ilp_path"):
                    f.write(f"iis_ilp_path={phase_result['iis_ilp_path']}\n")

    print(f"report_json={json_path}")
    print(f"report_txt={txt_path}")
    for phase_result in phase_results:
        print(f"{phase_result['phase']}_status={phase_result['status']}")
        if phase_result.get("obj_val") is not None:
            print(f"{phase_result['phase']}_obj={float(phase_result['obj_val']):.6f}")
        if phase_result.get("model_cmax") is not None:
            print(f"{phase_result['phase']}_model_cmax={float(phase_result['model_cmax']):.6f}")
        if phase_result.get("cmax_gap_vs_alns") is not None:
            print(f"{phase_result['phase']}_cmax_gap_vs_alns={float(phase_result['cmax_gap_vs_alns']):.6f}")
        if phase_result.get("iis"):
            print(f"{phase_result['phase']}_iis_count={int(phase_result['iis']['count'])}")


if __name__ == "__main__":
    main()
