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
            _add_fix(model, x[str(unit.unit_id), sid], take, f"FixX_{phase}_{sid}_{unit.sku_id}")
            fixed_counts["x"] += 1
            if (order_id, int(unit.sku_id), sid) in sku_use:
                _add_fix(model, sku_use[order_id, int(unit.sku_id), sid], take, f"FixSkuUse_{phase}_{sid}_{unit.sku_id}")
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
            _add_fix(model, flip[sid, int(stack_id)], 1.0 if int(stack_id) in selected_flip_stacks else 0.0, f"FixFlip_{phase}_{sid}_{stack_id}")
            fixed_counts["flip"] += 1
            for station_id in station_ids:
                val = 1.0 if (sid, int(stack_id), int(station_id)) in selected_pairs else 0.0
                _add_fix(model, pair_activate[sid, int(stack_id), int(station_id)], val, f"FixPair_{phase}_{sid}_{stack_id}_{station_id}")
                fixed_counts["pair_activate"] += 1
                if val > 0.5 and (sid, int(stack_id), int(station_id)) not in route_task_by_tuple:
                    missing_route_tuples.append(
                        {"slot_id": sid, "stack_id": int(stack_id), "station_id": int(station_id), "reason": "missing_route_task"}
                    )

        for key in sort_index:
            if int(key[0]) != sid:
                continue
            stack_id = int(key[1])
            low = int(key[2])
            high = int(key[3])
            val = 1.0 if (stack_id, low, high) in selected_sort_keys else 0.0
            _add_fix(model, sort_var[key], val, f"FixSort_{phase}_{sid}_{stack_id}_{low}_{high}")
            fixed_counts["sort"] += 1

        for slot_id, tote_id in list(carry.keys()):
            if int(slot_id) != sid:
                continue
            _add_fix(model, carry[int(slot_id), int(tote_id)], 1.0 if int(tote_id) in carry_set else 0.0, f"FixCarry_{phase}_{slot_id}_{tote_id}")
            fixed_counts["carry"] += 1
        for slot_id, tote_id in list(hit.keys()):
            if int(slot_id) != sid:
                continue
            _add_fix(model, hit[int(slot_id), int(tote_id)], 1.0 if int(tote_id) in hit_set else 0.0, f"FixHit_{phase}_{slot_id}_{tote_id}")
            fixed_counts["hit"] += 1
        for slot_id, tote_id in list(noise.keys()):
            if int(slot_id) != sid:
                continue
            _add_fix(model, noise[int(slot_id), int(tote_id)], 1.0 if int(tote_id) in noise_set else 0.0, f"FixNoise_{phase}_{slot_id}_{tote_id}")
            fixed_counts["noise"] += 1
        for slot_id, tote_id in list(flip_hit.keys()):
            if int(slot_id) != sid:
                continue
            _add_fix(model, flip_hit[int(slot_id), int(tote_id)], 1.0 if int(tote_id) in flip_hit_set else 0.0, f"FixFlipHit_{phase}_{slot_id}_{tote_id}")
            fixed_counts["flip_hit"] += 1

        if slot_robot is not None:
            chosen_robot = int(subtasks.get(sid, {}).get("assigned_robot_id", parsed["subtask_robot"].get(sid, -1)))
            for robot_id in robot_ids:
                val = 1.0 if active and int(robot_id) == chosen_robot else 0.0
                _add_fix(model, slot_robot[sid, int(robot_id)], val, f"FixSlotRobot_{phase}_{sid}_{robot_id}")
                fixed_counts["slot_robot"] += 1

        if phase == "full":
            slot_finish = 0.0
            slot_start = 0.0
            slot_arrival = 0.0
            if task_rows:
                slot_finish = max(float(row.get("end_process_time", 0.0) or 0.0) for row in task_rows)
                slot_start = min(float(row.get("start_process_time", 0.0) or 0.0) for row in task_rows)
                slot_arrival = min(float(row.get("arrival_station", 0.0) or 0.0) for row in task_rows)
            _add_fix(model, arrival[sid], slot_arrival if active else 0.0, f"FixArrival_{phase}_{sid}")
            _add_fix(model, start[sid], slot_start if active else 0.0, f"FixStart_{phase}_{sid}")
            _add_fix(model, finish[sid], slot_finish if active else 0.0, f"FixFinish_{phase}_{sid}")
            fixed_counts["slot_times"] += 3

    if route_visit is not None and route_arc is not None:
        selected_route_tuples: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for task_row in tasks.values():
            selected_route_tuples[(int(task_row["subtask_id"]), int(task_row["stack_id"]), int(task_row["station_id"]))] = task_row

        selected_task_keys: Set[int] = set()
        task_key_by_alns_task_id: Dict[int, int] = {}
        for key_tuple, task_row in selected_route_tuples.items():
            task_key = route_task_by_tuple.get(key_tuple)
            if task_key is None:
                continue
            selected_task_keys.add(int(task_key))
            task_key_by_alns_task_id[int(task_row["task_id"])] = int(task_key)

        for task_key, spec in route_tasks.items():
            chosen_robot = -1
            if int(task_key) in selected_task_keys:
                row = selected_route_tuples[(int(spec.slot_id), int(spec.stack_id), int(spec.station_id))]
                chosen_robot = int(row["robot_id"])
            for robot_id in robot_ids:
                val = 1.0 if int(robot_id) == chosen_robot else 0.0
                _add_fix(model, route_visit[int(spec.pickup_node), int(robot_id)], val, f"FixVisitP_{phase}_{task_key}_{robot_id}")
                _add_fix(model, route_visit[int(spec.delivery_node), int(robot_id)], val, f"FixVisitD_{phase}_{task_key}_{robot_id}")
                fixed_counts["route_visit"] += 2
                if phase == "full" and val > 0.5 and route_time is not None:
                    row = selected_route_tuples[(int(spec.slot_id), int(spec.stack_id), int(spec.station_id))]
                    _add_fix(model, route_time[int(spec.pickup_node), int(robot_id)], float(row.get("arrival_stack", 0.0) or 0.0), f"FixRouteTimeP_{phase}_{task_key}_{robot_id}")
                    _add_fix(model, route_time[int(spec.delivery_node), int(robot_id)], float(row.get("arrival_station", 0.0) or 0.0), f"FixRouteTimeD_{phase}_{task_key}_{robot_id}")
                    fixed_counts["route_time"] += 2

        selected_arcs: Set[Tuple[int, int, int]] = set()
        for robot_id in robot_ids:
            trips = list(parsed["trips_by_robot"].get(int(robot_id), []) or [])
            flattened: List[int] = []
            for trip in trips:
                flattened.extend(int(task_id) for task_id in trip)
            if not flattened:
                selected_arcs.add((route_start_node, route_end_node, int(robot_id)))
                continue
            first_key = task_key_by_alns_task_id.get(int(flattened[0]))
            if first_key is not None:
                selected_arcs.add((route_start_node, int(route_tasks[first_key].pickup_node), int(robot_id)))
            prev_key: Optional[int] = None
            for alns_task_id in flattened:
                task_key = task_key_by_alns_task_id.get(int(alns_task_id))
                if task_key is None:
                    continue
                spec = route_tasks[int(task_key)]
                selected_arcs.add((int(spec.pickup_node), int(spec.delivery_node), int(robot_id)))
                if prev_key is not None:
                    prev_spec = route_tasks[int(prev_key)]
                    selected_arcs.add((int(prev_spec.delivery_node), int(spec.pickup_node), int(robot_id)))
                prev_key = int(task_key)
            if prev_key is not None:
                prev_spec = route_tasks[int(prev_key)]
                selected_arcs.add((int(prev_spec.delivery_node), route_end_node, int(robot_id)))

        for i, j, robot_id in list(route_arc.keys()):
            val = 1.0 if (int(i), int(j), int(robot_id)) in selected_arcs else 0.0
            _add_fix(model, route_arc[int(i), int(j), int(robot_id)], val, f"FixArc_{phase}_{i}_{j}_{robot_id}")
            fixed_counts["route_arc"] += 1
        if phase == "full" and route_time is not None:
            for robot_id in robot_ids:
                _add_fix(model, route_time[route_start_node, int(robot_id)], 0.0, f"FixRouteTimeStart_{phase}_{robot_id}")
                fixed_counts["route_time"] += 1

    if phase == "full":
        model_cmax = float(parsed["header"].get("global_makespan", parsed["header"].get("best_z", 0.0)) or 0.0)
        _add_fix(model, cmax, model_cmax, f"FixCmax_{phase}")
        fixed_counts["cmax"] += 1

    return {
        "fixed_counts": dict(fixed_counts),
        "missing_route_tuples": missing_route_tuples,
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
) -> Dict[str, Any]:
    model = gp.Model(f"alns_xyzu_{phase}")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = float(cfg.time_limit_sec)
    model.Params.MIPGap = float(cfg.mip_gap)
    payload = solver._build_model(model, prepared, cfg)
    fix_diag = _add_alns_fix_constraints(model=model, payload=payload, prepared=prepared, parsed=parsed, phase=phase)
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
        "iis": None,
    }
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
        if phase_result.get("iis"):
            print(f"{phase_result['phase']}_iis_count={int(phase_result['iis']['count'])}")


if __name__ == "__main__":
    main()
