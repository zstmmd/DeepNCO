from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config.ofs_config import OFSConfig
from entity.subTask import SubTask
from entity.task import Task

from .state import ResourceConfig, ResourceSubtask, ZTaskDescriptor
from .utils import global_used_totes, pick_ranked_candidate, pick_soft_greedy_min


def _z_grid_cell(x: float, y: float, grid_size: float) -> Tuple[int, int]:
    size = max(1.0, float(grid_size))
    return int(math.floor(float(x) / size)), int(math.floor(float(y) / size))


def _z_manhattan_cells(x0: float, y0: float, x1: float, y1: float, grid_size: float) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    size = max(1.0, float(grid_size))
    x_step = 1 if x1 >= x0 else -1
    y_step = 1 if y1 >= y0 else -1
    x = float(x0)
    while (x_step > 0 and x <= float(x1)) or (x_step < 0 and x >= float(x1)):
        cell = _z_grid_cell(x, y0, size)
        if cell not in cells:
            cells.append(cell)
        x += float(x_step) * size
    y = float(y0)
    while (y_step > 0 and y <= float(y1)) or (y_step < 0 and y >= float(y1)):
        cell = _z_grid_cell(x1, y, size)
        if cell not in cells:
            cells.append(cell)
        y += float(y_step) * size
    end_cell = _z_grid_cell(x1, y1, size)
    if end_cell not in cells:
        cells.append(end_cell)
    return cells


def _z_station_point(opt, station_id: int):
    stations = list(getattr(getattr(opt, "problem", None), "station_list", []) or [])
    if not (0 <= int(station_id) < len(stations)):
        return None
    return stations[int(station_id)].point


def _z_stack_point(opt, stack_id: int):
    return getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id), None)


def _z_route_cells(opt, stack_id: int, station_id: int, grid_size: float) -> List[Tuple[int, int]]:
    stack = _z_stack_point(opt, int(stack_id))
    station_pt = _z_station_point(opt, int(station_id))
    if stack is None or station_pt is None:
        return []
    stack_pt = stack.store_point
    return _z_manhattan_cells(float(stack_pt.x), float(stack_pt.y), float(station_pt.x), float(station_pt.y), float(grid_size))


def _transient_conflict_constraints(config: ResourceConfig) -> Dict[str, object]:
    return dict((getattr(config, "metadata", {}) or {}).get("transient_conflict_constraints", {}) or {})


def _snapshot_subtask_arrivals(opt) -> Dict[int, Dict[str, float]]:
    snapshot = getattr(getattr(opt, "best_validated", None), "snapshot", None)
    if snapshot is None:
        snapshot = getattr(opt, "work", None)
    rows: Dict[int, Dict[str, float]] = {}
    for subtask in list(getattr(snapshot, "subtask_state", []) or []):
        subtask_id = int(getattr(subtask, "id", -1))
        task_rows = list(getattr(subtask, "execution_tasks", []) or [])
        if subtask_id < 0 or not task_rows:
            continue
        rows[int(subtask_id)] = {
            "arrival_station": min(float(getattr(task, "arrival_time_at_station", 0.0) or 0.0) for task in task_rows),
            "station_id": int(getattr(subtask, "assigned_station_id", -1)),
        }
    return rows


def _robot_start_positions(opt) -> List[Tuple[int, Tuple[float, float]]]:
    rows: List[Tuple[int, Tuple[float, float]]] = []
    for robot in getattr(getattr(opt, "problem", None), "robot_list", []) or []:
        point = getattr(robot, "start_point", None)
        robot_id = int(getattr(robot, "id", -1))
        if point is None or robot_id < 0:
            continue
        rows.append((int(robot_id), (float(point.x), float(point.y))))
    return rows


def _manhattan_distance_xy(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    return float(abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1])))


def _subtask_candidate_stack_points(opt, config: ResourceConfig, subtask: ResourceSubtask) -> List[Tuple[int, Tuple[float, float]]]:
    stack_ids: List[int] = []
    for descriptor in subtask.z_tasks or []:
        stack_id = int(getattr(descriptor, "stack_id", -1))
        if stack_id >= 0 and int(stack_id) not in stack_ids:
            stack_ids.append(int(stack_id))
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        for stack_id in getattr(opt, "_x_candidate_stack_ids_for_sku", lambda *_args, **_kwargs: [])(int(work_unit.sku_id)) or []:
            if int(stack_id) >= 0 and int(stack_id) not in stack_ids:
                stack_ids.append(int(stack_id))
    points: List[Tuple[int, Tuple[float, float]]] = []
    for stack_id in stack_ids:
        stack = _z_stack_point(opt, int(stack_id))
        if stack is None:
            continue
        points.append((int(stack_id), (float(stack.store_point.x), float(stack.store_point.y))))
    return points


def _station_service_proxy(descriptors: Sequence[ZTaskDescriptor]) -> float:
    total = 0.0
    for descriptor in descriptors or ():
        total += float(getattr(descriptor, "station_service_time", 0.0) or 0.0)
        total += float(max(1, int(getattr(descriptor, "sku_pick_count", 0) or 0))) * float(getattr(OFSConfig, "PICKING_TIME", 1.0))
    return float(max(total, 1.0))


def _z_station_load(config: ResourceConfig, station_id: int) -> float:
    return float(
        sum(
            _station_service_proxy(row.z_tasks or [])
            for row in config.subtasks.values()
            if int(row.station_id) == int(station_id)
        )
    )


def _z_robot_region_load(opt, station_id: int) -> float:
    snapshot = getattr(getattr(opt, "best_validated", None), "snapshot", None)
    if snapshot is None:
        snapshot = getattr(opt, "work", None)
    rows = list(getattr(snapshot, "subtask_state", []) or [])
    load = 0.0
    for subtask in rows:
        if int(getattr(subtask, "assigned_station_id", -1)) != int(station_id):
            continue
        robot_ids = {
            int(getattr(task, "robot_id", -1))
            for task in (getattr(subtask, "execution_tasks", []) or [])
            if int(getattr(task, "robot_id", -1)) >= 0
        }
        load += float(max(1, len(robot_ids)))
    return float(load)


def _estimate_arrival(opt, config: ResourceConfig, subtask: ResourceSubtask, snapshot_arrivals: Optional[Dict[int, Dict[str, float]]] = None) -> float:
    arrivals = dict(snapshot_arrivals or {})
    snapshot_row = dict(arrivals.get(int(subtask.subtask_id), {}) or {})
    if snapshot_row and int(snapshot_row.get("station_id", -1)) == int(subtask.station_id):
        return float(snapshot_row.get("arrival_station", 0.0))
    station_pt = _z_station_point(opt, int(subtask.station_id))
    stack_points = _subtask_candidate_stack_points(opt, config, subtask)
    if station_pt is None or not stack_points:
        return 0.0
    station_xy = (float(station_pt.x), float(station_pt.y))
    best = float("inf")
    for _robot_id, robot_xy in _robot_start_positions(opt):
        for _stack_id, stack_xy in stack_points:
            cost = _manhattan_distance_xy(robot_xy, stack_xy) + _manhattan_distance_xy(stack_xy, station_xy)
            if float(cost) < float(best):
                best = float(cost)
    return 0.0 if best == float("inf") else float(best)


def _predict_station_queues(
    opt,
    config: ResourceConfig,
    service_overrides: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    station_timelines: Dict[int, float] = defaultdict(float)
    expected_wait_times: Dict[int, float] = {}
    predicted_arrival_times: Dict[int, float] = {}
    snapshot_arrivals = _snapshot_subtask_arrivals(opt)
    rows = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) >= 0],
        key=lambda row: (int(row.station_id), int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)),
    )
    for row in rows:
        subtask_id = int(row.subtask_id)
        arrival = float(_estimate_arrival(opt, config, row, snapshot_arrivals=snapshot_arrivals))
        service = float(dict(service_overrides or {}).get(int(subtask_id), _station_service_proxy(row.z_tasks or [])))
        start = max(float(arrival), float(station_timelines.get(int(row.station_id), 0.0)))
        expected_wait_times[int(subtask_id)] = float(max(0.0, start - arrival))
        predicted_arrival_times[int(subtask_id)] = float(arrival)
        station_timelines[int(row.station_id)] = float(start + service)
    return {
        "station_timelines": dict(station_timelines),
        "expected_wait_times": dict(expected_wait_times),
        "predicted_arrival_times": dict(predicted_arrival_times),
    }


def _subtask_station_load_before(config: ResourceConfig, subtask: ResourceSubtask) -> float:
    station_rows = [
        row
        for row in config.subtasks.values()
        if int(row.station_id) == int(subtask.station_id)
        and int(row.subtask_id) != int(subtask.subtask_id)
        and int(row.station_rank) >= 0
        and int(row.station_rank) < int(subtask.station_rank)
    ]
    total = 0.0
    for row in station_rows:
        total += sum(float(getattr(task, "station_service_time", 0.0) or 0.0) for task in (row.z_tasks or []))
    return float(total)


def _subtask_station_window(config: ResourceConfig, subtask: ResourceSubtask, duration: float, bucket_sec: float) -> Tuple[float, float, List[int]]:
    load_before = _subtask_station_load_before(config, subtask)
    start = max(0.0, float(load_before))
    end = float(start + max(float(duration), 1.0))
    width = max(1.0, float(bucket_sec))
    lo = int(math.floor(float(start) / width))
    hi = int(math.floor(max(float(start), float(end) - 1e-9) / width))
    return float(start), float(end), list(range(int(lo), int(hi) + 1))


def _descriptor_duration_proxy(opt, descriptor: ZTaskDescriptor) -> float:
    detour = float(opt._z_best_insertion_detour(int(descriptor.stack_id)))
    travel = float(detour / max(1.0, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0))))
    mode_switch = float(getattr(opt.cfg, "resource_z_mode_switch_penalty", 6.0))
    if str(descriptor.mode).upper() == "FLIP":
        mode_switch *= 0.5
    sort_span = 0.0
    if descriptor.sort_layer_range is not None:
        sort_span = float(max(0, int(descriptor.sort_layer_range[1]) - int(descriptor.sort_layer_range[0]) + 1))
    tote_count = float(len(list(descriptor.target_tote_ids or ())))
    return float(travel + float(descriptor.robot_service_time) + float(descriptor.station_service_time) + mode_switch + 0.5 * tote_count + sort_span)


def _estimate_assignment_duration(opt, descriptors: Sequence[ZTaskDescriptor]) -> float:
    return float(sum(_descriptor_duration_proxy(opt, descriptor) for descriptor in (descriptors or ())))


def _rough_route_feasibility(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    descriptors: Sequence[ZTaskDescriptor],
) -> Tuple[bool, float, Dict[str, object]]:
    cfg = getattr(opt, "cfg", None)
    bucket_sec = max(1.0, float(getattr(cfg, "resource_z_contention_time_bucket_sec", 20.0)))
    grid_size = max(1.0, float(getattr(cfg, "resource_y_heat_grid_size", 4.0)))
    duration = _estimate_assignment_duration(opt, descriptors)
    cap = max(1.0, float(getattr(cfg, "resource_z_sequence_feasibility_cap", 180.0)))
    start, end, bucket_ids = _subtask_station_window(config, subtask, duration, bucket_sec)
    stack_budget = max(1, int(getattr(cfg, "resource_stack_concurrency_limit", 2)))
    tote_budget = max(1, int(getattr(cfg, "resource_tote_concurrency_limit", 1)))
    choke_budget = max(1, int(getattr(cfg, "resource_choke_point_budget", 2)))
    stack_counts: Dict[int, int] = defaultdict(int)
    tote_counts: Dict[int, int] = defaultdict(int)
    choke_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    station_buckets = set(int(x) for x in bucket_ids)
    descriptor_stacks = {int(item.stack_id) for item in (descriptors or ()) if int(item.stack_id) >= 0}
    descriptor_totes = {int(tid) for item in (descriptors or ()) for tid in (item.target_tote_ids or ()) if int(tid) >= 0}
    descriptor_cells = {
        cell
        for item in (descriptors or ())
        for cell in _z_route_cells(opt, int(item.stack_id), int(subtask.station_id), grid_size)
    }
    for row in config.subtasks.values():
        if int(row.subtask_id) == int(subtask.subtask_id) or int(row.station_id) < 0:
            continue
        other_duration = _estimate_assignment_duration(opt, row.z_tasks or [])
        _, _, other_buckets = _subtask_station_window(config, row, other_duration, bucket_sec)
        if not station_buckets.intersection(set(int(x) for x in other_buckets)):
            continue
        for descriptor in row.z_tasks or []:
            stack_counts[int(descriptor.stack_id)] += 1
            for tote_id in (descriptor.target_tote_ids or ()):
                tote_counts[int(tote_id)] += 1
            for cell in _z_route_cells(opt, int(descriptor.stack_id), int(row.station_id), grid_size):
                choke_counts[cell] += 1
    stack_over = sum(max(0, int(stack_counts.get(stack_id, 0)) + 1 - stack_budget) for stack_id in descriptor_stacks)
    tote_over = sum(max(0, int(tote_counts.get(tote_id, 0)) + 1 - tote_budget) for tote_id in descriptor_totes)
    choke_over = sum(max(0, int(choke_counts.get(cell, 0)) + 1 - choke_budget) for cell in descriptor_cells)
    penalty = 0.0
    penalty += float(getattr(cfg, "resource_z_stack_contention_penalty", 6.0)) * float(stack_over)
    penalty += float(getattr(cfg, "resource_z_tote_contention_penalty", 10.0)) * float(tote_over)
    conflict = _transient_conflict_constraints(config)
    conflict_stacks = {int(x) for x in (conflict.get("stack_ids", []) or [])}
    conflict_totes = {int(x) for x in (conflict.get("target_tote_ids", []) or [])}
    conflict_buckets = {int(x) for x in (conflict.get("time_bucket_ids", []) or [])}
    if descriptor_stacks & conflict_stacks:
        penalty += float(getattr(cfg, "resource_z_conflict_stack_penalty", 12.0))
    if descriptor_totes & conflict_totes:
        penalty += float(getattr(cfg, "resource_z_conflict_tote_penalty", 14.0))
    if station_buckets & conflict_buckets:
        penalty += float(getattr(cfg, "resource_z_conflict_time_bucket_penalty", 8.0))
    over_cap = max(0.0, float(duration) - cap)
    if over_cap > 0.0:
        penalty += float(getattr(cfg, "resource_z_route_feasibility_penalty", 20.0)) * float(over_cap / max(1.0, cap))
    ok = bool(over_cap <= 1e-9 and stack_over <= 0 and tote_over <= 0)
    return ok, float(penalty), {
        "duration": float(duration),
        "window_start": float(start),
        "window_end": float(end),
        "time_bucket_ids": list(bucket_ids),
        "stack_ids": sorted(descriptor_stacks),
        "target_tote_ids": sorted(descriptor_totes),
        "route_cells": [list(cell) for cell in sorted(descriptor_cells)],
        "stack_over": int(stack_over),
        "tote_over": int(tote_over),
        "choke_over": int(choke_over),
    }


def _demand_counts(config: ResourceConfig, subtask: ResourceSubtask) -> Dict[int, int]:
    demand: Dict[int, int] = defaultdict(int)
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        demand[int(work_unit.sku_id)] += 1
    return dict(demand)


def _descriptor_to_task(subtask: ResourceSubtask, descriptor: ZTaskDescriptor) -> Task:
    return Task(
        task_id=int(descriptor.task_id),
        sub_task_id=int(subtask.subtask_id),
        target_stack_id=int(descriptor.stack_id),
        target_station_id=int(subtask.station_id),
        operation_mode=str(descriptor.mode).upper(),
        station_sequence_rank=int(subtask.station_rank),
        target_tote_ids=list(int(x) for x in (descriptor.target_tote_ids or ())),
        hit_tote_ids=list(int(x) for x in (descriptor.hit_tote_ids or ())),
        noise_tote_ids=list(int(x) for x in (descriptor.noise_tote_ids or ())),
        sort_layer_range=None if descriptor.sort_layer_range is None else (int(descriptor.sort_layer_range[0]), int(descriptor.sort_layer_range[1])),
        station_service_time=float(descriptor.station_service_time),
        robot_service_time=float(descriptor.robot_service_time),
        sku_pick_count=int(descriptor.sku_pick_count),
    )


def _build_temp_subtask(opt, config: ResourceConfig, subtask: ResourceSubtask, descriptors: Sequence[ZTaskDescriptor]) -> SubTask:
    order_map = {int(getattr(order, "order_id", -1)): order for order in getattr(opt.problem, "order_list", []) or []}
    sku_map = {int(getattr(sku, "id", -1)): sku for sku in getattr(opt.problem, "skus_list", []) or []}
    order_obj = order_map.get(int(subtask.order_id))
    sku_list = [
        sku_map[int(config.work_units[str(work_unit_id)].sku_id)]
        for work_unit_id in (subtask.work_unit_ids or ())
        if str(work_unit_id) in config.work_units and int(config.work_units[str(work_unit_id)].sku_id) in sku_map
    ]
    temp_subtask = SubTask(id=int(subtask.subtask_id), parent_order=order_obj, sku_list=sku_list)
    temp_subtask.assigned_station_id = int(subtask.station_id)
    temp_subtask.station_sequence_rank = int(subtask.station_rank)
    for descriptor in descriptors:
        task = _descriptor_to_task(subtask, descriptor)
        stack = opt.problem.point_to_stack.get(int(task.target_stack_id))
        if stack is not None:
            temp_subtask.add_execution_detail(task, stack)
    return temp_subtask


def _coverage_gain(opt, remaining: Dict[int, int], hit_tote_ids: Sequence[int]) -> int:
    gain = 0
    local = dict(remaining)
    for tote_id in hit_tote_ids or ():
        tote = getattr(opt.problem, "id_to_tote", {}).get(int(tote_id))
        if tote is None:
            continue
        for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
            sku_id = int(sku_id)
            use = min(int(local.get(sku_id, 0)), int(qty))
            if use <= 0:
                continue
            local[sku_id] = int(local.get(sku_id, 0)) - int(use)
            gain += int(use)
    return int(gain)


def _consume_coverage(opt, remaining: Dict[int, int], hit_tote_ids: Sequence[int]) -> Dict[int, int]:
    updated = dict(remaining)
    for tote_id in hit_tote_ids or ():
        tote = getattr(opt.problem, "id_to_tote", {}).get(int(tote_id))
        if tote is None:
            continue
        for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
            sku_id = int(sku_id)
            use = min(int(updated.get(sku_id, 0)), int(qty))
            if use <= 0:
                continue
            updated[sku_id] = int(updated.get(sku_id, 0)) - int(use)
    return updated


def _candidate_centroid_xy(opt, config: ResourceConfig, subtask: ResourceSubtask) -> Optional[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    seen: Set[Tuple[float, float]] = set()
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        for stack_id in opt._x_candidate_stack_ids_for_sku(int(work_unit.sku_id)):
            xy = opt._stack_xy(int(stack_id))
            if xy is None or xy in seen:
                continue
            seen.add(xy)
            points.append(xy)
    if not points:
        return None
    return (
        float(sum(pt[0] for pt in points) / len(points)),
        float(sum(pt[1] for pt in points) / len(points)),
    )


def _candidate_stack_ids(opt, config: ResourceConfig, subtask: ResourceSubtask, seed_stack_ids: Optional[Sequence[int]] = None) -> List[int]:
    primary_stack_ids: List[int] = []
    extra_stack_ids: List[int] = []
    for stack_id in seed_stack_ids or ():
        if int(stack_id) >= 0 and int(stack_id) not in primary_stack_ids:
            primary_stack_ids.append(int(stack_id))
    for descriptor in subtask.z_tasks or []:
        if int(descriptor.stack_id) >= 0 and int(descriptor.stack_id) not in primary_stack_ids:
            primary_stack_ids.append(int(descriptor.stack_id))
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        for stack_id in opt._x_candidate_stack_ids_for_sku(int(work_unit.sku_id)):
            sid = int(stack_id)
            if sid < 0 or sid in primary_stack_ids or sid in extra_stack_ids:
                continue
            extra_stack_ids.append(sid)
    centroid = _candidate_centroid_xy(opt, config, subtask)
    if centroid is not None and extra_stack_ids:
        extra_stack_ids.sort(
            key=lambda sid: (
                float(opt._xy_manhattan(centroid, opt._stack_xy(int(sid)))) if opt._stack_xy(int(sid)) is not None else float("inf"),
                float(opt._z_best_insertion_detour(int(sid))),
                int(sid),
            )
        )
    topk = max(0, int(getattr(opt.cfg, "resource_z_candidate_stack_topk", 5)))
    if topk > 0:
        extra_stack_ids = extra_stack_ids[:topk]
    return list(primary_stack_ids) + list(extra_stack_ids)


def _estimate_wait_overflow(config: ResourceConfig, station_id: int) -> float:
    if int(station_id) < 0:
        return 1e9
    count = len([row for row in config.subtasks.values() if int(row.station_id) == int(station_id)])
    return float(count * getattr(OFSConfig, "PICKING_TIME", 1.0))


def _guard_reason(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    plan: Dict[str, object],
    queue_ctx: Optional[Dict[str, object]] = None,
) -> str:
    detour = float(opt._z_best_insertion_detour(int(plan.get("target_stack_id", -1))))
    arrival_shift = float(detour / max(1.0, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0))))
    wait_overflow = float(
        dict((queue_ctx or {}).get("expected_wait_times", {}) or {}).get(
            int(subtask.subtask_id),
            _estimate_wait_overflow(config, int(subtask.station_id)),
        )
    )
    target_count = len(list(plan.get("target_tote_ids", []) or []))
    tail_penalty = 0.5 * float(target_count)
    route_tail_delta = float(arrival_shift + tail_penalty)
    if arrival_shift > float(getattr(opt.cfg, "z_arrival_shift_soft_cap", 140.0)) + 1e-9:
        return "z_arrival_shift_soft_cap"
    if wait_overflow > float(getattr(opt.cfg, "z_wait_overflow_soft_cap", 180.0)) + 1e-9:
        return "z_wait_overflow_soft_cap"
    if route_tail_delta > float(getattr(opt.cfg, "z_route_tail_soft_cap", 90.0)) + 1e-9:
        return "z_route_tail_soft_cap"
    if detour > float(getattr(opt.cfg, "z_route_gap_soft_cap", 25.0)) + 1e-9:
        return "z_route_gap_soft_cap"
    return ""


def _descriptor_from_plan(opt, subtask: ResourceSubtask, plan: Dict[str, object], task_id: int, sku_pick_count: int) -> ZTaskDescriptor:
    plan = _canonicalize_z_plan(opt, plan)
    robot_service = max(float(plan.get("robot_service_time", 0.0) or 0.0), 0.5 * float(len(list(plan.get("target_tote_ids", []) or []))))
    return ZTaskDescriptor(
        task_id=int(task_id),
        stack_id=int(plan.get("target_stack_id", -1)),
        mode=str(plan.get("operation_mode", "FLIP")).upper(),
        target_tote_ids=tuple(int(x) for x in (plan.get("target_tote_ids", []) or [])),
        hit_tote_ids=tuple(int(x) for x in (plan.get("hit_tote_ids", []) or [])),
        noise_tote_ids=tuple(int(x) for x in (plan.get("noise_tote_ids", []) or [])),
        sort_layer_range=None if plan.get("sort_layer_range", None) is None else (
            int(plan.get("sort_layer_range", (0, 0))[0]),
            int(plan.get("sort_layer_range", (0, 0))[1]),
        ),
        station_service_time=float(plan.get("station_service_time", 0.0)),
        robot_service_time=float(robot_service),
        sku_pick_count=int(max(1, sku_pick_count)),
    )


def _is_joint_sort_strategy(strategy: str) -> bool:
    return str(strategy) == "z_repair_joint_sort_colocated_flip"


def _sort_plan_within_capacity(plan: Dict[str, object]) -> bool:
    if str(plan.get("operation_mode", "")).upper() != "SORT":
        return True
    capacity = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)))
    return len(list(plan.get("target_tote_ids", []) or [])) <= int(capacity)


def _dedupe_ints(values: Sequence[int]) -> List[int]:
    deduped: List[int] = []
    for value in values or ():
        item = int(value)
        if item >= 0 and item not in deduped:
            deduped.append(item)
    return deduped


def _stack_tote_ids(opt, stack_id: int) -> List[int]:
    stack = getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id))
    if stack is None:
        return []
    return [int(getattr(tote, "id", -1)) for tote in (getattr(stack, "totes", []) or []) if int(getattr(tote, "id", -1)) >= 0]


def _stack_interval_tote_ids(opt, stack_id: int, sort_layer_range: Optional[Tuple[int, int]]) -> Optional[List[int]]:
    if sort_layer_range is None:
        return None
    stack = getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id))
    if stack is None:
        return None
    totes = list(getattr(stack, "totes", []) or [])
    lo, hi = int(sort_layer_range[0]), int(sort_layer_range[1])
    if lo < 0 or hi < lo or hi >= len(totes):
        return None
    return [int(getattr(tote, "id", -1)) for tote in totes[lo:hi + 1] if int(getattr(tote, "id", -1)) >= 0]


def _tote_layer_map(opt, stack_id: int) -> Dict[int, int]:
    stack = getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id))
    if stack is None:
        return {}
    layer_by_tote: Dict[int, int] = {}
    for idx, tote in enumerate(getattr(stack, "totes", []) or []):
        tote_id = int(getattr(tote, "id", -1))
        if tote_id >= 0:
            layer_by_tote[int(tote_id)] = int(idx)
    return layer_by_tote


def _canonicalize_z_plan(opt, plan: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(plan or {})
    mode = str(normalized.get("operation_mode", "FLIP")).upper()
    stack_id = int(normalized.get("target_stack_id", -1))
    if mode != "SORT":
        hit_ids = _dedupe_ints(normalized.get("hit_tote_ids", []) or normalized.get("target_tote_ids", []) or [])
        normalized["operation_mode"] = "FLIP"
        normalized["target_tote_ids"] = list(hit_ids)
        normalized["hit_tote_ids"] = list(hit_ids)
        normalized["noise_tote_ids"] = []
        normalized["sort_layer_range"] = None
        normalized["station_service_time"] = 0.0
        return normalized

    hit_ids = _dedupe_ints(normalized.get("hit_tote_ids", []) or normalized.get("target_tote_ids", []) or [])
    target_seed_ids = _dedupe_ints(normalized.get("target_tote_ids", []) or []) + [tid for tid in hit_ids if tid not in set(normalized.get("target_tote_ids", []) or [])]
    layer_by_tote = _tote_layer_map(opt, stack_id)
    sort_range = normalized.get("sort_layer_range", None)
    lo = hi = None
    if sort_range is not None:
        lo, hi = int(sort_range[0]), int(sort_range[1])
    seed_layers = [int(layer_by_tote[int(tid)]) for tid in target_seed_ids if int(tid) in layer_by_tote]
    if seed_layers:
        lo = min(seed_layers) if lo is None else min(int(lo), min(seed_layers))
        hi = max(seed_layers) if hi is None else max(int(hi), max(seed_layers))
    if lo is not None and hi is not None:
        interval_ids = _stack_interval_tote_ids(opt, stack_id, (int(lo), int(hi)))
    else:
        interval_ids = None
    if interval_ids is not None:
        target_ids = list(interval_ids)
        target_set = set(target_ids)
        hit_ids = [int(tid) for tid in hit_ids if int(tid) in target_set]
        noise_ids = [int(tid) for tid in target_ids if int(tid) not in set(hit_ids)]
        normalized["sort_layer_range"] = (int(lo), int(hi))
        normalized["target_tote_ids"] = list(target_ids)
        normalized["hit_tote_ids"] = list(hit_ids)
        normalized["noise_tote_ids"] = list(noise_ids)
        normalized["station_service_time"] = float(len(noise_ids) * float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)))
    else:
        normalized["target_tote_ids"] = _dedupe_ints(normalized.get("target_tote_ids", []) or [])
        normalized["hit_tote_ids"] = list(hit_ids)
        normalized["noise_tote_ids"] = _dedupe_ints(normalized.get("noise_tote_ids", []) or [])
    normalized["operation_mode"] = "SORT"
    return normalized


def _normalize_joint_sort_plan(opt, plan: Dict[str, object], carry_tote_ids: Sequence[int]) -> Dict[str, object]:
    normalized = dict(plan or {})
    carried = _dedupe_ints(carry_tote_ids or ())
    normalized["operation_mode"] = "SORT"
    normalized["hit_tote_ids"] = list(carried)
    if not list(normalized.get("target_tote_ids", []) or []):
        normalized["target_tote_ids"] = list(carried)
    return _canonicalize_z_plan(opt, normalized)


def _tote_demand_overlap(opt, tote_id: int, remaining: Dict[int, int]) -> int:
    tote = getattr(opt.problem, "id_to_tote", {}).get(int(tote_id))
    if tote is None:
        return 0
    overlap = 0
    for sku_id, qty in (getattr(tote, "sku_quantity_map", {}) or {}).items():
        overlap += max(0, min(int(remaining.get(int(sku_id), 0)), int(qty)))
    return int(overlap)


def _enumerate_covering_hit_sets(
    opt,
    tote_ids: Sequence[int],
    remaining: Dict[int, int],
    capacity: int,
    max_sets: int,
) -> List[List[int]]:
    demand_total = sum(max(0, int(qty)) for qty in dict(remaining or {}).values())
    if demand_total <= 0:
        return [[]]
    ordered = sorted(
        [int(tote_id) for tote_id in (tote_ids or ()) if _tote_demand_overlap(opt, int(tote_id), remaining) > 0],
        key=lambda tote_id: (-_tote_demand_overlap(opt, int(tote_id), remaining), int(tote_id)),
    )
    results: List[List[int]] = []

    def dfs(idx: int, current: List[int], current_remaining: Dict[int, int]) -> None:
        if len(results) >= int(max_sets):
            return
        if all(int(qty) <= 0 for qty in current_remaining.values()):
            results.append(list(current))
            return
        if idx >= len(ordered) or len(current) >= int(capacity):
            return
        tote_id = int(ordered[idx])
        next_remaining = _consume_coverage(opt, current_remaining, [int(tote_id)])
        if next_remaining != current_remaining:
            current.append(int(tote_id))
            dfs(idx + 1, current, next_remaining)
            current.pop()
        dfs(idx + 1, current, current_remaining)

    dfs(0, [], dict(remaining))
    return results


def _replace_descriptor_group(
    subtask: ResourceSubtask,
    group_indices: Sequence[int],
    descriptor: ZTaskDescriptor,
) -> List[ZTaskDescriptor]:
    index_set = {int(idx) for idx in (group_indices or [])}
    updated_assignment: List[ZTaskDescriptor] = []
    inserted = False
    for task_idx, existing in enumerate(subtask.z_tasks or []):
        if int(task_idx) in index_set:
            if not inserted:
                updated_assignment.append(descriptor)
                inserted = True
            continue
        updated_assignment.append(existing)
    if not inserted:
        updated_assignment.append(descriptor)
    return updated_assignment


def _build_joint_sort_candidate_options(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    group_indices: Sequence[int],
    stack_id: int,
    shared_candidate_totes: Sequence[int],
    external_used_totes: Set[int],
) -> List[ZTaskDescriptor]:
    index_set = {int(idx) for idx in (group_indices or [])}
    group = [descriptor for idx, descriptor in enumerate(subtask.z_tasks or []) if int(idx) in index_set]
    if not group:
        return []
    preserved_before = [descriptor for idx, descriptor in enumerate(subtask.z_tasks or []) if int(idx) not in index_set]
    blocked_totes = {int(x) for x in (external_used_totes or set())}
    local_used_totes = {
        int(tote_id)
        for descriptor in preserved_before
        for tote_id in (descriptor.target_tote_ids or ())
        if int(tote_id) >= 0
    }
    available_totes = [
        int(tote_id)
        for tote_id in (shared_candidate_totes or ())
        if int(tote_id) not in blocked_totes and int(tote_id) not in local_used_totes
    ]
    remaining = _demand_counts(config, subtask)
    for descriptor in preserved_before:
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)
    temp_subtask = _build_temp_subtask(opt, config, subtask, list(preserved_before))
    dummy_task = Task(
        task_id=-1,
        sub_task_id=int(subtask.subtask_id),
        target_stack_id=int(stack_id),
        target_station_id=int(subtask.station_id),
        operation_mode="SORT",
    )
    max_sets = max(4, int(getattr(opt.cfg, "resource_joint_colocated_sort_candidate_limit", 12)))
    capacity = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)))
    option_rows: List[Tuple[Tuple[float, ...], ZTaskDescriptor]] = []
    for hit_ids in _enumerate_covering_hit_sets(opt, available_totes, remaining, capacity, max_sets):
        if not hit_ids:
            continue
        plan = opt._z_build_plan_from_hits(temp_subtask, dummy_task, int(stack_id), list(hit_ids), "SORT", {-1})
        if not bool(plan.get("valid", False)) or plan.get("sort_layer_range", None) is None:
            continue
        plan = _normalize_joint_sort_plan(opt, plan, hit_ids)
        if not _sort_plan_within_capacity(plan):
            continue
        target_ids = [int(x) for x in (plan.get("target_tote_ids", []) or [])]
        if any(int(tid) in blocked_totes or int(tid) in local_used_totes for tid in target_ids):
            continue
        coverage_gain = int(_coverage_gain(opt, remaining, list(plan.get("hit_tote_ids", []) or [])))
        if coverage_gain <= 0:
            continue
        descriptor = _descriptor_from_plan(opt, subtask, plan, int(config.next_task_id), coverage_gain)
        assignment = _replace_descriptor_group(subtask, group_indices, descriptor)
        if not validate_z_assignment(opt, config, subtask, assignment, external_used_totes=blocked_totes):
            continue
        score = (
            float(descriptor.robot_service_time + descriptor.station_service_time),
            float(len(descriptor.target_tote_ids or ())),
            float(-coverage_gain),
            float(descriptor.sort_layer_range[1] - descriptor.sort_layer_range[0]) if descriptor.sort_layer_range is not None else 0.0,
            float(min(descriptor.target_tote_ids) if descriptor.target_tote_ids else 10**9),
        )
        option_rows.append((score, descriptor))
    option_rows.sort(key=lambda item: item[0])
    unique: List[ZTaskDescriptor] = []
    seen = set()
    for _, descriptor in option_rows:
        sig = descriptor.signature()
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(descriptor)
        if len(unique) >= int(max_sets):
            break
    return unique


def _is_verified_joint_colocated_sort(descriptor: ZTaskDescriptor, group_size: int) -> bool:
    return (
        int(group_size) >= 2
        and str(descriptor.mode).upper() == "SORT"
        and descriptor.sort_layer_range is not None
        and bool(descriptor.hit_tote_ids)
    )


def _joint_sort_tie_break(descriptor: ZTaskDescriptor, group_size: int) -> float:
    return -0.05 if _is_verified_joint_colocated_sort(descriptor, group_size) else 0.0


def _joint_colocated_sort_objective(descriptors: Sequence[ZTaskDescriptor]) -> Tuple[float, ...]:
    robot_service = sum(float(getattr(item, "robot_service_time", 0.0) or 0.0) for item in (descriptors or ()))
    station_service = sum(float(getattr(item, "station_service_time", 0.0) or 0.0) for item in (descriptors or ()))
    noise = sum(len(getattr(item, "noise_tote_ids", ()) or ()) for item in (descriptors or ()))
    targets = sum(len(getattr(item, "target_tote_ids", ()) or ()) for item in (descriptors or ()))
    return (
        float(len(list(descriptors or ()))),
        float(noise),
        float(targets),
        float(robot_service + station_service),
    )


def _empty_joint_postprocess_stats() -> Dict[str, float]:
    return {
        "triggered": 0.0,
        "candidate_groups": 0.0,
        "submitted": 0.0,
        "applied": 0.0,
        "makespan_improvement": 0.0,
        "rejected_capacity": 0.0,
        "rejected_interval_illegal": 0.0,
        "rejected_noise": 0.0,
        "rejected_eval_not_better": 0.0,
        "rejected_validation": 0.0,
        "rejected_target_conflict": 0.0,
    }


def _joint_colocated_flip_groups(config: ResourceConfig) -> List[List[Tuple[int, int]]]:
    station_stack_groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for subtask_id in sorted(config.subtasks.keys()):
        subtask = config.subtasks.get(int(subtask_id))
        if subtask is None:
            continue
        for task_idx, descriptor in enumerate(subtask.z_tasks or []):
            if str(descriptor.mode).upper() != "FLIP" or int(descriptor.stack_id) < 0 or int(subtask.station_id) < 0:
                continue
            station_stack_groups[(int(subtask.station_id), int(descriptor.stack_id))].append((int(subtask_id), int(task_idx)))
    return [
        list(entries)
        for _, entries in sorted(station_stack_groups.items(), key=lambda item: item[0])
        if len(entries) >= 2
    ]


def apply_joint_colocated_sort_postprocess(
    opt,
    config: ResourceConfig,
    max_groups: int = 1,
) -> Tuple[ResourceConfig, Dict[str, float]]:
    candidate = config.clone()
    stats = _empty_joint_postprocess_stats()
    if int(max_groups) <= 0:
        return candidate, stats
    stats["triggered"] = 1.0
    applied_groups = 0
    for grouped_entries in _joint_colocated_flip_groups(candidate):
        if applied_groups >= int(max_groups):
            break
        stats["candidate_groups"] += 1.0
        per_subtask_indices: Dict[int, List[int]] = defaultdict(list)
        before_descriptors: List[ZTaskDescriptor] = []
        after_rows: Dict[int, ZTaskDescriptor] = {}
        reject_key = ""
        for subtask_id, task_idx in grouped_entries:
            subtask = candidate.subtasks.get(int(subtask_id))
            if subtask is None:
                reject_key = "rejected_validation"
                break
            per_subtask_indices[int(subtask_id)].append(int(task_idx))
            before_descriptors.append(subtask.z_tasks[int(task_idx)])
        if reject_key:
            stats[reject_key] += 1.0
            continue
        stack_id = int(before_descriptors[0].stack_id) if before_descriptors else -1
        shared_candidate_totes: List[int] = []
        for descriptor in before_descriptors:
            for tote_id in (descriptor.hit_tote_ids or ()):
                tid = int(tote_id)
                if tid >= 0 and tid not in shared_candidate_totes:
                    shared_candidate_totes.append(tid)
        stack = opt.problem.point_to_stack.get(int(stack_id)) if getattr(opt, "problem", None) is not None else None
        if stack is not None:
            for tote in getattr(stack, "totes", []) or []:
                tid = int(getattr(tote, "id", -1))
                if tid < 0 or tid in shared_candidate_totes:
                    continue
                if any(_tote_demand_overlap(opt, tid, _demand_counts(candidate, candidate.subtasks[int(subtask_id)])) > 0 for subtask_id in per_subtask_indices.keys()):
                    shared_candidate_totes.append(tid)
        option_map: Dict[int, List[ZTaskDescriptor]] = {}
        group_external_used = global_used_totes(candidate, exclude_subtask_ids=set(int(subtask_id) for subtask_id in per_subtask_indices.keys()))
        for subtask_id, indices in per_subtask_indices.items():
            subtask = candidate.subtasks.get(int(subtask_id))
            options = _build_joint_sort_candidate_options(
                opt=opt,
                config=candidate,
                subtask=subtask,
                group_indices=sorted(indices),
                stack_id=int(stack_id),
                shared_candidate_totes=list(shared_candidate_totes),
                external_used_totes=set(group_external_used),
            )
            if not options:
                reject_key = "rejected_interval_illegal"
                break
            option_map[int(subtask_id)] = list(options)
        if reject_key:
            stats[reject_key] += 1.0
            continue
        ordered_subtasks = sorted(option_map.keys())
        best_combo: Optional[Tuple[Tuple[float, ...], Dict[int, ZTaskDescriptor]]] = None

        def dfs_combo(pos: int, used_targets: Set[int], chosen: Dict[int, ZTaskDescriptor]) -> None:
            nonlocal best_combo
            if pos >= len(ordered_subtasks):
                combo_descriptors = [chosen[subtask_id] for subtask_id in ordered_subtasks]
                score = _joint_colocated_sort_objective(combo_descriptors)
                if best_combo is None or score < best_combo[0]:
                    best_combo = (score, dict(chosen))
                return
            subtask_id = int(ordered_subtasks[pos])
            for descriptor in option_map.get(int(subtask_id), []):
                target_set = {int(tid) for tid in (descriptor.target_tote_ids or ())}
                if target_set & used_targets:
                    continue
                chosen[int(subtask_id)] = descriptor
                dfs_combo(pos + 1, set(used_targets) | target_set, chosen)
                chosen.pop(int(subtask_id), None)

        dfs_combo(0, set(), {})
        if best_combo is None:
            stats["rejected_target_conflict"] += 1.0
            continue
        after_rows = dict(best_combo[1])
        before_objective = _joint_colocated_sort_objective(before_descriptors)
        after_objective = _joint_colocated_sort_objective(list(after_rows.values()))
        if after_objective >= before_objective:
            stats["rejected_eval_not_better"] += 1.0
            continue
        for subtask_id, indices in per_subtask_indices.items():
            subtask = candidate.subtasks.get(int(subtask_id))
            descriptor = after_rows[int(subtask_id)]
            subtask.z_tasks = _replace_descriptor_group(subtask, indices, descriptor)
            candidate.next_task_id = max(int(candidate.next_task_id), int(descriptor.task_id) + 1)
        candidate.metadata["joint_colocated_sort_postprocess"] = True
        stats["submitted"] += 1.0
        stats["applied"] += 1.0
        applied_groups += 1
    candidate.rebuild_indices()
    return candidate, stats


def apply_single_flip_sortify_polish(opt, config: ResourceConfig) -> Tuple[ResourceConfig, Dict[str, float]]:
    candidate = config.clone()
    stats = {
        "triggered": 1.0,
        "attempted": 0.0,
        "applied": 0.0,
        "rejected_invalid": 0.0,
        "rejected_capacity": 0.0,
        "rejected_duplicate_tote": 0.0,
    }
    for subtask_id in sorted(candidate.subtasks.keys()):
        subtask = candidate.subtasks.get(int(subtask_id))
        if subtask is None or not subtask.z_tasks:
            continue
        external_used = global_used_totes(candidate, exclude_subtask_ids={int(subtask_id)})
        updated: List[ZTaskDescriptor] = []
        local_used: Set[int] = set()
        changed = False
        for descriptor in list(subtask.z_tasks or []):
            if str(getattr(descriptor, "mode", "")).upper() != "FLIP":
                updated.append(descriptor)
                local_used.update(int(tid) for tid in (descriptor.target_tote_ids or ()) if int(tid) >= 0)
                continue
            stats["attempted"] += 1.0
            hit_ids = _dedupe_ints(descriptor.hit_tote_ids or descriptor.target_tote_ids or ())
            if not hit_ids:
                updated.append(descriptor)
                stats["rejected_invalid"] += 1.0
                continue
            temp_subtask = _build_temp_subtask(opt, candidate, subtask, [descriptor])
            dummy_task = _descriptor_to_task(subtask, descriptor)
            plan = opt._z_build_plan_from_hits(
                temp_subtask,
                dummy_task,
                int(descriptor.stack_id),
                hit_ids,
                "SORT",
                {int(descriptor.task_id)},
            )
            if not bool(plan.get("valid", False)):
                updated.append(descriptor)
                stats["rejected_invalid"] += 1.0
                continue
            plan = _canonicalize_z_plan(opt, plan)
            if not _sort_plan_within_capacity(plan):
                updated.append(descriptor)
                stats["rejected_capacity"] += 1.0
                continue
            target_ids = _dedupe_ints(plan.get("target_tote_ids", []) or [])
            if any(int(tid) in external_used or int(tid) in local_used for tid in target_ids):
                updated.append(descriptor)
                stats["rejected_duplicate_tote"] += 1.0
                continue
            replacement = _descriptor_from_plan(opt, subtask, plan, int(descriptor.task_id), int(max(1, descriptor.sku_pick_count)))
            updated.append(replacement)
            local_used.update(int(tid) for tid in (replacement.target_tote_ids or ()) if int(tid) >= 0)
            changed = True
            stats["applied"] += 1.0
        if changed:
            subtask.z_tasks = updated
    candidate.rebuild_indices()
    return candidate, stats


def _joint_sort_seed_hit_map(removed_window: Sequence[ZTaskDescriptor]) -> Dict[int, List[int]]:
    stack_to_rows: Dict[int, List[ZTaskDescriptor]] = defaultdict(list)
    for descriptor in removed_window or ():
        if str(getattr(descriptor, "mode", "")).upper() != "FLIP":
            continue
        stack_to_rows[int(getattr(descriptor, "stack_id", -1))].append(descriptor)
    seed_hits: Dict[int, List[int]] = {}
    for stack_id, rows in stack_to_rows.items():
        if int(stack_id) < 0 or len(rows) < 2:
            continue
        dedup: List[int] = []
        for row in rows:
            for tote_id in (getattr(row, "hit_tote_ids", ()) or ()):
                tid = int(tote_id)
                if tid >= 0 and tid not in dedup:
                    dedup.append(tid)
        if dedup:
            seed_hits[int(stack_id)] = list(dedup)
    return seed_hits


def _validation_meta(subtask: ResourceSubtask, descriptor: Optional[ZTaskDescriptor], idx: int, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "subtask_id": int(getattr(subtask, "subtask_id", -1)),
        "descriptor_index": int(idx),
    }
    if descriptor is not None:
        meta.update(
            {
                "task_id": int(getattr(descriptor, "task_id", -1)),
                "stack_id": int(getattr(descriptor, "stack_id", -1)),
                "mode": str(getattr(descriptor, "mode", "")).upper(),
                "target_tote_ids": list(int(x) for x in (getattr(descriptor, "target_tote_ids", ()) or ())),
                "hit_tote_ids": list(int(x) for x in (getattr(descriptor, "hit_tote_ids", ()) or ())),
                "noise_tote_ids": list(int(x) for x in (getattr(descriptor, "noise_tote_ids", ()) or ())),
                "sort_layer_range": None
                if getattr(descriptor, "sort_layer_range", None) is None
                else [int(descriptor.sort_layer_range[0]), int(descriptor.sort_layer_range[1])],
            }
        )
    if extra:
        meta.update(extra)
    return meta


def validate_z_assignment_detail(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    descriptors: Sequence[ZTaskDescriptor],
    external_used_totes: Optional[Set[int]] = None,
) -> Tuple[bool, str, Dict[str, object]]:
    used_totes: Set[int] = set()
    blocked = {int(x) for x in (external_used_totes or set())}
    for idx, descriptor in enumerate(descriptors or ()):
        mode = str(descriptor.mode).upper()
        target_ids = [int(x) for x in (descriptor.target_tote_ids or ())]
        hit_ids = [int(x) for x in (descriptor.hit_tote_ids or ())]
        noise_ids = [int(x) for x in (descriptor.noise_tote_ids or ())]
        if len(target_ids) != len(set(target_ids)) or len(hit_ids) != len(set(hit_ids)) or len(noise_ids) != len(set(noise_ids)):
            return False, "duplicate_or_blocked_tote", _validation_meta(subtask, descriptor, idx, {"duplicate_scope": "descriptor"})
        if mode == "FLIP":
            if tuple(target_ids) != tuple(hit_ids) or list(noise_ids) or descriptor.sort_layer_range is not None:
                return False, "flip_target_hit_mismatch", _validation_meta(subtask, descriptor, idx)
        if mode == "SORT":
            if descriptor.sort_layer_range is None:
                return False, "sort_missing_range", _validation_meta(subtask, descriptor, idx)
            interval_list = _stack_interval_tote_ids(opt, int(descriptor.stack_id), descriptor.sort_layer_range)
            if interval_list is None or not interval_list:
                return False, "sort_range_invalid", _validation_meta(subtask, descriptor, idx)
            interval_totes = set(interval_list)
            target_set = set(target_ids)
            hit_set = set(hit_ids)
            noise_set = set(noise_ids)
            if not target_set:
                return False, "sort_target_not_contiguous", _validation_meta(subtask, descriptor, idx, {"expected_target_tote_ids": list(interval_list)})
            if list(target_ids) != list(interval_list):
                return False, "sort_target_not_contiguous", _validation_meta(subtask, descriptor, idx, {"expected_target_tote_ids": list(interval_list)})
            if not hit_set.issubset(target_set):
                return False, "hit_not_subset_target", _validation_meta(subtask, descriptor, idx)
            if not noise_set.issubset(target_set):
                return False, "noise_not_subset_target", _validation_meta(subtask, descriptor, idx)
            if hit_set & noise_set:
                return False, "hit_noise_overlap", _validation_meta(subtask, descriptor, idx, {"overlap_tote_ids": sorted(hit_set & noise_set)})
            if target_set != (hit_set | noise_set):
                return False, "target_not_hit_plus_noise", _validation_meta(subtask, descriptor, idx, {"missing_tote_ids": sorted(target_set - (hit_set | noise_set))})
        for tote_id in target_ids:
            if int(tote_id) in used_totes or int(tote_id) in blocked:
                return False, "duplicate_or_blocked_tote", _validation_meta(subtask, descriptor, idx, {"tote_id": int(tote_id)})
            used_totes.add(int(tote_id))
    demand = _demand_counts(config, subtask)
    remaining = dict(demand)
    for descriptor in descriptors:
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)
    if not all(int(qty) <= 0 for qty in remaining.values()):
        return False, "unmet_demand", _validation_meta(subtask, None, -1, {"remaining_demand": dict(remaining)})
    rough_ok, _, rough_meta = _rough_route_feasibility(opt, config, subtask, descriptors)
    if not bool(rough_ok):
        return False, "rough_route_infeasible", _validation_meta(subtask, None, -1, {"rough_meta": rough_meta})
    return True, "", {}


def validate_z_assignment(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    descriptors: Sequence[ZTaskDescriptor],
    external_used_totes: Optional[Set[int]] = None,
) -> bool:
    ok, _, _ = validate_z_assignment_detail(opt, config, subtask, descriptors, external_used_totes=external_used_totes)
    return bool(ok)


def build_full_z_assignment(
    opt,
    config: ResourceConfig,
    subtask_id: int,
    preferred_stack_ids: Optional[Sequence[int]] = None,
    strategy: str = "fallback",
    allow_fallback: bool = True,
    external_used_totes: Optional[Set[int]] = None,
    rng=None,
) -> Tuple[bool, List[ZTaskDescriptor], Dict[str, object]]:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return False, [], {"reason": "subtask_missing"}
    preserved: List[ZTaskDescriptor] = []
    if external_used_totes is None:
        external_used_totes = global_used_totes(config, exclude_subtask_ids={int(subtask_id)})
    queue_ctx = _predict_station_queues(opt, config)
    return _rebuild_window(
        opt=opt,
        config=config,
        subtask=subtask,
        preserved_before=preserved,
        preserved_after=preserved,
        seed_stack_ids=list(preferred_stack_ids or ()),
        strategy=str(strategy),
        allow_fallback=bool(allow_fallback),
        external_used_totes=set(int(x) for x in (external_used_totes or set())),
        queue_ctx=queue_ctx,
        rng=rng,
    )


def _rebuild_window(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    preserved_before: Sequence[ZTaskDescriptor],
    preserved_after: Sequence[ZTaskDescriptor],
    seed_stack_ids: Sequence[int],
    strategy: str,
    allow_fallback: bool,
    removed_window: Optional[Sequence[ZTaskDescriptor]] = None,
    external_used_totes: Optional[Set[int]] = None,
    queue_ctx: Optional[Dict[str, object]] = None,
    rng=None,
) -> Tuple[bool, List[ZTaskDescriptor], Dict[str, object]]:
    preserved_all = list(preserved_before) + list(preserved_after)
    temp_subtask = _build_temp_subtask(opt, config, subtask, preserved_all)
    demand = _demand_counts(config, subtask)
    remaining = dict(demand)
    for descriptor in preserved_all:
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)
    created: List[ZTaskDescriptor] = []
    candidate_stack_ids = _candidate_stack_ids(opt, config, subtask, seed_stack_ids)
    fallback_used = False
    blocked_totes = {int(x) for x in (external_used_totes or set())}
    local_used_totes = {
        int(tote_id)
        for descriptor in preserved_all
        for tote_id in (descriptor.target_tote_ids or ())
        if int(tote_id) >= 0
    }
    joint_seed_hits = _joint_sort_seed_hit_map(removed_window or []) if _is_joint_sort_strategy(str(strategy)) else {}
    queue_waits = dict((queue_ctx or {}).get("expected_wait_times", {}) or {})
    predicted_arrivals = dict((queue_ctx or {}).get("predicted_arrival_times", {}) or {})
    wait_time = float(queue_waits.get(int(subtask.subtask_id), 0.0))
    threshold = float(getattr(opt.cfg, "resource_z_queue_wait_threshold_sec", 5.0))
    wait_multiplier = float(getattr(opt.cfg, "resource_z_queue_wait_multiplier", 0.10))
    wait_scale = 1.0 + max(0.0, float(wait_time) - threshold) * wait_multiplier if wait_time > threshold else 1.0
    noise_weight = float(getattr(opt.cfg, "resource_z_queue_noise_weight", 1.0))
    multistack_weight = float(getattr(opt.cfg, "resource_z_queue_multistack_weight", 0.8))
    sort_weight = float(getattr(opt.cfg, "resource_z_queue_sort_weight", 0.6))
    station_service_weight = float(getattr(opt.cfg, "resource_z_queue_station_service_weight", 0.15))
    spread_balance = str(strategy) in {"z_repair_spread_region_balance", "z_repair_load_balance_idle_robot"}
    structural_weight = float(getattr(opt.cfg, "resource_z_structural_score_weight", 0.08))

    def _queue_weighted_station_burden(descriptor: ZTaskDescriptor) -> float:
        stack_penalty = max(0.0, float(len({int(descriptor.stack_id)})) - 1.0)
        return float(
            wait_scale
            * (
                noise_weight * float(len(list(descriptor.noise_tote_ids or ())))
                + multistack_weight * stack_penalty
                + sort_weight * float(str(descriptor.mode).upper() == "SORT")
                + station_service_weight * float(getattr(descriptor, "station_service_time", 0.0) or 0.0)
            )
        )

    gurobi_like_sort = str(strategy) == "z_repair_gurobi_like_sort"
    flip_compact = str(strategy) == "z_repair_flip_compact"
    gurobi_noise_weight = float(getattr(opt.cfg, "resource_gurobi_like_sort_noise_weight", 2.0))
    gurobi_span_weight = float(getattr(opt.cfg, "resource_gurobi_like_sort_span_weight", 0.20))
    gurobi_sort_bonus = float(getattr(opt.cfg, "resource_gurobi_like_sort_bonus", 1.0))

    def _gurobi_like_sort_score(descriptor: ZTaskDescriptor, target_len: int) -> Tuple[float, float, float, float]:
        mode = str(descriptor.mode).upper()
        sort_range = getattr(descriptor, "sort_layer_range", None)
        span = 0.0
        if sort_range is not None:
            span = float(max(0, int(sort_range[1]) - int(sort_range[0]) + 1))
        noise_count = float(len(list(getattr(descriptor, "noise_tote_ids", []) or [])))
        service_time = float(getattr(descriptor, "robot_service_time", 0.0) or 0.0) + float(getattr(descriptor, "station_service_time", 0.0) or 0.0)
        mode_penalty = 0.0 if mode == "SORT" else float(gurobi_sort_bonus)
        return (
            float(mode_penalty),
            float(gurobi_noise_weight * noise_count),
            float(service_time + gurobi_span_weight * span),
            float(target_len),
        )

    while any(int(qty) > 0 for qty in remaining.values()):
        candidate_rows = []
        if joint_seed_hits:
            for stack_id, seed_hits in joint_seed_hits.items():
                hit_ids = [
                    int(tote_id)
                    for tote_id in (seed_hits or [])
                    if int(tote_id) not in blocked_totes and int(tote_id) not in local_used_totes
                ]
                if not hit_ids:
                    continue
                dummy_task = Task(
                    task_id=-1,
                    sub_task_id=int(subtask.subtask_id),
                    target_stack_id=int(stack_id),
                    target_station_id=int(subtask.station_id),
                    operation_mode="SORT",
                )
                plan = opt._z_build_plan_from_hits(temp_subtask, dummy_task, int(stack_id), hit_ids, "SORT", {-1})
                if not bool(plan.get("valid", False)):
                    continue
                plan = _canonicalize_z_plan(opt, plan)
                if not _sort_plan_within_capacity(plan):
                    continue
                target_ids = [int(x) for x in (plan.get("target_tote_ids", []) or [])]
                if any(int(tid) in blocked_totes or int(tid) in local_used_totes for tid in target_ids):
                    continue
                coverage_gain = int(_coverage_gain(opt, remaining, list(plan.get("hit_tote_ids", []) or [])))
                if coverage_gain <= 0:
                    continue
                guard_reason = _guard_reason(opt, config, subtask, plan, queue_ctx=queue_ctx)
                if guard_reason and not bool(fallback_used):
                    continue
                temp_descriptor = _descriptor_from_plan(opt, subtask, plan, -1, coverage_gain)
                rough_ok, rough_penalty, rough_meta = _rough_route_feasibility(opt, config, subtask, list(created) + [temp_descriptor])
                if not rough_ok and not bool(fallback_used):
                    continue
                station_load_soft = float(_z_station_load(config, int(subtask.station_id)))
                robot_region_load_soft = float(_z_robot_region_load(opt, int(subtask.station_id)))
                choke_over_soft = float(rough_meta.get("choke_over", 0.0) or 0.0)
                structural_score = float(choke_over_soft + 0.05 * station_load_soft + robot_region_load_soft)
                detour = float(opt._z_best_insertion_detour(int(stack_id)))
                target_len = len(list(plan.get("target_tote_ids", []) or []))
                queue_burden = float(_queue_weighted_station_burden(temp_descriptor))
                if gurobi_like_sort:
                    score = (
                        -float(coverage_gain),
                        *_gurobi_like_sort_score(temp_descriptor, target_len),
                        float(queue_burden),
                        float(rough_penalty),
                        float(structural_weight * structural_score),
                        float(detour),
                        int(stack_id),
                        "SORT",
                    )
                else:
                    score = (
                        -float(coverage_gain),
                        float(queue_burden),
                        float(rough_penalty),
                        float(structural_weight * structural_score),
                        float(detour),
                        float(target_len),
                        float(_joint_sort_tie_break(_descriptor_from_plan(opt, subtask, plan, -1, coverage_gain), len(seed_hits))),
                        int(stack_id),
                        "SORT",
                    )
                candidate_rows.append({"score": score, "plan": plan, "coverage_gain": coverage_gain, "structural_score": structural_score})

        for stack_id in candidate_stack_ids:
            dummy_task = Task(
                task_id=-1,
                sub_task_id=int(subtask.subtask_id),
                target_stack_id=int(stack_id),
                target_station_id=int(subtask.station_id),
                operation_mode="FLIP",
            )
            summary = opt._z_stack_summary(temp_subtask, int(stack_id), {-1})
            hit_ids = [
                int(tote_id)
                for tote_id in (summary.get("hit_tote_ids", []) or [])
                if int(tote_id) not in blocked_totes and int(tote_id) not in local_used_totes
            ]
            if not hit_ids:
                continue
            modes = ["FLIP", "SORT"]
            if str(strategy) in {"z_repair_sort_range_shrink_first", "z_repair_gurobi_like_sort"}:
                modes = ["SORT", "FLIP"]
            elif flip_compact:
                modes = ["FLIP", "SORT"]
            elif _is_joint_sort_strategy(str(strategy)) and int(stack_id) in set(int(x) for x in joint_seed_hits.keys()):
                modes = ["SORT", "FLIP"]
            for mode in modes:
                plan = opt._z_build_plan_from_hits(temp_subtask, dummy_task, int(stack_id), hit_ids, str(mode).upper(), {-1})
                if not bool(plan.get("valid", False)):
                    continue
                plan = _canonicalize_z_plan(opt, plan)
                if not _sort_plan_within_capacity(plan):
                    continue
                target_ids = [int(x) for x in (plan.get("target_tote_ids", []) or [])]
                if any(int(tid) in blocked_totes or int(tid) in local_used_totes for tid in target_ids):
                    continue
                coverage_gain = int(_coverage_gain(opt, remaining, list(plan.get("hit_tote_ids", []) or [])))
                if coverage_gain <= 0:
                    continue
                guard_reason = _guard_reason(opt, config, subtask, plan, queue_ctx=queue_ctx)
                if guard_reason and not bool(fallback_used):
                    continue
                temp_descriptor = _descriptor_from_plan(opt, subtask, plan, -1, coverage_gain)
                rough_ok, rough_penalty, rough_meta = _rough_route_feasibility(opt, config, subtask, list(created) + [temp_descriptor])
                if not rough_ok and not bool(fallback_used):
                    continue
                station_load_soft = float(_z_station_load(config, int(subtask.station_id)))
                robot_region_load_soft = float(_z_robot_region_load(opt, int(subtask.station_id)))
                choke_over_soft = float(rough_meta.get("choke_over", 0.0) or 0.0)
                structural_score = float(choke_over_soft + 0.05 * station_load_soft + robot_region_load_soft)
                spread_bonus = -0.25 if bool(spread_balance) and structural_score <= 1.0 else 0.0
                detour = float(opt._z_best_insertion_detour(int(stack_id)))
                target_len = len(list(plan.get("target_tote_ids", []) or []))
                queue_burden = float(_queue_weighted_station_burden(temp_descriptor))
                same_stack_bonus = 0.0 if int(stack_id) in set(int(x) for x in seed_stack_ids) else 1.0
                if _is_joint_sort_strategy(str(strategy)) and int(stack_id) in set(int(x) for x in joint_seed_hits.keys()):
                    same_stack_bonus -= 0.1
                if flip_compact:
                    score = (
                        -float(coverage_gain),
                        0.0 if str(mode).upper() == "FLIP" else 1.5,
                        float(rough_penalty),
                        float(detour),
                        float(queue_burden),
                        float(structural_weight * structural_score + spread_bonus),
                        float(same_stack_bonus),
                        float(len(list(plan.get("target_tote_ids", []) or []))),
                        int(stack_id),
                        str(mode).upper(),
                    )
                elif gurobi_like_sort:
                    score = (
                        -float(coverage_gain),
                        *_gurobi_like_sort_score(temp_descriptor, target_len),
                        float(queue_burden),
                        float(rough_penalty),
                        float(structural_weight * structural_score + spread_bonus),
                        float(detour),
                        float(same_stack_bonus),
                        int(stack_id),
                        str(mode).upper(),
                    )
                else:
                    score = (
                        -float(coverage_gain),
                        float(queue_burden),
                        float(rough_penalty),
                        float(structural_weight * structural_score + spread_bonus),
                        float(detour),
                        float(same_stack_bonus),
                        float(target_len),
                        float(_joint_sort_tie_break(_descriptor_from_plan(opt, subtask, plan, -1, coverage_gain), len(joint_seed_hits.get(int(stack_id), [])))),
                        int(stack_id),
                        str(mode).upper(),
                    )
                candidate_rows.append({"score": score, "plan": plan, "coverage_gain": coverage_gain, "structural_score": structural_score})
        chosen_row = pick_soft_greedy_min(rng, candidate_rows, opt.cfg, score_getter=lambda item: item["score"])
        if chosen_row is None:
            if not bool(allow_fallback) or bool(fallback_used):
                break
            fallback_used = True
            continue

        chosen_plan = chosen_row["plan"]
        coverage_gain = int(chosen_row["coverage_gain"])
        next_task_id = int(config.next_task_id)
        config.next_task_id += 1
        descriptor = _descriptor_from_plan(opt, subtask, chosen_plan, next_task_id, coverage_gain)
        created.append(descriptor)
        local_used_totes.update(int(tid) for tid in (descriptor.target_tote_ids or ()) if int(tid) >= 0)
        temp_task = _descriptor_to_task(subtask, descriptor)
        stack_obj = opt.problem.point_to_stack.get(int(temp_task.target_stack_id))
        if stack_obj is not None:
            temp_subtask.add_execution_detail(temp_task, stack_obj)
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)

    full_assignment = list(preserved_before) + list(created) + list(preserved_after)
    validation_ok, validation_reason, validation_detail = validate_z_assignment_detail(
        opt,
        config,
        subtask,
        full_assignment,
        external_used_totes=blocked_totes,
    )
    if validation_ok:
        _, rough_penalty, rough_meta = _rough_route_feasibility(opt, config, subtask, full_assignment)
        return True, full_assignment, {
            "fallback_used": bool(fallback_used),
            "rough_penalty": float(rough_penalty),
            "rough_meta": rough_meta,
            "predicted_wait_sec": float(wait_time),
            "predicted_arrival_sec": float(predicted_arrivals.get(int(subtask.subtask_id), 0.0)),
        }
    _, rough_penalty, rough_meta = _rough_route_feasibility(opt, config, subtask, full_assignment)
    return False, list(full_assignment), {
        "reason": str(validation_reason or "invalid_assignment"),
        "validation_detail": dict(validation_detail or {}),
        "fallback_used": bool(fallback_used),
        "rough_penalty": float(rough_penalty),
        "rough_meta": rough_meta,
        "predicted_wait_sec": float(wait_time),
        "predicted_arrival_sec": float(predicted_arrivals.get(int(subtask.subtask_id), 0.0)),
    }


def _expand_window(descriptors: Sequence[ZTaskDescriptor], center_idx: int, base_size: int, max_size: int, mode_sensitive: bool = False) -> Tuple[int, int]:
    if not descriptors:
        return 0, 0
    size = max(1, min(int(base_size), len(descriptors)))
    start = max(0, int(center_idx) - size // 2)
    end = min(len(descriptors), start + size)
    start = max(0, end - size)
    if bool(mode_sensitive):
        base_mode = str(descriptors[int(center_idx)].mode).upper()
        while start > 0 and str(descriptors[start - 1].mode).upper() == base_mode and (end - start) < int(max_size):
            start -= 1
        while end < len(descriptors) and str(descriptors[end].mode).upper() == base_mode and (end - start) < int(max_size):
            end += 1
    return int(start), int(end)


def _destroy_window(config: ResourceConfig, subtask_id: int, start: int, end: int) -> Dict[str, object]:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return {"success": False}
    before = list(subtask.z_tasks[:int(start)])
    removed = list(subtask.z_tasks[int(start):int(end)])
    after = list(subtask.z_tasks[int(end):])
    if not removed:
        return {"success": False}
    subtask.z_tasks = list(before) + list(after)
    return {
        "success": True,
        "subtask_id": int(subtask_id),
        "window_start": int(start),
        "window_end": int(end),
        "preserved_before": before,
        "removed_window": removed,
        "preserved_after": after,
        "seed_stack_ids": [int(row.stack_id) for row in removed],
    }


def _preview_destroy_window(config: ResourceConfig, subtask_id: int, start: int, end: int) -> Dict[str, object]:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return {"success": False}
    before = list(subtask.z_tasks[:int(start)])
    removed = list(subtask.z_tasks[int(start):int(end)])
    after = list(subtask.z_tasks[int(end):])
    if not removed:
        return {"success": False}
    return {
        "success": True,
        "subtask_id": int(subtask_id),
        "window_start": int(start),
        "window_end": int(end),
        "preserved_before": before,
        "removed_window": removed,
        "preserved_after": after,
        "seed_stack_ids": [int(row.stack_id) for row in removed],
    }


def _destroy_windows(opt, config: ResourceConfig, rng, degree: int, candidate_builder, mode_sensitive: bool) -> Dict[str, object]:
    budget_remaining = max(1, int(degree))
    windows = []
    touched_subtasks: Set[int] = set()
    removed_total = 0
    while budget_remaining > 0:
        candidates = list(candidate_builder(config, touched_subtasks))
        if not candidates:
            break
        picked = pick_ranked_candidate(rng, candidates, opt.cfg)
        if picked is None:
            break
        _, subtask_id, center_idx = picked
        row = config.subtasks.get(int(subtask_id))
        if row is None or not row.z_tasks:
            touched_subtasks.add(int(subtask_id))
            continue
        window_size = max(int(getattr(opt.cfg, "resource_z_window_size", 3)), min(int(budget_remaining), 5))
        start, end = _expand_window(row.z_tasks, int(center_idx), int(window_size), 5, mode_sensitive=bool(mode_sensitive))
        ctx = _destroy_window(config, int(subtask_id), int(start), int(end))
        touched_subtasks.add(int(subtask_id))
        if not bool(ctx.get("success", False)):
            continue
        removed_len = len(list(ctx.get("removed_window", []) or []))
        removed_total += int(removed_len)
        budget_remaining -= int(removed_len)
        windows.append(ctx)
    if not windows:
        return {"success": False}
    payload = {"success": True, "windows": windows, "released_task_count": int(removed_total)}
    if len(windows) == 1:
        payload.update(dict(windows[0]))
    return payload


def _plan_destroy_windows(opt, config: ResourceConfig, rng, degree: int, candidate_builder, mode_sensitive: bool) -> Dict[str, object]:
    budget_remaining = max(1, int(degree))
    windows = []
    touched_subtasks: Set[int] = set()
    removed_total = 0
    while budget_remaining > 0:
        candidates = list(candidate_builder(config, touched_subtasks))
        if not candidates:
            break
        picked = pick_ranked_candidate(rng, candidates, opt.cfg)
        if picked is None:
            break
        _, subtask_id, center_idx = picked
        row = config.subtasks.get(int(subtask_id))
        if row is None or not row.z_tasks:
            touched_subtasks.add(int(subtask_id))
            continue
        window_size = max(int(getattr(opt.cfg, "resource_z_window_size", 3)), min(int(budget_remaining), 5))
        start, end = _expand_window(row.z_tasks, int(center_idx), int(window_size), 5, mode_sensitive=bool(mode_sensitive))
        ctx = _preview_destroy_window(config, int(subtask_id), int(start), int(end))
        touched_subtasks.add(int(subtask_id))
        if not bool(ctx.get("success", False)):
            continue
        removed_len = len(list(ctx.get("removed_window", []) or []))
        removed_total += int(removed_len)
        budget_remaining -= int(removed_len)
        windows.append(ctx)
    if not windows:
        return {"success": False}
    payload = {"success": True, "windows": windows, "released_task_count": int(removed_total)}
    if len(windows) == 1:
        payload.update(dict(windows[0]))
    return payload


def z_destroy_noise_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            noise_scores = [len(task.noise_tote_ids) for task in row.z_tasks]
            center_idx = max(range(len(noise_scores)), key=lambda idx: (noise_scores[idx], -idx))
            candidates.append(((-max(noise_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_noise_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            noise_scores = [len(task.noise_tote_ids) for task in row.z_tasks]
            center_idx = max(range(len(noise_scores)), key=lambda idx: (noise_scores[idx], -idx))
            candidates.append(((-max(noise_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_multistack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or len(row.z_tasks) < 2:
                continue
            change_scores = []
            for idx in range(len(row.z_tasks)):
                left = int(row.z_tasks[idx - 1].stack_id) if idx > 0 else int(row.z_tasks[idx].stack_id)
                right = int(row.z_tasks[idx + 1].stack_id) if idx + 1 < len(row.z_tasks) else int(row.z_tasks[idx].stack_id)
                score = int(left != int(row.z_tasks[idx].stack_id)) + int(right != int(row.z_tasks[idx].stack_id))
                change_scores.append(score)
            center_idx = max(range(len(change_scores)), key=lambda idx: (change_scores[idx], -idx))
            candidates.append(((-max(change_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_multistack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or len(row.z_tasks) < 2:
                continue
            change_scores = []
            for idx in range(len(row.z_tasks)):
                left = int(row.z_tasks[idx - 1].stack_id) if idx > 0 else int(row.z_tasks[idx].stack_id)
                right = int(row.z_tasks[idx + 1].stack_id) if idx + 1 < len(row.z_tasks) else int(row.z_tasks[idx].stack_id)
                score = int(left != int(row.z_tasks[idx].stack_id)) + int(right != int(row.z_tasks[idx].stack_id))
                change_scores.append(score)
            center_idx = max(range(len(change_scores)), key=lambda idx: (change_scores[idx], -idx))
            candidates.append(((-max(change_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_detour_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            detours = [float(opt._z_best_insertion_detour(int(task.stack_id))) for task in row.z_tasks]
            center_idx = max(range(len(detours)), key=lambda idx: (detours[idx], -idx))
            candidates.append(((-max(detours), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_detour_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            detours = [float(opt._z_best_insertion_detour(int(task.stack_id))) for task in row.z_tasks]
            center_idx = max(range(len(detours)), key=lambda idx: (detours[idx], -idx))
            candidates.append(((-max(detours), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_mode_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            mode_scores = []
            for idx in range(len(row.z_tasks)):
                left = str(row.z_tasks[idx - 1].mode).upper() if idx > 0 else str(row.z_tasks[idx].mode).upper()
                right = str(row.z_tasks[idx + 1].mode).upper() if idx + 1 < len(row.z_tasks) else str(row.z_tasks[idx].mode).upper()
                cur = str(row.z_tasks[idx].mode).upper()
                mode_scores.append(int(left != cur) + int(right != cur) + int(cur == "SORT"))
            center_idx = max(range(len(mode_scores)), key=lambda idx: (mode_scores[idx], -idx))
            candidates.append(((-max(mode_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=True)


def z_plan_destroy_mode_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            mode_scores = []
            for idx in range(len(row.z_tasks)):
                left = str(row.z_tasks[idx - 1].mode).upper() if idx > 0 else str(row.z_tasks[idx].mode).upper()
                right = str(row.z_tasks[idx + 1].mode).upper() if idx + 1 < len(row.z_tasks) else str(row.z_tasks[idx].mode).upper()
                cur = str(row.z_tasks[idx].mode).upper()
                mode_scores.append(int(left != cur) + int(right != cur) + int(cur == "SORT"))
            center_idx = max(range(len(mode_scores)), key=lambda idx: (mode_scores[idx], -idx))
            candidates.append(((-max(mode_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=True)


def z_destroy_spread_hotspot_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    queue_ctx = _predict_station_queues(opt, config)
    wait_map = dict(queue_ctx.get("expected_wait_times", {}) or {})

    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks or int(row.station_id) < 0:
                continue
            wait = float(wait_map.get(int(row.subtask_id), 0.0) or 0.0)
            station_load = float(_z_station_load(config_obj, int(row.station_id)))
            robot_region = float(_z_robot_region_load(opt, int(row.station_id)))
            duration = float(_estimate_assignment_duration(opt, row.z_tasks or []))
            score = float(wait + 0.05 * station_load + robot_region + 0.10 * duration)
            center_idx = max(
                range(len(row.z_tasks)),
                key=lambda idx: (
                    float(opt._z_best_insertion_detour(int(row.z_tasks[idx].stack_id))),
                    len(row.z_tasks[idx].target_tote_ids or ()),
                    -idx,
                ),
            )
            candidates.append(((-score, int(row.station_id), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_spread_hotspot_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    queue_ctx = _predict_station_queues(opt, config)
    wait_map = dict(queue_ctx.get("expected_wait_times", {}) or {})

    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks or int(row.station_id) < 0:
                continue
            wait = float(wait_map.get(int(row.subtask_id), 0.0) or 0.0)
            station_load = float(_z_station_load(config_obj, int(row.station_id)))
            robot_region = float(_z_robot_region_load(opt, int(row.station_id)))
            duration = float(_estimate_assignment_duration(opt, row.z_tasks or []))
            score = float(wait + 0.05 * station_load + robot_region + 0.10 * duration)
            center_idx = max(
                range(len(row.z_tasks)),
                key=lambda idx: (
                    float(opt._z_best_insertion_detour(int(row.z_tasks[idx].stack_id))),
                    len(row.z_tasks[idx].target_tote_ids or ()),
                    -idx,
                ),
            )
            candidates.append(((-score, int(row.station_id), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_random_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            center_idx = int(rng.randrange(len(row.z_tasks))) if rng is not None else 0
            candidates.append(((float(rng.random() if rng is not None else 0.0), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_random_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            center_idx = int(rng.randrange(len(row.z_tasks))) if rng is not None else 0
            candidates.append(((float(rng.random() if rng is not None else 0.0), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def _related_stack_candidate_builder(opt, config: ResourceConfig, rng):
    stack_rows: Dict[int, List[Tuple[float, int, int]]] = defaultdict(list)
    for row in config.subtasks.values():
        if not row.z_tasks:
            continue
        wait = 0.0
        if int(row.station_id) >= 0:
            wait = float(_z_station_load(config, int(row.station_id)))
        for idx, descriptor in enumerate(row.z_tasks):
            stack_id = int(getattr(descriptor, "stack_id", -1))
            if stack_id < 0:
                continue
            detour = float(opt._z_best_insertion_detour(int(stack_id)))
            stack_rows[int(stack_id)].append((float(wait + detour), int(row.subtask_id), int(idx)))
    candidates = [
        ((-float(len(rows)), -float(sum(item[0] for item in rows) / max(1, len(rows))), int(stack_id)), int(stack_id), rows)
        for stack_id, rows in stack_rows.items()
        if rows
    ]
    if not candidates:
        return []
    picked = pick_ranked_candidate(rng, sorted(candidates, key=lambda item: item[0]), getattr(opt, "cfg", None))
    if picked is None:
        return []
    _, _stack_id, rows = picked
    rows = sorted(rows, key=lambda item: (-float(item[0]), int(item[1]), int(item[2])))
    return [((float(rank), int(subtask_id), int(idx)), int(subtask_id), int(idx)) for rank, (_score, subtask_id, idx) in enumerate(rows)]


def z_destroy_related_stack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        return [item for item in _related_stack_candidate_builder(opt, config_obj, rng) if int(item[1]) not in touched_subtasks]

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_related_stack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        return [item for item in _related_stack_candidate_builder(opt, config_obj, rng) if int(item[1]) not in touched_subtasks]

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def _shared_stack_candidate_builder(opt, config: ResourceConfig, rng):
    stack_rows: Dict[int, List[Tuple[float, int, int]]] = defaultdict(list)
    for row in config.subtasks.values():
        if not row.z_tasks:
            continue
        wait = float(_z_station_load(config, int(row.station_id))) if int(row.station_id) >= 0 else 0.0
        for idx, descriptor in enumerate(row.z_tasks):
            stack_id = int(getattr(descriptor, "stack_id", -1))
            if stack_id < 0:
                continue
            detour = float(opt._z_best_insertion_detour(int(stack_id)))
            stack_rows[int(stack_id)].append((float(wait + detour), int(row.subtask_id), int(idx)))
    candidates = [
        ((-float(len(rows)), -float(sum(item[0] for item in rows) / max(1, len(rows))), int(stack_id)), int(stack_id), rows)
        for stack_id, rows in stack_rows.items()
        if len({int(item[1]) for item in rows}) >= 2
    ]
    if not candidates:
        return []
    picked = pick_ranked_candidate(rng, sorted(candidates, key=lambda item: item[0]), getattr(opt, "cfg", None))
    if picked is None:
        return []
    _, _stack_id, rows = picked
    rows = sorted(rows, key=lambda item: (-float(item[0]), int(item[1]), int(item[2])))
    return [((float(rank), int(subtask_id), int(idx)), int(subtask_id), int(idx)) for rank, (_score, subtask_id, idx) in enumerate(rows)]


def z_destroy_shared_stack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        return [item for item in _shared_stack_candidate_builder(opt, config_obj, rng) if int(item[1]) not in touched_subtasks]

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_shared_stack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        return [item for item in _shared_stack_candidate_builder(opt, config_obj, rng) if int(item[1]) not in touched_subtasks]

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def _critical_z_candidate_builder(opt, config: ResourceConfig, touched_subtasks: Set[int]):
    snapshot = getattr(getattr(opt, "best_validated", None), "snapshot", None)
    if snapshot is None:
        snapshot = getattr(opt, "work", None)
    critical_by_subtask: Dict[int, float] = {}
    for subtask in list(getattr(snapshot, "subtask_state", []) or []):
        subtask_id = int(getattr(subtask, "id", -1))
        task_rows = list(getattr(subtask, "execution_tasks", []) or [])
        if subtask_id < 0 or not task_rows:
            continue
        critical_by_subtask[int(subtask_id)] = max(
            float(getattr(task, "end_process_time", 0.0) or getattr(task, "arrival_time_at_station", 0.0) or 0.0)
            for task in task_rows
        )
    candidates = []
    for row in config.subtasks.values():
        if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
            continue
        critical_time = float(critical_by_subtask.get(int(row.subtask_id), 0.0))
        if critical_time <= 0.0:
            continue
        center_idx = max(
            range(len(row.z_tasks)),
            key=lambda idx: (
                float(getattr(row.z_tasks[idx], "robot_service_time", 0.0) or 0.0)
                + float(opt._z_best_insertion_detour(int(row.z_tasks[idx].stack_id))),
                len(row.z_tasks[idx].target_tote_ids or ()),
                -idx,
            ),
        )
        candidates.append(((-critical_time, int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
    return sorted(candidates, key=lambda item: item[0])


def z_destroy_critical_path_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        return _critical_z_candidate_builder(opt, config_obj, touched_subtasks)

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_critical_path_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        return _critical_z_candidate_builder(opt, config_obj, touched_subtasks)

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def _build_z_action_signature(destroy_name: str, repair_name: str, windows: Sequence[Dict[str, object]], target_stack_ids: Sequence[int], mode_summary: Sequence[str]) -> Tuple[object, ...]:
    window_sig = tuple(
        sorted(
            (
                int(window_ctx.get("subtask_id", -1)),
                tuple(int(task.task_id) for task in (window_ctx.get("removed_window", []) or [])),
            )
            for window_ctx in (windows or [])
        )
    )
    return (
        "Z",
        str(destroy_name),
        window_sig,
        str(repair_name),
        tuple(int(x) for x in (target_stack_ids or ())),
        tuple(str(x) for x in (mode_summary or ())),
    )


def _build_z_rough_features(
    opt,
    windows: Sequence[Dict[str, object]],
    target_stack_ids: Sequence[int],
    mode_summary: Sequence[str],
    config: Optional[ResourceConfig] = None,
) -> Dict[str, float]:
    removed_noise = 0
    removed_target = 0
    removed_stacks = set()
    for window_ctx in windows or []:
        for descriptor in window_ctx.get("removed_window", []) or []:
            removed_noise += len(getattr(descriptor, "noise_tote_ids", []) or [])
            removed_target += len(getattr(descriptor, "target_tote_ids", []) or [])
            removed_stacks.add(int(getattr(descriptor, "stack_id", -1)))
    detour_proxy = 0.0
    if target_stack_ids:
        detour_proxy = float(sum(float(opt._z_best_insertion_detour(int(stack_id))) for stack_id in target_stack_ids) / max(1, len(list(target_stack_ids))))
    noise_ratio = float(removed_noise / max(1, removed_target))
    stack_delta = float(max(0, len(set(int(x) for x in (target_stack_ids or []))) - max(0, len(removed_stacks) - 1)))
    mode_penalty = 0.2 * float(sum(1 for mode in (mode_summary or []) if str(mode).upper() == "SORT"))
    sz_delta = float(0.35 * noise_ratio + 0.25 * stack_delta + 0.20 * detour_proxy / 100.0 + 0.20 * mode_penalty - 0.15 * len(target_stack_ids))
    predicted_wait_sec = 0.0
    z_choke_over_soft = 0.0
    z_station_load_soft = 0.0
    z_robot_region_load_soft = 0.0
    if config is not None:
        queue_ctx = _predict_station_queues(opt, config)
        wait_map = dict(queue_ctx.get("expected_wait_times", {}) or {})
        subtask_ids = [int(window_ctx.get("subtask_id", -1)) for window_ctx in (windows or []) if int(window_ctx.get("subtask_id", -1)) >= 0]
        if subtask_ids:
            predicted_wait_sec = float(sum(float(wait_map.get(int(subtask_id), 0.0)) for subtask_id in subtask_ids) / max(1, len(subtask_ids)))
        for subtask_id in subtask_ids:
            row = config.subtasks.get(int(subtask_id))
            if row is None:
                continue
            _ok, _penalty, meta = _rough_route_feasibility(opt, config, row, row.z_tasks or [])
            z_choke_over_soft += float(meta.get("choke_over", 0.0) or 0.0)
            z_station_load_soft += float(_z_station_load(config, int(row.station_id)))
            z_robot_region_load_soft += float(_z_robot_region_load(opt, int(row.station_id)))
        denom = max(1, len(subtask_ids))
        z_choke_over_soft = float(z_choke_over_soft / denom)
        z_station_load_soft = float(z_station_load_soft / denom)
        z_robot_region_load_soft = float(z_robot_region_load_soft / denom)
    z_structural_score = float(z_choke_over_soft + 0.05 * z_station_load_soft + z_robot_region_load_soft)
    sz_delta += float(getattr(opt.cfg, "resource_z_structural_score_weight", 0.08)) * z_structural_score
    return {
        "sz_delta": float(sz_delta),
        "affected_count": float(sum(len(window_ctx.get("removed_window", []) or []) for window_ctx in (windows or []))),
        "predicted_wait_sec": float(predicted_wait_sec),
        "queue_weighted_noise_delta": float(noise_ratio * max(1.0, predicted_wait_sec)),
        "queue_weighted_station_burden_delta": float((noise_ratio + mode_penalty + stack_delta) * max(1.0, predicted_wait_sec)),
        "z_structural_score": float(z_structural_score),
        "z_choke_over_soft": float(z_choke_over_soft),
        "z_station_load_soft": float(z_station_load_soft),
        "z_robot_region_load_soft": float(z_robot_region_load_soft),
    }


def plan_z_candidate(opt, config: ResourceConfig, destroy_name: str, repair_name: str, rng, degree: int) -> Dict[str, object]:
    destroy_planners = {
        "z_destroy_noise_window": z_plan_destroy_noise_window,
        "z_destroy_multistack_window": z_plan_destroy_multistack_window,
        "z_destroy_detour_window": z_plan_destroy_detour_window,
        "z_destroy_mode_window": z_plan_destroy_mode_window,
        "z_destroy_spread_hotspot_window": z_plan_destroy_spread_hotspot_window,
        "z_destroy_random_window": z_plan_destroy_random_window,
        "z_destroy_related_stack_window": z_plan_destroy_related_stack_window,
        "z_destroy_shared_stack_window": z_plan_destroy_shared_stack_window,
        "z_destroy_critical_path_window": z_plan_destroy_critical_path_window,
    }
    destroy_ctx = destroy_planners[str(destroy_name)](opt, config, rng, degree)
    if not bool(destroy_ctx.get("success", False)):
        return {"success": False}
    windows = list(destroy_ctx.get("windows", []) or [])
    target_stack_ids: List[int] = []
    mode_summary: List[str] = []
    for window_ctx in windows:
        subtask_id = int(window_ctx.get("subtask_id", -1))
        subtask = config.subtasks.get(int(subtask_id))
        if subtask is None:
            continue
        candidate_stacks = _candidate_stack_ids(opt, config, subtask, window_ctx.get("seed_stack_ids", []) or [])
        candidate_pick_topk = max(2, int(getattr(opt.cfg, "resource_z_plan_target_stack_topk", 3)))
        for stack_id in candidate_stacks[:candidate_pick_topk]:
            if int(stack_id) not in target_stack_ids:
                target_stack_ids.append(int(stack_id))
        removed_modes = [str(descriptor.mode).upper() for descriptor in (window_ctx.get("removed_window", []) or [])]
        mode_summary.append(
            "SORT"
            if str(repair_name) in {"z_repair_sort_range_shrink_first", "z_repair_joint_sort_colocated_flip", "z_repair_cross_subtask_shared_stack"}
            else (removed_modes[0] if removed_modes else "FLIP")
        )
    return {
        "success": True,
        "destroy_ctx": destroy_ctx,
        "strategy": str(repair_name),
        "target_stack_ids": list(target_stack_ids),
        "mode_summary": list(mode_summary),
        "fallback_used": False,
        "action_signature": _build_z_action_signature(str(destroy_name), str(repair_name), windows, target_stack_ids, mode_summary),
        "rough_features": _build_z_rough_features(opt, windows, target_stack_ids, mode_summary, config=config),
    }


def apply_exact_z_plan(opt, config: ResourceConfig, plan: Dict[str, object], rng=None) -> Dict[str, object]:
    destroy_ctx = dict(plan.get("destroy_ctx", {}) or {})
    windows = list(destroy_ctx.get("windows", []) or [])
    if not windows and int(destroy_ctx.get("subtask_id", -1)) >= 0:
        windows = [destroy_ctx]
    touched_subtask_ids = sorted({int(window_ctx.get("subtask_id", -1)) for window_ctx in windows if int(window_ctx.get("subtask_id", -1)) >= 0})
    if not touched_subtask_ids:
        return {"success": False, "reason": "no_touched_subtasks"}
    candidate = config.clone_for_layer("Z", touched_subtask_ids)
    exact_windows = []
    for window_ctx in windows:
        subtask_id = int(window_ctx.get("subtask_id", -1))
        exact_ctx = _destroy_window(candidate, subtask_id, int(window_ctx.get("window_start", 0)), int(window_ctx.get("window_end", 0)))
        if not bool(exact_ctx.get("success", False)):
            return {"success": False, "reason": "exact_destroy_window_fail"}
        exact_windows.append(exact_ctx)
    exact_ctx = {"success": True, "windows": exact_windows}
    if str(plan.get("strategy", "")) == "z_repair_stack_mode_joint_polish":
        return _apply_joint_z_repair_strategies(opt, candidate, exact_ctx, rng=rng)
    repair_result = _repair_window(opt, candidate, exact_ctx, str(plan.get("strategy", "z_repair_same_stack_window")), allow_fallback=False, rng=rng)
    fallback_used = False
    if not bool(repair_result.get("success", False)):
        repair_result = _repair_window(opt, candidate, exact_ctx, "z_repair_greedy_fallback", allow_fallback=True, rng=rng)
        fallback_used = bool(repair_result.get("success", False))
    if not bool(repair_result.get("success", False)):
        return {
            "success": False,
            "reason": str(repair_result.get("reason", "exact_repair_fail") or "exact_repair_fail"),
            "validation_detail": dict(repair_result.get("validation_detail", {}) or {}),
        }
    candidate.rebuild_indices()
    return {
        "success": True,
        "config": candidate,
        "score_cache": None,
        "affected_ids": set(int(x) for x in (repair_result.get("affected_subtask_ids", set()) or set())),
        "fallback_used": bool(fallback_used or repair_result.get("fallback_used", False)),
        "projection_mode": "",
        "projection_repaired_subtask_count": 0,
        "validation_signature": candidate.validation_signature(),
    }


def _repair_window(opt, config: ResourceConfig, ctx: Dict[str, object], strategy: str, allow_fallback: bool, rng=None) -> Dict[str, object]:
    if not bool(ctx.get("success", False)):
        return {"success": False}
    windows = list(ctx.get("windows", []) or [])
    if not windows and int(ctx.get("subtask_id", -1)) >= 0:
        windows = [ctx]
    if _is_joint_sort_strategy(str(strategy)) and len(windows) != 1:
        return {"success": False, "reason": "joint_sort_requires_single_subtask_window", "fallback_used": False}
    affected_subtasks: Set[int] = set()
    fallback_used = False
    original_assignments = {
        int(window_ctx.get("subtask_id", -1)): list((config.subtasks.get(int(window_ctx.get("subtask_id", -1))).z_tasks if config.subtasks.get(int(window_ctx.get("subtask_id", -1))) is not None else []))
        for window_ctx in windows
        if int(window_ctx.get("subtask_id", -1)) >= 0
    }
    for window_ctx in windows:
        subtask_id = int(window_ctx.get("subtask_id", -1))
        subtask = config.subtasks.get(int(subtask_id))
        if subtask is None:
            return {"success": False, "reason": "subtask_missing", "fallback_used": bool(fallback_used)}
        external_used = global_used_totes(config, exclude_subtask_ids={int(subtask_id)})
        queue_ctx = _predict_station_queues(opt, config)
        success, assignment, meta = _rebuild_window(
            opt=opt,
            config=config,
            subtask=subtask,
            preserved_before=list(window_ctx.get("preserved_before", []) or []),
            preserved_after=list(window_ctx.get("preserved_after", []) or []),
            seed_stack_ids=list(window_ctx.get("seed_stack_ids", []) or []),
            strategy=str(strategy),
            allow_fallback=bool(allow_fallback),
            removed_window=list(window_ctx.get("removed_window", []) or []),
            external_used_totes=external_used,
            queue_ctx=queue_ctx,
            rng=rng,
        )
        if not success:
            for restore_subtask_id, restore_assignment in original_assignments.items():
                restore_row = config.subtasks.get(int(restore_subtask_id))
                if restore_row is not None:
                    restore_row.z_tasks = list(restore_assignment)
            return {
                "success": False,
                "reason": str(meta.get("reason", "repair_fail")),
                "validation_detail": dict(meta.get("validation_detail", {}) or {}),
                "fallback_used": bool(fallback_used or meta.get("fallback_used", False)),
            }
        subtask.z_tasks = list(assignment)
        affected_subtasks.add(int(subtask_id))
        fallback_used = bool(fallback_used or meta.get("fallback_used", False))
    return {
        "success": bool(affected_subtasks),
        "affected_subtask_ids": affected_subtasks,
        "fallback_used": bool(fallback_used),
    }


def _z_assignment_proxy(opt, config: ResourceConfig, affected_subtask_ids: Sequence[int]) -> float:
    total = 0.0
    for subtask_id in affected_subtask_ids or ():
        subtask = config.subtasks.get(int(subtask_id))
        if subtask is None:
            continue
        rough_ok, rough_penalty, rough_meta = _rough_route_feasibility(opt, config, subtask, subtask.z_tasks or [])
        structural_score = float(rough_meta.get("choke_over", 0.0) or 0.0)
        structural_score += 0.05 * float(_z_station_load(config, int(subtask.station_id)))
        structural_score += float(_z_robot_region_load(opt, int(subtask.station_id)))
        noise_total = float(sum(len(list(descriptor.noise_tote_ids or ())) for descriptor in (subtask.z_tasks or [])))
        flip_total = float(sum(1 for descriptor in (subtask.z_tasks or []) if str(descriptor.mode).upper() == "FLIP"))
        robot_service = float(sum(float(getattr(descriptor, "robot_service_time", 0.0) or 0.0) for descriptor in (subtask.z_tasks or [])))
        station_service = float(sum(float(getattr(descriptor, "station_service_time", 0.0) or 0.0) for descriptor in (subtask.z_tasks or [])))
        stack_count = float(len({int(descriptor.stack_id) for descriptor in (subtask.z_tasks or []) if int(descriptor.stack_id) >= 0}))
        total += float(rough_penalty)
        total += float(getattr(opt.cfg, "resource_z_structural_score_weight", 0.08)) * structural_score
        total += 1.75 * noise_total + 0.50 * flip_total + 0.40 * robot_service + 0.20 * station_service + 0.75 * stack_count
        if not rough_ok:
            total += 1000.0
    return float(total)


def _apply_joint_z_repair_strategies(opt, candidate: ResourceConfig, exact_ctx: Dict[str, object], rng=None) -> Dict[str, object]:
    affected_ids = sorted(
        {
            int(window_ctx.get("subtask_id", -1))
            for window_ctx in (exact_ctx.get("windows", []) or [])
            if int(window_ctx.get("subtask_id", -1)) >= 0
        }
    )
    best_result = None
    best_proxy = float("inf")
    for strategy_name in (
        "z_repair_gurobi_like_sort",
        "z_repair_multistack_cover_compact",
        "z_repair_flip_compact",
        "z_repair_sort_range_shrink_first",
        "z_repair_mode_toggle_contextual",
        "z_repair_joint_sort_colocated_flip",
    ):
        trial = candidate.clone()
        trial_ctx = copy.deepcopy(exact_ctx)
        repair_result = _repair_window(opt, trial, trial_ctx, str(strategy_name), allow_fallback=False, rng=rng)
        if not bool(repair_result.get("success", False)):
            continue
        trial.rebuild_indices()
        proxy_value = _z_assignment_proxy(opt, trial, affected_ids)
        if proxy_value + 1e-9 < best_proxy:
            best_proxy = float(proxy_value)
            best_result = {
                "success": True,
                "config": trial,
                "score_cache": None,
                "affected_ids": set(int(x) for x in (repair_result.get("affected_subtask_ids", set()) or set())),
                "fallback_used": bool(repair_result.get("fallback_used", False)),
                "projection_mode": "",
                "projection_repaired_subtask_count": 0,
                "validation_signature": trial.validation_signature(),
                "selected_joint_strategy": str(strategy_name),
            }
    if best_result is not None:
        return best_result
    return {"success": False, "reason": "joint_z_repair_fail"}


def z_repair_same_stack_window(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_same_stack_window", allow_fallback=False, rng=rng)


def z_repair_bounded_detour_window(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_bounded_detour_window", allow_fallback=False, rng=rng)


def z_repair_sort_range_shrink_first(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_sort_range_shrink_first", allow_fallback=False, rng=rng)


def z_repair_gurobi_like_sort(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_gurobi_like_sort", allow_fallback=False, rng=rng)


def z_repair_mode_toggle_contextual(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_mode_toggle_contextual", allow_fallback=False, rng=rng)


def z_repair_flip_compact(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_flip_compact", allow_fallback=False, rng=rng)


def z_repair_joint_sort_colocated_flip(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_joint_sort_colocated_flip", allow_fallback=False, rng=rng)


def z_repair_spread_region_balance(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_spread_region_balance", allow_fallback=False, rng=rng)


def z_repair_load_balance_idle_robot(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_load_balance_idle_robot", allow_fallback=False, rng=rng)


def z_repair_cross_subtask_shared_stack(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_joint_sort_colocated_flip", allow_fallback=False, rng=rng)


def z_repair_multistack_cover_compact(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_gurobi_like_sort", allow_fallback=False, rng=rng)


def z_repair_stack_mode_joint_polish(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _apply_joint_z_repair_strategies(opt, config, dict(ctx or {}), rng=rng)


def z_repair_greedy_fallback(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_greedy_fallback", allow_fallback=True, rng=rng)


Z_DESTROY_OPERATORS = {
    "z_destroy_noise_window": z_destroy_noise_window,
    "z_destroy_multistack_window": z_destroy_multistack_window,
    "z_destroy_detour_window": z_destroy_detour_window,
    "z_destroy_mode_window": z_destroy_mode_window,
    "z_destroy_spread_hotspot_window": z_destroy_spread_hotspot_window,
    "z_destroy_random_window": z_destroy_random_window,
    "z_destroy_related_stack_window": z_destroy_related_stack_window,
    "z_destroy_shared_stack_window": z_destroy_shared_stack_window,
    "z_destroy_critical_path_window": z_destroy_critical_path_window,
}

Z_REPAIR_OPERATORS = {
    "z_repair_same_stack_window": z_repair_same_stack_window,
    "z_repair_bounded_detour_window": z_repair_bounded_detour_window,
    "z_repair_sort_range_shrink_first": z_repair_sort_range_shrink_first,
    "z_repair_gurobi_like_sort": z_repair_gurobi_like_sort,
    "z_repair_mode_toggle_contextual": z_repair_mode_toggle_contextual,
    "z_repair_flip_compact": z_repair_flip_compact,
    "z_repair_joint_sort_colocated_flip": z_repair_joint_sort_colocated_flip,
    "z_repair_spread_region_balance": z_repair_spread_region_balance,
    "z_repair_load_balance_idle_robot": z_repair_load_balance_idle_robot,
    "z_repair_cross_subtask_shared_stack": z_repair_cross_subtask_shared_stack,
    "z_repair_multistack_cover_compact": z_repair_multistack_cover_compact,
    "z_repair_stack_mode_joint_polish": z_repair_stack_mode_joint_polish,
}

Z_FALLBACK_OPERATOR = "z_repair_greedy_fallback"
