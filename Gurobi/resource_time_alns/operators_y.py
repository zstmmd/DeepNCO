from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config.ofs_config import OFSConfig

from .state import ResourceConfig, ResourceSubtask
from .utils import pick_ranked_candidate, pick_soft_greedy_min


def _validated_snapshot(opt):
    for candidate in [
        getattr(opt, "work", None),
        getattr(opt, "best", None),
        getattr(getattr(opt, "best_validated", None), "snapshot", None),
    ]:
        if candidate is not None and getattr(candidate, "subtask_state", None) is not None:
            return candidate
    return None


def _snapshot_subtasks(opt) -> List[object]:
    snapshot = _validated_snapshot(opt)
    if snapshot is None:
        return []
    return list(getattr(snapshot, "subtask_state", []) or [])


def _snapshot_station_idle_heads(opt) -> Dict[int, Dict[str, float]]:
    station_rows: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
    for subtask in _snapshot_subtasks(opt):
        station_id = int(getattr(subtask, "assigned_station_id", -1))
        rank = int(getattr(subtask, "station_sequence_rank", -1))
        if station_id < 0:
            continue
        task_rows = list(getattr(subtask, "execution_tasks", []) or [])
        if not task_rows:
            continue
        first_start = min(float(getattr(task, "start_process_time", 0.0) or 0.0) for task in task_rows)
        station_rows[station_id].append((rank if rank >= 0 else 10**9, float(first_start), int(getattr(subtask, "id", -1))))
    heads: Dict[int, Dict[str, float]] = {}
    for station_id, rows in station_rows.items():
        rows.sort(key=lambda item: (int(item[0]), float(item[1]), int(item[2])))
        rank, first_start, subtask_id = rows[0]
        heads[int(station_id)] = {
            "initial_idle_time": float(first_start),
            "subtask_id": int(subtask_id),
            "rank": int(rank if rank < 10**9 else -1),
        }
    return heads


def _snapshot_subtask_metrics(opt) -> Dict[int, Dict[str, object]]:
    rows: Dict[int, Dict[str, object]] = {}
    for subtask in _snapshot_subtasks(opt):
        subtask_id = int(getattr(subtask, "id", -1))
        task_rows = list(getattr(subtask, "execution_tasks", []) or [])
        if subtask_id < 0 or not task_rows:
            continue
        robot_ids = sorted({int(getattr(task, "robot_id", -1)) for task in task_rows if int(getattr(task, "robot_id", -1)) >= 0})
        assigned_robot_id = int(getattr(subtask, "assigned_robot_id", -1))
        if assigned_robot_id < 0 and robot_ids:
            assigned_robot_id = int(robot_ids[0])
        arrival_station = min(float(getattr(task, "arrival_time_at_station", 0.0) or 0.0) for task in task_rows)
        start_time = min(float(getattr(task, "start_process_time", 0.0) or 0.0) for task in task_rows)
        completion_time = max(float(getattr(task, "end_process_time", 0.0) or 0.0) for task in task_rows)
        first_stack_arrival = min(float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0) for task in task_rows)
        rows[int(subtask_id)] = {
            "subtask_id": int(subtask_id),
            "station_id": int(getattr(subtask, "assigned_station_id", -1)),
            "station_rank": int(getattr(subtask, "station_sequence_rank", -1)),
            "assigned_robot_id": int(assigned_robot_id),
            "arrival_station": float(arrival_station),
            "start_time": float(start_time),
            "completion_time": float(completion_time),
            "arrival_stack": float(first_stack_arrival),
        }
    return rows


def _snapshot_robot_chains(opt) -> Dict[int, List[int]]:
    per_robot: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    for subtask in _snapshot_subtasks(opt):
        subtask_id = int(getattr(subtask, "id", -1))
        task_rows = list(getattr(subtask, "execution_tasks", []) or [])
        if subtask_id < 0 or not task_rows:
            continue
        grouped: Dict[int, List[object]] = defaultdict(list)
        for task in task_rows:
            robot_id = int(getattr(task, "robot_id", -1))
            if robot_id >= 0:
                grouped[int(robot_id)].append(task)
        for robot_id, rows in grouped.items():
            per_robot[int(robot_id)].append(
                (
                    min(float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0) for task in rows),
                    min(float(getattr(task, "arrival_time_at_station", 0.0) or 0.0) for task in rows),
                    int(subtask_id),
                )
            )
    chains: Dict[int, List[int]] = {}
    for robot_id, rows in per_robot.items():
        rows.sort(key=lambda item: (float(item[0]), float(item[1]), int(item[2])))
        ordered: List[int] = []
        for _, _, subtask_id in rows:
            if int(subtask_id) not in ordered:
                ordered.append(int(subtask_id))
        chains[int(robot_id)] = ordered
    return chains


def _snapshot_station_chains(opt) -> Dict[int, List[int]]:
    station_rows: Dict[int, List[Tuple[int, float, float, int]]] = defaultdict(list)
    for subtask_id, meta in _snapshot_subtask_metrics(opt).items():
        station_id = int(meta.get("station_id", -1))
        if station_id < 0:
            continue
        station_rows[int(station_id)].append(
            (
                int(meta.get("station_rank", -1) if int(meta.get("station_rank", -1)) >= 0 else 10**9),
                float(meta.get("start_time", 0.0)),
                float(meta.get("completion_time", 0.0)),
                int(subtask_id),
            )
        )
    chains: Dict[int, List[int]] = {}
    for station_id, rows in station_rows.items():
        rows.sort(key=lambda item: (int(item[0]), float(item[1]), float(item[2]), int(item[3])))
        chains[int(station_id)] = [int(item[3]) for item in rows]
    return chains


def _robot_start_positions(opt) -> Dict[int, Tuple[float, float]]:
    problem = getattr(opt, "problem", None)
    rows: Dict[int, Tuple[float, float]] = {}
    for robot in getattr(problem, "robot_list", []) or []:
        point = getattr(robot, "start_point", None)
        robot_id = int(getattr(robot, "id", -1))
        if point is None or robot_id < 0:
            continue
        rows[int(robot_id)] = (float(point.x), float(point.y))
    return rows


def _snapshot_robot_frontier(opt) -> Dict[int, Dict[str, float]]:
    metrics = _snapshot_subtask_metrics(opt)
    chains = _snapshot_robot_chains(opt)
    stations = list(getattr(getattr(opt, "problem", None), "station_list", []) or [])
    frontiers: Dict[int, Dict[str, float]] = {}
    start_positions = _robot_start_positions(opt)
    for robot_id, xy in start_positions.items():
        frontiers[int(robot_id)] = {
            "available_time": 0.0,
            "x": float(xy[0]),
            "y": float(xy[1]),
            "assigned_count": 0.0,
        }
    for robot_id, subtask_ids in chains.items():
        if not subtask_ids:
            continue
        last_subtask_id = int(subtask_ids[-1])
        meta = dict(metrics.get(int(last_subtask_id), {}) or {})
        station_id = int(meta.get("station_id", -1))
        if not (0 <= station_id < len(stations)):
            continue
        point = stations[int(station_id)].point
        frontier = frontiers.setdefault(
            int(robot_id),
            {"available_time": 0.0, "x": float(point.x), "y": float(point.y), "assigned_count": 0.0},
        )
        frontier["available_time"] = float(meta.get("completion_time", 0.0))
        frontier["x"] = float(point.x)
        frontier["y"] = float(point.y)
        frontier["assigned_count"] = float(len(list(subtask_ids)))
    return frontiers


def _rank_released_subtasks(
    released_ids: Sequence[int],
    priority_ids: Sequence[int],
    metrics: Dict[int, Dict[str, object]],
) -> List[int]:
    priority_rank = {int(subtask_id): idx for idx, subtask_id in enumerate(int(x) for x in (priority_ids or []))}
    rows = sorted(
        {int(x) for x in (released_ids or []) if int(x) >= 0},
        key=lambda subtask_id: (
            int(priority_rank.get(int(subtask_id), 10**9)),
            -float(metrics.get(int(subtask_id), {}).get("completion_time", 0.0)),
            -float(metrics.get(int(subtask_id), {}).get("start_time", 0.0)),
            int(subtask_id),
        ),
    )
    return list(rows)


def _finalize_release_ids(
    config: ResourceConfig,
    release_ids: Sequence[int],
    degree: int,
    priority_ids: Sequence[int],
    preview: bool,
    metrics: Optional[Dict[int, Dict[str, object]]] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    ordered_ids = _rank_released_subtasks(release_ids, priority_ids, dict(metrics or {}))
    capped_ids = [int(subtask_id) for subtask_id in ordered_ids[: max(1, int(degree))] if int(subtask_id) in config.subtasks]
    if not capped_ids:
        return {"success": False}
    rows = [config.subtasks[int(subtask_id)] for subtask_id in capped_ids if int(subtask_id) in config.subtasks]
    released = _preview_release_rows(rows, len(rows)) if bool(preview) else _release_rows(rows, len(rows))
    payload = {
        "success": bool(released),
        "released_subtasks": released,
        "source_station_ids": sorted({int(station_id) for station_id, _ in released.values() if int(station_id) >= 0}),
    }
    payload.update(dict(extra or {}))
    return payload


def _station_count(opt) -> int:
    return max(1, len(getattr(getattr(opt, "problem", None), "station_list", []) or []))


def _station_loads(config: ResourceConfig) -> Dict[int, float]:
    loads: Dict[int, float] = defaultdict(float)
    for row in config.subtasks.values():
        if int(row.station_id) < 0:
            continue
        loads[int(row.station_id)] += float(_subtask_station_work(row))
    return dict(loads)


def _subtask_station_work(subtask: ResourceSubtask) -> float:
    total = 0.0
    for task in subtask.z_tasks or []:
        total += float(max(1, int(task.sku_pick_count or 0))) * float(getattr(OFSConfig, "PICKING_TIME", 1.0))
        total += float(task.station_service_time)
    return float(total)


def _arrival_proxy(opt, subtask: ResourceSubtask, station_id: int) -> float:
    problem = getattr(opt, "problem", None)
    if problem is None or not (0 <= int(station_id) < len(getattr(problem, "station_list", []) or [])):
        return 1e6
    rows = []
    for task in subtask.z_tasks or []:
        stack = problem.point_to_stack.get(int(task.stack_id))
        if stack is None:
            continue
        station = problem.station_list[int(station_id)]
        rows.append(abs(float(stack.store_point.x) - float(station.point.x)) + abs(float(stack.store_point.y) - float(station.point.y)))
    return float(sum(rows) / max(1, len(rows))) if rows else 0.0


def _manhattan_distance_xy(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    return float(abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1])))


def _station_xy(opt, station_id: int) -> Optional[Tuple[float, float]]:
    stations = list(getattr(getattr(opt, "problem", None), "station_list", []) or [])
    if not (0 <= int(station_id) < len(stations)):
        return None
    point = stations[int(station_id)].point
    return float(point.x), float(point.y)


def _subtask_candidate_stack_ids(opt, config: ResourceConfig, subtask: ResourceSubtask) -> List[int]:
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
    return stack_ids


def _robot_availability_proxy(opt, robot_frontier: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[int, Dict[str, float]]:
    frontier = dict(robot_frontier or {})
    if frontier:
        return frontier
    return _snapshot_robot_frontier(opt)


def _subtask_pick_count(config: ResourceConfig, subtask: ResourceSubtask) -> int:
    pick_count = 0
    for descriptor in subtask.z_tasks or []:
        pick_count += max(0, int(getattr(descriptor, "sku_pick_count", 0) or 0))
    if pick_count > 0:
        return int(pick_count)
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        pick_count += max(0, int(getattr(work_unit, "units", 0) or 0))
    return int(max(1, pick_count))


def _subtask_heaviness_penalty(opt, config: ResourceConfig, subtask: ResourceSubtask) -> float:
    has_sort = any(str(getattr(descriptor, "mode", "")).upper() == "SORT" for descriptor in (subtask.z_tasks or []))
    sort_penalty = float(getattr(opt.cfg, "resource_y_wave1_sort_penalty", 8.0)) if has_sort else 0.0
    pick_weight = float(getattr(opt.cfg, "resource_y_wave1_pick_weight", 0.75))
    pick_penalty = max(0, _subtask_pick_count(config, subtask) - 1) * pick_weight
    return float(sort_penalty + pick_penalty)


def _robot_station_arrival_cost(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    station_id: int,
    robot_id: int,
    robot_frontier: Optional[Dict[int, Dict[str, float]]] = None,
) -> float:
    station_xy = _station_xy(opt, int(station_id))
    frontier = _robot_availability_proxy(opt, robot_frontier)
    robot_meta = dict(frontier.get(int(robot_id), {}) or {})
    if station_xy is None or not robot_meta:
        return float(_arrival_proxy(opt, subtask, int(station_id)))
    candidate_stack_ids = _subtask_candidate_stack_ids(opt, config, subtask)
    stack_points: List[Tuple[int, Tuple[float, float]]] = []
    for stack_id in candidate_stack_ids:
        xy = getattr(opt, "_stack_xy", lambda *_args, **_kwargs: None)(int(stack_id))
        if xy is None:
            stack = getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id))
            if stack is not None:
                xy = (float(stack.store_point.x), float(stack.store_point.y))
        if xy is None:
            continue
        stack_points.append((int(stack_id), (float(xy[0]), float(xy[1]))))
    if not stack_points:
        return float(_arrival_proxy(opt, subtask, int(station_id)))
    robot_xy = (float(robot_meta.get("x", 0.0)), float(robot_meta.get("y", 0.0)))
    ready = float(robot_meta.get("available_time", 0.0))
    stack_leg = min(
        _manhattan_distance_xy(robot_xy, stack_xy) + _manhattan_distance_xy(stack_xy, station_xy)
        for _, stack_xy in stack_points
    )
    return float(ready + stack_leg)


def _triangle_arrival_cost(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    station_id: int,
    robot_frontier: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[float, int]:
    station_xy = _station_xy(opt, int(station_id))
    if station_xy is None:
        return float(_arrival_proxy(opt, subtask, int(station_id))), -1
    candidate_stack_ids = _subtask_candidate_stack_ids(opt, config, subtask)
    stack_points: List[Tuple[int, Tuple[float, float]]] = []
    for stack_id in candidate_stack_ids:
        xy = getattr(opt, "_stack_xy", lambda *_args, **_kwargs: None)(int(stack_id))
        if xy is None:
            stack = getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id))
            if stack is not None:
                xy = (float(stack.store_point.x), float(stack.store_point.y))
        if xy is None:
            continue
        stack_points.append((int(stack_id), (float(xy[0]), float(xy[1]))))
    if not stack_points:
        return float(_arrival_proxy(opt, subtask, int(station_id))), -1
    frontier = _robot_availability_proxy(opt, robot_frontier)
    if not frontier:
        return float(_arrival_proxy(opt, subtask, int(station_id))), -1
    best_cost = float("inf")
    best_robot_id = -1
    for robot_id, meta in frontier.items():
        total = _robot_station_arrival_cost(opt, config, subtask, int(station_id), int(robot_id), robot_frontier=frontier)
        total += float(getattr(opt.cfg, "resource_y_robot_available_time_weight", 0.25)) * float(meta.get("available_time", 0.0) or 0.0)
        total += float(getattr(opt.cfg, "resource_y_robot_count_penalty", 6.0)) * float(meta.get("assigned_count", 0.0) or 0.0)
        if (float(total), int(robot_id)) < (float(best_cost), int(best_robot_id if best_robot_id >= 0 else 10**9)):
            best_cost = float(total)
            best_robot_id = int(robot_id)
    if best_robot_id < 0:
        return float(_arrival_proxy(opt, subtask, int(station_id))), -1
    return float(best_cost), int(best_robot_id)


def _grid_cell(x: float, y: float, grid_size: float) -> Tuple[int, int]:
    size = max(1.0, float(grid_size))
    return int(math.floor(float(x) / size)), int(math.floor(float(y) / size))


def _manhattan_cells(x0: float, y0: float, x1: float, y1: float, grid_size: float) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    size = max(1.0, float(grid_size))
    x_step = 1 if x1 >= x0 else -1
    y_step = 1 if y1 >= y0 else -1
    x = float(x0)
    while (x_step > 0 and x <= float(x1)) or (x_step < 0 and x >= float(x1)):
        cell = _grid_cell(x, y0, size)
        if cell not in cells:
            cells.append(cell)
        x += float(x_step) * size
    y = float(y0)
    while (y_step > 0 and y <= float(y1)) or (y_step < 0 and y >= float(y1)):
        cell = _grid_cell(x1, y, size)
        if cell not in cells:
            cells.append(cell)
        y += float(y_step) * size
    end_cell = _grid_cell(x1, y1, size)
    if end_cell not in cells:
        cells.append(end_cell)
    return cells


def _stack_station_cells(opt, stack_id: int, station_id: int, grid_size: float) -> List[Tuple[int, int]]:
    problem = getattr(opt, "problem", None)
    if problem is None:
        return []
    stack = getattr(problem, "point_to_stack", {}).get(int(stack_id))
    stations = list(getattr(problem, "station_list", []) or [])
    if stack is None or not (0 <= int(station_id) < len(stations)):
        return []
    stack_pt = stack.store_point
    station_pt = stations[int(station_id)].point
    return _manhattan_cells(float(stack_pt.x), float(stack_pt.y), float(station_pt.x), float(station_pt.y), float(grid_size))


def _subtask_route_cells(opt, subtask: ResourceSubtask, station_id: int, grid_size: float) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    stack_ids = sorted({int(task.stack_id) for task in (subtask.z_tasks or []) if int(task.stack_id) >= 0})
    for stack_id in stack_ids:
        for cell in _stack_station_cells(opt, int(stack_id), int(station_id), float(grid_size)):
            if cell not in cells:
                cells.append(cell)
    return cells


def _station_neighbor_robot_budget(opt, station_id: int, grid_size: float) -> int:
    problem = getattr(opt, "problem", None)
    if problem is None:
        return 1
    stations = list(getattr(problem, "station_list", []) or [])
    if not (0 <= int(station_id) < len(stations)):
        return 1
    station_pt = stations[int(station_id)].point
    target = _grid_cell(float(station_pt.x), float(station_pt.y), float(grid_size))
    budget = 0
    for robot in getattr(problem, "robot_list", []) or []:
        start_point = getattr(robot, "start_point", None)
        if start_point is None:
            continue
        cell = _grid_cell(float(start_point.x), float(start_point.y), float(grid_size))
        if abs(int(cell[0]) - int(target[0])) <= 1 and abs(int(cell[1]) - int(target[1])) <= 1:
            budget += 1
    return max(1, int(budget or len(getattr(problem, "robot_list", []) or []) or 1))


def _subtask_station_window(subtask: ResourceSubtask, station_load_before: float, arrival_proxy: float) -> Tuple[float, float]:
    start = float(max(0.0, station_load_before) + max(0.0, arrival_proxy))
    end = float(start + _subtask_station_work(subtask))
    return float(start), float(end)


def _time_bucket_ids(start: float, end: float, bucket_sec: float) -> List[int]:
    width = max(1.0, float(bucket_sec))
    lo = int(math.floor(float(start) / width))
    hi = int(math.floor(max(float(start), float(end) - 1e-9) / width))
    return list(range(int(lo), int(hi) + 1))


def _transient_conflict_constraints(config: ResourceConfig) -> Dict[str, object]:
    return dict((getattr(config, "metadata", {}) or {}).get("transient_conflict_constraints", {}) or {})


def _existing_subtask_windows(config: ResourceConfig, station_id: int, released_ids: Optional[Sequence[int]] = None) -> List[Tuple[int, float, float, List[Tuple[int, int]]]]:
    released = {int(x) for x in (released_ids or [])}
    windows: List[Tuple[int, float, float, List[Tuple[int, int]]]] = []
    station_rows = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) == int(station_id) and int(row.subtask_id) not in released],
        key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)),
    )
    load_before = 0.0
    for row in station_rows:
        arrival = 0.0
        start, end = _subtask_station_window(row, load_before, arrival)
        windows.append((int(row.subtask_id), float(start), float(end), []))
        load_before = float(end)
    return windows


def _station_choice_penalties(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    station_id: int,
    loads: Dict[int, float],
    released_ids: Optional[Sequence[int]] = None,
) -> Tuple[float, float, float]:
    grid_size = max(1.0, float(getattr(opt.cfg, "resource_y_heat_grid_size", 4.0)))
    bucket_sec = max(1.0, float(getattr(opt.cfg, "resource_y_time_bucket_sec", 20.0)))
    route_cells = _subtask_route_cells(opt, subtask, int(station_id), grid_size)
    heat_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    existing_windows = []
    released = {int(x) for x in (released_ids or [])}
    for row in config.subtasks.values():
        if int(row.subtask_id) in released:
            continue
        if int(row.station_id) < 0:
            continue
        for cell in _subtask_route_cells(opt, row, int(row.station_id), grid_size):
            heat_counts[cell] += 1
        if int(row.station_id) == int(station_id):
            start, end = _subtask_station_window(row, float(loads.get(int(station_id), 0.0)), 0.0)
            existing_windows.append((float(start), float(end), _subtask_route_cells(opt, row, int(station_id), grid_size)))
    heat_penalty = float(sum(heat_counts.get(cell, 0) for cell in route_cells))
    arrival = float(_arrival_proxy(opt, subtask, int(station_id)))
    candidate_start, candidate_end = _subtask_station_window(subtask, float(loads.get(int(station_id), 0.0)), arrival)
    candidate_buckets = set(_time_bucket_ids(candidate_start, candidate_end, bucket_sec))
    overlap_penalty = 0.0
    for start, end, cells in existing_windows:
        overlap_buckets = candidate_buckets & set(_time_bucket_ids(float(start), float(end), bucket_sec))
        if not overlap_buckets:
            continue
        shared_cells = len(set(route_cells) & set(cells)) if route_cells and cells else 0
        overlap_penalty += float(len(overlap_buckets) * max(1, shared_cells or 1))
    nearby_budget = int(_station_neighbor_robot_budget(opt, int(station_id), grid_size))
    concurrent_demand = int(len(existing_windows) + 1)
    budget_penalty = float(max(0, concurrent_demand - nearby_budget))
    conflict = _transient_conflict_constraints(config)
    if int(station_id) in set(int(x) for x in (conflict.get("station_ids", []) or [])):
        heat_penalty += float(getattr(opt.cfg, "resource_y_conflict_station_penalty", 12.0))
    if candidate_buckets & set(int(x) for x in (conflict.get("time_bucket_ids", []) or [])):
        overlap_penalty += float(getattr(opt.cfg, "resource_y_conflict_time_bucket_penalty", 8.0))
    if route_cells and set(route_cells) & {tuple(cell) for cell in (conflict.get("route_cells", []) or [])}:
        heat_penalty += float(getattr(opt.cfg, "resource_y_conflict_heat_bonus", 4.0))
    return float(heat_penalty), float(overlap_penalty), float(budget_penalty)


def _station_choice_cost(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    station_id: int,
    loads: Dict[int, float],
    released_ids: Optional[Sequence[int]] = None,
    counts: Optional[Dict[int, int]] = None,
    assignment_index: int = 0,
    first_batch_limit: int = 0,
    used_station_ids: Optional[Set[int]] = None,
    used_robot_ids: Optional[Set[int]] = None,
    robot_frontier: Optional[Dict[int, Dict[str, float]]] = None,
) -> float:
    projected_finish = float(loads.get(int(station_id), 0.0) + _subtask_station_work(subtask))
    arrival = float(_arrival_proxy(opt, subtask, station_id))
    triangle_arrival, selected_robot_id = _triangle_arrival_cost(
        opt,
        config,
        subtask,
        int(station_id),
        robot_frontier=robot_frontier,
    )
    heat_penalty, overlap_penalty, budget_penalty = _station_choice_penalties(
        opt,
        config,
        subtask,
        int(station_id),
        loads,
        released_ids=released_ids,
    )
    counts_map = dict(counts or {})
    used_stations = set(int(x) for x in (used_station_ids or set()) if int(x) >= 0)
    used_robots = set(int(x) for x in (used_robot_ids or set()) if int(x) >= 0)
    station_count = _station_count(opt)
    is_starving = bool(int(counts_map.get(int(station_id), 0)) <= 0)
    is_first_batch = bool(int(assignment_index) < max(0, int(first_batch_limit)))
    arrival_term = float(triangle_arrival if is_starving else arrival)
    projected_finish_weight = 1.0
    if is_starving:
        projected_finish_weight = float(getattr(opt.cfg, "resource_y_triangle_projected_finish_weight", 0.35))
    starvation_bonus = float(getattr(opt.cfg, "resource_y_starvation_bonus", 8.0)) if is_starving else 0.0
    triangle_weight = float(getattr(opt.cfg, "resource_y_triangle_arrival_weight", 1.0))
    spread_penalty = 0.0
    robot_penalty = 0.0
    if is_first_batch:
        if len(used_stations) < int(station_count) and int(station_id) in used_stations:
            spread_penalty += float(getattr(opt.cfg, "resource_y_first_batch_station_penalty", 40.0))
        if int(selected_robot_id) >= 0 and int(selected_robot_id) in used_robots:
            robot_penalty += float(getattr(opt.cfg, "resource_y_first_batch_robot_penalty", 18.0))
        if is_starving:
            starvation_bonus += float(getattr(opt.cfg, "resource_y_first_batch_starvation_bonus", 6.0))
    return float(
        projected_finish_weight * projected_finish
        + 0.1 * triangle_weight * arrival_term
        + float(getattr(opt.cfg, "resource_y_heat_penalty_weight", 0.75)) * heat_penalty
        + float(getattr(opt.cfg, "resource_y_window_overlap_weight", 0.60)) * overlap_penalty
        + float(getattr(opt.cfg, "resource_y_nearby_robot_budget_weight", 0.80)) * budget_penalty
        + float(spread_penalty)
        + float(robot_penalty)
        - float(starvation_bonus)
    )


def _normalize_ranks(config: ResourceConfig) -> None:
    config.normalize_station_ranks()


def _release_rows(rows: List[ResourceSubtask], move_n: int) -> Dict[int, Tuple[int, int]]:
    released = {}
    for row in list(rows or [])[: max(0, int(move_n))]:
        released[int(row.subtask_id)] = (int(row.station_id), int(row.station_rank))
        row.station_id = -1
        row.station_rank = -1
    return released


def _preview_release_rows(rows: List[ResourceSubtask], move_n: int) -> Dict[int, Tuple[int, int]]:
    released = {}
    for row in list(rows or [])[: max(0, int(move_n))]:
        released[int(row.subtask_id)] = (int(row.station_id), int(row.station_rank))
    return released


def _normalize_station_ranks_subset(config: ResourceConfig, station_ids: List[int]) -> None:
    touched = {int(x) for x in (station_ids or []) if int(x) >= 0}
    if not touched:
        return
    for station_id in sorted(touched):
        rows = [row for row in config.subtasks.values() if int(row.station_id) == int(station_id)]
        rows.sort(key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)))
        for rank, row in enumerate(rows):
            row.station_rank = int(rank)


def y_destroy_congested_station_block(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    loads = _station_loads(config)
    if not loads:
        return {"success": False}
    ranked = sorted(
        [((-float(loads[sid]), int(sid)), int(sid)) for sid in loads.keys()],
        key=lambda item: item[0],
    )
    picked = pick_ranked_candidate(rng, ranked, opt.cfg)
    if picked is None:
        return {"success": False}
    _, heavy_station = picked
    rows = sorted(config.station_subtasks(int(heavy_station)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    if not rows:
        return {"success": False}
    move_n = max(1, min(int(degree), len(rows)))
    chosen = list(reversed(rows))[:move_n]
    released = _release_rows(chosen, move_n)
    return {"success": True, "released_subtasks": released}


def y_destroy_cross_station_fragment(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    order_station_map: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        order_station_map[int(row.order_id)].append(row)
    candidate_rows = []
    for order_id, rows in order_station_map.items():
        stations = {int(row.station_id) for row in rows if int(row.station_id) >= 0}
        if len(stations) >= 2:
            candidate_rows.append(((-len(stations), -len(rows), int(order_id)), int(order_id)))
    if not candidate_rows:
        return {"success": False}
    picked = pick_ranked_candidate(rng, sorted(candidate_rows, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False}
    _, target_order = picked
    rows = sorted(order_station_map[int(target_order)], key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    move_n = max(1, min(int(degree), len(rows)))
    released = _release_rows(rows[:move_n], move_n)
    return {"success": True, "released_subtasks": released}


def y_destroy_rank_window_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) >= 0],
        key=lambda row: (-int(row.station_rank), int(row.subtask_id)),
    )
    if not ranked:
        return {"success": False}
    candidates = [((float(-int(row.station_rank)), int(row.subtask_id)), int(row.subtask_id)) for row in ranked]
    picked = pick_ranked_candidate(rng, candidates, opt.cfg)
    if picked is None:
        return {"success": False}
    _, center_subtask_id = picked
    center_row = config.subtasks.get(int(center_subtask_id))
    if center_row is None or int(center_row.station_id) < 0:
        return {"success": False}
    station_rows = sorted(config.station_subtasks(int(center_row.station_id)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    center_idx = next((idx for idx, row in enumerate(station_rows) if int(row.subtask_id) == int(center_subtask_id)), 0)
    move_n = max(1, min(int(degree), len(station_rows)))
    start = max(0, int(center_idx) - move_n // 2)
    end = min(len(station_rows), start + move_n)
    start = max(0, end - move_n)
    released = _release_rows(station_rows[start:end], move_n)
    return {"success": True, "released_subtasks": released}


def y_destroy_load_skew_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return y_destroy_congested_station_block(opt, config, rng, degree)


def _random_release_payload(config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    rows = [row for row in config.subtasks.values() if int(row.station_id) >= 0]
    if not rows:
        return {"success": False}
    rng.shuffle(rows)
    move_n = max(1, min(int(degree), len(rows)))
    released = _preview_release_rows(rows[:move_n], move_n) if bool(preview) else _release_rows(rows[:move_n], move_n)
    return {
        "success": bool(released),
        "released_subtasks": released,
        "source_station_ids": sorted({int(station_id) for station_id, _ in released.values() if int(station_id) >= 0}),
        "trigger_reason": "random_release",
    }


def y_destroy_random_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del opt
    return _random_release_payload(config, rng, degree, preview=False)


def _related_station_release_payload(config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    station_rows: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        if int(row.station_id) >= 0:
            station_rows[int(row.station_id)].append(row)
    candidates = [
        ((-float(len(rows)), int(station_id)), int(station_id))
        for station_id, rows in station_rows.items()
        if rows
    ]
    if not candidates:
        return {"success": False}
    picked = pick_ranked_candidate(rng, sorted(candidates, key=lambda item: item[0]), None)
    if picked is None:
        return {"success": False}
    _, station_id = picked
    rows = sorted(station_rows[int(station_id)], key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    move_n = max(1, min(int(degree), len(rows)))
    released = _preview_release_rows(rows[-move_n:], move_n) if bool(preview) else _release_rows(rows[-move_n:], move_n)
    return {
        "success": bool(released),
        "released_subtasks": released,
        "source_station_ids": [int(station_id)],
        "trigger_reason": "related_station_release",
    }


def y_destroy_related_station_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del opt
    return _related_station_release_payload(config, rng, degree, preview=False)


def _related_robot_release_payload(opt, config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    metrics = _snapshot_subtask_metrics(opt)
    robot_chains = _snapshot_robot_chains(opt)
    if not metrics or not robot_chains:
        return _random_release_payload(config, rng, degree, preview=preview)
    candidates = []
    for robot_id, chain in robot_chains.items():
        ids = [int(subtask_id) for subtask_id in (chain or []) if int(subtask_id) in config.subtasks]
        if not ids:
            continue
        finish = max(float(metrics.get(int(subtask_id), {}).get("completion_time", 0.0)) for subtask_id in ids)
        candidates.append(((-float(finish), -float(len(ids)), int(robot_id)), int(robot_id), ids))
    if not candidates:
        return _random_release_payload(config, rng, degree, preview=preview)
    candidates.sort(key=lambda item: item[0])
    _, robot_id, chain = candidates[0]
    release_ids = list(reversed(chain))[: max(1, int(degree))]
    return _finalize_release_ids(
        config=config,
        release_ids=release_ids,
        degree=int(degree),
        priority_ids=release_ids,
        metrics=metrics,
        preview=bool(preview),
        extra={"source_robot_ids": [int(robot_id)], "trigger_reason": "related_robot_release"},
    )


def y_destroy_related_robot_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _related_robot_release_payload(opt, config, rng, int(degree), preview=False)


def _heavy_robot_release_payload(opt, config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    metrics = _snapshot_subtask_metrics(opt)
    robot_chains = _snapshot_robot_chains(opt)
    if not metrics or not robot_chains:
        fallback = y_plan_destroy_load_skew_release if bool(preview) else y_destroy_load_skew_release
        return fallback(opt, config, rng, degree)
    robot_scores = []
    for robot_id, chain in robot_chains.items():
        task_ids = [int(subtask_id) for subtask_id in (chain or []) if int(subtask_id) in config.subtasks]
        if not task_ids:
            continue
        finish = max(float(metrics.get(int(subtask_id), {}).get("completion_time", 0.0)) for subtask_id in task_ids)
        work = sum(float(_subtask_station_work(config.subtasks[int(subtask_id)])) for subtask_id in task_ids)
        robot_scores.append(((-float(finish), -float(work), -len(task_ids), int(robot_id)), int(robot_id), task_ids))
    if not robot_scores:
        fallback = y_plan_destroy_load_skew_release if bool(preview) else y_destroy_load_skew_release
        return fallback(opt, config, rng, degree)
    robot_scores.sort(key=lambda item: item[0])
    _, heavy_robot_id, chain = robot_scores[0]
    tail = list(reversed(chain))
    release_ids = tail[: max(1, int(degree))]
    if len(release_ids) < max(1, int(degree)):
        for subtask_id in chain:
            if int(subtask_id) not in release_ids:
                release_ids.append(int(subtask_id))
            if len(release_ids) >= max(1, int(degree)):
                break
    return _finalize_release_ids(
        config=config,
        release_ids=release_ids,
        degree=int(degree),
        priority_ids=release_ids,
        metrics=metrics,
        preview=bool(preview),
        extra={
            "source_robot_ids": [int(heavy_robot_id)],
            "trigger_reason": "heavy_robot_unload",
        },
    )


def y_destroy_heavy_robot_tail(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _heavy_robot_release_payload(opt, config, rng, int(degree), preview=False)


def _initial_idle_release_payload(opt, config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    heads = _snapshot_station_idle_heads(opt)
    metrics = _snapshot_subtask_metrics(opt)
    robot_chains = _snapshot_robot_chains(opt)
    if not heads or not metrics:
        fallback = y_plan_destroy_congested_station_block if bool(preview) else y_destroy_congested_station_block
        return fallback(opt, config, rng, degree)
    target = max(heads.items(), key=lambda item: (float(item[1].get("initial_idle_time", 0.0)), -int(item[0])))
    station_id = int(target[0])
    anchor_subtask_id = int(target[1].get("subtask_id", -1))
    anchor_meta = metrics.get(int(anchor_subtask_id), {})
    robot_id = int(anchor_meta.get("assigned_robot_id", -1))
    if int(anchor_subtask_id) not in config.subtasks or robot_id < 0:
        fallback = y_plan_destroy_congested_station_block if bool(preview) else y_destroy_congested_station_block
        return fallback(opt, config, rng, degree)
    chain = list(robot_chains.get(int(robot_id), []) or [])
    release_ids: List[int] = []
    if int(anchor_subtask_id) in chain:
        anchor_idx = chain.index(int(anchor_subtask_id))
        release_ids.extend(int(subtask_id) for subtask_id in chain[: anchor_idx + 1])
    else:
        release_ids.append(int(anchor_subtask_id))
    station_chain = list(_snapshot_station_chains(opt).get(int(station_id), []) or [])
    for subtask_id in station_chain:
        if int(subtask_id) == int(anchor_subtask_id):
            break
        if int(subtask_id) not in release_ids:
            release_ids.append(int(subtask_id))
        if len(release_ids) >= max(1, int(degree)):
            break
    return _finalize_release_ids(
        config=config,
        release_ids=release_ids,
        degree=int(degree),
        priority_ids=[int(anchor_subtask_id)] + [int(subtask_id) for subtask_id in chain if int(subtask_id) != int(anchor_subtask_id)],
        metrics=metrics,
        preview=bool(preview),
        extra={
            "source_station_ids": [int(station_id)],
            "source_robot_ids": [int(robot_id)],
            "trigger_reason": "initial_idle_head",
        },
    )


def y_destroy_initial_idle_head(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _initial_idle_release_payload(opt, config, rng, int(degree), preview=False)


def _max_tardiness_release_payload(opt, config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    metrics = _snapshot_subtask_metrics(opt)
    if not metrics:
        fallback = y_plan_destroy_rank_window_release if bool(preview) else y_destroy_rank_window_release
        return fallback(opt, config, rng, degree)
    station_chains = _snapshot_station_chains(opt)
    robot_chains = _snapshot_robot_chains(opt)
    ordered_targets = sorted(
        metrics.values(),
        key=lambda item: (-float(item.get("completion_time", 0.0)), -float(item.get("arrival_station", 0.0)), int(item.get("subtask_id", -1))),
    )
    target_count = max(1, int(math.ceil(0.10 * len(ordered_targets))))
    target_ids = [int(item.get("subtask_id", -1)) for item in ordered_targets[:target_count] if int(item.get("subtask_id", -1)) in config.subtasks]
    if not target_ids:
        fallback = y_plan_destroy_rank_window_release if bool(preview) else y_destroy_rank_window_release
        return fallback(opt, config, rng, degree)
    release_ids: List[int] = []
    priority_ids: List[int] = []
    source_station_ids: Set[int] = set()
    source_robot_ids: Set[int] = set()
    for target_subtask_id in target_ids:
        meta = metrics.get(int(target_subtask_id), {})
        station_id = int(meta.get("station_id", -1))
        robot_id = int(meta.get("assigned_robot_id", -1))
        arrival_station = float(meta.get("arrival_station", 0.0))
        start_time = float(meta.get("start_time", 0.0))
        if int(target_subtask_id) not in release_ids:
            release_ids.append(int(target_subtask_id))
        if int(target_subtask_id) not in priority_ids:
            priority_ids.append(int(target_subtask_id))
        if station_id >= 0:
            source_station_ids.add(int(station_id))
        if robot_id >= 0:
            source_robot_ids.add(int(robot_id))
        if robot_id >= 0:
            chain = list(robot_chains.get(int(robot_id), []) or [])
            if int(target_subtask_id) in chain:
                anchor_idx = chain.index(int(target_subtask_id))
                preceding = list(reversed(chain[:anchor_idx]))
                for subtask_id in preceding:
                    if int(subtask_id) not in release_ids:
                        release_ids.append(int(subtask_id))
                    if int(subtask_id) not in priority_ids:
                        priority_ids.append(int(subtask_id))
        if station_id >= 0:
            station_chain = list(station_chains.get(int(station_id), []) or [])
            for subtask_id in reversed(station_chain):
                if int(subtask_id) == int(target_subtask_id):
                    continue
                blocker = metrics.get(int(subtask_id), {})
                if int(blocker.get("station_rank", -1)) >= int(meta.get("station_rank", -1)):
                    continue
                blocker_completion = float(blocker.get("completion_time", 0.0))
                if blocker_completion + 1e-9 < min(arrival_station, start_time):
                    continue
                if int(subtask_id) not in release_ids:
                    release_ids.append(int(subtask_id))
            if int(target_subtask_id) in station_chain:
                idx = station_chain.index(int(target_subtask_id))
                if idx > 0:
                    immediate = int(station_chain[idx - 1])
                    if immediate not in release_ids:
                        release_ids.append(immediate)
    if not release_ids:
        fallback = y_plan_destroy_rank_window_release if bool(preview) else y_destroy_rank_window_release
        return fallback(opt, config, rng, degree)
    return _finalize_release_ids(
        config=config,
        release_ids=release_ids,
        degree=int(degree),
        priority_ids=priority_ids,
        metrics=metrics,
        preview=bool(preview),
        extra={
            "source_station_ids": sorted(int(x) for x in source_station_ids),
            "source_robot_ids": sorted(int(x) for x in source_robot_ids),
            "trigger_reason": "max_tardiness_blocker",
            "target_subtask_ids": list(target_ids),
        },
    )


def y_destroy_max_tardiness_blocker(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _max_tardiness_release_payload(opt, config, rng, int(degree), preview=False)


def _critical_path_release_payload(opt, config: ResourceConfig, rng, degree: int, preview: bool) -> Dict[str, object]:
    metrics = _snapshot_subtask_metrics(opt)
    if not metrics:
        return _heavy_robot_release_payload(opt, config, rng, int(degree), preview=bool(preview))
    ranked = sorted(
        [
            (
                (
                    -float(meta.get("completion_time", 0.0)),
                    -float(meta.get("arrival_station", 0.0)),
                    int(subtask_id),
                ),
                int(subtask_id),
            )
            for subtask_id, meta in metrics.items()
            if int(subtask_id) in config.subtasks and int(config.subtasks[int(subtask_id)].station_id) >= 0
        ],
        key=lambda item: item[0],
    )
    if not ranked:
        return {"success": False}
    picked = pick_ranked_candidate(rng, ranked, opt.cfg)
    if picked is None:
        return {"success": False}
    _, center_subtask_id = picked
    center = config.subtasks.get(int(center_subtask_id))
    if center is None:
        return {"success": False}
    station_rows = config.station_subtasks(int(center.station_id))
    center_idx = next((idx for idx, row in enumerate(station_rows) if int(row.subtask_id) == int(center_subtask_id)), 0)
    move_n = max(1, min(int(degree), int(getattr(opt.cfg, "resource_critical_path_window_size", 4)), len(station_rows)))
    start = max(0, int(center_idx) - move_n + 1)
    end = min(len(station_rows), start + move_n)
    release_ids = [int(row.subtask_id) for row in station_rows[start:end]]
    return _finalize_release_ids(
        config,
        release_ids,
        move_n,
        priority_ids=[int(center_subtask_id)],
        preview=bool(preview),
        metrics=metrics,
        extra={"critical_path_subtask_ids": [int(x) for x in release_ids]},
    )


def y_destroy_critical_path_block(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _critical_path_release_payload(opt, config, rng, int(degree), preview=False)


def y_plan_destroy_congested_station_block(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    loads = _station_loads(config)
    if not loads:
        return {"success": False}
    ranked = sorted([((-float(loads[sid]), int(sid)), int(sid)) for sid in loads.keys()], key=lambda item: item[0])
    picked = pick_ranked_candidate(rng, ranked, opt.cfg)
    if picked is None:
        return {"success": False}
    _, heavy_station = picked
    rows = sorted(config.station_subtasks(int(heavy_station)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    if not rows:
        return {"success": False}
    move_n = max(1, min(int(degree), len(rows)))
    chosen = list(reversed(rows))[:move_n]
    released = _preview_release_rows(chosen, move_n)
    return {
        "success": True,
        "released_subtasks": released,
        "source_station_ids": [int(heavy_station)],
    }


def y_plan_destroy_cross_station_fragment(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    order_station_map: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        order_station_map[int(row.order_id)].append(row)
    candidate_rows = []
    for order_id, rows in order_station_map.items():
        stations = {int(row.station_id) for row in rows if int(row.station_id) >= 0}
        if len(stations) >= 2:
            candidate_rows.append(((-len(stations), -len(rows), int(order_id)), int(order_id)))
    if not candidate_rows:
        return {"success": False}
    picked = pick_ranked_candidate(rng, sorted(candidate_rows, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False}
    _, target_order = picked
    rows = sorted(order_station_map[int(target_order)], key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    move_n = max(1, min(int(degree), len(rows)))
    released = _preview_release_rows(rows[:move_n], move_n)
    return {
        "success": True,
        "released_subtasks": released,
        "source_station_ids": sorted({int(station_id) for station_id, _ in released.values() if int(station_id) >= 0}),
    }


def y_plan_destroy_rank_window_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) >= 0],
        key=lambda row: (-int(row.station_rank), int(row.subtask_id)),
    )
    if not ranked:
        return {"success": False}
    candidates = [((float(-int(row.station_rank)), int(row.subtask_id)), int(row.subtask_id)) for row in ranked]
    picked = pick_ranked_candidate(rng, candidates, opt.cfg)
    if picked is None:
        return {"success": False}
    _, center_subtask_id = picked
    center_row = config.subtasks.get(int(center_subtask_id))
    if center_row is None or int(center_row.station_id) < 0:
        return {"success": False}
    station_rows = sorted(config.station_subtasks(int(center_row.station_id)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    center_idx = next((idx for idx, row in enumerate(station_rows) if int(row.subtask_id) == int(center_subtask_id)), 0)
    move_n = max(1, min(int(degree), len(station_rows)))
    start = max(0, int(center_idx) - move_n // 2)
    end = min(len(station_rows), start + move_n)
    start = max(0, end - move_n)
    released = _preview_release_rows(station_rows[start:end], move_n)
    return {
        "success": True,
        "released_subtasks": released,
        "source_station_ids": [int(center_row.station_id)],
    }


def y_plan_destroy_load_skew_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return y_plan_destroy_congested_station_block(opt, config, rng, degree)


def y_plan_destroy_random_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del opt
    return _random_release_payload(config, rng, degree, preview=True)


def y_plan_destroy_related_station_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del opt
    return _related_station_release_payload(config, rng, degree, preview=True)


def y_plan_destroy_related_robot_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _related_robot_release_payload(opt, config, rng, int(degree), preview=True)


def y_plan_destroy_heavy_robot_tail(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _heavy_robot_release_payload(opt, config, rng, int(degree), preview=True)


def y_plan_destroy_initial_idle_head(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _initial_idle_release_payload(opt, config, rng, int(degree), preview=True)


def y_plan_destroy_max_tardiness_blocker(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _max_tardiness_release_payload(opt, config, rng, int(degree), preview=True)


def y_plan_destroy_critical_path_block(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return _critical_path_release_payload(opt, config, rng, int(degree), preview=True)


def y_plan_destroy_global_station_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del opt, rng, degree
    rows = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) >= 0],
        key=lambda row: (int(row.station_id), int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)),
    )
    released = _preview_release_rows(rows, len(rows))
    return {
        "success": bool(released),
        "released_subtasks": released,
        "source_station_ids": sorted({int(station_id) for station_id, _rank in released.values() if int(station_id) >= 0}),
        "global_station_release": True,
    }


def y_destroy_global_station_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    payload = y_plan_destroy_global_station_release(opt, config, rng, degree)
    if not bool(payload.get("success", False)):
        return {"success": False}
    released_ids = [int(x) for x in (payload.get("released_subtasks", {}) or {}).keys()]
    for subtask_id in released_ids:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = -1
        row.station_rank = -1
    return payload


def _assign_station_rank(config: ResourceConfig, subtask_id: int, station_id: int) -> None:
    row = config.subtasks.get(int(subtask_id))
    if row is None:
        return
    row.station_id = int(station_id)
    current_rows = [item for item in config.subtasks.values() if int(item.station_id) == int(station_id) and int(item.subtask_id) != int(subtask_id)]
    row.station_rank = int(len(current_rows))


def _choose_station_earliest_finish(opt, config: ResourceConfig, subtask: ResourceSubtask, rng=None) -> int:
    loads = _station_loads(config)
    choices = []
    for station_id in range(_station_count(opt)):
        choices.append((_station_choice_cost(opt, config, subtask, int(station_id), loads), int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _choose_station_load_balance(opt, config: ResourceConfig, subtask: ResourceSubtask, rng=None) -> int:
    loads = _station_loads(config)
    choices = []
    for station_id in range(_station_count(opt)):
        projected = dict(loads)
        projected[int(station_id)] = projected.get(int(station_id), 0.0) + _subtask_station_work(subtask)
        vals = list(projected.values()) or [0.0]
        mean = sum(vals) / max(1, len(vals))
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
        choices.append((float(variance), int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _station_load_std_from_map(loads: Dict[int, float]) -> float:
    vals = list(loads.values()) or [0.0]
    mean = sum(vals) / max(1, len(vals))
    variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    return float(math.sqrt(max(0.0, variance)))


def _y_base_maps(config: ResourceConfig, released_ids: List[int]) -> Tuple[Dict[int, float], Dict[int, int]]:
    released = {int(x) for x in (released_ids or [])}
    loads: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    for row in config.subtasks.values():
        if int(row.subtask_id) in released or int(row.station_id) < 0:
            continue
        loads[int(row.station_id)] += float(_subtask_station_work(row))
        counts[int(row.station_id)] += 1
    return dict(loads), dict(counts)


def _choose_station_earliest_finish_from_maps(opt, loads: Dict[int, float], subtask: ResourceSubtask, rng=None) -> int:
    choices = []
    synthetic = ResourceConfig(work_units={}, subtasks={}, capacity_limits={}, next_subtask_id=0, next_task_id=0)
    for station_id in range(_station_count(opt)):
        choices.append((_station_choice_cost(opt, synthetic, subtask, int(station_id), loads), int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _choose_station_load_balance_from_maps(opt, loads: Dict[int, float], subtask: ResourceSubtask, rng=None) -> int:
    choices = []
    for station_id in range(_station_count(opt)):
        projected = dict(loads)
        projected[int(station_id)] = projected.get(int(station_id), 0.0) + _subtask_station_work(subtask)
        vals = list(projected.values()) or [0.0]
        mean = sum(vals) / max(1, len(vals))
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
        choices.append((float(variance), int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _subtask_sweep_features(opt, config: ResourceConfig, subtask: ResourceSubtask) -> Dict[str, float]:
    order = None
    problem = getattr(opt, "problem", None)
    if problem is not None:
        for row in getattr(problem, "order_list", []) or []:
            if int(getattr(row, "order_id", -1)) == int(subtask.order_id):
                order = row
                break
    order_in = 0.0
    if order is not None:
        raw_in = getattr(order, "order_in_time", None)
        if hasattr(raw_in, "timestamp"):
            try:
                order_in = float(raw_in.timestamp())
            except Exception:
                order_in = 0.0
    lst = float(getattr(order, "lst_sec", 0.0) or 0.0) if order is not None else 0.0
    total_qty = float(getattr(order, "total_qty", 0.0) or 0.0) if order is not None else float(_subtask_pick_count(config, subtask))
    stack_points: List[Tuple[float, float]] = []
    for stack_id in _subtask_candidate_stack_ids(opt, config, subtask):
        xy = getattr(opt, "_stack_xy", lambda *_args, **_kwargs: None)(int(stack_id))
        if xy is None:
            stack = getattr(getattr(opt, "problem", None), "point_to_stack", {}).get(int(stack_id))
            if stack is not None:
                xy = (float(stack.store_point.x), float(stack.store_point.y))
        if xy is not None:
            stack_points.append((float(xy[0]), float(xy[1])))
    if stack_points:
        avg_x = float(sum(pt[0] for pt in stack_points) / len(stack_points))
        avg_y = float(sum(pt[1] for pt in stack_points) / len(stack_points))
        min_x = float(min(pt[0] for pt in stack_points))
        max_x = float(max(pt[0] for pt in stack_points))
    else:
        avg_x = avg_y = min_x = max_x = 0.0
    best_station = 0
    best_station_cost = float("inf")
    for station_id in range(_station_count(opt)):
        cost = float(_arrival_proxy(opt, subtask, int(station_id)))
        if (float(cost), int(station_id)) < (float(best_station_cost), int(best_station)):
            best_station_cost = float(cost)
            best_station = int(station_id)
    return {
        "order_in": float(order_in),
        "lst": float(lst),
        "total_qty": float(total_qty),
        "avg_x": float(avg_x),
        "avg_y": float(avg_y),
        "min_x": float(min_x),
        "max_x": float(max_x),
        "best_station": float(best_station),
        "best_station_cost": float(best_station_cost),
    }


def _station_quota_pattern(station_count: int, item_count: int, pattern: str) -> List[int]:
    station_count = max(1, int(station_count))
    item_count = max(0, int(item_count))
    base = item_count // station_count
    rem = item_count % station_count
    quotas = [int(base) for _ in range(station_count)]
    if rem <= 0:
        return quotas
    if str(pattern) == "back":
        order = list(range(station_count - 1, -1, -1))
    elif str(pattern) == "middle":
        mid = (station_count - 1) / 2.0
        order = sorted(range(station_count), key=lambda sid: (abs(float(sid) - mid), int(sid)))
    elif str(pattern) == "edges":
        order = []
        lo, hi = 0, station_count - 1
        while lo <= hi:
            order.append(lo)
            if hi != lo:
                order.append(hi)
            lo += 1
            hi -= 1
    else:
        order = list(range(station_count))
    for sid in order[:rem]:
        quotas[int(sid)] += 1
    return quotas


def _plan_y_sweep_pack(opt, config: ResourceConfig, released_ids: Sequence[int], variant: int) -> Dict[int, Dict[str, int]]:
    rows = [config.subtasks[int(subtask_id)] for subtask_id in released_ids if int(subtask_id) in config.subtasks]
    if not rows:
        return {}
    features = {int(row.subtask_id): _subtask_sweep_features(opt, config, row) for row in rows}
    key_variants = [
        lambda row: (features[int(row.subtask_id)]["order_in"], features[int(row.subtask_id)]["avg_x"], int(row.order_id)),
        lambda row: (features[int(row.subtask_id)]["best_station"], features[int(row.subtask_id)]["order_in"], int(row.order_id)),
        lambda row: (features[int(row.subtask_id)]["avg_x"], features[int(row.subtask_id)]["order_in"], int(row.order_id)),
        lambda row: (features[int(row.subtask_id)]["lst"], features[int(row.subtask_id)]["avg_x"], int(row.order_id)),
        lambda row: (-features[int(row.subtask_id)]["total_qty"], features[int(row.subtask_id)]["order_in"], int(row.order_id)),
        lambda row: (features[int(row.subtask_id)]["min_x"], features[int(row.subtask_id)]["max_x"], int(row.order_id)),
    ]
    quota_patterns = ["middle", "front", "back", "edges"]
    key_index = int(variant) % len(key_variants)
    quota_index = (int(variant) // len(key_variants)) % len(quota_patterns)
    reverse = bool((int(variant) // max(1, len(key_variants) * len(quota_patterns))) % 2)
    ordered = sorted(rows, key=key_variants[int(key_index)], reverse=reverse)
    quotas = _station_quota_pattern(_station_count(opt), len(ordered), quota_patterns[int(quota_index)])
    assignments: Dict[int, Dict[str, int]] = {}
    pos = 0
    for station_id, quota in enumerate(quotas):
        station_rows = ordered[pos : pos + int(quota)]
        pos += int(quota)
        station_rows = sorted(
            station_rows,
            key=lambda row: (
                features[int(row.subtask_id)]["order_in"],
                features[int(row.subtask_id)]["best_station_cost"],
                int(row.order_id),
            ),
        )
        for rank, row in enumerate(station_rows):
            assignments[int(row.subtask_id)] = {
                "station_id": int(station_id),
                "station_rank": int(rank),
                "selected_robot_id": -1,
            }
    return assignments


def _plan_y_anchor_compact_sweep(opt, config: ResourceConfig, released_ids: Sequence[int]) -> Dict[int, Dict[str, int]]:
    rows = [config.subtasks[int(subtask_id)] for subtask_id in released_ids if int(subtask_id) in config.subtasks]
    station_count = _station_count(opt)
    if len(rows) < 3 or station_count < 3:
        return {}
    features = {int(row.subtask_id): _subtask_sweep_features(opt, config, row) for row in rows}
    earliest = min(rows, key=lambda row: (features[int(row.subtask_id)]["order_in"], int(row.order_id)))
    latest = max(rows, key=lambda row: (features[int(row.subtask_id)]["order_in"], -int(row.order_id)))
    if int(earliest.subtask_id) == int(latest.subtask_id):
        return {}
    assignments: Dict[int, Dict[str, int]] = {
        int(earliest.subtask_id): {"station_id": 0, "station_rank": 0, "selected_robot_id": -1},
        int(latest.subtask_id): {"station_id": station_count - 1, "station_rank": 0, "selected_robot_id": -1},
    }
    remaining = [row for row in rows if int(row.subtask_id) not in assignments]
    inner_station_ids = list(range(1, station_count - 1))
    if not inner_station_ids:
        return assignments
    quotas = _station_quota_pattern(len(inner_station_ids), len(remaining), "middle")
    station_quota = {int(station_id): int(quota) for station_id, quota in zip(inner_station_ids, quotas)}
    singleton_stations = [int(station_id) for station_id in inner_station_ids if int(station_quota.get(int(station_id), 0)) == 1]
    reserved_singletons: Dict[int, ResourceSubtask] = {}
    if singleton_stations:
        compact_rows = sorted(
            remaining,
            key=lambda row: (
                float(features[int(row.subtask_id)]["max_x"] - features[int(row.subtask_id)]["min_x"]),
                float(features[int(row.subtask_id)]["best_station_cost"]),
                int(row.order_id),
            ),
        )
        for station_id, row in zip(reversed(singleton_stations), compact_rows):
            reserved_singletons[int(station_id)] = row
        reserved_ids = {int(row.subtask_id) for row in reserved_singletons.values()}
        remaining = [row for row in remaining if int(row.subtask_id) not in reserved_ids]
    for station_id in inner_station_ids:
        quota = int(station_quota.get(int(station_id), 0))
        if quota <= 0:
            continue
        if int(station_id) in reserved_singletons:
            row = reserved_singletons[int(station_id)]
            assignments[int(row.subtask_id)] = {"station_id": int(station_id), "station_rank": 0, "selected_robot_id": -1}
            continue
        picked = sorted(
            remaining,
            key=lambda row: (
                abs(float(features[int(row.subtask_id)]["best_station"]) - float(station_id)),
                float(_arrival_proxy(opt, row, int(station_id))),
                float(features[int(row.subtask_id)]["min_x"]),
                int(row.order_id),
            ),
        )[:quota]
        picked_ids = {int(row.subtask_id) for row in picked}
        remaining = [row for row in remaining if int(row.subtask_id) not in picked_ids]
        picked = sorted(
            picked,
            key=lambda row: (
                float(_arrival_proxy(opt, row, int(station_id))),
                float(features[int(row.subtask_id)]["order_in"]),
                int(row.order_id),
            ),
        )
        for rank, row in enumerate(picked):
            assignments[int(row.subtask_id)] = {
                "station_id": int(station_id),
                "station_rank": int(rank),
                "selected_robot_id": -1,
            }
    if remaining:
        for row in remaining:
            best_station = min(
                inner_station_ids,
                key=lambda station_id: (
                    int(station_quota.get(int(station_id), 0))
                    - sum(1 for meta in assignments.values() if int(meta["station_id"]) == int(station_id)),
                    float(_arrival_proxy(opt, row, int(station_id))),
                    int(station_id),
                ),
            )
            rank = sum(1 for meta in assignments.values() if int(meta["station_id"]) == int(best_station))
            assignments[int(row.subtask_id)] = {"station_id": int(best_station), "station_rank": int(rank), "selected_robot_id": -1}
    return assignments


def _plan_y_left_right_tail_balance(opt, config: ResourceConfig, released_ids: Sequence[int]) -> Dict[int, Dict[str, int]]:
    rows = [config.subtasks[int(subtask_id)] for subtask_id in released_ids if int(subtask_id) in config.subtasks]
    station_count = _station_count(opt)
    if len(rows) < station_count or station_count < 4:
        return {}
    features = {int(row.subtask_id): _subtask_sweep_features(opt, config, row) for row in rows}
    ordered = sorted(
        rows,
        key=lambda row: (
            float(features[int(row.subtask_id)]["avg_x"]),
            float(features[int(row.subtask_id)]["min_x"]),
            float(features[int(row.subtask_id)]["order_in"]),
            int(row.order_id),
        ),
    )
    front_quota = max(1, int(math.ceil(float(len(rows)) / float(station_count))))
    front = ordered[:front_quota]
    tail = ordered[front_quota:]
    assignments: Dict[int, Dict[str, int]] = {}

    def assign_rows(station_id: int, station_rows: Sequence[ResourceSubtask]) -> None:
        ranked = sorted(
            list(station_rows or []),
            key=lambda row: (
                float(features[int(row.subtask_id)]["order_in"]),
                float(_arrival_proxy(opt, row, int(station_id))),
                int(row.order_id),
            ),
        )
        for rank, row in enumerate(ranked):
            assignments[int(row.subtask_id)] = {
                "station_id": int(station_id),
                "station_rank": int(rank),
                "selected_robot_id": -1,
            }

    assign_rows(0, front)
    middle_station_ids = list(range(1, max(1, station_count - 2)))
    for station_id in middle_station_ids:
        if not tail:
            break
        row = tail.pop(0)
        assign_rows(int(station_id), [row])
    if tail:
        rightmost = max(tail, key=lambda row: (float(features[int(row.subtask_id)]["avg_x"]), int(row.order_id)))
        tail = [row for row in tail if int(row.subtask_id) != int(rightmost.subtask_id)]
        assign_rows(station_count - 2, [rightmost])
    if tail:
        assign_rows(station_count - 1, tail)
    return assignments if len(assignments) == len(rows) else {}


def _plan_y_global_route_balance(opt, config: ResourceConfig, released_ids: Sequence[int], rng=None) -> Dict[int, Dict[str, int]]:
    del rng
    station_count = _station_count(opt)
    if station_count <= 0:
        return {}
    remaining = [int(subtask_id) for subtask_id in released_ids if int(subtask_id) in config.subtasks]
    if not remaining:
        return {}
    max_per_station = max(1, int(math.ceil(float(len(remaining)) / float(station_count))))
    robot_frontier = _snapshot_robot_frontier(opt)
    call_index = int(getattr(opt, "_resource_y_global_route_balance_calls", 0))
    setattr(opt, "_resource_y_global_route_balance_calls", call_index + 1)
    small_no_split_exhaustive = bool(
        bool(getattr(opt.cfg, "sp1_no_split", False))
        and str(getattr(opt.cfg, "resource_eval_backend", "") or "").strip().lower() != "surrogate"
        and len(remaining) <= 8
        and station_count <= 5
    )
    if (
        bool(getattr(opt.cfg, "sp1_no_split", False))
        and str(getattr(opt.cfg, "resource_eval_backend", "") or "").strip().lower() == "surrogate"
        and call_index == 0
    ):
        tail_balanced = _plan_y_left_right_tail_balance(opt, config, remaining)
        if tail_balanced:
            return tail_balanced
    if call_index > 0 and not small_no_split_exhaustive:
        if bool(getattr(opt.cfg, "sp1_no_split", False)) and call_index == 1:
            tail_balanced = _plan_y_left_right_tail_balance(opt, config, remaining)
            if tail_balanced:
                return tail_balanced
        if call_index == 1:
            anchored = _plan_y_anchor_compact_sweep(opt, config, remaining)
            if anchored:
                return anchored
        sweep = _plan_y_sweep_pack(opt, config, remaining, int(call_index - 1))
        if sweep:
            return sweep

    if small_no_split_exhaustive and call_index > 0:
        top_plans: List[Tuple[Tuple[float, ...], Dict[int, Dict[str, int]]]] = []
        min_nonempty = min(station_count, len(remaining))
        for station_tuple in itertools.product(range(station_count), repeat=len(remaining)):
            counts_tmp: Dict[int, int] = defaultdict(int)
            for station_id in station_tuple:
                counts_tmp[int(station_id)] += 1
            if any(int(count) > int(max_per_station) for count in counts_tmp.values()):
                continue
            if sum(1 for count in counts_tmp.values() if int(count) > 0) < int(min_nonempty):
                continue
            station_rows: Dict[int, List[Tuple[float, float, int, ResourceSubtask]]] = defaultdict(list)
            feasible = True
            for subtask_id, station_id in zip(remaining, station_tuple):
                row = config.subtasks.get(int(subtask_id))
                if row is None:
                    feasible = False
                    break
                arrival, selected_robot_id = _triangle_arrival_cost(opt, config, row, int(station_id), robot_frontier=robot_frontier)
                if not math.isfinite(float(arrival)):
                    arrival = _arrival_proxy(opt, row, int(station_id))
                station_rows[int(station_id)].append((float(arrival), float(_subtask_station_work(row)), int(subtask_id), row))
            if not feasible:
                continue
            assignments: Dict[int, Dict[str, int]] = {}
            station_finish_times: List[float] = []
            total_arrival = 0.0
            total_work = 0.0
            for station_id, rows in station_rows.items():
                rows.sort(key=lambda item: (float(item[0]), -float(item[1]), int(item[2])))
                station_time = 0.0
                for rank, (arrival, work, subtask_id, row) in enumerate(rows):
                    station_time = max(float(station_time), float(arrival)) + float(work)
                    selected_robot_id = int(_triangle_arrival_cost(opt, config, row, int(station_id), robot_frontier=robot_frontier)[1])
                    assignments[int(subtask_id)] = {
                        "station_id": int(station_id),
                        "station_rank": int(rank),
                        "selected_robot_id": int(selected_robot_id),
                    }
                    total_arrival += float(arrival)
                    total_work += float(work)
                station_finish_times.append(float(station_time))
            if len(assignments) != len(remaining):
                continue
            count_vals = [float(counts_tmp.get(station_id, 0)) for station_id in range(station_count)]
            mean_count = sum(count_vals) / max(1, len(count_vals))
            count_var = sum((value - mean_count) ** 2 for value in count_vals) / max(1, len(count_vals))
            score = (
                float(max(station_finish_times) if station_finish_times else float("inf")),
                float(0.02 * total_arrival + 0.002 * total_work + 10.0 * count_var),
                tuple(int(assignments[int(subtask_id)]["station_id"]) for subtask_id in sorted(assignments)),
            )
            top_plans.append((score, assignments))
        if top_plans:
            top_plans.sort(key=lambda item: item[0])
            unique: List[Tuple[Tuple[float, ...], Dict[int, Dict[str, int]]]] = []
            seen = set()
            for score, assignments in top_plans:
                signature = tuple((int(subtask_id), int(meta["station_id"]), int(meta["station_rank"])) for subtask_id, meta in sorted(assignments.items()))
                if signature in seen:
                    continue
                seen.add(signature)
                unique.append((score, assignments))
                if len(unique) >= 32:
                    break
            return dict(unique[int(call_index - 1) % len(unique)][1])

    loads: Dict[int, float] = {int(station_id): 0.0 for station_id in range(station_count)}
    counts: Dict[int, int] = {int(station_id): 0 for station_id in range(station_count)}

    def station_cost(row: ResourceSubtask, station_id: int, assignment_index: int) -> float:
        base = _station_choice_cost(
            opt,
            config,
            row,
            int(station_id),
            loads,
            released_ids=remaining,
            counts=counts,
            assignment_index=int(assignment_index),
            first_batch_limit=min(station_count, len(remaining)),
            used_station_ids={sid for sid, cnt in counts.items() if int(cnt) > 0},
            used_robot_ids=set(),
            robot_frontier=robot_frontier,
        )
        overflow = max(0, int(counts.get(int(station_id), 0)) + 1 - int(max_per_station))
        return float(base + 10000.0 * overflow + 18.0 * float(counts.get(int(station_id), 0)))

    def regret_key(subtask_id: int) -> Tuple[float, float, int]:
        row = config.subtasks[int(subtask_id)]
        costs = sorted(float(station_cost(row, station_id, 0)) for station_id in range(station_count))
        regret = float(costs[1] - costs[0]) if len(costs) >= 2 else float(costs[0])
        return (-regret, float(costs[0]), int(subtask_id))

    assignments: Dict[int, Dict[str, int]] = {}
    while remaining:
        remaining.sort(key=regret_key)
        subtask_id = int(remaining.pop(0))
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        choices = sorted(
            [(station_cost(row, station_id, len(assignments)), int(station_id)) for station_id in range(station_count)],
            key=lambda item: (float(item[0]), int(item[1])),
        )
        if not choices:
            continue
        station_id = int(choices[0][1])
        selected_robot_id = int(_triangle_arrival_cost(opt, config, row, station_id, robot_frontier=robot_frontier)[1])
        assignments[int(subtask_id)] = {
            "station_id": int(station_id),
            "station_rank": int(counts.get(int(station_id), 0)),
            "selected_robot_id": int(selected_robot_id),
        }
        loads[int(station_id)] = loads.get(int(station_id), 0.0) + _subtask_station_work(row)
        counts[int(station_id)] = counts.get(int(station_id), 0) + 1
    return assignments


def _build_y_action_signature(destroy_name: str, repair_name: str, released_subtasks: Dict[int, Tuple[int, int]], assignments: Dict[int, Dict[str, int]]) -> Tuple[object, ...]:
    release_sig = tuple(sorted((int(subtask_id), int(station_id), int(rank)) for subtask_id, (station_id, rank) in (released_subtasks or {}).items()))
    assign_sig = tuple(sorted((int(subtask_id), int(meta["station_id"]), int(meta["station_rank"])) for subtask_id, meta in (assignments or {}).items()))
    return ("Y", str(destroy_name), release_sig, str(repair_name), assign_sig)


def _build_y_rough_features(opt, config: ResourceConfig, released_subtasks: Dict[int, Tuple[int, int]], assignments: Dict[int, Dict[str, int]]) -> Dict[str, float]:
    released_ids = [int(x) for x in (released_subtasks or {}).keys()]
    base_loads, base_counts = _y_base_maps(config, released_ids)
    old_loads = _station_loads(config)
    old_std = _station_load_std_from_map(old_loads)
    new_loads = dict(base_loads)
    old_arrival = 0.0
    new_arrival = 0.0
    old_rank_sum = 0.0
    new_rank_sum = 0.0
    old_heat = 0.0
    new_heat = 0.0
    old_overlap = 0.0
    new_overlap = 0.0
    old_triangle = 0.0
    new_triangle = 0.0
    old_starvation = 0.0
    new_starvation = 0.0
    new_first_batch_spread = 0.0
    old_first_batch_spread = 0.0
    robot_frontier = _snapshot_robot_frontier(opt)
    first_batch_limit = min(_station_count(opt), len(assignments))
    seen_new_stations: Set[int] = set()
    seen_old_stations: Set[int] = set()
    for subtask_id, meta in (assignments or {}).items():
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        work = float(_subtask_station_work(row))
        target_station = int(meta["station_id"])
        new_loads[target_station] = new_loads.get(target_station, 0.0) + work
        new_arrival += float(_arrival_proxy(opt, row, target_station))
        new_rank_sum += float(meta["station_rank"])
        origin_station = int(released_subtasks.get(int(subtask_id), (-1, -1))[0])
        origin_rank = int(released_subtasks.get(int(subtask_id), (-1, -1))[1])
        if origin_station >= 0:
            old_arrival += float(_arrival_proxy(opt, row, origin_station))
            old_triangle += float(_triangle_arrival_cost(opt, config, row, int(origin_station), robot_frontier=robot_frontier)[0])
            old_heat_i, old_overlap_i, _ = _station_choice_penalties(opt, config, row, int(origin_station), old_loads, released_ids=released_ids)
            old_heat += float(old_heat_i)
            old_overlap += float(old_overlap_i)
            if int(base_counts.get(int(origin_station), 0)) <= 0:
                old_starvation += 1.0
            if len(seen_old_stations) < int(first_batch_limit):
                seen_old_stations.add(int(origin_station))
        if origin_rank >= 0:
            old_rank_sum += float(origin_rank)
        new_heat_i, new_overlap_i, _ = _station_choice_penalties(opt, config, row, int(target_station), new_loads, released_ids=released_ids)
        new_heat += float(new_heat_i)
        new_overlap += float(new_overlap_i)
        new_triangle += float(_triangle_arrival_cost(opt, config, row, int(target_station), robot_frontier=robot_frontier)[0])
        if int(base_counts.get(int(target_station), 0)) <= 0:
            new_starvation += 1.0
        if len(seen_new_stations) < int(first_batch_limit):
            seen_new_stations.add(int(target_station))
    new_std = _station_load_std_from_map(new_loads)
    count = max(1, len(assignments))
    sy_delta = float((new_std - old_std) / max(1.0, float(getattr(opt, "work_z", 1.0) or 1.0)))
    sy_delta += 0.01 * float((new_arrival - old_arrival) / count)
    sy_delta += 0.02 * float((new_rank_sum - old_rank_sum) / count)
    sy_delta += 0.01 * float((new_heat - old_heat) / count)
    sy_delta += 0.01 * float((new_overlap - old_overlap) / count)
    return {
        "sy_delta": float(sy_delta),
        "affected_count": float(len(assignments)),
        "load_std_before": float(old_std),
        "load_std_after": float(new_std),
        "heat_delta": float(new_heat - old_heat),
        "window_overlap_delta": float(new_overlap - old_overlap),
        "triangle_arrival_delta": float((new_triangle - old_triangle) / count),
        "starvation_relief_delta": float(old_starvation - new_starvation),
        "first_batch_spread_delta": float(len(seen_new_stations) - len(seen_old_stations)),
    }


def _station_rank_permutation_proxy(
    opt,
    config: ResourceConfig,
    station_id: int,
    ordered_subtask_ids: Sequence[int],
) -> float:
    cumulative = 0.0
    completion_sum = 0.0
    max_tail = 0.0
    weighted_rank_penalty = 0.0
    for rank, subtask_id in enumerate(ordered_subtask_ids):
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        station_work = float(_subtask_station_work(row))
        arrival = float(_arrival_proxy(opt, row, int(station_id)))
        cumulative += station_work
        completion = float(arrival + cumulative)
        completion_sum += completion
        max_tail = max(max_tail, completion)
        weighted_rank_penalty += float(rank) * station_work
    return float(max_tail + 0.15 * completion_sum + 0.10 * weighted_rank_penalty)


def _enumerate_station_rank_permutations(
    opt,
    config: ResourceConfig,
    station_id: int,
    released_ids: Sequence[int],
    max_candidates: int,
) -> List[Tuple[float, List[int]]]:
    released_ids = [int(x) for x in (released_ids or []) if int(x) in config.subtasks]
    if not released_ids or int(station_id) < 0:
        return []
    released_set = set(int(x) for x in released_ids)
    fixed_ids = [
        int(row.subtask_id)
        for row in sorted(config.station_subtasks(int(station_id)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
        if int(row.subtask_id) not in released_set
    ]
    candidates: List[Tuple[float, List[int]]] = []
    seen: Set[Tuple[int, ...]] = set()
    perm_limit = max(1, int(max_candidates))
    release_perms = list(itertools.permutations(released_ids))
    if len(release_perms) > perm_limit:
        release_perms = release_perms[:perm_limit]
    for perm in release_perms:
        for insert_start in range(0, len(fixed_ids) + 1):
            sequence = list(fixed_ids)
            for offset, subtask_id in enumerate(perm):
                sequence.insert(int(insert_start) + int(offset), int(subtask_id))
            signature = tuple(int(x) for x in sequence)
            if signature in seen:
                continue
            seen.add(signature)
            score = _station_rank_permutation_proxy(opt, config, int(station_id), sequence)
            candidates.append((float(score), list(sequence)))
    candidates.sort(key=lambda item: (float(item[0]), tuple(int(x) for x in item[1])))
    return candidates[:perm_limit]


def _apply_station_sequence(candidate: ResourceConfig, station_id: int, ordered_subtask_ids: Sequence[int]) -> None:
    for rank, subtask_id in enumerate(ordered_subtask_ids):
        row = candidate.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = int(station_id)
        row.station_rank = int(rank)


def _wave1_minimax_match(
    opt,
    config: ResourceConfig,
    subtask_ids: Sequence[int],
    counts: Dict[int, int],
    robot_frontier: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[int, Dict[str, float]]:
    prefix_ids = [int(subtask_id) for subtask_id in (subtask_ids or []) if int(subtask_id) in config.subtasks]
    if not prefix_ids or not bool(getattr(opt.cfg, "resource_y_wave1_enable", True)):
        return {}
    station_ids = list(range(_station_count(opt)))
    frontier = _robot_availability_proxy(opt, robot_frontier)
    robot_ids = sorted(int(robot_id) for robot_id in frontier.keys())
    if not station_ids or not robot_ids:
        return {}
    task_count = min(len(prefix_ids), len(station_ids), len(robot_ids))
    prefix_ids = prefix_ids[:task_count]
    if not prefix_ids:
        return {}
    cost_cache: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for subtask_id in prefix_ids:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        heavy_penalty = _subtask_heaviness_penalty(opt, config, row)
        for station_id in station_ids:
            for robot_id in robot_ids:
                arrival = _robot_station_arrival_cost(opt, config, row, int(station_id), int(robot_id), robot_frontier=frontier)
                cost_cache[(int(subtask_id), int(station_id), int(robot_id))] = (float(arrival), float(arrival + heavy_penalty))
    if not cost_cache:
        return {}
    best_score = None
    best_assignment: Dict[int, Dict[str, float]] = {}
    for station_perm in itertools.permutations(station_ids, len(prefix_ids)):
        for robot_perm in itertools.permutations(robot_ids, len(prefix_ids)):
            raw_times: List[float] = []
            adjusted_times: List[float] = []
            trial: Dict[int, Dict[str, float]] = {}
            feasible = True
            for index, subtask_id in enumerate(prefix_ids):
                key = (int(subtask_id), int(station_perm[index]), int(robot_perm[index]))
                if key not in cost_cache:
                    feasible = False
                    break
                raw_eta, adjusted_eta = cost_cache[key]
                raw_times.append(float(raw_eta))
                adjusted_times.append(float(adjusted_eta))
                trial[int(subtask_id)] = {
                    "station_id": int(station_perm[index]),
                    "station_rank": int(counts.get(int(station_perm[index]), 0)),
                    "selected_robot_id": int(robot_perm[index]),
                    "projected_arrival": float(raw_eta),
                    "adjusted_arrival": float(adjusted_eta),
                }
            if not feasible:
                continue
            score = (
                float(max(adjusted_times) if adjusted_times else float("inf")),
                float(sum(adjusted_times)),
                float(max(raw_times) if raw_times else float("inf")),
                float(sum(raw_times)),
                tuple(int(x) for x in station_perm),
                tuple(int(x) for x in robot_perm),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = trial
    return dict(best_assignment)


def _plan_y_assignments(opt, config: ResourceConfig, released_subtasks: Dict[int, Tuple[int, int]], strategy: str, rng=None) -> Dict[str, object]:
    released_ids = [int(x) for x in (released_subtasks or {}).keys()]
    if not released_ids:
        return {"success": False}
    if str(strategy) == "y_repair_station_rank_permutation":
        source_station_ids = sorted({int(station_id) for station_id, _ in released_subtasks.values() if int(station_id) >= 0})
        if len(source_station_ids) != 1:
            return {"success": False}
        station_id = int(source_station_ids[0])
        assignments = {
            int(subtask_id): {
                "station_id": int(station_id),
                "station_rank": int(released_subtasks[int(subtask_id)][1]),
                "selected_robot_id": -1,
            }
            for subtask_id in released_ids
        }
        return {"success": True, "assignments": assignments}
    loads, counts = _y_base_maps(config, released_ids)
    assignments: Dict[int, Dict[str, int]] = {}
    remaining = list(released_ids)
    first_batch_limit = min(_station_count(opt), len(released_ids))
    robot_frontier = _snapshot_robot_frontier(opt)
    used_station_ids: Set[int] = set()
    used_robot_ids: Set[int] = set()
    wave1_size = min(int(first_batch_limit), len(released_ids), max(0, len(_robot_availability_proxy(opt, robot_frontier))))
    wave1_assignments = _wave1_minimax_match(opt, config, released_ids[:wave1_size], counts, robot_frontier=robot_frontier)
    if wave1_assignments:
        for subtask_id in released_ids[:wave1_size]:
            meta = dict(wave1_assignments.get(int(subtask_id), {}) or {})
            if not meta:
                continue
            station_id = int(meta["station_id"])
            assignments[int(subtask_id)] = meta
            row = config.subtasks.get(int(subtask_id))
            if row is not None:
                loads[int(station_id)] = loads.get(int(station_id), 0.0) + _subtask_station_work(row)
            used_station_ids.add(int(station_id))
            selected_robot_id = int(meta.get("selected_robot_id", -1))
            if selected_robot_id >= 0:
                used_robot_ids.add(int(selected_robot_id))
            counts[int(station_id)] = counts.get(int(station_id), 0) + 1
        remaining = [int(subtask_id) for subtask_id in remaining if int(subtask_id) not in assignments]
    if str(strategy) == "y_repair_regret2_station":
        while remaining:
            best = None
            for subtask_id in list(remaining):
                row = config.subtasks.get(int(subtask_id))
                if row is None:
                    continue
                choices = []
                for station_id in range(_station_count(opt)):
                    choices.append(
                        (
                            _station_choice_cost(
                                opt,
                                config,
                                row,
                                int(station_id),
                                loads,
                                released_ids=released_ids,
                                counts=counts,
                                assignment_index=len(assignments),
                                first_batch_limit=first_batch_limit,
                                used_station_ids=used_station_ids,
                                used_robot_ids=used_robot_ids,
                                robot_frontier=robot_frontier,
                            ),
                            int(station_id),
                        )
                    )
                if not choices:
                    continue
                ranked_choices = sorted(choices, key=lambda item: (item[0], item[1]))
                regret = float(ranked_choices[1][0] - ranked_choices[0][0]) if len(ranked_choices) >= 2 else float(ranked_choices[0][0])
                candidate = {
                    "regret_score": -regret,
                    "best_cost": float(ranked_choices[0][0]),
                    "subtask_id": int(subtask_id),
                    "station_choice": pick_soft_greedy_min(rng, ranked_choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1]))),
                }
                if best is None or (float(candidate["regret_score"]), float(candidate["best_cost"]), int(candidate["subtask_id"])) < (
                    float(best["regret_score"]),
                    float(best["best_cost"]),
                    int(best["subtask_id"]),
                ):
                    best = candidate
            if best is None:
                break
            subtask_id = int(best["subtask_id"])
            station_id = int(best["station_choice"][1])
            selected_robot_id = -1
            row = config.subtasks.get(int(subtask_id))
            if row is not None:
                selected_robot_id = int(_triangle_arrival_cost(opt, config, row, int(station_id), robot_frontier=robot_frontier)[1])
            assignments[int(subtask_id)] = {
                "station_id": int(station_id),
                "station_rank": int(counts.get(int(station_id), 0)),
                "selected_robot_id": int(selected_robot_id),
            }
            if row is not None:
                loads[int(station_id)] = loads.get(int(station_id), 0.0) + _subtask_station_work(row)
                if len(assignments) <= int(first_batch_limit):
                    used_station_ids.add(int(station_id))
                    if selected_robot_id >= 0:
                        used_robot_ids.add(int(selected_robot_id))
            counts[int(station_id)] = counts.get(int(station_id), 0) + 1
            remaining.remove(int(subtask_id))
    else:
        ejection_sources = {
            int(subtask_id): int(station_id)
            for subtask_id, (station_id, _rank) in (released_subtasks or {}).items()
            if int(station_id) >= 0
        }
        chooser = {
            "y_repair_earliest_finish": lambda row: int(
                (
                    pick_soft_greedy_min(
                        rng,
                        [
                            (
                                _station_choice_cost(
                                    opt,
                                    config,
                                    row,
                                    int(station_id),
                                    loads,
                                    released_ids=released_ids,
                                    counts=counts,
                                    assignment_index=len(assignments),
                                    first_batch_limit=first_batch_limit,
                                    used_station_ids=used_station_ids,
                                    used_robot_ids=used_robot_ids,
                                    robot_frontier=robot_frontier,
                                ),
                                int(station_id),
                            )
                            for station_id in range(_station_count(opt))
                        ],
                        opt.cfg,
                        score_getter=lambda item: (float(item[0]), int(item[1])),
                    )
                    or (0.0, 0)
                )[1]
            ),
            "y_repair_arrival_aware_rank": lambda row: int(
                (
                    pick_soft_greedy_min(
                        rng,
                        [
                            (
                                _station_choice_cost(
                                    opt,
                                    config,
                                    row,
                                    int(station_id),
                                    loads,
                                    released_ids=released_ids,
                                    counts=counts,
                                    assignment_index=len(assignments),
                                    first_batch_limit=first_batch_limit,
                                    used_station_ids=used_station_ids,
                                    used_robot_ids=used_robot_ids,
                                    robot_frontier=robot_frontier,
                                ),
                                int(station_id),
                            )
                            for station_id in range(_station_count(opt))
                        ],
                        opt.cfg,
                        score_getter=lambda item: (float(item[0]), int(item[1])),
                    )
                    or (0.0, 0)
                )[1]
            ),
            "y_repair_load_balance": lambda row: _choose_station_load_balance_from_maps(opt, loads, row, rng),
            "y_repair_ejection_chain_balance": lambda row: int(
                (
                    pick_soft_greedy_min(
                        rng,
                        [
                            (
                                _station_choice_cost(
                                    opt,
                                    config,
                                    row,
                                    int(station_id),
                                    loads,
                                    released_ids=released_ids,
                                    counts=counts,
                                    assignment_index=len(assignments),
                                    first_batch_limit=first_batch_limit,
                                    used_station_ids=used_station_ids,
                                    used_robot_ids=used_robot_ids,
                                    robot_frontier=robot_frontier,
                                )
                                + (25.0 if int(station_id) == int(ejection_sources.get(int(row.subtask_id), -1)) else 0.0),
                                int(station_id),
                            )
                            for station_id in range(_station_count(opt))
                        ],
                        opt.cfg,
                        score_getter=lambda item: (float(item[0]), int(item[1])),
                    )
                    or (0.0, 0)
                )[1]
            ),
        }.get(
            str(strategy),
            lambda row: int(
                (
                    pick_soft_greedy_min(
                        rng,
                        [
                            (
                                _station_choice_cost(
                                    opt,
                                    config,
                                    row,
                                    int(station_id),
                                    loads,
                                    released_ids=released_ids,
                                    counts=counts,
                                    assignment_index=len(assignments),
                                    first_batch_limit=first_batch_limit,
                                    used_station_ids=used_station_ids,
                                    used_robot_ids=used_robot_ids,
                                    robot_frontier=robot_frontier,
                                ),
                                int(station_id),
                            )
                            for station_id in range(_station_count(opt))
                        ],
                        opt.cfg,
                        score_getter=lambda item: (float(item[0]), int(item[1])),
                    )
                    or (0.0, 0)
                )[1]
            ),
        )
        for subtask_id in released_ids:
            if int(subtask_id) in assignments:
                continue
            row = config.subtasks.get(int(subtask_id))
            if row is None:
                continue
            station_id = int(chooser(row))
            selected_robot_id = int(_triangle_arrival_cost(opt, config, row, int(station_id), robot_frontier=robot_frontier)[1])
            assignments[int(subtask_id)] = {
                "station_id": int(station_id),
                "station_rank": int(counts.get(int(station_id), 0)),
                "selected_robot_id": int(selected_robot_id),
            }
            loads[int(station_id)] = loads.get(int(station_id), 0.0) + _subtask_station_work(row)
            if len(assignments) <= int(first_batch_limit):
                used_station_ids.add(int(station_id))
                if selected_robot_id >= 0:
                    used_robot_ids.add(int(selected_robot_id))
            counts[int(station_id)] = counts.get(int(station_id), 0) + 1
    if not assignments:
        return {"success": False}
    return {"success": True, "assignments": assignments}


def plan_y_candidate(opt, config: ResourceConfig, destroy_name: str, repair_name: str, rng, degree: int) -> Dict[str, object]:
    destroy_planners = {
        "y_destroy_congested_station_block": y_plan_destroy_congested_station_block,
        "y_destroy_cross_station_fragment": y_plan_destroy_cross_station_fragment,
        "y_destroy_rank_window_release": y_plan_destroy_rank_window_release,
        "y_destroy_load_skew_release": y_plan_destroy_load_skew_release,
        "y_destroy_random_release": y_plan_destroy_random_release,
        "y_destroy_related_station_release": y_plan_destroy_related_station_release,
        "y_destroy_related_robot_release": y_plan_destroy_related_robot_release,
        "y_destroy_heavy_robot_tail": y_plan_destroy_heavy_robot_tail,
        "y_destroy_initial_idle_head": y_plan_destroy_initial_idle_head,
        "y_destroy_max_tardiness_blocker": y_plan_destroy_max_tardiness_blocker,
        "y_destroy_critical_path_block": y_plan_destroy_critical_path_block,
        "y_destroy_global_station_release": y_plan_destroy_global_station_release,
    }
    destroy_ctx = destroy_planners[str(destroy_name)](opt, config, rng, degree)
    if not bool(destroy_ctx.get("success", False)):
        return {"success": False}
    if str(repair_name) == "y_repair_global_route_balance":
        released_subtasks = dict(destroy_ctx.get("released_subtasks", {}) or {})
        assignments = _plan_y_global_route_balance(opt, config, list(released_subtasks.keys()), rng)
        repair_plan = {"success": bool(assignments), "assignments": assignments}
    else:
        repair_plan = _plan_y_assignments(opt, config, destroy_ctx.get("released_subtasks", {}), str(repair_name), rng)
    fallback_used = False
    effective_repair_name = str(repair_name)
    if not bool(repair_plan.get("success", False)):
        repair_plan = _plan_y_assignments(opt, config, destroy_ctx.get("released_subtasks", {}), "y_repair_earliest_finish", rng)
        fallback_used = bool(repair_plan.get("success", False))
        if bool(fallback_used):
            effective_repair_name = "y_repair_earliest_finish"
    if not bool(repair_plan.get("success", False)):
        return {"success": False}
    assignments = dict(repair_plan.get("assignments", {}) or {})
    return {
        "success": True,
        "destroy_ctx": destroy_ctx,
        "assignments": assignments,
        "repair_name": str(effective_repair_name),
        "fallback_used": bool(fallback_used),
        "action_signature": _build_y_action_signature(str(destroy_name), str(effective_repair_name), destroy_ctx.get("released_subtasks", {}), assignments),
        "rough_features": _build_y_rough_features(opt, config, destroy_ctx.get("released_subtasks", {}), assignments),
    }


def apply_exact_y_plan(opt, config: ResourceConfig, plan: Dict[str, object], rng=None) -> Dict[str, object]:
    del rng
    destroy_ctx = dict(plan.get("destroy_ctx", {}) or {})
    assignments = dict(plan.get("assignments", {}) or {})
    released_subtasks = dict(destroy_ctx.get("released_subtasks", {}) or {})
    if not released_subtasks or not assignments:
        return {"success": False}
    if str(plan.get("repair_name", "")) == "y_repair_station_rank_permutation":
        source_station_ids = sorted({int(station_id) for station_id, _ in released_subtasks.values() if int(station_id) >= 0})
        if len(source_station_ids) != 1:
            return {"success": False}
        station_id = int(source_station_ids[0])
        touched_subtask_ids = {
            int(row.subtask_id)
            for row in config.subtasks.values()
            if int(row.station_id) == int(station_id) or int(row.subtask_id) in set(int(x) for x in released_subtasks.keys())
        }
        candidate = config.clone_for_layer("Y", sorted(touched_subtask_ids))
        ranked_sequences = _enumerate_station_rank_permutations(
            opt,
            candidate,
            int(station_id),
            [int(x) for x in released_subtasks.keys()],
            max(8, int(getattr(opt.cfg, "resource_y_rank_permutation_cap", 24))),
        )
        if not ranked_sequences:
            return {"success": False}
        _apply_station_sequence(candidate, int(station_id), ranked_sequences[0][1])
        _normalize_station_ranks_subset(candidate, [int(station_id)])
        return {
            "success": True,
            "config": candidate,
            "score_cache": None,
            "affected_ids": set(int(x) for x in touched_subtask_ids),
            "fallback_used": bool(plan.get("fallback_used", False)),
            "projection_mode": "",
            "projection_repaired_subtask_count": 0,
            "validation_signature": candidate.validation_signature(),
        }
    source_station_ids = [int(station_id) for station_id, _ in released_subtasks.values() if int(station_id) >= 0]
    target_station_ids = [int(meta.get("station_id", -1)) for meta in assignments.values() if int(meta.get("station_id", -1)) >= 0]
    touched_station_ids = sorted(set(source_station_ids + target_station_ids))
    touched_subtask_ids = set(int(released_id) for released_id in released_subtasks.keys())
    for row in config.subtasks.values():
        if int(row.station_id) in set(touched_station_ids):
            touched_subtask_ids.add(int(row.subtask_id))
    candidate = config.clone_for_layer("Y", sorted(touched_subtask_ids))
    for subtask_id in released_subtasks.keys():
        row = candidate.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = -1
        row.station_rank = -1
    for subtask_id, meta in assignments.items():
        row = candidate.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = int(meta["station_id"])
        row.station_rank = int(meta["station_rank"])
    _normalize_station_ranks_subset(candidate, touched_station_ids)
    return {
        "success": True,
        "config": candidate,
        "score_cache": None,
        "affected_ids": set(int(x) for x in touched_subtask_ids),
        "fallback_used": bool(plan.get("fallback_used", False)),
        "projection_mode": "",
        "projection_repaired_subtask_count": 0,
        "validation_signature": candidate.validation_signature(),
    }


def y_repair_earliest_finish(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        _assign_station_rank(config, int(subtask_id), _choose_station_earliest_finish(opt, config, row, rng))
    _normalize_ranks(config)
    return {"success": True, "affected_subtask_ids": set(released)}


def y_repair_regret2_station(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    remaining = set(released)
    affected = set()
    while remaining:
        best_row = None
        for subtask_id in list(remaining):
            row = config.subtasks.get(int(subtask_id))
            if row is None:
                remaining.remove(int(subtask_id))
                continue
            choices = []
            for station_id in range(_station_count(opt)):
                loads = _station_loads(config)
                projected_finish = float(loads.get(int(station_id), 0.0) + _subtask_station_work(row))
                arrival = float(_arrival_proxy(opt, row, station_id))
                choices.append((projected_finish + 0.1 * arrival, int(station_id)))
            if not choices:
                continue
            ranked_choices = sorted(choices, key=lambda item: (item[0], item[1]))
            regret = float(ranked_choices[1][0] - ranked_choices[0][0]) if len(ranked_choices) >= 2 else float(ranked_choices[0][0])
            station_choice = pick_soft_greedy_min(rng, ranked_choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
            candidate = {
                "regret_score": -regret,
                "best_cost": float(ranked_choices[0][0]),
                "subtask_id": int(subtask_id),
                "station_choice": station_choice or ranked_choices[0],
            }
            if best_row is None or (float(candidate["regret_score"]), float(candidate["best_cost"]), int(candidate["subtask_id"])) < (
                float(best_row["regret_score"]),
                float(best_row["best_cost"]),
                int(best_row["subtask_id"]),
            ):
                best_row = candidate
        if best_row is None:
            break
        chosen_subtask_id = int(best_row["subtask_id"])
        chosen_station_id = int(best_row["station_choice"][1])
        _assign_station_rank(config, int(chosen_subtask_id), int(chosen_station_id))
        affected.add(int(chosen_subtask_id))
        remaining.remove(int(chosen_subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_arrival_aware_rank(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    affected = set()
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        choices = []
        for station_id in range(_station_count(opt)):
            arrival = float(_arrival_proxy(opt, row, station_id))
            load = float(_station_loads(config).get(int(station_id), 0.0))
            choices.append((arrival + 0.1 * load, int(station_id)))
        picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
        _assign_station_rank(config, int(subtask_id), int((picked or choices[0])[1]))
        affected.add(int(subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_load_balance(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    affected = set()
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        _assign_station_rank(config, int(subtask_id), _choose_station_load_balance(opt, config, row, rng))
        affected.add(int(subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_ejection_chain_balance(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released_subtasks = dict(ctx.get("released_subtasks", {}) or {})
    if not released_subtasks:
        return {"success": False}
    plan = _plan_y_assignments(opt, config, released_subtasks, "y_repair_ejection_chain_balance", rng)
    assignments = dict(plan.get("assignments", {}) or {})
    if not bool(plan.get("success", False)) or not assignments:
        return {"success": False}
    touched_station_ids = {
        int(station_id)
        for station_id, _rank in released_subtasks.values()
        if int(station_id) >= 0
    }
    for meta in assignments.values():
        station_id = int(meta.get("station_id", -1))
        if station_id >= 0:
            touched_station_ids.add(station_id)
    affected = set(int(x) for x in released_subtasks.keys())
    for subtask_id in released_subtasks.keys():
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = -1
        row.station_rank = -1
    for subtask_id, meta in assignments.items():
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = int(meta["station_id"])
        row.station_rank = int(meta["station_rank"])
        affected.add(int(subtask_id))
    _normalize_station_ranks_subset(config, sorted(touched_station_ids))
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_global_route_balance(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released_subtasks = dict(ctx.get("released_subtasks", {}) or {})
    if not released_subtasks:
        return {"success": False}
    assignments = _plan_y_global_route_balance(opt, config, list(released_subtasks.keys()), rng)
    if not assignments:
        return {"success": False}
    touched_station_ids = {
        int(station_id)
        for station_id, _rank in released_subtasks.values()
        if int(station_id) >= 0
    }
    affected = set(int(x) for x in released_subtasks.keys())
    for meta in assignments.values():
        station_id = int(meta.get("station_id", -1))
        if station_id >= 0:
            touched_station_ids.add(station_id)
    for subtask_id in released_subtasks.keys():
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = -1
        row.station_rank = -1
    for subtask_id, meta in assignments.items():
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = int(meta["station_id"])
        row.station_rank = int(meta["station_rank"])
    _normalize_station_ranks_subset(config, sorted(touched_station_ids))
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_greedy_fallback(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    affected = set()
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        station_id = _choose_station_earliest_finish(opt, config, row, rng)
        _assign_station_rank(config, int(subtask_id), int(station_id))
        affected.add(int(subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


Y_DESTROY_OPERATORS = {
    "y_destroy_congested_station_block": y_destroy_congested_station_block,
    "y_destroy_cross_station_fragment": y_destroy_cross_station_fragment,
    "y_destroy_rank_window_release": y_destroy_rank_window_release,
    "y_destroy_load_skew_release": y_destroy_load_skew_release,
    "y_destroy_random_release": y_destroy_random_release,
    "y_destroy_related_station_release": y_destroy_related_station_release,
    "y_destroy_related_robot_release": y_destroy_related_robot_release,
    "y_destroy_heavy_robot_tail": y_destroy_heavy_robot_tail,
    "y_destroy_initial_idle_head": y_destroy_initial_idle_head,
    "y_destroy_max_tardiness_blocker": y_destroy_max_tardiness_blocker,
    "y_destroy_critical_path_block": y_destroy_critical_path_block,
    "y_destroy_global_station_release": y_destroy_global_station_release,
}

Y_REPAIR_OPERATORS = {
    "y_repair_earliest_finish": y_repair_earliest_finish,
    "y_repair_regret2_station": y_repair_regret2_station,
    "y_repair_arrival_aware_rank": y_repair_arrival_aware_rank,
    "y_repair_load_balance": y_repair_load_balance,
    "y_repair_station_rank_permutation": y_repair_arrival_aware_rank,
    "y_repair_ejection_chain_balance": y_repair_ejection_chain_balance,
    "y_repair_global_route_balance": y_repair_global_route_balance,
}

Y_FALLBACK_OPERATOR = "y_repair_greedy_fallback"
