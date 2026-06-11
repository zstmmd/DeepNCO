from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from config.ofs_config import OFSConfig

from .state import ResourceConfig, ResourceSubtask


@dataclass
class URoutePlan:
    fixed_route_task_sequence_by_robot: Dict[int, List[Dict[str, Any]]]
    u_fast_cmax: float
    u_route_lb: float
    changed_robot_ids: List[int] = field(default_factory=list)
    u_repair_time: float = 0.0
    feasible: bool = True
    reason: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "fixed_route_task_sequence_by_robot": self.fixed_route_task_sequence_by_robot,
            "u_fast_cmax": float(self.u_fast_cmax),
            "u_route_lb": float(self.u_route_lb),
            "u_changed_robot_ids": [int(x) for x in self.changed_robot_ids],
            "u_changed_robot_count": int(len(set(int(x) for x in self.changed_robot_ids))),
            "u_repair_time": float(self.u_repair_time),
            "u_repair_feasible": bool(self.feasible),
            "u_repair_reason": str(self.reason),
        }


def _xy(point: Any) -> Tuple[float, float]:
    return float(getattr(point, "x", 0.0) or 0.0), float(getattr(point, "y", 0.0) or 0.0)


def _manhattan(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    return float(abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1])))


def _station_xy(problem: Any, station_id: int) -> Tuple[float, float]:
    stations = list(getattr(problem, "station_list", []) or [])
    if 0 <= int(station_id) < len(stations):
        return _xy(getattr(stations[int(station_id)], "point", None))
    return 0.0, 0.0


def _stack_xy(problem: Any, stack_id: int) -> Tuple[float, float]:
    stack = dict(getattr(problem, "point_to_stack", {}) or {}).get(int(stack_id))
    if stack is None:
        return 0.0, 0.0
    return _xy(getattr(stack, "store_point", None))


def _robot_start_rows(problem: Any) -> List[Tuple[int, Tuple[float, float]]]:
    rows: List[Tuple[int, Tuple[float, float]]] = []
    for robot in getattr(problem, "robot_list", []) or []:
        robot_id = int(getattr(robot, "id", -1))
        if robot_id < 0:
            continue
        rows.append((int(robot_id), _xy(getattr(robot, "start_point", None))))
    if not rows:
        rows.append((0, (0.0, 0.0)))
    rows.sort(key=lambda item: int(item[0]))
    return rows


def _task_service_time(descriptor: Any) -> float:
    station_service = float(getattr(descriptor, "station_service_time", 0.0) or 0.0)
    robot_service = float(getattr(descriptor, "robot_service_time", 0.0) or 0.0)
    pick_count = max(1, int(getattr(descriptor, "sku_pick_count", 0) or 0))
    return float(station_service + robot_service + pick_count * float(getattr(OFSConfig, "PICKING_TIME", 1.0)))


def _ordered_subtasks(config: ResourceConfig) -> List[ResourceSubtask]:
    return sorted(
        list(config.subtasks.values()),
        key=lambda row: (
            int(row.station_rank if int(row.station_rank) >= 0 else 10**9),
            int(row.station_id if int(row.station_id) >= 0 else 10**9),
            int(row.order_id),
            int(row.subtask_id),
        ),
    )


def _local_slot_index_by_subtask(config: ResourceConfig) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    by_order: Dict[int, List[ResourceSubtask]] = {}
    for row in config.subtasks.values():
        by_order.setdefault(int(row.order_id), []).append(row)
    for order_id, rows in by_order.items():
        rows.sort(key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)))
        for idx, row in enumerate(rows):
            out[int(row.subtask_id)] = (int(order_id), int(idx))
    return out


def _route_rows_for_config(problem: Any, config: ResourceConfig) -> List[Dict[str, Any]]:
    slot_lookup = _local_slot_index_by_subtask(config)
    rows_by_key: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for subtask in _ordered_subtasks(config):
        station_id = int(subtask.station_id)
        if station_id < 0:
            continue
        order_id, local_idx = slot_lookup.get(int(subtask.subtask_id), (int(subtask.order_id), 0))
        for descriptor in subtask.z_tasks or []:
            stack_id = int(getattr(descriptor, "stack_id", -1))
            if stack_id < 0:
                continue
            stack_xy = _stack_xy(problem, int(stack_id))
            station_xy = _station_xy(problem, int(station_id))
            key = (int(order_id), int(local_idx), int(stack_id), int(station_id))
            if key not in rows_by_key:
                rows_by_key[key] = {
                    "subtask_id": int(subtask.subtask_id),
                    "order_id": int(order_id),
                    "local_slot_index": int(local_idx),
                    "group_key": (int(order_id), int(local_idx)),
                    "stack_id": int(stack_id),
                    "station_id": int(station_id),
                    "stack_xy": stack_xy,
                    "station_xy": station_xy,
                    "service_time": float(_task_service_time(descriptor)),
                    "sort_key": (int(subtask.station_rank), int(station_id), int(subtask.order_id), int(subtask.subtask_id)),
                }
            else:
                rows_by_key[key]["service_time"] = float(rows_by_key[key].get("service_time", 0.0) or 0.0) + float(_task_service_time(descriptor))
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: tuple(row["sort_key"]))
    return rows


def _append_route_row(state: Dict[str, Any], task: Dict[str, Any]) -> None:
        travel_to_stack = _manhattan(tuple(state["xy"]), tuple(task["stack_xy"]))
        travel_to_station = _manhattan(tuple(task["stack_xy"]), tuple(task["station_xy"]))
        arrival_stack = float(state["time"]) + float(travel_to_stack)
        arrival_station = float(arrival_stack) + float(travel_to_station)
        finish = float(arrival_station) + float(task["service_time"])
        seq = list(state["seq"])
        seq.append(
            {
                "sequence": int(len(seq)),
                "trip_id": int(len(seq)),
                "order_id": int(task["order_id"]),
                "local_slot_index": int(task["local_slot_index"]),
                "subtask_id": int(task["subtask_id"]),
                "stack_id": int(task["stack_id"]),
                "station_id": int(task["station_id"]),
                "arrival_stack": float(arrival_stack),
                "arrival_station": float(arrival_station),
                "finish_time": float(finish),
            }
        )
        state["seq"] = seq
        state["time"] = float(finish)
        state["xy"] = tuple(task["station_xy"])


def _slot_groups(task_rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for task in task_rows:
        key = tuple(task.get("group_key", (int(task.get("order_id", -1)), int(task.get("local_slot_index", -1)))))
        grouped.setdefault((int(key[0]), int(key[1])), []).append(dict(task))
    groups = list(grouped.values())
    groups.sort(key=lambda rows: min(tuple(row["sort_key"]) for row in rows))
    for rows in groups:
        rows.sort(key=lambda row: tuple(row["sort_key"]))
    return groups


def _initial_dispatch_assignments(problem: Any, task_rows: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    robot_state: Dict[int, Dict[str, Any]] = {
        int(robot_id): {"time": 0.0, "xy": tuple(xy), "seq": []}
        for robot_id, xy in _robot_start_rows(problem)
    }
    assignments: Dict[int, List[Dict[str, Any]]] = {int(robot_id): [] for robot_id in robot_state.keys()}
    for group in _slot_groups(task_rows):
        first_task = group[0]
        best_robot_id = min(
            robot_state.keys(),
            key=lambda rid: (
                float(robot_state[rid]["time"])
                + _manhattan(tuple(robot_state[rid]["xy"]), tuple(first_task["stack_xy"]))
                + sum(
                    _manhattan(tuple(task["stack_xy"]), tuple(task["station_xy"])) + float(task.get("service_time", 0.0) or 0.0)
                    for task in group
                ),
                float(robot_state[rid]["time"]),
                int(rid),
            ),
        )
        for task in group:
            assignments[int(best_robot_id)].append(dict(task))
            _append_route_row(robot_state[int(best_robot_id)], task)
    return assignments


def _schedule_assignments(
    problem: Any,
    assignments: Dict[int, List[Dict[str, Any]]],
) -> Tuple[Dict[int, List[Dict[str, Any]]], float, Dict[int, float]]:
    start_xy = {int(robot_id): tuple(xy) for robot_id, xy in _robot_start_rows(problem)}
    robot_ids = sorted(set(start_xy.keys()) | set(int(x) for x in assignments.keys()))
    robot_state: Dict[int, Dict[str, Any]] = {
        int(robot_id): {"time": 0.0, "xy": tuple(start_xy.get(int(robot_id), (0.0, 0.0))), "seq": [], "idx": 0}
        for robot_id in robot_ids
    }
    station_available: Dict[int, float] = {}
    remaining = sum(len(rows or []) for rows in assignments.values())
    while remaining > 0:
        candidates: List[Tuple[float, float, int, Dict[str, Any]]] = []
        for robot_id in robot_ids:
            idx = int(robot_state[int(robot_id)]["idx"])
            rows = list(assignments.get(int(robot_id), []) or [])
            if idx >= len(rows):
                continue
            task = rows[idx]
            arrival_stack = float(robot_state[int(robot_id)]["time"]) + _manhattan(
                tuple(robot_state[int(robot_id)]["xy"]),
                tuple(task["stack_xy"]),
            )
            arrival_station = float(arrival_stack) + _manhattan(tuple(task["stack_xy"]), tuple(task["station_xy"]))
            candidates.append((float(arrival_station), float(arrival_stack), int(robot_id), task))
        if not candidates:
            break
        arrival_station, arrival_stack, robot_id, task = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        station_id = int(task["station_id"])
        start_service = max(float(arrival_station), float(station_available.get(station_id, 0.0)))
        finish = float(start_service) + float(task["service_time"])
        seq = list(robot_state[int(robot_id)]["seq"])
        seq.append(
            {
                "sequence": int(len(seq)),
                "trip_id": int(len(seq)),
                "order_id": int(task["order_id"]),
                "local_slot_index": int(task["local_slot_index"]),
                "subtask_id": int(task["subtask_id"]),
                "stack_id": int(task["stack_id"]),
                "station_id": int(station_id),
                "arrival_stack": float(arrival_stack),
                "arrival_station": float(arrival_station),
                "start_service": float(start_service),
                "finish_time": float(finish),
            }
        )
        robot_state[int(robot_id)]["seq"] = seq
        robot_state[int(robot_id)]["time"] = float(finish)
        robot_state[int(robot_id)]["xy"] = tuple(task["station_xy"])
        robot_state[int(robot_id)]["idx"] = int(robot_state[int(robot_id)]["idx"]) + 1
        station_available[int(station_id)] = float(finish)
        remaining -= 1
    route_by_robot: Dict[int, List[Dict[str, Any]]] = {
        int(rid): list(state["seq"])
        for rid, state in robot_state.items()
        if state["seq"]
    }
    finish_by_robot: Dict[int, float] = {int(rid): float(state["time"]) for rid, state in robot_state.items()}
    cmax = max(list(finish_by_robot.values()) + [0.0])
    return route_by_robot, float(cmax), finish_by_robot


def _candidate_insert_positions(length: int) -> List[int]:
    if length <= 8:
        return list(range(0, int(length) + 1))
    rows = {0, int(length), max(0, int(length) // 2), max(0, int(length) - 1)}
    return sorted(rows)


def _improve_assignments(
    problem: Any,
    assignments: Dict[int, List[Dict[str, Any]]],
    *,
    max_moves: int,
    time_limit_sec: float,
    neighborhood_robots: int,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]], float, Dict[int, float]]:
    deadline = time.perf_counter() + max(0.0, float(time_limit_sec))
    best_assignments = {int(rid): [dict(task) for task in rows] for rid, rows in assignments.items()}
    best_route, best_cmax, best_finish = _schedule_assignments(problem, best_assignments)
    moves = 0
    while moves < int(max_moves) and time.perf_counter() < deadline:
        heavy_robot = max(best_finish.keys(), key=lambda rid: (float(best_finish.get(rid, 0.0)), int(rid)))
        heavy_rows = list(best_assignments.get(int(heavy_robot), []) or [])
        if len(heavy_rows) <= 1:
            break
        target_robots = sorted(
            [rid for rid in best_finish.keys() if int(rid) != int(heavy_robot)],
            key=lambda rid: (float(best_finish.get(rid, 0.0)), int(rid)),
        )[: max(1, int(neighborhood_robots))]
        best_trial = None
        best_trial_cmax = float(best_cmax)
        group_positions: Dict[Tuple[int, int], List[int]] = {}
        for pos, row in enumerate(heavy_rows):
            key = tuple(row.get("group_key", (int(row.get("order_id", -1)), int(row.get("local_slot_index", -1)))))
            group_positions.setdefault((int(key[0]), int(key[1])), []).append(int(pos))
        candidate_groups = sorted(
            group_positions.items(),
            key=lambda item: max(item[1]),
            reverse=True,
        )[:6]
        for _group_key, positions in candidate_groups:
            moving_group = [dict(heavy_rows[int(pos)]) for pos in positions]
            for target_robot in target_robots:
                target_len = len(best_assignments.get(int(target_robot), []) or [])
                for insert_pos in _candidate_insert_positions(int(target_len)):
                    trial = {int(rid): [dict(row) for row in rows] for rid, rows in best_assignments.items()}
                    for pos in sorted(positions, reverse=True):
                        trial[int(heavy_robot)].pop(int(pos))
                    target_rows = trial.setdefault(int(target_robot), [])
                    for offset, moved in enumerate(moving_group):
                        target_rows.insert(int(insert_pos) + int(offset), dict(moved))
                    _, trial_cmax, _ = _schedule_assignments(problem, trial)
                    if float(trial_cmax) + 1e-9 < float(best_trial_cmax):
                        best_trial = trial
                        best_trial_cmax = float(trial_cmax)
        if best_trial is None:
            for pos in range(max(0, len(heavy_rows) - 5), len(heavy_rows) - 1):
                left_key = heavy_rows[int(pos)].get("group_key", None)
                right_key = heavy_rows[int(pos) + 1].get("group_key", None)
                if left_key == right_key:
                    continue
                if len(group_positions.get(tuple(left_key), [])) != 1 or len(group_positions.get(tuple(right_key), [])) != 1:
                    continue
                trial = {int(rid): [dict(row) for row in rows] for rid, rows in best_assignments.items()}
                trial[int(heavy_robot)][int(pos)], trial[int(heavy_robot)][int(pos) + 1] = (
                    trial[int(heavy_robot)][int(pos) + 1],
                    trial[int(heavy_robot)][int(pos)],
                )
                _, trial_cmax, _ = _schedule_assignments(problem, trial)
                if float(trial_cmax) + 1e-9 < float(best_trial_cmax):
                    best_trial = trial
                    best_trial_cmax = float(trial_cmax)
        if best_trial is None:
            break
        best_assignments = best_trial
        best_route, best_cmax, best_finish = _schedule_assignments(problem, best_assignments)
        moves += 1
    return best_assignments, best_route, float(best_cmax), best_finish
    route_by_robot = {int(rid): list(state["seq"]) for rid, state in robot_state.items() if state["seq"]}
    cmax = max([float(state["time"]) for state in robot_state.values()] + [0.0])
    return route_by_robot, float(cmax)


def _route_lb(problem: Any, task_rows: Sequence[Dict[str, Any]]) -> float:
    robot_count = max(1, len(_robot_start_rows(problem)))
    total_service = float(sum(float(row.get("service_time", 0.0) or 0.0) for row in task_rows))
    total_trip = float(
        sum(_manhattan(tuple(row["stack_xy"]), tuple(row["station_xy"])) for row in task_rows)
    )
    return float((total_service + total_trip) / float(robot_count))


def _route_signature(route_by_robot: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Tuple[Tuple[int, int, int], ...]]:
    return {
        int(robot_id): tuple(
            (int(row.get("order_id", -1)), int(row.get("local_slot_index", -1)), int(row.get("stack_id", -1)))
            for row in rows
        )
        for robot_id, rows in dict(route_by_robot or {}).items()
    }


class UFastRepairSolver:
    def __init__(self, opt) -> None:
        self.opt = opt

    def repair(
        self,
        config: ResourceConfig,
        *,
        previous_route_plan: Dict[int, List[Dict[str, Any]]] | None = None,
        affected_subtask_ids: Iterable[int] | None = None,
    ) -> URoutePlan:
        start = time.perf_counter()
        problem = getattr(self.opt, "problem", None)
        if problem is None:
            return URoutePlan({}, float("inf"), float("inf"), feasible=False, reason="missing_problem")
        task_rows = _route_rows_for_config(problem, config)
        if not task_rows:
            return URoutePlan({}, float("inf"), float("inf"), feasible=False, reason="empty_route_tasks")
        assignments = _initial_dispatch_assignments(problem, task_rows)
        _assignments, route_by_robot, cmax, _finish_by_robot = _improve_assignments(
            problem,
            assignments,
            max_moves=int(getattr(self.opt.cfg, "u_repair_max_local_moves", 200) or 200),
            time_limit_sec=float(getattr(self.opt.cfg, "u_repair_time_limit_sec", 5.0) or 5.0),
            neighborhood_robots=int(getattr(self.opt.cfg, "u_repair_neighborhood_robots", 3) or 3),
        )
        lb = min(float(cmax), float(_route_lb(problem, task_rows)))
        prev_sig = _route_signature(previous_route_plan or {})
        new_sig = _route_signature(route_by_robot)
        changed_robot_ids = sorted(
            set(new_sig.keys()) | set(prev_sig.keys())
            if not prev_sig
            else {rid for rid in (set(new_sig.keys()) | set(prev_sig.keys())) if new_sig.get(rid) != prev_sig.get(rid)}
        )
        if affected_subtask_ids:
            affected = {int(x) for x in affected_subtask_ids}
            touched = set()
            for robot_id, rows in route_by_robot.items():
                if any(int(row.get("subtask_id", -1)) in affected for row in rows):
                    touched.add(int(robot_id))
            if touched:
                changed_robot_ids = sorted(set(changed_robot_ids) | touched)
        return URoutePlan(
            fixed_route_task_sequence_by_robot=route_by_robot,
            u_fast_cmax=float(cmax),
            u_route_lb=float(lb),
            changed_robot_ids=[int(x) for x in changed_robot_ids],
            u_repair_time=float(time.perf_counter() - start),
            feasible=bool(math.isfinite(float(cmax))),
            reason="dispatch_local_repair",
        )
