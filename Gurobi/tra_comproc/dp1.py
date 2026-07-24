from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

from Gurobi.tra_comproc.types import DP1RouteResult


def _name(variable: Any) -> str:
    return str(variable.VarName)


def _selected(values: Mapping[str, float], variable: Any) -> bool:
    return float(values.get(_name(variable), 0.0)) > 0.5


def evaluate_dp1_route(
    values_by_name: Mapping[str, float],
    payload: Mapping[str, Any],
) -> DP1RouteResult:
    """Read the no-wait route solution into the paper's DP1 path output."""

    if not bool(payload.get("integrate_u_route", False)):
        return DP1RouteResult(
            feasible=True,
            route_end_sec=0.0,
            slot_arrival_lower={},
            robot_paths={},
        )
    route_arc = payload.get("route_arc") or {}
    route_time = payload.get("route_time") or {}
    pass_x = payload.get("pass_x") or {}
    route_finish = payload.get("route_finish") or {}
    start_nodes = {
        int(robot_id): int(node_id)
        for robot_id, node_id in dict(payload.get("route_start_nodes", {}) or {}).items()
    }
    end_nodes = {
        int(robot_id): int(node_id)
        for robot_id, node_id in dict(payload.get("route_end_nodes", {}) or {}).items()
    }
    robot_ids = sorted(set(start_nodes) | set(end_nodes))
    errors: list[str] = []
    selected_arcs = {
        (int(key[0]), int(key[1]))
        for key, variable in route_arc.items()
        if _selected(values_by_name, variable)
    }
    selected_nodes_by_robot: dict[int, set[int]] = defaultdict(set)
    for key, variable in pass_x.items():
        if _selected(values_by_name, variable):
            selected_nodes_by_robot[int(key[1])].add(int(key[0]))

    robot_paths: dict[int, tuple[int, ...]] = {}
    for robot_id in robot_ids:
        start = start_nodes.get(robot_id)
        end = end_nodes.get(robot_id)
        if start is None or end is None:
            errors.append("DP1_MISSING_DEPOT")
            continue
        owned = set(selected_nodes_by_robot.get(robot_id, set())) | {start, end}
        successors: dict[int, list[int]] = defaultdict(list)
        for left, right in selected_arcs:
            if left in owned and right in owned:
                successors[left].append(right)
        path = [start]
        seen = {start}
        current = start
        while current != end:
            next_nodes = sorted(successors.get(current, []))
            if len(next_nodes) != 1:
                errors.append("DP1_ROUTE_DEGREE")
                break
            current = int(next_nodes[0])
            if current in seen:
                errors.append("DP1_ROUTE_CYCLE")
                break
            seen.add(current)
            path.append(current)
            if len(path) > len(owned) + 1:
                errors.append("DP1_ROUTE_OVERFLOW")
                break
        if current == end and not (owned - set(path)):
            robot_paths[robot_id] = tuple(path)
        elif current == end:
            errors.append("DP1_DISCONNECTED_NODES")

    slot_arrival: dict[int, float] = defaultdict(float)
    for spec in dict(payload.get("route_tasks", {}) or {}).values():
        delivery_node = int(getattr(spec, "delivery_node", -1))
        slot_id = int(getattr(spec, "slot_id", -1))
        variable = route_time.get(delivery_node)
        if variable is None or slot_id < 0:
            continue
        value = float(values_by_name.get(_name(variable), 0.0))
        if not math.isfinite(value) or value < -1e-6:
            errors.append("DP1_INVALID_ROUTE_TIME")
            continue
        slot_arrival[slot_id] = max(float(slot_arrival[slot_id]), value)

    route_end = 0.0
    for variable in route_finish.values():
        value = float(values_by_name.get(_name(variable), 0.0))
        if math.isfinite(value):
            route_end = max(route_end, value)
    return DP1RouteResult(
        feasible=not errors,
        route_end_sec=float(route_end),
        slot_arrival_lower=dict(slot_arrival),
        robot_paths=robot_paths,
        error_codes=tuple(sorted(set(errors))),
    )
