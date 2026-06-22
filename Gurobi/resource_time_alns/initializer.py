from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

from .state import ResourceConfig, ResourceSubtask, WorkUnitInfo, ZTaskDescriptor


def build_resource_config_from_problem(opt, problem=None) -> ResourceConfig:
    problem = problem if problem is not None else opt.problem
    assert problem is not None
    work_units: Dict[str, WorkUnitInfo] = {}
    available_by_order_sku: Dict[Tuple[int, int], deque] = {}
    capacity_limits: Dict[int, int] = {}

    for order in getattr(problem, "order_list", []) or []:
        order_id = int(getattr(order, "order_id", -1))
        sku_seen: Dict[int, int] = defaultdict(int)
        queue_map: Dict[int, List[str]] = defaultdict(list)
        for sku_id in getattr(order, "order_product_id_list", []) or []:
            sku_id = int(sku_id)
            occ = int(sku_seen[sku_id])
            sku_seen[sku_id] += 1
            work_unit_id = f"{order_id}:{sku_id}:{occ}"
            work_units[work_unit_id] = WorkUnitInfo(
                work_unit_id=work_unit_id,
                order_id=order_id,
                sku_id=sku_id,
                occurrence_index=occ,
                origin_subtask_id=-1,
            )
            queue_map[sku_id].append(work_unit_id)
        for sku_id, rows in queue_map.items():
            available_by_order_sku[(order_id, int(sku_id))] = deque(rows)
        capacity_limits[order_id] = int(max(1, getattr(opt.sp1, "order_capacity_limits", {}).get(order_id, 0) or 0))

    subtasks: Dict[int, ResourceSubtask] = {}
    route_rows_by_robot: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    local_slot_index_by_subtask: Dict[int, int] = {}
    local_slot_count_by_order: Dict[int, int] = defaultdict(int)
    next_task_id = 1
    for st in sorted(getattr(problem, "subtask_list", []) or [], key=lambda row: int(getattr(row, "id", -1))):
        subtask_id = int(getattr(st, "id", -1))
        order_id = int(getattr(getattr(st, "parent_order", None), "order_id", -1))
        local_slot_index = int(local_slot_count_by_order[int(order_id)])
        local_slot_count_by_order[int(order_id)] += 1
        local_slot_index_by_subtask[int(subtask_id)] = int(local_slot_index)
        work_unit_ids: List[str] = []
        for sku in getattr(st, "sku_list", []) or []:
            sku_id = int(getattr(sku, "id", -1))
            pool = available_by_order_sku.get((order_id, sku_id))
            if pool is None or not pool:
                synthetic_idx = sum(1 for key in work_units if key.startswith(f"{order_id}:{sku_id}:"))
                work_unit_id = f"{order_id}:{sku_id}:{synthetic_idx}"
                work_units[work_unit_id] = WorkUnitInfo(
                    work_unit_id=work_unit_id,
                    order_id=order_id,
                    sku_id=sku_id,
                    occurrence_index=synthetic_idx,
                    origin_subtask_id=subtask_id,
                )
            else:
                work_unit_id = str(pool.popleft())
                work_units[work_unit_id].origin_subtask_id = subtask_id
            work_unit_ids.append(work_unit_id)

        descriptors: List[ZTaskDescriptor] = []
        for task in sorted(getattr(st, "execution_tasks", []) or [], key=lambda row: int(getattr(row, "task_id", -1))):
            task_id = int(getattr(task, "task_id", next_task_id))
            next_task_id = max(next_task_id, task_id + 1)
            robot_id = int(getattr(task, "robot_id", -1))
            if robot_id >= 0:
                route_rows_by_robot[int(robot_id)].append(
                    {
                        "subtask_id": int(subtask_id),
                        "order_id": int(order_id),
                        "local_slot_index": int(local_slot_index_by_subtask.get(int(subtask_id), local_slot_index)),
                        "task_id": int(task_id),
                        "stack_id": int(getattr(task, "target_stack_id", -1)),
                        "station_id": int(getattr(st, "assigned_station_id", -1)),
                        "trip_id": int(getattr(task, "trip_id", 0) or 0),
                        "arrival_stack": float(getattr(task, "arrival_time_at_stack", 0.0) or 0.0),
                        "arrival_station": float(getattr(task, "arrival_time_at_station", 0.0) or 0.0),
                    }
                )
            sort_range = getattr(task, "sort_layer_range", None)
            descriptors.append(
                ZTaskDescriptor(
                    task_id=task_id,
                    stack_id=int(getattr(task, "target_stack_id", -1)),
                    mode=str(getattr(task, "operation_mode", "FLIP")).upper(),
                    target_tote_ids=tuple(int(x) for x in (getattr(task, "target_tote_ids", []) or [])),
                    hit_tote_ids=tuple(int(x) for x in (getattr(task, "hit_tote_ids", []) or [])),
                    noise_tote_ids=tuple(int(x) for x in (getattr(task, "noise_tote_ids", []) or [])),
                    sort_layer_range=None if sort_range is None else (int(sort_range[0]), int(sort_range[1])),
                    station_service_time=float(getattr(task, "station_service_time", 0.0)),
                    robot_service_time=float(getattr(task, "robot_service_time", 0.0)),
                    sku_pick_count=int(getattr(task, "sku_pick_count", 0) or 0),
                )
            )

        subtasks[subtask_id] = ResourceSubtask(
            subtask_id=subtask_id,
            order_id=order_id,
            work_unit_ids=tuple(sorted(work_unit_ids)),
            station_id=int(getattr(st, "assigned_station_id", -1)),
            station_rank=int(getattr(st, "station_sequence_rank", -1)),
            z_tasks=descriptors,
            origin_group_ids=(f"st_{subtask_id}",),
        )

    next_subtask_id = max([1] + [int(subtask_id) + 1 for subtask_id in subtasks.keys()])
    if not any(int(v) > 0 for v in capacity_limits.values()):
        order_unit_counts: Dict[int, int] = defaultdict(int)
        for work_unit in work_units.values():
            order_unit_counts[int(work_unit.order_id)] += 1
        for order_id, total_units in order_unit_counts.items():
            count = len([st for st in subtasks.values() if int(st.order_id) == int(order_id)])
            capacity_limits[int(order_id)] = int(max(1, round(total_units / max(1, count))))

    config = ResourceConfig(
        work_units=work_units,
        subtasks=subtasks,
        capacity_limits=capacity_limits,
        next_subtask_id=next_subtask_id,
        next_task_id=next_task_id,
    ).rebuild_indices()
    config.metadata["tote_list"] = list(getattr(problem, "tote_list", []) or [])
    if route_rows_by_robot:
        config.metadata["fixed_route_task_sequence_by_robot"] = {
            int(robot_id): sorted(
                rows,
                key=lambda row: (
                    int(row.get("trip_id", 0) or 0),
                    float(row.get("arrival_stack", 0.0) or 0.0),
                    int(row.get("task_id", 0) or 0),
                ),
            )
            for robot_id, rows in sorted(route_rows_by_robot.items())
            if rows
        }
        config.metadata["fixed_route_node_sequence_by_robot"] = {
            int(robot_id): [
                {
                    "kind": str(kind),
                    "subtask_id": int(row.get("subtask_id", -1)),
                    "order_id": int(row.get("order_id", -1)),
                    "local_slot_index": int(row.get("local_slot_index", -1)),
                    "task_id": int(row.get("task_id", -1)),
                    "stack_id": int(row.get("stack_id", -1)),
                    "station_id": int(row.get("station_id", -1)),
                    "time": float(time_value),
                }
                for time_value, kind, row in sorted(
                    [
                        (float(row.get("arrival_stack", 0.0) or 0.0), "pickup", row)
                        for row in rows
                    ]
                    + [
                        (float(row.get("arrival_station", 0.0) or 0.0), "delivery", row)
                        for row in rows
                    ],
                    key=lambda item: (
                        float(item[0]),
                        0 if str(item[1]) == "pickup" else 1,
                        int(item[2].get("task_id", 0) or 0),
                    ),
                )
            ]
            for robot_id, rows in sorted(route_rows_by_robot.items())
            if rows
        }
    return config


def build_initial_resource_config(opt) -> ResourceConfig:
    return build_resource_config_from_problem(opt, getattr(opt, "problem", None))
