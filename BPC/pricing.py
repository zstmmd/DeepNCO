from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Sequence, Tuple

from config.ofs_config import OFSConfig

from BPC.models import BPCRouteColumn, BPCRouteTask, PricingResult


def manhattan(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))


def travel_time(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    speed = float(getattr(OFSConfig, "ROBOT_SPEED", 1.0) or 1.0)
    if speed <= 0.0:
        speed = 1.0
    return manhattan(a, b) / speed


def validate_column(column: BPCRouteColumn, tasks_by_key: Dict[int, BPCRouteTask], robot_capacity: int) -> Tuple[bool, str]:
    seen = set()
    current_time = 0.0
    for task_key in column.sequence:
        if int(task_key) in seen:
            return False, f"duplicate_task:{task_key}"
        task = tasks_by_key.get(int(task_key))
        if task is None:
            return False, f"unknown_task:{task_key}"
        if int(task.load) > int(robot_capacity):
            return False, f"capacity_violation:{task_key}"
        current_time += float(task.service_time)
        arrival = float(column.arrival_at_station.get(int(task_key), -1.0))
        if arrival + 1e-9 < current_time:
            return False, f"time_inconsistent:{task_key}"
        current_time = max(current_time, arrival)
        seen.add(int(task_key))
    if float(column.finish_time) + 1e-9 < current_time:
        return False, "finish_before_last_arrival"
    return True, ""


class LabelSettingPricer:
    """Exact resource-constrained route pricing over the supplied task universe.

    The pricer enumerates elementary pickup-delivery sequences for one robot. It
    is exact only when the search exhausts without hitting time or label limits.
    """

    def __init__(
        self,
        robot_id: int,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float] | None = None,
        robot_capacity: int | None = None,
    ) -> None:
        self.robot_id = int(robot_id)
        self.start_xy = (float(start_xy[0]), float(start_xy[1]))
        self.end_xy = self.start_xy if end_xy is None else (float(end_xy[0]), float(end_xy[1]))
        self.robot_capacity = int(robot_capacity or getattr(OFSConfig, "ROBOT_CAPACITY", 8))

    def price(
        self,
        tasks: Sequence[BPCRouteTask],
        dual_task_cover: Dict[int, float],
        existing_column_count: int = 0,
        time_limit_sec: float = 30.0,
        max_labels: int = 200000,
        reduced_cost_tol: float = 1e-9,
    ) -> PricingResult:
        start = time.perf_counter()
        task_list = [task for task in tasks if int(task.load) <= self.robot_capacity]
        by_key = {int(task.task_key): task for task in task_list}
        best_columns: List[BPCRouteColumn] = []
        best_reduced_cost = 0.0
        timed_out = False
        label_limit_hit = False
        expanded = 0

        # Label tuple: current_xy, elapsed_time, cost, reduced_cost, sequence, arrivals
        stack: List[Tuple[Tuple[float, float], float, float, float, Tuple[int, ...], Dict[int, float]]] = [
            (self.start_xy, 0.0, 0.0, 0.0, tuple(), {})
        ]
        while stack:
            if time.perf_counter() - start > float(time_limit_sec):
                timed_out = True
                break
            if expanded >= int(max_labels):
                label_limit_hit = True
                break
            current_xy, elapsed, cost, red_cost, sequence, arrivals = stack.pop()
            expanded += 1
            used = set(int(x) for x in sequence)
            if sequence:
                finish = elapsed + travel_time(current_xy, self.end_xy)
                total_cost = cost + travel_time(current_xy, self.end_xy)
                total_red = red_cost + travel_time(current_xy, self.end_xy)
                if total_red < best_reduced_cost:
                    best_reduced_cost = float(total_red)
                if total_red < -float(reduced_cost_tol):
                    best_columns.append(
                        BPCRouteColumn(
                            column_id=int(existing_column_count + len(best_columns)),
                            robot_id=self.robot_id,
                            task_keys=tuple(int(x) for x in sequence),
                            sequence=tuple(int(x) for x in sequence),
                            arrival_at_station=dict(arrivals),
                            finish_time=float(finish),
                            travel_time=float(total_cost - sum(float(by_key[int(k)].service_time) for k in sequence)),
                            service_time=float(sum(float(by_key[int(k)].service_time) for k in sequence)),
                            reduced_cost=float(total_red),
                        )
                    )
            for task in task_list:
                key = int(task.task_key)
                if key in used:
                    continue
                to_pickup = travel_time(current_xy, task.pickup_xy)
                pickup_done = elapsed + to_pickup + float(task.service_time)
                to_delivery = travel_time(task.pickup_xy, task.delivery_xy)
                delivered = pickup_done + to_delivery
                new_cost = cost + to_pickup + float(task.service_time) + to_delivery
                new_red = red_cost + to_pickup + float(task.service_time) + to_delivery - float(dual_task_cover.get(key, 0.0) or 0.0)
                new_arrivals = dict(arrivals)
                new_arrivals[key] = float(delivered)
                stack.append((task.delivery_xy, delivered, new_cost, new_red, tuple(list(sequence) + [key]), new_arrivals))

        exact = bool(not timed_out and not label_limit_hit)
        return PricingResult(
            columns=best_columns,
            exact=exact,
            timed_out=timed_out,
            label_limit_hit=label_limit_hit,
            expanded_labels=int(expanded),
            best_reduced_cost=float(best_reduced_cost),
        )


def build_single_task_columns(
    tasks: Iterable[BPCRouteTask],
    robot_id: int,
    start_xy: Tuple[float, float],
    first_column_id: int = 0,
) -> List[BPCRouteColumn]:
    columns: List[BPCRouteColumn] = []
    for offset, task in enumerate(tasks):
        to_pickup = travel_time(start_xy, task.pickup_xy)
        to_delivery = travel_time(task.pickup_xy, task.delivery_xy)
        finish = to_pickup + float(task.service_time) + to_delivery + travel_time(task.delivery_xy, start_xy)
        columns.append(
            BPCRouteColumn(
                column_id=int(first_column_id + offset),
                robot_id=int(robot_id),
                task_keys=(int(task.task_key),),
                sequence=(int(task.task_key),),
                arrival_at_station={int(task.task_key): float(to_pickup + float(task.service_time) + to_delivery)},
                finish_time=float(finish),
                travel_time=float(to_pickup + to_delivery + travel_time(task.delivery_xy, start_xy)),
                service_time=float(task.service_time),
            )
        )
    return columns
