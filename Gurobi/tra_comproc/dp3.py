from __future__ import annotations

import math
import statistics
from collections import defaultdict

from Gurobi.tra_comproc.types import DP2ServiceResult, DP3RecoveryResult


def evaluate_dp3_recovery(
    dp2: DP2ServiceResult,
    *,
    no_wait_cmax_floor: float = 0.0,
) -> DP3RecoveryResult:
    """Estimate route-recourse potential while constructing station FCFS orders."""

    intervals_by_station: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    errors: list[str] = []
    for slot_id, station_id in dp2.station_by_slot.items():
        arrival = float(dp2.slot_arrival.get(int(slot_id), float("nan")))
        duration = float(dp2.slot_process_duration.get(int(slot_id), float("nan")))
        if not math.isfinite(arrival) or not math.isfinite(duration):
            errors.append("DP3_NONFINITE_INTERVAL")
            continue
        if arrival < -1e-6 or duration < -1e-6:
            errors.append("DP3_NEGATIVE_INTERVAL")
            continue
        intervals_by_station[int(station_id)].append(
            (int(slot_id), max(0.0, arrival), max(0.0, duration))
        )

    station_orders: dict[int, tuple[int, ...]] = {}
    no_wait_cmax = max(0.0, float(no_wait_cmax_floor))
    if not math.isfinite(no_wait_cmax):
        errors.append("DP3_NONFINITE_CMAX_FLOOR")
        no_wait_cmax = 0.0
    feasible_start_cmax = 0.0
    overlap_sec = 0.0
    active_count = 0
    station_workloads: list[float] = []
    for station_id, intervals in sorted(intervals_by_station.items()):
        ordered = sorted(intervals, key=lambda row: (row[1], row[0]))
        station_workloads.append(sum(row[2] for row in ordered))
        station_orders[int(station_id)] = tuple(row[0] for row in ordered)
        previous_finish = 0.0
        for index, (_slot_id, arrival, duration) in enumerate(ordered):
            no_wait_cmax = max(no_wait_cmax, arrival + duration)
            previous_finish = max(previous_finish, arrival) + duration
            feasible_start_cmax = max(feasible_start_cmax, previous_finish)
            active_count += 1
            right = arrival + duration
            for _other_slot, other_arrival, other_duration in ordered[index + 1 :]:
                overlap_sec += max(
                    0.0,
                    min(right, other_arrival + other_duration)
                    - max(arrival, other_arrival),
                )

    if active_count <= 0:
        errors.append("DP3_NO_ACTIVE_SLOTS")
        recourse_score = float("inf")
        workload_imbalance = 0.0
    else:
        workload_imbalance = (
            statistics.pstdev(station_workloads)
            if len(station_workloads) > 1
            else 0.0
        )
        # Pairwise overlap charges both jobs. Dividing by twice the active
        # count estimates per-job queue pressure without double counting.
        average_congestion = overlap_sec / float(2 * active_count)
        recourse_score = min(
            feasible_start_cmax,
            no_wait_cmax
            + average_congestion
            + workload_imbalance / float(max(1, len(station_workloads))),
        )
    return DP3RecoveryResult(
        feasible=bool(dp2.feasible and not errors),
        no_wait_cmax=float(no_wait_cmax),
        feasible_start_cmax=float(feasible_start_cmax),
        recourse_score=float(recourse_score),
        station_overlap_sec=float(overlap_sec),
        station_workload_imbalance=float(workload_imbalance),
        active_slot_count=int(active_count),
        station_orders=station_orders,
        error_codes=tuple(sorted(set(errors))),
    )
