from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping

from Gurobi.tra_model_state import ModelSnapshot
from Gurobi.tra_projection import INACTIVE_LABEL, StructuralShell, raw_one_hot_hamming


@dataclass(frozen=True)
class RepairRisk:
    total: float
    station_overlap_sec: float
    station_workload_imbalance: float
    warm_disturbance_hamming: int


def _snapshot_value(snapshot: ModelSnapshot, variable: Any) -> float:
    return float(snapshot.values_by_name.get(str(variable.VarName), 0.0))


def compute_repair_risk(
    shell: StructuralShell,
    snapshot: ModelSnapshot,
    payload: Mapping[str, Any],
    reference: StructuralShell,
) -> RepairRisk:
    station_by_slot: dict[int, int] = {}
    for (slot_id, _stack_id), station_id in shell.projection.s_visit.items():
        if int(station_id) == INACTIVE_LABEL:
            continue
        station_by_slot[int(slot_id)] = int(station_id)

    intervals_by_station: dict[int, list[tuple[float, float]]] = {}
    arrival = payload.get("arrival", {})
    finish = payload.get("finish", {})
    for slot_id, station_id in station_by_slot.items():
        if slot_id not in arrival or slot_id not in finish:
            continue
        left = _snapshot_value(snapshot, arrival[slot_id])
        right = max(left, _snapshot_value(snapshot, finish[slot_id]))
        intervals_by_station.setdefault(station_id, []).append((left, right))

    overlap = 0.0
    workloads = []
    for intervals in intervals_by_station.values():
        workloads.append(sum(max(0.0, right - left) for left, right in intervals))
        for index, (left, right) in enumerate(intervals):
            for other_left, other_right in intervals[index + 1 :]:
                overlap += max(0.0, min(right, other_right) - max(left, other_left))
    imbalance = statistics.pstdev(workloads) if len(workloads) > 1 else 0.0

    disturbance = (
        raw_one_hot_hamming(reference.projection.x_group, shell.projection.x_group)
        + raw_one_hot_hamming(reference.projection.s_visit, shell.projection.s_visit)
        + raw_one_hot_hamming(reference.projection.r_assign, shell.projection.r_assign)
    )
    total = float(overlap) + 0.25 * float(imbalance) + 0.01 * float(disturbance)
    return RepairRisk(
        total=total,
        station_overlap_sec=float(overlap),
        station_workload_imbalance=float(imbalance),
        warm_disturbance_hamming=int(disturbance),
    )
