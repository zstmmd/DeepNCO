from __future__ import annotations

import math
from typing import Any, Mapping

from Gurobi.tra_comproc.types import DP1RouteResult, DP2ServiceResult
from Gurobi.tra_projection import INACTIVE_LABEL, StructuralShell


def _read(values: Mapping[str, float], variable: Any) -> float:
    return float(values.get(str(variable.VarName), 0.0))


def evaluate_dp2_service(
    values_by_name: Mapping[str, float],
    payload: Mapping[str, Any],
    shell: StructuralShell,
    dp1: DP1RouteResult,
) -> DP2ServiceResult:
    """Propagate route arrivals and processing durations before station waiting."""

    arrival = payload.get("arrival") or {}
    start = payload.get("start") or {}
    finish = payload.get("finish") or {}
    active_slots = {int(slot_id) for slot_id in shell.projection.x_group.values()}
    station_by_slot: dict[int, int] = {}
    errors: list[str] = []
    for (slot_id, _stack_id), station_id in shell.projection.s_visit.items():
        slot = int(slot_id)
        station = int(station_id)
        if station == INACTIVE_LABEL:
            continue
        previous = station_by_slot.setdefault(slot, station)
        if previous != station:
            errors.append("DP2_MULTIPLE_SLOT_STATIONS")

    slot_arrival: dict[int, float] = {}
    slot_duration: dict[int, float] = {}
    for slot_id in sorted(active_slots):
        if slot_id not in arrival or slot_id not in start or slot_id not in finish:
            errors.append("DP2_MISSING_SLOT_TIME")
            continue
        arrival_value = max(
            _read(values_by_name, arrival[slot_id]),
            float(dp1.slot_arrival_lower.get(slot_id, 0.0)),
        )
        duration = _read(values_by_name, finish[slot_id]) - _read(
            values_by_name,
            start[slot_id],
        )
        if not math.isfinite(arrival_value) or arrival_value < -1e-6:
            errors.append("DP2_INVALID_ARRIVAL")
        if not math.isfinite(duration) or duration < -1e-6:
            errors.append("DP2_INVALID_DURATION")
        slot_arrival[slot_id] = max(0.0, float(arrival_value))
        slot_duration[slot_id] = max(0.0, float(duration))
        if slot_id not in station_by_slot:
            errors.append("DP2_MISSING_SLOT_STATION")
    return DP2ServiceResult(
        feasible=not errors,
        slot_arrival=slot_arrival,
        slot_process_duration=slot_duration,
        station_by_slot=station_by_slot,
        error_codes=tuple(sorted(set(errors))),
    )
