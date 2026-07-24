from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from Gurobi.tra_projection import INACTIVE_LABEL, StructuralShell
from Gurobi.tra_start_validation import FullStartValidation, validate_full_start


class OuterStartProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class OuterStartProjection:
    values_by_name: Mapping[str, float]
    station_orders: Mapping[int, tuple[int, ...]]
    projected_cmax: float
    added_station_wait_sec: float


@dataclass(frozen=True)
class FullStartVector:
    values_by_name: Mapping[str, float]
    safe_values_by_name: Mapping[str, float]
    projection: OuterStartProjection
    validation: FullStartValidation


def positive_family_start_values(
    values_by_name: Mapping[str, float],
    payload: Mapping[str, Any],
    *family_names: str,
) -> dict[str, float]:
    """Keep only selected primary decisions and let Gurobi complete their recourse."""

    selected: dict[str, float] = {}
    for family_name in family_names:
        family = payload.get(str(family_name))
        if family is None:
            raise OuterStartProjectionError(f"outer payload is missing {family_name}")
        for variable in family.values():
            name = _name(variable)
            value = float(values_by_name.get(name, 0.0))
            if value > 0.5:
                selected[name] = 1.0
    return selected


def _name(variable: Any) -> str:
    return str(variable.VarName)


def _read(values: Mapping[str, float], variable: Any) -> float:
    name = _name(variable)
    if name not in values:
        raise OuterStartProjectionError(f"outer start is missing {name}")
    value = float(values[name])
    if not math.isfinite(value):
        raise OuterStartProjectionError(f"outer start has nonfinite value for {name}")
    return value


def restore_station_wait_start(
    values_by_name: Mapping[str, float],
    payload: Mapping[str, Any],
    shell: StructuralShell,
) -> OuterStartProjection:
    """Apply the paper's arrival-order waiting repair to a no-wait inner start."""

    values = {str(name): float(value) for name, value in values_by_name.items()}
    y = payload.get("y")
    arrival = payload.get("arrival")
    start = payload.get("start")
    finish = payload.get("finish")
    cmax = payload.get("cmax")
    if not y or not arrival or not start or not finish or cmax is None:
        raise OuterStartProjectionError("outer payload is missing station scheduling variables")

    active_slots = {int(slot_id) for slot_id in shell.projection.x_group.values()}
    station_by_slot: dict[int, int] = {}
    for (slot_id, _stack_id), station_id in shell.projection.s_visit.items():
        slot = int(slot_id)
        station = int(station_id)
        if station == INACTIVE_LABEL:
            continue
        previous = station_by_slot.setdefault(slot, station)
        if previous != station:
            raise OuterStartProjectionError(f"slot {slot} visits more than one station")
    missing_stations = active_slots - set(station_by_slot)
    if missing_stations:
        raise OuterStartProjectionError(
            f"active slots have no station assignment: {sorted(missing_stations)}"
        )

    ranks_by_station: dict[int, set[int]] = {}
    for (_slot_id, station_id, rank), variable in y.items():
        values[_name(variable)] = 0.0
        ranks_by_station.setdefault(int(station_id), set()).add(int(rank))

    station_arrival_clock = payload.get("station_arrival_clock")
    station_finish_clock = payload.get("station_finish_clock")
    for family in (station_arrival_clock, station_finish_clock):
        if family is not None:
            for variable in family.values():
                values[_name(variable)] = 0.0

    slots_by_station: dict[int, list[int]] = {}
    for slot_id in active_slots:
        slots_by_station.setdefault(station_by_slot[slot_id], []).append(slot_id)

    station_orders: dict[int, tuple[int, ...]] = {}
    projected_cmax = 0.0
    total_wait = 0.0
    for station_id, slot_ids in sorted(slots_by_station.items()):
        ordered_slots = sorted(
            slot_ids,
            key=lambda slot_id: (_read(values, arrival[slot_id]), int(slot_id)),
        )
        ranks = sorted(ranks_by_station.get(int(station_id), set()))
        if len(ranks) < len(ordered_slots):
            raise OuterStartProjectionError(
                f"station {station_id} has {len(ordered_slots)} slots but only {len(ranks)} ranks"
            )
        station_orders[int(station_id)] = tuple(ordered_slots)
        previous_finish = 0.0
        for slot_id, rank in zip(ordered_slots, ranks):
            y_variable = y.get((int(slot_id), int(station_id), int(rank)))
            if y_variable is None:
                raise OuterStartProjectionError(
                    f"missing y variable for slot {slot_id}, station {station_id}, rank {rank}"
                )
            values[_name(y_variable)] = 1.0
            arrival_value = _read(values, arrival[slot_id])
            old_start = _read(values, start[slot_id])
            old_finish = _read(values, finish[slot_id])
            service_time = max(0.0, old_finish - old_start)
            start_value = max(arrival_value, previous_finish)
            finish_value = start_value + service_time
            values[_name(start[slot_id])] = start_value
            values[_name(finish[slot_id])] = finish_value
            total_wait += max(0.0, start_value - arrival_value)
            previous_finish = finish_value
            projected_cmax = max(projected_cmax, finish_value)
            if station_arrival_clock is not None:
                values[_name(station_arrival_clock[int(station_id), int(rank)])] = arrival_value
            if station_finish_clock is not None:
                values[_name(station_finish_clock[int(station_id), int(rank)])] = finish_value

    values[_name(cmax)] = float(projected_cmax)
    return OuterStartProjection(
        values_by_name=values,
        station_orders=station_orders,
        projected_cmax=float(projected_cmax),
        added_station_wait_sec=float(total_wait),
    )


def build_full_start_vector(
    model: Any,
    values_by_name: Mapping[str, float],
    payload: Mapping[str, Any],
    shell: StructuralShell,
) -> FullStartVector:
    """Project no-wait values into the complete model and validate the result."""

    projection = restore_station_wait_start(values_by_name, payload, shell)
    full_values = {
        str(variable.VarName): 0.0
        for variable in model.getVars()
    }
    for name, value in projection.values_by_name.items():
        if str(name) in full_values and math.isfinite(float(value)):
            full_values[str(name)] = float(value)
    validation = validate_full_start(model, full_values)
    safe_values = positive_family_start_values(
        projection.values_by_name,
        payload,
        "y",
    )
    return FullStartVector(
        values_by_name=full_values,
        safe_values_by_name=safe_values,
        projection=projection,
        validation=validation,
    )
