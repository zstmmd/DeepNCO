from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, Optional

from gurobipy import GRB

from Gurobi.tra_model_state import ModelSnapshot, PersistentCompiledTemplate
from Gurobi.tra_projection import ProjectionError, ProjectionRegistry, StructuralShell
from Gurobi.tra_verifier import SnapshotVerifier, VerifiedSnapshot


@dataclass(frozen=True)
class CanonicalInitialState:
    search_shell: StructuralShell
    start_values: Dict[str, float]
    verified_incumbent: Optional[VerifiedSnapshot]


def _is_defined_start(value: float) -> bool:
    return math.isfinite(float(value)) and abs(float(value)) < 0.5 * float(GRB.UNDEFINED)


def _complete_primary_projection(
    template: PersistentCompiledTemplate,
    values: Dict[str, float],
    search_values: Dict[str, float],
) -> None:
    slot_order = {
        int(getattr(slot, "slot_id", -1)): int(getattr(slot, "order_id", -1))
        for slot in list(template.compiled.prepared.get("slots", []) or [])
    }
    station_by_slot: Dict[int, int] = {}
    station_rank_by_slot: Dict[int, tuple[int, int]] = {}
    for (slot_id, station_id, rank), variable in template.payload["y"].items():
        if search_values.get(str(variable.VarName), 0.0) > 0.5:
            station_by_slot[int(slot_id)] = int(station_id)
            station_rank_by_slot[int(slot_id)] = (int(station_id), int(rank))

    slot_ids_by_order = {
        int(order_id): [int(slot_id) for slot_id in list(slot_ids or [])]
        for order_id, slot_ids in dict(template.compiled.prepared.get("slot_ids_by_order", {}) or {}).items()
    }
    warm_rows: list[tuple[int, int, Any]] = []
    for raw_order_id, subtasks in dict(getattr(template.compiled.warm, "subtask_by_order", {}) or {}).items():
        order_id = int(raw_order_id)
        available_slots = list(slot_ids_by_order.get(order_id, []))
        used_slots: set[int] = set()
        for index, subtask in enumerate(list(subtasks or [])):
            station_id = int(getattr(subtask, "assigned_station_id", -1))
            rank = int(getattr(subtask, "station_sequence_rank", -1))
            matching = [
                slot_id
                for slot_id in available_slots
                if slot_id not in used_slots and station_rank_by_slot.get(slot_id) == (station_id, rank)
            ]
            remaining = [slot_id for slot_id in available_slots if slot_id not in used_slots]
            if matching:
                slot_id = int(matching[0])
            elif index < len(available_slots) and available_slots[index] not in used_slots:
                slot_id = int(available_slots[index])
            elif remaining:
                slot_id = int(remaining[0])
            else:
                continue
            used_slots.add(slot_id)
            station_by_slot[slot_id] = station_id
            warm_rows.append((order_id, slot_id, subtask))

    preferred_slot_by_order_sku: Dict[tuple[int, int], int] = {}
    preferred_robot_by_slot: Dict[int, int] = {}
    for order_id, slot_id, subtask in warm_rows:
        preferred_robot_by_slot[slot_id] = int(getattr(subtask, "assigned_robot_id", -1))
        for sku in list(getattr(subtask, "sku_list", []) or []):
            sku_id = int(getattr(sku, "id", sku))
            preferred_slot_by_order_sku.setdefault((order_id, sku_id), slot_id)

    a = template.payload.get("a", {})
    warm_active_slots = {slot_id for _order_id, slot_id, _subtask in warm_rows}
    warm_active_slots.update(
        int(slot_id)
        for slot_id, variable in a.items()
        if float(search_values.get(str(variable.VarName), 0.0)) > 0.5
    )
    unit_semantics = {
        str(getattr(unit, "unit_id", "")): (
            int(getattr(unit, "order_id", -1)),
            int(getattr(unit, "sku_id", -1)),
        )
        for unit in list(template.compiled.prepared.get("work_units", []) or [])
    }
    x_groups: Dict[Any, list[tuple[int, Any]]] = defaultdict(list)
    for (unit_id, slot_id), variable in template.payload["x"].items():
        x_groups[unit_id].append((int(slot_id), variable))
    load_by_slot: Dict[int, int] = defaultdict(int)
    for unit_id, candidates in sorted(x_groups.items(), key=lambda item: str(item[0])):
        selected = [slot_id for slot_id, variable in candidates if search_values.get(str(variable.VarName), 0.0) > 0.5]
        semantic = unit_semantics.get(str(unit_id), (-1, -1))
        preferred = preferred_slot_by_order_sku.get(semantic)
        candidate_slot_ids = {slot_id for slot_id, _variable in candidates}
        if preferred in candidate_slot_ids:
            chosen = int(preferred)
        elif selected:
            chosen = int(selected[0])
        else:
            eligible = [
                slot_id
                for slot_id, _variable in candidates
                if int(slot_order.get(slot_id, -1)) == int(semantic[0])
                and slot_id in warm_active_slots
            ]
            if not eligible:
                eligible = [slot_id for slot_id, _variable in candidates]
            chosen = min(eligible, key=lambda slot_id: (load_by_slot[int(slot_id)], int(slot_id)))
        load_by_slot[chosen] += 1
        for slot_id, variable in candidates:
            value = 1.0 if int(slot_id) == chosen else 0.0
            search_values[str(variable.VarName)] = value
            values[str(variable.VarName)] = value

    active_slots = set(int(slot_id) for slot_id in load_by_slot)
    robot_groups: Dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for (slot_id, robot_id), variable in template.payload["slot_robot"].items():
        robot_groups[int(slot_id)].append((int(robot_id), variable))
    for slot_id, candidates in robot_groups.items():
        selected = [robot_id for robot_id, variable in candidates if search_values.get(str(variable.VarName), 0.0) > 0.5]
        candidate_robot_ids = {robot_id for robot_id, _variable in candidates}
        preferred_robot = preferred_robot_by_slot.get(slot_id, -1)
        if selected:
            chosen_robot = int(selected[0])
        elif slot_id in active_slots and preferred_robot in candidate_robot_ids:
            chosen_robot = int(preferred_robot)
        elif slot_id in active_slots:
            chosen_robot = min(candidate_robot_ids)
        else:
            chosen_robot = -1
        for robot_id, variable in candidates:
            value = 1.0 if int(robot_id) == chosen_robot else 0.0
            search_values[str(variable.VarName)] = value
            values[str(variable.VarName)] = value

    visit_groups: Dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
    for (slot_id, stack_id, station_id), variable in template.payload["pair_activate"].items():
        visit_groups[(int(slot_id), int(stack_id))].append((int(station_id), variable))
    warm_visit_station: Dict[tuple[int, int], int] = {}
    for _order_id, slot_id, subtask in warm_rows:
        station_id = int(getattr(subtask, "assigned_station_id", -1))
        for task in list(getattr(subtask, "execution_tasks", []) or []):
            stack_id = int(getattr(task, "target_stack_id", -1))
            if stack_id >= 0:
                warm_visit_station[(slot_id, stack_id)] = station_id
    selected_stack_visits: set[tuple[int, int]] = set()
    for (slot_id, stack_id), variable in template.payload["flip"].items():
        if search_values.get(str(variable.VarName), 0.0) > 0.5:
            selected_stack_visits.add((int(slot_id), int(stack_id)))
    for (slot_id, stack_id, _low, _high), variable in template.payload["sort"].items():
        if search_values.get(str(variable.VarName), 0.0) > 0.5:
            selected_stack_visits.add((int(slot_id), int(stack_id)))
    for visit_key, candidates in visit_groups.items():
        selected = [station_id for station_id, variable in candidates if search_values.get(str(variable.VarName), 0.0) > 0.5]
        chosen_station = int(selected[0]) if selected else -1
        if chosen_station < 0 and visit_key in warm_visit_station:
            candidate_station = int(warm_visit_station[visit_key])
            if any(int(station_id) == candidate_station for station_id, _variable in candidates):
                chosen_station = candidate_station
            else:
                raise ProjectionError(f"warm visit {visit_key} station {candidate_station} is outside manifest domain")
        if chosen_station < 0 and visit_key in selected_stack_visits:
            candidate_station = station_by_slot.get(int(visit_key[0]), -1)
            if any(int(station_id) == int(candidate_station) for station_id, _variable in candidates):
                chosen_station = int(candidate_station)
        for station_id, variable in candidates:
            value = 1.0 if int(station_id) == chosen_station else 0.0
            search_values[str(variable.VarName)] = value
            if str(variable.VarName) in values:
                values[str(variable.VarName)] = value


def build_canonical_initial_state(
    template: PersistentCompiledTemplate,
    verifier: SnapshotVerifier,
) -> CanonicalInitialState:
    defined_values: Dict[str, float] = {}
    all_values: Dict[str, float] = {}
    for variable in template.model.getVars():
        try:
            value = float(variable.Start)
        except Exception:
            value = float(GRB.UNDEFINED)
        if _is_defined_start(value):
            defined_values[str(variable.VarName)] = value
            all_values[str(variable.VarName)] = value
        else:
            all_values[str(variable.VarName)] = 0.0

    _complete_primary_projection(template, defined_values, all_values)
    start_registry = ProjectionRegistry.from_payload(
        template.payload,
        value_getter=lambda variable: all_values[str(variable.VarName)],
    )
    search_shell = StructuralShell.extract(start_registry)

    verified: Optional[VerifiedSnapshot] = None
    if len(defined_values) == len(template.model.getVars()):
        objective = sum(
            float(variable.Obj) * float(defined_values[str(variable.VarName)])
            for variable in template.model.getVars()
        )
        cmax_var = template.payload["cmax"]
        snapshot = ModelSnapshot(
            values_by_name=defined_values,
            solver_objective=float(objective),
            solver_cmax=float(defined_values[str(cmax_var.VarName)]),
            callback_runtime_sec=0.0,
        )
        candidate = verifier.verify(snapshot)
        if candidate.internal_feasible:
            verified = candidate
            search_shell = candidate.shell
    return CanonicalInitialState(
        search_shell=search_shell,
        start_values=defined_values,
        verified_incumbent=verified,
    )
