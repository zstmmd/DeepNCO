from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, Optional

from gurobipy import GRB

from Gurobi.tra_model_state import ModelSnapshot, PersistentCompiledTemplate
from Gurobi.tra_projection import (
    INACTIVE_LABEL,
    ProjectionError,
    ProjectionRegistry,
    StructuralShell,
)
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
    for (slot_id, station_id, rank), variable in template.payload["y"].items():
        if search_values.get(str(variable.VarName), 0.0) > 0.5:
            station_by_slot[int(slot_id)] = int(station_id)

    slot_ids_by_order = {
        int(order_id): [int(slot_id) for slot_id in list(slot_ids or [])]
        for order_id, slot_ids in dict(template.compiled.prepared.get("slot_ids_by_order", {}) or {}).items()
    }
    route_available_pairs: set[tuple[int, int, int]] = set()
    route_stations_by_visit: Dict[tuple[int, int], set[int]] = defaultdict(set)
    for spec in dict(template.payload.get("route_tasks", {}) or {}).values():
        try:
            row = (
                int(getattr(spec, "slot_id")),
                int(getattr(spec, "stack_id")),
                int(getattr(spec, "station_id")),
            )
        except (TypeError, ValueError):
            continue
        route_available_pairs.add(row)
        route_stations_by_visit[(row[0], row[1])].add(row[2])
    if not route_available_pairs:
        for node in dict(template.payload.get("route_nodes", {}) or {}).values():
            try:
                if str(getattr(node, "kind", "")) not in {"pickup", "delivery"}:
                    continue
                row = (
                    int(getattr(node, "slot_id")),
                    int(getattr(node, "stack_id")),
                    int(getattr(node, "station_id")),
                )
            except (TypeError, ValueError):
                continue
            route_available_pairs.add(row)
            route_stations_by_visit[(row[0], row[1])].add(row[2])

    def route_available(slot_id: int, stack_id: int, station_id: int) -> bool:
        if not route_available_pairs:
            return True
        return (int(slot_id), int(stack_id), int(station_id)) in route_available_pairs

    def subtask_stack_ids(subtask: Any) -> list[int]:
        stack_ids: list[int] = []
        for task in list(getattr(subtask, "execution_tasks", []) or []):
            stack_id = int(getattr(task, "target_stack_id", -1))
            if stack_id >= 0 and stack_id not in stack_ids:
                stack_ids.append(stack_id)
        return stack_ids

    def common_route_stations(slot_id: int, subtask: Any) -> set[int]:
        assigned_station = int(getattr(subtask, "assigned_station_id", -1))
        stack_ids = subtask_stack_ids(subtask)
        if not stack_ids:
            return {assigned_station} if assigned_station >= 0 else set()
        if not route_available_pairs:
            return {assigned_station} if assigned_station >= 0 else set()
        station_sets = [
            set(route_stations_by_visit.get((int(slot_id), int(stack_id)), set()))
            for stack_id in stack_ids
        ]
        if not station_sets or any(not station_set for station_set in station_sets):
            return set()
        return set.intersection(*station_sets)

    def choose_warm_slot_assignment(
        available_slots: list[int],
        subtasks: list[Any],
    ) -> list[tuple[int, int]]:
        if not available_slots or not subtasks:
            return []
        slot_position = {int(slot_id): index for index, slot_id in enumerate(available_slots)}
        prefix_slots = list(available_slots[: min(len(available_slots), len(subtasks))])
        slot_pools = [prefix_slots]
        if len(prefix_slots) < len(subtasks):
            slot_pools.append(list(available_slots))

        def row_cost(index: int, slot_id: int, subtask: Any) -> tuple[int, int, int, int]:
            common = common_route_stations(slot_id, subtask)
            assigned_station = int(getattr(subtask, "assigned_station_id", -1))
            route_miss = int(not common)
            station_mismatch = int(bool(common) and assigned_station not in common)
            movement = abs(int(slot_position.get(int(slot_id), index)) - int(index))
            return (route_miss, station_mismatch, movement, int(slot_id))

        best_rows: list[tuple[int, int]] = []
        best_cost: tuple[int, int, int, int] | None = None

        def add_cost(
            left: tuple[int, int, int, int],
            right: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            return tuple(int(a) + int(b) for a, b in zip(left, right))  # type: ignore[return-value]

        for slot_pool in slot_pools:
            if len(slot_pool) < len(subtasks):
                continue
            pool_best_rows: list[tuple[int, int]] = []
            pool_best_cost: tuple[int, int, int, int] | None = None

            def search(
                index: int,
                used_slots: set[int],
                rows: list[tuple[int, int]],
                cost: tuple[int, int, int, int],
            ) -> None:
                nonlocal pool_best_rows, pool_best_cost
                if pool_best_cost is not None and cost >= pool_best_cost:
                    return
                if index >= len(subtasks):
                    pool_best_cost = cost
                    pool_best_rows = list(rows)
                    return
                ranked_slots = sorted(
                    (int(slot_id) for slot_id in slot_pool if int(slot_id) not in used_slots),
                    key=lambda slot_id: row_cost(index, slot_id, subtasks[index]),
                )
                for slot_id in ranked_slots:
                    used_slots.add(slot_id)
                    rows.append((slot_id, index))
                    search(
                        index + 1,
                        used_slots,
                        rows,
                        add_cost(cost, row_cost(index, slot_id, subtasks[index])),
                    )
                    rows.pop()
                    used_slots.remove(slot_id)

            search(0, set(), [], (0, 0, 0, 0))
            if pool_best_cost is None:
                continue
            best_rows = pool_best_rows
            best_cost = pool_best_cost
            if int(pool_best_cost[0]) == 0:
                break
        if best_cost is None:
            return []
        return best_rows

    warm_rows: list[tuple[int, int, Any, int]] = []
    for raw_order_id, subtasks in dict(getattr(template.compiled.warm, "subtask_by_order", {}) or {}).items():
        order_id = int(raw_order_id)
        available_slots = list(slot_ids_by_order.get(order_id, []))
        subtask_rows = list(subtasks or [])
        for slot_id, index in choose_warm_slot_assignment(available_slots, subtask_rows):
            subtask = subtask_rows[int(index)]
            station_id = int(getattr(subtask, "assigned_station_id", -1))
            common = common_route_stations(slot_id, subtask)
            if common:
                station_id = station_id if station_id in common else min(common)
            station_by_slot[int(slot_id)] = station_id
            warm_rows.append((order_id, int(slot_id), subtask, station_id))

    preferred_slot_by_order_sku: Dict[tuple[int, int], int] = {}
    preferred_robot_by_slot: Dict[int, int] = {}
    for order_id, slot_id, subtask, _station_id in warm_rows:
        preferred_robot = int(getattr(subtask, "assigned_robot_id", -1))
        if preferred_robot < 0:
            for task in list(getattr(subtask, "execution_tasks", []) or []):
                task_robot = int(getattr(task, "robot_id", -1))
                if task_robot >= 0:
                    preferred_robot = task_robot
                    break
        if preferred_robot >= 0:
            preferred_robot_by_slot[slot_id] = int(preferred_robot)
        for sku in list(getattr(subtask, "sku_list", []) or []):
            sku_id = int(getattr(sku, "id", sku))
            preferred_slot_by_order_sku.setdefault((order_id, sku_id), slot_id)

    a = template.payload.get("a", {})
    warm_active_slots = {slot_id for _order_id, slot_id, _subtask, _station_id in warm_rows}
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
    for slot_id, variable in (template.payload.get("a", {}) or {}).items():
        value = 1.0 if int(slot_id) in active_slots else 0.0
        search_values[str(variable.VarName)] = value
        values[str(variable.VarName)] = value

    robot_groups: Dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for (slot_id, robot_id), variable in template.payload["slot_robot"].items():
        robot_groups[int(slot_id)].append((int(robot_id), variable))
    for slot_id, candidates in robot_groups.items():
        selected = [robot_id for robot_id, variable in candidates if search_values.get(str(variable.VarName), 0.0) > 0.5]
        candidate_robot_ids = {robot_id for robot_id, _variable in candidates}
        preferred_robot = preferred_robot_by_slot.get(slot_id, -1)
        if slot_id not in active_slots:
            chosen_robot = -1
        elif selected:
            chosen_robot = int(selected[0])
        elif preferred_robot in candidate_robot_ids:
            chosen_robot = int(preferred_robot)
        else:
            chosen_robot = min(candidate_robot_ids)
        for robot_id, variable in candidates:
            value = 1.0 if int(robot_id) == chosen_robot else 0.0
            search_values[str(variable.VarName)] = value
            values[str(variable.VarName)] = value

    visit_groups: Dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
    for (slot_id, stack_id, station_id), variable in template.payload["pair_activate"].items():
        visit_groups[(int(slot_id), int(stack_id))].append((int(station_id), variable))

    warm_visit_station: Dict[tuple[int, int], int] = {}
    selected_stack_visits: set[tuple[int, int]] = set()
    for _order_id, slot_id, subtask, station_id in warm_rows:
        for task in list(getattr(subtask, "execution_tasks", []) or []):
            stack_id = int(getattr(task, "target_stack_id", -1))
            if stack_id >= 0:
                warm_visit_station[(slot_id, stack_id)] = station_id
                selected_stack_visits.add((int(slot_id), stack_id))
    if not selected_stack_visits:
        for (slot_id, stack_id), variable in template.payload["flip"].items():
            if search_values.get(str(variable.VarName), 0.0) > 0.5:
                selected_stack_visits.add((int(slot_id), int(stack_id)))
        for (slot_id, stack_id, _low, _high), variable in template.payload["sort"].items():
            if search_values.get(str(variable.VarName), 0.0) > 0.5:
                selected_stack_visits.add((int(slot_id), int(stack_id)))
    common_station_by_slot: Dict[int, int] = {}
    selected_by_slot: Dict[int, list[int]] = defaultdict(list)
    for slot_id, stack_id in selected_stack_visits:
        if int(slot_id) in active_slots:
            selected_by_slot[int(slot_id)].append(int(stack_id))
    for slot_id, stack_ids in selected_by_slot.items():
        station_sets: list[set[int]] = []
        station_sets_by_stack: Dict[int, set[int]] = {}
        station_scores: Dict[int, int] = defaultdict(int)
        for stack_id in stack_ids:
            candidates = visit_groups.get((int(slot_id), int(stack_id)), ())
            available = {
                int(station_id)
                for station_id, _variable in candidates
                if route_available(slot_id, stack_id, station_id)
            }
            if available:
                station_sets_by_stack[int(stack_id)] = set(available)
                station_sets.append(available)
                for station_id in available:
                    station_scores[int(station_id)] += 1
        if not station_sets:
            continue
        common = set.intersection(*station_sets)
        preferred_station = int(station_by_slot.get(int(slot_id), INACTIVE_LABEL))
        if common:
            chosen_slot_station = (
                preferred_station if preferred_station in common else min(common)
            )
        else:
            chosen_slot_station = min(
                station_scores,
                key=lambda station_id: (
                    -int(station_scores[int(station_id)]),
                    int(station_id) != preferred_station,
                    int(station_id),
                ),
            )
        common_station_by_slot[int(slot_id)] = int(chosen_slot_station)
        for stack_id, available in station_sets_by_stack.items():
            if int(chosen_slot_station) not in available:
                selected_stack_visits.discard((int(slot_id), int(stack_id)))
    for visit_key, candidates in visit_groups.items():
        if int(visit_key[0]) not in active_slots:
            chosen_station = -1
        elif visit_key in selected_stack_visits and int(visit_key[0]) in common_station_by_slot:
            chosen_station = int(common_station_by_slot[int(visit_key[0])])
        elif visit_key in selected_stack_visits:
            selected = [
                station_id
                for station_id, variable in candidates
                if search_values.get(str(variable.VarName), 0.0) > 0.5
                and route_available(visit_key[0], visit_key[1], station_id)
            ]
            chosen_station = int(selected[0]) if selected else -1
        else:
            chosen_station = -1
        if (
            int(visit_key[0]) in active_slots
            and chosen_station < 0
            and visit_key in selected_stack_visits
            and visit_key in warm_visit_station
        ):
            candidate_station = int(warm_visit_station[visit_key])
            if any(
                int(station_id) == candidate_station
                and route_available(visit_key[0], visit_key[1], station_id)
                for station_id, _variable in candidates
            ):
                chosen_station = candidate_station
            else:
                available_stations = [
                    int(station_id)
                    for station_id, _variable in candidates
                    if route_available(visit_key[0], visit_key[1], station_id)
                ]
                if available_stations:
                    chosen_station = min(available_stations)
                else:
                    raise ProjectionError(f"warm visit {visit_key} station {candidate_station} is outside manifest domain")
        if int(visit_key[0]) in active_slots and chosen_station < 0 and visit_key in selected_stack_visits:
            candidate_station = station_by_slot.get(int(visit_key[0]), -1)
            if any(
                int(station_id) == int(candidate_station)
                and route_available(visit_key[0], visit_key[1], station_id)
                for station_id, _variable in candidates
            ):
                chosen_station = int(candidate_station)
            else:
                available_stations = [
                    int(station_id)
                    for station_id, _variable in candidates
                    if route_available(visit_key[0], visit_key[1], station_id)
                ]
                if available_stations:
                    chosen_station = min(available_stations)
        for station_id, variable in candidates:
            value = 1.0 if int(station_id) == chosen_station else 0.0
            search_values[str(variable.VarName)] = value
            values[str(variable.VarName)] = value

    def set_action_start(family_name: str, key: tuple[int, ...]) -> None:
        family = template.payload.get(str(family_name), {}) or {}
        variable = family.get(tuple(int(part) for part in key))
        if variable is None:
            return
        values[str(variable.VarName)] = 1.0
        search_values[str(variable.VarName)] = 1.0

    for family_name in ("flip", "sort", "carry", "hit", "noise", "flip_hit"):
        for key, variable in (template.payload.get(family_name, {}) or {}).items():
            values[str(variable.VarName)] = 0.0
            search_values[str(variable.VarName)] = 0.0

    for _order_id, slot_id, subtask, _station_id in warm_rows:
        for task in list(getattr(subtask, "execution_tasks", []) or []):
            stack_id = int(getattr(task, "target_stack_id", -1))
            if stack_id < 0:
                continue
            if (int(slot_id), stack_id) not in selected_stack_visits:
                continue
            mode = str(getattr(task, "operation_mode", "FLIP") or "FLIP").upper()
            target_totes = [
                int(tote_id)
                for tote_id in list(getattr(task, "target_tote_ids", []) or [])
            ]
            hit_totes = [
                int(tote_id)
                for tote_id in list(getattr(task, "hit_tote_ids", []) or [])
            ] or list(target_totes)
            noise_totes = [
                int(tote_id)
                for tote_id in list(getattr(task, "noise_tote_ids", []) or [])
            ]
            carry_totes = list(target_totes or hit_totes)
            if mode == "SORT":
                sort_range = getattr(task, "sort_layer_range", None)
                if sort_range is not None:
                    set_action_start(
                        "sort",
                        (
                            int(slot_id),
                            stack_id,
                            int(sort_range[0]),
                            int(sort_range[1]),
                        ),
                    )
            else:
                set_action_start("flip", (int(slot_id), stack_id))
                for tote_id in hit_totes:
                    set_action_start("flip_hit", (int(slot_id), int(tote_id)))
            for tote_id in carry_totes:
                set_action_start("carry", (int(slot_id), int(tote_id)))
            for tote_id in hit_totes:
                set_action_start("hit", (int(slot_id), int(tote_id)))
            for tote_id in noise_totes:
                set_action_start("noise", (int(slot_id), int(tote_id)))


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

    if not bool(
        dict(getattr(template.compiled, "diagnostics", {}) or {}).get(
            "warm_start_mip_start_ready",
            False,
        )
    ):
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
        start_values=all_values,
        verified_incumbent=verified,
    )
