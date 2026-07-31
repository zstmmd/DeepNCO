from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import gurobipy as gp

from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_projection import INACTIVE_LABEL, StructuralShell


_INVENTORY_ACTION_FAMILIES = (
    "flip",
    "sort",
    "carry",
    "hit",
    "noise",
    "flip_hit",
)

_ROUTE_RECOURSE_FAMILIES = (
    "route_arc",
    "route_time",
    "route_load",
    "route_finish",
    "arrival",
    "start",
    "finish",
)

_STATION_TIME_FAMILIES = (
    "start",
    "finish",
    "cmax",
)

_DERIVED_ROUTE_FAMILIES = (
    "pass_x",
    "route_owner",
)

PHASE_TWO_CANDIDATE_TARGET = 6
PHASE_TWO_BASE_OBJECTIVE_TIEBREAK = 1e-4
PHASE_TWO_MAX_SEED_ATTEMPTS = 3
PHASE_TWO_MIN_RESERVED_ROUND_SEC = 0.75
PHASE_TWO_SEED_STRIDE = 7919


def _variable_names(value: Any) -> set[str]:
    if hasattr(value, "VarName"):
        return {str(value.VarName)}
    if isinstance(value, Mapping):
        names: set[str] = set()
        for item in value.values():
            names.update(_variable_names(item))
        return names
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        names = set()
        for item in value:
            names.update(_variable_names(item))
        return names
    return set()


def project_inner_start(
    values_by_name: Mapping[str, float],
    payload: Mapping[str, Any],
    procedure: Procedure,
) -> dict[str, float]:
    """Remove values that directly conflict with the requested local move."""

    procedure = Procedure(procedure)
    if procedure is Procedure.F1:
        return {}
    excluded_families = {
        Procedure.F2: (
            "x",
            *_INVENTORY_ACTION_FAMILIES,
            *_STATION_TIME_FAMILIES,
            *_DERIVED_ROUTE_FAMILIES,
        ),
        Procedure.F3: (
            "slot_robot",
            *_ROUTE_RECOURSE_FAMILIES,
            *_DERIVED_ROUTE_FAMILIES,
            "cmax",
        ),
    }[procedure]
    excluded_names: set[str] = set()
    for family in excluded_families:
        excluded_names.update(_variable_names(payload.get(family, {})))
    return {
        str(name): float(value)
        for name, value in values_by_name.items()
        if str(name) not in excluded_names
    }


def initial_inner_start_values(
    procedure: Procedure,
    *,
    projected_start: Mapping[str, float],
    vns_start_values: Sequence[Mapping[str, float]],
    f1_live_seed_starts: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    values = {
        str(name): float(value)
        for name, value in projected_start.items()
    }
    if Procedure(procedure) is Procedure.F1:
        if f1_live_seed_starts:
            values.update(
                {
                    str(name): float(value)
                    for name, value in f1_live_seed_starts[0].items()
                }
            )
        return values
    if vns_start_values:
        values.update(
            {
                str(name): float(value)
                for name, value in vns_start_values[0].items()
            }
        )
    return values


def phase_two_inner_start_values(
    procedure: Procedure,
    *,
    projected_start: Mapping[str, float],
    vns_start_values: Sequence[Mapping[str, float]],
    f1_live_seed_starts: Sequence[Mapping[str, float]],
    phase_two_attempt: int,
) -> dict[str, float]:
    """Return the deterministic start for one phase-two round."""

    values = {
        str(name): float(value)
        for name, value in projected_start.items()
    }
    seed_index = max(0, int(phase_two_attempt))
    starts = (
        f1_live_seed_starts
        if Procedure(procedure) is Procedure.F1
        else vns_start_values
    )
    if seed_index < len(starts):
        values.update(
            {
                str(name): float(value)
                for name, value in starts[seed_index].items()
            }
        )
    return values


def phase_two_start_seed_sha256(
    procedure: Procedure,
    *,
    vns_seed_sha256: Sequence[str],
    f1_live_seed_sha256: Sequence[str],
    phase_two_attempt: int,
) -> str:
    """Return the source-start hash associated with one phase-two round."""

    seed_index = max(0, int(phase_two_attempt))
    hashes = (
        f1_live_seed_sha256
        if Procedure(procedure) is Procedure.F1 and f1_live_seed_sha256
        else vns_seed_sha256
    )
    return str(hashes[seed_index]) if seed_index < len(hashes) else ""


def configure_inner_search(model: Any) -> None:
    """Favor natural feasible MIPSOL discovery without enabling pool search."""

    model.Params.MIPFocus = 1
    model.Params.MIPGap = 0.0
    model.Params.PoolSearchMode = 0
    model.Params.Heuristics = max(0.5, float(model.Params.Heuristics))
    model.Params.PumpPasses = max(20, int(model.Params.PumpPasses))
    model.Params.StartNodeLimit = max(1000, int(model.Params.StartNodeLimit))


def f3_balance_coefficients(
    assignments: Mapping[int, int],
    robot_labels: Iterable[int],
) -> dict[int, float]:
    counts = Counter(
        int(robot_id)
        for robot_id in assignments.values()
        if int(robot_id) != INACTIVE_LABEL
    )
    return {
        int(robot_id): float(counts.get(int(robot_id), 0))
        for robot_id in sorted({int(value) for value in robot_labels})
    }


def phase_two_recourse_objective(
    payload: Mapping[str, Any],
    incumbent: StructuralShell,
    procedure: Procedure,
    neighborhood: NeighborhoodLevel,
) -> Any | None:
    """Return a target-blind secondary objective inside the relaxed-quality band."""

    if (
        Procedure(procedure) is not Procedure.F3
        or NeighborhoodLevel(neighborhood)
        not in (NeighborhoodLevel.N1, NeighborhoodLevel.N3)
    ):
        return None
    slot_robot = dict(payload.get("slot_robot", {}) or {})
    coefficients = f3_balance_coefficients(
        incumbent.projection.r_assign,
        (int(robot_id) for _slot_id, robot_id in slot_robot),
    )
    return gp.quicksum(
        float(coefficients[int(robot_id)]) * variable
        for (_slot_id, robot_id), variable in slot_robot.items()
    )


def phase_one_time_limit(total_time_limit_sec: float) -> float:
    return max(1e-3, 0.60 * max(0.0, float(total_time_limit_sec)))


def relaxed_quality_tolerance(best_relaxed_objective: float) -> float:
    value = abs(float(best_relaxed_objective))
    return max(0.5, 0.002 * value)


def phase_two_quality_limit(
    best_relaxed_objective: float | None,
    incumbent_objective: float | None,
) -> float | None:
    for value in (best_relaxed_objective, incumbent_objective):
        if value is not None and math.isfinite(float(value)):
            numeric = float(value)
            return numeric + relaxed_quality_tolerance(numeric)
    return None


def should_run_phase_two(
    *,
    candidate_count: int,
    phase_one_timed_out: bool,
    remaining_sec: float,
    candidate_target: int = PHASE_TWO_CANDIDATE_TARGET,
) -> bool:
    count = max(0, int(candidate_count))
    return (
        float(remaining_sec) > 1e-3
        and count < max(1, int(candidate_target))
        and (count > 0 or bool(phase_one_timed_out))
    )


def phase_two_pool_complete(
    candidate_count: int,
    candidate_target: int = PHASE_TWO_CANDIDATE_TARGET,
) -> bool:
    return int(candidate_count) >= max(1, int(candidate_target))


def phase_two_attempt_limit(
    procedure: Procedure,
    neighborhood: NeighborhoodLevel,
    *,
    vns_seed_count: int,
    f1_live_seed_count: int = 0,
    recourse_active: bool,
) -> int:
    """Return how many independent natural phase-two solves to permit."""

    if (
        Procedure(procedure) is Procedure.F1
        and NeighborhoodLevel(neighborhood) is NeighborhoodLevel.N2
        and int(f1_live_seed_count) > 1
    ):
        return min(
            PHASE_TWO_MAX_SEED_ATTEMPTS,
            int(f1_live_seed_count) - 1,
        )
    if (
        not bool(recourse_active)
        or Procedure(procedure) is not Procedure.F3
        or NeighborhoodLevel(neighborhood)
        not in (NeighborhoodLevel.N1, NeighborhoodLevel.N3)
    ):
        return 1
    available_phase_two_seeds = max(1, int(vns_seed_count) - 1)
    return min(PHASE_TWO_MAX_SEED_ATTEMPTS, available_phase_two_seeds)


def phase_two_round_time_limit(
    remaining_sec: float,
    *,
    rounds_remaining: int,
) -> float:
    """Reserve a small deterministic slice for each unused VNS seed."""

    remaining = max(0.0, float(remaining_sec))
    rounds = max(1, int(rounds_remaining))
    if rounds == 1:
        return remaining
    reserved = PHASE_TWO_MIN_RESERVED_ROUND_SEC * float(rounds - 1)
    if remaining <= PHASE_TWO_MIN_RESERVED_ROUND_SEC * float(rounds):
        return remaining / float(rounds)
    return remaining - reserved


def phase_two_search_seed(base_seed: int, *, phase_two_attempt: int) -> int:
    return (
        int(base_seed)
        + PHASE_TWO_SEED_STRIDE * max(1, int(phase_two_attempt))
    ) % 2_000_000_000


def phase_two_round_complete(
    *,
    candidate_count: int,
    round_start_count: int,
    stop_after_new_candidate: bool,
    candidate_target: int = PHASE_TWO_CANDIDATE_TARGET,
) -> bool:
    if phase_two_pool_complete(candidate_count, candidate_target):
        return True
    return bool(
        stop_after_new_candidate
        and int(candidate_count) > int(round_start_count)
    )
