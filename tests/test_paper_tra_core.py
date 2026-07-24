from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from Gurobi.tra_elite import candidate_preference_key, select_pareto_elites
from Gurobi.tra_neighborhood import (
    NeighborhoodLevel,
    Procedure,
    validate_transition,
)
from Gurobi.tra_outer_start import positive_family_start_values, restore_station_wait_start
from Gurobi.tra_projection import (
    INACTIVE_LABEL,
    CoreProjection,
    ProjectionError,
    ProjectionRegistry,
    StructuralShell,
)


@dataclass(frozen=True)
class _EliteRisk:
    total: float


@dataclass(frozen=True)
class _EliteShell:
    sha256: str


@dataclass(frozen=True)
class _EliteCandidate:
    shell: _EliteShell
    relaxed_objective: float
    repair_risk: _EliteRisk


class _Value:
    def __init__(self, value: float) -> None:
        self.X = float(value)


@dataclass(frozen=True)
class _NamedVar:
    VarName: str


def _vars(keys, selected):
    return {key: _Value(1.0 if key in selected else 0.0) for key in keys}


def _payload():
    x_keys = [("u1", 10), ("u1", 11), ("u2", 10), ("u2", 11)]
    pair_keys = [
        (slot, stack, station)
        for slot in (10, 11)
        for stack in (20, 21)
        for station in (30, 31)
    ]
    robot_keys = [(slot, robot) for slot in (10, 11) for robot in (40, 41)]
    y_keys = [(slot, station, rank) for slot in (10, 11) for station in (30, 31) for rank in (0, 1)]
    return {
        "x": _vars(x_keys, {("u1", 10), ("u2", 11)}),
        "pair_activate": _vars(pair_keys, {(10, 20, 30), (11, 21, 31)}),
        "slot_robot": _vars(robot_keys, {(10, 40), (11, 41)}),
        "y": _vars(y_keys, {(10, 30, 0), (11, 31, 1)}),
        "flip": _vars([(10, 20), (10, 21), (11, 20), (11, 21)], {(10, 20)}),
        "sort": _vars([(11, 21, 0, 1)], {(11, 21, 0, 1)}),
        "carry": _vars([(10, 100), (11, 101)], {(10, 100), (11, 101)}),
        "hit": _vars([(10, 100), (11, 101)], {(10, 100), (11, 101)}),
        "noise": _vars([(10, 100), (11, 101)], set()),
        "flip_hit": _vars([(10, 100), (11, 101)], {(10, 100)}),
        "route_arc": _vars([(1, 2), (2, 3)], {(1, 2)}),
        "route_time": {1: _Value(12.0)},
        "arrival": {10: _Value(15.0)},
        "cmax": _Value(99.0),
    }


def _projection() -> CoreProjection:
    return ProjectionRegistry.from_payload(_payload()).extract()


def test_projection_extracts_three_primary_carriers_and_is_canonical() -> None:
    projection = _projection()

    assert projection.x_group == {"u1": 10, "u2": 11}
    assert projection.s_visit == {
        (10, 20): 30,
        (10, 21): INACTIVE_LABEL,
        (11, 20): INACTIVE_LABEL,
        (11, 21): 31,
    }
    assert projection.r_assign == {10: 40, 11: 41}
    assert projection.sha256 == projection.canonicalized().sha256


def test_projection_rejects_non_one_hot_primary_assignment() -> None:
    payload = _payload()
    payload["x"][("u1", 11)].X = 1.0

    with pytest.raises(ProjectionError, match="x_group.*one-hot"):
        ProjectionRegistry.from_payload(payload).extract()


def test_structural_shell_round_trip_fixes_marginals_but_not_rank_or_route() -> None:
    payload = _payload()
    registry = ProjectionRegistry.from_payload(payload)
    shell = StructuralShell.extract(registry)
    plan = shell.fixing_plan(registry)

    assert set(plan.binary_values) == {
        "x",
        "pair_activate",
        "slot_robot",
        "flip",
        "sort",
        "carry",
        "hit",
        "noise",
        "flip_hit",
    }
    assert "y" not in plan.binary_values
    assert "route_arc" not in plan.binary_values
    assert "route_time" not in plan.binary_values
    assert plan.station_marginals[(10, 30)] == 1
    assert plan.station_marginals[(10, 31)] == 0
    assert plan.station_marginals[(11, 30)] == 0
    assert plan.station_marginals[(11, 31)] == 1
    assert shell.projection == registry.extract()


@pytest.mark.parametrize(
    ("procedure", "block_name"),
    [
        (Procedure.F1, "s_visit"),
        (Procedure.F2, "x_group"),
        (Procedure.F3, "r_assign"),
    ],
)
def test_n1_changes_one_carrier_with_raw_hamming_two(procedure, block_name) -> None:
    before = _projection()
    if procedure is Procedure.F2:
        before = CoreProjection(
            x_group={"u1": 10, "u2": 10, "u3": 11},
            s_visit=before.s_visit,
            r_assign=before.r_assign,
        )
    mapping = dict(getattr(before, block_name))
    first = next(iter(mapping))
    mapping[first] = {
        "s_visit": 31,
        "x_group": 11,
        "r_assign": 41,
    }[block_name]
    if mapping[first] == getattr(before, block_name)[first]:
        mapping[first] = {
            "s_visit": 30,
            "x_group": 10,
            "r_assign": 40,
        }[block_name]
    after = before.replace_block(block_name, mapping)

    audit = validate_transition(before, after, procedure, NeighborhoodLevel.N1)

    assert audit.changed_carriers == 1
    assert audit.raw_one_hot_hamming == 2


def test_n2_is_a_true_label_swap_not_two_unrelated_relocations() -> None:
    before = _projection()
    swapped = dict(before.x_group)
    swapped["u1"], swapped["u2"] = swapped["u2"], swapped["u1"]
    after = before.replace_block("x_group", swapped)

    audit = validate_transition(before, after, Procedure.F2, NeighborhoodLevel.N2)

    assert audit.changed_carriers == 2
    assert audit.raw_one_hot_hamming == 4
    assert Counter(before.x_group.values()) == Counter(after.x_group.values())

    unrelated = before.replace_block("x_group", {"u1": 12, "u2": 13})
    with pytest.raises(ProjectionError, match="label-count conservation"):
        validate_transition(before, unrelated, Procedure.F2, NeighborhoodLevel.N2)


def test_n3_changes_at_most_four_carriers() -> None:
    before = CoreProjection(
        x_group={f"u{i}": i for i in range(5)},
        s_visit={},
        r_assign={},
    )
    four = before.replace_block("x_group", {**before.x_group, **{f"u{i}": i + 10 for i in range(4)}})
    audit = validate_transition(before, four, Procedure.F2, NeighborhoodLevel.N3)
    assert audit.raw_one_hot_hamming == 8

    one = before.replace_block("x_group", {**before.x_group, "u0": 10})
    with pytest.raises(ProjectionError, match="three or four"):
        validate_transition(before, one, Procedure.F2, NeighborhoodLevel.N3)

    five = before.replace_block("x_group", {f"u{i}": i + 10 for i in range(5)})
    with pytest.raises(ProjectionError, match="three or four"):
        validate_transition(before, five, Procedure.F2, NeighborhoodLevel.N3)


def test_elite_pool_keeps_only_relax_risk_pareto_candidates() -> None:
    candidates = [
        _EliteCandidate(_EliteShell("low-risk"), relaxed_objective=15.0, repair_risk=_EliteRisk(1.0)),
        _EliteCandidate(_EliteShell("balanced"), relaxed_objective=12.0, repair_risk=_EliteRisk(2.0)),
        _EliteCandidate(_EliteShell("low-relax"), relaxed_objective=10.0, repair_risk=_EliteRisk(4.0)),
        _EliteCandidate(_EliteShell("dominated"), relaxed_objective=16.0, repair_risk=_EliteRisk(3.0)),
    ]

    selected = select_pareto_elites(candidates, limit=8)

    assert [candidate.shell.sha256 for candidate in selected] == [
        "low-relax",
        "balanced",
        "low-risk",
    ]


def test_elite_pool_diversity_clipping_is_deterministic_and_keeps_endpoints() -> None:
    candidates = [
        _EliteCandidate(
            _EliteShell(f"candidate-{index:02d}"),
            relaxed_objective=float(100 - index),
            repair_risk=_EliteRisk(float(index)),
        )
        for index in range(12)
    ]

    first = select_pareto_elites(candidates, limit=8)
    second = select_pareto_elites(reversed(candidates), limit=8)
    first_hashes = [candidate.shell.sha256 for candidate in first]

    assert first_hashes == [candidate.shell.sha256 for candidate in second]
    assert "candidate-00" in first_hashes
    assert "candidate-11" in first_hashes
    assert len(first_hashes) == 8


def test_candidate_score_ignores_tiny_risk_delta_but_penalizes_large_repair_burden() -> None:
    lower_risk = _EliteCandidate(
        _EliteShell("lower-risk"),
        relaxed_objective=579.268,
        repair_risk=_EliteRisk(2036.0),
    )
    lower_relax = _EliteCandidate(
        _EliteShell("lower-relax"),
        relaxed_objective=579.244,
        repair_risk=_EliteRisk(2038.0),
    )
    high_risk = _EliteCandidate(
        _EliteShell("high-risk"),
        relaxed_objective=579.228,
        repair_risk=_EliteRisk(637.0),
    )
    repairable = _EliteCandidate(
        _EliteShell("repairable"),
        relaxed_objective=579.248,
        repair_risk=_EliteRisk(273.0),
    )

    assert candidate_preference_key(lower_relax) < candidate_preference_key(lower_risk)
    assert candidate_preference_key(repairable) < candidate_preference_key(high_risk)


def test_outer_start_restores_station_wait_in_arrival_order() -> None:
    y = {
        (slot_id, 30, rank): _NamedVar(f"y[{slot_id},30,{rank}]")
        for slot_id in (10, 11)
        for rank in (0, 1)
    }
    arrival = {slot_id: _NamedVar(f"arrival[{slot_id}]") for slot_id in (10, 11)}
    start = {slot_id: _NamedVar(f"start[{slot_id}]") for slot_id in (10, 11)}
    finish = {slot_id: _NamedVar(f"finish[{slot_id}]") for slot_id in (10, 11)}
    route_finish = {40: _NamedVar("route_finish[40]")}
    station_arrival = {(30, rank): _NamedVar(f"sa[30,{rank}]") for rank in (0, 1)}
    station_finish = {(30, rank): _NamedVar(f"sf[30,{rank}]") for rank in (0, 1)}
    cmax = _NamedVar("cmax")
    values = {
        "arrival[10]": 10.0,
        "start[10]": 10.0,
        "finish[10]": 14.0,
        "arrival[11]": 5.0,
        "start[11]": 5.0,
        "finish[11]": 11.0,
        "route_finish[40]": 2007.0,
        "cmax": 14.0,
    }
    shell = StructuralShell(
        projection=CoreProjection(
            x_group={"u1": 10, "u2": 11},
            s_visit={(10, 20): 30, (11, 21): 30},
            r_assign={10: 40, 11: 41},
        )
    )

    projected = restore_station_wait_start(
        values,
        {
            "y": y,
            "arrival": arrival,
            "start": start,
            "finish": finish,
            "route_finish": route_finish,
            "cmax": cmax,
            "station_arrival_clock": station_arrival,
            "station_finish_clock": station_finish,
        },
        shell,
    )

    assert projected.station_orders == {30: (11, 10)}
    assert projected.values_by_name["y[11,30,0]"] == 1.0
    assert projected.values_by_name["y[10,30,1]"] == 1.0
    assert projected.values_by_name["start[10]"] == 11.0
    assert projected.values_by_name["finish[10]"] == 15.0
    assert projected.projected_cmax == 15.0
    assert projected.added_station_wait_sec == 1.0


def test_outer_start_submits_only_selected_primary_route_arcs() -> None:
    route_arc = {
        (1, 2, 40): _NamedVar("route_arc[1,2,40]"),
        (2, 3, 40): _NamedVar("route_arc[2,3,40]"),
    }

    selected = positive_family_start_values(
        {"route_arc[1,2,40]": 1.0, "route_arc[2,3,40]": 0.0, "pass_x[2,40]": 1.0},
        {"route_arc": route_arc},
        "route_arc",
    )

    assert selected == {"route_arc[1,2,40]": 1.0}
