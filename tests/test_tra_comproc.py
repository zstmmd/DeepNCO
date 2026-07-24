from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from Gurobi.tra_comproc.dp1 import evaluate_dp1_route
from Gurobi.tra_comproc.dp3 import evaluate_dp3_recovery
from Gurobi.tra_comproc.ranking import comproc_candidate_key
from Gurobi.tra_comproc.types import DP2ServiceResult


@dataclass(frozen=True)
class _Var:
    VarName: str


@dataclass(frozen=True)
class _Risk:
    total: float


def test_dp1_extracts_owned_start_to_end_paths_and_delivery_arrivals() -> None:
    payload = {
        "integrate_u_route": True,
        "route_start_nodes": {7: 0},
        "route_end_nodes": {7: 3},
        "route_arc": {
            (0, 1): _Var("arc01"),
            (1, 2): _Var("arc12"),
            (2, 3): _Var("arc23"),
        },
        "pass_x": {
            (0, 7): _Var("pass0"),
            (1, 7): _Var("pass1"),
            (2, 7): _Var("pass2"),
            (3, 7): _Var("pass3"),
        },
        "route_time": {2: _Var("time2")},
        "route_finish": {7: _Var("finish7")},
        "route_tasks": {
            10: SimpleNamespace(slot_id=5, delivery_node=2),
        },
    }
    values = {
        "arc01": 1.0,
        "arc12": 1.0,
        "arc23": 1.0,
        "pass0": 1.0,
        "pass1": 1.0,
        "pass2": 1.0,
        "pass3": 1.0,
        "time2": 11.0,
        "finish7": 15.0,
    }

    result = evaluate_dp1_route(values, payload)

    assert result.feasible
    assert result.robot_paths == {7: (0, 1, 2, 3)}
    assert result.slot_arrival_lower == {5: 11.0}
    assert result.route_end_sec == 15.0


def test_dp3_scores_recoverable_congestion_below_the_fcfs_upper_bound() -> None:
    dp2 = DP2ServiceResult(
        feasible=True,
        slot_arrival={0: 0.0, 1: 1.0, 2: 2.0},
        slot_process_duration={0: 10.0, 1: 10.0, 2: 10.0},
        station_by_slot={0: 0, 1: 0, 2: 0},
    )

    result = evaluate_dp3_recovery(dp2)

    assert result.feasible
    assert result.station_orders == {0: (0, 1, 2)}
    assert result.no_wait_cmax == 12.0
    assert result.feasible_start_cmax == 30.0
    assert result.station_overlap_sec == 26.0
    assert result.recourse_score == 12.0 + 26.0 / 6.0

    floored = evaluate_dp3_recovery(dp2, no_wait_cmax_floor=20.0)
    assert floored.no_wait_cmax == 20.0
    assert floored.recourse_score == 20.0 + 26.0 / 6.0


def test_dp3_recourse_score_penalizes_station_workload_imbalance() -> None:
    dp2 = DP2ServiceResult(
        feasible=True,
        slot_arrival={0: 0.0, 1: 1.0, 2: 0.0},
        slot_process_duration={0: 10.0, 1: 10.0, 2: 2.0},
        station_by_slot={0: 0, 1: 0, 2: 1},
    )

    result = evaluate_dp3_recovery(dp2)

    assert result.station_overlap_sec == 9.0
    assert result.station_workload_imbalance == 9.0
    assert result.recourse_score == 17.0


def test_comproc_ranking_prefers_recourse_score_before_feasible_start_cmax() -> None:
    def candidate(
        name: str,
        feasible: bool,
        cmax: float,
        objective: float,
        recourse_score: float,
    ):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=name),
            repair_risk=_Risk(1.0),
            relaxed_objective=10.0,
            comproc=SimpleNamespace(
                feasible=feasible,
                projected_cmax=cmax,
                recourse_score=recourse_score,
                verified_cmax=cmax,
                projected_objective=objective,
            ),
        )

    infeasible = candidate("a", False, 1.0, 1.0, 1.0)
    infeasible.comproc.verified_cmax = 21.0
    recoverable = candidate("b", True, 20.0, 20.0, 18.0)
    smaller_start = candidate("c", True, 19.0, 30.0, 19.0)

    assert sorted(
        [infeasible, recoverable, smaller_start],
        key=comproc_candidate_key,
    ) == [
        recoverable,
        smaller_start,
        infeasible,
    ]
