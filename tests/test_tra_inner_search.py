from __future__ import annotations

from types import SimpleNamespace

from Gurobi.tra_inner_search import (
    PHASE_TWO_BASE_OBJECTIVE_TIEBREAK,
    configure_inner_search,
    f3_balance_coefficients,
    phase_one_time_limit,
    phase_two_attempt_limit,
    phase_two_pool_complete,
    phase_two_quality_limit,
    phase_two_round_complete,
    phase_two_round_time_limit,
    phase_two_search_seed,
    project_inner_start,
    relaxed_quality_tolerance,
    should_run_phase_two,
)
from Gurobi.tra_inner_trace import InnerAttemptTrace
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure


class _Variable:
    def __init__(self, name: str) -> None:
        self.VarName = name


def _family(prefix: str):
    return {prefix: _Variable(prefix)}


def _payload():
    return {
        "x": _family("x"),
        "pair_activate": _family("pair"),
        "slot_robot": _family("robot"),
        "y": _family("y"),
        "hit": _family("hit"),
        "route_arc": _family("arc"),
        "pass_x": _family("pass"),
        "route_owner": _family("owner"),
        "route_time": _family("route_time"),
        "arrival": _family("arrival"),
        "start": _family("start"),
        "finish": _family("finish"),
    }


def test_inner_partial_start_clears_released_block_and_direct_recourse() -> None:
    values = {
        name: 1.0
        for name in (
            "x",
            "pair",
            "robot",
            "y",
            "hit",
            "arc",
            "pass",
            "owner",
            "route_time",
            "arrival",
            "start",
            "finish",
        )
    }
    payload = _payload()

    f1 = project_inner_start(values, payload, Procedure.F1)
    f2 = project_inner_start(values, payload, Procedure.F2)
    f3 = project_inner_start(values, payload, Procedure.F3)

    assert f1 == {}
    assert set(f2) == {"pair", "robot", "y", "arc", "route_time", "arrival"}
    assert set(f3) == {"x", "pair", "y", "hit"}


def test_inner_search_parameters_prioritize_natural_feasible_solutions() -> None:
    params = SimpleNamespace(
        MIPFocus=0,
        Heuristics=0.05,
        PumpPasses=0,
        StartNodeLimit=500,
        MIPGap=0.1,
        PoolSearchMode=2,
    )
    configure_inner_search(SimpleNamespace(Params=params))

    assert params.MIPFocus == 1
    assert params.MIPGap == 0.0
    assert params.PoolSearchMode == 0
    assert params.Heuristics == 0.5
    assert params.PumpPasses == 20
    assert params.StartNodeLimit == 1000


def test_f3_phase_two_balance_coefficients_prioritize_unused_robots() -> None:
    coefficients = f3_balance_coefficients(
        {
            **{slot_id: 0 for slot_id in range(9)},
            **{slot_id: 1 for slot_id in range(9, 12)},
        },
        (0, 1, 2),
    )

    assert coefficients == {0: 9.0, 1: 3.0, 2: 0.0}


def test_inner_two_phase_policy_is_deterministic_and_target_blind() -> None:
    assert PHASE_TWO_BASE_OBJECTIVE_TIEBREAK == 1e-4
    assert phase_one_time_limit(10.0) == 6.0
    assert relaxed_quality_tolerance(100.0) == 0.5
    assert relaxed_quality_tolerance(1000.0) == 2.0
    assert phase_two_quality_limit(580.0, 589.0) == 581.16
    assert phase_two_quality_limit(None, 589.0) == 590.178
    assert phase_two_quality_limit(None, None) is None
    assert should_run_phase_two(
        candidate_count=0,
        phase_one_timed_out=True,
        remaining_sec=4.0,
    )
    assert should_run_phase_two(
        candidate_count=5,
        phase_one_timed_out=False,
        remaining_sec=4.0,
    )
    assert not should_run_phase_two(
        candidate_count=6,
        phase_one_timed_out=True,
        remaining_sec=4.0,
    )
    assert not phase_two_pool_complete(5)
    assert phase_two_pool_complete(6)
    assert not should_run_phase_two(
        candidate_count=0,
        phase_one_timed_out=False,
        remaining_sec=4.0,
    )


def test_f3_phase_two_enumerates_natural_solutions_across_vns_seeds() -> None:
    assert phase_two_attempt_limit(
        Procedure.F3,
        NeighborhoodLevel.N1,
        vns_seed_count=4,
        recourse_active=True,
    ) == 3
    assert phase_two_attempt_limit(
        Procedure.F3,
        NeighborhoodLevel.N3,
        vns_seed_count=4,
        recourse_active=True,
    ) == 3
    assert phase_two_attempt_limit(
        Procedure.F3,
        NeighborhoodLevel.N2,
        vns_seed_count=4,
        recourse_active=True,
    ) == 1
    assert phase_two_attempt_limit(
        Procedure.F3,
        NeighborhoodLevel.N1,
        vns_seed_count=4,
        recourse_active=False,
    ) == 1

    assert phase_two_round_time_limit(5.8, rounds_remaining=3) == 4.3
    assert phase_two_round_time_limit(1.0, rounds_remaining=3) == 1.0 / 3.0
    assert phase_two_search_seed(42, phase_two_attempt=1) == 7961
    assert phase_two_search_seed(42, phase_two_attempt=2) == 15880
    assert phase_two_round_complete(
        candidate_count=2,
        round_start_count=1,
        stop_after_new_candidate=True,
    )
    assert not phase_two_round_complete(
        candidate_count=2,
        round_start_count=1,
        stop_after_new_candidate=False,
    )


def test_inner_attempt_trace_is_append_only_audit_data() -> None:
    trace = InnerAttemptTrace(
        phase="phase_two",
        attempt_index=2,
        search_seed=15880,
        vns_seed_sha256="seed-b",
        requested_time_limit_sec=1.5,
        runtime_sec=1.4,
        solver_status_code=9,
        candidate_count_before=1,
        candidate_count_after=2,
    )

    assert trace.as_audit_payload() == {
        "phase": "phase_two",
        "attempt_index": 2,
        "search_seed": 15880,
        "vns_seed_sha256": "seed-b",
        "requested_time_limit_sec": 1.5,
        "runtime_sec": 1.4,
        "solver_status_code": 9,
        "candidate_count_before": 1,
        "candidate_count_after": 2,
    }
