from __future__ import annotations

from types import SimpleNamespace

import gurobipy as gp
import pytest

import Gurobi.tra_inner as tra_inner_module
from Gurobi.tra_f1_live_seed import build_f1_live_seed_start
from Gurobi.tra_inner import PaperInnerTemplate
from Gurobi.tra_inner_search import (
    PHASE_TWO_BASE_OBJECTIVE_TIEBREAK,
    configure_inner_search,
    f3_balance_coefficients,
    initial_inner_start_values,
    phase_one_time_limit,
    phase_two_attempt_limit,
    phase_two_inner_start_values,
    phase_two_pool_complete,
    phase_two_quality_limit,
    phase_two_round_complete,
    phase_two_round_time_limit,
    phase_two_search_seed,
    phase_two_start_seed_sha256,
    project_inner_start,
    relaxed_quality_tolerance,
    should_run_phase_two,
)
from Gurobi.tra_inner_trace import InnerAttemptTrace
from Gurobi.tra_neighborhood import DualBlockSpec, NeighborhoodLevel, Procedure
from Gurobi.tra_projection import CoreProjection, StructuralShell


class _Variable:
    def __init__(self, name: str) -> None:
        self.VarName = name


def _family(prefix: str):
    return {prefix: _Variable(prefix)}


def _f1_seed_fixture():
    payload = {
        "a": {1: _Variable("a[1]")},
        "x": {("unit-1", 1): _Variable("x[unit-1,1]")},
        "slot_robot": {(1, 7): _Variable("slot_robot[1,7]")},
        "flip": {(1, 5): _Variable("flip[1,5]")},
        "sort": {},
        "carry": {},
        "hit": {},
        "noise": {},
        "flip_hit": {},
        "pair_activate": {
            (1, 5, 0): _Variable("pair_act[1,5,0]"),
            (1, 5, 1): _Variable("pair_act[1,5,1]"),
        },
        "y": {
            (1, 0, 0): _Variable("y[1,0,0]"),
            (1, 1, 0): _Variable("y[1,1,0]"),
        },
        "arrival": {1: _Variable("arrival[1]")},
        "start": {1: _Variable("start[1]")},
        "finish": {1: _Variable("finish[1]")},
        "cmax": _Variable("Cmax"),
    }
    source_values = {
        "a[1]": 1.0,
        "x[unit-1,1]": 1.0,
        "slot_robot[1,7]": 1.0,
        "flip[1,5]": 1.0,
        "pair_act[1,5,0]": 1.0,
        "pair_act[1,5,1]": 0.0,
        "y[1,0,0]": 1.0,
        "y[1,1,0]": 0.0,
        "arrival[1]": 2.0,
        "start[1]": 2.0,
        "finish[1]": 5.0,
        "Cmax": 5.0,
    }
    reference_shell = StructuralShell(
        projection=CoreProjection(
            x_group={"unit-1": 1},
            s_visit={(1, 5): 0},
            r_assign={1: 7},
        )
    )
    seed_projection = reference_shell.projection.replace_block(
        "s_visit",
        {(1, 5): 1},
    )
    return payload, source_values, reference_shell, seed_projection


def test_f1_live_seed_start_repairs_y_and_pair_without_route_or_time_values() -> None:
    payload, source_values, reference_shell, seed_projection = _f1_seed_fixture()

    values = build_f1_live_seed_start(
        source_values,
        reference_shell,
        seed_projection,
        payload,
    )

    assert values == {
        "a[1]": 1.0,
        "x[unit-1,1]": 1.0,
        "slot_robot[1,7]": 1.0,
        "flip[1,5]": 1.0,
        "pair_act[1,5,1]": 1.0,
        "y[1,1,0]": 1.0,
    }
    assert not any(
        name.startswith(("route_", "arrival", "start", "finish", "Cmax"))
        for name in values
    )


def test_inner_start_selection_uses_live_f1_seed_only_for_f1() -> None:
    f1_seed = {"pair_act[1,5,1]": 1.0, "y[1,1,0]": 1.0}

    assert initial_inner_start_values(
        Procedure.F1,
        projected_start={},
        vns_start_values=({"pair_act[1,5,0]": 1.0},),
        f1_live_seed_starts=(f1_seed,),
    ) == f1_seed
    assert initial_inner_start_values(
        Procedure.F2,
        projected_start={"pair_act[1,5,0]": 1.0},
        vns_start_values=({"x[unit-1,2]": 1.0},),
        f1_live_seed_starts=(f1_seed,),
    ) == {
        "pair_act[1,5,0]": 1.0,
        "x[unit-1,2]": 1.0,
    }


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
    assert phase_two_attempt_limit(
        Procedure.F1,
        NeighborhoodLevel.N1,
        vns_seed_count=4,
        f1_live_seed_count=4,
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
    assert phase_two_inner_start_values(
        Procedure.F2,
        projected_start={"fixed": 1.0},
        vns_start_values=({"vns-1": 1.0}, {"vns-2": 1.0}),
        f1_live_seed_starts=({"f1-0": 1.0}, {"f1-1": 1.0}),
        phase_two_attempt=1,
    ) == {"fixed": 1.0, "vns-2": 1.0}


def test_f1_n2_phase_two_uses_later_repaired_live_starts() -> None:
    f1_starts = (
        {"f1-0": 1.0},
        {"f1-1": 1.0},
        {"f1-2": 1.0},
        {"f1-3": 1.0},
    )
    f1_hashes = ("f1-hash-0", "f1-hash-1", "f1-hash-2", "f1-hash-3")

    assert phase_two_attempt_limit(
        Procedure.F1,
        NeighborhoodLevel.N2,
        vns_seed_count=4,
        f1_live_seed_count=len(f1_starts),
        recourse_active=False,
    ) == 3
    assert phase_two_inner_start_values(
        Procedure.F1,
        projected_start={"stale": 1.0},
        vns_start_values=({"vns-1": 1.0},),
        f1_live_seed_starts=f1_starts,
        phase_two_attempt=1,
    ) == {"stale": 1.0, "f1-1": 1.0}
    assert phase_two_inner_start_values(
        Procedure.F1,
        projected_start={},
        vns_start_values=({"vns-2": 1.0},),
        f1_live_seed_starts=f1_starts,
        phase_two_attempt=3,
    ) == {"f1-3": 1.0}
    assert phase_two_start_seed_sha256(
        Procedure.F1,
        vns_seed_sha256=("vns-hash-0",),
        f1_live_seed_sha256=f1_hashes,
        phase_two_attempt=2,
    ) == "f1-hash-2"


def test_f1_start_hash_follows_the_repaired_start_sequence() -> None:
    assert phase_two_start_seed_sha256(
        Procedure.F1,
        vns_seed_sha256=("unrelated-vns-hash",),
        f1_live_seed_sha256=("f1-hash-0", "f1-hash-1", "f1-hash-2"),
        phase_two_attempt=1,
    ) == "f1-hash-1"
    assert phase_two_start_seed_sha256(
        Procedure.F1,
        vns_seed_sha256=("vns-hash-0", "vns-hash-1"),
        f1_live_seed_sha256=(),
        phase_two_attempt=1,
    ) == "vns-hash-1"


def test_inner_f1_n2_phase_two_installs_later_repaired_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __init__(self) -> None:
            self.ModelSense = gp.GRB.MINIMIZE
            self.Params = SimpleNamespace(Seed=0)
            self.Status = gp.GRB.TIME_LIMIT
            self.SolCount = 0
            self.ObjBound = 0.0

        def getObjective(self):
            return gp.LinExpr()

        def optimize(self, _callback) -> None:
            self.Status = gp.GRB.TIME_LIMIT

        def reset(self, _clear: int) -> None:
            return None

        def update(self) -> None:
            return None

    class _Template:
        def __init__(self) -> None:
            self.model = _Model()
            self.compiled = SimpleNamespace(
                cfg=SimpleNamespace(tra_inner_no_station_wait=True),
            )
            self.payload = {}
            self.solver = SimpleNamespace(
                _status_label=lambda status: f"status-{status}",
            )
            self.installed_starts: list[dict[str, float]] = []

        def reset_for_solve(self) -> None:
            return None

        def install_start(self, values, *, clear_existing: bool) -> None:
            assert clear_existing is True
            self.installed_starts.append(dict(values))

        def set_internal_cutoff(self, objective, tolerance: float) -> None:
            assert objective is None
            assert tolerance > 0.0

        def set_time_limit(self, seconds: float) -> None:
            self.model.Params.TimeLimit = seconds

    template = _Template()
    monkeypatch.setattr(
        tra_inner_module,
        "configure_inner_search",
        lambda _model: None,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "apply_local_neighborhood",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "should_run_phase_two",
        lambda **_kwargs: True,
    )

    result = PaperInnerTemplate(template).solve(
        StructuralShell(
            projection=CoreProjection(
                x_group={},
                s_visit={},
                r_assign={},
            )
        ),
        procedure=Procedure.F1,
        neighborhood=NeighborhoodLevel.N2,
        time_limit_sec=0.3,
        incumbent_objective=None,
        f1_live_seed_starts=(
            {"f1-0": 1.0},
            {"f1-1": 1.0},
            {"f1-2": 1.0},
            {"f1-3": 1.0},
        ),
        f1_live_seed_sha256=(
            "f1-hash-0",
            "f1-hash-1",
            "f1-hash-2",
            "f1-hash-3",
        ),
    )

    assert template.installed_starts == [
        {"f1-0": 1.0},
        {"f1-1": 1.0},
        {"f1-2": 1.0},
        {"f1-3": 1.0},
    ]
    assert [trace.vns_seed_sha256 for trace in result.attempt_traces] == [
        "f1-hash-0",
        "f1-hash-1",
        "f1-hash-2",
        "f1-hash-3",
    ]


def test_inner_f1_phase_one_trace_falls_back_to_vns_hash_without_live_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __init__(self) -> None:
            self.ModelSense = gp.GRB.MINIMIZE
            self.Params = SimpleNamespace(Seed=0)
            self.Status = gp.GRB.TIME_LIMIT
            self.SolCount = 0
            self.ObjBound = 0.0

        def getObjective(self):
            return gp.LinExpr()

        def optimize(self, _callback) -> None:
            self.Status = gp.GRB.TIME_LIMIT

        def update(self) -> None:
            return None

    class _Template:
        def __init__(self) -> None:
            self.model = _Model()
            self.compiled = SimpleNamespace(
                cfg=SimpleNamespace(tra_inner_no_station_wait=True),
            )
            self.payload = {}
            self.solver = SimpleNamespace(
                _status_label=lambda status: f"status-{status}",
            )

        def reset_for_solve(self) -> None:
            return None

        def install_start(self, _values, *, clear_existing: bool) -> None:
            assert clear_existing is True

        def set_internal_cutoff(self, objective, tolerance: float) -> None:
            assert objective is None
            assert tolerance > 0.0

        def set_time_limit(self, seconds: float) -> None:
            self.model.Params.TimeLimit = seconds

    monkeypatch.setattr(
        tra_inner_module,
        "configure_inner_search",
        lambda _model: None,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "apply_local_neighborhood",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "should_run_phase_two",
        lambda **_kwargs: False,
    )

    result = PaperInnerTemplate(_Template()).solve(
        StructuralShell(
            projection=CoreProjection(
                x_group={},
                s_visit={},
                r_assign={},
            )
        ),
        procedure=Procedure.F1,
        neighborhood=NeighborhoodLevel.N1,
        time_limit_sec=0.1,
        incumbent_objective=None,
        vns_seed_sha256=("vns-hash-0",),
    )

    assert result.attempt_traces[0].vns_seed_sha256 == "vns-hash-0"


def test_inner_dual_block_solve_applies_diagnostic_neighborhood_and_start_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __init__(self) -> None:
            self.ModelSense = gp.GRB.MINIMIZE
            self.Params = SimpleNamespace(Seed=0)
            self.Status = gp.GRB.TIME_LIMIT
            self.SolCount = 0
            self.ObjBound = 0.0

        def getObjective(self):
            return gp.LinExpr()

        def optimize(self, _callback) -> None:
            self.Status = gp.GRB.TIME_LIMIT

        def update(self) -> None:
            return None

    class _Template:
        def __init__(self) -> None:
            self.model = _Model()
            self.compiled = SimpleNamespace(
                cfg=SimpleNamespace(tra_inner_no_station_wait=True),
            )
            self.payload = {
                "x": _family("x"),
                "pair_activate": _family("pair"),
                "slot_robot": _family("robot"),
                "y": _family("y"),
                "flip": _family("flip"),
                "sort": _family("sort"),
                "carry": _family("carry"),
                "hit": _family("hit"),
                "noise": _family("noise"),
                "flip_hit": _family("flip_hit"),
                "route_arc": _family("arc"),
                "route_time": _family("route_time"),
                "route_load": _family("route_load"),
                "route_finish": _family("route_finish"),
                "arrival": _family("arrival"),
                "start": _family("start"),
                "finish": _family("finish"),
                "pass_x": _family("pass"),
                "route_owner": _family("owner"),
                "cmax": _Variable("Cmax"),
            }
            self.solver = SimpleNamespace(
                _status_label=lambda status: f"status-{status}",
            )
            self.installed_starts: list[dict[str, float]] = []

        def reset_for_solve(self) -> None:
            return None

        def install_start(self, values, *, clear_existing: bool) -> None:
            assert clear_existing is True
            self.installed_starts.append(dict(values))

        def set_internal_cutoff(self, objective, tolerance: float) -> None:
            assert objective is None
            assert tolerance > 0.0

        def set_time_limit(self, seconds: float) -> None:
            self.model.Params.TimeLimit = seconds

    template = _Template()
    applied: list[DualBlockSpec] = []
    monkeypatch.setattr(
        tra_inner_module,
        "configure_inner_search",
        lambda _model: None,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "apply_local_neighborhood",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dual-block solve must not use single-block local branch")
        ),
    )
    monkeypatch.setattr(
        tra_inner_module,
        "apply_dual_block_neighborhood",
        lambda _template, _shell, spec: applied.append(spec),
        raising=False,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "should_run_phase_two",
        lambda **_kwargs: False,
    )
    spec = DualBlockSpec("F2_F3", ("x_group", "r_assign"), hamming_limit=8)

    result = PaperInnerTemplate(template).solve_dual_block(
        StructuralShell(
            projection=CoreProjection(
                x_group={},
                s_visit={},
                r_assign={},
            )
        ),
        spec=spec,
        time_limit_sec=0.1,
        incumbent_objective=None,
        start_values={
            "x": 1.0,
            "pair": 1.0,
            "robot": 1.0,
            "y": 1.0,
            "arc": 1.0,
            "route_time": 1.0,
            "route_load": 1.0,
            "route_finish": 1.0,
            "arrival": 1.0,
            "start": 1.0,
            "finish": 1.0,
            "pass": 1.0,
            "owner": 1.0,
            "Cmax": 1.0,
        },
        search_seed=123,
        vns_seed_sha256=("dual-seed-0",),
    )

    assert applied == [spec]
    assert template.installed_starts == [{"pair": 1.0, "y": 1.0}]
    assert result.procedure == "F2_F3"
    assert result.neighborhood == "DUAL"
    assert result.search_seeds == (123,)
    assert result.vns_seed_sha256 == ("dual-seed-0",)
    assert result.attempt_traces[0].phase == "phase_one"
    assert result.attempt_traces[0].vns_seed_sha256 == "dual-seed-0"


def test_inner_dual_block_phase_two_enumerates_later_vns_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __init__(self) -> None:
            self.ModelSense = gp.GRB.MINIMIZE
            self.Params = SimpleNamespace(Seed=0)
            self.Status = gp.GRB.TIME_LIMIT
            self.SolCount = 0
            self.ObjBound = 0.0
            self.reset_count = 0

        def getObjective(self):
            return gp.LinExpr()

        def optimize(self, _callback) -> None:
            self.Status = gp.GRB.TIME_LIMIT

        def reset(self, _clear: int) -> None:
            self.reset_count += 1

        def update(self) -> None:
            return None

    class _Template:
        def __init__(self) -> None:
            self.model = _Model()
            self.compiled = SimpleNamespace(
                cfg=SimpleNamespace(tra_inner_no_station_wait=True),
            )
            self.payload = {
                "pair_activate": _family("pair"),
                "y": _family("y"),
            }
            self.solver = SimpleNamespace(
                _status_label=lambda status: f"status-{status}",
            )
            self.installed_starts: list[dict[str, float]] = []

        def reset_for_solve(self) -> None:
            return None

        def install_start(self, values, *, clear_existing: bool) -> None:
            assert clear_existing is True
            self.installed_starts.append(dict(values))

        def set_internal_cutoff(self, objective, tolerance: float) -> None:
            assert objective is None
            assert tolerance > 0.0

        def set_time_limit(self, seconds: float) -> None:
            self.model.Params.TimeLimit = seconds

    template = _Template()
    monkeypatch.setattr(
        tra_inner_module,
        "configure_inner_search",
        lambda _model: None,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "apply_dual_block_neighborhood",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        tra_inner_module,
        "should_run_phase_two",
        lambda **_kwargs: True,
    )

    result = PaperInnerTemplate(template).solve_dual_block(
        StructuralShell(
            projection=CoreProjection(
                x_group={},
                s_visit={},
                r_assign={},
            )
        ),
        spec=DualBlockSpec("F2_F3", ("x_group", "r_assign"), hamming_limit=8),
        time_limit_sec=0.5,
        incumbent_objective=None,
        start_values={"pair": 1.0, "y": 1.0},
        search_seed=42,
        vns_start_values=(
            {"dual-start-0": 1.0},
            {"dual-start-1": 1.0},
            {"dual-start-2": 1.0},
        ),
        vns_seed_sha256=("dual-hash-0", "dual-hash-1", "dual-hash-2"),
    )

    assert template.installed_starts == [
        {"pair": 1.0, "y": 1.0, "dual-start-0": 1.0},
        {"pair": 1.0, "y": 1.0, "dual-start-1": 1.0},
        {"pair": 1.0, "y": 1.0, "dual-start-2": 1.0},
    ]
    assert result.attempt_count == 3
    assert result.search_seeds == (42, 7961, 15880)
    assert [trace.phase for trace in result.attempt_traces] == [
        "phase_one",
        "phase_two",
        "phase_two",
    ]
    assert [trace.vns_seed_sha256 for trace in result.attempt_traces] == [
        "dual-hash-0",
        "dual-hash-1",
        "dual-hash-2",
    ]


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
