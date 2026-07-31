from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_inner import InnerCandidate
from Gurobi.tra_model_state import ModelSnapshot
from Gurobi.tra_outer import OuterDisposition, has_unresolved_improvement_potential
from Gurobi.tra_outer_continuation import OuterContinuationState
from Gurobi.tra_outer_sequence import ImmediateOuterSequence
from Gurobi.tra_projection import CoreProjection, StructuralShell
from Gurobi.tra_risk import RepairRisk
from Gurobi.tra_reserve_phase import ReservePhase
from Gurobi.tra_budget_policy import (
    OuterBudgetPolicy,
    RegularInnerBudgetPolicy,
    ReserveBudgetPolicy,
    f3_support_expansion_needed,
)
from Gurobi.tra_scheduler import ProcedureStep, RetryRegistry, RotationScheduler, RuntimeLedger
from Gurobi.tra_search_state import (
    InnerSearchEvidence,
    RecourseCalibration,
    SearchState,
    TRAIncumbent,
)
from Gurobi.tra_work_queue import (
    DeferredInnerStep,
    PendingOuterShell,
    ReserveStage,
    SearchWorkQueues,
)


def test_scheduler_keeps_strict_f1_f2_f3_order_and_escalates_per_block() -> None:
    scheduler = RotationScheduler(max_procedures=12)
    observed = []
    for _ in range(6):
        step = scheduler.current_step()
        observed.append((step.procedure, step.neighborhood))
        scheduler.complete_step(improved=False)

    assert observed == [
        (Procedure.F1, NeighborhoodLevel.N1),
        (Procedure.F2, NeighborhoodLevel.N1),
        (Procedure.F3, NeighborhoodLevel.N1),
        (Procedure.F1, NeighborhoodLevel.N2),
        (Procedure.F2, NeighborhoodLevel.N2),
        (Procedure.F3, NeighborhoodLevel.N2),
    ]


def test_rotation_yields_pending_outer_only_after_a_complete_f1_f2_f3_cycle() -> None:
    scheduler = RotationScheduler(max_procedures=12)

    assert not scheduler.should_yield_to_reserve(pending_outer_count=1)
    scheduler.complete_step(improved=False)  # F1
    assert not scheduler.should_yield_to_reserve(pending_outer_count=1)
    scheduler.complete_step(improved=False)  # F2
    assert not scheduler.should_yield_to_reserve(pending_outer_count=1)
    scheduler.complete_step(improved=False)  # F3
    assert scheduler.should_yield_to_reserve(pending_outer_count=1)
    assert not scheduler.should_yield_to_reserve(pending_outer_count=0)


def test_plateau_transition_keeps_other_processes_vns_progress() -> None:
    scheduler = RotationScheduler(max_procedures=12)
    scheduler.complete_step(improved=False)
    scheduler.complete_step(improved=False)
    scheduler.complete_step(improved=True, primary_improved=False)

    step = scheduler.current_step()
    assert step.procedure is Procedure.F1
    assert step.neighborhood is NeighborhoodLevel.N2


def test_plateau_transition_rechecks_n2_on_other_blocks_from_n3() -> None:
    scheduler = RotationScheduler(max_procedures=12)
    for _ in range(5):
        scheduler.complete_step(improved=False)
    assert scheduler.current_step().procedure is Procedure.F3
    assert scheduler.current_step().neighborhood is NeighborhoodLevel.N2

    scheduler.complete_step(improved=True, primary_improved=False)

    step = scheduler.current_step()
    assert step.procedure is Procedure.F1
    assert step.neighborhood is NeighborhoodLevel.N2
    scheduler.complete_step(improved=False)
    assert scheduler.current_step().procedure is Procedure.F2
    assert scheduler.current_step().neighborhood is NeighborhoodLevel.N2


def test_default_f1_plateau_transition_resets_f1_to_n1() -> None:
    scheduler = RotationScheduler(max_procedures=12)

    scheduler.complete_step(improved=True, primary_improved=False)  # F1 N1
    scheduler.complete_step(improved=False)  # F2
    scheduler.complete_step(improved=False)  # F3

    step = scheduler.current_step()
    assert step.procedure is Procedure.F1
    assert step.neighborhood is NeighborhoodLevel.N1


def test_diagnostic_f1_plateau_transition_escalates_to_n2_then_n3() -> None:
    scheduler = RotationScheduler(
        max_procedures=12,
        enable_f1_plateau_escalation=True,
    )

    scheduler.complete_step(improved=True, primary_improved=False)  # F1 N1
    scheduler.complete_step(improved=False)  # F2
    scheduler.complete_step(improved=False)  # F3
    assert scheduler.current_step() == ProcedureStep(
        4,
        2,
        Procedure.F1,
        NeighborhoodLevel.N2,
    )

    scheduler.complete_step(improved=True, primary_improved=False)  # F1 N2
    scheduler.complete_step(improved=False)  # F2
    scheduler.complete_step(improved=False)  # F3

    assert scheduler.current_step() == ProcedureStep(
        7,
        3,
        Procedure.F1,
        NeighborhoodLevel.N3,
    )


def test_diagnostic_f2_plateau_transition_escalates_to_n2() -> None:
    scheduler = RotationScheduler(
        max_procedures=12,
        plateau_escalation_procedures=(Procedure.F2,),
    )

    scheduler.complete_step(improved=False)  # F1
    scheduler.complete_step(improved=True, primary_improved=False)  # F2 N1
    scheduler.complete_step(improved=False)  # F3
    scheduler.complete_step(improved=False)  # F1

    assert scheduler.current_step() == ProcedureStep(
        5,
        2,
        Procedure.F2,
        NeighborhoodLevel.N2,
    )


def test_strict_improvement_resets_all_procedures_to_n1() -> None:
    scheduler = RotationScheduler()
    scheduler.complete_step(improved=False)  # F1 -> N2
    scheduler.complete_step(improved=True)   # F2 improvement resets every block

    assert scheduler.current_step().procedure is Procedure.F3
    assert scheduler.current_step().neighborhood is NeighborhoodLevel.N1
    scheduler.complete_step(improved=False)
    assert scheduler.current_step().procedure is Procedure.F1
    assert scheduler.current_step().neighborhood is NeighborhoodLevel.N1


def test_three_stagnant_cycles_stop_only_after_deferred_queue_is_empty() -> None:
    scheduler = RotationScheduler(stagnant_cycle_limit=3)
    for _ in range(9):
        scheduler.complete_step(improved=False)

    assert not scheduler.should_stop(runtime_remaining_sec=10.0, deferred_empty=False)
    assert scheduler.should_stop(runtime_remaining_sec=10.0, deferred_empty=True)


def test_reserve_improvement_restarts_strict_rotation_at_f1_n1() -> None:
    scheduler = RotationScheduler(stagnant_cycle_limit=3)
    for _ in range(9):
        scheduler.complete_step(improved=False)

    scheduler.restart_after_external_improvement()

    assert scheduler.stagnant_cycles == 0
    assert scheduler.current_step().procedure is Procedure.F1
    assert scheduler.current_step().neighborhood is NeighborhoodLevel.N1


def test_external_restart_can_preserve_f1_escalation_only_when_requested() -> None:
    scheduler = RotationScheduler(
        max_procedures=12,
        enable_f1_plateau_escalation=True,
    )
    for _ in range(3):
        scheduler.complete_step(improved=False)
    assert scheduler.current_step() == ProcedureStep(
        4,
        2,
        Procedure.F1,
        NeighborhoodLevel.N2,
    )

    scheduler.restart_after_external_improvement(preserve_f1_level=True)

    assert scheduler.current_step() == ProcedureStep(
        4,
        2,
        Procedure.F1,
        NeighborhoodLevel.N2,
    )

    scheduler.restart_after_external_improvement()

    assert scheduler.current_step() == ProcedureStep(
        4,
        2,
        Procedure.F1,
        NeighborhoodLevel.N1,
    )


def test_external_restart_can_preserve_configured_f2_level() -> None:
    scheduler = RotationScheduler(
        max_procedures=12,
        plateau_escalation_procedures=(Procedure.F2,),
    )
    scheduler.complete_step(improved=False)  # F1 -> N2
    scheduler.complete_step(improved=False)  # F2 -> N2
    scheduler.complete_step(improved=False)  # F3 -> N2

    scheduler.restart_after_external_improvement(
        preserve_procedure_levels=(Procedure.F2,),
    )
    scheduler.complete_step(improved=False)  # F1

    assert scheduler.current_step() == ProcedureStep(
        5,
        2,
        Procedure.F2,
        NeighborhoodLevel.N2,
    )


def test_unresolved_shell_gets_exactly_one_reserve_retry() -> None:
    retries = RetryRegistry()
    assert retries.register_unresolved("shell-a")
    assert retries.can_retry("shell-a")
    retries.mark_retried("shell-a")
    assert not retries.can_retry("shell-a")
    assert not retries.register_unresolved("shell-a")


def test_runtime_ledger_uses_soft_30_55_15_quotas_under_one_hard_clock() -> None:
    now = [100.0]
    ledger = RuntimeLedger(
        hard_limit_sec=80.0,
        inner_quota_sec=24.0,
        outer_quota_sec=44.0,
        reserve_quota_sec=12.0,
        clock=lambda: now[0],
    )
    ledger.start()
    assert ledger.slice_for("inner", 4) == pytest.approx(6.0)
    ledger.record("inner", 4.0)
    assert ledger.slice_for("inner", 4) == pytest.approx(5.0)

    now[0] = 179.0
    assert ledger.remaining_sec == pytest.approx(1.0)
    assert ledger.slice_for("outer", 1) == pytest.approx(1.0)


def test_reserve_can_borrow_unused_regular_soft_quotas_after_stagnation() -> None:
    now = [100.0]
    ledger = RuntimeLedger(
        hard_limit_sec=80.0,
        inner_quota_sec=24.0,
        outer_quota_sec=44.0,
        reserve_quota_sec=12.0,
        clock=lambda: now[0],
    )
    ledger.start()
    ledger.record("inner", 10.0)
    ledger.record("outer", 20.0)

    assert ledger.slice_for("reserve", 2) == pytest.approx(6.0)
    assert ledger.slice_for("reserve", 2, borrow_unused=True) == pytest.approx(25.0)


def test_runtime_slice_preserves_formal_clock_safety_buffer() -> None:
    now = [100.0]
    ledger = RuntimeLedger(
        hard_limit_sec=10.0,
        inner_quota_sec=3.0,
        outer_quota_sec=5.5,
        reserve_quota_sec=1.5,
        safety_buffer_sec=0.5,
        clock=lambda: now[0],
    )
    ledger.start()
    now[0] = 109.0

    assert ledger.remaining_sec == pytest.approx(1.0)
    assert ledger.slice_for("outer", 1) == pytest.approx(0.5)


def test_runtime_slice_does_not_launch_subminimum_solver_fragments() -> None:
    now = [100.0]
    ledger = RuntimeLedger(
        hard_limit_sec=10.0,
        inner_quota_sec=3.0,
        outer_quota_sec=5.5,
        reserve_quota_sec=1.5,
        safety_buffer_sec=1.5,
        minimum_solver_slice_sec=2.0,
        clock=lambda: now[0],
    )
    ledger.start()
    now[0] = 106.8

    assert ledger.allocatable_remaining_sec == pytest.approx(1.7)
    assert ledger.slice_for("outer", 1) == 0.0


def test_runtime_slice_raises_small_quota_share_to_minimum_when_time_remains() -> None:
    ledger = RuntimeLedger(
        hard_limit_sec=20.0,
        inner_quota_sec=3.0,
        outer_quota_sec=5.5,
        reserve_quota_sec=1.5,
        minimum_solver_slice_sec=2.0,
        clock=lambda: 100.0,
    )
    ledger.start()

    assert ledger.slice_for("inner", 4) == pytest.approx(2.0)


def test_work_queue_deduplicates_only_the_same_deferred_neighborhood() -> None:
    queues = SearchWorkQueues()
    shell = SimpleNamespace(sha256="incumbent-a")
    queues.add_deferred(
        DeferredInnerStep(
            reference_shell=shell,
            start_values={},
            step=ProcedureStep(7, 3, Procedure.F1, NeighborhoodLevel.N1),
        )
    )
    queues.add_deferred(
        DeferredInnerStep(
            reference_shell=shell,
            start_values={},
            step=ProcedureStep(10, 4, Procedure.F1, NeighborhoodLevel.N2),
        )
    )

    assert queues.deferred_count == 2
    assert queues.pop_deferred().step.procedure_index == 7
    assert queues.pop_deferred().step.procedure_index == 10


def test_deferred_queue_accepts_run_evidence_priority_without_reordering_regular_rotation() -> None:
    queues = SearchWorkQueues()
    shell = SimpleNamespace(sha256="incumbent-a")
    for index, procedure in ((7, Procedure.F1), (12, Procedure.F3)):
        queues.add_deferred(
            DeferredInnerStep(
                reference_shell=shell,
                start_values={},
                step=ProcedureStep(index, 4, procedure, NeighborhoodLevel.N3),
            )
        )

    selected = queues.pop_deferred(
        priority=lambda item: 0 if item.step.procedure is Procedure.F3 else 1
    )

    assert selected.step.procedure is Procedure.F3


def test_inner_evidence_prefers_candidate_producing_block_for_deferred_work() -> None:
    empty = InnerSearchEvidence()
    productive = InnerSearchEvidence()
    for _ in range(3):
        empty.observe(candidate_count=0, timed_out=True)
    productive.observe(candidate_count=1, timed_out=False)
    productive.observe(candidate_count=0, timed_out=True)

    assert productive.reserve_priority < empty.reserve_priority


def test_post_n3_reserve_priority_rotates_with_cold_f1_last_on_equal_evidence() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
        last_n3_improving_procedure=Procedure.F2,
    )
    items = {
        procedure: DeferredInnerStep(
            reference_shell=state.search_shell,
            start_values={},
            step=ProcedureStep(index, 4, procedure, NeighborhoodLevel.N3),
        )
        for index, procedure in enumerate(Procedure, start=10)
    }

    ordered = sorted(items, key=lambda procedure: state.deferred_priority(items[procedure]))

    assert ordered == [Procedure.F3, Procedure.F2, Procedure.F1]


def test_f2_deferred_work_precedes_repeating_f3_after_an_f3_n3_success() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
        last_n3_improving_procedure=Procedure.F3,
    )
    items = {
        procedure: DeferredInnerStep(
            reference_shell=state.search_shell,
            start_values={},
            step=ProcedureStep(index, 4, procedure, NeighborhoodLevel.N3),
        )
        for index, procedure in enumerate(Procedure, start=10)
    }

    ordered = sorted(
        items,
        key=lambda procedure: state.deferred_priority(items[procedure]),
    )

    assert ordered == [Procedure.F2, Procedure.F3, Procedure.F1]


def test_f2_reserve_prioritizes_size_preserving_n2_exchange() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
    )
    items = {
        neighborhood: DeferredInnerStep(
            reference_shell=state.search_shell,
            start_values={},
            step=ProcedureStep(
                index,
                4,
                Procedure.F2,
                neighborhood,
            ),
        )
        for index, neighborhood in enumerate(NeighborhoodLevel, start=10)
    }

    ordered = sorted(
        items,
        key=lambda neighborhood: state.deferred_priority(items[neighborhood]),
    )

    assert ordered == [
        NeighborhoodLevel.N2,
        NeighborhoodLevel.N3,
        NeighborhoodLevel.N1,
    ]


def test_inner_effort_shrinks_empty_probes_but_keeps_productive_blocks_at_full_slice() -> None:
    empty = InnerSearchEvidence()
    productive = InnerSearchEvidence()
    assert empty.regular_effort_multiplier == pytest.approx(1.0)

    for _ in range(3):
        empty.observe(candidate_count=0, timed_out=True)
        productive.observe(candidate_count=1, timed_out=False)

    assert empty.regular_effort_multiplier == pytest.approx(0.4)
    assert productive.regular_effort_multiplier == pytest.approx(1.0)


def test_f3_support_pressure_gets_a_stable_regular_candidate_pool_slice() -> None:
    policy = RegularInnerBudgetPolicy(
        f3_n2_cross_process_hard_fraction=0.025,
    )

    assert policy.stabilize_slice(
        3.0,
        hard_limit_sec=288.0,
        allocatable_remaining_sec=100.0,
        f3_n1_support_expansion=False,
        cross_process_f3_n2=True,
        f3_n3_balance=False,
    ) == pytest.approx(7.2)
    assert policy.stabilize_slice(
        3.0,
        hard_limit_sec=288.0,
        allocatable_remaining_sec=100.0,
        f3_n1_support_expansion=False,
        cross_process_f3_n2=False,
        f3_n3_balance=False,
    ) == pytest.approx(3.0)
    assert policy.stabilize_slice(
        3.0,
        hard_limit_sec=288.0,
        allocatable_remaining_sec=100.0,
        f3_n1_support_expansion=True,
        cross_process_f3_n2=False,
        f3_n3_balance=False,
    ) == pytest.approx(14.4)
    assert policy.stabilize_slice(
        3.0,
        hard_limit_sec=288.0,
        allocatable_remaining_sec=100.0,
        f3_n1_support_expansion=False,
        cross_process_f3_n2=False,
        f3_n3_balance=True,
    ) == pytest.approx(21.6)


def test_f3_support_pressure_is_target_blind_and_stops_at_capacity_band() -> None:
    labels = (0, 1, 2)

    assert f3_support_expansion_needed(
        {**{slot: 0 for slot in range(9)}, **{slot: 1 for slot in range(9, 12)}},
        labels,
    )
    assert f3_support_expansion_needed(
        {
            **{slot: 0 for slot in range(8)},
            **{slot: 1 for slot in range(8, 11)},
            11: 2,
        },
        labels,
    )
    assert f3_support_expansion_needed(
        {
            **{slot: 0 for slot in range(7)},
            **{slot: 1 for slot in range(7, 10)},
            **{slot: 2 for slot in range(10, 12)},
        },
        labels,
    )
    assert not f3_support_expansion_needed(
        {
            **{slot: 0 for slot in range(6)},
            **{slot: 1 for slot in range(6, 9)},
            **{slot: 2 for slot in range(9, 12)},
        },
        labels,
    )
    assert not f3_support_expansion_needed(
        {
            **{slot: 0 for slot in range(4)},
            **{slot: 1 for slot in range(4, 6)},
            **{slot: 2 for slot in range(6, 12)},
        },
        labels,
    )


def test_outer_attempts_are_capped_at_four_percent_of_the_hard_budget() -> None:
    policy = ReserveBudgetPolicy(restart_hard_fraction=0.04)

    assert policy.cap_outer_slice(
        100.0,
        hard_limit_sec=288.0,
        reserve_retry=False,
    ) == pytest.approx(11.52)
    assert policy.cap_outer_slice(
        100.0,
        hard_limit_sec=288.0,
        reserve_retry=True,
    ) == pytest.approx(11.52)


def test_outer_retry_promotes_only_a_finite_improving_objective_bound() -> None:
    policy = OuterBudgetPolicy()

    assert policy.retry_is_bound_promoted(
        objective_bound=579.2,
        incumbent_objective=589.22,
    )
    assert not policy.retry_is_bound_promoted(
        objective_bound=589.22,
        incumbent_objective=589.22,
    )
    assert not policy.retry_is_bound_promoted(
        objective_bound=float("nan"),
        incumbent_objective=589.22,
    )
    assert not policy.retry_is_bound_promoted(
        objective_bound=579.2,
        incumbent_objective=None,
    )


def test_promoted_outer_retry_gets_fifteen_percent_cap() -> None:
    policy = ReserveBudgetPolicy()

    assert policy.cap_outer_slice(
        100.0,
        hard_limit_sec=288.0,
        reserve_retry=True,
        bound_promoted=False,
    ) == pytest.approx(11.52)
    assert policy.cap_outer_slice(
        100.0,
        hard_limit_sec=288.0,
        reserve_retry=True,
        bound_promoted=True,
    ) == pytest.approx(43.2)


def test_reserve_phase_assigns_promoted_slice_to_improving_bound_retry() -> None:
    shell = SimpleNamespace(sha256="candidate-shell")
    state = SearchState(search_shell=SimpleNamespace(sha256="parent-shell"), start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.22,
        shell=state.search_shell,
    )
    state.queues.add_pending(
        PendingOuterShell(
            shell=shell,
            start_values={},
            step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3),
            reserve_retry=True,
            relaxed_objective=579.2,
            repair_risk_total=2.0,
            validation_bound=579.2,
        )
    )
    runtime = SimpleNamespace(
        allocatable_remaining_sec=100.0,
        hard_limit_sec=288.0,
        slice_for=lambda *args, **kwargs: 100.0,
    )
    phase = ReservePhase(
        templates=SimpleNamespace(),
        runtime=runtime,
        scheduler=SimpleNamespace(),
        audit=SimpleNamespace(),
        record_verified=lambda *args: None,
    )
    observed = {}

    def run_outer(fake_state, time_limit_sec, *, bound_promoted):
        observed["time_limit_sec"] = time_limit_sec
        observed["bound_promoted"] = bound_promoted
        fake_state.queues.pop_pending()
        return None

    phase._run_outer = run_outer

    assert not phase.run(state)
    assert observed == {
        "time_limit_sec": pytest.approx(43.2),
        "bound_promoted": True,
    }


@pytest.mark.parametrize(
    ("enabled", "cmax_improved", "expected_preserve"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_reserve_preserves_f1_level_only_for_diagnostic_plateau_refinement(
    enabled: bool,
    cmax_improved: bool,
    expected_preserve: bool,
) -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="parent-shell"),
        start_values={},
    )
    state.queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="candidate-shell"),
            start_values={},
            step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3),
            reserve_retry=False,
            relaxed_objective=579.2,
            repair_risk_total=2.0,
            validation_bound=579.2,
        )
    )
    scheduler = SimpleNamespace(
        restart_after_external_improvement=Mock(),
    )
    phase = ReservePhase(
        templates=SimpleNamespace(),
        runtime=SimpleNamespace(
            allocatable_remaining_sec=100.0,
            hard_limit_sec=288.0,
            slice_for=lambda *args, **kwargs: 10.0,
        ),
        scheduler=scheduler,
        audit=SimpleNamespace(),
        record_verified=lambda *args: None,
        enable_f1_plateau_escalation=enabled,
    )

    def run_outer(fake_state, _time_limit_sec, *, bound_promoted):
        assert bound_promoted is False
        fake_state.queues.pop_pending()
        return SimpleNamespace(
            structural_change=True,
            cmax_improved=cmax_improved,
        )

    phase._run_outer = run_outer

    assert phase.run(state)
    scheduler.restart_after_external_improvement.assert_called_once_with(
        preserve_f1_level=expected_preserve
    )


def test_deferred_inner_slices_use_process_specific_hard_caps() -> None:
    policy = ReserveBudgetPolicy(f2_n2_inner_hard_fraction=0.135)

    assert policy.cap_deferred_inner(
        100.0,
        hard_limit_sec=288.0,
        procedure=Procedure.F2,
        neighborhood=NeighborhoodLevel.N2,
    ) == pytest.approx(38.88)
    assert policy.cap_deferred_inner(
        100.0,
        hard_limit_sec=288.0,
        procedure=Procedure.F2,
        neighborhood=NeighborhoodLevel.N3,
    ) == pytest.approx(38.88)
    assert policy.cap_deferred_inner(
        100.0,
        hard_limit_sec=288.0,
        procedure=Procedure.F3,
        neighborhood=NeighborhoodLevel.N2,
    ) == pytest.approx(28.8)
    assert policy.cap_deferred_inner(
        100.0,
        hard_limit_sec=288.0,
        procedure=Procedure.F1,
        neighborhood=NeighborhoodLevel.N3,
    ) == pytest.approx(14.4)


def test_outer_policy_allows_one_promising_time_limit_continuation() -> None:
    policy = OuterBudgetPolicy()
    result = SimpleNamespace(
        solver_status_code=9,
        resumed_search=False,
        objective_bound=579.0,
    )

    assert policy.should_continue(
        result,
        incumbent_objective=607.0,
        incumbent_cmax=607.0,
        projected_cmax=610.0,
    )
    result.resumed_search = True
    assert not policy.should_continue(
        result,
        incumbent_objective=607.0,
        incumbent_cmax=607.0,
        projected_cmax=600.0,
    )


def test_outer_continuation_resumes_only_the_untouched_same_shell() -> None:
    continuation = OuterContinuationState()
    continuation.remember("shell-a")

    assert continuation.plan("shell-a", resume_requested=True) is True
    assert continuation.plan("shell-a", resume_requested=False) is False
    continuation.remember("shell-a")
    assert continuation.plan("shell-b", resume_requested=True) is False


def test_same_shell_refinement_updates_incumbent_without_clearing_deferred_work() -> None:
    shell = SimpleNamespace(sha256="shell-a")

    def verified(objective: float, cmax: float, candidate_shell=shell):
        return SimpleNamespace(
            shell=candidate_shell,
            snapshot=SimpleNamespace(
                solver_objective=objective,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(search_shell=shell, start_values={})
    state.accept(verified(596.212, 596.0))
    state.queues.add_deferred(
        DeferredInnerStep(
            reference_shell=shell,
            start_values={},
            step=ProcedureStep(8, 3, Procedure.F1, NeighborhoodLevel.N3),
        )
    )

    outcome = state.accept(verified(596.210, 596.0))

    assert not outcome.structural_change
    assert not outcome.cmax_improved
    assert state.queues.deferred_count == 1

    new_shell = SimpleNamespace(sha256="shell-b")
    outcome = state.accept(verified(595.21, 595.0, new_shell))
    assert outcome.structural_change
    assert outcome.cmax_improved
    assert state.queues.empty


def test_equal_cmax_shell_advances_rotation_without_replacing_global_best() -> None:
    first_shell = SimpleNamespace(sha256="shell-a")
    plateau_shell = SimpleNamespace(sha256="shell-b")

    def verified(shell, objective: float, cmax: float):
        return SimpleNamespace(
            shell=shell,
            snapshot=SimpleNamespace(
                solver_objective=objective,
                values_by_name={"Cmax": cmax, "marker": objective},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(search_shell=first_shell, start_values={})
    state.accept(verified(first_shell, 589.198, 589.0))

    outcome = state.accept(verified(plateau_shell, 589.245, 589.0))

    assert outcome.structural_change
    assert not outcome.cmax_improved
    assert state.incumbent is not None
    assert state.incumbent.shell.sha256 == "shell-a"
    assert state.search_shell.sha256 == "shell-b"
    assert state.search_incumbent is not None
    assert state.start_values["marker"] == pytest.approx(589.245)


def test_worse_cmax_shell_cannot_replace_the_search_or_global_incumbent() -> None:
    first_shell = SimpleNamespace(sha256="shell-a")
    worse_shell = SimpleNamespace(sha256="shell-b")

    def verified(shell, objective: float, cmax: float):
        return SimpleNamespace(
            shell=shell,
            snapshot=SimpleNamespace(
                solver_objective=objective,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(search_shell=first_shell, start_values={})
    state.accept(verified(first_shell, 589.198, 589.0))

    outcome = state.accept(verified(worse_shell, 590.150, 590.0))

    assert not outcome.structural_change
    assert not outcome.cmax_improved
    assert state.incumbent is not None
    assert state.incumbent.shell.sha256 == "shell-a"
    assert state.search_shell.sha256 == "shell-a"


def test_n3_record_to_record_shake_can_cross_one_bounded_cmax_valley() -> None:
    first_shell = SimpleNamespace(sha256="shell-a")
    shake_shell = SimpleNamespace(sha256="shell-b")
    descent_shell = SimpleNamespace(sha256="shell-c")

    def verified(shell, objective: float, cmax: float):
        return SimpleNamespace(
            shell=shell,
            snapshot=SimpleNamespace(
                solver_objective=objective,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(search_shell=first_shell, start_values={})
    state.accept(verified(first_shell, 589.198, 589.0))
    n3 = ProcedureStep(9, 3, Procedure.F2, NeighborhoodLevel.N3)

    shake = state.accept(verified(shake_shell, 603.2, 603.0), step=n3)

    assert shake.structural_change
    assert shake.uphill_shake
    assert state.incumbent is not None
    assert state.incumbent.verified_cmax == pytest.approx(589.0)
    assert state.search_incumbent is not None
    assert state.search_incumbent.verified_cmax == pytest.approx(603.0)
    assert state.certification_cmax_limit(n3) == pytest.approx(603.0)

    descent = state.accept(
        verified(descent_shell, 597.2, 597.0),
        step=ProcedureStep(10, 4, Procedure.F3, NeighborhoodLevel.N1),
    )

    assert descent.structural_change
    assert not descent.uphill_shake
    assert state.search_incumbent is not None
    assert state.search_incumbent.verified_cmax == pytest.approx(597.0)
    assert state.incumbent.verified_cmax == pytest.approx(589.0)


def test_record_to_record_rejects_a_fourth_n1_uphill_or_out_of_band_shell() -> None:
    first_shell = SimpleNamespace(sha256="shell-a")

    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(search_shell=first_shell, start_values={})
    state.accept(verified("shell-a", 589.0))
    n1 = ProcedureStep(9, 3, Procedure.F2, NeighborhoodLevel.N1)
    state.accept(verified("shell-b", 603.0), step=n1)
    state.accept(verified("shell-c", 605.0), step=n1)
    state.accept(verified("shell-d", 606.0), step=n1)

    assert not state.accept(
        verified("shell-e", 606.0),
        step=n1,
    ).structural_change
    state.restore_global_search()
    assert not state.accept(
        verified("shell-f", 608.0),
        step=n1,
    ).structural_change


def test_n1_uphill_shake_allows_three_same_process_transitions_before_rotation() -> None:
    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(
        search_shell=SimpleNamespace(sha256="warm-shell"),
        start_values={},
    )
    f3_n1 = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N1)
    state.accept(verified("shell-a", 589.0), step=f3_n1)

    assert state.certification_cmax_limit(f3_n1) == pytest.approx(606.67)
    assert state.accept(
        verified("shell-b", 597.0),
        step=f3_n1,
    ).uphill_shake
    assert state.accept(
        verified("shell-c", 598.0),
        step=f3_n1,
    ).uphill_shake
    assert state.certification_cmax_limit(f3_n1) == pytest.approx(598.0)

    f2_n3 = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N3)
    assert state.certification_cmax_limit(f2_n3) == pytest.approx(606.67)


def test_same_process_n2_must_rotate_after_a_structural_transition() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="warm-shell"),
        start_values={},
    )

    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    f3_n3 = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)
    state.accept(verified("global-shell", 589.0), step=f3_n3)
    f3_n2 = ProcedureStep(10, 4, Procedure.F3, NeighborhoodLevel.N2)

    assert state.certification_cmax_limit(f3_n2) == pytest.approx(589.0)
    assert not state.accept(
        verified("same-process-n2", 597.0),
        step=f3_n2,
    ).structural_change


def test_n1_can_cross_the_record_band_to_expand_assignment_support() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="global-shell"),
        start_values={},
    )

    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state.accept(verified("global-shell", 589.0))
    f3_n1 = ProcedureStep(10, 4, Procedure.F3, NeighborhoodLevel.N1)

    assert state.certification_cmax_limit(f3_n1) == pytest.approx(606.67)
    outcome = state.accept(verified("support-expanded", 592.0), step=f3_n1)
    assert outcome.structural_change
    assert outcome.uphill_shake


def test_record_band_allows_three_alternating_cross_process_shakes() -> None:
    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(
        search_shell=SimpleNamespace(sha256="warm-shell"),
        start_values={},
    )
    state.accept(
        verified("global-shell", 589.0),
        step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3),
    )
    outcomes = (
        state.accept(
            verified("f2-shake-1", 603.0),
            step=ProcedureStep(12, 4, Procedure.F2, NeighborhoodLevel.N3),
        ),
        state.accept(
            verified("f3-shake-2", 605.0),
            step=ProcedureStep(15, 5, Procedure.F3, NeighborhoodLevel.N3),
        ),
        state.accept(
            verified("f2-shake-3", 606.0),
            step=ProcedureStep(18, 6, Procedure.F2, NeighborhoodLevel.N3),
        ),
    )

    assert all(outcome.structural_change for outcome in outcomes)
    assert all(outcome.uphill_shake for outcome in outcomes)
    assert state.uphill_shakes == 3
    assert state.incumbent is not None
    assert state.incumbent.verified_cmax == pytest.approx(589.0)


def test_size_preserving_f2_n2_exchange_can_enter_the_record_band() -> None:
    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(
        search_shell=SimpleNamespace(sha256="warm-shell"),
        start_values={},
    )
    state.accept(
        verified("f3-global", 589.0),
        step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3),
    )
    f2_n2 = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N2)

    assert state.certification_cmax_limit(f2_n2) == pytest.approx(606.67)
    outcome = state.accept(verified("f2-n2-shake", 605.0), step=f2_n2)

    assert outcome.structural_change
    assert outcome.uphill_shake
    assert state.last_n3_improving_procedure is Procedure.F2
    assert state.certification_cmax_limit(f2_n2) == pytest.approx(605.0)


def test_same_process_candidate_yields_after_three_structural_transitions() -> None:
    def verified(shell_hash: str, cmax: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            snapshot=SimpleNamespace(
                solver_objective=cmax + 0.2,
                values_by_name={"Cmax": cmax},
            ),
            verified_cmax=cmax,
        )

    state = SearchState(
        search_shell=SimpleNamespace(sha256="warm-shell"),
        start_values={},
    )
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)
    state.accept(verified("f3-a", 589.0), step=step)
    state.accept(verified("f3-b", 589.0), step=step)
    state.accept(verified("f3-c", 589.0), step=step)
    ordinary = SimpleNamespace(
        shell=SimpleNamespace(sha256="f3-d"),
        comproc=SimpleNamespace(projected_cmax=600.0),
    )

    assert state.select_unattempted_candidate(
        state.search_shell,
        step,
        (ordinary,),
    ) is None

    improving_projection = SimpleNamespace(
        shell=SimpleNamespace(sha256="f3-e"),
        comproc=SimpleNamespace(projected_cmax=580.0),
    )
    assert state.select_unattempted_candidate(
        state.search_shell,
        step,
        (improving_projection,),
    ) is improving_projection


def test_initial_f1_selection_prefers_sparse_z_for_same_core_projection() -> None:
    projection = CoreProjection(
        x_group={"u0": 0, "u1": 1},
        s_visit={(0, 10): 0, (1, 11): 1},
        r_assign={0: 0, 1: 0},
    )
    parent = StructuralShell(projection=projection)
    dense_shell = StructuralShell(
        projection=projection,
        z_actions={
            "hit": {(0, 1): 1, (0, 2): 1, (1, 3): 1},
            "carry": {(0, 1): 1, (0, 2): 1, (1, 3): 1},
        },
    )
    sparse_shell = StructuralShell(
        projection=projection,
        z_actions={
            "hit": {(0, 1): 1, (1, 3): 1},
            "carry": {(0, 1): 1, (1, 3): 1},
        },
    )

    def candidate(shell: StructuralShell, projected_cmax: float) -> InnerCandidate:
        return InnerCandidate(
            shell=shell,
            snapshot=ModelSnapshot(
                values_by_name={},
                solver_objective=855.0,
                solver_cmax=projected_cmax,
                callback_runtime_sec=0.0,
            ),
            relaxed_objective=855.0,
            repair_risk=RepairRisk(
                total=0.0,
                station_overlap_sec=0.0,
                station_workload_imbalance=0.0,
                warm_disturbance_hamming=2,
            ),
            comproc=SimpleNamespace(
                feasible=True,
                projected_cmax=projected_cmax,
                projected_objective=projected_cmax,
                recourse_score=projected_cmax,
            ),
        )

    dense = candidate(dense_shell, 100.0)
    sparse = candidate(sparse_shell, 200.0)

    initial_state = SearchState(search_shell=parent, start_values={})
    assert (
        initial_state.select_unattempted_candidate(
            parent,
            ProcedureStep(1, 1, Procedure.F1, NeighborhoodLevel.N1),
            (dense, sparse),
        )
        is sparse
    )

    incumbent_state = SearchState(search_shell=parent, start_values={})
    incumbent_state.incumbent = TRAIncumbent(
        shell=parent,
        snapshot=ModelSnapshot(
            values_by_name={},
            solver_objective=300.0,
            solver_cmax=300.0,
            callback_runtime_sec=0.0,
        ),
        verified_cmax=300.0,
        objective=300.0,
    )
    assert (
        incumbent_state.select_unattempted_candidate(
            parent,
            ProcedureStep(2, 1, Procedure.F1, NeighborhoodLevel.N1),
            (dense, sparse),
        )
        is dense
    )


def test_diversified_search_yields_after_three_neighborhood_submissions() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=600.0)
    f3_n2 = ProcedureStep(8, 3, Procedure.F3, NeighborhoodLevel.N2)
    f3_n3 = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)
    f3_n1 = ProcedureStep(10, 3, Procedure.F3, NeighborhoodLevel.N1)

    def candidate(shell_hash: str, projected_cmax: float = 600.0):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            comproc=SimpleNamespace(projected_cmax=projected_cmax),
        )

    assert state.select_unattempted_candidate(
        shell,
        f3_n2,
        (candidate("f3-a"),),
    ) is not None
    assert state.select_unattempted_candidate(
        shell,
        f3_n3,
        (candidate("f3-b"),),
    ) is not None
    assert state.select_unattempted_candidate(
        shell,
        f3_n1,
        (candidate("f3-c"),),
    ) is not None
    assert state.select_unattempted_candidate(
        shell,
        f3_n2,
        (candidate("f3-d"),),
        allow_diverse_neighborhood_repeat=True,
    ) is None


def test_diversified_search_yields_after_one_submission_per_neighborhood() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=600.0)
    f3_n3 = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)

    def candidate(shell_hash: str, projected_cmax: float = 600.0):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            comproc=SimpleNamespace(projected_cmax=projected_cmax),
        )

    assert state.select_unattempted_candidate(
        shell,
        f3_n3,
        (candidate("f3-a"),),
    ) is not None
    assert state.select_unattempted_candidate(
        shell,
        f3_n3,
        (candidate("f3-b"),),
    ) is None
    diverse_repeat = candidate("f3-diverse")
    assert state.select_unattempted_candidate(
        shell,
        f3_n3,
        (diverse_repeat,),
        allow_diverse_neighborhood_repeat=True,
    ) is diverse_repeat
    assert state.select_unattempted_candidate(
        shell,
        f3_n3,
        (candidate("f3-third"),),
        allow_diverse_neighborhood_repeat=True,
    ) is None

    improving = candidate("f3-c", projected_cmax=580.0)
    assert state.select_unattempted_candidate(
        shell,
        f3_n3,
        (improving,),
    ) is improving


def test_global_search_yields_after_one_outer_submission() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=589.0)
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)

    def candidate(shell_hash: str):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            comproc=SimpleNamespace(projected_cmax=600.0),
        )

    assert state.select_unattempted_candidate(
        shell,
        step,
        (candidate("f3-a"),),
    ) is not None
    assert state.select_unattempted_candidate(
        shell,
        step,
        (candidate("f3-b"),),
    ) is None


def test_first_full_incumbent_does_not_consume_rotation_submission_quota() -> None:
    warm_shell = SimpleNamespace(sha256="warm-shell")
    first_shell = SimpleNamespace(sha256="first-shell")
    state = SearchState(search_shell=warm_shell, start_values={})
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)

    first_candidate = SimpleNamespace(
        shell=first_shell,
        comproc=SimpleNamespace(
            projected_cmax=1005.0,
            recourse_score=676.5,
        ),
    )
    assert state.select_unattempted_candidate(
        warm_shell,
        step,
        (first_candidate,),
    ) is first_candidate
    assert state.submission_neighborhood_counts[NeighborhoodLevel.N3] == 1

    state.accept(
        SimpleNamespace(
            shell=first_shell,
            snapshot=SimpleNamespace(
                solver_objective=589.2,
                values_by_name={},
            ),
            verified_cmax=589.0,
        ),
        step=step,
    )

    assert state.submission_neighborhood_counts == {}
    assert state.consecutive_submission_count == 0
    assert state.consecutive_transition_count == 1
    next_candidate = SimpleNamespace(
        shell=SimpleNamespace(sha256="post-bootstrap-n3"),
        comproc=SimpleNamespace(
            projected_cmax=605.0,
            recourse_score=580.0,
        ),
    )
    assert state.select_unattempted_candidate(
        first_shell,
        ProcedureStep(18, 6, Procedure.F3, NeighborhoodLevel.N3),
        (next_candidate,),
    ) is next_candidate


def test_n3_candidate_above_the_target_blind_record_band_is_not_submitted() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=589.0)
    step = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N3)

    def candidate(shell_hash: str, recourse_score: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            comproc=SimpleNamespace(
                projected_cmax=590.0,
                recourse_score=recourse_score,
            ),
        )

    assert state.select_unattempted_candidate(
        shell,
        step,
        (candidate("outside", 622.0),),
    ) is None
    inside = candidate("inside", 600.0)
    assert state.select_unattempted_candidate(
        shell,
        step,
        (inside,),
    ) is inside


def test_candidate_selection_reports_certification_band_rejection() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=589.0)
    step = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N3)
    outside = SimpleNamespace(
        shell=SimpleNamespace(sha256="outside-band"),
        comproc=SimpleNamespace(
            projected_cmax=590.0,
            recourse_score=622.0,
        ),
    )

    selection = state.select_unattempted_candidate_with_dispositions(
        shell,
        step,
        (outside,),
    )

    assert selection.candidate is None
    assert selection.dispositions == (
        {
            "shell_sha256": "outside-band",
            "disposition": "certification_band",
        },
    )


def test_n3_band_uses_only_post_incumbent_observed_recourse_error() -> None:
    calibration = RecourseCalibration()
    calibration.remember_submission(
        "first-shell",
        Procedure.F3,
        676.5,
        calibration_eligible=False,
    )
    calibration.observe_verification(
        "first-shell",
        verified_cmax=589.0,
        prior_incumbent_cmax=None,
    )
    calibration.observe_verification(
        "first-shell",
        verified_cmax=589.0,
        prior_incumbent_cmax=589.0,
    )
    assert calibration.allowance(Procedure.F3) == 0.0

    calibration.remember_submission(
        "rotated-shell",
        Procedure.F3,
        625.0,
        calibration_eligible=True,
    )
    calibration.observe_verification(
        "rotated-shell",
        verified_cmax=589.0,
        prior_incumbent_cmax=589.0,
    )
    assert calibration.allowance(Procedure.F3) == pytest.approx(36.0)
    assert calibration.allowance(Procedure.F2) == 0.0

    shell = SimpleNamespace(sha256="rotated-shell")
    state = SearchState(
        search_shell=shell,
        start_values={},
        recourse_calibration=calibration,
    )
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=589.0)
    step = ProcedureStep(12, 5, Procedure.F3, NeighborhoodLevel.N3)
    calibrated = SimpleNamespace(
        shell=SimpleNamespace(sha256="balanced-n3"),
        comproc=SimpleNamespace(
            projected_cmax=629.0,
            recourse_score=629.0,
        ),
    )

    assert state.candidate_within_certification_band(step, calibrated)


def test_small_neighborhood_candidate_is_not_pruned_by_dp3_proxy() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=589.0)
    step = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N2)
    candidate = SimpleNamespace(
        shell=SimpleNamespace(sha256="n2-conservative-proxy"),
        comproc=SimpleNamespace(
            projected_cmax=623.0,
            recourse_score=622.0,
        ),
    )

    assert state.select_unattempted_candidate(
        shell,
        step,
        (candidate,),
    ) is candidate


def test_f2_n2_uses_projected_order_inside_the_recourse_uncertainty_band() -> None:
    shell = SimpleNamespace(sha256="record-shell")
    state = SearchState(search_shell=shell, start_values={})
    step = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N2)

    def candidate(
        shell_hash: str,
        *,
        recourse_score: float,
        projected_cmax: float,
    ):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            relaxed_objective=579.2,
            repair_risk=SimpleNamespace(total=1.0),
            comproc=SimpleNamespace(
                feasible=True,
                projected_cmax=projected_cmax,
                projected_objective=projected_cmax + 0.2,
                recourse_score=recourse_score,
            ),
        )

    recourse_first = candidate(
        "recourse-first",
        recourse_score=604.21,
        projected_cmax=744.0,
    )
    projected_first = candidate(
        "projected-first",
        recourse_score=604.58,
        projected_cmax=726.0,
    )
    outside_band = candidate(
        "outside-band",
        recourse_score=609.12,
        projected_cmax=671.0,
    )

    assert state.select_unattempted_candidate(
        shell,
        step,
        (recourse_first, projected_first, outside_band),
    ) is projected_first


def test_same_process_record_n2_uses_target_blind_recourse_band() -> None:
    shell = SimpleNamespace(sha256="record-shell")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=589.0)
    state.consecutive_transition_procedure = Procedure.F3
    state.consecutive_transition_count = 1
    step = ProcedureStep(11, 4, Procedure.F3, NeighborhoodLevel.N2)

    def candidate(shell_hash: str, recourse_score: float):
        return SimpleNamespace(
            shell=SimpleNamespace(sha256=shell_hash),
            comproc=SimpleNamespace(
                projected_cmax=recourse_score,
                recourse_score=recourse_score,
            ),
        )

    assert state.select_unattempted_candidate(
        shell,
        step,
        (candidate("far-same-process", 622.0),),
    ) is None
    near = candidate("near-same-process", 590.0)
    assert state.select_unattempted_candidate(
        shell,
        step,
        (near,),
    ) is near


def test_uphill_branch_defers_archive_candidates_from_older_parents() -> None:
    old_parent = SimpleNamespace(
        sha256="old-parent",
        projection=SimpleNamespace(
            x_group={"a": 0, "b": 1},
            s_visit={(0, 1): 0},
            r_assign={0: 0},
        ),
    )
    current = SimpleNamespace(
        sha256="current-uphill",
        projection=old_parent.projection,
    )
    archived_candidate = SimpleNamespace(
        shell=SimpleNamespace(
            sha256="old-sibling",
            projection=SimpleNamespace(
                x_group={"a": 1, "b": 0},
                s_visit={(0, 1): 0},
                r_assign={0: 0},
            ),
        ),
        relaxed_objective=590.0,
        repair_risk=SimpleNamespace(total=1.0),
        comproc=SimpleNamespace(
            feasible=True,
            recourse_score=590.0,
            projected_cmax=590.0,
            projected_objective=590.0,
            verified_cmax=590.0,
        ),
    )
    state = SearchState(search_shell=current, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=old_parent,
    )
    state.search_incumbent = SimpleNamespace(
        verified_cmax=596.0,
        objective=596.2,
        shell=current,
    )
    state.candidate_archive.remember(
        old_parent,
        ProcedureStep(2, 1, Procedure.F2, NeighborhoodLevel.N2),
        (archived_candidate,),
    )
    state.queues = SimpleNamespace(empty=False)

    assert state.on_uphill_branch
    assert state.ranked_archive(Procedure.F2) == ()
    assert not state.has_compatible_archive

    state.queues = SimpleNamespace(empty=True)
    assert [
        item.candidate.shell.sha256
        for item in state.ranked_archive(Procedure.F2)
    ] == ["old-sibling"]
    assert state.has_compatible_archive


def test_uphill_branch_allows_one_same_process_sibling_before_rotation() -> None:
    old_parent = SimpleNamespace(
        sha256="old-parent",
        projection=SimpleNamespace(
            x_group={"a": 0, "b": 1},
            s_visit={(0, 1): 0},
            r_assign={0: 0},
        ),
    )
    current = SimpleNamespace(
        sha256="current-uphill",
        projection=SimpleNamespace(
            x_group={"a": 1, "b": 0},
            s_visit={(0, 1): 0},
            r_assign={0: 0},
        ),
    )
    sibling = SimpleNamespace(
        shell=SimpleNamespace(
            sha256="same-process-sibling",
            projection=SimpleNamespace(
                x_group={"a": 2, "b": 0},
                s_visit={(0, 1): 0},
                r_assign={0: 0},
            ),
        ),
        relaxed_objective=590.0,
        repair_risk=SimpleNamespace(total=1.0),
        comproc=SimpleNamespace(
            feasible=True,
            recourse_score=590.0,
            projected_cmax=590.0,
            projected_objective=590.0,
            verified_cmax=590.0,
        ),
    )
    state = SearchState(search_shell=current, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=old_parent,
    )
    state.search_incumbent = SimpleNamespace(
        verified_cmax=591.0,
        objective=591.2,
        shell=current,
    )
    state.consecutive_transition_procedure = Procedure.F2
    state.consecutive_transition_count = 1
    state.last_transition_neighborhood = NeighborhoodLevel.N1
    state.queues = SimpleNamespace(empty=False)
    state.candidate_archive.remember(
        old_parent,
        ProcedureStep(2, 1, Procedure.F2, NeighborhoodLevel.N1),
        (sibling,),
    )

    assert [
        item.candidate.shell.sha256
        for item in state.ranked_archive(Procedure.F2)
    ] == ["same-process-sibling"]
    assert state.ranked_archive(Procedure.F3) == ()

    state.consecutive_transition_count = 2
    assert state.ranked_archive(Procedure.F2) == ()


def test_archive_repeat_must_match_the_uphill_transition_neighborhood() -> None:
    shell = SimpleNamespace(sha256="uphill-shell")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=SimpleNamespace(sha256="global-shell"),
    )
    state.search_incumbent = SimpleNamespace(verified_cmax=592.0)
    state.consecutive_transition_procedure = Procedure.F3
    state.last_transition_neighborhood = NeighborhoodLevel.N1

    assert state.allow_archive_neighborhood_repeat(
        ProcedureStep(10, 4, Procedure.F3, NeighborhoodLevel.N1)
    )
    assert not state.allow_archive_neighborhood_repeat(
        ProcedureStep(11, 4, Procedure.F3, NeighborhoodLevel.N2)
    )
    assert not state.allow_archive_neighborhood_repeat(
        ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N1)
    )


def test_outer_horizon_uses_observed_natural_candidate_rate() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
    )
    assert state.estimated_outer_horizon(9) == 5

    for _ in range(3):
        state.observe_inner(Procedure.F1, candidate_count=0, timed_out=True)
        state.observe_inner(Procedure.F2, candidate_count=1, timed_out=False)
        state.observe_inner(Procedure.F3, candidate_count=1, timed_out=False)

    assert state.estimated_outer_horizon(9) == 6


def test_vns_windows_advance_per_shell_procedure_and_neighborhood() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
    )

    assert state.next_vns_offset(
        state.search_shell,
        Procedure.F3,
        NeighborhoodLevel.N3,
    ) == 0
    assert state.next_vns_offset(
        state.search_shell,
        Procedure.F3,
        NeighborhoodLevel.N3,
    ) == 4
    assert state.next_vns_offset(
        SimpleNamespace(sha256="incumbent-b"),
        Procedure.F3,
        NeighborhoodLevel.N3,
    ) == 0
    assert state.next_vns_offset(
        state.search_shell,
        Procedure.F2,
        NeighborhoodLevel.N3,
    ) == 0


def test_work_queue_prioritizes_repair_risk_then_outer_bound() -> None:
    queues = SearchWorkQueues()
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N2)
    for shell_hash, bound, risk in (
        ("higher-bound", 580.0, 1.0),
        ("lower-bound-higher-risk", 579.0, 5.0),
        ("lower-bound-lower-risk", 579.0, 2.0),
    ):
        queues.add_pending(
            PendingOuterShell(
                shell=SimpleNamespace(sha256=shell_hash),
                start_values={},
                step=step,
                reserve_retry=True,
                relaxed_objective=578.0,
                repair_risk_total=risk,
                validation_bound=bound,
            )
        )

    assert queues.next_reserve_stage(prefer_deferred=True) is ReserveStage.OUTER
    assert queues.reserve_horizon(ReserveStage.OUTER) == 2
    assert queues.pop_pending().shell.sha256 == "higher-bound"
    assert queues.pop_pending().shell.sha256 == "lower-bound-lower-risk"


def test_work_queue_prioritizes_comproc_projection_before_relaxed_bound() -> None:
    queues = SearchWorkQueues()
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N2)
    for shell_hash, projected_cmax, bound in (
        ("better-bound", 700.0, 578.0),
        ("better-projection", 620.0, 580.0),
    ):
        queues.add_pending(
            PendingOuterShell(
                shell=SimpleNamespace(sha256=shell_hash),
                start_values={},
                step=step,
                reserve_retry=True,
                relaxed_objective=579.0,
                repair_risk_total=2.0,
                validation_bound=bound,
                projected_cmax=projected_cmax,
                projected_objective=projected_cmax + 0.2,
                start_feasible=True,
            )
        )

    assert queues.pop_pending().shell.sha256 == "better-projection"


def test_deferred_priority_uses_recoverable_candidate_evidence_before_rotation() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
    )
    for _ in range(4):
        state.observe_inner(
            Procedure.F1,
            candidate_count=0,
            recoverable_count=0,
            timed_out=True,
        )
        state.observe_inner(
            Procedure.F2,
            candidate_count=1,
            recoverable_count=1,
            timed_out=True,
        )
    state.last_n3_improving_procedure = Procedure.F3
    f1 = DeferredInnerStep(
        reference_shell=state.search_shell,
        start_values={},
        step=ProcedureStep(10, 4, Procedure.F1, NeighborhoodLevel.N3),
    )
    f2 = DeferredInnerStep(
        reference_shell=state.search_shell,
        start_values={},
        step=ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N3),
    )

    assert state.deferred_priority(f2) < state.deferred_priority(f1)


def test_deferred_priority_uses_process_start_before_cold_f1_on_equal_evidence() -> None:
    state = SearchState(
        search_shell=SimpleNamespace(sha256="incumbent-a"),
        start_values={},
    )
    for procedure in (Procedure.F1, Procedure.F2):
        state.observe_inner(
            procedure,
            candidate_count=0,
            recoverable_count=0,
            timed_out=True,
        )
    f1 = DeferredInnerStep(
        reference_shell=state.search_shell,
        start_values={},
        step=ProcedureStep(10, 4, Procedure.F1, NeighborhoodLevel.N3),
    )
    f2 = DeferredInnerStep(
        reference_shell=state.search_shell,
        start_values={},
        step=ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N3),
    )

    assert state.deferred_priority(f2) < state.deferred_priority(f1)


def test_unexplored_deferred_process_precedes_a_worse_projected_restart() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        verified_cmax=589.0,
        objective=589.2,
        shell=shell,
    )
    step = ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N3)
    state.observe_inner(
        Procedure.F2,
        candidate_count=0,
        recoverable_count=0,
        timed_out=True,
    )
    state.queues.add_deferred(
        DeferredInnerStep(reference_shell=shell, start_values={}, step=step)
    )
    state.queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="worse-restart"),
            start_values={},
            step=step,
            reserve_retry=True,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=578.0,
            projected_cmax=601.0,
            projected_objective=601.2,
            start_feasible=True,
        )
    )

    assert state.allow_deferred_before_pending()


def test_reserve_stage_policy_alternates_new_structure_and_outer_validation() -> None:
    queues = SearchWorkQueues()
    shell = SimpleNamespace(sha256="incumbent-a")
    queues.add_deferred(
        DeferredInnerStep(
            reference_shell=shell,
            start_values={},
            step=ProcedureStep(7, 3, Procedure.F1, NeighborhoodLevel.N1),
        )
    )
    queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="candidate-a"),
            start_values={},
            step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N2),
            reserve_retry=True,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=580.0,
        )
    )

    assert queues.next_reserve_stage(prefer_deferred=True) is ReserveStage.DEFERRED_INNER
    assert queues.reserve_horizon(ReserveStage.DEFERRED_INNER) == 3
    assert queues.next_reserve_stage(prefer_deferred=False) is ReserveStage.OUTER
    assert queues.reserve_horizon(ReserveStage.OUTER) == 3


def test_unvalidated_outer_gets_a_half_slice_even_with_deferred_work_waiting() -> None:
    queues = SearchWorkQueues()
    incumbent = SimpleNamespace(sha256="incumbent-a")
    step = ProcedureStep(9, 3, Procedure.F2, NeighborhoodLevel.N3)
    queues.add_deferred(
        DeferredInnerStep(reference_shell=incumbent, start_values={}, step=step)
    )
    queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="candidate-a"),
            start_values={},
            step=step,
            reserve_retry=False,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=float("nan"),
        )
    )

    assert queues.reserve_horizon(ReserveStage.OUTER) == 2


def test_outer_retry_keeps_a_third_slice_when_deferred_generation_is_waiting() -> None:
    queues = SearchWorkQueues()
    incumbent = SimpleNamespace(sha256="incumbent-a")
    step = ProcedureStep(9, 3, Procedure.F2, NeighborhoodLevel.N3)
    queues.add_deferred(
        DeferredInnerStep(reference_shell=incumbent, start_values={}, step=step)
    )
    queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="retry-a"),
            start_values={},
            step=step,
            reserve_retry=True,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=580.0,
        )
    )

    assert queues.reserve_horizon(ReserveStage.OUTER) == 3


def test_low_evidence_deferred_work_waits_behind_pending_outer() -> None:
    queues = SearchWorkQueues()
    shell = SimpleNamespace(sha256="incumbent-a")
    queues.add_deferred(
        DeferredInnerStep(
            reference_shell=shell,
            start_values={},
            step=ProcedureStep(7, 3, Procedure.F1, NeighborhoodLevel.N3),
        )
    )
    queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="candidate-a"),
            start_values={},
            step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N2),
            reserve_retry=True,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=580.0,
        )
    )

    assert queues.next_reserve_stage(
        prefer_deferred=True,
        allow_deferred_before_pending=False,
    ) is ReserveStage.OUTER


def test_accepted_refinement_alternates_after_one_deferred_probe() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)
    state.queues.add_deferred(
        DeferredInnerStep(reference_shell=shell, start_values={}, step=step)
    )
    state.queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="accepted-a"),
            start_values={},
            step=step,
            reserve_retry=True,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=578.0,
            accepted_refinement=True,
        )
    )
    for _ in range(4):
        state.observe_inner(Procedure.F3, candidate_count=0, timed_out=True)

    assert state.allow_deferred_before_pending()
    assert state.queues.next_reserve_stage(
        prefer_deferred=True,
        allow_deferred_before_pending=state.allow_deferred_before_pending(),
    ) is ReserveStage.DEFERRED_INNER
    assert state.queues.next_reserve_stage(
        prefer_deferred=False,
        allow_deferred_before_pending=state.allow_deferred_before_pending(),
    ) is ReserveStage.OUTER


def test_resumable_unresolved_retry_precedes_deferred_generation() -> None:
    shell = SimpleNamespace(sha256="incumbent-a")
    state = SearchState(search_shell=shell, start_values={})
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)
    state.queues.add_deferred(
        DeferredInnerStep(reference_shell=shell, start_values={}, step=step)
    )
    state.queues.add_pending(
        PendingOuterShell(
            shell=SimpleNamespace(sha256="unresolved-a"),
            start_values={},
            step=step,
            reserve_retry=True,
            relaxed_objective=579.0,
            repair_risk_total=2.0,
            validation_bound=578.0,
        )
    )

    assert not state.allow_deferred_before_pending()


def test_time_limited_accepted_shell_can_keep_one_certified_refinement() -> None:
    assert has_unresolved_improvement_potential(
        solver_status_code=9,
        objective_bound=579.0,
        accepted_objective=607.2,
    )
    assert not has_unresolved_improvement_potential(
        solver_status_code=2,
        objective_bound=579.0,
        accepted_objective=607.2,
    )
    assert not has_unresolved_improvement_potential(
        solver_status_code=9,
        objective_bound=607.2,
        accepted_objective=607.2,
    )


def test_certified_bound_breaks_a_projection_tie() -> None:
    queues = SearchWorkQueues()
    step = ProcedureStep(15, 5, Procedure.F3, NeighborhoodLevel.N3)
    for shell_hash, reserve_retry in (("retry", True), ("first-attempt", False)):
        queues.add_pending(
            PendingOuterShell(
                shell=SimpleNamespace(sha256=shell_hash),
                start_values={},
                step=step,
                reserve_retry=reserve_retry,
                relaxed_objective=579.0,
                repair_risk_total=2.0,
                validation_bound=578.0 if reserve_retry else float("nan"),
            )
        )

    assert queues.pop_pending().shell.sha256 == "retry"


def test_accepted_refinement_uses_the_same_projection_ranking() -> None:
    queues = SearchWorkQueues()
    step = ProcedureStep(15, 5, Procedure.F2, NeighborhoodLevel.N3)
    for shell_hash, accepted_refinement in (("old-accepted", True), ("unresolved", False)):
        queues.add_pending(
            PendingOuterShell(
                shell=SimpleNamespace(sha256=shell_hash),
                start_values={},
                step=step,
                reserve_retry=True,
                relaxed_objective=579.0,
                repair_risk_total=2.0,
                validation_bound=578.0,
                accepted_refinement=accepted_refinement,
                projected_cmax=589.0 if accepted_refinement else 600.0,
                projected_objective=589.2 if accepted_refinement else 600.2,
                start_feasible=True,
            )
        )

    assert queues.pop_pending().shell.sha256 == "old-accepted"


def test_accepted_shell_restart_refreshes_projection_from_verified_incumbent() -> None:
    shell = SimpleNamespace(sha256="accepted-shell")
    state = SearchState(search_shell=shell, start_values={})
    state.incumbent = SimpleNamespace(
        shell=shell,
        snapshot=SimpleNamespace(
            values_by_name={"verified": 1.0},
            solver_objective=589.2,
        ),
        verified_cmax=589.0,
        objective=589.2,
    )
    candidate = SimpleNamespace(
        shell=shell,
        snapshot=SimpleNamespace(values_by_name={"candidate": 1.0}),
        relaxed_objective=579.2,
        repair_risk=SimpleNamespace(total=20.0),
        comproc=SimpleNamespace(
            feasible=True,
            projected_cmax=1005.0,
            projected_objective=1005.2,
        ),
    )
    result = SimpleNamespace(solver_status_code=9, objective_bound=579.1)
    sequence = object.__new__(ImmediateOuterSequence)
    sequence.audit = SimpleNamespace(queue=lambda *args, **kwargs: None)

    assert sequence._queue_restart(
        candidate=candidate,
        step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3),
        start_values={"candidate": 1.0},
        result=result,
        state=state,
    )
    pending = state.queues.pop_pending()

    assert pending.projected_cmax == 589.0
    assert pending.projected_objective == 589.2
    assert pending.start_values == {"verified": 1.0}


def test_immediate_continuation_unresolved_shell_is_queued_for_reserve_retry() -> None:
    parent_shell = SimpleNamespace(sha256="parent-shell")
    candidate_shell = SimpleNamespace(sha256="candidate-shell")
    state = SearchState(search_shell=parent_shell, start_values={})
    candidate = SimpleNamespace(
        shell=candidate_shell,
        snapshot=SimpleNamespace(values_by_name={"candidate": 1.0}),
        relaxed_objective=579.2,
        repair_risk=SimpleNamespace(total=20.0),
        comproc=SimpleNamespace(
            feasible=True,
            full_start=SimpleNamespace(values_by_name={"start": 1.0}),
            projected_cmax=600.0,
            projected_objective=600.2,
        ),
    )
    unresolved = SimpleNamespace(
        disposition=OuterDisposition.UNRESOLVED,
        accepted=None,
        solver_status_code=9,
        objective_bound=579.1,
        runtime_sec=1.0,
    )
    results = iter((unresolved, unresolved))
    sequence = object.__new__(ImmediateOuterSequence)
    sequence.runtime = SimpleNamespace(
        hard_limit_sec=288.0,
        slice_for=lambda *args, **kwargs: 10.0,
    )
    sequence.audit = SimpleNamespace(queue=lambda *args, **kwargs: None)
    sequence.budget_policy = SimpleNamespace(
        initial_slice=lambda *args, **kwargs: 10.0,
        continuation_slice=lambda *args, **kwargs: 10.0,
        should_continue=lambda *args, **kwargs: True,
    )
    sequence._solve = lambda **kwargs: next(results)

    outcome = sequence.run(
        candidate,
        step=ProcedureStep(11, 4, Procedure.F2, NeighborhoodLevel.N1),
        state=state,
        suggested_initial_sec=10.0,
        continuation_horizon=1,
    )

    assert outcome.continuation_attempted
    assert outcome.restart_queued
    pending = state.queues.pop_pending()
    assert pending.shell.sha256 == "candidate-shell"
    assert pending.reserve_retry is True


def test_immediate_continuation_accepted_shell_with_promising_bound_is_queued() -> None:
    parent_shell = SimpleNamespace(sha256="parent-shell")
    candidate_shell = SimpleNamespace(sha256="candidate-shell")
    state = SearchState(search_shell=parent_shell, start_values={})
    candidate = SimpleNamespace(
        shell=candidate_shell,
        snapshot=SimpleNamespace(values_by_name={"candidate": 1.0}),
        relaxed_objective=579.2,
        repair_risk=SimpleNamespace(total=20.0),
        comproc=SimpleNamespace(
            feasible=True,
            full_start=SimpleNamespace(values_by_name={"start": 1.0}),
            projected_cmax=600.0,
            projected_objective=600.2,
        ),
    )
    accepted = SimpleNamespace(
        shell=candidate_shell,
        snapshot=SimpleNamespace(
            values_by_name={"accepted": 1.0},
            solver_objective=589.2,
        ),
        verified_cmax=589.0,
    )
    result = SimpleNamespace(
        disposition=OuterDisposition.ACCEPTED,
        accepted=accepted,
        solver_status_code=9,
        objective_bound=579.1,
        runtime_sec=1.0,
    )
    results = iter((result, result))
    sequence = object.__new__(ImmediateOuterSequence)
    sequence.runtime = SimpleNamespace(
        hard_limit_sec=288.0,
        slice_for=lambda *args, **kwargs: 10.0,
    )
    sequence.audit = SimpleNamespace(queue=lambda *args, **kwargs: None)
    sequence.budget_policy = SimpleNamespace(
        initial_slice=lambda *args, **kwargs: 10.0,
        continuation_slice=lambda *args, **kwargs: 10.0,
        should_continue=lambda *args, **kwargs: True,
    )
    sequence._solve = lambda **kwargs: next(results)
    sequence._accept = lambda *args, **kwargs: None

    outcome = sequence.run(
        candidate,
        step=ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3),
        state=state,
        suggested_initial_sec=10.0,
        continuation_horizon=1,
    )

    assert outcome.continuation_attempted
    assert outcome.restart_queued
    assert state.queues.pop_pending().reserve_retry is True
