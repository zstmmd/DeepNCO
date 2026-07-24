from __future__ import annotations

from dataclasses import replace
from typing import Callable

from gurobipy import GRB

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_budget_policy import (
    RegularInnerBudgetPolicy,
    f3_support_expansion_needed,
)
from Gurobi.tra_candidate_archive import released_block_distance
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer_sequence import ImmediateOuterSequence
from Gurobi.tra_scheduler import ProcedureStep, RotationScheduler, RuntimeLedger
from Gurobi.tra_search_state import SearchState
from Gurobi.tra_templates import PaperTRATemplates
from Gurobi.tra_verifier import VerifiedSnapshot
from Gurobi.tra_vns import rotating_search_seed
from Gurobi.tra_work_queue import DeferredInnerStep


VerifiedRecorder = Callable[[VerifiedSnapshot, ProcedureStep, float, str, bool], None]


class RegularRotationPhase:
    """Run strict F1/F2/F3 steps until stagnation, budget, or procedure limit."""

    def __init__(
        self,
        templates: PaperTRATemplates,
        runtime: RuntimeLedger,
        scheduler: RotationScheduler,
        audit: SearchAuditTrail,
        record_verified: VerifiedRecorder,
        inner_budget_policy: RegularInnerBudgetPolicy | None = None,
    ) -> None:
        self.templates = templates
        self.runtime = runtime
        self.scheduler = scheduler
        self.audit = audit
        self.record_verified = record_verified
        self.inner_budget_policy = inner_budget_policy or RegularInnerBudgetPolicy()
        self.outer_sequence = ImmediateOuterSequence(
            templates,
            runtime,
            audit,
            record_verified,
        )

    def run(self, state: SearchState) -> None:
        while not self.scheduler.should_stop(
            runtime_remaining_sec=self.runtime.allocatable_remaining_sec,
            deferred_empty=(
                state.queues.empty and not state.has_compatible_archive
            ),
        ):
            if self.scheduler.stagnant_cycles >= self.scheduler.stagnant_cycle_limit and not state.queues.empty:
                return
            step = self.scheduler.current_step()
            horizon = min(9, max(1, self.scheduler.remaining_regular_steps))
            effort_multiplier = state.inner_effort_multiplier(step.procedure)
            inner_slice = self.runtime.slice_for("inner", horizon) * effort_multiplier
            robot_labels = {
                int(robot_id)
                for _slot_id, robot_id in (
                    self.templates.inner.template.payload.get("slot_robot", {}) or {}
                )
            }
            f3_support_pressure = bool(
                state.incumbent_cmax is not None
                and f3_support_expansion_needed(
                    state.search_shell.projection.r_assign,
                    robot_labels,
                )
            )
            inner_slice = self.inner_budget_policy.stabilize_slice(
                inner_slice,
                hard_limit_sec=self.runtime.hard_limit_sec,
                allocatable_remaining_sec=self.runtime.allocatable_remaining_sec,
                f3_n1_support_expansion=bool(
                    f3_support_pressure
                    and step.procedure is Procedure.F3
                    and step.neighborhood is NeighborhoodLevel.N1
                ),
                cross_process_f3_n2=bool(
                    step.procedure is Procedure.F3
                    and step.neighborhood is NeighborhoodLevel.N2
                    and state.consecutive_transition_procedure is not Procedure.F3
                ),
                f3_n3_balance=bool(
                    f3_support_pressure
                    and step.procedure is Procedure.F3
                    and step.neighborhood is NeighborhoodLevel.N3
                ),
            )
            if inner_slice <= 1e-3:
                return
            vns_offset = state.next_vns_offset(
                state.search_shell,
                step.procedure,
                step.neighborhood,
            )
            vns_seeds = self.templates.vns.generate(
                state.search_shell,
                procedure=step.procedure,
                neighborhood=step.neighborhood,
                offset=vns_offset,
                balance_support=state.incumbent_cmax is not None,
            )
            base_seed = int(
                getattr(
                    self.templates.inner.template.compiled.cfg,
                    "gurobi_seed",
                    0,
                )
                or 0
            )
            inner_result = self.templates.inner.solve(
                state.search_shell,
                procedure=step.procedure,
                neighborhood=step.neighborhood,
                time_limit_sec=inner_slice,
                incumbent_objective=state.incumbent_objective,
                start_values=state.start_values,
                search_seed=rotating_search_seed(
                    base_seed,
                    offset=vns_offset,
                ),
                vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
                vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
                incumbent_cmax=state.incumbent_cmax,
            )
            self.runtime.record("inner", inner_result.runtime_sec)
            inner_result = replace(
                inner_result,
                candidates=self.templates.comproc.evaluate_many(inner_result.candidates),
            )
            state.observe_inner(
                step.procedure,
                candidate_count=len(inner_result.candidates),
                recoverable_count=sum(
                    int(
                        candidate.comproc is not None
                        and candidate.comproc.feasible
                    )
                    for candidate in inner_result.candidates
                ),
                timed_out=inner_result.solver_status_code == GRB.TIME_LIMIT,
            )
            transitioned = False
            cmax_improved = False
            certified_prune = state.certified_prune(inner_result.certified_obj_bound)
            candidate = None
            submission_step = step
            if not certified_prune:
                candidate = state.select_unattempted_candidate(
                    state.search_shell,
                    step,
                    inner_result.candidates,
                )
                state.candidate_archive.remember(
                    state.search_shell,
                    step,
                    (
                        item
                        for item in inner_result.candidates
                        if state.candidate_within_certification_band(step, item)
                    ),
                    excluded_hashes=(
                        () if candidate is None else (candidate.shell.sha256,)
                    ),
                )
                if candidate is None:
                    for archived in state.ranked_archive(step.procedure):
                        if (
                            released_block_distance(
                                step.procedure,
                                state.search_shell,
                                archived.candidate.shell,
                            )
                            <= 0
                        ):
                            state.candidate_archive.discard(
                                step.procedure,
                                archived.candidate.shell.sha256,
                            )
                            continue
                        archived_step = ProcedureStep(
                            procedure_index=step.procedure_index,
                            cycle=step.cycle,
                            procedure=step.procedure,
                            neighborhood=archived.step.neighborhood,
                        )
                        candidate = state.select_unattempted_candidate(
                            archived.reference_shell,
                            archived_step,
                            (archived.candidate,),
                            allow_diverse_neighborhood_repeat=(
                                state.allow_archive_neighborhood_repeat(
                                    archived_step
                                )
                            ),
                        )
                        if candidate is None:
                            continue
                        submission_step = archived_step
                        state.candidate_archive.discard(
                            step.procedure,
                            candidate.shell.sha256,
                        )
                        self.audit.queue(
                            submission_step,
                            queue_name="candidate_archive",
                            reason="diverse_runner_up_submission",
                            shell_sha256=candidate.shell.sha256,
                        )
                        break
            self.audit.inner(
                step,
                inner_result,
                incumbent_objective=state.incumbent_objective,
                certified_prune=certified_prune,
                selected_shell_sha256=None if candidate is None else candidate.shell.sha256,
                requested_time_limit_sec=inner_slice,
                effort_multiplier=effort_multiplier,
                recourse_calibration_allowance_sec=(
                    state.recourse_calibration.allowance(step.procedure)
                ),
            )

            if candidate is not None and self.runtime.allocatable_remaining_sec > 1e-3:
                suggested_outer_slice = self.runtime.slice_for(
                    "outer",
                    state.estimated_outer_horizon(horizon),
                )
                if suggested_outer_slice > 1e-3:
                    outcome = self.outer_sequence.run(
                        candidate,
                        step=submission_step,
                        state=state,
                        suggested_initial_sec=suggested_outer_slice,
                        continuation_horizon=state.estimated_outer_horizon(horizon),
                    )
                    transitioned = bool(outcome.structural_improvement)
                    cmax_improved = bool(outcome.cmax_improvement)
                    if outcome.hard_failure:
                        return

            if (
                not transitioned
                and not certified_prune
                and not inner_result.candidates
                and inner_result.solver_status_code == GRB.TIME_LIMIT
            ):
                queued = state.queues.add_deferred(
                    DeferredInnerStep(
                        reference_shell=state.search_shell,
                        start_values=dict(state.start_values),
                        step=step,
                    )
                )
                if queued:
                    self.audit.queue(
                        step,
                        queue_name="deferred_inner",
                        reason="inner_time_limit_without_candidate",
                    )
            self.scheduler.complete_step(
                improved=transitioned,
                primary_improved=cmax_improved,
            )
