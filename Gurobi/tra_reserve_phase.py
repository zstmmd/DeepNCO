from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

from gurobipy import GRB

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_budget_policy import OuterBudgetPolicy, ReserveBudgetPolicy
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_refinement import retain_accepted_shell_refinement
from Gurobi.tra_regular_phase import VerifiedRecorder
from Gurobi.tra_scheduler import ProcedureStep, RotationScheduler, RuntimeLedger
from Gurobi.tra_search_state import AcceptanceOutcome, SearchState
from Gurobi.tra_templates import PaperTRATemplates
from Gurobi.tra_verifier import VerifiedSnapshot
from Gurobi.tra_vns import rotating_search_seed
from Gurobi.tra_work_queue import PendingOuterShell, ReserveStage


class ReservePhase:
    """Alternate deferred N3 generation with one-shot unresolved outer retries."""

    def __init__(
        self,
        templates: PaperTRATemplates,
        runtime: RuntimeLedger,
        scheduler: RotationScheduler,
        audit: SearchAuditTrail,
        record_verified: VerifiedRecorder,
        budget_policy: Optional[ReserveBudgetPolicy] = None,
        enable_f1_plateau_escalation: bool = False,
        plateau_escalation_procedures: Iterable[Procedure] = (),
    ) -> None:
        self.templates = templates
        self.runtime = runtime
        self.scheduler = scheduler
        self.audit = audit
        self.record_verified = record_verified
        self.budget_policy = budget_policy or ReserveBudgetPolicy()
        self.enable_f1_plateau_escalation = bool(enable_f1_plateau_escalation)
        self.plateau_escalation_procedures = {
            Procedure(procedure)
            for procedure in plateau_escalation_procedures
        }
        if self.enable_f1_plateau_escalation:
            self.plateau_escalation_procedures.add(Procedure.F1)

    def run(self, state: SearchState) -> bool:
        prefer_deferred = True
        while self.runtime.allocatable_remaining_sec > 1e-3 and not state.queues.empty:
            allow_deferred = state.allow_deferred_before_pending()
            reserve_stage = state.queues.next_reserve_stage(
                prefer_deferred=prefer_deferred,
                allow_deferred_before_pending=allow_deferred,
            )
            reserve_slice = self.runtime.slice_for(
                "reserve",
                state.queues.reserve_horizon(reserve_stage),
                borrow_unused=True,
            )
            if reserve_stage is ReserveStage.OUTER:
                pending = state.queues.peek_pending()
                bound_promoted = bool(
                    pending.reserve_retry
                    and OuterBudgetPolicy.retry_is_bound_promoted(
                        objective_bound=pending.validation_bound,
                        incumbent_objective=state.incumbent_objective,
                    )
                )
                reserve_slice = self.budget_policy.cap_outer_slice(
                    reserve_slice,
                    hard_limit_sec=self.runtime.hard_limit_sec,
                    reserve_retry=pending.reserve_retry,
                    bound_promoted=bound_promoted,
                )
            if reserve_slice <= 1e-3:
                return False
            if state.queues.pending_count and state.queues.deferred_count and allow_deferred:
                prefer_deferred = reserve_stage is ReserveStage.OUTER
            if reserve_stage is ReserveStage.OUTER:
                acceptance = self._run_outer(
                    state,
                    reserve_slice,
                    bound_promoted=bound_promoted,
                )
                if acceptance is not None and acceptance.structural_change:
                    preserve_f1 = bool(
                        self.enable_f1_plateau_escalation
                        and not acceptance.cmax_improved
                    )
                    extra_preserved = (
                        tuple(
                            sorted(
                                (
                                    procedure
                                    for procedure in self.plateau_escalation_procedures
                                    if procedure is not Procedure.F1
                                ),
                                key=lambda item: item.value,
                            )
                        )
                        if not acceptance.cmax_improved
                        else ()
                    )
                    kwargs = {"preserve_f1_level": preserve_f1}
                    if extra_preserved:
                        kwargs["preserve_procedure_levels"] = extra_preserved
                    self.scheduler.restart_after_external_improvement(**kwargs)
                    return True
                if state.status == "ENGINE_FAILED":
                    return False
            else:
                self._run_deferred_inner(state, reserve_slice)
        return False

    def _run_outer(
        self,
        state: SearchState,
        time_limit_sec: float,
        *,
        bound_promoted: bool = False,
    ) -> Optional[AcceptanceOutcome]:
        item = state.queues.pop_pending()
        if item.reserve_retry:
            if not state.retry_registry.can_retry(item.shell.sha256):
                return None
            state.retry_registry.mark_retried(item.shell.sha256)

        def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
            self.record_verified(
                verified,
                item.step,
                solver_timestamp,
                "outer_mipsol",
                item.reserve_retry,
            )

        result = self.templates.outer.solve_shell(
            item.shell,
            time_limit_sec=time_limit_sec,
            incumbent_objective=state.incumbent_objective,
            start_values=item.start_values,
            formal_elapsed_at_start=self.runtime.elapsed_sec,
            verified_sink=sink,
            reserve_retry=item.reserve_retry,
            # Queue order may have changed the persistent outer shell. Delayed
            # work is therefore always an explicit restart, never a resume.
            resume_if_available=False,
            incumbent_cmax=state.certification_cmax_limit(item.step),
        )
        self.runtime.record("reserve", result.runtime_sec)
        self.audit.outer(
            item.step,
            result,
            submitted_shell_sha256=item.shell.sha256,
            reserve_retry=item.reserve_retry,
            requested_time_limit_sec=time_limit_sec,
            budget_mode=(
                "reserve_bound_promoted"
                if bound_promoted
                else "reserve_standard"
            ),
            stage="reserve_outer",
        )
        if result.disposition is OuterDisposition.ACCEPTED and result.accepted is not None:
            acceptance = state.accept(result.accepted, step=item.step)
            if retain_accepted_shell_refinement(
                state,
                result,
                step=item.step,
                relaxed_objective=item.relaxed_objective,
                repair_risk_total=item.repair_risk_total,
                allow_retry=not item.reserve_retry,
            ):
                self.audit.queue(
                    item.step,
                    queue_name="pending_outer",
                    reason="accepted_shell_refinement",
                    shell_sha256=result.accepted.shell.sha256,
                )
            return acceptance
        if result.disposition is OuterDisposition.UNRESOLVED and not item.reserve_retry:
            if state.retry_registry.register_unresolved(item.shell.sha256):
                state.queues.add_pending(
                    PendingOuterShell(
                        shell=item.shell,
                        start_values=dict(item.start_values),
                        step=item.step,
                        reserve_retry=True,
                        relaxed_objective=float(item.relaxed_objective),
                        repair_risk_total=float(item.repair_risk_total),
                        validation_bound=float(result.objective_bound),
                        projected_cmax=float(item.projected_cmax),
                        projected_objective=float(item.projected_objective),
                        start_feasible=bool(item.start_feasible),
                    )
                )
                self.audit.queue(
                    item.step,
                    queue_name="pending_outer",
                    reason="unresolved_restart",
                    shell_sha256=item.shell.sha256,
                )
        elif result.disposition is OuterDisposition.HARD_FAILURE:
            state.error = str(result.error)
            state.status = "ENGINE_FAILED"
        return None

    def _run_deferred_inner(self, state: SearchState, time_limit_sec: float) -> None:
        item = state.queues.pop_deferred(priority=state.deferred_priority)
        time_limit_sec = self.budget_policy.cap_deferred_inner(
            time_limit_sec,
            hard_limit_sec=self.runtime.hard_limit_sec,
            procedure=item.step.procedure,
            neighborhood=item.step.neighborhood,
        )
        reserve_step = ProcedureStep(
            procedure_index=item.step.procedure_index,
            cycle=item.step.cycle,
            procedure=item.step.procedure,
            neighborhood=item.step.neighborhood,
        )
        vns_offset = state.next_vns_offset(
            item.reference_shell,
            item.step.procedure,
            reserve_step.neighborhood,
        )
        vns_seeds = self.templates.vns.generate(
            item.reference_shell,
            procedure=item.step.procedure,
            neighborhood=reserve_step.neighborhood,
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
        inner = self.templates.inner.solve(
            item.reference_shell,
            procedure=item.step.procedure,
            neighborhood=reserve_step.neighborhood,
            time_limit_sec=time_limit_sec,
            incumbent_objective=state.incumbent_objective,
            start_values=item.start_values,
            search_seed=rotating_search_seed(
                base_seed,
                offset=vns_offset,
            ),
            vns_start_values=tuple(seed.values_by_name for seed in vns_seeds),
            vns_seed_sha256=tuple(seed.sha256 for seed in vns_seeds),
            incumbent_cmax=state.incumbent_cmax,
        )
        self.runtime.record("reserve", inner.runtime_sec)
        inner = replace(
            inner,
            candidates=self.templates.comproc.evaluate_many(
                inner.candidates,
                source="deferred_inner_mipsol",
            ),
        )
        state.observe_inner(
            item.step.procedure,
            candidate_count=len(inner.candidates),
            recoverable_count=sum(
                int(candidate.comproc is not None and candidate.comproc.feasible)
                for candidate in inner.candidates
            ),
            timed_out=inner.solver_status_code == GRB.TIME_LIMIT,
        )
        certified_prune = state.certified_prune(inner.certified_obj_bound)
        candidate = None
        submission_step = reserve_step
        archived_submission = False
        selection_dispositions: list[dict[str, str]] = []
        if not certified_prune:
            selection = state.select_unattempted_candidate_with_dispositions(
                item.reference_shell,
                reserve_step,
                inner.candidates,
            )
            candidate = selection.candidate
            selection_dispositions.extend(selection.dispositions)
            state.candidate_archive.remember(
                item.reference_shell,
                reserve_step,
                (
                    inner_candidate
                    for inner_candidate in inner.candidates
                    if state.candidate_within_certification_band(
                        reserve_step,
                        inner_candidate,
                    )
                ),
                excluded_hashes=(
                    () if candidate is None else (candidate.shell.sha256,)
                ),
            )
            if candidate is None and not inner.candidates:
                for archived in state.ranked_archive(reserve_step.procedure):
                    archived_step = ProcedureStep(
                        procedure_index=reserve_step.procedure_index,
                        cycle=reserve_step.cycle,
                        procedure=reserve_step.procedure,
                        neighborhood=archived.step.neighborhood,
                    )
                    archive_selection = state.select_unattempted_candidate_with_dispositions(
                        archived.reference_shell,
                        archived_step,
                        (archived.candidate,),
                        allow_diverse_neighborhood_repeat=(
                            state.allow_archive_neighborhood_repeat(
                                archived_step
                            )
                        ),
                    )
                    candidate = archive_selection.candidate
                    selection_dispositions.extend(
                        archive_selection.dispositions
                    )
                    if candidate is None:
                        continue
                    submission_step = archived_step
                    archived_submission = True
                    state.candidate_archive.discard(
                        reserve_step.procedure,
                        candidate.shell.sha256,
                    )
                    break
        self.audit.inner(
            reserve_step,
            inner,
            incumbent_objective=state.incumbent_objective,
            certified_prune=certified_prune,
            selected_shell_sha256=None if candidate is None else candidate.shell.sha256,
            selection_dispositions=tuple(selection_dispositions),
            requested_time_limit_sec=time_limit_sec,
            effort_multiplier=1.0,
            recourse_calibration_allowance_sec=(
                state.recourse_calibration.allowance(reserve_step.procedure)
            ),
            stage="reserve_inner",
        )
        if candidate is None:
            return
        if archived_submission:
            self.audit.queue(
                submission_step,
                queue_name="candidate_archive",
                reason="compatible_after_deferred_exhaustion",
                shell_sha256=candidate.shell.sha256,
            )
        candidate_start_values = (
            candidate.comproc.full_start.values_by_name
            if candidate.comproc is not None
            and candidate.comproc.full_start is not None
            else candidate.snapshot.values_by_name
        )
        state.queues.add_pending(
            PendingOuterShell(
                shell=candidate.shell,
                start_values=dict(candidate_start_values),
                step=submission_step,
                reserve_retry=False,
                relaxed_objective=float(candidate.relaxed_objective),
                repair_risk_total=float(candidate.repair_risk.total),
                validation_bound=float("nan"),
                projected_cmax=float(
                    candidate.comproc.projected_cmax
                    if candidate.comproc is not None
                    else float("inf")
                ),
                projected_objective=float(
                    candidate.comproc.projected_objective
                    if candidate.comproc is not None
                    else float("inf")
                ),
                start_feasible=bool(
                    candidate.comproc is not None
                    and candidate.comproc.feasible
                ),
            )
        )
        self.audit.queue(
            submission_step,
            queue_name="pending_outer",
            reason="deferred_inner_candidate",
            shell_sha256=candidate.shell.sha256,
        )
