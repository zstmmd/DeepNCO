from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from gurobipy import GRB

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_budget_policy import OuterBudgetPolicy
from Gurobi.tra_outer import OuterDisposition, OuterSolveResult, objective_tolerance
from Gurobi.tra_scheduler import ProcedureStep, RuntimeLedger
from Gurobi.tra_search_state import AcceptanceOutcome, SearchState
from Gurobi.tra_templates import PaperTRATemplates
from Gurobi.tra_verifier import VerifiedSnapshot
from Gurobi.tra_work_queue import PendingOuterShell


VerifiedRecorder = Callable[[VerifiedSnapshot, ProcedureStep, float, str, bool], None]


@dataclass(frozen=True)
class OuterSequenceOutcome:
    structural_improvement: bool
    cmax_improvement: bool
    continuation_attempted: bool
    restart_queued: bool
    hard_failure: bool


class ImmediateOuterSequence:
    """Certify one shell and immediately continue its untouched search once."""

    def __init__(
        self,
        templates: PaperTRATemplates,
        runtime: RuntimeLedger,
        audit: SearchAuditTrail,
        record_verified: VerifiedRecorder,
        *,
        budget_policy: OuterBudgetPolicy | None = None,
    ) -> None:
        self.templates = templates
        self.runtime = runtime
        self.audit = audit
        self.record_verified = record_verified
        self.budget_policy = budget_policy or OuterBudgetPolicy()

    def _solve(
        self,
        *,
        candidate: Any,
        step: ProcedureStep,
        start_values: dict[str, float],
        time_limit_sec: float,
        state: SearchState,
        resume: bool,
    ) -> OuterSolveResult:
        formal_start = self.runtime.elapsed_sec

        def sink(verified: VerifiedSnapshot, solver_timestamp: float) -> None:
            provenance = "outer_continuation_mipsol" if resume else "outer_mipsol"
            self.record_verified(verified, step, solver_timestamp, provenance, False)

        result = self.templates.outer.solve_shell(
            candidate.shell,
            time_limit_sec=time_limit_sec,
            incumbent_objective=state.incumbent_objective,
            start_values=start_values,
            formal_elapsed_at_start=formal_start,
            verified_sink=sink,
            reserve_retry=False,
            resume_if_available=resume,
            incumbent_cmax=state.certification_cmax_limit(step),
        )
        self.runtime.record("outer", result.runtime_sec)
        self.audit.outer(
            step,
            result,
            submitted_shell_sha256=candidate.shell.sha256,
            reserve_retry=False,
            requested_time_limit_sec=time_limit_sec,
            stage="outer_continuation" if resume else "outer",
        )
        return result

    @staticmethod
    def _accept(
        state: SearchState,
        result: OuterSolveResult,
        step: ProcedureStep,
    ) -> AcceptanceOutcome | None:
        if result.disposition is not OuterDisposition.ACCEPTED or result.accepted is None:
            return None
        return state.accept(result.accepted, step=step)

    def _queue_restart(
        self,
        *,
        candidate: Any,
        step: ProcedureStep,
        start_values: dict[str, float],
        result: OuterSolveResult,
        state: SearchState,
    ) -> bool:
        if int(result.solver_status_code) != GRB.TIME_LIMIT:
            return False
        incumbent_objective = state.incumbent_objective
        if (
            state.incumbent_cmax is None
            and incumbent_objective is not None
            and math.isfinite(float(result.objective_bound))
            and float(result.objective_bound)
            >= float(incumbent_objective) - objective_tolerance(incumbent_objective)
        ):
            return False
        if not state.retry_registry.register_unresolved(candidate.shell.sha256):
            return False
        restart_values = dict(start_values)
        accepted_refinement = False
        projected_cmax = float(
            candidate.comproc.projected_cmax
            if candidate.comproc is not None
            else float("inf")
        )
        projected_objective = float(
            candidate.comproc.projected_objective
            if candidate.comproc is not None
            else float("inf")
        )
        start_feasible = bool(
            candidate.comproc is not None
            and candidate.comproc.feasible
        )
        active_search_incumbent = state.search_incumbent
        if (
            active_search_incumbent is None
            and state.incumbent is not None
            and state.incumbent.shell.sha256 == state.search_shell.sha256
        ):
            active_search_incumbent = state.incumbent
        if (
            active_search_incumbent is not None
            and active_search_incumbent.shell.sha256 == candidate.shell.sha256
        ):
            restart_values = dict(active_search_incumbent.snapshot.values_by_name)
            accepted_refinement = True
            projected_cmax = float(active_search_incumbent.verified_cmax)
            projected_objective = float(active_search_incumbent.objective)
            start_feasible = True
        queued = state.queues.add_pending(
            PendingOuterShell(
                shell=candidate.shell,
                start_values=restart_values,
                step=step,
                reserve_retry=True,
                relaxed_objective=float(candidate.relaxed_objective),
                repair_risk_total=float(candidate.repair_risk.total),
                validation_bound=float(result.objective_bound),
                accepted_refinement=accepted_refinement,
                projected_cmax=projected_cmax,
                projected_objective=projected_objective,
                start_feasible=start_feasible,
            )
        )
        if queued:
            self.audit.queue(
                step,
                queue_name="pending_outer",
                reason="restart_after_immediate_continuation",
                shell_sha256=candidate.shell.sha256,
            )
        return bool(queued)

    def run(
        self,
        candidate: Any,
        *,
        step: ProcedureStep,
        state: SearchState,
        suggested_initial_sec: float,
        continuation_horizon: int,
    ) -> OuterSequenceOutcome:
        candidate_start_values = dict(
            candidate.comproc.full_start.values_by_name
            if candidate.comproc is not None and candidate.comproc.full_start is not None
            else candidate.snapshot.values_by_name
        )
        initial_slice = self.budget_policy.initial_slice(
            suggested_initial_sec,
            hard_limit_sec=self.runtime.hard_limit_sec,
        )
        if initial_slice <= 1e-3:
            return OuterSequenceOutcome(False, False, False, False, False)

        result = self._solve(
            candidate=candidate,
            step=step,
            start_values=candidate_start_values,
            time_limit_sec=initial_slice,
            state=state,
            resume=False,
        )
        acceptance = self._accept(state, result, step)
        structural_improvement = bool(
            acceptance is not None and acceptance.structural_change
        )
        cmax_improvement = bool(
            acceptance is not None and acceptance.cmax_improved
        )
        if result.disposition is OuterDisposition.HARD_FAILURE:
            state.error = str(result.error)
            state.status = "ENGINE_FAILED"
            return OuterSequenceOutcome(
                structural_improvement,
                cmax_improvement,
                False,
                False,
                True,
            )

        incumbent_cmax = None if state.incumbent is None else state.incumbent.verified_cmax
        projected_cmax = (
            float(candidate.comproc.projected_cmax)
            if candidate.comproc is not None
            else float(result.projected_start_cmax)
        )
        continuation_attempted = False
        if self.budget_policy.should_continue(
            result,
            incumbent_objective=state.incumbent_objective,
            incumbent_cmax=incumbent_cmax,
            projected_cmax=projected_cmax,
        ):
            suggested = self.runtime.slice_for(
                "outer",
                max(1, int(continuation_horizon)),
            )
            continuation_slice = self.budget_policy.continuation_slice(
                suggested,
                hard_limit_sec=self.runtime.hard_limit_sec,
            )
            if continuation_slice > 1e-3:
                continuation_attempted = True
                result = self._solve(
                    candidate=candidate,
                    step=step,
                    start_values=candidate_start_values,
                    time_limit_sec=continuation_slice,
                    state=state,
                    resume=True,
                )
                continued_acceptance = self._accept(state, result, step)
                structural_improvement = bool(
                    structural_improvement
                    or (
                        continued_acceptance is not None
                        and continued_acceptance.structural_change
                    )
                )
                cmax_improvement = bool(
                    cmax_improvement
                    or (
                        continued_acceptance is not None
                        and continued_acceptance.cmax_improved
                    )
                )
                if result.disposition is OuterDisposition.HARD_FAILURE:
                    state.error = str(result.error)
                    state.status = "ENGINE_FAILED"
                    return OuterSequenceOutcome(
                        structural_improvement,
                        cmax_improvement,
                        continuation_attempted,
                        False,
                        True,
                    )

        restart_queued = False
        if not continuation_attempted:
            restart_queued = self._queue_restart(
                candidate=candidate,
                step=step,
                start_values=candidate_start_values,
                result=result,
                state=state,
            )
        return OuterSequenceOutcome(
            structural_improvement=structural_improvement,
            cmax_improvement=cmax_improvement,
            continuation_attempted=continuation_attempted,
            restart_queued=restart_queued,
            hard_failure=False,
        )
