from __future__ import annotations

import uuid
from typing import Optional

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_budget_policy import OuterBudgetPolicy
from Gurobi.tra_events import EventLedger, SearchAuditLedger
from Gurobi.tra_fast_events import TRAFastEngineResult, build_fast_event
from Gurobi.tra_fast_search import (
    FastNeighborhoodResult,
    PaperFastNeighborhoodTemplate,
)
from Gurobi.tra_initial import build_canonical_initial_state
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_scheduler import ProcedureStep, RotationScheduler, RuntimeLedger
from Gurobi.tra_search_state import SearchState
from Gurobi.tra_templates import PaperTRATemplates
from Gurobi.tra_verifier import VerifiedSnapshot

class PaperTRAFastEngine:
    """Strict F1/F2/F3 full-model VNS without the no-wait inner MIP."""

    def __init__(
        self,
        templates: PaperTRATemplates,
        fast: PaperFastNeighborhoodTemplate,
        runtime: RuntimeLedger,
        *,
        max_procedures: int = 50,
    ) -> None:
        self.templates = templates
        self.fast = fast
        self.runtime = runtime
        self.scheduler = RotationScheduler(
            max_procedures=max_procedures,
            stagnant_cycle_limit=3,
        )
        self.budget_policy = OuterBudgetPolicy()

    def _solve_step(
        self,
        *,
        state: SearchState,
        step: ProcedureStep,
        time_limit_sec: float,
        audit: SearchAuditTrail,
        record,
        bucket: str,
        resume: bool,
        vns_offset: int,
    ) -> FastNeighborhoodResult:
        seeds = self.templates.vns.generate(
            state.search_shell,
            procedure=step.procedure,
            neighborhood=step.neighborhood,
            offset=vns_offset,
            balance_support=state.incumbent_cmax is not None,
        )
        formal_start = self.runtime.elapsed_sec

        def sink(verified: VerifiedSnapshot, timestamp: float) -> None:
            record(
                verified,
                step,
                timestamp,
                "fast_continuation_mipsol" if resume else "fast_mipsol",
            )

        result = self.fast.solve(
            state.search_shell,
            procedure=step.procedure,
            neighborhood=step.neighborhood,
            time_limit_sec=time_limit_sec,
            incumbent_objective=state.incumbent_objective,
            start_values=state.start_values,
            vns_start_values=tuple(seed.values_by_name for seed in seeds),
            formal_elapsed_at_start=formal_start,
            verified_sink=sink,
            resume_if_available=resume,
            incumbent_cmax=state.certification_cmax_limit(step),
        )
        self.runtime.record(bucket, result.runtime_sec)
        audit.outer(
            step,
            result,
            submitted_shell_sha256=state.search_shell.sha256,
            reserve_retry=bucket == "reserve",
            requested_time_limit_sec=time_limit_sec,
            stage=(
                "fast_reserve"
                if bucket == "reserve"
                else ("fast_continuation" if resume else "fast_neighborhood")
            ),
        )
        return result

    def run(
        self,
        *,
        case: str,
        ledger: EventLedger,
        audit_ledger: Optional[SearchAuditLedger] = None,
        run_id: Optional[str] = None,
    ) -> TRAFastEngineResult:
        run_id = str(run_id or uuid.uuid4().hex)
        initial = build_canonical_initial_state(
            self.fast.template,
            self.fast.verifier,
        )
        state = SearchState(
            search_shell=initial.search_shell,
            start_values=dict(initial.start_values),
        )
        event_count = 0
        audit = SearchAuditTrail(
            audit_ledger,
            run_id=run_id,
            case=case,
            elapsed_sec=lambda: self.runtime.elapsed_sec,
        )

        def record(
            verified: VerifiedSnapshot,
            step: Optional[ProcedureStep],
            timestamp: float,
            source: str,
        ) -> None:
            nonlocal event_count
            ledger.append(
                build_fast_event(
                    manifest=self.templates.manifest,
                    run_id=run_id,
                    case=case,
                    elapsed_sec=self.runtime.elapsed_sec,
                    verified=verified,
                    step=step,
                    solver_timestamp_sec=timestamp,
                    source=source,
                )
            )
            event_count += 1

        if initial.verified_incumbent is not None:
            state.accept(initial.verified_incumbent)
            record(initial.verified_incumbent, None, 0.0, "canonical_warm_start")

        self.runtime.start()
        try:
            while not self.scheduler.should_stop(
                runtime_remaining_sec=self.runtime.allocatable_remaining_sec,
                deferred_empty=True,
            ):
                step = self.scheduler.current_step()
                horizon = min(9, max(1, self.scheduler.remaining_regular_steps))
                suggested = self.runtime.slice_for("outer", horizon)
                time_slice = self.budget_policy.initial_slice(
                    suggested,
                    hard_limit_sec=self.runtime.hard_limit_sec,
                )
                if time_slice <= 1e-3:
                    break
                vns_offset = state.next_vns_offset(
                    state.search_shell,
                    step.procedure,
                    step.neighborhood,
                )
                result = self._solve_step(
                    state=state,
                    step=step,
                    time_limit_sec=time_slice,
                    audit=audit,
                    record=record,
                    bucket="outer",
                    resume=False,
                    vns_offset=vns_offset,
                )
                accepted = result.accepted
                if result.disposition is OuterDisposition.HARD_FAILURE:
                    state.error = result.error
                    state.status = "ENGINE_FAILED"
                    break
                if self.budget_policy.should_continue(
                    result,
                    incumbent_objective=state.incumbent_objective,
                    incumbent_cmax=(
                        None
                        if state.incumbent is None
                        else state.incumbent.verified_cmax
                    ),
                    projected_cmax=(
                        float("inf")
                        if result.accepted is None
                        else result.accepted.verified_cmax
                    ),
                ):
                    continuation_slice = self.budget_policy.continuation_slice(
                        self.runtime.slice_for("outer", horizon),
                        hard_limit_sec=self.runtime.hard_limit_sec,
                    )
                    if continuation_slice > 1e-3:
                        continued = self._solve_step(
                            state=state,
                            step=step,
                            time_limit_sec=continuation_slice,
                            audit=audit,
                            record=record,
                            bucket="outer",
                            resume=True,
                            vns_offset=vns_offset,
                        )
                        if (
                            continued.disposition is OuterDisposition.ACCEPTED
                            and continued.accepted is not None
                        ):
                            accepted = continued.accepted
                transitioned = False
                cmax_improved = False
                if accepted is not None:
                    acceptance = state.accept(
                        accepted,
                        step=step,
                    )
                    transitioned = bool(acceptance.structural_change)
                    cmax_improved = bool(acceptance.cmax_improved)
                self.scheduler.complete_step(
                    improved=transitioned,
                    primary_improved=cmax_improved,
                )

            for procedure in Procedure:
                if self.runtime.allocatable_remaining_sec <= 1e-3:
                    break
                step = ProcedureStep(
                    procedure_index=self.scheduler.procedure_count + 1,
                    cycle=self.scheduler.cycle,
                    procedure=procedure,
                    neighborhood=NeighborhoodLevel.N3,
                )
                suggested = self.runtime.slice_for(
                    "reserve",
                    max(1, len(Procedure)),
                    borrow_unused=True,
                )
                time_slice = self.budget_policy.restart_slice(
                    suggested,
                    hard_limit_sec=self.runtime.hard_limit_sec,
                )
                if time_slice <= 1e-3:
                    break
                vns_offset = state.next_vns_offset(
                    state.search_shell,
                    step.procedure,
                    step.neighborhood,
                )
                result = self._solve_step(
                    state=state,
                    step=step,
                    time_limit_sec=time_slice,
                    audit=audit,
                    record=record,
                    bucket="reserve",
                    resume=False,
                    vns_offset=vns_offset,
                )
                if (
                    result.disposition is OuterDisposition.ACCEPTED
                    and result.accepted is not None
                ):
                    state.accept(result.accepted, step=step)
        except Exception as exc:
            state.error = str(exc)
            state.status = "ENGINE_FAILED"

        if state.incumbent is not None and state.status != "ENGINE_FAILED":
            state.status = "FEASIBLE"
        result = TRAFastEngineResult(
            run_id=run_id,
            case=str(case),
            status=state.status,
            runtime_sec=float(self.runtime.elapsed_sec),
            procedure_count=int(self.scheduler.procedure_count),
            cycle_count=max(0, int(self.scheduler.cycle - 1)),
            event_count=int(event_count),
            best_objective=float(
                state.incumbent.objective if state.incumbent else float("nan")
            ),
            best_verified_cmax=float(
                state.incumbent.verified_cmax
                if state.incumbent
                else float("nan")
            ),
            manifest_sha256=str(self.templates.manifest["manifest_sha256"]),
            regular_runtime_sec=float(self.runtime.outer_used_sec),
            reserve_runtime_sec=float(self.runtime.reserve_used_sec),
            error=state.error,
        )
        audit.finish(
            {
                "algorithm": "paper-tra-fast",
                "status": result.status,
                "runtime_sec": result.runtime_sec,
                "procedure_count": result.procedure_count,
                "cycle_count": result.cycle_count,
                "event_count": result.event_count,
                "best_objective": result.best_objective,
                "best_verified_cmax": result.best_verified_cmax,
                "error": result.error,
            }
        )
        return result
