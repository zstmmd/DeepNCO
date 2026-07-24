from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_events import EventLedger, FeasibleSolutionEvent, SearchAuditLedger
from Gurobi.tra_initial import CanonicalInitialState, build_canonical_initial_state
from Gurobi.tra_regular_phase import RegularRotationPhase
from Gurobi.tra_reserve_phase import ReservePhase
from Gurobi.tra_scheduler import ProcedureStep, RotationScheduler, RuntimeLedger
from Gurobi.tra_search_state import SearchState
from Gurobi.tra_templates import PaperTRATemplates
from Gurobi.tra_verifier import VerifiedSnapshot


@dataclass(frozen=True)
class TRAEngineResult:
    run_id: str
    case: str
    status: str
    runtime_sec: float
    procedure_count: int
    cycle_count: int
    event_count: int
    best_objective: float
    best_verified_cmax: float
    manifest_sha256: str
    inner_runtime_sec: float
    outer_runtime_sec: float
    reserve_runtime_sec: float
    unresolved_remaining: int
    deferred_remaining: int
    error: str = ""


class PaperTRAEngine:
    def __init__(
        self,
        templates: PaperTRATemplates,
        runtime: RuntimeLedger,
        *,
        max_procedures: int = 50,
    ) -> None:
        self.templates = templates
        self.runtime = runtime
        self.scheduler = RotationScheduler(max_procedures=max_procedures, stagnant_cycle_limit=3)

    def _event(
        self,
        *,
        run_id: str,
        case: str,
        verified: VerifiedSnapshot,
        step: Optional[ProcedureStep],
        solver_timestamp_sec: float,
        provenance: str,
        reserve_retry: bool,
    ) -> FeasibleSolutionEvent:
        manifest = self.templates.manifest
        return FeasibleSolutionEvent(
            run_id=run_id,
            case=case,
            wall_timestamp_sec=float(self.runtime.elapsed_sec),
            solver_incumbent_timestamp_sec=float(solver_timestamp_sec),
            cycle=int(step.cycle if step else 0),
            procedure=str(step.procedure.value if step else "WARM"),
            neighborhood=str(step.neighborhood.value if step else "CANONICAL"),
            manifest_sha256=str(manifest["manifest_sha256"]),
            objective_sha256=str(manifest["model_fingerprints"]["objective_sha256"]),
            structural_hash=verified.shell.sha256,
            solver_objective=float(verified.snapshot.solver_objective),
            solver_cmax=float(verified.snapshot.solver_cmax),
            verified_cmax=float(verified.verified_cmax),
            internal_feasible=True,
            verifier_error_codes=verified.verifier_error_codes,
            provenance={
                "source": provenance,
                "reserve_retry": bool(reserve_retry),
                "retry_mode": "restart" if reserve_retry else "initial_or_continuation",
            },
            snapshot_sha256=verified.snapshot_sha256,
            structural_projection=verified.shell.projection.as_canonical_payload(),
        )

    def run(
        self,
        *,
        case: str,
        ledger: EventLedger,
        audit_ledger: Optional[SearchAuditLedger] = None,
        run_id: Optional[str] = None,
    ) -> TRAEngineResult:
        run_id = str(run_id or uuid.uuid4().hex)
        audit = SearchAuditTrail(
            audit_ledger,
            run_id=run_id,
            case=case,
            elapsed_sec=lambda: self.runtime.elapsed_sec,
        )
        initial: CanonicalInitialState = build_canonical_initial_state(
            self.templates.outer.template,
            self.templates.outer.verifier,
        )
        state = SearchState(
            search_shell=initial.search_shell,
            start_values=dict(initial.start_values),
        )
        event_count = 0

        def record_verified(
            verified: VerifiedSnapshot,
            step: ProcedureStep,
            solver_timestamp_sec: float,
            provenance: str,
            reserve_retry: bool,
        ) -> None:
            nonlocal event_count
            ledger.append(
                self._event(
                    run_id=run_id,
                    case=case,
                    verified=verified,
                    step=step,
                    solver_timestamp_sec=solver_timestamp_sec,
                    provenance=provenance,
                    reserve_retry=reserve_retry,
                )
            )
            event_count += 1

        if initial.verified_incumbent is not None:
            state.accept(initial.verified_incumbent)
            state.status = "WARM_INCUMBENT"
            ledger.append(
                self._event(
                    run_id=run_id,
                    case=case,
                    verified=initial.verified_incumbent,
                    step=None,
                    solver_timestamp_sec=0.0,
                    provenance="canonical_warm_start",
                    reserve_retry=False,
                )
            )
            event_count += 1

        # The formal clock begins immediately before the first F1 reset/fixing.
        self.runtime.start()
        regular = RegularRotationPhase(
            self.templates,
            self.runtime,
            self.scheduler,
            audit,
            record_verified,
        )
        reserve = ReservePhase(
            self.templates,
            self.runtime,
            self.scheduler,
            audit,
            record_verified,
        )
        empty_queue_restarts = 0
        try:
            while (
                self.runtime.allocatable_remaining_sec > 1e-3
                and self.scheduler.procedure_count < self.scheduler.max_procedures
            ):
                shell_before_regular = str(state.search_shell.sha256)
                regular.run(state)
                if str(state.search_shell.sha256) != shell_before_regular:
                    empty_queue_restarts = 0
                if state.status == "ENGINE_FAILED":
                    break
                if state.queues.empty:
                    if state.restore_global_search():
                        self.scheduler.restart_after_external_improvement()
                        empty_queue_restarts = 0
                        continue
                    if (
                        empty_queue_restarts < 2
                        and self.runtime.allocatable_remaining_sec > 1.0
                        and self.scheduler.procedure_count
                        < self.scheduler.max_procedures
                    ):
                        empty_queue_restarts += 1
                        self.scheduler.restart_after_external_improvement()
                        continue
                    break
                if reserve.run(state):
                    empty_queue_restarts = 0
                    continue
                if state.restore_global_search():
                    self.scheduler.restart_after_external_improvement()
                    empty_queue_restarts = 0
                    continue
                if (
                    empty_queue_restarts < 2
                    and self.runtime.allocatable_remaining_sec > 1.0
                    and self.scheduler.procedure_count
                    < self.scheduler.max_procedures
                ):
                    empty_queue_restarts += 1
                    self.scheduler.restart_after_external_improvement()
                    continue
                break
        except Exception as exc:
            state.error = str(exc)
            state.status = "ENGINE_FAILED"

        if state.incumbent is not None and state.status != "ENGINE_FAILED":
            state.status = "FEASIBLE"
        result = TRAEngineResult(
            run_id=run_id,
            case=str(case),
            status=state.status,
            runtime_sec=float(self.runtime.elapsed_sec),
            procedure_count=int(self.scheduler.procedure_count),
            cycle_count=max(0, int(self.scheduler.cycle - 1)),
            event_count=int(event_count),
            best_objective=float(state.incumbent.objective if state.incumbent else float("nan")),
            best_verified_cmax=float(state.incumbent.verified_cmax if state.incumbent else float("nan")),
            manifest_sha256=str(self.templates.manifest["manifest_sha256"]),
            inner_runtime_sec=float(self.runtime.inner_used_sec),
            outer_runtime_sec=float(self.runtime.outer_used_sec),
            reserve_runtime_sec=float(self.runtime.reserve_used_sec),
            unresolved_remaining=state.queues.pending_count,
            deferred_remaining=state.queues.deferred_count,
            error=state.error,
        )
        audit.finish(
            {
                "status": result.status,
                "runtime_sec": result.runtime_sec,
                "procedure_count": result.procedure_count,
                "cycle_count": result.cycle_count,
                "event_count": result.event_count,
                "best_objective": result.best_objective,
                "best_verified_cmax": result.best_verified_cmax,
                "pending_outer_remaining": result.unresolved_remaining,
                "deferred_inner_remaining": result.deferred_remaining,
                "error": result.error,
            }
        )
        return result
