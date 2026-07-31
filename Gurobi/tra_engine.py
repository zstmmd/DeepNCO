from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_candidate_census import CandidateObserver
from Gurobi.tra_events import EventLedger, FeasibleSolutionEvent, SearchAuditLedger
from Gurobi.tra_initial import CanonicalInitialState, build_canonical_initial_state
from Gurobi.tra_regular_phase import RegularRotationPhase
from Gurobi.tra_reserve_phase import ReservePhase
from Gurobi.tra_scheduler import ProcedureStep, RotationScheduler, RuntimeLedger
from Gurobi.tra_neighborhood import Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_projection import INACTIVE_LABEL
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
        candidate_observer: CandidateObserver | None = None,
        enable_f1_live_seed_starts: bool = False,
        enable_f1_plateau_escalation: bool = False,
        plateau_escalation_procedures: Iterable[Procedure] = (),
        enable_station_balance_repair: bool = False,
        enable_canonical_initial_outer_seed: bool = True,
        canonical_initial_outer_active_slot_threshold: int = 20,
        canonical_initial_outer_hard_fraction: float = 0.18,
    ) -> None:
        self.templates = templates
        self.runtime = runtime
        self.plateau_escalation_procedures = {
            Procedure(procedure)
            for procedure in plateau_escalation_procedures
        }
        if bool(enable_f1_plateau_escalation):
            self.plateau_escalation_procedures.add(Procedure.F1)
        self.scheduler = RotationScheduler(
            max_procedures=max_procedures,
            stagnant_cycle_limit=3,
            enable_f1_plateau_escalation=enable_f1_plateau_escalation,
            plateau_escalation_procedures=self.plateau_escalation_procedures,
        )
        self.candidate_observer = candidate_observer
        self.enable_f1_live_seed_starts = bool(enable_f1_live_seed_starts)
        self.enable_f1_plateau_escalation = bool(enable_f1_plateau_escalation)
        self.enable_station_balance_repair = bool(enable_station_balance_repair)
        self.enable_canonical_initial_outer_seed = bool(
            enable_canonical_initial_outer_seed
        )
        self.canonical_initial_outer_active_slot_threshold = max(
            0,
            int(canonical_initial_outer_active_slot_threshold),
        )
        self.canonical_initial_outer_hard_fraction = max(
            0.0,
            float(canonical_initial_outer_hard_fraction),
        )
        self.best_verified_snapshot: Optional[VerifiedSnapshot] = None
        self.best_verified_wall_timestamp_sec = float("nan")

    @staticmethod
    def _verified_sort_key(verified: VerifiedSnapshot) -> tuple[float, float, float, str]:
        return (
            float(verified.verified_cmax),
            float(verified.snapshot.solver_objective),
            float(verified.snapshot.callback_runtime_sec),
            str(verified.snapshot_sha256),
        )

    def _record_best_verified(
        self,
        verified: VerifiedSnapshot,
        *,
        wall_timestamp_sec: float,
    ) -> None:
        if not verified.internal_feasible:
            return
        if not math.isfinite(float(wall_timestamp_sec)):
            return
        if (
            self.best_verified_snapshot is None
            or self._verified_sort_key(verified)
            < self._verified_sort_key(self.best_verified_snapshot)
        ):
            self.best_verified_snapshot = verified
            self.best_verified_wall_timestamp_sec = float(wall_timestamp_sec)

    def _restart_regular_cycle(self, state: SearchState) -> None:
        preserve_f1 = bool(
            self.enable_f1_plateau_escalation
            and state.incumbent is not None
        )
        preserved = set(getattr(self, "plateau_escalation_procedures", set()))
        if preserve_f1:
            preserved.add(Procedure.F1)
        extra_preserved = (
            tuple(
                sorted(
                    (
                        procedure
                        for procedure in preserved
                        if procedure is not Procedure.F1
                    ),
                    key=lambda item: item.value,
                )
            )
            if state.incumbent is not None
            else ()
        )
        kwargs = {"preserve_f1_level": preserve_f1}
        if extra_preserved:
            kwargs["preserve_procedure_levels"] = extra_preserved
        self.scheduler.restart_after_external_improvement(**kwargs)

    @staticmethod
    def _initial_active_slot_count(initial: CanonicalInitialState) -> int:
        return len(
            {
                int(slot_id)
                for slot_id in initial.search_shell.projection.x_group.values()
                if int(slot_id) != INACTIVE_LABEL
            }
        )

    def _should_run_canonical_initial_outer(
        self,
        initial: CanonicalInitialState,
    ) -> bool:
        return bool(
            self.enable_canonical_initial_outer_seed
            and initial.verified_incumbent is None
            and self._initial_active_slot_count(initial)
            >= int(self.canonical_initial_outer_active_slot_threshold)
        )

    def _canonical_initial_outer_slice(self) -> float:
        if self.runtime.allocatable_remaining_sec <= 1e-3:
            return 0.0
        repair_floor = max(
            2.0,
            float(self.runtime.hard_limit_sec)
            * float(self.canonical_initial_outer_hard_fraction),
        )
        return min(float(self.runtime.allocatable_remaining_sec), repair_floor)

    def _should_stop_after_reserve_progress(self) -> bool:
        """Avoid opening a fresh solver slice when reserve has almost exhausted cap."""

        low_remaining_guard = max(
            6.0,
            4.0 * max(0.0, float(self.runtime.minimum_solver_slice_sec)),
        )
        return bool(
            float(self.runtime.allocatable_remaining_sec)
            <= float(low_remaining_guard)
        )

    def _event(
        self,
        *,
        run_id: str,
        case: str,
        verified: VerifiedSnapshot,
        step: Optional[ProcedureStep],
        solver_timestamp_sec: float,
        wall_timestamp_sec: float,
        provenance: str,
        reserve_retry: bool,
    ) -> FeasibleSolutionEvent:
        manifest = self.templates.manifest
        return FeasibleSolutionEvent(
            run_id=run_id,
            case=case,
            wall_timestamp_sec=float(wall_timestamp_sec),
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
        self.best_verified_snapshot = None
        self.best_verified_wall_timestamp_sec = float("nan")
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
            wall_timestamp_sec = float(self.runtime.elapsed_sec)
            self._record_best_verified(
                verified,
                wall_timestamp_sec=wall_timestamp_sec,
            )
            ledger.append(
                self._event(
                    run_id=run_id,
                    case=case,
                    verified=verified,
                    step=step,
                    solver_timestamp_sec=solver_timestamp_sec,
                    wall_timestamp_sec=wall_timestamp_sec,
                    provenance=provenance,
                    reserve_retry=reserve_retry,
                )
            )
            event_count += 1

        if initial.verified_incumbent is not None:
            state.accept(initial.verified_incumbent)
            state.status = "WARM_INCUMBENT"
            self._record_best_verified(
                initial.verified_incumbent,
                wall_timestamp_sec=0.0,
            )
            ledger.append(
                self._event(
                    run_id=run_id,
                    case=case,
                    verified=initial.verified_incumbent,
                    step=None,
                    solver_timestamp_sec=0.0,
                    wall_timestamp_sec=0.0,
                    provenance="canonical_warm_start",
                    reserve_retry=False,
                )
            )
            event_count += 1

        # The formal clock begins immediately before the first F1 reset/fixing.
        self.runtime.start()

        if self._should_run_canonical_initial_outer(initial):
            outer_slice = self._canonical_initial_outer_slice()
            if outer_slice > 1e-3:

                def record_initial_verified(
                    verified: VerifiedSnapshot,
                    solver_timestamp_sec: float,
                ) -> None:
                    nonlocal event_count
                    wall_timestamp_sec = float(self.runtime.elapsed_sec)
                    self._record_best_verified(
                        verified,
                        wall_timestamp_sec=wall_timestamp_sec,
                    )
                    ledger.append(
                        self._event(
                            run_id=run_id,
                            case=case,
                            verified=verified,
                            step=None,
                            solver_timestamp_sec=solver_timestamp_sec,
                            wall_timestamp_sec=wall_timestamp_sec,
                            provenance="canonical_initial_outer_mipsol",
                            reserve_retry=False,
                        )
                    )
                    event_count += 1

                result = self.templates.outer.solve_shell(
                    initial.search_shell,
                    time_limit_sec=outer_slice,
                    incumbent_objective=None,
                    start_values=dict(initial.start_values),
                    formal_elapsed_at_start=self.runtime.elapsed_sec,
                    verified_sink=record_initial_verified,
                    reserve_retry=False,
                    resume_if_available=False,
                    incumbent_cmax=None,
                )
                self.runtime.record("outer", result.runtime_sec)
                audit.outer(
                    None,
                    result,
                    submitted_shell_sha256=initial.search_shell.sha256,
                    reserve_retry=False,
                    requested_time_limit_sec=outer_slice,
                    budget_mode="canonical_initial_outer",
                    stage="canonical_initial_outer",
                )
                if result.disposition is OuterDisposition.HARD_FAILURE:
                    state.error = str(result.error)
                    state.status = "ENGINE_FAILED"
                elif (
                    result.disposition is OuterDisposition.ACCEPTED
                    and result.accepted is not None
                ):
                    state.accept(result.accepted)

        regular = RegularRotationPhase(
            self.templates,
            self.runtime,
            self.scheduler,
            audit,
            record_verified,
            candidate_observer=self.candidate_observer,
            enable_f1_live_seed_starts=self.enable_f1_live_seed_starts,
            enable_station_balance_repair=self.enable_station_balance_repair,
        )
        reserve = ReservePhase(
            self.templates,
            self.runtime,
            self.scheduler,
            audit,
            record_verified,
            enable_f1_plateau_escalation=self.enable_f1_plateau_escalation,
            plateau_escalation_procedures=self.plateau_escalation_procedures,
        )
        empty_queue_restarts = 0
        try:
            while (
                self.runtime.allocatable_remaining_sec > 1e-3
                and self.scheduler.procedure_count < self.scheduler.max_procedures
            ):
                shell_before_regular = str(state.search_shell.sha256)
                stop_requested = regular.run(state)
                if str(state.search_shell.sha256) != shell_before_regular:
                    empty_queue_restarts = 0
                if stop_requested:
                    break
                if state.status == "ENGINE_FAILED":
                    break
                if state.queues.empty:
                    if state.restore_global_search():
                        self._restart_regular_cycle(state)
                        empty_queue_restarts = 0
                        continue
                    if (
                        empty_queue_restarts < 2
                        and self.runtime.allocatable_remaining_sec > 1.0
                        and self.scheduler.procedure_count
                        < self.scheduler.max_procedures
                    ):
                        empty_queue_restarts += 1
                        self._restart_regular_cycle(state)
                        continue
                    break
                if reserve.run(state):
                    empty_queue_restarts = 0
                    if self._should_stop_after_reserve_progress():
                        break
                    continue
                if state.restore_global_search():
                    self._restart_regular_cycle(state)
                    empty_queue_restarts = 0
                    continue
                if (
                    empty_queue_restarts < 2
                    and self.runtime.allocatable_remaining_sec > 1.0
                    and self.scheduler.procedure_count
                    < self.scheduler.max_procedures
                ):
                    empty_queue_restarts += 1
                    self._restart_regular_cycle(state)
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
            best_verified_cmax=float(
                self.best_verified_snapshot.verified_cmax
                if self.best_verified_snapshot is not None
                else float("nan")
            ),
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
