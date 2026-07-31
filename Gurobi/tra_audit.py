from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from Gurobi.tra_events import SearchAuditLedger
from Gurobi.tra_scheduler import ProcedureStep


class SearchAuditTrail:
    """Small facade that keeps process-ledger formatting out of the engine."""

    def __init__(
        self,
        ledger: Optional[SearchAuditLedger],
        *,
        run_id: str,
        case: str,
        elapsed_sec: Callable[[], float],
    ) -> None:
        self.ledger = ledger
        self.run_id = str(run_id)
        self.case = str(case)
        self.elapsed_sec = elapsed_sec

    def _append(self, stage: str, step: Optional[ProcedureStep], payload: Mapping[str, Any]) -> None:
        if self.ledger is None:
            return
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "case": self.case,
            "stage": str(stage),
            "elapsed_sec": float(self.elapsed_sec()),
        }
        if step is not None:
            row.update(
                {
                    "procedure_index": int(step.procedure_index),
                    "cycle": int(step.cycle),
                    "procedure": str(step.procedure.value),
                    "neighborhood": str(step.neighborhood.value),
                }
            )
        row.update(dict(payload))
        self.ledger.append(row)

    @staticmethod
    def _candidate_row(candidate: Any) -> dict[str, Any]:
        risk = candidate.repair_risk
        row = {
            "shell_sha256": str(candidate.shell.sha256),
            "structural_projection": (
                candidate.shell.projection.as_canonical_payload()
            ),
            "relaxed_objective": float(candidate.relaxed_objective),
            "callback_runtime_sec": float(candidate.snapshot.callback_runtime_sec),
            "repair_risk": {
                "total": float(risk.total),
                "station_overlap_sec": float(risk.station_overlap_sec),
                "station_workload_imbalance": float(risk.station_workload_imbalance),
                "warm_disturbance_hamming": int(risk.warm_disturbance_hamming),
            },
        }
        comproc = getattr(candidate, "comproc", None)
        if comproc is not None:
            row["comproc"] = {
                "feasible": bool(comproc.feasible),
                "projected_cmax": float(comproc.projected_cmax),
                "recourse_score": float(
                    getattr(comproc, "recourse_score", comproc.projected_cmax)
                ),
                "verified_cmax": float(comproc.verified_cmax),
                "projected_objective": float(comproc.projected_objective),
                "feasibility_residual": float(comproc.feasibility_residual),
                "source": str(comproc.source),
                "error_codes": list(comproc.error_codes),
                "dp3": {
                    "no_wait_cmax": float(comproc.dp3.no_wait_cmax),
                    "feasible_start_cmax": float(
                        comproc.dp3.feasible_start_cmax
                    ),
                    "station_overlap_sec": float(
                        comproc.dp3.station_overlap_sec
                    ),
                    "station_workload_imbalance": float(
                        comproc.dp3.station_workload_imbalance
                    ),
                    "active_slot_count": int(comproc.dp3.active_slot_count),
                },
            }
        return row

    def inner(
        self,
        step: ProcedureStep,
        result: Any,
        *,
        incumbent_objective: Optional[float],
        certified_prune: bool,
        selected_shell_sha256: Optional[str],
        selection_dispositions: tuple[dict[str, str], ...] = (),
        requested_time_limit_sec: Optional[float] = None,
        effort_multiplier: Optional[float] = None,
        recourse_calibration_allowance_sec: float = 0.0,
        stage: str = "inner",
    ) -> None:
        self._append(
            stage,
            step,
            {
                "runtime_sec": float(result.runtime_sec),
                "requested_time_limit_sec": (
                    None if requested_time_limit_sec is None else float(requested_time_limit_sec)
                ),
                "effort_multiplier": (
                    None if effort_multiplier is None else float(effort_multiplier)
                ),
                "recourse_calibration_allowance_sec": float(
                    recourse_calibration_allowance_sec
                ),
                "solver_status": str(result.status),
                "solver_status_code": int(result.solver_status_code),
                "attempt_count": int(getattr(result, "attempt_count", 1)),
                "search_seeds": [int(value) for value in getattr(result, "search_seeds", ())],
                "vns_seed_sha256": list(
                    getattr(result, "vns_seed_sha256", ()) or ()
                ),
                "attempt_traces": [
                    trace.as_audit_payload()
                    for trace in getattr(result, "attempt_traces", ()) or ()
                ],
                "certified_obj_bound": float(result.certified_obj_bound),
                "certified_prune": bool(certified_prune),
                "incumbent_objective": (
                    None if incumbent_objective is None else float(incumbent_objective)
                ),
                "candidate_count": len(result.candidates),
                "selected_shell_sha256": selected_shell_sha256,
                "selection_dispositions": [
                    dict(disposition)
                    for disposition in selection_dispositions
                ],
                "candidates": [self._candidate_row(candidate) for candidate in result.candidates],
                "error": str(result.error or ""),
            },
        )

    def outer(
        self,
        step: ProcedureStep,
        result: Any,
        *,
        submitted_shell_sha256: str,
        reserve_retry: bool,
        requested_time_limit_sec: Optional[float] = None,
        budget_mode: str = "regular",
        stage: str = "outer",
        candidate_kind: Optional[str] = None,
    ) -> None:
        accepted = result.accepted
        candidate_kind_payload = (
            {}
            if candidate_kind is None
            else {"candidate_kind": str(candidate_kind)}
        )
        self._append(
            stage,
            step,
            {
                "runtime_sec": float(result.runtime_sec),
                "requested_time_limit_sec": (
                    None if requested_time_limit_sec is None else float(requested_time_limit_sec)
                ),
                "budget_mode": str(budget_mode),
                "solver_status": str(result.solver_status),
                "solver_status_code": int(result.solver_status_code),
                "objective_bound": float(result.objective_bound),
                "disposition": str(result.disposition.value),
                "submitted_shell_sha256": str(submitted_shell_sha256),
                "reserve_retry": bool(reserve_retry),
                "retry_mode": "restart" if reserve_retry else "initial_or_continuation",
                "resumed_search": bool(getattr(result, "resumed_search", False)),
                "projected_start_cmax": float(getattr(result, "projected_start_cmax", float("nan"))),
                "projected_start_wait_sec": float(
                    getattr(result, "projected_start_wait_sec", float("nan"))
                ),
                "installed_start_count": int(getattr(result, "installed_start_count", 0)),
                "full_start_complete": bool(getattr(result, "full_start_complete", False)),
                "full_start_feasible": bool(getattr(result, "full_start_feasible", False)),
                "full_start_max_residual": float(
                    getattr(result, "full_start_max_residual", float("nan"))
                ),
                "full_start_error_codes": list(
                    getattr(result, "full_start_error_codes", ()) or ()
                ),
                "start_projection_error": str(getattr(result, "start_projection_error", "") or ""),
                "verified_snapshot_count": len(result.verified_snapshots),
                "accepted_shell_sha256": None if accepted is None else str(accepted.shell.sha256),
                "accepted_objective": (
                    None if accepted is None else float(accepted.snapshot.solver_objective)
                ),
                "accepted_cmax": None if accepted is None else float(accepted.verified_cmax),
                "error": str(result.error or ""),
                **candidate_kind_payload,
            },
        )

    def queue(self, step: ProcedureStep, *, queue_name: str, reason: str, shell_sha256: str = "") -> None:
        self._append(
            "queue",
            step,
            {
                "queue_name": str(queue_name),
                "reason": str(reason),
                "shell_sha256": str(shell_sha256),
            },
        )

    def diagnostic(
        self,
        step: Optional[ProcedureStep],
        *,
        stage: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._append(str(stage), step, payload)

    def finish(self, payload: Mapping[str, Any]) -> None:
        self._append("finish", None, payload)
