from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional

from gurobipy import GRB

from Gurobi.tra_model_state import ModelSnapshot, PersistentCompiledTemplate, TemplateStateError
from Gurobi.tra_outer_continuation import OuterContinuationState
from Gurobi.tra_outer_search import configure_outer_certification_search
from Gurobi.tra_outer_start_install import install_outer_start
from Gurobi.tra_projection import StructuralShell
from Gurobi.tra_verifier import SnapshotVerifier, VerifiedSnapshot


class OuterDisposition(str, Enum):
    ACCEPTED = "accepted"
    PROVED_REJECT = "proved_reject"
    UNRESOLVED = "unresolved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    HARD_FAILURE = "hard_failure"


@dataclass(frozen=True)
class OuterSolveResult:
    disposition: OuterDisposition
    runtime_sec: float
    solver_status_code: int
    solver_status: str
    objective_bound: float
    verified_snapshots: tuple[VerifiedSnapshot, ...]
    accepted: Optional[VerifiedSnapshot]
    resumed_search: bool = False
    projected_start_cmax: float = float("nan")
    projected_start_wait_sec: float = float("nan")
    installed_start_count: int = 0
    full_start_complete: bool = False
    full_start_feasible: bool = False
    full_start_max_residual: float = float("nan")
    full_start_error_codes: tuple[str, ...] = ()
    start_projection_error: str = ""
    error: str = ""


def objective_tolerance(value: Optional[float]) -> float:
    numeric = float(value or 0.0)
    return max(1e-6, 1e-8 * max(1.0, abs(numeric)))


def has_unresolved_improvement_potential(
    *,
    solver_status_code: int,
    objective_bound: float,
    accepted_objective: float,
) -> bool:
    if int(solver_status_code) != GRB.TIME_LIMIT:
        return False
    if not math.isfinite(float(objective_bound)):
        return False
    return float(objective_bound) < (
        float(accepted_objective) - objective_tolerance(accepted_objective)
    )


def _solution_snapshot(template: PersistentCompiledTemplate, runtime_sec: float) -> ModelSnapshot:
    values = {str(variable.VarName): float(variable.X) for variable in template.model.getVars()}
    cmax_var = template.payload["cmax"]
    return ModelSnapshot(
        values_by_name=values,
        solver_objective=float(template.model.ObjVal),
        solver_cmax=float(values[str(cmax_var.VarName)]),
        callback_runtime_sec=float(runtime_sec),
    )


class PaperOuterTemplate:
    def __init__(
        self,
        template: PersistentCompiledTemplate,
        *,
        verifier: Optional[SnapshotVerifier] = None,
    ) -> None:
        if bool(getattr(template.compiled.cfg, "tra_inner_no_station_wait", False)):
            raise TemplateStateError("outer template must retain full station waiting")
        self.template = template
        self.verifier = verifier or SnapshotVerifier(template.compiled, solver=template.solver)
        self.continuation = OuterContinuationState()

    def solve_shell(
        self,
        shell: StructuralShell,
        *,
        time_limit_sec: float,
        incumbent_objective: Optional[float],
        start_values: Optional[Mapping[str, float]],
        formal_elapsed_at_start: float,
        verified_sink: Optional[Callable[[VerifiedSnapshot, float], None]] = None,
        reserve_retry: bool = False,
        resume_if_available: bool = False,
        incumbent_cmax: Optional[float] = None,
    ) -> OuterSolveResult:
        started = time.perf_counter()
        model = self.template.model
        callback_snapshots: list[ModelSnapshot] = []
        resumed_search = False
        projected_start_cmax = float("nan")
        projected_start_wait_sec = float("nan")
        installed_start_count = 0
        full_start_complete = False
        full_start_feasible = False
        full_start_max_residual = float("nan")
        full_start_error_codes: tuple[str, ...] = ()
        start_projection_error = ""
        try:
            resumed_search = self.continuation.plan(
                shell.sha256,
                resume_requested=resume_if_available,
            )
            tolerance = objective_tolerance(incumbent_objective)
            cmax_tolerance = objective_tolerance(incumbent_cmax)
            primary_incumbent = (
                incumbent_cmax is not None
                and math.isfinite(float(incumbent_cmax))
            )
            if not resumed_search:
                self.template.reset_for_solve()
                configure_outer_certification_search(model)
                plan = shell.fixing_plan(self.template.registry)
                self.template.fix_binary_families(plan)
                self.template.add_station_marginal_fixings(plan, prefix="TRA_Outer")
                if start_values:
                    installed = install_outer_start(
                        self.template,
                        self.verifier,
                        shell,
                        start_values,
                    )
                    projected_start_cmax = installed.projected_cmax
                    projected_start_wait_sec = installed.projected_wait_sec
                    installed_start_count = installed.installed_count
                    full_start_complete = installed.complete
                    full_start_feasible = installed.feasible
                    full_start_max_residual = installed.max_residual
                    full_start_error_codes = installed.error_codes
                    start_projection_error = installed.projection_error
                if not primary_incumbent:
                    self.template.set_internal_cutoff(incumbent_objective, tolerance)
            elif not primary_incumbent:
                self.template.set_internal_cutoff(incumbent_objective, tolerance)
            prep_elapsed = float(time.perf_counter() - started)
            remaining = float(time_limit_sec) - prep_elapsed
            verifier_reserve = min(max(0.02, 0.05 * float(time_limit_sec)), max(0.0, 0.25 * remaining))
            # Gurobi resumes the search tree, but TimeLimit is scoped to this
            # optimize() call rather than the cumulative resumed runtime.
            self.template.set_time_limit(remaining - verifier_reserve)
            model.update()

            def callback(callback_model: Any, where: int) -> None:
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    callback_runtime = float(time.perf_counter() - started)
                    callback_snapshots.append(self.template.snapshot_from_callback(callback_runtime))
                except Exception:
                    return

            model.optimize(callback)
            solve_runtime = float(time.perf_counter() - started)
            if int(model.SolCount) > 0:
                final_snapshot = _solution_snapshot(self.template, solve_runtime)
                if not callback_snapshots or (
                    abs(callback_snapshots[-1].solver_objective - final_snapshot.solver_objective) > 1e-8
                    or abs(callback_snapshots[-1].solver_cmax - final_snapshot.solver_cmax) > 1e-8
                ):
                    callback_snapshots.append(final_snapshot)

            verified: list[VerifiedSnapshot] = []
            seen_snapshot_hashes: set[str] = set()
            for snapshot in callback_snapshots:
                result = self.verifier.verify(snapshot)
                if result.snapshot_sha256 in seen_snapshot_hashes:
                    continue
                seen_snapshot_hashes.add(result.snapshot_sha256)
                if result.internal_feasible:
                    verified.append(result)
                    if verified_sink is not None:
                        incumbent_timestamp = float(formal_elapsed_at_start) + float(snapshot.callback_runtime_sec)
                        verified_sink(result, incumbent_timestamp)

            accepted: Optional[VerifiedSnapshot] = None
            feasible_by_primary = sorted(
                verified,
                key=lambda result: (
                    result.verified_cmax,
                    result.snapshot.solver_objective,
                    result.snapshot.callback_runtime_sec,
                ),
            )
            if feasible_by_primary:
                best = feasible_by_primary[0]
                if primary_incumbent:
                    if (
                        float(best.verified_cmax)
                        <= float(incumbent_cmax) + cmax_tolerance
                    ):
                        accepted = best
                elif incumbent_objective is None or not math.isfinite(float(incumbent_objective)):
                    accepted = best
                elif float(best.snapshot.solver_objective) < float(incumbent_objective) - tolerance:
                    accepted = best

            try:
                objective_bound = float(model.ObjBound)
            except Exception:
                objective_bound = float("nan")
            status_code = int(model.Status)
            status = self.template.solver._status_label(status_code)
            if status_code == GRB.TIME_LIMIT:
                self.continuation.remember(shell.sha256)
            else:
                self.continuation.clear()
            if accepted is not None:
                disposition = OuterDisposition.ACCEPTED
            elif status_code == GRB.INFEASIBLE:
                disposition = OuterDisposition.PROVED_REJECT
            elif primary_incumbent and status_code == GRB.OPTIMAL:
                disposition = OuterDisposition.PROVED_REJECT
            elif (
                not primary_incumbent
                and incumbent_objective is not None
                and math.isfinite(float(objective_bound))
                and float(objective_bound) >= float(incumbent_objective) - tolerance
                and status_code in {GRB.OPTIMAL, GRB.INFEASIBLE, GRB.CUTOFF}
            ):
                disposition = OuterDisposition.PROVED_REJECT
            elif reserve_retry:
                disposition = OuterDisposition.BUDGET_EXHAUSTED
            else:
                disposition = OuterDisposition.UNRESOLVED
            return OuterSolveResult(
                disposition=disposition,
                runtime_sec=float(time.perf_counter() - started),
                solver_status_code=status_code,
                solver_status=status,
                objective_bound=objective_bound,
                verified_snapshots=tuple(verified),
                accepted=accepted,
                resumed_search=resumed_search,
                projected_start_cmax=projected_start_cmax,
                projected_start_wait_sec=projected_start_wait_sec,
                installed_start_count=installed_start_count,
                full_start_complete=full_start_complete,
                full_start_feasible=full_start_feasible,
                full_start_max_residual=full_start_max_residual,
                full_start_error_codes=full_start_error_codes,
                start_projection_error=start_projection_error,
            )
        except Exception as exc:
            self.continuation.clear()
            disposition = (
                OuterDisposition.BUDGET_EXHAUSTED
                if isinstance(exc, TemplateStateError)
                else OuterDisposition.HARD_FAILURE
            )
            return OuterSolveResult(
                disposition=disposition,
                runtime_sec=float(time.perf_counter() - started),
                solver_status_code=int(getattr(model, "Status", 0) or 0),
                solver_status=(
                    "BUDGET_EXHAUSTED"
                    if disposition is OuterDisposition.BUDGET_EXHAUSTED
                    else "OUTER_FAILED"
                ),
                objective_bound=float("nan"),
                verified_snapshots=(),
                accepted=None,
                resumed_search=resumed_search,
                projected_start_cmax=projected_start_cmax,
                projected_start_wait_sec=projected_start_wait_sec,
                installed_start_count=installed_start_count,
                full_start_complete=full_start_complete,
                full_start_feasible=full_start_feasible,
                full_start_max_residual=full_start_max_residual,
                full_start_error_codes=full_start_error_codes,
                start_projection_error=start_projection_error,
                error=str(exc),
            )
