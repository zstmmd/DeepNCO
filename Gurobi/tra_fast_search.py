from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from gurobipy import GRB

from Gurobi.tra_inner_search import configure_inner_search, project_inner_start
from Gurobi.tra_local_branching import apply_local_neighborhood
from Gurobi.tra_model_state import (
    ModelSnapshot,
    PersistentCompiledTemplate,
    TemplateStateError,
)
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition, objective_tolerance
from Gurobi.tra_outer_continuation import OuterContinuationState
from Gurobi.tra_verifier import SnapshotVerifier, VerifiedSnapshot


@dataclass(frozen=True)
class FastNeighborhoodResult:
    disposition: OuterDisposition
    runtime_sec: float
    solver_status_code: int
    solver_status: str
    objective_bound: float
    verified_snapshots: tuple[VerifiedSnapshot, ...]
    accepted: Optional[VerifiedSnapshot]
    resumed_search: bool
    verifier_error_codes: tuple[str, ...] = ()
    error: str = ""


def _solution_snapshot(
    template: PersistentCompiledTemplate,
    runtime_sec: float,
) -> ModelSnapshot:
    values = {
        str(variable.VarName): float(variable.X)
        for variable in template.model.getVars()
    }
    cmax = template.payload["cmax"]
    return ModelSnapshot(
        values_by_name=values,
        solver_objective=float(template.model.ObjVal),
        solver_cmax=float(values[str(cmax.VarName)]),
        callback_runtime_sec=float(runtime_sec),
    )


class PaperFastNeighborhoodTemplate:
    """Full-model local branching used by TRA-Fast without a no-wait inner MIP."""

    def __init__(
        self,
        template: PersistentCompiledTemplate,
        *,
        verifier: SnapshotVerifier,
    ) -> None:
        if bool(getattr(template.compiled.cfg, "tra_inner_no_station_wait", False)):
            raise TemplateStateError("TRA-Fast requires the full station-wait model")
        self.template = template
        self.verifier = verifier
        self.continuation = OuterContinuationState()

    def solve(
        self,
        reference_shell: Any,
        *,
        procedure: Procedure,
        neighborhood: NeighborhoodLevel,
        time_limit_sec: float,
        incumbent_objective: Optional[float],
        start_values: Mapping[str, float],
        vns_start_values: Sequence[Mapping[str, float]],
        formal_elapsed_at_start: float,
        verified_sink: Optional[Callable[[VerifiedSnapshot, float], None]] = None,
        resume_if_available: bool = False,
        incumbent_cmax: Optional[float] = None,
    ) -> FastNeighborhoodResult:
        procedure = Procedure(procedure)
        neighborhood = NeighborhoodLevel(neighborhood)
        key = f"{reference_shell.sha256}:{procedure.value}:{neighborhood.value}"
        started = time.perf_counter()
        model = self.template.model
        callback_snapshots: list[ModelSnapshot] = []
        rejected_codes: list[str] = []
        resumed = False
        try:
            resumed = self.continuation.plan(
                key,
                resume_requested=resume_if_available,
            )
            if not resumed:
                self.template.reset_for_solve()
                configure_inner_search(model)
                projected_start = project_inner_start(
                    start_values,
                    self.template.payload,
                    procedure,
                )
                if vns_start_values and procedure is not Procedure.F1:
                    projected_start.update(
                        {
                            str(name): float(value)
                            for name, value in vns_start_values[0].items()
                        }
                    )
                self.template.install_start(projected_start, clear_existing=True)
                apply_local_neighborhood(
                    self.template,
                    reference_shell,
                    procedure,
                    neighborhood,
                )
            tolerance = objective_tolerance(incumbent_objective)
            cmax_tolerance = objective_tolerance(incumbent_cmax)
            primary_incumbent = (
                incumbent_cmax is not None
                and math.isfinite(float(incumbent_cmax))
            )
            if not primary_incumbent:
                self.template.set_internal_cutoff(incumbent_objective, tolerance)
            elapsed = float(time.perf_counter() - started)
            remaining = float(time_limit_sec) - elapsed
            verifier_reserve = min(
                max(0.02, 0.05 * float(time_limit_sec)),
                max(0.0, 0.25 * remaining),
            )
            self.template.set_time_limit(remaining - verifier_reserve)
            model.update()

            def callback(callback_model: Any, where: int) -> None:
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    callback_runtime = float(time.perf_counter() - started)
                    callback_snapshots.append(
                        self.template.snapshot_from_callback(callback_runtime)
                    )
                except Exception:
                    return

            model.optimize(callback)
            solve_runtime = float(time.perf_counter() - started)
            if int(model.SolCount) > 0:
                final_snapshot = _solution_snapshot(self.template, solve_runtime)
                if not callback_snapshots or (
                    abs(
                        callback_snapshots[-1].solver_objective
                        - final_snapshot.solver_objective
                    )
                    > 1e-8
                ):
                    callback_snapshots.append(final_snapshot)

            verified: list[VerifiedSnapshot] = []
            seen: set[str] = set()
            for snapshot in callback_snapshots:
                result = self.verifier.verify(snapshot)
                if result.snapshot_sha256 in seen:
                    continue
                seen.add(result.snapshot_sha256)
                if not result.internal_feasible:
                    rejected_codes.extend(result.verifier_error_codes)
                    continue
                verified.append(result)
                if verified_sink is not None:
                    verified_sink(
                        result,
                        float(formal_elapsed_at_start)
                        + float(snapshot.callback_runtime_sec),
                    )

            accepted = None
            if verified:
                best = min(
                    verified,
                    key=lambda item: (
                        item.verified_cmax,
                        item.snapshot.solver_objective,
                        item.snapshot.callback_runtime_sec,
                    ),
                )
                if primary_incumbent:
                    if (
                        float(best.verified_cmax)
                        <= float(incumbent_cmax) + cmax_tolerance
                    ):
                        accepted = best
                elif (
                    incumbent_objective is None
                    or not math.isfinite(float(incumbent_objective))
                    or float(best.snapshot.solver_objective)
                    < float(incumbent_objective) - tolerance
                ):
                    accepted = best
            try:
                bound = float(model.ObjBound)
            except Exception:
                bound = float("nan")
            status_code = int(model.Status)
            if status_code == GRB.TIME_LIMIT:
                self.continuation.remember(key)
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
                and math.isfinite(bound)
                and bound >= float(incumbent_objective) - tolerance
                and status_code in {GRB.OPTIMAL, GRB.CUTOFF}
            ):
                disposition = OuterDisposition.PROVED_REJECT
            else:
                disposition = OuterDisposition.UNRESOLVED
            return FastNeighborhoodResult(
                disposition=disposition,
                runtime_sec=float(time.perf_counter() - started),
                solver_status_code=status_code,
                solver_status=self.template.solver._status_label(status_code),
                objective_bound=bound,
                verified_snapshots=tuple(verified),
                accepted=accepted,
                resumed_search=resumed,
                verifier_error_codes=tuple(sorted(set(rejected_codes))),
            )
        except Exception as exc:
            self.continuation.clear()
            return FastNeighborhoodResult(
                disposition=OuterDisposition.HARD_FAILURE,
                runtime_sec=float(time.perf_counter() - started),
                solver_status_code=int(getattr(model, "Status", 0) or 0),
                solver_status="FAST_NEIGHBORHOOD_FAILED",
                objective_bound=float("nan"),
                verified_snapshots=(),
                accepted=None,
                resumed_search=resumed,
                verifier_error_codes=tuple(sorted(set(rejected_codes))),
                error=str(exc),
            )
