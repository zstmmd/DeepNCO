from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from gurobipy import GRB

from Gurobi.tra_model_state import ModelSnapshot, PersistentCompiledTemplate
from Gurobi.tra_outer_start import (
    OuterStartProjectionError,
    build_full_start_vector,
    positive_family_start_values,
)
from Gurobi.tra_projection import StructuralShell
from Gurobi.tra_verifier import SnapshotVerifier


@dataclass(frozen=True)
class InstalledOuterStart:
    projected_cmax: float
    projected_wait_sec: float
    installed_count: int
    complete: bool
    feasible: bool
    max_residual: float
    error_codes: tuple[str, ...]
    projection_error: str = ""


def install_outer_start(
    template: PersistentCompiledTemplate,
    verifier: SnapshotVerifier,
    shell: StructuralShell,
    start_values: Mapping[str, float],
) -> InstalledOuterStart:
    model = template.model
    projected_cmax = float("nan")
    projected_wait = float("nan")
    complete = False
    feasible = False
    max_residual = float("nan")
    error_codes: tuple[str, ...] = ()
    projection_error = ""
    outer_start: Mapping[str, float] = {}
    try:
        model.update()
        full_start = build_full_start_vector(
            model,
            start_values,
            template.payload,
            shell,
        )
        projected_cmax = float(full_start.projection.projected_cmax)
        projected_wait = float(full_start.projection.added_station_wait_sec)
        complete = bool(full_start.validation.complete)
        feasible = bool(full_start.validation.feasible)
        max_residual = float(full_start.validation.max_residual)
        error_codes = tuple(full_start.validation.error_codes)
        if feasible:
            cmax_var = template.payload["cmax"]
            objective = float(
                sum(
                    float(variable.Obj)
                    * float(full_start.values_by_name[str(variable.VarName)])
                    for variable in model.getVars()
                )
            )
            semantic = verifier.verify(
                ModelSnapshot(
                    values_by_name=full_start.values_by_name,
                    solver_objective=objective,
                    solver_cmax=float(
                        full_start.values_by_name[str(cmax_var.VarName)]
                    ),
                    callback_runtime_sec=0.0,
                )
            )
            if not semantic.internal_feasible:
                feasible = False
                error_codes = tuple(
                    [
                        *error_codes,
                        *(
                            f"START_SEMANTIC:{code}"
                            for code in semantic.verifier_error_codes
                        ),
                    ]
                )
        outer_start = (
            full_start.values_by_name
            if feasible
            else full_start.safe_values_by_name
        )
    except OuterStartProjectionError as exc:
        projection_error = str(exc)
        outer_start = positive_family_start_values(
            start_values,
            template.payload,
            "y",
        )
    installed = template.install_start(
        outer_start,
        clear_existing=True,
        variable_types=None if feasible else (GRB.BINARY, GRB.INTEGER),
    )
    return InstalledOuterStart(
        projected_cmax=projected_cmax,
        projected_wait_sec=projected_wait,
        installed_count=int(installed),
        complete=complete,
        feasible=feasible,
        max_residual=max_residual,
        error_codes=error_codes,
        projection_error=projection_error,
    )
