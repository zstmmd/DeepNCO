from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import gurobipy as gp

from Gurobi.global_xyzu import CompiledGlobalXYZUModel, GlobalXYZUSolver, RankAwareGlobalTimeCalculator
from Gurobi.tra_model_state import ModelSnapshot
from Gurobi.tra_projection import ProjectionRegistry, StructuralShell


@dataclass(frozen=True)
class SnapshotValue:
    VarName: str
    X: float


@dataclass(frozen=True)
class VerifiedSnapshot:
    snapshot: ModelSnapshot
    shell: StructuralShell
    snapshot_sha256: str
    internal_feasible: bool
    verified_cmax: float
    verifier_error_codes: tuple[str, ...]
    verifier_runtime_sec: float
    materialized_problem: Any = None


def _snapshot_hash(values_by_name: Mapping[str, float]) -> str:
    rows = [[str(name), float(value)] for name, value in sorted(values_by_name.items())]
    text = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot_payload(payload: Mapping[str, Any], values_by_name: Mapping[str, float]) -> Dict[str, Any]:
    def remap(value: Any) -> Any:
        try:
            if isinstance(value, gp.Var):
                name = str(value.VarName)
                return SnapshotValue(name, float(values_by_name[name]))
            if isinstance(value, gp.tupledict):
                return {key: remap(item) for key, item in value.items()}
        except Exception:
            pass
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(remap(item) for item in value)
        if isinstance(value, set):
            return {remap(item) for item in value}
        return value

    return {key: remap(value) for key, value in payload.items()}


class SnapshotVerifier:
    """Read-only verifier: it reports violations and never repairs a snapshot."""

    def __init__(
        self,
        compiled: CompiledGlobalXYZUModel,
        *,
        solver: GlobalXYZUSolver | None = None,
        feasibility_tolerance: float = 1e-5,
        cmax_tolerance: float = 1e-2,
    ) -> None:
        self.compiled = compiled
        self.solver = solver or GlobalXYZUSolver()
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.cmax_tolerance = float(cmax_tolerance)
        self._variables = {str(variable.VarName): variable for variable in compiled.model.getVars()}

    def verify(self, snapshot: ModelSnapshot) -> VerifiedSnapshot:
        started = time.perf_counter()
        errors: list[str] = []
        values = snapshot.values_by_name
        missing = set(self._variables) - set(values)
        extra = set(values) - set(self._variables)
        if missing:
            errors.append("missing_variable_values")
        if extra:
            errors.append("unknown_variable_values")

        for name, variable in self._variables.items():
            if name not in values:
                continue
            value = float(values[name])
            if not math.isfinite(value):
                errors.append("nonfinite_variable_value")
                break
            if value < float(variable.LB) - self.feasibility_tolerance or value > float(variable.UB) + self.feasibility_tolerance:
                errors.append("variable_bound_violation")
                break
            if str(variable.VType) in {"B", "I"} and abs(value - round(value)) > self.feasibility_tolerance:
                errors.append("integrality_violation")
                break

        shell: StructuralShell
        materialized_problem = None
        verified_cmax = float("nan")
        try:
            snapshot_payload = _snapshot_payload(self.compiled.vars_payload, values)
            registry = ProjectionRegistry.from_payload(snapshot_payload)
            shell = StructuralShell.extract(registry)
            extraction = self.solver._extract_xyz_solution(snapshot_payload, self.compiled.prepared)
            materialized_problem = copy.deepcopy(self.compiled.problem_template)
            self.solver._materialize_solution(materialized_problem, extraction, self.compiled.prepared)
            verified_cmax = float(RankAwareGlobalTimeCalculator(materialized_problem).calculate())
            if not math.isfinite(verified_cmax):
                errors.append("nonfinite_verified_cmax")
            elif abs(float(snapshot.solver_cmax) - verified_cmax) > self.cmax_tolerance:
                errors.append("cmax_semantic_mismatch")
        except Exception:
            shell = StructuralShell(
                projection=ProjectionRegistry.from_payload(
                    self.compiled.vars_payload,
                    value_getter=lambda variable: values[str(variable.VarName)],
                ).extract(),
                z_actions={},
            )
            errors.append("semantic_extraction_failed")

        runtime = float(time.perf_counter() - started)
        return VerifiedSnapshot(
            snapshot=snapshot,
            shell=shell,
            snapshot_sha256=_snapshot_hash(values),
            internal_feasible=not errors,
            verified_cmax=verified_cmax,
            verifier_error_codes=tuple(dict.fromkeys(errors)),
            verifier_runtime_sec=runtime,
            materialized_problem=materialized_problem,
        )
