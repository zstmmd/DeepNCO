from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import gurobipy as gp
from gurobipy import GRB

from Gurobi.global_xyzu import CompiledGlobalXYZUModel, GlobalXYZUSolver
from Gurobi.tra_projection import ProjectionRegistry, StructuralFixingPlan


class TemplateStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSnapshot:
    values_by_name: Mapping[str, float]
    solver_objective: float
    solver_cmax: float
    callback_runtime_sec: float


class PersistentCompiledTemplate:
    """One precompiled model with explicit, reversible per-rotation mutations."""

    def __init__(self, compiled: CompiledGlobalXYZUModel, solver: Optional[GlobalXYZUSolver] = None) -> None:
        self.compiled = compiled
        self.solver = solver or GlobalXYZUSolver()
        self.model = compiled.model.copy()
        self.payload = self.solver._remap_vars_payload_for_model(compiled.vars_payload, self.model)
        self.registry = ProjectionRegistry.from_payload(self.payload)
        self._vars_by_name = {str(variable.VarName): variable for variable in self.model.getVars()}
        self._initial_bounds = {
            name: (float(variable.LB), float(variable.UB))
            for name, variable in self._vars_by_name.items()
        }
        self._initial_starts = {
            name: self._read_start(variable)
            for name, variable in self._vars_by_name.items()
        }
        self._touched_bound_names: set[str] = set()
        self._touched_start_names: set[str] = set()
        self._temporary_constraints: list[Any] = []
        self._solve_generation = 0

    @staticmethod
    def _read_start(variable: Any) -> float:
        try:
            return float(variable.Start)
        except Exception:
            return float(GRB.UNDEFINED)

    @property
    def solve_generation(self) -> int:
        return self._solve_generation

    def reset_for_solve(self) -> None:
        self.model.reset(0)
        if self._temporary_constraints:
            self.model.remove(self._temporary_constraints)
            self._temporary_constraints.clear()
        for name in self._touched_bound_names:
            variable = self._vars_by_name[name]
            lower, upper = self._initial_bounds[name]
            variable.LB = lower
            variable.UB = upper
        self._touched_bound_names.clear()
        for name in self._touched_start_names:
            self._vars_by_name[name].Start = self._initial_starts[name]
        self._touched_start_names.clear()
        self.model.resetParams()
        self.solver._set_solve_params(self.model, self.compiled.cfg)
        self.model.update()
        self._solve_generation += 1

    def add_constraint(self, expression: Any, *, name: str) -> Any:
        constraint = self.model.addConstr(expression, name=name)
        self._temporary_constraints.append(constraint)
        return constraint

    def fix_variable(self, variable: Any, value: int | float) -> None:
        name = str(variable.VarName)
        if name not in self._initial_bounds:
            raise TemplateStateError(f"variable is not part of this template: {name}")
        numeric = float(value)
        variable.LB = numeric
        variable.UB = numeric
        self._touched_bound_names.add(name)

    def fix_binary_families(
        self,
        plan: StructuralFixingPlan,
        *,
        families: Optional[Iterable[str]] = None,
    ) -> None:
        selected = set(families or plan.binary_values.keys())
        for family in selected:
            values = plan.binary_values.get(family)
            if values is None:
                raise TemplateStateError(f"fixing plan has no family {family}")
            variables = self.registry.family(family)
            if set(variables) != set(values):
                raise TemplateStateError(f"fixing domain differs for family {family}")
            for key, value in values.items():
                self.fix_variable(variables[key], int(value))

    def add_station_marginal_fixings(self, plan: StructuralFixingPlan, *, prefix: str) -> None:
        y = self.registry.family("y")
        by_slot_station: Dict[tuple[int, int], list[Any]] = {}
        for (slot_id, station_id, _rank), variable in y.items():
            by_slot_station.setdefault((int(slot_id), int(station_id)), []).append(variable)
        if set(by_slot_station) != set(plan.station_marginals):
            raise TemplateStateError("station marginal domain differs from structural shell")
        for (slot_id, station_id), variables in sorted(by_slot_station.items()):
            self.add_constraint(
                gp.quicksum(variables) == int(plan.station_marginals[(slot_id, station_id)]),
                name=f"{prefix}_StationMarginal_{slot_id}_{station_id}",
            )

    def install_start(
        self,
        values_by_name: Mapping[str, float],
        *,
        clear_existing: bool = False,
        variable_types: Optional[Iterable[str]] = None,
    ) -> int:
        selected_types = None if variable_types is None else {str(value) for value in variable_types}
        if clear_existing:
            for name, variable in self._vars_by_name.items():
                variable.Start = GRB.UNDEFINED
                self._touched_start_names.add(name)
        installed = 0
        for name, value in values_by_name.items():
            variable = self._vars_by_name.get(str(name))
            if variable is None or not math.isfinite(float(value)):
                continue
            if selected_types is not None and str(variable.VType) not in selected_types:
                continue
            variable.Start = float(value)
            self._touched_start_names.add(str(name))
            installed += 1
        return installed

    def set_internal_cutoff(self, incumbent_objective: Optional[float], tolerance: float) -> None:
        if incumbent_objective is None or not math.isfinite(float(incumbent_objective)):
            return
        self.model.Params.Cutoff = float(incumbent_objective) - float(tolerance)

    def set_time_limit(self, seconds: float) -> None:
        if not math.isfinite(float(seconds)) or float(seconds) <= 0.0:
            raise TemplateStateError("timed solve has no remaining budget")
        self.model.Params.TimeLimit = max(1e-3, float(seconds))

    def snapshot_from_callback(self, callback_runtime_sec: float) -> ModelSnapshot:
        variables = list(self._vars_by_name.values())
        values = self.model.cbGetSolution(variables)
        values_by_name = {
            str(variable.VarName): float(value)
            for variable, value in zip(variables, values)
        }
        objective = float(self.model.cbGet(GRB.Callback.MIPSOL_OBJ))
        cmax_var = self.payload.get("cmax")
        cmax = float(values_by_name.get(str(cmax_var.VarName), float("nan")))
        return ModelSnapshot(
            values_by_name=values_by_name,
            solver_objective=objective,
            solver_cmax=cmax,
            callback_runtime_sec=float(callback_runtime_sec),
        )
