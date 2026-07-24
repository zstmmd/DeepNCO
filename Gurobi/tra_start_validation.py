from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from gurobipy import GRB


@dataclass(frozen=True)
class FullStartValidation:
    complete: bool
    feasible: bool
    variable_count: int
    supplied_count: int
    missing_names: tuple[str, ...]
    nonfinite_names: tuple[str, ...]
    bound_violation_names: tuple[str, ...]
    integrality_violation_names: tuple[str, ...]
    constraint_violation_names: tuple[str, ...]
    unsupported_general_constraint_types: tuple[int, ...]
    max_residual: float

    @property
    def error_codes(self) -> tuple[str, ...]:
        codes: list[str] = []
        if self.missing_names:
            codes.append("START_MISSING_VARIABLES")
        if self.nonfinite_names:
            codes.append("START_NONFINITE_VALUES")
        if self.bound_violation_names:
            codes.append("START_BOUND_VIOLATION")
        if self.integrality_violation_names:
            codes.append("START_INTEGRALITY_VIOLATION")
        if self.constraint_violation_names:
            codes.append("START_CONSTRAINT_VIOLATION")
        if self.unsupported_general_constraint_types:
            codes.append("START_UNSUPPORTED_GENERAL_CONSTRAINT")
        return tuple(codes)


def _linear_value(expression: Any, values: Mapping[str, float]) -> float:
    total = float(expression.getConstant())
    for index in range(int(expression.size())):
        variable = expression.getVar(index)
        total += float(expression.getCoeff(index)) * float(values[str(variable.VarName)])
    return float(total)


def _sense_residual(lhs: float, sense: str, rhs: float) -> float:
    if sense == GRB.LESS_EQUAL:
        return max(0.0, float(lhs) - float(rhs))
    if sense == GRB.GREATER_EQUAL:
        return max(0.0, float(rhs) - float(lhs))
    return abs(float(lhs) - float(rhs))


def validate_full_start(
    model: Any,
    values_by_name: Mapping[str, float],
    *,
    tolerance: float = 1e-5,
    max_reported_names: int = 20,
) -> FullStartValidation:
    """Evaluate a complete candidate start without invoking optimize()."""

    model.update()
    variables = list(model.getVars())
    normalized = {
        str(name): float(value)
        for name, value in values_by_name.items()
        if str(name)
    }
    variable_names = {str(variable.VarName) for variable in variables}
    missing = sorted(variable_names - set(normalized))
    nonfinite: list[str] = []
    bound_violations: list[str] = []
    integrality_violations: list[str] = []
    max_residual = 0.0

    for variable in variables:
        name = str(variable.VarName)
        if name not in normalized:
            continue
        value = float(normalized[name])
        if not math.isfinite(value):
            nonfinite.append(name)
            continue
        lower_residual = max(0.0, float(variable.LB) - value)
        upper_residual = max(0.0, value - float(variable.UB))
        bound_residual = max(lower_residual, upper_residual)
        max_residual = max(max_residual, bound_residual)
        if bound_residual > tolerance:
            bound_violations.append(name)
        if str(variable.VType) in {GRB.BINARY, GRB.INTEGER}:
            integrality_residual = abs(value - round(value))
            max_residual = max(max_residual, integrality_residual)
            if integrality_residual > tolerance:
                integrality_violations.append(name)

    constraint_violations: list[str] = []
    if not missing and not nonfinite:
        for constraint in model.getConstrs():
            row = model.getRow(constraint)
            lhs = _linear_value(row, normalized)
            residual = _sense_residual(lhs, str(constraint.Sense), float(constraint.RHS))
            max_residual = max(max_residual, residual)
            if residual > tolerance:
                constraint_violations.append(str(constraint.ConstrName))

    unsupported_types: set[int] = set()
    if not missing and not nonfinite:
        for constraint in model.getGenConstrs():
            constraint_type = int(constraint.GenConstrType)
            if constraint_type != int(GRB.GENCONSTR_INDICATOR):
                unsupported_types.add(constraint_type)
                continue
            binvar, binval, expression, sense, rhs = model.getGenConstrIndicator(constraint)
            trigger_value = float(normalized[str(binvar.VarName)])
            if abs(trigger_value - float(bool(binval))) > tolerance:
                continue
            lhs = _linear_value(expression, normalized)
            residual = _sense_residual(lhs, str(sense), float(rhs))
            max_residual = max(max_residual, residual)
            if residual > tolerance:
                constraint_violations.append(str(constraint.GenConstrName))

    complete = not missing and not nonfinite
    feasible = bool(
        complete
        and not bound_violations
        and not integrality_violations
        and not constraint_violations
        and not unsupported_types
    )
    limit = max(1, int(max_reported_names))
    return FullStartValidation(
        complete=bool(complete),
        feasible=bool(feasible),
        variable_count=len(variables),
        supplied_count=len(variable_names & set(normalized)),
        missing_names=tuple(missing[:limit]),
        nonfinite_names=tuple(sorted(nonfinite)[:limit]),
        bound_violation_names=tuple(sorted(bound_violations)[:limit]),
        integrality_violation_names=tuple(sorted(integrality_violations)[:limit]),
        constraint_violation_names=tuple(sorted(set(constraint_violations))[:limit]),
        unsupported_general_constraint_types=tuple(sorted(unsupported_types)),
        max_residual=float(max_residual),
    )
