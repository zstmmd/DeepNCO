from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from Gurobi.tra_start_validation import validate_full_start


def _model():
    model = gp.Model("start-validation")
    model.Params.OutputFlag = 0
    x = model.addVar(vtype=GRB.BINARY, name="x")
    y = model.addVar(lb=0.0, ub=5.0, name="y")
    model.addConstr(y >= 2.0 * x, name="linear")
    model.addGenConstrIndicator(x, True, y == 3.0, name="indicator")
    model.update()
    return model


def test_complete_start_accepts_linear_and_indicator_constraints() -> None:
    result = validate_full_start(_model(), {"x": 1.0, "y": 3.0})

    assert result.complete
    assert result.feasible
    assert result.error_codes == ()


def test_complete_start_reports_missing_and_constraint_violations() -> None:
    missing = validate_full_start(_model(), {"x": 0.0})
    violated = validate_full_start(_model(), {"x": 1.0, "y": 2.0})

    assert not missing.complete
    assert missing.error_codes == ("START_MISSING_VARIABLES",)
    assert not violated.feasible
    assert violated.constraint_violation_names == ("indicator",)


def test_complete_start_reports_bounds_and_integrality() -> None:
    result = validate_full_start(_model(), {"x": 0.5, "y": 6.0})

    assert not result.feasible
    assert result.bound_violation_names == ("y",)
    assert result.integrality_violation_names == ("x",)
