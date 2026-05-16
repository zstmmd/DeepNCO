from __future__ import annotations

import math
from typing import Dict, List, Sequence

from BPC.models import BPCRouteColumn, BPCRouteTask, MasterResult

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover - fallback is tested without requiring Gurobi.
    gp = None
    GRB = None


class RestrictedMasterProblem:
    def __init__(self, route_tasks: Sequence[BPCRouteTask], columns: Sequence[BPCRouteColumn]) -> None:
        self.route_tasks = list(route_tasks)
        self.columns = list(columns)
        self.task_keys = sorted(int(task.task_key) for task in self.route_tasks)

    def solve_relaxation(self, output: bool = False) -> MasterResult:
        if gp is None:
            return self._fallback_cover_bound()
        try:
            model = gp.Model("BPC_RMP")
        except Exception:
            return self._fallback_cover_bound()
        model.Params.OutputFlag = 1 if bool(output) else 0
        lamb = {
            int(col.column_id): model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"lambda_{int(col.column_id)}")
            for col in self.columns
        }
        cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")
        cover_constr = {}
        for task_key in self.task_keys:
            relevant = [lamb[int(col.column_id)] for col in self.columns if int(task_key) in set(int(k) for k in col.task_keys)]
            if relevant:
                cover_constr[int(task_key)] = model.addConstr(gp.quicksum(relevant) >= 1.0, name=f"cover_{task_key}")
            else:
                model.addConstr(0.0 >= 1.0, name=f"cover_{task_key}")
        for col in self.columns:
            model.addConstr(cmax >= float(col.finish_time) * lamb[int(col.column_id)], name=f"cmax_{int(col.column_id)}")
        model.setObjective(cmax, GRB.MINIMIZE)
        model.optimize()
        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return MasterResult(status=str(model.Status), objective=float("inf"), lower_bound=float("inf"))
        selected = {
            int(col_id): float(var.X)
            for col_id, var in lamb.items()
            if float(var.X) > 1e-9
        }
        duals = {
            int(task_key): float(constr.Pi)
            for task_key, constr in cover_constr.items()
        }
        integer = all(abs(value - round(value)) <= 1e-9 for value in selected.values())
        return MasterResult(
            status="OPTIMAL" if model.Status == GRB.OPTIMAL else "SUBOPTIMAL",
            objective=float(model.ObjVal),
            lower_bound=float(model.ObjBound if math.isfinite(model.ObjBound) else model.ObjVal),
            dual_task_cover=duals,
            selected_columns=selected,
            integer=bool(integer),
        )

    def _fallback_cover_bound(self) -> MasterResult:
        selected: Dict[int, float] = {}
        max_finish = 0.0
        covered = set()
        for task_key in self.task_keys:
            candidates = [col for col in self.columns if int(task_key) in set(int(k) for k in col.task_keys)]
            if not candidates:
                return MasterResult(status="INFEASIBLE", objective=float("inf"), lower_bound=float("inf"))
            best = min(candidates, key=lambda col: (float(col.finish_time), int(col.column_id)))
            selected[int(best.column_id)] = 1.0
            covered.update(int(k) for k in best.task_keys)
            max_finish = max(max_finish, float(best.finish_time))
        return MasterResult(
            status="FALLBACK",
            objective=float(max_finish),
            lower_bound=0.0,
            dual_task_cover={int(k): 0.0 for k in self.task_keys},
            selected_columns=selected,
            integer=True,
        )

    def active_task_coverage(self, selected_columns: Dict[int, float] | None = None) -> Dict[int, float]:
        selected_columns = selected_columns or {}
        by_id = {int(col.column_id): col for col in self.columns}
        coverage = {int(key): 0.0 for key in self.task_keys}
        for col_id, value in selected_columns.items():
            col = by_id.get(int(col_id))
            if col is None:
                continue
            for task_key in col.task_keys:
                coverage[int(task_key)] = float(coverage.get(int(task_key), 0.0) + float(value))
        return coverage
