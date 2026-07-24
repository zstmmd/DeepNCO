from __future__ import annotations

from Gurobi.tra_outer import OuterSolveResult, has_unresolved_improvement_potential
from Gurobi.tra_scheduler import ProcedureStep
from Gurobi.tra_search_state import SearchState
from Gurobi.tra_work_queue import PendingOuterShell


def retain_accepted_shell_refinement(
    state: SearchState,
    result: OuterSolveResult,
    *,
    step: ProcedureStep,
    relaxed_objective: float,
    repair_risk_total: float,
    allow_retry: bool,
) -> bool:
    accepted = result.accepted
    if not allow_retry or accepted is None:
        return False
    if not has_unresolved_improvement_potential(
        solver_status_code=result.solver_status_code,
        objective_bound=result.objective_bound,
        accepted_objective=accepted.snapshot.solver_objective,
    ):
        return False
    shell_hash = str(accepted.shell.sha256)
    if not state.retry_registry.register_unresolved(shell_hash):
        return False
    return state.queues.add_pending(
        PendingOuterShell(
            shell=accepted.shell,
            start_values=dict(accepted.snapshot.values_by_name),
            step=step,
            reserve_retry=True,
            relaxed_objective=float(relaxed_objective),
            repair_risk_total=float(repair_risk_total),
            validation_bound=float(result.objective_bound),
            accepted_refinement=True,
            projected_cmax=float(accepted.verified_cmax),
            projected_objective=float(accepted.snapshot.solver_objective),
            start_feasible=True,
        )
    )
