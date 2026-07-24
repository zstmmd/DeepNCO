from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

import gurobipy as gp
from gurobipy import GRB

from Gurobi.tra_elite import candidate_preference_key
from Gurobi.tra_inner_search import (
    PHASE_TWO_BASE_OBJECTIVE_TIEBREAK,
    configure_inner_search,
    phase_one_time_limit,
    phase_two_attempt_limit,
    phase_two_pool_complete,
    phase_two_quality_limit,
    phase_two_recourse_objective,
    phase_two_round_complete,
    phase_two_round_time_limit,
    phase_two_search_seed,
    project_inner_start,
    should_run_phase_two,
)
from Gurobi.tra_inner_trace import InnerAttemptTrace
from Gurobi.tra_local_branching import add_shell_exclusion, apply_local_neighborhood
from Gurobi.tra_model_state import ModelSnapshot, PersistentCompiledTemplate, TemplateStateError
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure, validate_transition
from Gurobi.tra_projection import ProjectionRegistry, StructuralShell
from Gurobi.tra_risk import RepairRisk, compute_repair_risk

if TYPE_CHECKING:
    from Gurobi.tra_comproc.types import ComProcResult


@dataclass(frozen=True)
class InnerCandidate:
    shell: StructuralShell
    snapshot: ModelSnapshot
    relaxed_objective: float
    repair_risk: RepairRisk
    comproc: Optional["ComProcResult"] = None


@dataclass(frozen=True)
class InnerSolveResult:
    status: str
    procedure: Procedure
    neighborhood: NeighborhoodLevel
    runtime_sec: float
    solver_status_code: int
    certified_obj_bound: float
    candidates: tuple[InnerCandidate, ...]
    attempt_count: int = 1
    search_seeds: tuple[int, ...] = ()
    vns_seed_sha256: tuple[str, ...] = ()
    attempt_traces: tuple[InnerAttemptTrace, ...] = ()
    error: str = ""


def _snapshot_from_solution(template: PersistentCompiledTemplate, runtime_sec: float) -> ModelSnapshot:
    values = {str(variable.VarName): float(variable.X) for variable in template.model.getVars()}
    cmax_var = template.payload["cmax"]
    return ModelSnapshot(
        values_by_name=values,
        solver_objective=float(template.model.ObjVal),
        solver_cmax=float(values[str(cmax_var.VarName)]),
        callback_runtime_sec=float(runtime_sec),
    )


def _linear_objective_value(expression: Any, values_by_name: Mapping[str, float]) -> float:
    value = float(expression.getConstant())
    for index in range(int(expression.size())):
        variable = expression.getVar(index)
        value += float(expression.getCoeff(index)) * float(
            values_by_name[str(variable.VarName)]
        )
    return float(value)


class PaperInnerTemplate:
    def __init__(self, template: PersistentCompiledTemplate, *, elite_pool_size: int = 24) -> None:
        if not bool(getattr(template.compiled.cfg, "tra_inner_no_station_wait", False)):
            raise TemplateStateError("paper inner template must be compiled with no station waiting")
        self.template = template
        self.elite_pool_size = max(1, min(32, int(elite_pool_size)))

    def solve(
        self,
        incumbent: StructuralShell,
        *,
        procedure: Procedure,
        neighborhood: NeighborhoodLevel,
        time_limit_sec: float,
        incumbent_objective: Optional[float],
        start_values: Optional[Mapping[str, float]] = None,
        search_seed: Optional[int] = None,
        vns_start_values: Sequence[Mapping[str, float]] = (),
        vns_seed_sha256: Sequence[str] = (),
        incumbent_cmax: Optional[float] = None,
    ) -> InnerSolveResult:
        procedure = Procedure(procedure)
        neighborhood = NeighborhoodLevel(neighborhood)
        started = time.perf_counter()
        candidates_by_hash: Dict[str, InnerCandidate] = {}
        model = self.template.model
        phase_two_active = False
        phase_two_round_start_count = 0
        phase_two_stop_after_new_candidate = False
        attempt_traces: list[InnerAttemptTrace] = []
        base_objective = None
        base_objective_sense = int(model.ModelSense)
        objective_replaced = False

        def retain_candidate(candidate: InnerCandidate) -> None:
            current = candidates_by_hash.get(candidate.shell.sha256)
            if current is None or candidate_preference_key(candidate) < candidate_preference_key(
                current
            ):
                candidates_by_hash[candidate.shell.sha256] = candidate
            if len(candidates_by_hash) <= self.elite_pool_size:
                return
            retained = sorted(
                candidates_by_hash.values(),
                key=candidate_preference_key,
            )[: self.elite_pool_size]
            candidates_by_hash.clear()
            candidates_by_hash.update(
                (item.shell.sha256, item)
                for item in retained
            )

        try:
            self.template.reset_for_solve()
            configure_inner_search(model)
            base_objective = gp.LinExpr(model.getObjective())
            base_objective_sense = int(model.ModelSense)
            projected_start = project_inner_start(
                start_values or {},
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
            if search_seed is not None:
                model.Params.Seed = int(search_seed)
            apply_local_neighborhood(
                self.template,
                incumbent,
                procedure,
                neighborhood,
            )

            tolerance = max(
                1e-6,
                1e-8 * max(1.0, abs(float(incumbent_objective or 0.0))),
            )
            primary_incumbent = (
                incumbent_cmax is not None
                and math.isfinite(float(incumbent_cmax))
            )
            if not primary_incumbent:
                self.template.set_internal_cutoff(incumbent_objective, tolerance)
            elapsed = time.perf_counter() - started
            total_remaining = float(time_limit_sec) - elapsed
            phase_one_remaining = min(
                total_remaining,
                max(1e-3, phase_one_time_limit(time_limit_sec) - elapsed),
            )
            self.template.set_time_limit(phase_one_remaining)
            model.update()

            def callback(callback_model: Any, where: int) -> None:
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    callback_runtime = float(time.perf_counter() - started)
                    snapshot = self.template.snapshot_from_callback(callback_runtime)
                    if phase_two_active and base_objective is not None:
                        snapshot = replace(
                            snapshot,
                            solver_objective=_linear_objective_value(
                                base_objective,
                                snapshot.values_by_name,
                            ),
                        )
                    registry = ProjectionRegistry.from_payload(
                        self.template.payload,
                        value_getter=lambda variable: snapshot.values_by_name[str(variable.VarName)],
                    )
                    shell = StructuralShell.extract(registry)
                    validate_transition(incumbent.projection, shell.projection, procedure, neighborhood)
                    risk = compute_repair_risk(shell, snapshot, self.template.payload, incumbent)
                    candidate = InnerCandidate(
                        shell=shell,
                        snapshot=snapshot,
                        relaxed_objective=float(snapshot.solver_objective),
                        repair_risk=risk,
                    )
                    retain_candidate(candidate)
                    if phase_two_active and phase_two_round_complete(
                        candidate_count=len(candidates_by_hash),
                        round_start_count=phase_two_round_start_count,
                        stop_after_new_candidate=phase_two_stop_after_new_candidate,
                    ):
                        callback_model.terminate()
                except Exception:
                    return

            def capture_final_solution() -> None:
                if int(model.SolCount) <= 0:
                    return
                runtime_now = float(time.perf_counter() - started)
                final_snapshot = _snapshot_from_solution(self.template, runtime_now)
                if phase_two_active and base_objective is not None:
                    final_snapshot = replace(
                        final_snapshot,
                        solver_objective=_linear_objective_value(
                            base_objective,
                            final_snapshot.values_by_name,
                        ),
                    )
                registry = ProjectionRegistry.from_payload(
                    self.template.payload,
                    value_getter=lambda variable: final_snapshot.values_by_name[str(variable.VarName)],
                )
                final_shell = StructuralShell.extract(registry)
                validate_transition(incumbent.projection, final_shell.projection, procedure, neighborhood)
                final_candidate = InnerCandidate(
                    shell=final_shell,
                    snapshot=final_snapshot,
                    relaxed_objective=float(final_snapshot.solver_objective),
                    repair_risk=compute_repair_risk(
                        final_shell,
                        final_snapshot,
                        self.template.payload,
                        incumbent,
                    ),
                )
                retain_candidate(final_candidate)

            phase_one_candidate_count = len(candidates_by_hash)
            phase_one_started = time.perf_counter()
            model.optimize(callback)
            capture_final_solution()
            attempt_traces.append(
                InnerAttemptTrace(
                    phase="phase_one",
                    attempt_index=1,
                    search_seed=int(model.Params.Seed),
                    vns_seed_sha256=(
                        str(vns_seed_sha256[0])
                        if vns_seed_sha256
                        else ""
                    ),
                    requested_time_limit_sec=float(phase_one_remaining),
                    runtime_sec=float(time.perf_counter() - phase_one_started),
                    solver_status_code=int(model.Status),
                    candidate_count_before=phase_one_candidate_count,
                    candidate_count_after=len(candidates_by_hash),
                )
            )
            try:
                phase_one_bound = float(model.ObjBound)
            except Exception:
                phase_one_bound = float("nan")
            search_seeds = [int(model.Params.Seed)]
            attempt_count = 1

            remaining = float(time_limit_sec) - float(time.perf_counter() - started)
            if should_run_phase_two(
                candidate_count=len(candidates_by_hash),
                phase_one_timed_out=int(model.Status) == GRB.TIME_LIMIT,
                remaining_sec=remaining,
            ):
                best_relaxed = (
                    min(
                        float(candidate.relaxed_objective)
                        for candidate in candidates_by_hash.values()
                    )
                    if candidates_by_hash
                    else None
                )
                quality_limit = phase_two_quality_limit(
                    best_relaxed,
                    incumbent_objective,
                )
                if quality_limit is not None:
                    self.template.add_constraint(
                        base_objective <= float(quality_limit),
                        name="TRA_Phase2_RelaxedQualityBand",
                    )
                recourse_objective = (
                    phase_two_recourse_objective(
                        self.template.payload,
                        incumbent,
                        procedure,
                        neighborhood,
                    )
                    if incumbent_objective is not None
                    and math.isfinite(float(incumbent_objective))
                    else None
                )
                recourse_active = bool(
                    recourse_objective is not None
                    and quality_limit is not None
                )
                if recourse_active:
                    model.setObjective(
                        recourse_objective
                        + PHASE_TWO_BASE_OBJECTIVE_TIEBREAK * base_objective,
                        GRB.MINIMIZE,
                    )
                    objective_replaced = True
                attempt_limit = phase_two_attempt_limit(
                    procedure,
                    neighborhood,
                    vns_seed_count=len(vns_start_values),
                    recourse_active=recourse_active,
                )
                excluded_shells: set[str] = set()
                exclusion_index = 0
                phase_two_active = True
                for phase_two_attempt in range(1, attempt_limit + 1):
                    remaining = float(time_limit_sec) - float(
                        time.perf_counter() - started
                    )
                    if remaining <= 1e-3:
                        break
                    for candidate in sorted(
                        candidates_by_hash.values(),
                        key=lambda item: item.shell.sha256,
                    ):
                        if candidate.shell.sha256 in excluded_shells:
                            continue
                        add_shell_exclusion(
                            self.template,
                            candidate.shell,
                            procedure.released_block,
                            index=exclusion_index,
                        )
                        excluded_shells.add(candidate.shell.sha256)
                        exclusion_index += 1

                    model.update()
                    model.reset(0)
                    configure_inner_search(model)
                    phase_two_seed = phase_two_search_seed(
                        int(search_seeds[0]),
                        phase_two_attempt=phase_two_attempt,
                    )
                    model.Params.Seed = int(phase_two_seed)
                    self.template.install_start(
                        projected_start,
                        clear_existing=True,
                    )
                    vns_index = phase_two_attempt
                    if (
                        vns_index < len(vns_start_values)
                        and procedure is not Procedure.F1
                    ):
                        self.template.install_start(
                            vns_start_values[vns_index],
                            clear_existing=False,
                        )
                    search_seeds.append(int(phase_two_seed))
                    phase_two_round_start_count = len(candidates_by_hash)
                    phase_two_stop_after_new_candidate = attempt_limit > 1
                    round_limit = phase_two_round_time_limit(
                        remaining,
                        rounds_remaining=attempt_limit - phase_two_attempt + 1,
                    )
                    if round_limit <= 1e-3:
                        break
                    self.template.set_time_limit(round_limit)
                    round_started = time.perf_counter()
                    model.optimize(callback)
                    capture_final_solution()
                    attempt_count += 1
                    attempt_traces.append(
                        InnerAttemptTrace(
                            phase="phase_two",
                            attempt_index=phase_two_attempt,
                            search_seed=int(phase_two_seed),
                            vns_seed_sha256=(
                                str(vns_seed_sha256[vns_index])
                                if vns_index < len(vns_seed_sha256)
                                else ""
                            ),
                            requested_time_limit_sec=float(round_limit),
                            runtime_sec=float(time.perf_counter() - round_started),
                            solver_status_code=int(model.Status),
                            candidate_count_before=phase_two_round_start_count,
                            candidate_count_after=len(candidates_by_hash),
                        )
                    )
                    new_candidate_found = (
                        len(candidates_by_hash) > phase_two_round_start_count
                    )
                    if phase_two_pool_complete(len(candidates_by_hash)):
                        break
                    if (
                        not new_candidate_found
                        and int(model.Status) in (GRB.OPTIMAL, GRB.INFEASIBLE)
                    ):
                        break

            runtime = float(time.perf_counter() - started)
            status = self.template.solver._status_label(int(model.Status))
            candidates = tuple(
                sorted(
                    candidates_by_hash.values(),
                    key=candidate_preference_key,
                )[: self.elite_pool_size]
            )
            return InnerSolveResult(
                status=status,
                procedure=procedure,
                neighborhood=neighborhood,
                runtime_sec=runtime,
                solver_status_code=int(model.Status),
                certified_obj_bound=phase_one_bound,
                candidates=candidates,
                attempt_count=attempt_count,
                search_seeds=tuple(search_seeds),
                vns_seed_sha256=tuple(str(value) for value in vns_seed_sha256),
                attempt_traces=tuple(attempt_traces),
            )
        except Exception as exc:
            runtime = float(time.perf_counter() - started)
            return InnerSolveResult(
                status="BUDGET_EXHAUSTED" if isinstance(exc, TemplateStateError) else "INNER_FAILED",
                procedure=procedure,
                neighborhood=neighborhood,
                runtime_sec=runtime,
                solver_status_code=int(getattr(model, "Status", 0) or 0),
                certified_obj_bound=float("nan"),
                candidates=(),
                search_seeds=(int(getattr(model.Params, "Seed", 0)),),
                vns_seed_sha256=tuple(str(value) for value in vns_seed_sha256),
                attempt_traces=tuple(attempt_traces),
                error=str(exc),
            )
        finally:
            if objective_replaced and base_objective is not None:
                try:
                    model.setObjective(base_objective, base_objective_sense)
                    model.update()
                except Exception:
                    pass
