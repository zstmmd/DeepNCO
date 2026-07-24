from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable, Mapping

from Gurobi.tra_comproc.dp1 import evaluate_dp1_route
from Gurobi.tra_comproc.dp2 import evaluate_dp2_service
from Gurobi.tra_comproc.dp3 import evaluate_dp3_recovery
from Gurobi.tra_comproc.ranking import comproc_candidate_key
from Gurobi.tra_comproc.types import ComProcResult
from Gurobi.tra_model_state import ModelSnapshot
from Gurobi.tra_outer_start import OuterStartProjectionError, build_full_start_vector
from Gurobi.tra_verifier import SnapshotVerifier


class ComProcEvaluator:
    """Paper-style DP1/DP2/DP3 evaluation against an untouched full model."""

    def __init__(
        self,
        model: Any,
        payload: Mapping[str, Any],
        verifier: SnapshotVerifier,
    ) -> None:
        self.model = model
        self.payload = payload
        self.verifier = verifier

    def evaluate(self, candidate: Any, *, source: str = "inner_mipsol") -> Any:
        values = candidate.snapshot.values_by_name
        dp1 = evaluate_dp1_route(values, self.payload)
        dp2 = evaluate_dp2_service(values, self.payload, candidate.shell, dp1)
        dp3 = evaluate_dp3_recovery(
            dp2,
            no_wait_cmax_floor=float(candidate.snapshot.solver_cmax),
        )
        full_start = None
        errors = list(dp1.error_codes) + list(dp2.error_codes) + list(dp3.error_codes)
        projected_cmax = float("inf")
        verified_cmax = float("nan")
        projected_objective = float("inf")
        residual = float("inf")
        try:
            full_start = build_full_start_vector(
                self.model,
                values,
                self.payload,
                candidate.shell,
            )
            projected_cmax = float(full_start.projection.projected_cmax)
            projected_objective = float(
                sum(
                    float(variable.Obj)
                    * float(full_start.values_by_name[str(variable.VarName)])
                    for variable in self.model.getVars()
                )
            )
            residual = float(full_start.validation.max_residual)
            errors.extend(full_start.validation.error_codes)
            if not math.isclose(
                float(full_start.projection.projected_cmax),
                float(dp3.feasible_start_cmax),
                rel_tol=1e-8,
                abs_tol=1e-6,
            ):
                errors.append("DP3_START_CMAX_MISMATCH")
            cmax_var = self.payload["cmax"]
            semantic = self.verifier.verify(
                ModelSnapshot(
                    values_by_name=full_start.values_by_name,
                    solver_objective=projected_objective,
                    solver_cmax=float(
                        full_start.values_by_name[str(cmax_var.VarName)]
                    ),
                    callback_runtime_sec=float(candidate.snapshot.callback_runtime_sec),
                )
            )
            verified_cmax = float(semantic.verified_cmax)
            errors.extend(
                f"DP3_SEMANTIC:{code}"
                for code in semantic.verifier_error_codes
            )
        except OuterStartProjectionError as exc:
            errors.append(f"DP3_PROJECTION:{exc}")
        feasible = bool(
            dp1.feasible
            and dp2.feasible
            and dp3.feasible
            and full_start is not None
            and full_start.validation.feasible
            and math.isfinite(verified_cmax)
            and not any(code.startswith("DP3_SEMANTIC:") for code in errors)
            and math.isfinite(projected_objective)
        )
        result = ComProcResult(
            feasible=feasible,
            projected_cmax=projected_cmax,
            recourse_score=float(dp3.recourse_score),
            verified_cmax=verified_cmax,
            projected_objective=projected_objective,
            feasibility_residual=residual,
            source=str(source),
            dp1=dp1,
            dp2=dp2,
            dp3=dp3,
            full_start=full_start,
            error_codes=tuple(sorted(set(errors))),
        )
        return replace(candidate, comproc=result)

    def evaluate_many(self, candidates: Iterable[Any], *, source: str = "inner_mipsol") -> tuple[Any, ...]:
        evaluated = [self.evaluate(candidate, source=source) for candidate in candidates]
        return tuple(sorted(evaluated, key=comproc_candidate_key))
