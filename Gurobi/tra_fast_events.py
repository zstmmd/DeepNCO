from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from Gurobi.tra_events import FeasibleSolutionEvent
from Gurobi.tra_scheduler import ProcedureStep
from Gurobi.tra_verifier import VerifiedSnapshot


@dataclass(frozen=True)
class TRAFastEngineResult:
    run_id: str
    case: str
    status: str
    runtime_sec: float
    procedure_count: int
    cycle_count: int
    event_count: int
    best_objective: float
    best_verified_cmax: float
    manifest_sha256: str
    regular_runtime_sec: float
    reserve_runtime_sec: float
    error: str = ""


def build_fast_event(
    *,
    manifest: Mapping[str, object],
    run_id: str,
    case: str,
    elapsed_sec: float,
    verified: VerifiedSnapshot,
    step: Optional[ProcedureStep],
    solver_timestamp_sec: float,
    source: str,
) -> FeasibleSolutionEvent:
    fingerprints = dict(manifest["model_fingerprints"])  # type: ignore[arg-type]
    return FeasibleSolutionEvent(
        run_id=run_id,
        case=case,
        wall_timestamp_sec=float(elapsed_sec),
        solver_incumbent_timestamp_sec=float(solver_timestamp_sec),
        cycle=int(step.cycle if step else 0),
        procedure=str(step.procedure.value if step else "WARM"),
        neighborhood=str(step.neighborhood.value if step else "CANONICAL"),
        manifest_sha256=str(manifest["manifest_sha256"]),
        objective_sha256=str(fingerprints["objective_sha256"]),
        structural_hash=verified.shell.sha256,
        solver_objective=float(verified.snapshot.solver_objective),
        solver_cmax=float(verified.snapshot.solver_cmax),
        verified_cmax=float(verified.verified_cmax),
        internal_feasible=True,
        verifier_error_codes=verified.verifier_error_codes,
        provenance={"source": str(source), "algorithm": "paper-tra-fast"},
        snapshot_sha256=verified.snapshot_sha256,
        structural_projection=verified.shell.projection.as_canonical_payload(),
    )
