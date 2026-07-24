from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InnerAttemptTrace:
    phase: str
    attempt_index: int
    search_seed: int
    vns_seed_sha256: str
    requested_time_limit_sec: float
    runtime_sec: float
    solver_status_code: int
    candidate_count_before: int
    candidate_count_after: int

    def as_audit_payload(self) -> dict[str, object]:
        return {
            "phase": str(self.phase),
            "attempt_index": int(self.attempt_index),
            "search_seed": int(self.search_seed),
            "vns_seed_sha256": str(self.vns_seed_sha256),
            "requested_time_limit_sec": float(self.requested_time_limit_sec),
            "runtime_sec": float(self.runtime_sec),
            "solver_status_code": int(self.solver_status_code),
            "candidate_count_before": int(self.candidate_count_before),
            "candidate_count_after": int(self.candidate_count_after),
        }
