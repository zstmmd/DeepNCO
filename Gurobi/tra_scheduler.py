from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure


@dataclass(frozen=True)
class ProcedureStep:
    procedure_index: int
    cycle: int
    procedure: Procedure
    neighborhood: NeighborhoodLevel


@dataclass
class RuntimeLedger:
    hard_limit_sec: float
    inner_quota_sec: float
    outer_quota_sec: float
    reserve_quota_sec: float
    safety_buffer_sec: float = 0.0
    minimum_solver_slice_sec: float = 0.0
    clock: callable = time.perf_counter
    started_at: Optional[float] = None
    inner_used_sec: float = 0.0
    outer_used_sec: float = 0.0
    reserve_used_sec: float = 0.0

    def start(self) -> None:
        if self.started_at is not None:
            raise RuntimeError("formal runtime ledger has already started")
        self.started_at = float(self.clock())

    @property
    def elapsed_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, float(self.clock()) - float(self.started_at))

    @property
    def remaining_sec(self) -> float:
        return max(0.0, float(self.hard_limit_sec) - self.elapsed_sec)

    @property
    def allocatable_remaining_sec(self) -> float:
        return max(0.0, self.remaining_sec - max(0.0, float(self.safety_buffer_sec)))

    def record(self, bucket: str, runtime_sec: float) -> None:
        value = max(0.0, float(runtime_sec))
        if bucket == "inner":
            self.inner_used_sec += value
        elif bucket == "outer":
            self.outer_used_sec += value
        elif bucket == "reserve":
            self.reserve_used_sec += value
        else:
            raise ValueError(f"unknown runtime bucket: {bucket}")

    def slice_for(
        self,
        bucket: str,
        remaining_regular_steps: int,
        *,
        borrow_unused: bool = False,
    ) -> float:
        if self.started_at is None:
            raise RuntimeError("formal runtime ledger has not started")
        if bucket == "inner":
            quota_remaining = max(0.0, self.inner_quota_sec - self.inner_used_sec)
        elif bucket == "outer":
            quota_remaining = max(0.0, self.outer_quota_sec - self.outer_used_sec)
        elif bucket == "reserve":
            quota_remaining = max(0.0, self.reserve_quota_sec - self.reserve_used_sec)
        else:
            raise ValueError(f"unknown runtime bucket: {bucket}")
        if borrow_unused:
            if bucket != "reserve":
                raise ValueError("only the post-stagnation reserve may borrow unused soft quota")
            quota_remaining += max(0.0, self.inner_quota_sec - self.inner_used_sec)
            quota_remaining += max(0.0, self.outer_quota_sec - self.outer_used_sec)
            quota_remaining = min(self.allocatable_remaining_sec, quota_remaining)
        divisor = max(1, int(remaining_regular_steps))
        suggested = quota_remaining / divisor
        if suggested <= 0.0 and self.allocatable_remaining_sec > 0.0:
            fallback_share = {"inner": 0.30, "outer": 0.55, "reserve": 0.15}[bucket]
            suggested = fallback_share * self.allocatable_remaining_sec / divisor
        allocation = max(0.0, min(self.allocatable_remaining_sec, suggested))
        minimum_slice = max(0.0, float(self.minimum_solver_slice_sec))
        if 0.0 < allocation < minimum_slice:
            if self.allocatable_remaining_sec >= minimum_slice:
                return minimum_slice
            return 0.0
        return allocation


@dataclass
class RetryRegistry:
    unresolved_attempts: Dict[str, int] = field(default_factory=dict)

    def register_unresolved(self, shell_hash: str) -> bool:
        count = int(self.unresolved_attempts.get(str(shell_hash), 0))
        self.unresolved_attempts[str(shell_hash)] = count + 1
        return count == 0

    def can_retry(self, shell_hash: str) -> bool:
        return int(self.unresolved_attempts.get(str(shell_hash), 0)) == 1

    def mark_retried(self, shell_hash: str) -> None:
        if not self.can_retry(shell_hash):
            raise RuntimeError(f"shell {shell_hash} does not have exactly one unresolved first attempt")
        self.unresolved_attempts[str(shell_hash)] = 2


class RotationScheduler:
    ORDER = (Procedure.F1, Procedure.F2, Procedure.F3)

    def __init__(self, *, max_procedures: int = 50, stagnant_cycle_limit: int = 3) -> None:
        self.max_procedures = int(max_procedures)
        self.stagnant_cycle_limit = int(stagnant_cycle_limit)
        self.procedure_count = 0
        self.cycle = 1
        self._position = 0
        self._cycle_improved = False
        self.stagnant_cycles = 0
        self._level_by_procedure = {procedure: NeighborhoodLevel.N1 for procedure in self.ORDER}

    def current_step(self) -> ProcedureStep:
        procedure = self.ORDER[self._position]
        return ProcedureStep(
            procedure_index=self.procedure_count + 1,
            cycle=self.cycle,
            procedure=procedure,
            neighborhood=self._level_by_procedure[procedure],
        )

    def complete_step(
        self,
        *,
        improved: bool,
        primary_improved: Optional[bool] = None,
    ) -> None:
        procedure = self.ORDER[self._position]
        self.procedure_count += 1
        primary_change = bool(improved) if primary_improved is None else bool(primary_improved)
        if primary_change:
            self._cycle_improved = True
            self._level_by_procedure = {item: NeighborhoodLevel.N1 for item in self.ORDER}
        elif improved:
            self._cycle_improved = True
            for item in self.ORDER:
                if item is procedure:
                    self._level_by_procedure[item] = NeighborhoodLevel.N1
                elif self._level_by_procedure[item] is NeighborhoodLevel.N3:
                    self._level_by_procedure[item] = NeighborhoodLevel.N2
        else:
            current = self._level_by_procedure[procedure]
            self._level_by_procedure[procedure] = {
                NeighborhoodLevel.N1: NeighborhoodLevel.N2,
                NeighborhoodLevel.N2: NeighborhoodLevel.N3,
                NeighborhoodLevel.N3: NeighborhoodLevel.N3,
            }[current]

        self._position = (self._position + 1) % len(self.ORDER)
        if self._position == 0:
            if self._cycle_improved:
                self.stagnant_cycles = 0
            else:
                self.stagnant_cycles += 1
            self._cycle_improved = False
            self.cycle += 1

    def should_stop(self, *, runtime_remaining_sec: float, deferred_empty: bool) -> bool:
        if float(runtime_remaining_sec) <= 0.0:
            return True
        if self.procedure_count >= self.max_procedures:
            return True
        return bool(deferred_empty) and self.stagnant_cycles >= self.stagnant_cycle_limit

    def restart_after_external_improvement(self) -> None:
        """Resume a strict F1/F2/F3 cycle after a reserve-phase incumbent change."""

        if self._position != 0:
            self.cycle += 1
        self._position = 0
        self._cycle_improved = False
        self.stagnant_cycles = 0
        self._level_by_procedure = {item: NeighborhoodLevel.N1 for item in self.ORDER}

    @property
    def remaining_regular_steps(self) -> int:
        return max(0, self.max_procedures - self.procedure_count)
