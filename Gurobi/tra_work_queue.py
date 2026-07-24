from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from Gurobi.tra_projection import StructuralShell
from Gurobi.tra_scheduler import ProcedureStep


class ReserveStage(str, Enum):
    DEFERRED_INNER = "deferred_inner"
    OUTER = "outer"


@dataclass(frozen=True)
class DeferredInnerStep:
    reference_shell: StructuralShell
    start_values: dict[str, float]
    step: ProcedureStep


@dataclass(frozen=True)
class PendingOuterShell:
    shell: StructuralShell
    start_values: dict[str, float]
    step: ProcedureStep
    reserve_retry: bool
    relaxed_objective: float
    repair_risk_total: float
    validation_bound: float
    accepted_refinement: bool = False
    projected_cmax: float = float("inf")
    projected_objective: float = float("inf")
    start_feasible: bool = False

    @property
    def priority_key(
        self,
    ) -> tuple[int, float, float, float, float, str]:
        bound = float(self.validation_bound)
        if not math.isfinite(bound):
            bound = float(self.relaxed_objective)
        projected_cmax = float(self.projected_cmax)
        if not math.isfinite(projected_cmax):
            projected_cmax = float("inf")
        projected_objective = float(self.projected_objective)
        if not math.isfinite(projected_objective):
            projected_objective = float("inf")
        return (
            0 if self.start_feasible else 1,
            projected_cmax,
            projected_objective,
            float(self.repair_risk_total),
            bound,
            str(self.shell.sha256),
        )


class SearchWorkQueues:
    """Incumbent-scoped deferred work with deterministic deduplication and priority."""

    def __init__(self) -> None:
        self._pending_by_hash: dict[str, PendingOuterShell] = {}
        self._deferred_by_key: dict[tuple[str, str, str], DeferredInnerStep] = {}

    @property
    def empty(self) -> bool:
        return not self._pending_by_hash and not self._deferred_by_key

    @property
    def pending_count(self) -> int:
        return len(self._pending_by_hash)

    @property
    def deferred_count(self) -> int:
        return len(self._deferred_by_key)

    def next_reserve_stage(
        self,
        *,
        prefer_deferred: bool,
        allow_deferred_before_pending: bool = True,
    ) -> ReserveStage:
        if self._deferred_by_key and (
            not self._pending_by_hash
            or (prefer_deferred and allow_deferred_before_pending)
        ):
            return ReserveStage.DEFERRED_INNER
        if self._pending_by_hash:
            return ReserveStage.OUTER
        if self._deferred_by_key:
            return ReserveStage.DEFERRED_INNER
        raise IndexError("reserve work queue is empty")

    def reserve_horizon(self, stage: ReserveStage) -> int:
        stage = ReserveStage(stage)
        if stage is ReserveStage.OUTER:
            next_pending = min(self._pending_by_hash.values(), key=lambda pending: pending.priority_key)
            if not next_pending.reserve_retry:
                # A newly generated shell needs enough uninterrupted repair time.
                # Its one permitted retry remains available if this half-slice is unresolved.
                return 2
            if self._deferred_by_key:
                return 3
            return max(1, min(2, len(self._pending_by_hash)))
        deferred_stages = min(2, len(self._deferred_by_key))
        potential_outer_stage = 1
        competing_outer_stage = int(bool(self._pending_by_hash))
        return max(2, deferred_stages + potential_outer_stage + competing_outer_stage)

    def add_pending(self, item: PendingOuterShell) -> bool:
        shell_hash = str(item.shell.sha256)
        current = self._pending_by_hash.get(shell_hash)
        if current is not None and current.priority_key <= item.priority_key:
            return False
        self._pending_by_hash[shell_hash] = item
        return True

    def pop_pending(self) -> PendingOuterShell:
        item = self.peek_pending()
        self._pending_by_hash.pop(str(item.shell.sha256))
        return item

    def peek_pending(self) -> PendingOuterShell:
        if not self._pending_by_hash:
            raise IndexError("pending outer queue is empty")
        return min(self._pending_by_hash.values(), key=lambda pending: pending.priority_key)

    def add_deferred(self, item: DeferredInnerStep) -> bool:
        key = (
            str(item.reference_shell.sha256),
            str(item.step.procedure.value),
            str(item.step.neighborhood.value),
        )
        if key in self._deferred_by_key:
            return False
        self._deferred_by_key[key] = item
        return True

    def pop_deferred(
        self,
        *,
        priority: Optional[Callable[[DeferredInnerStep], Any]] = None,
    ) -> DeferredInnerStep:
        key, item = min(
            self._deferred_by_key.items(),
            key=lambda pair: (
                priority(pair[1]) if priority is not None else 0,
                int(pair[1].step.procedure_index),
                pair[0],
            ),
        )
        self._deferred_by_key.pop(key)
        return item

    def deferred_items(self) -> tuple[DeferredInnerStep, ...]:
        return tuple(self._deferred_by_key.values())

    def clear(self) -> None:
        self._pending_by_hash.clear()
        self._deferred_by_key.clear()
