from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from gurobipy import GRB

from Gurobi.tra_outer import objective_tolerance
from Gurobi.tra_projection import INACTIVE_LABEL


def f3_support_expansion_needed(
    assignments: Mapping[int, int],
    robot_labels: Iterable[int],
    *,
    dominance_ratio: float = 1.5,
) -> bool:
    """Detect target-blind pressure to move work off a dominant robot."""

    labels = tuple(
        sorted(
            {
                int(label)
                for label in robot_labels
                if int(label) != INACTIVE_LABEL
            }
        )
    )
    if len(labels) <= 1:
        return False
    counts = Counter(
        int(robot_id)
        for robot_id in assignments.values()
        if int(robot_id) in labels
    )
    active_count = sum(int(counts[label]) for label in labels)
    if active_count <= 0:
        return False
    if any(int(counts[label]) == 0 for label in labels):
        return True
    mean_load = float(active_count) / float(len(labels))
    dominant_limit = int(math.ceil(max(1.0, float(dominance_ratio)) * mean_load))
    return max(int(counts[label]) for label in labels) > dominant_limit

@dataclass(frozen=True)
class OuterBudgetPolicy:
    initial_hard_fraction: float = 0.04
    continuation_hard_fraction: float = 0.04
    restart_hard_fraction: float = 0.04

    @staticmethod
    def _cap(suggested_sec: float, hard_limit_sec: float, fraction: float) -> float:
        suggested = max(0.0, float(suggested_sec))
        hard_cap = max(1.0, max(0.0, float(hard_limit_sec)) * max(0.0, float(fraction)))
        return min(suggested, hard_cap)

    def initial_slice(self, suggested_sec: float, *, hard_limit_sec: float) -> float:
        return self._cap(suggested_sec, hard_limit_sec, self.initial_hard_fraction)

    def continuation_slice(self, suggested_sec: float, *, hard_limit_sec: float) -> float:
        return self._cap(suggested_sec, hard_limit_sec, self.continuation_hard_fraction)

    def restart_slice(self, suggested_sec: float, *, hard_limit_sec: float) -> float:
        return self._cap(suggested_sec, hard_limit_sec, self.restart_hard_fraction)

    @staticmethod
    def retry_is_bound_promoted(
        *,
        objective_bound: float,
        incumbent_objective: Optional[float],
    ) -> bool:
        if incumbent_objective is None:
            return False
        if not math.isfinite(float(objective_bound)):
            return False
        if not math.isfinite(float(incumbent_objective)):
            return False
        return bool(
            float(objective_bound)
            < float(incumbent_objective)
            - objective_tolerance(float(incumbent_objective))
        )

    def should_continue(
        self,
        result: Any,
        *,
        incumbent_objective: Optional[float],
        incumbent_cmax: Optional[float],
        projected_cmax: float,
    ) -> bool:
        if int(result.solver_status_code) != GRB.TIME_LIMIT:
            return False
        if bool(getattr(result, "resumed_search", False)):
            return False
        bound = float(result.objective_bound)
        bound_promising = (
            incumbent_objective is None
            or not math.isfinite(float(incumbent_objective))
            or not math.isfinite(bound)
            or bound < float(incumbent_objective)
        )
        projection_promising = (
            incumbent_cmax is None
            or not math.isfinite(float(incumbent_cmax))
            or (
                math.isfinite(float(projected_cmax))
                and float(projected_cmax) < float(incumbent_cmax)
            )
        )
        return bool(bound_promising or projection_promising)


@dataclass(frozen=True)
class RegularInnerBudgetPolicy:
    """Stabilize natural candidate pools at expensive rotation boundaries."""

    f3_n1_support_hard_fraction: float = 0.05
    f3_n2_cross_process_hard_fraction: float = 0.025
    f3_n3_balance_hard_fraction: float = 0.075

    def stabilize_slice(
        self,
        suggested_sec: float,
        *,
        hard_limit_sec: float,
        allocatable_remaining_sec: float,
        f3_n1_support_expansion: bool,
        cross_process_f3_n2: bool,
        f3_n3_balance: bool,
    ) -> float:
        suggested = max(0.0, float(suggested_sec))
        remaining = max(0.0, float(allocatable_remaining_sec))
        if (
            not f3_n1_support_expansion
            and not cross_process_f3_n2
            and not f3_n3_balance
        ):
            return min(suggested, remaining)
        hard_fraction = (
            self.f3_n1_support_hard_fraction
            if f3_n1_support_expansion
            else (
                self.f3_n2_cross_process_hard_fraction
                if cross_process_f3_n2
                else self.f3_n3_balance_hard_fraction
            )
        )
        stable_pool_floor = max(
            2.0,
            max(0.0, float(hard_limit_sec))
            * max(0.0, float(hard_fraction)),
        )
        return min(remaining, max(suggested, stable_pool_floor))


@dataclass(frozen=True)
class ReserveBudgetPolicy:
    """Compatibility facade for reserve restart budgeting."""

    restart_hard_fraction: float = 0.04
    promoted_restart_hard_fraction: float = 0.15
    f2_n2_inner_hard_fraction: float = 0.135
    f2_n3_inner_hard_fraction: float = 0.135
    f1_inner_hard_fraction: float = 0.05
    f3_inner_hard_fraction: float = 0.10

    def cap_outer_slice(
        self,
        suggested_sec: float,
        *,
        hard_limit_sec: float,
        reserve_retry: bool,
        bound_promoted: bool = False,
    ) -> float:
        fraction = (
            self.promoted_restart_hard_fraction
            if reserve_retry and bound_promoted
            else self.restart_hard_fraction
        )
        policy = OuterBudgetPolicy(restart_hard_fraction=fraction)
        return policy.restart_slice(suggested_sec, hard_limit_sec=hard_limit_sec)

    def cap_deferred_inner(
        self,
        suggested_sec: float,
        *,
        hard_limit_sec: float,
        procedure: Any,
        neighborhood: Any,
    ) -> float:
        from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure

        suggested = max(0.0, float(suggested_sec))
        current_procedure = Procedure(procedure)
        if current_procedure is Procedure.F1:
            return min(
                suggested,
                max(
                    2.0,
                    max(0.0, float(hard_limit_sec))
                    * max(0.0, float(self.f1_inner_hard_fraction)),
                ),
            )
        if current_procedure is Procedure.F3:
            return min(
                suggested,
                max(
                    2.0,
                    max(0.0, float(hard_limit_sec))
                    * max(0.0, float(self.f3_inner_hard_fraction)),
                ),
            )
        if (
            current_procedure is Procedure.F2
            and NeighborhoodLevel(neighborhood)
            in (NeighborhoodLevel.N2, NeighborhoodLevel.N3)
        ):
            hard_fraction = (
                self.f2_n2_inner_hard_fraction
                if NeighborhoodLevel(neighborhood) is NeighborhoodLevel.N2
                else self.f2_n3_inner_hard_fraction
            )
            cap = max(
                2.0,
                max(0.0, float(hard_limit_sec))
                * max(0.0, float(hard_fraction)),
            )
            return min(suggested, cap)
        return suggested
