from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from Gurobi.tra_neighborhood import NeighborhoodLevel

_SHAKE_NEIGHBORHOODS = frozenset(NeighborhoodLevel)


def cmax_tolerance(value: Optional[float]) -> float:
    numeric = float(value or 0.0)
    return max(1e-6, 1e-8 * max(1.0, abs(numeric)))


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    primary_nonworsening: bool
    current_improving: bool
    uphill_shake: bool


@dataclass(frozen=True)
class RecordToRecordPolicy:
    """Target-blind VNS diversification around the best verified Cmax."""

    relative_band: float = 0.03
    absolute_floor: float = 2.0
    max_uphill_shakes: int = 3

    def band_width(self, best_cmax: float) -> float:
        return max(
            float(self.absolute_floor),
            float(self.relative_band) * max(1.0, abs(float(best_cmax))),
        )

    def certification_limit(
        self,
        *,
        best_cmax: Optional[float],
        current_cmax: Optional[float],
        neighborhood: Optional[NeighborhoodLevel],
        uphill_shakes: int,
    ) -> Optional[float]:
        if best_cmax is None:
            return None
        best = float(best_cmax)
        limit = best
        record_limit = best + self.band_width(best)
        if (
            current_cmax is not None
            and float(current_cmax) > best + cmax_tolerance(best)
        ):
            limit = min(float(current_cmax), record_limit)
        if (
            neighborhood is not None
            and NeighborhoodLevel(neighborhood) in _SHAKE_NEIGHBORHOODS
            and int(uphill_shakes) < max(0, int(self.max_uphill_shakes))
        ):
            limit = record_limit
        return float(limit)

    def decide(
        self,
        *,
        best_cmax: Optional[float],
        current_cmax: Optional[float],
        candidate_cmax: float,
        neighborhood: Optional[NeighborhoodLevel],
        uphill_shakes: int,
    ) -> AcceptanceDecision:
        if best_cmax is None:
            return AcceptanceDecision(True, True, True, False)
        tolerance = cmax_tolerance(best_cmax)
        primary_nonworsening = (
            float(candidate_cmax) <= float(best_cmax) + tolerance
        )
        current = float(best_cmax if current_cmax is None else current_cmax)
        current_improving = float(candidate_cmax) < (
            current - cmax_tolerance(current)
        )
        inside_record_band = float(candidate_cmax) <= (
            float(best_cmax) + self.band_width(float(best_cmax)) + tolerance
        )
        uphill_shake = bool(
            inside_record_band
            and not current_improving
            and NeighborhoodLevel(neighborhood) in _SHAKE_NEIGHBORHOODS
            and int(uphill_shakes) < max(0, int(self.max_uphill_shakes))
        ) if neighborhood is not None else False
        return AcceptanceDecision(
            accepted=bool(
                primary_nonworsening
                or (inside_record_band and current_improving)
                or uphill_shake
            ),
            primary_nonworsening=bool(primary_nonworsening),
            current_improving=bool(current_improving),
            uphill_shake=bool(uphill_shake and not primary_nonworsening),
        )
