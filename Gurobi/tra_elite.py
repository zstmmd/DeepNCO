from __future__ import annotations

import math
from typing import Iterable, MutableMapping, Protocol, TypeVar


class _RiskLike(Protocol):
    total: float


class _ShellLike(Protocol):
    sha256: str


class EliteCandidate(Protocol):
    shell: _ShellLike
    relaxed_objective: float
    repair_risk: _RiskLike


CandidateT = TypeVar("CandidateT", bound=EliteCandidate)
REPAIR_RISK_WEIGHT = 1e-3


def candidate_preference_key(candidate: EliteCandidate) -> tuple[float, float, float, str]:
    """Balance the no-wait objective with its predicted full-repair burden."""

    objective = float(candidate.relaxed_objective)
    risk = float(candidate.repair_risk.total)
    return (
        objective + REPAIR_RISK_WEIGHT * risk,
        objective,
        risk,
        str(candidate.shell.sha256),
    )


def _comparison_tolerance(left: float, right: float) -> float:
    return 1e-9 * max(1.0, abs(float(left)), abs(float(right)))


def _dominates(left: EliteCandidate, right: EliteCandidate) -> bool:
    left_objective = float(left.relaxed_objective)
    right_objective = float(right.relaxed_objective)
    left_risk = float(left.repair_risk.total)
    right_risk = float(right.repair_risk.total)
    objective_tolerance = _comparison_tolerance(left_objective, right_objective)
    risk_tolerance = _comparison_tolerance(left_risk, right_risk)
    no_worse = (
        left_objective <= right_objective + objective_tolerance
        and left_risk <= right_risk + risk_tolerance
    )
    strictly_better = (
        left_objective < right_objective - objective_tolerance
        or left_risk < right_risk - risk_tolerance
    )
    return bool(no_worse and strictly_better)


def _normalized_coordinates(candidates: list[CandidateT]) -> dict[str, tuple[float, float]]:
    objectives = [float(candidate.relaxed_objective) for candidate in candidates]
    risks = [float(candidate.repair_risk.total) for candidate in candidates]
    objective_span = max(objectives) - min(objectives)
    risk_span = max(risks) - min(risks)
    return {
        str(candidate.shell.sha256): (
            0.0 if objective_span <= 0.0 else (float(candidate.relaxed_objective) - min(objectives)) / objective_span,
            0.0 if risk_span <= 0.0 else (float(candidate.repair_risk.total) - min(risks)) / risk_span,
        )
        for candidate in candidates
    }


def _diversity_clip(front: list[CandidateT], limit: int) -> list[CandidateT]:
    if len(front) <= limit:
        return front

    by_hash = {str(candidate.shell.sha256): candidate for candidate in front}
    risk_endpoint = min(
        front,
        key=lambda candidate: (
            float(candidate.repair_risk.total),
            float(candidate.relaxed_objective),
            str(candidate.shell.sha256),
        ),
    )
    objective_endpoint = min(front, key=candidate_preference_key)
    selected_hashes = {
        str(risk_endpoint.shell.sha256),
        str(objective_endpoint.shell.sha256),
    }
    coordinates = _normalized_coordinates(front)

    while len(selected_hashes) < limit:
        remaining = [candidate for candidate in front if str(candidate.shell.sha256) not in selected_hashes]

        def diversity_key(candidate: CandidateT) -> tuple[float, tuple[float, float, float, str]]:
            candidate_point = coordinates[str(candidate.shell.sha256)]
            nearest_distance = min(
                math.dist(candidate_point, coordinates[selected_hash])
                for selected_hash in selected_hashes
            )
            return (-nearest_distance, candidate_preference_key(candidate))

        selected_hashes.add(str(min(remaining, key=diversity_key).shell.sha256))

    return [candidate for candidate_hash, candidate in by_hash.items() if candidate_hash in selected_hashes]


def select_pareto_elites(candidates: Iterable[CandidateT], *, limit: int = 8) -> tuple[CandidateT, ...]:
    """Return a deterministic, objective-first slice of the non-dominated front."""

    capacity = max(1, min(8, int(limit)))
    best_by_hash: dict[str, CandidateT] = {}
    for candidate in candidates:
        candidate_hash = str(candidate.shell.sha256)
        current = best_by_hash.get(candidate_hash)
        if current is None or candidate_preference_key(candidate) < candidate_preference_key(current):
            best_by_hash[candidate_hash] = candidate

    unique = list(best_by_hash.values())
    front = [
        candidate
        for candidate in unique
        if not any(_dominates(other, candidate) for other in unique if other is not candidate)
    ]
    clipped = _diversity_clip(front, capacity)
    return tuple(sorted(clipped, key=candidate_preference_key))


def update_pareto_elites(
    elites_by_hash: MutableMapping[str, CandidateT],
    candidate: CandidateT,
    *,
    limit: int = 8,
) -> None:
    """Insert one natural MIPSOL candidate and keep only the bounded Pareto front."""

    combined = list(elites_by_hash.values()) + [candidate]
    selected = select_pareto_elites(combined, limit=limit)
    elites_by_hash.clear()
    elites_by_hash.update((str(item.shell.sha256), item) for item in selected)
