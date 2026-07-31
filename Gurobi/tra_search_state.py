from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from Gurobi.tra_acceptance import RecordToRecordPolicy, cmax_tolerance
from Gurobi.tra_candidate_archive import CandidateArchive
from Gurobi.tra_inner import InnerCandidate
from Gurobi.tra_model_state import ModelSnapshot
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import objective_tolerance
from Gurobi.tra_projection import StructuralShell
from Gurobi.tra_scheduler import ProcedureStep, RetryRegistry
from Gurobi.tra_verifier import VerifiedSnapshot
from Gurobi.tra_work_queue import SearchWorkQueues


@dataclass(frozen=True)
class TRAIncumbent:
    shell: StructuralShell
    snapshot: ModelSnapshot
    verified_cmax: float
    objective: float

    @classmethod
    def from_verified(cls, verified: VerifiedSnapshot) -> "TRAIncumbent":
        return cls(
            shell=verified.shell,
            snapshot=verified.snapshot,
            verified_cmax=float(verified.verified_cmax),
            objective=float(verified.snapshot.solver_objective),
        )


@dataclass(frozen=True)
class AcceptanceOutcome:
    structural_change: bool
    cmax_improved: bool
    uphill_shake: bool = False


@dataclass(frozen=True)
class CandidateSelection:
    candidate: Optional[InnerCandidate]
    dispositions: tuple[dict[str, str], ...]


@dataclass
class InnerSearchEvidence:
    attempts: int = 0
    solves_with_candidates: int = 0
    solves_with_recoverable_candidates: int = 0
    total_candidates: int = 0
    total_recoverable_candidates: int = 0
    empty_timeouts: int = 0

    def observe(
        self,
        *,
        candidate_count: int,
        timed_out: bool,
        recoverable_count: int = 0,
    ) -> None:
        count = max(0, int(candidate_count))
        recoverable = max(0, min(count, int(recoverable_count)))
        self.attempts += 1
        self.total_candidates += count
        self.total_recoverable_candidates += recoverable
        self.solves_with_candidates += int(count > 0)
        self.solves_with_recoverable_candidates += int(recoverable > 0)
        self.empty_timeouts += int(bool(timed_out) and count == 0)

    @property
    def smoothed_candidate_probability(self) -> float:
        return (self.solves_with_candidates + 1.0) / (self.attempts + 2.0)

    @property
    def smoothed_recoverable_probability(self) -> float:
        return (self.solves_with_recoverable_candidates + 1.0) / (
            self.attempts + 2.0
        )

    @property
    def reserve_priority(self) -> tuple[float, float, float, int, int]:
        timeout_rate = self.empty_timeouts / max(1, self.attempts)
        return (
            -self.smoothed_recoverable_probability,
            -self.smoothed_candidate_probability,
            timeout_rate,
            -self.total_recoverable_candidates,
            -self.total_candidates,
        )

    @property
    def regular_effort_multiplier(self) -> float:
        """Shorten repeatedly empty probes without changing the rotation itself."""

        return max(0.25, min(1.0, 2.0 * self.smoothed_candidate_probability))


@dataclass
class RecourseCalibration:
    """Learn conservative DP3 overprediction from completed certifications."""

    relative_cap: float = 0.10
    submitted_by_shell: dict[str, tuple[Procedure, float, bool]] = field(
        default_factory=dict
    )
    allowance_by_procedure: dict[Procedure, float] = field(default_factory=dict)

    def remember_submission(
        self,
        shell_sha256: str,
        procedure: Procedure,
        recourse_score: float,
        *,
        calibration_eligible: bool = True,
    ) -> None:
        score = float(recourse_score)
        if not math.isfinite(score):
            return
        self.submitted_by_shell[str(shell_sha256)] = (
            Procedure(procedure),
            score,
            bool(calibration_eligible),
        )

    def observe_verification(
        self,
        shell_sha256: str,
        *,
        verified_cmax: float,
        prior_incumbent_cmax: Optional[float],
    ) -> None:
        if (
            prior_incumbent_cmax is None
            or not math.isfinite(float(prior_incumbent_cmax))
        ):
            return
        submitted = self.submitted_by_shell.get(str(shell_sha256))
        if submitted is None or not math.isfinite(float(verified_cmax)):
            return
        procedure, predicted, calibration_eligible = submitted
        if not calibration_eligible:
            return
        observed = max(0.0, float(predicted) - float(verified_cmax))
        cap = max(
            0.5,
            max(0.0, float(self.relative_cap))
            * abs(float(prior_incumbent_cmax)),
        )
        bounded = min(observed, cap)
        self.allowance_by_procedure[procedure] = max(
            float(self.allowance_by_procedure.get(procedure, 0.0)),
            float(bounded),
        )

    def allowance(self, procedure: Procedure) -> float:
        return max(
            0.0,
            float(
                self.allowance_by_procedure.get(
                    Procedure(procedure),
                    0.0,
                )
            ),
        )


@dataclass
class SearchState:
    search_shell: StructuralShell
    start_values: dict[str, float]
    incumbent: Optional[TRAIncumbent] = None
    search_incumbent: Optional[TRAIncumbent] = None
    acceptance_policy: RecordToRecordPolicy = field(default_factory=RecordToRecordPolicy)
    uphill_shakes: int = 0
    queues: SearchWorkQueues = field(default_factory=SearchWorkQueues)
    candidate_archive: CandidateArchive = field(default_factory=CandidateArchive)
    attempted_shells: set[tuple[str, str, str, str]] = field(default_factory=set)
    explored_shells: set[str] = field(default_factory=set)
    retry_registry: RetryRegistry = field(default_factory=RetryRegistry)
    inner_evidence: dict[Procedure, InnerSearchEvidence] = field(default_factory=dict)
    recourse_calibration: RecourseCalibration = field(
        default_factory=RecourseCalibration
    )
    vns_window_counts: dict[tuple[str, str, str], int] = field(default_factory=dict)
    last_n3_improving_procedure: Optional[Procedure] = None
    last_transition_neighborhood: Optional[NeighborhoodLevel] = None
    consecutive_transition_procedure: Optional[Procedure] = None
    consecutive_transition_count: int = 0
    max_consecutive_process_transitions: int = 3
    consecutive_submission_procedure: Optional[Procedure] = None
    consecutive_submission_count: int = 0
    max_consecutive_process_submissions: int = 3
    max_global_process_submissions: int = 3
    submission_neighborhood_procedure: Optional[Procedure] = None
    submitted_neighborhoods: set[NeighborhoodLevel] = field(default_factory=set)
    submission_neighborhood_counts: dict[NeighborhoodLevel, int] = field(
        default_factory=dict
    )
    status: str = "NO_FULL_FEASIBLE_INCUMBENT"
    error: str = ""
    best_objective_bound: Optional[float] = None

    def __post_init__(self) -> None:
        self.explored_shells.add(str(self.search_shell.sha256))

    @property
    def incumbent_objective(self) -> Optional[float]:
        return None if self.incumbent is None else float(self.incumbent.objective)

    @property
    def incumbent_cmax(self) -> Optional[float]:
        return None if self.incumbent is None else float(self.incumbent.verified_cmax)

    def certification_cmax_limit(
        self,
        step: Optional[ProcedureStep],
    ) -> Optional[float]:
        effective_uphill_shakes = self._effective_uphill_shakes(step)
        return self.acceptance_policy.certification_limit(
            best_cmax=self.incumbent_cmax,
            current_cmax=(
                None
                if self.search_incumbent is None
                else float(self.search_incumbent.verified_cmax)
            ),
            neighborhood=None if step is None else step.neighborhood,
            uphill_shakes=effective_uphill_shakes,
        )

    def _effective_uphill_shakes(
        self,
        step: Optional[ProcedureStep],
    ) -> int:
        if (
            step is not None
            and self.consecutive_transition_procedure is not None
            and Procedure(step.procedure)
            is self.consecutive_transition_procedure
            and (
                NeighborhoodLevel(step.neighborhood) is not NeighborhoodLevel.N1
                or self.consecutive_transition_count
                >= max(1, int(self.max_consecutive_process_transitions))
            )
        ):
            return max(
                int(self.uphill_shakes),
                int(self.acceptance_policy.max_uphill_shakes),
            )
        return int(self.uphill_shakes)

    def _reset_submission_history_after_bootstrap(self) -> None:
        self.consecutive_submission_procedure = None
        self.consecutive_submission_count = 0
        self.submission_neighborhood_procedure = None
        self.submitted_neighborhoods.clear()
        self.submission_neighborhood_counts.clear()

    def accept(
        self,
        verified: VerifiedSnapshot,
        *,
        step: Optional[ProcedureStep] = None,
    ) -> AcceptanceOutcome:
        previous_shell_hash = str(self.search_shell.sha256)
        previous_cmax = None if self.incumbent is None else float(self.incumbent.verified_cmax)
        current_cmax = (
            None
            if self.search_incumbent is None
            else float(self.search_incumbent.verified_cmax)
        )
        candidate = TRAIncumbent.from_verified(verified)
        cmax_tolerance_value = cmax_tolerance(previous_cmax)
        cmax_improved = previous_cmax is None or (
            float(candidate.verified_cmax)
            < float(previous_cmax) - cmax_tolerance_value
        )
        decision = self.acceptance_policy.decide(
            best_cmax=previous_cmax,
            current_cmax=current_cmax,
            candidate_cmax=float(candidate.verified_cmax),
            neighborhood=None if step is None else step.neighborhood,
            uphill_shakes=self._effective_uphill_shakes(step),
        )
        if not decision.accepted:
            return AcceptanceOutcome(structural_change=False, cmax_improved=False)
        self.recourse_calibration.observe_verification(
            candidate.shell.sha256,
            verified_cmax=float(candidate.verified_cmax),
            prior_incumbent_cmax=previous_cmax,
        )

        structural_change = str(candidate.shell.sha256) != previous_shell_hash
        incumbent_objective = None if self.incumbent is None else float(self.incumbent.objective)
        objective_tolerance_value = objective_tolerance(incumbent_objective)
        primary_equal = previous_cmax is not None and (
            abs(float(candidate.verified_cmax) - float(previous_cmax))
            <= cmax_tolerance_value
        )
        global_improvement = (
            self.incumbent is None
            or cmax_improved
            or (
                primary_equal
                and float(candidate.objective)
                < float(incumbent_objective) - objective_tolerance_value
            )
        )
        if (
            step is not None
            and structural_change
            and (
                step.neighborhood is NeighborhoodLevel.N3
                or decision.uphill_shake
            )
        ):
            self.last_n3_improving_procedure = Procedure(step.procedure)
        if step is not None and structural_change:
            transition_procedure = Procedure(step.procedure)
            self.last_transition_neighborhood = NeighborhoodLevel(
                step.neighborhood
            )
            if transition_procedure is self.consecutive_transition_procedure:
                self.consecutive_transition_count += 1
            else:
                self.consecutive_transition_procedure = transition_procedure
                self.consecutive_transition_count = 1
        if global_improvement:
            self.incumbent = candidate
        if decision.primary_nonworsening:
            self.uphill_shakes = 0
        elif decision.uphill_shake:
            self.uphill_shakes += 1
        self.search_incumbent = candidate
        self.search_shell = candidate.shell
        self.start_values = dict(candidate.snapshot.values_by_name)
        self.explored_shells.add(str(candidate.shell.sha256))
        if structural_change:
            self.queues.clear()
            self.attempted_shells.clear()
            self.retry_registry = RetryRegistry()
        if previous_cmax is None:
            self._reset_submission_history_after_bootstrap()
        self.status = "INCUMBENT_FOUND"
        return AcceptanceOutcome(
            structural_change=bool(structural_change),
            cmax_improved=bool(cmax_improved),
            uphill_shake=bool(decision.uphill_shake),
        )

    def restore_global_search(self) -> bool:
        if self.incumbent is None:
            return False
        if str(self.search_shell.sha256) == str(self.incumbent.shell.sha256):
            return False
        self.search_incumbent = self.incumbent
        self.search_shell = self.incumbent.shell
        self.start_values = dict(self.incumbent.snapshot.values_by_name)
        self.uphill_shakes = 0
        self.queues.clear()
        self.attempted_shells.clear()
        self.retry_registry = RetryRegistry()
        return True

    def next_vns_offset(
        self,
        reference_shell: StructuralShell,
        procedure: Procedure,
        neighborhood: NeighborhoodLevel,
        *,
        width: int = 4,
    ) -> int:
        key = (
            str(reference_shell.sha256),
            str(Procedure(procedure).value),
            str(NeighborhoodLevel(neighborhood).value),
        )
        count = int(self.vns_window_counts.get(key, 0))
        self.vns_window_counts[key] = count + 1
        return max(1, int(width)) * count

    def select_unattempted_candidate(
        self,
        reference_shell: StructuralShell,
        step: ProcedureStep,
        candidates: Iterable[InnerCandidate],
        *,
        allow_diverse_neighborhood_repeat: bool = False,
    ) -> Optional[InnerCandidate]:
        return self.select_unattempted_candidate_with_dispositions(
            reference_shell,
            step,
            candidates,
            allow_diverse_neighborhood_repeat=allow_diverse_neighborhood_repeat,
        ).candidate

    def select_unattempted_candidate_with_dispositions(
        self,
        reference_shell: StructuralShell,
        step: ProcedureStep,
        candidates: Iterable[InnerCandidate],
        *,
        allow_diverse_neighborhood_repeat: bool = False,
    ) -> CandidateSelection:
        dispositions = []
        for candidate in self._submission_candidates(step, candidates):
            shell_sha256 = str(candidate.shell.sha256)
            if str(candidate.shell.sha256) in self.explored_shells:
                dispositions.append(
                    {
                        "shell_sha256": shell_sha256,
                        "disposition": "explored_shell",
                    }
                )
                continue
            if not self.candidate_within_certification_band(step, candidate):
                dispositions.append(
                    {
                        "shell_sha256": shell_sha256,
                        "disposition": "certification_band",
                    }
                )
                continue
            comproc = getattr(candidate, "comproc", None)
            projected_cmax = float(
                comproc.projected_cmax
                if comproc is not None
                else float("inf")
            )
            recourse_score = float(
                getattr(comproc, "recourse_score", float("inf"))
                if comproc is not None
                else float("inf")
            )
            submission_procedure = Procedure(step.procedure)
            submission_neighborhood = NeighborhoodLevel(step.neighborhood)
            neighborhood_submission_count = (
                int(
                    self.submission_neighborhood_counts.get(
                        submission_neighborhood,
                        0,
                    )
                )
                if self.submission_neighborhood_procedure
                is submission_procedure
                else 0
            )
            if (
                neighborhood_submission_count
                >= (2 if allow_diverse_neighborhood_repeat else 1)
                and (
                    self.incumbent_cmax is None
                    or not math.isfinite(projected_cmax)
                    or projected_cmax
                    >= float(self.incumbent_cmax)
                    - objective_tolerance(self.incumbent_cmax)
                )
            ):
                dispositions.append(
                    {
                        "shell_sha256": shell_sha256,
                        "disposition": "neighborhood_submission_quota",
                    }
                )
                continue
            if (
                self.consecutive_submission_procedure
                is Procedure(step.procedure)
                and self.consecutive_submission_count
                >= self._submission_limit()
                and (
                    self.incumbent_cmax is None
                    or not math.isfinite(projected_cmax)
                    or projected_cmax
                    >= float(self.incumbent_cmax)
                    - objective_tolerance(self.incumbent_cmax)
                )
            ):
                dispositions.append(
                    {
                        "shell_sha256": shell_sha256,
                        "disposition": "process_submission_quota",
                    }
                )
                continue
            if (
                self.consecutive_transition_procedure is Procedure(step.procedure)
                and self.consecutive_transition_count
                >= max(1, int(self.max_consecutive_process_transitions))
            ):
                if (
                    self.incumbent_cmax is None
                    or not math.isfinite(projected_cmax)
                    or projected_cmax
                    >= float(self.incumbent_cmax)
                    - objective_tolerance(self.incumbent_cmax)
                ):
                    dispositions.append(
                        {
                            "shell_sha256": shell_sha256,
                            "disposition": "transition_quota",
                        }
                    )
                    continue
            attempt_key = (
                str(reference_shell.sha256),
                str(step.procedure.value),
                str(step.neighborhood.value),
                str(candidate.shell.sha256),
            )
            if attempt_key in self.attempted_shells:
                dispositions.append(
                    {
                        "shell_sha256": shell_sha256,
                        "disposition": "attempted_shell",
                    }
                )
                continue
            self.attempted_shells.add(attempt_key)
            if submission_procedure is self.consecutive_submission_procedure:
                self.consecutive_submission_count += 1
            else:
                self.consecutive_submission_procedure = submission_procedure
                self.consecutive_submission_count = 1
            if (
                submission_procedure
                is not self.submission_neighborhood_procedure
            ):
                self.submission_neighborhood_procedure = submission_procedure
                self.submitted_neighborhoods.clear()
                self.submission_neighborhood_counts.clear()
            self.submitted_neighborhoods.add(submission_neighborhood)
            self.submission_neighborhood_counts[submission_neighborhood] = (
                int(
                    self.submission_neighborhood_counts.get(
                        submission_neighborhood,
                        0,
                    )
                )
                + 1
            )
            self.recourse_calibration.remember_submission(
                candidate.shell.sha256,
                submission_procedure,
                recourse_score,
                calibration_eligible=self.incumbent_cmax is not None,
            )
            dispositions.append(
                {
                    "shell_sha256": shell_sha256,
                    "disposition": "selected",
                }
            )
            return CandidateSelection(
                candidate=candidate,
                dispositions=tuple(dispositions),
            )
        return CandidateSelection(candidate=None, dispositions=tuple(dispositions))

    @staticmethod
    def _projected_candidate_key(
        candidate: InnerCandidate,
    ) -> tuple[float, float, float, float, str]:
        comproc = getattr(candidate, "comproc", None)

        def finite(value: float) -> float:
            numeric = float(value)
            return numeric if math.isfinite(numeric) else float("inf")

        return (
            finite(getattr(comproc, "projected_cmax", float("inf"))),
            finite(getattr(comproc, "projected_objective", float("inf"))),
            finite(
                getattr(
                    getattr(candidate, "repair_risk", None),
                    "total",
                    float("inf"),
                )
            ),
            finite(getattr(candidate, "relaxed_objective", float("inf"))),
            str(candidate.shell.sha256),
        )

    @staticmethod
    def _core_projection_key(candidate: InnerCandidate) -> str:
        projection = getattr(getattr(candidate, "shell", None), "projection", None)
        if projection is None:
            return str(getattr(getattr(candidate, "shell", None), "sha256", ""))
        if hasattr(projection, "as_canonical_payload"):
            return repr(projection.as_canonical_payload())
        return repr(projection)

    @staticmethod
    def _z_action_active_count(candidate: InnerCandidate) -> int:
        z_actions = getattr(getattr(candidate, "shell", None), "z_actions", {}) or {}
        active = 0
        for family_values in dict(z_actions).values():
            for value in dict(family_values or {}).values():
                try:
                    if abs(float(value)) > 1e-9:
                        active += 1
                except (TypeError, ValueError):
                    continue
        return active

    def _submission_candidates(
        self,
        step: ProcedureStep,
        candidates: Iterable[InnerCandidate],
    ) -> tuple[InnerCandidate, ...]:
        ranked = tuple(candidates)
        if (
            self.incumbent is None
            and Procedure(step.procedure) is Procedure.F1
            and NeighborhoodLevel(step.neighborhood) is NeighborhoodLevel.N1
        ):
            by_projection: dict[str, list[InnerCandidate]] = {}
            for candidate in ranked:
                by_projection.setdefault(
                    self._core_projection_key(candidate),
                    [],
                ).append(candidate)
            emitted: set[str] = set()
            reordered: list[InnerCandidate] = []
            for candidate in ranked:
                projection_key = self._core_projection_key(candidate)
                if projection_key in emitted:
                    continue
                emitted.add(projection_key)
                group = by_projection[projection_key]
                if len(group) > 1:
                    group = sorted(
                        group,
                        key=lambda item: (
                            self._z_action_active_count(item),
                            self._projected_candidate_key(item),
                        ),
                    )
                reordered.extend(group)
            ranked = tuple(reordered)
        if (
            Procedure(step.procedure) is not Procedure.F2
            or NeighborhoodLevel(step.neighborhood) is not NeighborhoodLevel.N2
        ):
            return ranked
        finite_recourse = [
            float(candidate.comproc.recourse_score)
            for candidate in ranked
            if getattr(candidate, "comproc", None) is not None
            and bool(getattr(candidate.comproc, "feasible", True))
            and math.isfinite(
                float(getattr(candidate.comproc, "recourse_score", float("nan")))
            )
        ]
        if not finite_recourse:
            return ranked
        best_recourse = min(finite_recourse)
        uncertainty = max(0.5, 0.002 * abs(best_recourse))
        near = [
            candidate
            for candidate in ranked
            if getattr(candidate, "comproc", None) is not None
            and bool(getattr(candidate.comproc, "feasible", True))
            and math.isfinite(
                float(getattr(candidate.comproc, "recourse_score", float("nan")))
            )
            and float(getattr(candidate.comproc, "recourse_score"))
            <= best_recourse + uncertainty
        ]
        near_hashes = {str(candidate.shell.sha256) for candidate in near}
        far = [
            candidate
            for candidate in ranked
            if str(candidate.shell.sha256) not in near_hashes
        ]
        return tuple(sorted(near, key=self._projected_candidate_key) + far)

    def candidate_within_certification_band(
        self,
        step: ProcedureStep,
        candidate: InnerCandidate,
    ) -> bool:
        # DP3 is a ranking proxy, not a certified upper bound.  Small
        # cross-process neighborhoods must reach the exact outer model even
        # when that proxy is conservative. The band protects N3 and redundant
        # same-process N2 work after a record-level transition.
        neighborhood = NeighborhoodLevel(step.neighborhood)
        same_process_record_n2 = bool(
            neighborhood is NeighborhoodLevel.N2
            and self.consecutive_transition_procedure is Procedure(step.procedure)
            and not self.on_uphill_branch
        )
        if (
            neighborhood is not NeighborhoodLevel.N3
            and not same_process_record_n2
        ):
            return True
        comproc = getattr(candidate, "comproc", None)
        recourse_score = float(
            getattr(comproc, "recourse_score", float("nan"))
        )
        limit = self.certification_cmax_limit(step)
        if limit is None or not math.isfinite(float(limit)):
            return True
        if not math.isfinite(recourse_score):
            return True
        uncertainty = max(0.5, 0.002 * abs(float(limit)))
        calibrated_score = (
            recourse_score
            - self.recourse_calibration.allowance(step.procedure)
        )
        return bool(calibrated_score <= float(limit) + uncertainty)

    def _submission_limit(self) -> int:
        diversified = bool(
            self.incumbent is not None
            and self.search_incumbent is not None
            and float(self.search_incumbent.verified_cmax)
            > float(self.incumbent.verified_cmax)
            + objective_tolerance(self.incumbent.verified_cmax)
        )
        return max(
            1,
            int(
                self.max_consecutive_process_submissions
                if diversified
                else self.max_global_process_submissions
            ),
        )

    def certified_prune(self, objective_bound: float) -> bool:
        self.observe_objective_bound(objective_bound)
        # The inner objective includes secondary route terms. Its bound can
        # prove objective improvement impossible, but cannot rule out a new
        # equal-Cmax shell that is useful for the next rotation.
        return False

    def observe_objective_bound(self, objective_bound: float) -> None:
        bound = float(objective_bound)
        if not math.isfinite(bound):
            return
        if self.best_objective_bound is None:
            self.best_objective_bound = bound
            return
        # Minimization lower bounds are tighter when they increase.
        self.best_objective_bound = max(float(self.best_objective_bound), bound)

    def objective_gap_satisfied(self, mip_gap: float) -> bool:
        if self.incumbent is None or self.best_objective_bound is None:
            return False
        gap_limit = float(mip_gap)
        if not math.isfinite(gap_limit) or gap_limit < 0.0:
            return False
        objective = float(self.incumbent.objective)
        bound = float(self.best_objective_bound)
        if not math.isfinite(objective) or not math.isfinite(bound):
            return False
        denominator = max(1.0, abs(objective))
        relative_gap = max(0.0, (objective - bound) / denominator)
        return bool(relative_gap <= gap_limit)

    def observe_inner(
        self,
        procedure: Procedure,
        *,
        candidate_count: int,
        timed_out: bool,
        recoverable_count: int = 0,
    ) -> None:
        evidence = self.inner_evidence.setdefault(Procedure(procedure), InnerSearchEvidence())
        evidence.observe(
            candidate_count=candidate_count,
            timed_out=timed_out,
            recoverable_count=recoverable_count,
        )

    def inner_effort_multiplier(self, procedure: Procedure) -> float:
        evidence = self.inner_evidence.get(Procedure(procedure), InnerSearchEvidence())
        return evidence.regular_effort_multiplier

    @property
    def has_compatible_archive(self) -> bool:
        return any(
            self.ranked_archive(procedure)
            for procedure in Procedure
        )

    @property
    def on_uphill_branch(self) -> bool:
        if self.incumbent is None or self.search_incumbent is None:
            return False
        best_cmax = float(self.incumbent.verified_cmax)
        return bool(
            float(self.search_incumbent.verified_cmax)
            > best_cmax + cmax_tolerance(best_cmax)
        )

    @property
    def archive_reference_sha256(self) -> Optional[str]:
        # Once a record-to-record shake is accepted, deepen that branch before
        # admitting sibling candidates generated from an older parent shell.
        if not self.on_uphill_branch or self.queues.empty:
            return None
        return str(self.search_shell.sha256)

    def ranked_archive(self, procedure: Procedure):
        procedure = Procedure(procedure)
        required_reference_sha256 = self.archive_reference_sha256
        if (
            required_reference_sha256 is not None
            and procedure is self.consecutive_transition_procedure
            and self.consecutive_transition_count <= 1
        ):
            # Before rotating another block, test one sibling of a merely
            # uphill move. It is still a neighbor of the same fixed shell and
            # may recover the global record without abandoning intensification.
            required_reference_sha256 = None
        return self.candidate_archive.ranked(
            procedure,
            self.search_shell,
            excluded_hashes=self.explored_shells,
            required_reference_sha256=required_reference_sha256,
        )

    def allow_archive_neighborhood_repeat(
        self,
        step: ProcedureStep,
    ) -> bool:
        return bool(
            self.on_uphill_branch
            and self.consecutive_transition_procedure is Procedure(step.procedure)
            and self.last_transition_neighborhood
            is NeighborhoodLevel(step.neighborhood)
        )

    def deferred_priority(
        self,
        item: object,
    ) -> tuple[int, float, float, float, int, int, int, int, int, int]:
        step = getattr(item, "step")
        procedure = Procedure(step.procedure)
        evidence = self.inner_evidence.get(procedure, InnerSearchEvidence())
        process_rank = 0
        if procedure is Procedure.F1:
            process_rank = 2
        elif self.last_n3_improving_procedure is procedure:
            process_rank = 1
        rotation_rank = 0
        if self.last_n3_improving_procedure is not None:
            order = tuple(Procedure)
            previous_index = order.index(self.last_n3_improving_procedure)
            candidate_index = order.index(procedure)
            distance = (candidate_index - previous_index) % len(order)
            rotation_rank = len(order) if distance == 0 else distance
        cold_start_rank = int(procedure is Procedure.F1)
        neighborhood = NeighborhoodLevel(step.neighborhood)
        if procedure is Procedure.F2:
            neighborhood_rank = {
                NeighborhoodLevel.N2: 0,
                NeighborhoodLevel.N3: 1,
                NeighborhoodLevel.N1: 2,
            }[neighborhood]
        else:
            neighborhood_rank = {
                NeighborhoodLevel.N1: 0,
                NeighborhoodLevel.N2: 1,
                NeighborhoodLevel.N3: 2,
            }[neighborhood]
        return (
            process_rank,
            *evidence.reserve_priority,
            cold_start_rank,
            rotation_rank,
            neighborhood_rank,
            int(step.procedure_index),
        )

    def estimated_outer_horizon(self, regular_steps: int) -> int:
        steps = max(1, int(regular_steps))
        probabilities = [
            self.inner_evidence.get(procedure, InnerSearchEvidence()).smoothed_candidate_probability
            for procedure in Procedure
        ]
        expected_submissions = steps * sum(probabilities) / len(probabilities)
        return max(1, min(steps, int(math.ceil(expected_submissions))))

    def allow_deferred_before_pending(self) -> bool:
        items = self.queues.deferred_items()
        if not items or not self.queues.pending_count:
            return True
        next_pending = self.queues.peek_pending()
        if next_pending.accepted_refinement:
            return True
        if not next_pending.reserve_retry:
            return False
        if (
            self.incumbent is not None
            and math.isfinite(float(next_pending.projected_cmax))
            and float(next_pending.projected_cmax)
            > float(self.incumbent.verified_cmax)
            + objective_tolerance(self.incumbent.verified_cmax)
        ):
            for item in items:
                evidence = self.inner_evidence.get(
                    Procedure(item.step.procedure),
                    InnerSearchEvidence(),
                )
                if evidence.attempts and not evidence.total_recoverable_candidates:
                    return True
        probabilities = [
            evidence.smoothed_recoverable_probability
            for item in items
            if (
                evidence := self.inner_evidence.get(
                    Procedure(item.step.procedure),
                    InnerSearchEvidence(),
                )
            ).attempts
        ]
        return bool(probabilities) and max(probabilities) >= 0.25
