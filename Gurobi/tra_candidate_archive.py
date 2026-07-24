from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from Gurobi.tra_comproc.ranking import comproc_candidate_key
from Gurobi.tra_neighborhood import Procedure
from Gurobi.tra_projection import StructuralShell, raw_one_hot_hamming
from Gurobi.tra_scheduler import ProcedureStep


@dataclass(frozen=True)
class ArchivedCandidate:
    reference_shell: StructuralShell
    step: ProcedureStep
    candidate: Any


def released_block_distance(
    procedure: Procedure,
    left: StructuralShell,
    right: StructuralShell,
) -> int:
    procedure = Procedure(procedure)
    if procedure is Procedure.F1:
        left_block = left.projection.s_visit
        right_block = right.projection.s_visit
    elif procedure is Procedure.F2:
        left_block = left.projection.x_group
        right_block = right.projection.x_group
    else:
        left_block = left.projection.r_assign
        right_block = right.projection.r_assign
    return int(raw_one_hot_hamming(left_block, right_block))


def fixed_blocks_compatible(
    procedure: Procedure,
    left: StructuralShell,
    right: StructuralShell,
) -> bool:
    procedure = Procedure(procedure)
    left_projection = left.projection
    right_projection = right.projection
    if procedure is Procedure.F1:
        return bool(
            left_projection.x_group == right_projection.x_group
            and left_projection.r_assign == right_projection.r_assign
        )
    if procedure is Procedure.F2:
        return bool(
            left_projection.s_visit == right_projection.s_visit
            and left_projection.r_assign == right_projection.r_assign
        )
    return bool(
        left_projection.s_visit == right_projection.s_visit
        and left_projection.x_group == right_projection.x_group
    )


class CandidateArchive:
    """Keep a bounded, target-blind beam of fully projected inner candidates."""

    def __init__(self, *, per_procedure_limit: int = 8) -> None:
        self.per_procedure_limit = max(1, min(8, int(per_procedure_limit)))
        self._items: dict[Procedure, dict[str, ArchivedCandidate]] = {
            procedure: {} for procedure in Procedure
        }

    @property
    def empty(self) -> bool:
        return not any(self._items.values())

    @property
    def count(self) -> int:
        return sum(len(items) for items in self._items.values())

    def remember(
        self,
        reference_shell: StructuralShell,
        step: ProcedureStep,
        candidates: Iterable[Any],
        *,
        excluded_hashes: Iterable[str] = (),
    ) -> None:
        procedure = Procedure(step.procedure)
        excluded = {str(value) for value in excluded_hashes}
        items = self._items[procedure]
        for shell_hash in excluded:
            items.pop(shell_hash, None)
        for candidate in candidates:
            shell_hash = str(candidate.shell.sha256)
            comproc = getattr(candidate, "comproc", None)
            if (
                shell_hash in excluded
                or comproc is None
                or not bool(comproc.feasible)
            ):
                continue
            archived = ArchivedCandidate(reference_shell, step, candidate)
            current = items.get(shell_hash)
            if (
                current is None
                or comproc_candidate_key(candidate)
                < comproc_candidate_key(current.candidate)
            ):
                items[shell_hash] = archived
        if len(items) > self.per_procedure_limit:
            ordered = sorted(
                items.values(),
                key=lambda item: comproc_candidate_key(item.candidate),
            )
            retained = [ordered.pop(0)]
            while ordered and len(retained) < self.per_procedure_limit:
                next_item = min(
                    ordered,
                    key=lambda item: (
                        -min(
                            released_block_distance(
                                procedure,
                                item.candidate.shell,
                                selected.candidate.shell,
                            )
                            for selected in retained
                        ),
                        comproc_candidate_key(item.candidate),
                    ),
                )
                ordered.remove(next_item)
                retained.append(next_item)
            self._items[procedure] = {
                str(item.candidate.shell.sha256): item for item in retained
            }

    def ranked(
        self,
        procedure: Procedure,
        anchor_shell: StructuralShell,
        *,
        excluded_hashes: Iterable[str] = (),
        required_reference_sha256: str | None = None,
    ) -> tuple[ArchivedCandidate, ...]:
        procedure = Procedure(procedure)
        excluded = {str(value) for value in excluded_hashes}
        required_reference = (
            None
            if required_reference_sha256 is None
            else str(required_reference_sha256)
        )
        for shell_hash in excluded:
            self._items[procedure].pop(shell_hash, None)
        eligible = [
            item
            for shell_hash, item in self._items[procedure].items()
            if (
                shell_hash not in excluded
                and fixed_blocks_compatible(
                    procedure,
                    anchor_shell,
                    item.candidate.shell,
                )
                and (
                    required_reference is None
                    or str(item.reference_shell.sha256) == required_reference
                )
            )
        ]
        return tuple(
            sorted(
                eligible,
                key=lambda item: (
                    -released_block_distance(
                        procedure,
                        anchor_shell,
                        item.candidate.shell,
                    ),
                    comproc_candidate_key(item.candidate),
                ),
            )
        )

    def discard(self, procedure: Procedure, shell_hash: str) -> None:
        self._items[Procedure(procedure)].pop(str(shell_hash), None)

    def has_compatible(
        self,
        anchor_shell: StructuralShell,
        *,
        excluded_hashes: Iterable[str] = (),
        required_reference_sha256: str | None = None,
    ) -> bool:
        return any(
            self.ranked(
                procedure,
                anchor_shell,
                excluded_hashes=excluded_hashes,
                required_reference_sha256=required_reference_sha256,
            )
            for procedure in Procedure
        )
