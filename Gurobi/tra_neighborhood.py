from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Mapping

from Gurobi.tra_projection import CoreProjection, ProjectionError, raw_one_hot_hamming


class Procedure(str, Enum):
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"

    @property
    def released_block(self) -> str:
        return {
            Procedure.F1: "s_visit",
            Procedure.F2: "x_group",
            Procedure.F3: "r_assign",
        }[self]


class NeighborhoodLevel(str, Enum):
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"

    @property
    def raw_hamming_limit(self) -> int:
        return {
            NeighborhoodLevel.N1: 2,
            NeighborhoodLevel.N2: 4,
            NeighborhoodLevel.N3: 8,
        }[self]


@dataclass(frozen=True)
class TransitionAudit:
    procedure: Procedure
    neighborhood: NeighborhoodLevel
    released_block: str
    changed_carriers: int
    raw_one_hot_hamming: int


def _same_domain(before: Mapping[Hashable, int], after: Mapping[Hashable, int], block_name: str) -> None:
    if set(before) != set(after):
        missing = sorted((str(value) for value in set(before) - set(after)))[:5]
        extra = sorted((str(value) for value in set(after) - set(before)))[:5]
        raise ProjectionError(f"{block_name} carrier domain changed: missing={missing}, extra={extra}")


def _validate_fixed_blocks(before: CoreProjection, after: CoreProjection, released_block: str) -> None:
    for block_name in ("x_group", "s_visit", "r_assign"):
        if block_name == released_block:
            continue
        if dict(before.block(block_name)) != dict(after.block(block_name)):
            raise ProjectionError(f"{block_name} changed while {released_block} was the released block")


def _validate_f2_relocate_keeps_source_nonempty(
    before: Mapping[Hashable, int],
    after: Mapping[Hashable, int],
    changed: list[Hashable],
) -> None:
    carrier = changed[0]
    source = int(before[carrier])
    if sum(int(label) == source for label in after.values()) < 1:
        raise ProjectionError("F2 N1 relocate would leave its source slot empty")


def validate_transition(
    before: CoreProjection,
    after: CoreProjection,
    procedure: Procedure,
    neighborhood: NeighborhoodLevel,
) -> TransitionAudit:
    """Validate paper TRA neighborhood semantics on the primary carriers only."""

    procedure = Procedure(procedure)
    neighborhood = NeighborhoodLevel(neighborhood)
    released_block = procedure.released_block
    _validate_fixed_blocks(before, after, released_block)

    old = before.block(released_block)
    new = after.block(released_block)
    _same_domain(old, new, released_block)
    changed = [key for key in old if int(old[key]) != int(new[key])]
    hamming = raw_one_hot_hamming(old, new)

    if neighborhood is NeighborhoodLevel.N1:
        if len(changed) != 1 or hamming != 2:
            raise ProjectionError("N1 must change exactly one carrier with raw one-hot Hamming 2")
        if procedure is Procedure.F2:
            _validate_f2_relocate_keeps_source_nonempty(old, new, changed)
    elif neighborhood is NeighborhoodLevel.N2:
        if len(changed) != 2 or hamming != 4:
            raise ProjectionError("N2 must change exactly two carriers with raw one-hot Hamming 4")
        if Counter(int(value) for value in old.values()) != Counter(int(value) for value in new.values()):
            raise ProjectionError("N2 requires label-count conservation and therefore a true swap")
    else:
        if not 3 <= len(changed) <= 4 or not 6 <= hamming <= 8:
            raise ProjectionError("N3 must change three or four carriers with raw one-hot Hamming 6 to 8")

    return TransitionAudit(
        procedure=procedure,
        neighborhood=neighborhood,
        released_block=released_block,
        changed_carriers=len(changed),
        raw_one_hot_hamming=hamming,
    )
