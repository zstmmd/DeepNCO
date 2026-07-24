from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Hashable, Iterable, Mapping

from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure, validate_transition
from Gurobi.tra_projection import (
    INACTIVE_LABEL,
    CoreProjection,
    ProjectionRegistry,
    StructuralShell,
)


@dataclass(frozen=True)
class VNSSeed:
    projection: CoreProjection
    values_by_name: Mapping[str, float]
    changed_carriers: tuple[Hashable, ...]
    sha256: str


def _sort_key(value: Hashable) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def rotating_search_seed(base_seed: int, *, offset: int, width: int = 4) -> int:
    window = max(0, int(offset)) // max(1, int(width))
    return (int(base_seed) + 104729 * window) % 2_000_000_000


class PaperVNSGenerator:
    """Deterministic paper-style relocate, swap, and cyclic neighborhood seeds."""

    def __init__(self, registry: ProjectionRegistry) -> None:
        self.registry = registry

    def _carrier_labels(
        self,
        procedure: Procedure,
    ) -> tuple[str, dict[Hashable, tuple[int, ...]], Mapping[Any, Any]]:
        if procedure is Procedure.F1:
            family_name = "pair_activate"
            family = self.registry.family(family_name)
            grouped: dict[Hashable, set[int]] = {}
            for slot_id, stack_id, station_id in family:
                grouped.setdefault((int(slot_id), int(stack_id)), set()).add(int(station_id))
            return family_name, {
                carrier: tuple([INACTIVE_LABEL, *sorted(labels)])
                for carrier, labels in grouped.items()
            }, family
        if procedure is Procedure.F2:
            family_name = "x"
            family = self.registry.family(family_name)
            grouped = {}
            for unit_id, slot_id in family:
                grouped.setdefault(unit_id, set()).add(int(slot_id))
            return family_name, {
                carrier: tuple(sorted(labels))
                for carrier, labels in grouped.items()
            }, family
        family_name = "slot_robot"
        family = self.registry.family(family_name)
        grouped = {}
        for slot_id, robot_id in family:
            grouped.setdefault(int(slot_id), set()).add(int(robot_id))
        return family_name, {
            carrier: tuple([INACTIVE_LABEL, *sorted(labels)])
            for carrier, labels in grouped.items()
        }, family

    @staticmethod
    def _n1_moves(
        current: Mapping[Hashable, int],
        labels: Mapping[Hashable, tuple[int, ...]],
        procedure: Procedure,
    ) -> Iterable[dict[Hashable, int]]:
        counts = Counter(int(value) for value in current.values())
        carriers = sorted(current, key=_sort_key)
        if procedure is Procedure.F3:
            moves: list[tuple[int, int, str, int, Hashable]] = []
            for carrier in carriers:
                old = int(current[carrier])
                if old == INACTIVE_LABEL:
                    continue
                for label in labels[carrier]:
                    target = int(label)
                    if target == old or target == INACTIVE_LABEL:
                        continue
                    moves.append(
                        (
                            int(counts[target]),
                            -int(counts[old]),
                            _sort_key(carrier),
                            target,
                            carrier,
                        )
                    )
            for _, _, _, target, carrier in sorted(moves):
                moved = dict(current)
                moved[carrier] = int(target)
                yield moved
            return
        for carrier in carriers:
            old = int(current[carrier])
            if procedure in {Procedure.F1, Procedure.F3} and old == INACTIVE_LABEL:
                continue
            if procedure is Procedure.F2 and counts[old] <= 1:
                continue
            for label in labels[carrier]:
                if int(label) == old or int(label) == INACTIVE_LABEL:
                    continue
                moved = dict(current)
                moved[carrier] = int(label)
                yield moved

    @staticmethod
    def _n2_moves(
        current: Mapping[Hashable, int],
        labels: Mapping[Hashable, tuple[int, ...]],
    ) -> Iterable[dict[Hashable, int]]:
        carriers = sorted(current, key=_sort_key)
        for left, right in combinations(carriers, 2):
            left_label = int(current[left])
            right_label = int(current[right])
            if left_label == right_label:
                continue
            if right_label not in labels[left] or left_label not in labels[right]:
                continue
            moved = dict(current)
            moved[left], moved[right] = right_label, left_label
            yield moved

    @staticmethod
    def _n3_moves(
        current: Mapping[Hashable, int],
        labels: Mapping[Hashable, tuple[int, ...]],
        procedure: Procedure,
        balance_support: bool,
    ) -> Iterable[dict[Hashable, int]]:
        carriers = sorted(current, key=_sort_key)
        if procedure is Procedure.F3 and bool(balance_support):
            counts = Counter(int(value) for value in current.values())
            source_order = sorted(
                carriers,
                key=lambda carrier: (
                    -int(counts[int(current[carrier])]),
                    _sort_key(carrier),
                ),
            )
            for size in (3, 4):
                for selected in combinations(source_order, size):
                    moved = dict(current)
                    working_counts = Counter(counts)
                    for carrier in selected:
                        old = int(moved[carrier])
                        targets = [
                            int(label)
                            for label in labels[carrier]
                            if int(label) not in (old, INACTIVE_LABEL)
                        ]
                        if not targets:
                            break
                        target = min(
                            targets,
                            key=lambda label: (
                                int(working_counts[label]),
                                int(label),
                            ),
                        )
                        moved[carrier] = int(target)
                        working_counts[old] -= 1
                        working_counts[target] += 1
                    else:
                        yield moved
        for size in (3, 4):
            for selected in combinations(carriers, size):
                old_labels = [int(current[carrier]) for carrier in selected]
                rotated = old_labels[1:] + old_labels[:1]
                if all(
                    rotated[index] != old_labels[index]
                    and rotated[index] in labels[carrier]
                    for index, carrier in enumerate(selected)
                ):
                    moved = dict(current)
                    for carrier, label in zip(selected, rotated):
                        moved[carrier] = int(label)
                    yield moved

    @staticmethod
    def _projection_values(
        family_name: str,
        family: Mapping[Any, Any],
        projection: Mapping[Hashable, int],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for key, variable in family.items():
            if family_name == "x":
                carrier, label = key[0], int(key[1])
            elif family_name == "pair_activate":
                carrier, label = (int(key[0]), int(key[1])), int(key[2])
            else:
                carrier, label = int(key[0]), int(key[1])
            values[str(variable.VarName)] = float(int(projection[carrier]) == label)
        return values

    def generate(
        self,
        shell: StructuralShell,
        *,
        procedure: Procedure,
        neighborhood: NeighborhoodLevel,
        limit: int = 4,
        offset: int = 0,
        balance_support: bool = True,
    ) -> tuple[VNSSeed, ...]:
        procedure = Procedure(procedure)
        neighborhood = NeighborhoodLevel(neighborhood)
        family_name, labels, family = self._carrier_labels(procedure)
        current = shell.projection.block(procedure.released_block)
        moves = {
            NeighborhoodLevel.N1: self._n1_moves(current, labels, procedure),
            NeighborhoodLevel.N2: self._n2_moves(current, labels),
            NeighborhoodLevel.N3: self._n3_moves(
                current,
                labels,
                procedure,
                balance_support,
            ),
        }[neighborhood]
        seeds: list[VNSSeed] = []
        seen: set[str] = set()
        for moved in moves:
            projection = shell.projection.replace_block(procedure.released_block, moved)
            try:
                audit = validate_transition(
                    shell.projection,
                    projection,
                    procedure,
                    neighborhood,
                )
            except Exception:
                continue
            values = self._projection_values(family_name, family, moved)
            payload = json.dumps(
                sorted(values.items()),
                ensure_ascii=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            if len(seen) <= max(0, int(offset)):
                continue
            seeds.append(
                VNSSeed(
                    projection=projection,
                    values_by_name=values,
                    changed_carriers=tuple(
                        carrier
                        for carrier in current
                        if int(current[carrier]) != int(moved[carrier])
                    ),
                    sha256=digest,
                )
            )
            if len(seeds) >= max(1, int(limit)):
                break
        return tuple(seeds)
