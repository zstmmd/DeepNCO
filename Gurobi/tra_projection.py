from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Tuple


INACTIVE_LABEL = -1
ACTION_FAMILIES = ("flip", "sort", "carry", "hit", "noise", "flip_hit")


class ProjectionError(ValueError):
    pass


def _canonical_key(value: Hashable) -> list[Any]:
    if isinstance(value, tuple):
        return ["tuple", *[_canonical_atom(item) for item in value]]
    return ["atom", _canonical_atom(value)]


def _canonical_atom(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _mapping_rows(values: Mapping[Hashable, int]) -> list[list[Any]]:
    rows = [[_canonical_key(key), int(label)] for key, label in values.items()]
    return sorted(rows, key=lambda row: json.dumps(row[0], ensure_ascii=True, separators=(",", ":")))


def _sha256(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_value_getter(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.X)
    except Exception as exc:
        raise ProjectionError("variable value is unavailable; pass an explicit value_getter") from exc


def _selected_label(
    family: str,
    carrier: Hashable,
    candidates: Iterable[Tuple[int, Any]],
    getter: Callable[[Any], float],
    *,
    allow_inactive: bool,
    tolerance: float,
) -> int:
    selected = [int(label) for label, variable in candidates if float(getter(variable)) > 0.5]
    if len(selected) > 1:
        raise ProjectionError(f"{family} carrier {carrier!r} is not one-hot: selected={selected}")
    if not selected:
        if allow_inactive:
            return INACTIVE_LABEL
        raise ProjectionError(f"{family} carrier {carrier!r} is not one-hot: no selected label")
    label = int(selected[0])
    for candidate_label, variable in candidates:
        value = float(getter(variable))
        expected = 1.0 if int(candidate_label) == label else 0.0
        if abs(value - expected) > tolerance:
            raise ProjectionError(
                f"{family} carrier {carrier!r} is not integral one-hot: label={candidate_label}, value={value}"
            )
    return label


@dataclass(frozen=True)
class CoreProjection:
    """The three paper-level TRA carriers, independent of derived model variables."""

    x_group: Mapping[Hashable, int]
    s_visit: Mapping[Tuple[int, int], int]
    r_assign: Mapping[int, int]

    def canonicalized(self) -> "CoreProjection":
        return CoreProjection(
            x_group=dict(sorted(self.x_group.items(), key=lambda item: str(item[0]))),
            s_visit=dict(sorted(self.s_visit.items())),
            r_assign=dict(sorted((int(key), int(value)) for key, value in self.r_assign.items())),
        )

    def as_canonical_payload(self) -> Dict[str, Any]:
        return {
            "x_group": _mapping_rows(self.x_group),
            "s_visit": _mapping_rows(self.s_visit),
            "r_assign": _mapping_rows(self.r_assign),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_canonical_payload())

    def block(self, block_name: str) -> Mapping[Hashable, int]:
        if block_name not in {"x_group", "s_visit", "r_assign"}:
            raise ProjectionError(f"unknown projection block: {block_name}")
        return getattr(self, block_name)

    def replace_block(self, block_name: str, values: Mapping[Hashable, int]) -> "CoreProjection":
        if block_name not in {"x_group", "s_visit", "r_assign"}:
            raise ProjectionError(f"unknown projection block: {block_name}")
        return replace(self, **{block_name: dict(values)})


@dataclass(frozen=True)
class StructuralFixingPlan:
    binary_values: Mapping[str, Mapping[Hashable, int]]
    station_marginals: Mapping[Tuple[int, int], int]


@dataclass(frozen=True)
class StructuralShell:
    projection: CoreProjection
    z_actions: Mapping[str, Mapping[Hashable, int]] = field(default_factory=dict)

    @classmethod
    def extract(cls, registry: "ProjectionRegistry") -> "StructuralShell":
        actions: Dict[str, Dict[Hashable, int]] = {}
        for family in ACTION_FAMILIES:
            variables = registry.family(family)
            actions[family] = {
                key: int(float(registry.value_getter(variable)) > 0.5)
                for key, variable in variables.items()
            }
        return cls(projection=registry.extract(), z_actions=actions)

    @property
    def sha256(self) -> str:
        payload = {
            "projection": self.projection.as_canonical_payload(),
            "z_actions": {
                family: _mapping_rows(values)
                for family, values in sorted(self.z_actions.items())
            },
        }
        return _sha256(payload)

    def fixing_plan(self, registry: "ProjectionRegistry") -> StructuralFixingPlan:
        binary_values: Dict[str, Dict[Hashable, int]] = {
            "x": {
                key: int(self.projection.x_group[key[0]] == int(key[1]))
                for key in registry.family("x")
            },
            "pair_activate": {
                key: int(self.projection.s_visit[(int(key[0]), int(key[1]))] == int(key[2]))
                for key in registry.family("pair_activate")
            },
            "slot_robot": {
                key: int(self.projection.r_assign[int(key[0])] == int(key[1]))
                for key in registry.family("slot_robot")
            },
        }
        for family in ACTION_FAMILIES:
            domain = registry.family(family)
            shell_values = self.z_actions.get(family, {})
            if set(domain) != set(shell_values):
                raise ProjectionError(f"structural shell {family} domain does not match compiled model")
            binary_values[family] = {key: int(shell_values[key]) for key in domain}

        station_marginals: Dict[Tuple[int, int], int] = {}
        station_by_slot: Dict[int, int] = {}
        for (slot_id, _stack_id), station_id in self.projection.s_visit.items():
            if int(station_id) == INACTIVE_LABEL:
                continue
            existing = station_by_slot.setdefault(int(slot_id), int(station_id))
            if existing != int(station_id):
                raise ProjectionError(f"slot {slot_id} has visits assigned to multiple stations")
        for slot_id, station_id, _rank in registry.family("y"):
            key = (int(slot_id), int(station_id))
            station_marginals[key] = int(station_by_slot.get(int(slot_id), INACTIVE_LABEL) == int(station_id))

        return StructuralFixingPlan(binary_values=binary_values, station_marginals=station_marginals)


class ProjectionRegistry:
    """Indexes compiled variables once so TRA consumers never rediscover a domain."""

    def __init__(
        self,
        families: Mapping[str, Mapping[Hashable, Any]],
        *,
        value_getter: Optional[Callable[[Any], float]] = None,
        tolerance: float = 1e-6,
    ) -> None:
        self._families = {str(name): dict(values) for name, values in families.items()}
        self.value_getter = value_getter or _default_value_getter
        self.tolerance = float(tolerance)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        value_getter: Optional[Callable[[Any], float]] = None,
        tolerance: float = 1e-6,
    ) -> "ProjectionRegistry":
        required = ("x", "pair_activate", "slot_robot", "y", *ACTION_FAMILIES)
        missing = [name for name in required if payload.get(name) is None]
        if missing:
            raise ProjectionError(f"compiled model is missing TRA variable families: {missing}")
        return cls(
            {name: payload[name] for name in required},
            value_getter=value_getter,
            tolerance=tolerance,
        )

    def family(self, name: str) -> Mapping[Hashable, Any]:
        if name not in self._families:
            raise ProjectionError(f"unknown variable family: {name}")
        return self._families[name]

    def extract(self) -> CoreProjection:
        x_candidates: Dict[Hashable, list[Tuple[int, Any]]] = {}
        for (unit_id, slot_id), variable in self.family("x").items():
            x_candidates.setdefault(unit_id, []).append((int(slot_id), variable))
        x_group = {
            unit_id: _selected_label(
                "x_group",
                unit_id,
                candidates,
                self.value_getter,
                allow_inactive=False,
                tolerance=self.tolerance,
            )
            for unit_id, candidates in x_candidates.items()
        }

        visit_candidates: Dict[Tuple[int, int], list[Tuple[int, Any]]] = {}
        for (slot_id, stack_id, station_id), variable in self.family("pair_activate").items():
            visit_candidates.setdefault((int(slot_id), int(stack_id)), []).append((int(station_id), variable))
        s_visit = {
            key: _selected_label(
                "s_visit",
                key,
                candidates,
                self.value_getter,
                allow_inactive=True,
                tolerance=self.tolerance,
            )
            for key, candidates in visit_candidates.items()
        }

        robot_candidates: Dict[int, list[Tuple[int, Any]]] = {}
        for (slot_id, robot_id), variable in self.family("slot_robot").items():
            robot_candidates.setdefault(int(slot_id), []).append((int(robot_id), variable))
        r_assign = {
            slot_id: _selected_label(
                "r_assign",
                slot_id,
                candidates,
                self.value_getter,
                allow_inactive=True,
                tolerance=self.tolerance,
            )
            for slot_id, candidates in robot_candidates.items()
        }

        active_slots = set(int(slot_id) for slot_id in x_group.values())
        for slot_id, robot_id in r_assign.items():
            if (slot_id in active_slots) != (int(robot_id) != INACTIVE_LABEL):
                raise ProjectionError(f"r_assign active state disagrees with x_group for slot {slot_id}")

        station_by_slot: Dict[int, int] = {}
        for (slot_id, _stack_id), station_id in s_visit.items():
            if int(station_id) == INACTIVE_LABEL:
                continue
            existing = station_by_slot.setdefault(int(slot_id), int(station_id))
            if existing != int(station_id):
                raise ProjectionError(f"s_visit assigns slot {slot_id} to multiple stations")
        return CoreProjection(x_group=x_group, s_visit=s_visit, r_assign=r_assign).canonicalized()


def raw_one_hot_hamming(before: Mapping[Hashable, int], after: Mapping[Hashable, int]) -> int:
    if set(before) != set(after):
        raise ProjectionError("projection carrier domains differ")
    return 2 * sum(int(before[key]) != int(after[key]) for key in before)
