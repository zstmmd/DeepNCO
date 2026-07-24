from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from Gurobi.master_domain import MasterDomainError
from Gurobi.master_domain_fingerprint import decode_key


@dataclass(frozen=True)
class MasterNumericBounds:
    slot_time_ub: float
    route_big_m: float
    route_node_time_ub: Mapping[int, float]
    route_arc_time_m: Mapping[tuple[int, int], float]
    pickup_service_lb_by_node: Mapping[int, float]
    pickup_service_ub_by_node: Mapping[int, float]


def _finite_nonnegative(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise MasterDomainError(f"master numeric bound {field_name} is not numeric") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise MasterDomainError(f"master numeric bound {field_name} must be finite and nonnegative")
    return numeric


def _number_rows(
    numeric_bounds: Mapping[str, Any],
    field_name: str,
    *,
    key_arity: int,
) -> dict[Any, float]:
    result: dict[Any, float] = {}
    for row in list(numeric_bounds.get(field_name, ()) or ()):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise MasterDomainError(f"master numeric bound {field_name} has an invalid row")
        key = decode_key(row[0])
        normalized_key = key if isinstance(key, tuple) else (key,)
        if len(normalized_key) != int(key_arity):
            raise MasterDomainError(f"master numeric bound {field_name} has an invalid key")
        typed_key: Any
        if key_arity == 1:
            typed_key = int(normalized_key[0])
        else:
            typed_key = tuple(int(value) for value in normalized_key)
        if typed_key in result:
            raise MasterDomainError(f"master numeric bound {field_name} contains duplicate keys")
        result[typed_key] = _finite_nonnegative(row[1], field_name=field_name)
    return result


def numeric_bounds_from_manifest(manifest: Mapping[str, Any]) -> MasterNumericBounds:
    numeric_bounds = dict(manifest.get("numeric_bounds", {}) or {})
    required = {
        "slot_time_ub",
        "route_big_m",
        "route_node_time_ub",
        "route_arc_time_m",
        "pickup_service_lb_by_node",
        "pickup_service_ub_by_node",
    }
    missing = required - set(numeric_bounds)
    if missing:
        raise MasterDomainError(f"master domain is missing numeric bounds: {sorted(missing)}")
    return MasterNumericBounds(
        slot_time_ub=_finite_nonnegative(numeric_bounds["slot_time_ub"], field_name="slot_time_ub"),
        route_big_m=_finite_nonnegative(numeric_bounds["route_big_m"], field_name="route_big_m"),
        route_node_time_ub=_number_rows(
            numeric_bounds,
            "route_node_time_ub",
            key_arity=1,
        ),
        route_arc_time_m=_number_rows(
            numeric_bounds,
            "route_arc_time_m",
            key_arity=2,
        ),
        pickup_service_lb_by_node=_number_rows(
            numeric_bounds,
            "pickup_service_lb_by_node",
            key_arity=1,
        ),
        pickup_service_ub_by_node=_number_rows(
            numeric_bounds,
            "pickup_service_ub_by_node",
            key_arity=1,
        ),
    )


def require_same_keys(
    field_name: str,
    actual: Mapping[Any, Any],
    expected: Mapping[Any, Any],
) -> None:
    if set(actual) == set(expected):
        return
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    raise MasterDomainError(
        f"master numeric domain mismatch for {field_name}: missing={missing[:10]}, extra={extra[:10]}"
    )
