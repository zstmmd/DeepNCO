from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


MASTER_DOMAIN_SCHEMA_VERSION = 2


_CANONICAL_WARM_CONFIG_FIELDS = (
    "warm_start_sp4_time_limit_sec",
    "warm_start_subtask_ordering",
    "warm_start_use_sp2_mip_initial",
    "warm_start_sp2_mip_time_limit_sec",
    "warm_start_refine_sp2_after_sp4",
    "warm_start_use_sp4",
    "sort_hit_tote_threshold",
)


class MasterDomainError(ValueError):
    pass


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _point_payload(point: Any) -> Optional[Sequence[float]]:
    if point is None:
        return None
    return [float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0))]


def _problem_fingerprint(problem: Any) -> str:
    orders = []
    for order in list(getattr(problem, "order_list", []) or []):
        orders.append(
            {
                "order_id": int(getattr(order, "order_id", -1)),
                "sku_ids": sorted(int(v) for v in list(getattr(order, "order_product_id_list", []) or [])),
                "total_qty": int(getattr(order, "total_qty", 0) or 0),
            }
        )

    stations = []
    for index, station in enumerate(list(getattr(problem, "station_list", []) or [])):
        stations.append(
            {
                "station_id": int(getattr(station, "id", index)),
                "point": _point_payload(getattr(station, "point", None)),
            }
        )

    robots = []
    for index, robot in enumerate(list(getattr(problem, "robot_list", []) or [])):
        robots.append(
            {
                "robot_id": int(getattr(robot, "id", index)),
                "start_point": _point_payload(getattr(robot, "start_point", None)),
            }
        )

    stacks = []
    for raw_stack_id, stack in dict(getattr(problem, "point_to_stack", {}) or {}).items():
        totes = []
        for tote in list(getattr(stack, "totes", []) or []):
            totes.append(
                {
                    "tote_id": int(getattr(tote, "id", getattr(tote, "tote_id", -1))),
                    "sku_id": int(getattr(tote, "sku_id", getattr(tote, "product_id", -1))),
                    "quantity": int(getattr(tote, "quantity", getattr(tote, "qty", 0)) or 0),
                }
            )
        stacks.append(
            {
                "stack_id": int(raw_stack_id),
                "point": _point_payload(getattr(stack, "store_point", None)),
                "totes": sorted(totes, key=lambda row: (row["tote_id"], row["sku_id"], row["quantity"])),
            }
        )

    payload = {
        "scale_name": str(getattr(problem, "scale_name", "") or ""),
        "orders": sorted(orders, key=lambda row: row["order_id"]),
        "stations": sorted(stations, key=lambda row: row["station_id"]),
        "robots": sorted(robots, key=lambda row: row["robot_id"]),
        "stacks": sorted(stacks, key=lambda row: row["stack_id"]),
    }
    return _sha256(payload)


def _warm_start_fingerprint(warm: Any) -> Tuple[str, Dict[str, Sequence[int]]]:
    rows = []
    protected_stacks: Dict[str, list[int]] = {}
    for raw_order_id, subtasks in dict(getattr(warm, "subtask_by_order", {}) or {}).items():
        order_id = int(raw_order_id)
        order_stacks = set()
        for index, subtask in enumerate(list(subtasks or [])):
            stack_ids = sorted(
                {
                    int(getattr(task, "target_stack_id", -1))
                    for task in list(getattr(subtask, "execution_tasks", []) or [])
                    if int(getattr(task, "target_stack_id", -1)) >= 0
                }
            )
            order_stacks.update(stack_ids)
            rows.append(
                {
                    "order_id": order_id,
                    "slot_index": int(index),
                    "station_id": int(getattr(subtask, "assigned_station_id", -1)),
                    "stack_ids": stack_ids,
                }
            )
        protected_stacks[str(order_id)] = sorted(order_stacks)
    rows.sort(key=lambda row: (row["order_id"], row["slot_index"]))
    return _sha256({"rows": rows}), dict(sorted(protected_stacks.items()))


def _canonical_warm_config(cfg: Any) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for field_name in _CANONICAL_WARM_CONFIG_FIELDS:
        value = getattr(cfg, field_name, None)
        if isinstance(value, bool):
            values[field_name] = bool(value)
        elif isinstance(value, int):
            values[field_name] = int(value)
        elif isinstance(value, float):
            values[field_name] = float(value)
        elif value is not None:
            values[field_name] = str(value)
    return values


def _sorted_pairs(values: Iterable[Sequence[Any]]) -> list[list[int]]:
    return [
        [first, second]
        for first, second in sorted({(int(value[0]), int(value[1])) for value in values})
    ]


def build_master_domain_manifest(
    compiled: Any,
    *,
    canonical_seed: int,
    instance_name: str = "",
) -> Dict[str, Any]:
    prepared = dict(getattr(compiled, "prepared", {}) or {})
    payload = dict(getattr(compiled, "vars_payload", {}) or {})
    problem = getattr(compiled, "problem_template", None)
    warm = getattr(compiled, "warm", None)
    cfg = prepared.get("cfg")
    warm_sha256, protected_stacks = _warm_start_fingerprint(warm)

    route_task_tuples = sorted(
        {
            (
                int(getattr(task, "slot_id", -1)),
                int(getattr(task, "stack_id", -1)),
                int(getattr(task, "station_id", -1)),
            )
            for task in dict(payload.get("route_tasks", {}) or {}).values()
        }
    )
    body: Dict[str, Any] = {
        "schema_version": MASTER_DOMAIN_SCHEMA_VERSION,
        "instance_name": str(instance_name or getattr(problem, "scale_name", "") or ""),
        "canonical_seed": int(canonical_seed),
        "problem_sha256": _problem_fingerprint(problem),
        "warm_start_sha256": warm_sha256,
        "canonical_warm_config": _canonical_warm_config(cfg),
        "slot_count_by_order": {
            str(int(order_id)): int(len(list(slot_ids or [])))
            for order_id, slot_ids in sorted(
                dict(prepared.get("slot_ids_by_order", {}) or {}).items(), key=lambda item: int(item[0])
            )
        },
        "candidate_stacks_by_order": {
            str(int(order_id)): sorted({int(stack_id) for stack_id in list(stack_ids or [])})
            for order_id, stack_ids in sorted(
                dict(prepared.get("candidate_stacks_by_order", {}) or {}).items(), key=lambda item: int(item[0])
            )
        },
        "route_task_tuples": [list(value) for value in route_task_tuples],
        "route_arcs": _sorted_pairs(payload.get("route_arcs", []) or []),
        "protected_route_arcs": _sorted_pairs(payload.get("protected_route_arcs", []) or []),
        "protected_warm_stacks_by_order": protected_stacks,
    }
    body["manifest_sha256"] = _sha256(body)
    return body


def normalize_master_domain_manifest(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MasterDomainError("master domain manifest must be a mapping")
    supplied_hash = str(raw.get("manifest_sha256", "") or "")
    body = {str(key): value for key, value in raw.items() if str(key) != "manifest_sha256"}
    if int(body.get("schema_version", 0) or 0) != MASTER_DOMAIN_SCHEMA_VERSION:
        raise MasterDomainError(f"unsupported master domain schema: {body.get('schema_version')}")

    normalized: Dict[str, Any] = {
        "schema_version": MASTER_DOMAIN_SCHEMA_VERSION,
        "instance_name": str(body.get("instance_name", "") or ""),
        "canonical_seed": int(body.get("canonical_seed", 0) or 0),
        "problem_sha256": str(body.get("problem_sha256", "") or ""),
        "warm_start_sha256": str(body.get("warm_start_sha256", "") or ""),
        "canonical_warm_config": {
            str(key): value
            for key, value in sorted(dict(body.get("canonical_warm_config", {}) or {}).items())
            if str(key) in _CANONICAL_WARM_CONFIG_FIELDS
        },
        "slot_count_by_order": {
            str(int(order_id)): int(count)
            for order_id, count in sorted(
                dict(body.get("slot_count_by_order", {}) or {}).items(), key=lambda item: int(item[0])
            )
        },
        "candidate_stacks_by_order": {
            str(int(order_id)): sorted({int(stack_id) for stack_id in list(stack_ids or [])})
            for order_id, stack_ids in sorted(
                dict(body.get("candidate_stacks_by_order", {}) or {}).items(), key=lambda item: int(item[0])
            )
        },
        "route_task_tuples": [
            list(value)
            for value in sorted(
                {
                    (int(value[0]), int(value[1]), int(value[2]))
                    for value in list(body.get("route_task_tuples", []) or [])
                }
            )
        ],
        "route_arcs": _sorted_pairs(body.get("route_arcs", []) or []),
        "protected_route_arcs": _sorted_pairs(body.get("protected_route_arcs", []) or []),
        "protected_warm_stacks_by_order": {
            str(int(order_id)): sorted({int(stack_id) for stack_id in list(stack_ids or [])})
            for order_id, stack_ids in sorted(
                dict(body.get("protected_warm_stacks_by_order", {}) or {}).items(), key=lambda item: int(item[0])
            )
        },
    }
    calculated_hash = _sha256(normalized)
    if supplied_hash and supplied_hash != calculated_hash:
        raise MasterDomainError(
            f"master domain manifest hash mismatch: supplied={supplied_hash}, calculated={calculated_hash}"
        )
    normalized["manifest_sha256"] = calculated_hash
    return normalized


def verify_manifest_problem(manifest: Mapping[str, Any], problem: Any) -> None:
    expected = str(manifest.get("problem_sha256", "") or "")
    actual = _problem_fingerprint(problem)
    if expected and expected != actual:
        raise MasterDomainError(f"master domain problem hash mismatch: expected={expected}, actual={actual}")


def verify_manifest_warm_start(manifest: Mapping[str, Any], warm: Any) -> None:
    expected = str(manifest.get("warm_start_sha256", "") or "")
    actual, _protected_stacks = _warm_start_fingerprint(warm)
    if expected and expected != actual:
        raise MasterDomainError(f"master domain warm-start hash mismatch: expected={expected}, actual={actual}")
