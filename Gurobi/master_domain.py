from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from Gurobi.master_domain_fingerprint import (
    DOMAIN_FAMILIES,
    build_domain_partitions,
    build_model_fingerprints,
    build_numeric_bounds,
    build_pruning_rules,
    build_route_node_contract,
    canonical_json,
    decode_key,
    partition_from_keys,
    sha256_payload,
)


MASTER_DOMAIN_SCHEMA_VERSION = 3
LEGACY_MASTER_DOMAIN_SCHEMA_VERSION = 2


_CANONICAL_WARM_CONFIG_FIELDS = (
    "warm_start_sp4_time_limit_sec",
    "warm_start_sp4_guided_local_search",
    "warm_start_subtask_ordering",
    "warm_start_use_sp2_mip_initial",
    "warm_start_sp2_mip_time_limit_sec",
    "warm_start_refine_sp2_after_sp4",
    "warm_start_use_sp4",
    "enable_sort_hit_tote_threshold",
    "sort_hit_tote_threshold",
)


class MasterDomainError(ValueError):
    pass


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _point_payload(point: Any) -> Optional[Mapping[str, Any]]:
    if point is None:
        return None
    return {
        "idx": int(getattr(point, "idx", -1)),
        "x": float(getattr(point, "x", 0.0)),
        "y": float(getattr(point, "y", 0.0)),
        "type": int(getattr(point, "type", -1)),
    }


def _problem_contract(problem: Any) -> Dict[str, Any]:
    orders = []
    for order in list(getattr(problem, "order_list", []) or []):
        orders.append(
            {
                "order_id": int(getattr(order, "order_id", -1)),
                "sku_ids": sorted(int(v) for v in list(getattr(order, "order_product_id_list", []) or [])),
                "total_qty": int(getattr(order, "total_qty", 0) or 0),
                "unique_sku_count": int(getattr(order, "unique_sku_count", 0) or 0),
                "batch_quantity": int(getattr(order, "batch_quantity", 1) or 1),
                "bom_part_quantity_by_sku": {
                    str(int(sku_id)): int(quantity)
                    for sku_id, quantity in sorted(
                        dict(getattr(order, "bom_part_quantity_by_sku", {}) or {}).items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "bom_total_quantity_by_sku": {
                    str(int(sku_id)): int(quantity)
                    for sku_id, quantity in sorted(
                        dict(getattr(order, "bom_total_quantity_by_sku", {}) or {}).items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "est_sec": float(getattr(order, "est_sec", 0.0) or 0.0),
                "kitting_span_limit_sec": float(getattr(order, "kitting_span_limit_sec", 0.0) or 0.0),
                "lst_sec": float(getattr(order, "lst_sec", 0.0) or 0.0),
            }
        )

    stations = []
    for index, station in enumerate(list(getattr(problem, "station_list", []) or [])):
        stations.append(
            {
                "station_id": int(getattr(station, "id", index)),
                "point": _point_payload(getattr(station, "point", None)),
                "picking_time": float(getattr(station, "picking_time", 0.0) or 0.0),
                "buffer": int(getattr(station, "picking_station_buffer", 0) or 0),
            }
        )

    robots = []
    for index, robot in enumerate(list(getattr(problem, "robot_list", []) or [])):
        robots.append(
            {
                "robot_id": int(getattr(robot, "id", index)),
                "start_point": _point_payload(getattr(robot, "start_point", None)),
                "capacity": int(getattr(robot, "capacity", getattr(robot, "max_stack_height", 0)) or 0),
                "velocity": float(getattr(robot, "velocity", 0.0) or 0.0),
                "packing_time": float(getattr(robot, "packing_time", 0.0) or 0.0),
                "lifting_time": float(getattr(robot, "lifting_time", 0.0) or 0.0),
            }
        )

    stacks = []
    stack_by_id = {
        int(getattr(stack, "stack_id", -1)): stack
        for stack in list(getattr(problem, "stack_list", []) or [])
        if int(getattr(stack, "stack_id", -1)) >= 0
    }
    for raw_stack_id, stack in dict(getattr(problem, "point_to_stack", {}) or {}).items():
        stack_by_id[int(getattr(stack, "stack_id", raw_stack_id))] = stack
    for raw_stack_id, stack in sorted(stack_by_id.items()):
        totes = []
        for layer, tote in enumerate(list(getattr(stack, "totes", []) or [])):
            sku_quantity_map = dict(getattr(tote, "sku_quantity_map", {}) or {})
            if not sku_quantity_map:
                sku_ids = [int(getattr(sku, "id", -1)) for sku in list(getattr(tote, "skus_list", []) or [])]
                capacities = list(getattr(tote, "capacity", []) or [])
                sku_quantity_map = {
                    sku_id: int(capacities[index]) if index < len(capacities) else 1
                    for index, sku_id in enumerate(sku_ids)
                    if sku_id >= 0
                }
            totes.append(
                {
                    "tote_id": int(getattr(tote, "id", getattr(tote, "tote_id", -1))),
                    "layer": int(getattr(tote, "layer", layer) or layer),
                    "sku_quantity": {
                        str(int(sku_id)): int(quantity)
                        for sku_id, quantity in sorted(sku_quantity_map.items(), key=lambda item: int(item[0]))
                    },
                }
            )
        stacks.append(
            {
                "stack_id": int(raw_stack_id),
                "point": _point_payload(getattr(stack, "store_point", None)),
                "max_height": int(getattr(stack, "max_height", 0) or 0),
                "totes": sorted(totes, key=lambda row: (row["layer"], row["tote_id"])),
            }
        )

    map_obj = getattr(problem, "map", None)
    map_points = [
        _point_payload(point)
        for point in list(getattr(map_obj, "point_list", []) or [])
    ]
    return {
        "scale_name": str(getattr(problem, "scale_name", "") or ""),
        "orders": sorted(orders, key=lambda row: row["order_id"]),
        "stations": sorted(stations, key=lambda row: row["station_id"]),
        "robots": sorted(robots, key=lambda row: row["robot_id"]),
        "stacks": sorted(stacks, key=lambda row: row["stack_id"]),
        "map_points": sorted(
            [point for point in map_points if point is not None],
            key=lambda row: (int(row["idx"]), float(row["x"]), float(row["y"])),
        ),
    }


def _problem_fingerprint(problem: Any) -> str:
    return _sha256(_problem_contract(problem))


def _warm_start_contract(warm: Any) -> Tuple[Dict[str, Any], Dict[str, Sequence[int]]]:
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
            tasks = []
            for task in list(getattr(subtask, "execution_tasks", []) or []):
                tasks.append(
                    {
                        "task_id": int(getattr(task, "task_id", -1)),
                        "stack_id": int(getattr(task, "target_stack_id", -1)),
                        "station_id": int(getattr(task, "target_station_id", getattr(subtask, "assigned_station_id", -1))),
                        "robot_id": int(getattr(task, "robot_id", getattr(subtask, "assigned_robot_id", -1))),
                        "mode": str(getattr(task, "operation_mode", "") or ""),
                        "target_tote_ids": sorted(int(value) for value in list(getattr(task, "target_tote_ids", []) or [])),
                        "hit_tote_ids": sorted(int(value) for value in list(getattr(task, "hit_tote_ids", []) or [])),
                        "noise_tote_ids": sorted(int(value) for value in list(getattr(task, "noise_tote_ids", []) or [])),
                        "sort_layer_range": (
                            [int(value) for value in getattr(task, "sort_layer_range")]
                            if getattr(task, "sort_layer_range", None) is not None
                            else None
                        ),
                    }
                )
            rows.append(
                {
                    "order_id": order_id,
                    "slot_index": int(index),
                    "station_id": int(getattr(subtask, "assigned_station_id", -1)),
                    "robot_id": int(getattr(subtask, "assigned_robot_id", -1)),
                    "sku_ids": sorted(
                        int(getattr(sku, "id", sku))
                        for sku in list(getattr(subtask, "sku_list", []) or [])
                    ),
                    "stack_ids": stack_ids,
                    "tasks": sorted(tasks, key=lambda row: (row["task_id"], row["stack_id"])),
                }
            )
        protected_stacks[str(order_id)] = sorted(order_stacks)
    rows.sort(key=lambda row: (row["order_id"], row["slot_index"]))
    contract = {
        "rows": rows,
        "makespan": float(getattr(warm, "makespan", 0.0) or 0.0),
        "sp2_mode": str(getattr(warm, "sp2_mode", "") or ""),
        "sp4_mode": str(getattr(warm, "sp4_mode", "") or ""),
    }
    return contract, dict(sorted(protected_stacks.items()))


def _warm_start_fingerprint(warm: Any) -> Tuple[str, Dict[str, Sequence[int]]]:
    contract, protected_stacks = _warm_start_contract(warm)
    return _sha256(contract), protected_stacks


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


def _domain_semantics(prepared: Mapping[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
    work_units = [
        {
            "unit_id": str(getattr(unit, "unit_id", "")),
            "order_id": int(getattr(unit, "order_id", -1)),
            "sku_id": int(getattr(unit, "sku_id", -1)),
            "demand_qty": int(getattr(unit, "demand_qty", 1) or 1),
        }
        for unit in list(prepared.get("work_units", []) or [])
    ]
    slots = [
        {
            "slot_id": int(getattr(slot, "slot_id", -1)),
            "order_id": int(getattr(slot, "order_id", -1)),
            "local_index": int(getattr(slot, "local_index", -1)),
        }
        for slot in list(prepared.get("slots", []) or [])
    ]
    sort_intervals = []
    for raw_stack_id, intervals in dict(prepared.get("sort_intervals_by_stack", {}) or {}).items():
        for interval in list(intervals or []):
            sort_intervals.append(
                {
                    "stack_id": int(getattr(interval, "stack_id", raw_stack_id)),
                    "low": int(getattr(interval, "low", -1)),
                    "high": int(getattr(interval, "high", -1)),
                    "tote_ids": [int(value) for value in list(getattr(interval, "tote_ids", []) or [])],
                    "robot_service_time": float(getattr(interval, "robot_service_time", 0.0) or 0.0),
                }
            )
    route_tau = [
        [[int(key[0]), int(key[1])], float(value)]
        for key, value in sorted(dict(payload.get("route_tau", {}) or {}).items())
    ]
    return {
        "work_units": sorted(work_units, key=lambda row: row["unit_id"]),
        "slots": sorted(slots, key=lambda row: row["slot_id"]),
        "sort_intervals": sorted(
            sort_intervals,
            key=lambda row: (row["stack_id"], row["low"], row["high"]),
        ),
        "route_nodes": build_route_node_contract(payload),
        "route_tau": route_tau,
        "route_start_nodes": {
            str(int(robot_id)): int(node_id)
            for robot_id, node_id in sorted(dict(payload.get("route_start_nodes", {}) or {}).items())
        },
        "route_end_nodes": {
            str(int(robot_id)): int(node_id)
            for robot_id, node_id in sorted(dict(payload.get("route_end_nodes", {}) or {}).items())
        },
        "demand_hit_totes_by_order": {
            str(int(order_id)): sorted(int(value) for value in list(tote_ids or []))
            for order_id, tote_ids in sorted(dict(payload.get("demand_hit_totes_by_order", {}) or {}).items())
        },
        "support_totes_by_order": {
            str(int(order_id)): sorted(int(value) for value in list(tote_ids or []))
            for order_id, tote_ids in sorted(dict(payload.get("support_totes_by_order", {}) or {}).items())
        },
    }


def _component_hashes(body: Mapping[str, Any]) -> Dict[str, str]:
    components = (
        "problem_contract",
        "warm_start_contract",
        "domain_semantics",
        "domain_partitions",
        "numeric_bounds",
        "model_fingerprints",
        "pruning_rules",
    )
    return {name: sha256_payload(body[name]) for name in components}


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
    problem_contract = _problem_contract(problem)
    warm_start_contract, protected_stacks = _warm_start_contract(warm)
    domain_partitions = build_domain_partitions(payload)
    model_fingerprints = build_model_fingerprints(getattr(compiled, "model", None))
    diagnostics = dict(getattr(compiled, "diagnostics", {}) or {})

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
        "problem_contract": problem_contract,
        "problem_sha256": _sha256(problem_contract),
        "warm_start_contract": warm_start_contract,
        "warm_start_sha256": _sha256(warm_start_contract),
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
        "domain_semantics": _domain_semantics(prepared, payload),
        "domain_partitions": domain_partitions,
        "numeric_bounds": build_numeric_bounds(payload, model_fingerprints),
        "model_fingerprints": model_fingerprints,
        "pruning_rules": sorted(
            build_pruning_rules(cfg, diagnostics, domain_partitions),
            key=lambda row: str(row.get("rule_id", "")),
        ),
    }
    body["component_sha256"] = _component_hashes(body)
    body["manifest_sha256"] = _sha256(body)
    return body


_V3_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "instance_name",
        "canonical_seed",
        "problem_contract",
        "problem_sha256",
        "warm_start_contract",
        "warm_start_sha256",
        "canonical_warm_config",
        "slot_count_by_order",
        "candidate_stacks_by_order",
        "route_task_tuples",
        "route_arcs",
        "protected_route_arcs",
        "protected_warm_stacks_by_order",
        "domain_semantics",
        "domain_partitions",
        "numeric_bounds",
        "model_fingerprints",
        "pruning_rules",
        "component_sha256",
        "manifest_sha256",
    }
)


def _json_normalized(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _normalized_common(body: Mapping[str, Any], schema_version: int) -> Dict[str, Any]:
    return {
        "schema_version": int(schema_version),
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


def _normalize_v2(raw: Mapping[str, Any]) -> Dict[str, Any]:
    supplied_hash = str(raw.get("manifest_sha256", "") or "")
    body = {str(key): value for key, value in raw.items() if str(key) != "manifest_sha256"}
    normalized = _normalized_common(body, LEGACY_MASTER_DOMAIN_SCHEMA_VERSION)
    calculated_hash = _sha256(normalized)
    if supplied_hash and supplied_hash != calculated_hash:
        raise MasterDomainError(
            f"master domain manifest hash mismatch: supplied={supplied_hash}, calculated={calculated_hash}"
        )
    normalized["manifest_sha256"] = calculated_hash
    return normalized


def _normalize_partitions(raw: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    unknown = set(raw) - set(DOMAIN_FAMILIES)
    missing = set(DOMAIN_FAMILIES) - set(raw)
    if unknown or missing:
        raise MasterDomainError(f"domain partition families differ: missing={sorted(missing)}, unknown={sorted(unknown)}")
    normalized: Dict[str, Dict[str, Any]] = {}
    for family in DOMAIN_FAMILIES:
        partition = dict(raw[family] or {})
        keys = sorted([list(row) for row in list(partition.get("keys", []) or [])], key=canonical_json)
        expected = partition_from_keys(decode_key(row) for row in keys)
        supplied_count = int(partition.get("count", -1))
        supplied_hash = str(partition.get("sha256", "") or "")
        if supplied_count != expected["count"] or supplied_hash != expected["sha256"]:
            raise MasterDomainError(f"domain partition hash/count mismatch for {family}")
        normalized[family] = expected
    return normalized


def _normalize_v3(raw: Mapping[str, Any]) -> Dict[str, Any]:
    supplied_fields = {str(key) for key in raw}
    missing = _V3_REQUIRED_FIELDS - supplied_fields
    unknown = supplied_fields - _V3_REQUIRED_FIELDS
    if missing or unknown:
        raise MasterDomainError(f"manifest v3 fields differ: missing={sorted(missing)}, unknown={sorted(unknown)}")
    supplied_hash = str(raw.get("manifest_sha256", "") or "")
    common = _normalized_common(raw, MASTER_DOMAIN_SCHEMA_VERSION)
    normalized: Dict[str, Any] = {
        "schema_version": MASTER_DOMAIN_SCHEMA_VERSION,
        "instance_name": common["instance_name"],
        "canonical_seed": common["canonical_seed"],
        "problem_contract": _json_normalized(raw["problem_contract"]),
        "problem_sha256": common["problem_sha256"],
        "warm_start_contract": _json_normalized(raw["warm_start_contract"]),
        "warm_start_sha256": common["warm_start_sha256"],
        "canonical_warm_config": common["canonical_warm_config"],
        "slot_count_by_order": common["slot_count_by_order"],
        "candidate_stacks_by_order": common["candidate_stacks_by_order"],
        "route_task_tuples": common["route_task_tuples"],
        "route_arcs": common["route_arcs"],
        "protected_route_arcs": common["protected_route_arcs"],
        "protected_warm_stacks_by_order": common["protected_warm_stacks_by_order"],
        "domain_semantics": _json_normalized(raw["domain_semantics"]),
        "domain_partitions": _normalize_partitions(dict(raw["domain_partitions"] or {})),
        "numeric_bounds": _json_normalized(raw["numeric_bounds"]),
        "model_fingerprints": _json_normalized(raw["model_fingerprints"]),
        "pruning_rules": sorted(
            [_json_normalized(row) for row in list(raw["pruning_rules"] or [])],
            key=lambda row: str(row.get("rule_id", "")),
        ),
    }
    if normalized["problem_sha256"] != _sha256(normalized["problem_contract"]):
        raise MasterDomainError("problem contract hash mismatch")
    if normalized["warm_start_sha256"] != _sha256(normalized["warm_start_contract"]):
        raise MasterDomainError("warm-start contract hash mismatch")

    calculated_components = _component_hashes(normalized)
    supplied_components = {
        str(key): str(value)
        for key, value in sorted(dict(raw["component_sha256"] or {}).items())
    }
    if supplied_components != calculated_components:
        raise MasterDomainError("manifest component hash mismatch")
    normalized["component_sha256"] = calculated_components
    calculated_hash = _sha256(normalized)
    if not supplied_hash or supplied_hash != calculated_hash:
        raise MasterDomainError(
            f"master domain manifest hash mismatch: supplied={supplied_hash}, calculated={calculated_hash}"
        )
    normalized["manifest_sha256"] = calculated_hash
    return normalized


def normalize_master_domain_manifest(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MasterDomainError("master domain manifest must be a mapping")
    schema_version = int(raw.get("schema_version", 0) or 0)
    if schema_version == LEGACY_MASTER_DOMAIN_SCHEMA_VERSION:
        return _normalize_v2(raw)
    if schema_version == MASTER_DOMAIN_SCHEMA_VERSION:
        return _normalize_v3(raw)
    raise MasterDomainError(f"unsupported master domain schema: {schema_version}")


@dataclass(frozen=True)
class PreparedDomainFromManifest:
    manifest_sha256: str
    family_keys_by_name: Mapping[str, Tuple[Any, ...]]
    slot_count_by_order: Mapping[int, int]
    candidate_stacks_by_order: Mapping[int, Tuple[int, ...]]
    route_task_tuples: Tuple[Tuple[int, int, int], ...]
    route_arcs: Tuple[Tuple[int, int], ...]
    protected_route_arcs: Tuple[Tuple[int, int], ...]

    def family_keys(self, family: str) -> Tuple[Any, ...]:
        try:
            return self.family_keys_by_name[str(family)]
        except KeyError as exc:
            raise MasterDomainError(f"unknown manifest variable family: {family}") from exc

    def assert_payload_compatible(self, payload: Mapping[str, Any]) -> None:
        actual = build_domain_partitions(payload)
        for family in DOMAIN_FAMILIES:
            expected_keys = self.family_keys(family)
            expected = partition_from_keys(expected_keys)
            if actual[family]["count"] != expected["count"] or actual[family]["sha256"] != expected["sha256"]:
                raise MasterDomainError(f"compiled payload domain differs from manifest for {family}")


def prepared_domain_from_manifest(raw: Mapping[str, Any]) -> PreparedDomainFromManifest:
    manifest = normalize_master_domain_manifest(raw)
    if int(manifest["schema_version"]) != MASTER_DOMAIN_SCHEMA_VERSION:
        raise MasterDomainError("formal TRA consumers require master domain schema v3")
    family_keys = {
        family: tuple(decode_key(row) for row in manifest["domain_partitions"][family]["keys"])
        for family in DOMAIN_FAMILIES
    }
    return PreparedDomainFromManifest(
        manifest_sha256=str(manifest["manifest_sha256"]),
        family_keys_by_name=MappingProxyType(family_keys),
        slot_count_by_order=MappingProxyType(
            {int(order_id): int(count) for order_id, count in manifest["slot_count_by_order"].items()}
        ),
        candidate_stacks_by_order=MappingProxyType(
            {
                int(order_id): tuple(int(stack_id) for stack_id in stack_ids)
                for order_id, stack_ids in manifest["candidate_stacks_by_order"].items()
            }
        ),
        route_task_tuples=tuple(tuple(int(value) for value in row) for row in manifest["route_task_tuples"]),
        route_arcs=tuple(tuple(int(value) for value in row) for row in manifest["route_arcs"]),
        protected_route_arcs=tuple(
            tuple(int(value) for value in row) for row in manifest["protected_route_arcs"]
        ),
    )


def verify_manifest_problem(manifest: Mapping[str, Any], problem: Any) -> None:
    expected = str(manifest.get("problem_sha256", "") or "")
    actual = _problem_fingerprint(problem)
    if expected and expected != actual:
        raise MasterDomainError(f"master domain problem hash mismatch: expected={expected}, actual={actual}")


def manifest_warm_start_hash_status(manifest: Mapping[str, Any], warm: Any) -> Dict[str, Any]:
    expected = str(manifest.get("warm_start_sha256", "") or "")
    actual, _protected_stacks = _warm_start_fingerprint(warm)
    return {
        "master_domain_expected_warm_start_sha256": expected,
        "master_domain_actual_warm_start_sha256": actual,
        "master_domain_warm_start_sha256_matches": bool((not expected) or expected == actual),
    }


def verify_manifest_warm_start(manifest: Mapping[str, Any], warm: Any) -> None:
    status = manifest_warm_start_hash_status(manifest, warm)
    expected = str(status["master_domain_expected_warm_start_sha256"])
    actual = str(status["master_domain_actual_warm_start_sha256"])
    if expected and expected != actual:
        raise MasterDomainError(f"master domain warm-start hash mismatch: expected={expected}, actual={actual}")
