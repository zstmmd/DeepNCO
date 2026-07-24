from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence


DOMAIN_FAMILIES = (
    "x",
    "a",
    "sku_use",
    "y",
    "flip",
    "sort",
    "carry",
    "hit",
    "noise",
    "flip_hit",
    "pair_activate",
    "pass_x",
    "route_owner",
    "route_arc",
    "route_time",
    "route_load",
    "route_finish",
    "slot_robot",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _json_atom(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return "inf" if value > 0 else "-inf"
    if value is None:
        return None
    return str(value)


def encode_key(key: Any) -> list[Any]:
    if isinstance(key, tuple):
        return [_json_atom(value) for value in key]
    return [_json_atom(key)]


def decode_key(row: Sequence[Any]) -> Any:
    values = tuple(row)
    return values[0] if len(values) == 1 else values


def _sorted_key_rows(keys: Iterable[Any]) -> list[list[Any]]:
    rows = [encode_key(key) for key in keys]
    return sorted(rows, key=canonical_json)


def partition_from_keys(keys: Iterable[Any]) -> Dict[str, Any]:
    rows = _sorted_key_rows(keys)
    return {
        "count": len(rows),
        "keys": rows,
        "sha256": sha256_payload(rows),
    }


def build_domain_partitions(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    partitions: Dict[str, Dict[str, Any]] = {}
    for family in DOMAIN_FAMILIES:
        container = payload.get(family)
        keys = [] if container is None else list(container.keys())
        partitions[family] = partition_from_keys(keys)
    return partitions


def _linexpr_row(expression: Any) -> list[list[Any]]:
    terms = []
    try:
        for index in range(int(expression.size())):
            variable = expression.getVar(index)
            coefficient = float(expression.getCoeff(index))
            terms.append([str(variable.VarName), coefficient])
    except Exception:
        return []
    return sorted(terms, key=lambda row: row[0])


def _hash_rows(rows: Iterable[Any]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def build_model_fingerprints(model: Any) -> Dict[str, Any]:
    if model is None or not hasattr(model, "getVars"):
        empty_hash = hashlib.sha256(b"").hexdigest()
        return {
            "variable_count": 0,
            "variable_domain_sha256": empty_hash,
            "objective_sha256": sha256_payload({"sense": 1, "terms": []}),
            "linear_constraint_count": 0,
            "linear_constraints_sha256": empty_hash,
            "general_constraint_count": 0,
            "general_constraints_sha256": empty_hash,
        }

    try:
        model.update()
    except Exception:
        pass

    variables = sorted(model.getVars(), key=lambda variable: str(variable.VarName))
    variable_rows = [
        [
            str(variable.VarName),
            str(variable.VType),
            _json_atom(float(variable.LB)),
            _json_atom(float(variable.UB)),
        ]
        for variable in variables
    ]
    variable_count, variable_hash = _hash_rows(variable_rows)
    objective_terms = [
        [str(variable.VarName), float(variable.Obj)]
        for variable in variables
        if abs(float(variable.Obj)) > 0.0
    ]
    objective_payload = {
        "sense": int(getattr(model, "ModelSense", 1)),
        "terms": objective_terms,
    }

    def linear_rows():
        for constraint in sorted(model.getConstrs(), key=lambda item: str(item.ConstrName)):
            expression = model.getRow(constraint)
            yield [
                str(constraint.ConstrName),
                str(constraint.Sense),
                _json_atom(float(constraint.RHS)),
                _linexpr_row(expression),
            ]

    linear_count, linear_hash = _hash_rows(linear_rows())

    def general_rows():
        for constraint in sorted(model.getGenConstrs(), key=lambda item: str(item.GenConstrName)):
            constraint_type = int(constraint.GenConstrType)
            row: list[Any] = [str(constraint.GenConstrName), constraint_type]
            try:
                data = model.getGenConstrIndicator(constraint)
                binary, binary_value, expression, sense, rhs = data
                row.extend(
                    [
                        str(binary.VarName),
                        int(bool(binary_value)),
                        str(sense),
                        _json_atom(float(rhs)),
                        _linexpr_row(expression),
                    ]
                )
            except Exception:
                row.append("unsupported-general-constraint-payload")
            yield row

    general_count, general_hash = _hash_rows(general_rows())
    return {
        "variable_count": variable_count,
        "variable_domain_sha256": variable_hash,
        "objective_sha256": sha256_payload(objective_payload),
        "linear_constraint_count": linear_count,
        "linear_constraints_sha256": linear_hash,
        "general_constraint_count": general_count,
        "general_constraints_sha256": general_hash,
    }


def _number_rows(values: Mapping[Any, Any]) -> list[list[Any]]:
    return [
        [encode_key(key), _json_atom(float(value))]
        for key, value in sorted(values.items(), key=lambda item: canonical_json(encode_key(item[0])))
    ]


def build_numeric_bounds(payload: Mapping[str, Any], model_fingerprints: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "slot_time_ub": _json_atom(float(payload.get("slot_time_ub", 0.0) or 0.0)),
        "route_big_m": _json_atom(float(payload.get("route_big_m", 0.0) or 0.0)),
        "route_node_time_ub": _number_rows(dict(payload.get("route_node_time_ub", {}) or {})),
        "route_arc_time_m": _number_rows(dict(payload.get("route_arc_time_m", {}) or {})),
        "pickup_service_lb_by_node": _number_rows(dict(payload.get("pickup_service_lb_by_node", {}) or {})),
        "pickup_service_ub_by_node": _number_rows(dict(payload.get("pickup_service_ub_by_node", {}) or {})),
        "variable_domain_sha256": str(model_fingerprints.get("variable_domain_sha256", "")),
    }


def build_route_node_contract(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for raw_node_id, node in dict(payload.get("route_nodes", {}) or {}).items():
        rows.append(
            {
                "node_id": int(getattr(node, "node_id", raw_node_id)),
                "kind": str(getattr(node, "kind", "")),
                "task_key": int(getattr(node, "task_key", -1)),
                "slot_id": int(getattr(node, "slot_id", -1)),
                "stack_id": int(getattr(node, "stack_id", -1)),
                "station_id": int(getattr(node, "station_id", -1)),
                "x": float(getattr(node, "x", 0.0)),
                "y": float(getattr(node, "y", 0.0)),
                "robot_id": int(getattr(node, "robot_id", -1)),
            }
        )
    return sorted(rows, key=lambda row: row["node_id"])


def build_pruning_rules(cfg: Any, diagnostics: Mapping[str, Any], partitions: Mapping[str, Any]) -> list[Dict[str, Any]]:
    specs = (
        ("candidate_stack_topk", "heuristic"),
        ("max_candidate_stacks_per_order", "heuristic"),
        ("candidate_station_topk_per_stack", "heuristic"),
        ("route_pickup_neighbor_limit", "heuristic"),
        ("enable_tight_slot_upper_bound", "heuristic"),
        ("enable_warm_candidate_stack_prune", "heuristic"),
        ("route_arc_prune", "heuristic"),
        ("enable_route_time_window_arc_prune", "heuristic"),
        ("enable_route_load_interval_arc_prune", "safe"),
        ("enable_route_directional_arc_prune", "heuristic"),
    )
    rows = []
    for rule_id, classification in specs:
        value = getattr(cfg, rule_id, diagnostics.get(rule_id)) if cfg is not None else diagnostics.get(rule_id)
        rows.append(
            {
                "rule_id": rule_id,
                "classification": classification,
                "value": _json_atom(value),
                "source": "compiled_config",
                "before_count": diagnostics.get(f"{rule_id}_before_count"),
                "after_count": diagnostics.get(f"{rule_id}_after_count"),
            }
        )
    rows.append(
        {
            "rule_id": "compiled_domain_partitions",
            "classification": "safe",
            "value": True,
            "source": "manifest",
            "before_count": None,
            "after_count": sum(int(partition.get("count", 0) or 0) for partition in partitions.values()),
        }
    )
    return rows
