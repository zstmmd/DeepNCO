from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from Gurobi.global_xyzu import CompiledGlobalXYZUModel
from Gurobi.master_domain_fingerprint import (
    DOMAIN_FAMILIES,
    build_route_node_contract,
    canonical_json,
    encode_key,
    sha256_payload,
)


class TemplateContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class SharedTemplateContract:
    variable_families_sha256: str
    route_semantics_sha256: str
    route_constraints_sha256: str
    route_constraint_count: int


_ROUTE_VARIABLE_FAMILIES = (
    "pass_x",
    "route_owner",
    "route_arc",
    "route_time",
    "route_load",
    "route_finish",
)


def _number(value: Any) -> int | float | str:
    numeric = float(value)
    if math.isfinite(numeric):
        return int(numeric) if numeric.is_integer() else numeric
    return "inf" if numeric > 0 else "-inf"


def _family_rows(payload: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for family_name in DOMAIN_FAMILIES:
        family = payload.get(family_name)
        for key, variable in dict(family or {}).items():
            rows.append(
                [
                    str(family_name),
                    encode_key(key),
                    str(variable.VarName),
                    str(variable.VType),
                    _number(variable.LB),
                    _number(variable.UB),
                ]
            )
    return sorted(rows, key=lambda row: (row[0], str(row[1])))


def _route_semantics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    tasks = []
    for task_key, task in dict(payload.get("route_tasks", {}) or {}).items():
        tasks.append(
            {
                "task_key": int(getattr(task, "task_key", task_key)),
                "slot_id": int(getattr(task, "slot_id")),
                "stack_id": int(getattr(task, "stack_id")),
                "station_id": int(getattr(task, "station_id")),
                "pickup_node": int(getattr(task, "pickup_node")),
                "delivery_node": int(getattr(task, "delivery_node")),
                "estimated_load": int(getattr(task, "estimated_load", 0)),
            }
        )

    def keyed_numbers(name: str) -> list[list[Any]]:
        return sorted(
            [
                [encode_key(key), _number(value)]
                for key, value in dict(payload.get(name, {}) or {}).items()
            ],
            key=lambda row: str(row[0]),
        )

    return {
        "route_nodes": build_route_node_contract(payload),
        "route_tasks": sorted(tasks, key=lambda row: row["task_key"]),
        "route_arcs": sorted([list(map(int, arc)) for arc in payload.get("route_arcs", ())]),
        "protected_route_arcs": sorted(
            [list(map(int, arc)) for arc in payload.get("protected_route_arcs", ())]
        ),
        "route_tau": keyed_numbers("route_tau"),
        "route_node_time_ub": keyed_numbers("route_node_time_ub"),
        "route_arc_time_m": keyed_numbers("route_arc_time_m"),
        "route_start_nodes": sorted(
            [list(map(int, item)) for item in dict(payload.get("route_start_nodes", {}) or {}).items()]
        ),
        "route_end_nodes": sorted(
            [list(map(int, item)) for item in dict(payload.get("route_end_nodes", {}) or {}).items()]
        ),
    }


def _expression_terms(expression: Any) -> list[list[Any]]:
    return sorted(
        [
            [str(expression.getVar(index).VarName), _number(expression.getCoeff(index))]
            for index in range(int(expression.size()))
        ],
        key=lambda row: row[0],
    )


def _route_constraint_rows(compiled: CompiledGlobalXYZUModel) -> list[list[Any]]:
    model = compiled.model
    model.update()
    route_names = {
        str(variable.VarName)
        for family_name in _ROUTE_VARIABLE_FAMILIES
        for variable in dict(compiled.vars_payload.get(family_name, {}) or {}).values()
    }
    rows: list[list[Any]] = []
    for constraint in model.getConstrs():
        terms = _expression_terms(model.getRow(constraint))
        if any(name in route_names for name, _coefficient in terms):
            constraint_name = str(constraint.ConstrName)
            if re.fullmatch(r"R\d+", constraint_name):
                constraint_name = ""
            rows.append(
                [
                    "linear",
                    constraint_name,
                    str(constraint.Sense),
                    _number(constraint.RHS),
                    terms,
                ]
            )
    for constraint in model.getGenConstrs():
        name = str(constraint.GenConstrName)
        try:
            binary, binary_value, expression, sense, rhs = model.getGenConstrIndicator(constraint)
        except Exception:
            if name.startswith("Route"):
                rows.append(["general", name, int(constraint.GenConstrType)])
            continue
        terms = _expression_terms(expression)
        binary_name = str(binary.VarName)
        if binary_name in route_names or any(term_name in route_names for term_name, _coefficient in terms):
            rows.append(
                [
                    "indicator",
                    name,
                    binary_name,
                    int(bool(binary_value)),
                    str(sense),
                    _number(rhs),
                    terms,
                ]
            )
    return sorted(rows, key=canonical_json)


def build_shared_template_contract(compiled: CompiledGlobalXYZUModel) -> SharedTemplateContract:
    route_constraints = _route_constraint_rows(compiled)
    return SharedTemplateContract(
        variable_families_sha256=sha256_payload(_family_rows(compiled.vars_payload)),
        route_semantics_sha256=sha256_payload(_route_semantics(compiled.vars_payload)),
        route_constraints_sha256=sha256_payload(route_constraints),
        route_constraint_count=len(route_constraints),
    )


def assert_shared_template_contract(
    full: CompiledGlobalXYZUModel,
    inner: CompiledGlobalXYZUModel,
) -> SharedTemplateContract:
    full_contract = build_shared_template_contract(full)
    inner_contract = build_shared_template_contract(inner)
    for field_name in (
        "variable_families_sha256",
        "route_semantics_sha256",
        "route_constraints_sha256",
        "route_constraint_count",
    ):
        if getattr(full_contract, field_name) != getattr(inner_contract, field_name):
            detail = ""
            if field_name == "variable_families_sha256":
                full_rows = _family_rows(full.vars_payload)
                inner_rows = _family_rows(inner.vars_payload)
                for index in range(max(len(full_rows), len(inner_rows))):
                    full_row = full_rows[index] if index < len(full_rows) else None
                    inner_row = inner_rows[index] if index < len(inner_rows) else None
                    if full_row != inner_row:
                        detail = f", first_difference=full:{full_row!r}/inner:{inner_row!r}"
                        break
            elif field_name == "route_semantics_sha256":
                full_semantics = _route_semantics(full.vars_payload)
                inner_semantics = _route_semantics(inner.vars_payload)
                for component in full_semantics:
                    if full_semantics[component] != inner_semantics.get(component):
                        component_detail = ""
                        full_value = full_semantics[component]
                        inner_value = inner_semantics.get(component)
                        if isinstance(full_value, list) and isinstance(inner_value, list):
                            for index in range(max(len(full_value), len(inner_value))):
                                full_row = full_value[index] if index < len(full_value) else None
                                inner_row = inner_value[index] if index < len(inner_value) else None
                                if full_row != inner_row:
                                    component_detail = (
                                        f", first_difference=full:{full_row!r}/inner:{inner_row!r}"
                                    )
                                    break
                        detail = (
                            f", first_component={component}, "
                            f"full_sha256={sha256_payload(full_semantics[component])}, "
                            f"inner_sha256={sha256_payload(inner_semantics.get(component))}"
                            f"{component_detail}"
                        )
                        break
            elif field_name in {"route_constraints_sha256", "route_constraint_count"}:
                full_rows = _route_constraint_rows(full)
                inner_rows = _route_constraint_rows(inner)
                for index in range(max(len(full_rows), len(inner_rows))):
                    full_row = full_rows[index] if index < len(full_rows) else None
                    inner_row = inner_rows[index] if index < len(inner_rows) else None
                    if full_row != inner_row:
                        detail = f", first_difference=full:{full_row!r}/inner:{inner_row!r}"
                        break
            raise TemplateContractError(
                f"full/inner template contract differs for {field_name}: "
                f"full={getattr(full_contract, field_name)}, "
                f"inner={getattr(inner_contract, field_name)}{detail}"
            )
    return full_contract
