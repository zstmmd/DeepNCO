from __future__ import annotations

from typing import Any, Dict, Hashable

import gurobipy as gp

from Gurobi.tra_model_state import PersistentCompiledTemplate
from Gurobi.tra_neighborhood import DualBlockSpec, NeighborhoodLevel, Procedure
from Gurobi.tra_projection import (
    INACTIVE_LABEL,
    ProjectionError,
    ProjectionRegistry,
    StructuralShell,
)


def carrier_variables(
    registry: ProjectionRegistry,
    block_name: str,
) -> Dict[Hashable, Dict[int, Any]]:
    grouped: Dict[Hashable, Dict[int, Any]] = {}
    if block_name == "x_group":
        for (unit_id, slot_id), variable in registry.family("x").items():
            grouped.setdefault(unit_id, {})[int(slot_id)] = variable
    elif block_name == "s_visit":
        for (slot_id, stack_id, station_id), variable in registry.family(
            "pair_activate"
        ).items():
            grouped.setdefault((int(slot_id), int(stack_id)), {})[
                int(station_id)
            ] = variable
    elif block_name == "r_assign":
        for (slot_id, robot_id), variable in registry.family("slot_robot").items():
            grouped.setdefault(int(slot_id), {})[int(robot_id)] = variable
    else:
        raise ProjectionError(f"unknown projection block: {block_name}")
    return grouped


def local_hamming_expression(
    registry: ProjectionRegistry,
    shell: StructuralShell,
    block_name: str,
) -> gp.LinExpr:
    incumbent = shell.projection.block(block_name)
    grouped = carrier_variables(registry, block_name)
    if set(grouped) != set(incumbent):
        raise ProjectionError(f"{block_name} carrier domain differs from incumbent")
    expression = gp.LinExpr(0.0)
    for carrier, labels in grouped.items():
        old_label = int(incumbent[carrier])
        if old_label == INACTIVE_LABEL:
            expression += 2.0 * gp.quicksum(labels.values())
        else:
            if old_label not in labels:
                raise ProjectionError(
                    f"incumbent label {old_label} is outside {block_name} domain"
                )
            expression += 2.0 * (1.0 - labels[old_label])
    return expression


def add_shell_exclusion(
    template: PersistentCompiledTemplate,
    shell: StructuralShell,
    block_name: str,
    *,
    index: int,
) -> None:
    incumbent = shell.projection.block(block_name)
    grouped = carrier_variables(template.registry, block_name)
    matches = gp.LinExpr(0.0)
    for carrier, labels in grouped.items():
        label = int(incumbent[carrier])
        if label == INACTIVE_LABEL:
            matches += 1.0 - gp.quicksum(labels.values())
        else:
            matches += labels[label]
    template.add_constraint(
        matches <= max(0, len(grouped) - 1),
        name=f"TRA_Phase2_Exclude_{block_name}_{index}",
    )


def fix_complement_blocks(
    template: PersistentCompiledTemplate,
    shell: StructuralShell,
    procedure: Procedure,
) -> None:
    plan = shell.fixing_plan(template.registry)
    released = Procedure(procedure).released_block
    if released != "x_group":
        template.fix_binary_families(plan, families=("x",))
    if released != "s_visit":
        template.fix_binary_families(plan, families=("pair_activate",))
        template.add_station_marginal_fixings(plan, prefix=f"TRA_{procedure.value}")
    if released != "r_assign":
        template.fix_binary_families(plan, families=("slot_robot",))


def fix_dual_complement_blocks(
    template: PersistentCompiledTemplate,
    shell: StructuralShell,
    spec: DualBlockSpec,
) -> None:
    plan = shell.fixing_plan(template.registry)
    released = {str(block) for block in spec.released_blocks}
    if "x_group" not in released:
        template.fix_binary_families(plan, families=("x",))
    if "s_visit" not in released:
        template.fix_binary_families(plan, families=("pair_activate",))
        template.add_station_marginal_fixings(plan, prefix=f"TRA_{spec.name}")
    if "r_assign" not in released:
        template.fix_binary_families(plan, families=("slot_robot",))


def apply_dual_block_neighborhood(
    template: PersistentCompiledTemplate,
    shell: StructuralShell,
    spec: DualBlockSpec,
) -> None:
    """Apply a diagnostic local branch that releases two primary blocks."""

    fix_dual_complement_blocks(template, shell, spec)
    distances = []
    for block_name in spec.released_blocks:
        block = str(block_name)
        distance = local_hamming_expression(template.registry, shell, block)
        template.add_constraint(
            distance >= 2,
            name=f"TRA_{spec.name}_{block}_HammingLB",
        )
        distances.append(distance)
    template.add_constraint(
        gp.quicksum(distances) <= max(0, int(spec.hamming_limit)),
        name=f"TRA_{spec.name}_CombinedHammingUB",
    )


def apply_local_neighborhood(
    template: PersistentCompiledTemplate,
    shell: StructuralShell,
    procedure: Procedure,
    neighborhood: NeighborhoodLevel,
) -> None:
    procedure = Procedure(procedure)
    neighborhood = NeighborhoodLevel(neighborhood)
    fix_complement_blocks(template, shell, procedure)
    block_name = procedure.released_block
    distance = local_hamming_expression(template.registry, shell, block_name)
    if neighborhood is NeighborhoodLevel.N1:
        template.add_constraint(
            distance == 2,
            name=f"TRA_{procedure.value}_N1_Hamming",
        )
        if procedure is Procedure.F2:
            grouped = carrier_variables(template.registry, "x_group")
            active_slots = sorted(
                {int(slot_id) for slot_id in shell.projection.x_group.values()}
            )
            for carrier_labels in grouped.values():
                for slot_id, variable in carrier_labels.items():
                    if int(slot_id) not in active_slots:
                        template.fix_variable(variable, 0)
            for slot_id in active_slots:
                variables = [
                    labels[slot_id]
                    for labels in grouped.values()
                    if slot_id in labels
                ]
                template.add_constraint(
                    gp.quicksum(variables) >= 1,
                    name=f"TRA_F2_N1_SourceNonempty_{slot_id}",
                )
    elif neighborhood is NeighborhoodLevel.N2:
        template.add_constraint(
            distance == 4,
            name=f"TRA_{procedure.value}_N2_Hamming",
        )
        incumbent = shell.projection.block(block_name)
        grouped = carrier_variables(template.registry, block_name)
        labels = sorted(
            {
                label
                for carrier_labels in grouped.values()
                for label in carrier_labels
            }
        )
        for label in labels:
            old_count = sum(
                int(value) == int(label)
                for value in incumbent.values()
            )
            variables = [
                carrier_labels[label]
                for carrier_labels in grouped.values()
                if label in carrier_labels
            ]
            template.add_constraint(
                gp.quicksum(variables) == int(old_count),
                name=f"TRA_N2_LabelCount_{block_name}_{label}",
            )
    else:
        template.add_constraint(
            distance >= 6,
            name=f"TRA_{procedure.value}_N3_HammingLB",
        )
        template.add_constraint(
            distance <= 8,
            name=f"TRA_{procedure.value}_N3_HammingUB",
        )
