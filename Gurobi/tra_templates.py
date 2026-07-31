from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from Gurobi.global_xyzu import (
    CompiledGlobalXYZUModel,
    GlobalXYZUConfig,
    GlobalXYZUSolver,
)
from Gurobi.master_domain import (
    MasterDomainError,
    build_master_domain_manifest,
    normalize_master_domain_manifest,
    prepared_domain_from_manifest,
)
from Gurobi.tra_comproc import ComProcEvaluator
from Gurobi.tra_inner import PaperInnerTemplate
from Gurobi.tra_model_state import PersistentCompiledTemplate
from Gurobi.tra_outer import PaperOuterTemplate
from Gurobi.tra_template_contract import assert_shared_template_contract
from Gurobi.tra_vns import PaperVNSGenerator


@dataclass(frozen=True)
class PaperTRATemplates:
    manifest: Mapping[str, Any]
    full_compiled: CompiledGlobalXYZUModel
    inner_compiled: CompiledGlobalXYZUModel
    outer: PaperOuterTemplate
    inner: PaperInnerTemplate
    comproc: ComProcEvaluator
    vns: PaperVNSGenerator


def global_config_from_policy(
    policy_values: Mapping[str, Any],
    *,
    gurobi_output: bool = False,
    gurobi_seed: Optional[int] = None,
) -> GlobalXYZUConfig:
    unknown = set(policy_values) - set(GlobalXYZUConfig.__dataclass_fields__)
    if unknown:
        raise MasterDomainError(f"sanitized policy has unsupported GlobalXYZU fields: {sorted(unknown)}")
    cfg = GlobalXYZUConfig(**dict(policy_values))
    cfg.gurobi_output = bool(gurobi_output)
    cfg.gurobi_seed = int(gurobi_seed) if gurobi_seed is not None else cfg.gurobi_seed
    cfg.enable_warm_start = True
    cfg.fixgurobi_no_warm_start = False
    cfg.fixgurobi_allow_warm_start_fallback = False
    cfg.gurobi_best_obj_stop = None
    cfg.gurobi_cutoff = None
    # Wall-clock-limited OR-Tools local search changes protected warm arcs.
    # Formal TRA keeps the deterministic first solution unless the archived
    # policy explicitly records the guided-local-search setting.
    if "warm_start_sp4_guided_local_search" not in policy_values:
        cfg.warm_start_sp4_guided_local_search = False
    cfg.master_domain_manifest = None
    cfg.master_domain_strict = False
    cfg.master_domain_enforce_warm_start_contract = True
    cfg.tra_inner_no_station_wait = False
    for field_name in (
        "fixed_work_units_by_order_slot",
        "fixed_station_rank_by_order_slot",
        "fixed_z_descriptors_by_order_slot",
        "fixed_used_stack_ids_by_order",
        "fixed_route_arcs_by_robot",
        "fixed_route_task_sequence_by_robot",
        "fixed_route_node_sequence_by_robot",
    ):
        setattr(cfg, field_name, None)
    return cfg


def compile_paper_tra_templates(
    problem: Any,
    cfg: GlobalXYZUConfig,
    *,
    canonical_seed: int,
    instance_name: str = "",
) -> PaperTRATemplates:
    """Compile both templates before the formal timer and lock them to one domain."""

    frozen_manifest = (
        normalize_master_domain_manifest(cfg.master_domain_manifest)
        if cfg.master_domain_manifest
        else None
    )
    full_cfg = copy.deepcopy(cfg)
    full_cfg.master_domain_manifest = (
        dict(frozen_manifest) if frozen_manifest is not None else None
    )
    full_cfg.master_domain_strict = bool(frozen_manifest is not None)
    full_cfg.master_domain_enforce_warm_start_contract = bool(frozen_manifest is None)
    full_cfg.tra_inner_no_station_wait = False
    full_solver = GlobalXYZUSolver()
    full_compiled = full_solver.compile_model(copy.deepcopy(problem), full_cfg)
    manifest = (
        dict(frozen_manifest)
        if frozen_manifest is not None
        else normalize_master_domain_manifest(
            build_master_domain_manifest(
                full_compiled,
                canonical_seed=int(canonical_seed),
                instance_name=instance_name,
            )
        )
    )
    prepared_domain = prepared_domain_from_manifest(manifest)
    prepared_domain.assert_payload_compatible(full_compiled.vars_payload)
    full_compiled.cfg.master_domain_manifest = dict(manifest)
    full_compiled.cfg.master_domain_strict = True
    full_compiled.diagnostics["master_domain_sha256"] = str(manifest["manifest_sha256"])
    full_compiled.diagnostics["master_domain_payload_verified"] = True

    inner_cfg = copy.deepcopy(cfg)
    inner_cfg.master_domain_manifest = dict(manifest)
    inner_cfg.master_domain_strict = True
    inner_cfg.master_domain_enforce_warm_start_contract = False
    inner_cfg.tra_inner_no_station_wait = True
    # The formal child shares the full model's manifest/domain but should not
    # regenerate a separate warm start; large M cases can otherwise produce a
    # different protected warm fingerprint while retaining the same domain.
    inner_cfg.enable_warm_start = False
    inner_solver = GlobalXYZUSolver()
    inner_compiled = inner_solver.compile_model(copy.deepcopy(problem), inner_cfg)
    prepared_domain.assert_payload_compatible(inner_compiled.vars_payload)
    shared_contract = assert_shared_template_contract(full_compiled, inner_compiled)
    for compiled in (full_compiled, inner_compiled):
        compiled.diagnostics.update(
            {
                "tra_shared_variable_families_sha256": shared_contract.variable_families_sha256,
                "tra_shared_route_semantics_sha256": shared_contract.route_semantics_sha256,
                "tra_shared_route_constraints_sha256": shared_contract.route_constraints_sha256,
                "tra_shared_route_constraint_count": shared_contract.route_constraint_count,
            }
        )

    outer_template = PaperOuterTemplate(
        PersistentCompiledTemplate(full_compiled, solver=full_solver)
    )
    inner_template = PaperInnerTemplate(
        PersistentCompiledTemplate(inner_compiled, solver=inner_solver),
        elite_pool_size=24,
    )
    return PaperTRATemplates(
        manifest=manifest,
        full_compiled=full_compiled,
        inner_compiled=inner_compiled,
        outer=outer_template,
        inner=inner_template,
        comproc=ComProcEvaluator(
            full_compiled.model,
            full_compiled.vars_payload,
            outer_template.verifier,
        ),
        vns=PaperVNSGenerator(inner_template.template.registry),
    )
