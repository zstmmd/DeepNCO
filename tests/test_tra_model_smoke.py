from __future__ import annotations

import math

from gurobipy import GRB

from Gurobi.global_xyzu import GlobalXYZUConfig
from Gurobi.master_domain_fingerprint import build_domain_partitions
from Gurobi.tra_initial import build_canonical_initial_state
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_templates import compile_paper_tra_templates, global_config_from_policy
from problemDto.createInstance import CreateOFSProblem


def _defined_start_count(model) -> int:
    count = 0
    for variable in model.getVars():
        try:
            value = float(variable.Start)
        except Exception:
            continue
        if math.isfinite(value) and abs(value) < 0.5 * float(GRB.UNDEFINED):
            count += 1
    return count


def test_policy_can_preserve_explicit_warm_start_guided_local_search() -> None:
    cfg = global_config_from_policy(
        {
            "warm_start_sp4_guided_local_search": True,
        },
        gurobi_output=False,
        gurobi_seed=42,
    )

    assert cfg.warm_start_sp4_guided_local_search is True


def test_full_and_no_wait_templates_share_the_manifest_domain() -> None:
    problem = CreateOFSProblem.generate_problem_by_scale("TEST", seed=42)
    cfg = GlobalXYZUConfig(
        time_limit_sec=2.0,
        mip_gap=0.1,
        gurobi_output=False,
        warm_start_sp4_time_limit_sec=1,
        gurobi_threads=1,
        gurobi_seed=42,
        enable_resource_lex_symmetry=False,
        enable_slot_lex_symmetry=False,
    )

    templates = compile_paper_tra_templates(
        problem,
        cfg,
        canonical_seed=42,
        instance_name="TEST",
    )

    assert templates.manifest["schema_version"] == 3
    assert templates.full_compiled.cfg.enable_warm_start is True
    assert templates.inner_compiled.cfg.enable_warm_start is False
    assert templates.inner_compiled.diagnostics["tra_inner_no_station_wait"] is True
    assert templates.full_compiled.diagnostics["tra_inner_no_station_wait"] is False
    full_names = {variable.VarName for variable in templates.full_compiled.model.getVars()}
    inner_names = {variable.VarName for variable in templates.inner_compiled.model.getVars()}
    assert any(name.startswith("station_arrival_clock") for name in full_names)
    assert not any(name.startswith("station_arrival_clock") for name in inner_names)
    assert build_domain_partitions(templates.full_compiled.vars_payload) == build_domain_partitions(
        templates.inner_compiled.vars_payload
    )
    assert templates.full_compiled.diagnostics["tra_shared_route_constraint_count"] > 0
    assert (
        templates.full_compiled.diagnostics["tra_shared_route_constraints_sha256"]
        == templates.inner_compiled.diagnostics["tra_shared_route_constraints_sha256"]
    )
    assert _defined_start_count(templates.full_compiled.model) > 0
    assert _defined_start_count(templates.outer.template.model) == _defined_start_count(
        templates.full_compiled.model
    )
    assert templates.inner_compiled.diagnostics["master_domain_numeric_bounds_applied"] is True
    assert {
        key: (
            task.task_key,
            task.pickup_node,
            task.delivery_node,
        )
        for key, task in templates.full_compiled.vars_payload["route_tasks"].items()
    } == {
        key: (
            task.task_key,
            task.pickup_node,
            task.delivery_node,
        )
        for key, task in templates.inner_compiled.vars_payload["route_tasks"].items()
    }

    initial = build_canonical_initial_state(
        templates.outer.template,
        templates.outer.verifier,
    )
    assert initial.search_shell.projection.x_group
    assert initial.search_shell.projection.s_visit
    assert initial.search_shell.projection.r_assign

    inner = templates.inner.solve(
        initial.search_shell,
        procedure=Procedure.F1,
        neighborhood=NeighborhoodLevel.N1,
        time_limit_sec=5.0,
        incumbent_objective=None,
        start_values=initial.start_values,
    )
    assert inner.status != "INNER_FAILED", inner.error
    assert inner.candidates, {
        key: value
        for key, value in initial.search_shell.projection.s_visit.items()
        if value >= 0
    }
    candidate = templates.comproc.evaluate_many(inner.candidates)[0]
    assert candidate.comproc is not None
    assert candidate.comproc.dp1.feasible
    assert candidate.comproc.dp2.feasible
    assert candidate.comproc.full_start is not None
    assert candidate.comproc.full_start.validation.feasible
    outer = templates.outer.solve_shell(
        candidate.shell,
        time_limit_sec=2.0,
        incumbent_objective=None,
        start_values=candidate.comproc.full_start.values_by_name,
        formal_elapsed_at_start=0.0,
    )
    assert outer.disposition is not OuterDisposition.HARD_FAILURE, outer.error
    assert outer.disposition in {
        OuterDisposition.ACCEPTED,
        OuterDisposition.PROVED_REJECT,
        OuterDisposition.UNRESOLVED,
    }
    assert outer.full_start_complete
    if candidate.comproc.feasible:
        assert outer.full_start_feasible
        assert outer.installed_start_count == len(templates.outer.template.model.getVars())
    else:
        assert not outer.full_start_feasible
        assert any(
            code.startswith("START_SEMANTIC:")
            for code in outer.full_start_error_codes
        )
    if outer.accepted is not None:
        assert outer.accepted.internal_feasible

    exhausted = templates.outer.solve_shell(
        candidate.shell,
        time_limit_sec=-1.0,
        incumbent_objective=None,
        start_values=candidate.comproc.full_start.values_by_name,
        formal_elapsed_at_start=0.0,
    )
    assert exhausted.disposition is OuterDisposition.BUDGET_EXHAUSTED


def test_compile_templates_can_reuse_frozen_master_domain_manifest() -> None:
    problem = CreateOFSProblem.generate_problem_by_scale("TEST", seed=42)
    cfg = GlobalXYZUConfig(
        time_limit_sec=2.0,
        mip_gap=0.1,
        gurobi_output=False,
        warm_start_sp4_time_limit_sec=1,
        gurobi_threads=1,
        gurobi_seed=42,
        enable_resource_lex_symmetry=False,
        enable_slot_lex_symmetry=False,
    )
    frozen = compile_paper_tra_templates(
        problem,
        cfg,
        canonical_seed=42,
        instance_name="TEST",
    ).manifest

    replay_cfg = GlobalXYZUConfig(
        time_limit_sec=2.0,
        mip_gap=0.1,
        gurobi_output=False,
        warm_start_sp4_time_limit_sec=1,
        gurobi_threads=1,
        gurobi_seed=42,
        enable_resource_lex_symmetry=False,
        enable_slot_lex_symmetry=False,
        master_domain_manifest=dict(frozen),
        master_domain_strict=True,
    )
    replayed = compile_paper_tra_templates(
        problem,
        replay_cfg,
        canonical_seed=7,
        instance_name="SHOULD_NOT_REBUILD_DOMAIN",
    )

    assert replayed.manifest == frozen
    assert replayed.full_compiled.cfg.master_domain_strict is True
    assert replayed.inner_compiled.cfg.master_domain_strict is True


def test_frozen_master_domain_allows_regenerated_warm_fingerprint_mismatch() -> None:
    problem = CreateOFSProblem.generate_problem_by_scale("TEST", seed=42)
    cfg = GlobalXYZUConfig(
        time_limit_sec=2.0,
        mip_gap=0.1,
        gurobi_output=False,
        warm_start_sp4_time_limit_sec=1,
        gurobi_threads=1,
        gurobi_seed=42,
        enable_resource_lex_symmetry=False,
        enable_slot_lex_symmetry=False,
    )
    frozen = compile_paper_tra_templates(
        problem,
        cfg,
        canonical_seed=42,
        instance_name="TEST",
    ).manifest

    replay_cfg = GlobalXYZUConfig(
        time_limit_sec=2.0,
        mip_gap=0.1,
        gurobi_output=False,
        warm_start_sp4_time_limit_sec=0,
        gurobi_threads=1,
        gurobi_seed=42,
        enable_resource_lex_symmetry=False,
        enable_slot_lex_symmetry=False,
        master_domain_manifest=dict(frozen),
        master_domain_strict=True,
    )
    replayed = compile_paper_tra_templates(
        problem,
        replay_cfg,
        canonical_seed=42,
        instance_name="TEST",
    )

    assert replayed.manifest == frozen
    assert replayed.full_compiled.diagnostics["master_domain_payload_verified"] is True
    assert replayed.full_compiled.diagnostics["master_domain_warm_start_sha256_matches"] is False
