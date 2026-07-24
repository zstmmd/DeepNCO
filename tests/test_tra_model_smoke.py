from __future__ import annotations

from Gurobi.global_xyzu import GlobalXYZUConfig
from Gurobi.master_domain_fingerprint import build_domain_partitions
from Gurobi.tra_initial import build_canonical_initial_state
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_templates import compile_paper_tra_templates
from problemDto.createInstance import CreateOFSProblem


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
