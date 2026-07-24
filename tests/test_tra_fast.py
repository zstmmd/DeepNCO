from __future__ import annotations

from pathlib import Path
import json

import pytest

from Gurobi.global_xyzu import GlobalXYZUConfig
from Gurobi.tra_fast_search import PaperFastNeighborhoodTemplate
from Gurobi.tra_initial import build_canonical_initial_state
from Gurobi.tra_model_state import PersistentCompiledTemplate
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_templates import compile_paper_tra_templates
from experiments.m_tra_fast_policy import load_fast_runtime_budget
from experiments.freeze_m_tra_fast_runtime_policy import build_policy
from experiments.m_tra_policy import assert_target_blind_payload
from experiments.run_m_tra_fast_formal import build_parser
from problemDto.createInstance import CreateOFSProblem


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_fast_runner_has_no_target_or_solution_replay_argument() -> None:
    destinations = {action.dest.lower() for action in build_parser()._actions}

    assert not any(
        marker in destination
        for destination in destinations
        for marker in ("target", "replay", "cutoff", "best_obj")
    )


def test_fast_budget_is_eighty_percent_of_frozen_tra_gurobi_runtime() -> None:
    budget = load_fast_runtime_budget(
        ROOT_DIR / "experiments/configs/m_tra_fast_runtime_budgets_v1.json",
        "M1",
    )

    assert budget.hard_limit_sec == pytest.approx(
        0.8 * budget.source_tra_gurobi_runtime_sec
    )
    assert budget.regular_quota_sec + budget.reserve_quota_sec == pytest.approx(
        budget.hard_limit_sec
    )


def test_freezer_removes_objective_values_from_fast_runtime_policy(
    tmp_path: Path,
) -> None:
    paths = {}
    for index in range(1, 10):
        case_id = f"M{index}"
        path = tmp_path / f"{case_id}.json"
        path.write_text(
            json.dumps(
                {
                    "target_cmax": 1000 + index,
                    "cmax_equal": True,
                    "runtime_ok": True,
                    "first_verified_target_time_sec": 10.0 * index,
                }
            ),
            encoding="utf-8",
        )
        paths[case_id] = path

    payload = build_policy(paths)

    assert_target_blind_payload(payload)
    assert payload["cases"]["M3"]["hard_limit_sec"] == pytest.approx(24.0)


def test_fast_full_model_neighborhood_smoke() -> None:
    problem = CreateOFSProblem.generate_problem_by_scale("TEST", seed=42)
    cfg = GlobalXYZUConfig(
        time_limit_sec=3.0,
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
    fast = PaperFastNeighborhoodTemplate(
        PersistentCompiledTemplate(
            templates.full_compiled,
            solver=templates.outer.template.solver,
        ),
        verifier=templates.outer.verifier,
    )
    initial = build_canonical_initial_state(fast.template, fast.verifier)
    seeds = templates.vns.generate(
        initial.search_shell,
        procedure=Procedure.F1,
        neighborhood=NeighborhoodLevel.N1,
    )

    result = fast.solve(
        initial.search_shell,
        procedure=Procedure.F1,
        neighborhood=NeighborhoodLevel.N1,
        time_limit_sec=3.0,
        incumbent_objective=None,
        start_values=initial.start_values,
        vns_start_values=tuple(seed.values_by_name for seed in seeds),
        formal_elapsed_at_start=0.0,
    )

    assert result.disposition is not OuterDisposition.HARD_FAILURE, result.error
    assert all(item.internal_feasible for item in result.verified_snapshots)
