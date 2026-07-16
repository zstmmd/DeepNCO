from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

from experiments.m_current_tra_baselines import (
    CurrentMCase,
    build_gurobi_baseline_rows,
    build_runtime_alias_config,
    build_structure_export_map,
)
from experiments.run_m_current_tra_acceptance import _accept_tra_gurobi, compare_cmax, runtime_speedup_ok
from experiments.run_m_current_tra_acceptance import _run_tra_fast_case, _run_tra_gurobi_case
from experiments.run_m_current_tra_acceptance import _tra_gurobi_internal_budget_args, _tra_gurobi_policy_args


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_baseline_rows_from_gurobi_summary(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "cfg.json"
    summary_path = tmp_path / "result" / "gurobi_summary.json"
    _write_json(config_path, {"configs": {"M1": {"resources": [1, 2, 3]}}})
    _write_json(
        summary_path,
        {
            "status": "OPTIMAL",
            "global_makespan": 100.0,
            "true_global_makespan": 100.0,
            "gurobi_runtime_sec": 123.45,
            "gap": 0.009,
            "diagnostics": {
                "model_best_bound": 99.0,
                "model_var_count_total": 10,
                "model_constr_count_total": 20,
                "time_verify_mismatch": False,
                "warm_start_mip_start_ready": True,
                "warm_start_missing_arc_count": 0,
            },
        },
    )
    from experiments import m_current_tra_baselines as baselines

    monkeypatch.setattr(baselines, "ROOT_DIR", tmp_path)
    case = CurrentMCase("M1", "GUROBI-M1", "cfg.json", "result/gurobi_summary.json")
    row = build_gurobi_baseline_rows([case])[0]
    assert row["case"] == "GUROBI-M1"
    assert row["model_cmax"] == 100.0
    assert row["runtime_sec"] == 123.45
    assert row["model_gap"] == 0.009
    assert row["warm_start_missing_arc_count"] == 0


def test_runtime_alias_contains_m_and_gurobi_m_keys(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "cfg.json"
    _write_json(config_path, {"configs": {"M1": {"resources": [5, 4, 300], "data": [8, 340]}}})
    from experiments import m_current_tra_baselines as baselines

    monkeypatch.setattr(baselines, "ROOT_DIR", tmp_path)
    case = CurrentMCase("M1", "GUROBI-M1", "cfg.json", "unused.json")
    payload = build_runtime_alias_config([case])
    assert payload["configs"]["M1"] == payload["configs"]["GUROBI-M1"]
    assert payload["configs"]["GUROBI-M1"]["resources"] == [5, 4, 300]


def test_m1_baseline_config_regenerates_authoritative_order_total() -> None:
    from experiments.m_current_tra_baselines import load_current_m_cases
    from problemDto.createInstance import CreateOFSProblem

    case = next(item for item in load_current_m_cases() if item.case_id == "M1")
    config_payload = json.loads(case.config_abs_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(case.gurobi_summary_abs_path.read_text(encoding="utf-8"))
    expected_total_qty = sum(int(row.get("total_qty", 0) or 0) for row in summary_payload.get("orders", []))
    configs = config_payload.get("configs", config_payload)

    previous_configs = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    try:
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(previous_configs)
        for name, cfg in dict(configs).items():
            if isinstance(cfg, dict):
                CreateOFSProblem.RUNTIME_SCALE_CONFIGS[str(name).upper()] = dict(cfg)
        problem = CreateOFSProblem.generate_problem_by_scale(case.case_id, seed=42)
    finally:
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = previous_configs

    generated_total_qty = sum(int(getattr(order, "total_qty", 0) or 0) for order in problem.order_list)
    assert generated_total_qty == expected_total_qty
    assert build_gurobi_baseline_rows([case])[0]["sort_hit_tote_threshold"] == 1


def test_m2_baseline_config_regenerates_authoritative_tote_stack() -> None:
    from experiments.m_current_tra_baselines import load_current_m_cases
    from problemDto.createInstance import CreateOFSProblem

    case = next(item for item in load_current_m_cases() if item.case_id == "M2")
    config_payload = json.loads(case.config_abs_path.read_text(encoding="utf-8"))
    configs = config_payload.get("configs", config_payload)

    previous_configs = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    try:
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(previous_configs)
        for name, cfg in dict(configs).items():
            if isinstance(cfg, dict):
                CreateOFSProblem.RUNTIME_SCALE_CONFIGS[str(name).upper()] = dict(cfg)
        problem = CreateOFSProblem.generate_problem_by_scale(case.case_id, seed=42)
    finally:
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = previous_configs

    tote_stack = {}
    for stack in problem.stack_list:
        for tote in getattr(stack, "totes", []) or []:
            tote_stack[int(tote.id)] = int(stack.stack_id)
    assert tote_stack[39] == 44
    row = build_gurobi_baseline_rows([case])[0]
    assert row["candidate_stack_count_max"] == 3
    assert row["enable_slot_lex_symmetry"] is False
    assert row["enable_resource_lex_symmetry"] is False


def test_structure_export_map_uses_algorithm_case_key() -> None:
    case = CurrentMCase(
        "M1",
        "GUROBI-M1",
        "cfg.json",
        "result/example_case/gurobi_summary.json",
    )
    payload = build_structure_export_map([case])
    assert payload["exports"]["GUROBI-M1"] == "result/example_case/gurobi_solution_export"


def test_cmax_lower_than_gurobi_is_fail_fast() -> None:
    result = compare_cmax(100.0, 99.999)
    assert result["lower_than_gurobi"] is True
    assert result["cmax_equal"] is False


def test_cmax_equal_with_float_noise() -> None:
    result = compare_cmax(863.0, 863.000001)
    assert result["cmax_equal"] is True
    assert result["lower_than_gurobi"] is False


def test_runtime_speedup_passes_at_20_percent() -> None:
    result = runtime_speedup_ok(candidate_runtime=800.0, baseline_runtime=1000.0, min_speedup=0.20)
    assert result["runtime_ok"] is True
    assert result["runtime_ratio"] == 0.8


def test_runtime_speedup_fails_when_fast_not_20_percent_faster_than_tra_gurobi() -> None:
    result = runtime_speedup_ok(candidate_runtime=700.0, baseline_runtime=790.0, min_speedup=0.20)
    assert result["runtime_ok"] is False
    assert result["runtime_ratio"] > 0.8


def test_missing_tra_gurobi_summary_is_nonfinite_cmax_failure(tmp_path: Path) -> None:
    args = Namespace(cmax_abs_tol=1e-5, min_tra_gurobi_speedup=0.2)
    accepted = _accept_tra_gurobi(
        args,
        "GUROBI-M1",
        {"model_cmax": 582.0, "runtime_sec": 360.0},
        {},
        tmp_path,
    )
    assert accepted["tra_gurobi_acceptance_ok"] is False
    assert accepted["tra_gurobi_failure_reason"] == "tra_gurobi_missing_or_nonfinite_cmax"


def test_tra_gurobi_command_uses_internal_speed_budget() -> None:
    args = _tra_gurobi_internal_budget_args({"budget_sec": 288.0}, min_speedup=0.2)
    assert "--enforce-speed-budget" in args
    assert "--resource-wall-time-limit-sec" in args
    assert args[args.index("--resource-wall-time-limit-sec") + 1] == "288.0"
    assert "--speed-budget-factor" in args
    assert args[args.index("--speed-budget-factor") + 1] == "0.8"


def test_tra_gurobi_policy_args_replay_baseline_prune_flags() -> None:
    args = _tra_gurobi_policy_args(
        {
            "candidate_stack_topk": 999,
            "max_candidate_stacks_per_order": 0,
            "candidate_station_topk_per_stack": 999,
            "route_pickup_neighbor_limit": 0,
            "route_arc_prune": True,
            "route_time_window_arc_prune": False,
            "route_load_interval_arc_prune": True,
            "sort_hit_tote_threshold": 1,
        }
    )
    assert "--fixgurobi-route-arc-prune" in args
    assert "--no-fixgurobi-route-time-window-arc-prune" in args
    assert "--fixgurobi-route-load-interval-arc-prune" in args
    assert "--fixgurobi-sort-hit-tote-threshold" in args
    assert args[args.index("--fixgurobi-sort-hit-tote-threshold") + 1] == "1"


def test_tra_gurobi_command_can_disable_structure_guidance(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd, **kwargs):
        del kwargs
        captured["cmd"] = list(cmd)
        return {"returncode": 0, "runtime_sec": 0.1, "timeout": False, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr("experiments.run_m_current_tra_acceptance._run_command", fake_run_command)
    monkeypatch.setattr(
        "experiments.run_m_current_tra_acceptance._latest_case_row",
        lambda *_args, **_kwargs: {"case": "GUROBI-M1", "tra_gurobi_cmax": 582.0, "tra_gurobi_total_runtime_sec": 10.0},
    )
    args = Namespace(
        python_bin="python3",
        seed=42,
        min_tra_gurobi_speedup=0.2,
        tra_gurobi_structure_guidance=False,
        tra_gurobi_structure_required=False,
    )

    row = _run_tra_gurobi_case(
        case="GUROBI-M1",
        args=args,
        output_root=tmp_path,
        runtime_alias_json="runtime.json",
        baseline_json="baseline.json",
        structure_exports_json="exports.json",
        gurobi_row={
            "candidate_stack_topk": 999,
            "max_candidate_stacks_per_order": 0,
            "candidate_station_topk_per_stack": 999,
            "route_pickup_neighbor_limit": 0,
            "route_arc_prune": True,
            "route_time_window_arc_prune": False,
            "route_load_interval_arc_prune": True,
            "sort_hit_tote_threshold": 1,
            "solver_mip_gap": 0.002,
        },
        gurobi_runtime=360.0,
    )

    assert "--gurobi-structure-guidance" not in captured["cmd"]
    assert "--gurobi-structure-required" not in captured["cmd"]
    assert "--no-gurobi-structure-guidance" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--fixgurobi-mip-gap") + 1] == "0.002"
    assert row["command_returncode"] == 0


def test_tra_gurobi_command_can_use_no_structure_target_probe_warm_start(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd, **kwargs):
        del kwargs
        captured["cmd"] = list(cmd)
        return {"returncode": 0, "runtime_sec": 0.1, "timeout": False, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr("experiments.run_m_current_tra_acceptance._run_command", fake_run_command)
    monkeypatch.setattr(
        "experiments.run_m_current_tra_acceptance._latest_case_row",
        lambda *args, **kwargs: {"case": "GUROBI-M1", "tra_gurobi_cmax": 582.0, "tra_gurobi_total_runtime_sec": 216.0},
    )
    args = Namespace(
        python_bin="python3",
        seed=42,
        min_tra_gurobi_speedup=0.2,
        tra_gurobi_structure_guidance=False,
        tra_gurobi_structure_required=False,
        tra_gurobi_revolving_mode=False,
        tra_gurobi_known_target_guidance=True,
        tra_gurobi_global_target_probe=True,
        tra_gurobi_global_target_probe_warm_start=True,
        tra_gurobi_global_target_probe_time_factor=1.0,
        tra_gurobi_resource_global_decomp_repair=False,
    )

    _run_tra_gurobi_case(
        case="GUROBI-M1",
        args=args,
        output_root=tmp_path,
        runtime_alias_json="runtime.json",
        baseline_json="baseline.json",
        structure_exports_json="exports.json",
        gurobi_row={
            "candidate_stack_topk": 999,
            "max_candidate_stacks_per_order": 0,
            "candidate_station_topk_per_stack": 999,
            "route_pickup_neighbor_limit": 0,
            "route_arc_prune": True,
            "route_time_window_arc_prune": False,
            "route_load_interval_arc_prune": True,
            "sort_hit_tote_threshold": 1,
            "candidate_stack_count_max": 3,
            "enable_hard_candidate_stack_cap": True,
        },
        gurobi_runtime=360.0,
    )

    assert "--tra-revolving-mode" not in captured["cmd"]
    assert "--known-target-guidance" in captured["cmd"]
    assert "--global-target-probe" in captured["cmd"]
    assert "--global-target-probe-warm-start" in captured["cmd"]
    assert "--global-target-probe-candidate-stack-topk" in captured["cmd"]
    assert "--global-target-probe-hard-candidate-stack-cap" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--global-target-probe-max-candidate-stacks-per-order") + 1] == "3"


def test_tra_fast_command_can_disable_structure_fastpath(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd, **kwargs):
        del kwargs
        captured["cmd"] = list(cmd)
        return {"returncode": 0, "runtime_sec": 0.1, "timeout": False, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr("experiments.run_m_current_tra_acceptance._run_command", fake_run_command)
    monkeypatch.setattr(
        "experiments.run_m_current_tra_acceptance._latest_case_row",
        lambda *_args, **_kwargs: {"case": "GUROBI-M1", "tra_fast_cmax": 582.0, "tra_fast_runtime_sec": 1.0},
    )
    args = Namespace(
        python_bin="python3",
        seed=42,
        min_tra_fast_speedup=0.2,
        tra_fast_calibration_mode="auto",
        tra_fast_structure_fastpath=False,
    )

    _run_tra_fast_case(
        case="GUROBI-M1",
        args=args,
        output_root=tmp_path,
        runtime_alias_json="runtime.json",
        structure_exports_json="exports.json",
        fast_baseline_csv="fast.csv",
        tra_gurobi_runtime=100.0,
    )

    assert "--structure-fastpath" not in captured["cmd"]
    assert "--no-structure-fastpath" in captured["cmd"]
    assert "--direct-calibration-for-m" in captured["cmd"]
    assert "--calibration-full-candidates" in captured["cmd"]
    assert "--calibration-target-obj-slack" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--calibration-target-obj-slack") + 1] == "1.0"


def test_structure_guided_replay_accepts_float_noise_without_local_audit(tmp_path: Path, monkeypatch) -> None:
    if "gurobipy" not in sys.modules:
        fake_gurobipy = types.ModuleType("gurobipy")
        fake_gurobipy.GRB = types.SimpleNamespace(
            BINARY="B",
            INTEGER="I",
            CONTINUOUS="C",
            MINIMIZE=1,
            OPTIMAL=2,
            TIME_LIMIT=9,
            INFEASIBLE=3,
            INTERRUPTED=11,
            CUTOFF=6,
            USER_OBJ_LIMIT=15,
            SUBOPTIMAL=13,
        )
        fake_gurobipy.LinExpr = object
        fake_gurobipy.Model = object
        fake_gurobipy.Var = object
        fake_gurobipy.quicksum = sum
        monkeypatch.setitem(sys.modules, "gurobipy", fake_gurobipy)
    fake_tra = types.ModuleType("Gurobi.tra")
    fake_tra.TRAOptimizer = object
    fake_tra.TRARunConfig = object
    monkeypatch.setitem(sys.modules, "Gurobi.tra", fake_tra)

    from Gurobi import tra_gurobi

    monkeypatch.setattr(
        "sys.argv",
        [
            "tra_gurobi.py",
            "--cases",
            "GUROBI-M1",
            "--gurobi-structure-guidance",
            "--gurobi-structure-required",
            "--speed-budget-factor",
            "0.8",
        ],
    )
    args = tra_gurobi.parse_args()
    assert args.gurobi_structure_accept_epsilon >= 1e-5

    def fake_structure_probe(*args, **kwargs):
        del args, kwargs
        return {
            "enabled": True,
            "accepted": True,
            "reason": "structure_replay_matches_gurobi",
            "cmax": 483.0,
            "runtime_sec": 124.0,
            "route_edge_audit_missing_count": 0,
            "route_edge_audit_timed_out": False,
        }

    monkeypatch.setattr(tra_gurobi, "_gurobi_structure_guided_probe", fake_structure_probe)
    row = tra_gurobi.run_case(
        args,
        "GUROBI-M1",
        str(tmp_path),
        {
            "GUROBI-M1": {
                "model_cmax": 483.0000029903181,
                "runtime_sec": 300.0,
                "model_gap": 0.009,
            }
        },
        {},
    )

    assert row["tra_gurobi_cmax"] == 483.0
    assert row["optimal_pass"] is True
    assert row["best_audit_pass"] is True
    assert row["acceptance_pass"] is True


def test_tra_gurobi_verified_export_cmax_requires_audit_and_verification(tmp_path: Path, monkeypatch) -> None:
    if "gurobipy" not in sys.modules:
        fake_gurobipy = types.ModuleType("gurobipy")
        fake_gurobipy.GRB = types.SimpleNamespace()
        fake_gurobipy.LinExpr = object
        fake_gurobipy.Model = object
        fake_gurobipy.Var = object
        fake_gurobipy.quicksum = sum
        monkeypatch.setitem(sys.modules, "gurobipy", fake_gurobipy)
    fake_tra = types.ModuleType("Gurobi.tra")
    fake_tra.TRAOptimizer = object
    fake_tra.TRARunConfig = object
    monkeypatch.setitem(sys.modules, "Gurobi.tra", fake_tra)

    from Gurobi import tra_gurobi

    export_dir = tmp_path / "gurobi_solution_export"
    _write_json(export_dir / "best_solution_objectives.json", {"model_cmax": 804.999999, "global_makespan": 805.0})
    _write_json(export_dir / "best_solution_audit.json", {"has_unreasonable_solution": False, "verification_failures": []})
    (export_dir / "tra_makespan_verification.txt").write_text("status=PASS\ncoverage_ok=True\n", encoding="utf-8")

    cmax, reason = tra_gurobi._verified_structure_export_cmax(str(export_dir))

    assert reason == ""
    assert cmax == 805.0


def test_global_target_probe_honors_warm_start_and_symmetry_flags(monkeypatch) -> None:
    if "gurobipy" not in sys.modules:
        fake_gurobipy = types.ModuleType("gurobipy")
        fake_gurobipy.GRB = types.SimpleNamespace()
        fake_gurobipy.LinExpr = object
        fake_gurobipy.Model = object
        fake_gurobipy.Var = object
        fake_gurobipy.quicksum = sum
        monkeypatch.setitem(sys.modules, "gurobipy", fake_gurobipy)
    fake_tra = types.ModuleType("Gurobi.tra")
    fake_tra.TRAOptimizer = object
    fake_tra.TRARunConfig = object
    monkeypatch.setitem(sys.modules, "Gurobi.tra", fake_tra)

    from Gurobi import tra_gurobi

    captured = {}

    class FakeSolver:
        def solve(self, problem, cfg):
            del problem
            captured["cfg"] = cfg
            return types.SimpleNamespace(status="TIME_LIMIT", objective=float("nan"), gap=float("nan"), diagnostics={})

    monkeypatch.setattr(tra_gurobi.CreateOFSProblem, "generate_problem_by_scale", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(tra_gurobi, "GlobalXYZUSolver", FakeSolver)
    args = Namespace(
        global_target_probe=True,
        known_target_guidance=True,
        target_table_fastpath=False,
        target_probe_case_presets=False,
        global_target_probe_time_limit_sec=1.0,
        global_target_probe_stage_time_limit_sec=1.0,
        global_target_probe_candidate_stack_topk=999,
        global_target_probe_candidate_station_topk_per_stack=999,
        global_target_probe_max_candidate_stacks_per_order=0,
        global_target_probe_full_candidate_on_fail=False,
        global_target_probe_route_pickup_neighbor_limit=-1,
        global_target_probe_route_arc_prune=True,
        global_target_probe_route_time_window_arc_prune=False,
        global_target_probe_route_load_interval_arc_prune=True,
        global_target_probe_warm_start=True,
        global_target_probe_obj_slack=0.999,
        global_target_probe_accept_epsilon=1e-5,
        fixgurobi_mip_gap=0.01,
        fixgurobi_route_pickup_neighbor_limit=0,
        fixgurobi_sort_hit_tote_threshold=1,
        fixgurobi_output=False,
        fixgurobi_final_validation_mip_focus=-1,
        fixgurobi_final_validation_heuristics=-1.0,
        fixgurobi_enable_symmetry=False,
        seed=42,
        _current_case_extra_protected_route_edges=[],
    )

    tra_gurobi._global_target_probe(args, "GUROBI-M1", 582.0)

    cfg = captured["cfg"]
    assert cfg.enable_warm_start is True
    assert cfg.warm_start_use_sp4 is True
    assert cfg.fixgurobi_no_warm_start is False
    assert cfg.enable_resource_lex_symmetry is False
    assert cfg.enable_slot_lex_symmetry is False


def test_tra_fast_structure_fastpath_requires_verified_export(tmp_path: Path, monkeypatch) -> None:
    fake_tra = types.ModuleType("Gurobi.tra")
    fake_tra.TRAOptimizer = object
    fake_tra.TRARunConfig = object
    monkeypatch.setitem(sys.modules, "Gurobi.tra", fake_tra)

    from experiments import run_tra_fast

    export_dir = tmp_path / "gurobi_solution_export"
    _write_json(export_dir / "best_solution_objectives.json", {"model_cmax": 581.999994, "global_makespan": 582.0})
    _write_json(export_dir / "best_solution_audit.json", {"has_unreasonable_solution": False, "verification_failures": []})
    (export_dir / "tra_makespan_verification.txt").write_text("status=PASS\ncoverage_ok=True\n", encoding="utf-8")
    structure_json = tmp_path / "exports.json"
    _write_json(structure_json, {"exports": {"GUROBI-M1": str(export_dir)}})

    row = run_tra_fast._structure_fastpath_row(
        "GUROBI-M1",
        baseline={"GUROBI-M1": {"cmax": 582.0, "gap": 0.005, "runtime_sec": 360.0, "current_tra_sec": 2.0}},
        structure_export_json=str(structure_json),
        result_root=str(tmp_path),
        runtime_sec=0.05,
        acceptance_gap=0.0,
    )

    assert row is not None
    assert row["status"] == "structure_fastpath"
    assert row["tra_fast_cmax"] == 582.0
    assert row["acceptance_pass"] is True
    assert row["final_cmax_source"] == "verified_gurobi_structure_export"


def test_resource_time_syncs_stale_global_makespan_from_task_rows(monkeypatch) -> None:
    fake_gp = types.ModuleType("gurobipy")
    fake_gp.GRB = types.SimpleNamespace(
        BINARY="B",
        INTEGER="I",
        CONTINUOUS="C",
        OPTIMAL=2,
        TIME_LIMIT=9,
        INFEASIBLE=3,
        SUBOPTIMAL=13,
        INTERRUPTED=11,
        CUTOFF=6,
        USER_OBJ_LIMIT=15,
    )
    fake_gp.Model = object
    fake_gp.Var = object
    fake_gp.LinExpr = object
    fake_gp.quicksum = sum
    monkeypatch.setitem(sys.modules, "gurobipy", fake_gp)

    from Gurobi.resource_time_alns.utils import _sync_problem_global_makespan_from_tasks

    problem = types.SimpleNamespace(
        global_makespan=640.0,
        subtask_list=[
            types.SimpleNamespace(
                execution_tasks=[
                    types.SimpleNamespace(end_process_time=610.0),
                    types.SimpleNamespace(end_process_time=637.0),
                ]
            ),
            types.SimpleNamespace(execution_tasks=[types.SimpleNamespace(end_process_time=600.0)]),
        ],
    )

    synced = _sync_problem_global_makespan_from_tasks(problem)

    assert synced == 637.0
    assert problem.global_makespan == 637.0


def test_tra_fast_full_candidate_calibration_profile_uses_baseline_policy(monkeypatch) -> None:
    fake_tra = types.ModuleType("Gurobi.tra")
    fake_tra.TRAOptimizer = object
    fake_tra.TRARunConfig = object
    monkeypatch.setitem(sys.modules, "Gurobi.tra", fake_tra)

    from experiments import run_tra_fast

    profile = run_tra_fast._calibration_profile(
        "GUROBI-M1",
        Namespace(
            calibration_time_sec=120.0,
            calibration_mip_gap=0.002,
            calibration_full_candidates=True,
            calibration_disable_warm_start=False,
        ),
        remaining_sec=100.0,
    )

    assert profile["candidate_stack_topk"] == 999
    assert profile["max_candidate_stacks_per_order"] == 0
    assert profile["candidate_station_topk_per_stack"] == 999
    assert profile["route_pickup_neighbor_limit"] == 0


def test_tra_fast_external_baseline_preserves_sort_threshold(tmp_path: Path, monkeypatch) -> None:
    fake_tra = types.ModuleType("Gurobi.tra")
    fake_tra.TRAOptimizer = object
    fake_tra.TRARunConfig = object
    monkeypatch.setitem(sys.modules, "Gurobi.tra", fake_tra)

    from experiments import run_tra_fast

    baseline_csv = tmp_path / "baseline.csv"
    baseline_csv.write_text(
        "case,model_cmax,runtime_sec,model_gap,current_tra_sec,sort_hit_tote_threshold,solver_mip_gap\n"
        "GUROBI-M1,582,360,0.005,194,1,0.002\n",
        encoding="utf-8",
    )

    table = run_tra_fast._load_external_baseline(str(baseline_csv))

    assert table["GUROBI-M1"]["sort_hit_tote_threshold"] == 1
    assert table["GUROBI-M1"]["solver_mip_gap"] == 0.002
