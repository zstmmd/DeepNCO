from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from Gurobi.tra_audit import SearchAuditTrail
from Gurobi.tra_events import EventLedger, FeasibleSolutionEvent, SearchAuditLedger, read_event_rows
from Gurobi.tra_neighborhood import NeighborhoodLevel, Procedure
from Gurobi.tra_outer import OuterDisposition
from Gurobi.tra_scheduler import ProcedureStep
from Gurobi.tra_templates import global_config_from_policy
from experiments.harvest_m_tra_gurobi import harvest
from experiments.m_tra_policy import (
    PolicyError,
    normalize_serialized_case_policy,
    sanitize_case_policy,
)
from experiments.run_m_tra_gurobi_formal import (
    build_formal_engine,
    build_parser as build_formal_parser,
)
from experiments.run_m_tra_gurobi_candidate_census import (
    build_parser as build_candidate_census_parser,
    validate_diagnostic_flags,
)


def test_formal_runner_exposes_no_target_replay_export_or_cutoff_argument() -> None:
    destinations = {action.dest.lower() for action in build_formal_parser()._actions}
    forbidden_fragments = ("target", "replay", "export", "cutoff", "best_obj")

    assert not any(fragment in destination for destination in destinations for fragment in forbidden_fragments)


def test_candidate_census_runner_has_only_target_blind_arguments() -> None:
    destinations = {
        action.dest.lower()
        for action in build_candidate_census_parser()._actions
    }
    forbidden_fragments = ("target", "replay", "export", "cutoff", "best_obj")

    assert not any(
        fragment in destination
        for destination in destinations
        for fragment in forbidden_fragments
    )


def test_candidate_census_runner_supports_opt_in_f1_live_seeds_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--enable-f1-live-seed-starts",
            "--outer-census-slice-sec",
            "0",
        ]
    )

    assert args.enable_f1_live_seed_starts is True
    assert args.outer_census_slice_sec == 0.0
    formal_destinations = {
        action.dest
        for action in build_formal_parser()._actions
    }
    assert "enable_f1_live_seed_starts" not in formal_destinations


def test_outer_census_procedure_filter_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--outer-census-procedure",
            "F1",
        ]
    )

    assert args.outer_census_procedure == "F1"
    assert "outer_census_procedure" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_outer_start_mode_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--outer-start-mode",
            "none",
        ]
    )

    assert args.outer_start_mode == "none"
    assert "outer_start_mode" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_incumbent_values_dir_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M5",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--incumbent-values-dir",
            "formal/M5/paper_tra_solution_export",
        ]
    )

    assert args.incumbent_values_dir == "formal/M5/paper_tra_solution_export"
    assert "incumbent_values_dir" not in {
        action.dest
        for action in build_formal_parser()._actions
    }
    with pytest.raises(SystemExit):
        build_formal_parser().parse_args(
            [
                "--case",
                "M5",
                "--runtime-config",
                "runtime.json",
                "--output-dir",
                "formal",
                "--incumbent-values-dir",
                "formal/M5/paper_tra_solution_export",
            ]
        )


def test_dual_block_census_flag_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--dual-block-census",
            "F2_F3",
        ]
    )

    assert args.dual_block_census == "F2_F3"
    assert "dual_block_census" not in {
        action.dest
        for action in build_formal_parser()._actions
    }
    with pytest.raises(SystemExit):
        build_formal_parser().parse_args(
            [
                "--case",
                "M1",
                "--runtime-config",
                "runtime.json",
                "--output-dir",
                "formal",
                "--dual-block-census",
                "F2_F3",
            ]
        )


def test_dual_block_hamming_limit_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M5",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--dual-block-census",
            "F1_F3",
            "--dual-block-hamming-limit",
            "16",
        ]
    )

    assert args.dual_block_hamming_limit == 16
    assert "dual_block_hamming_limit" not in {
        action.dest
        for action in build_formal_parser()._actions
    }
    with pytest.raises(SystemExit):
        build_formal_parser().parse_args(
            [
                "--case",
                "M5",
                "--runtime-config",
                "runtime.json",
                "--output-dir",
                "formal",
                "--dual-block-hamming-limit",
                "16",
            ]
        )


def test_dual_block_census_flag_accepts_all_diagnostic_pairs() -> None:
    for label in ("F1_F2", "F1_F3", "F2_F3"):
        args = build_candidate_census_parser().parse_args(
            [
                "--case",
                "M1",
                "--runtime-config",
                "runtime.json",
                "--output-dir",
                "diagnostics",
                "--dual-block-census",
                label,
            ]
        )

        assert args.dual_block_census == label


def test_dual_block_census_flag_accepts_all3_diagnostic() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--dual-block-census",
            "ALL3",
        ]
    )

    assert args.dual_block_census == "ALL3"
    assert "dual_block_census" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_dual_block_census_flag_accepts_hybrid_diagnostic() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--dual-block-census",
            "F2_N3_F3_N1",
        ]
    )

    assert args.dual_block_census == "F2_N3_F3_N1"
    assert "dual_block_census" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_f1_plateau_escalation_is_diagnostic_and_requires_live_starts() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--enable-f1-live-seed-starts",
            "--enable-f1-plateau-escalation",
        ]
    )

    assert args.enable_f1_plateau_escalation is True
    assert "enable_f1_plateau_escalation" not in {
        action.dest
        for action in build_formal_parser()._actions
    }
    validate_diagnostic_flags(
        enable_f1_live_seed_starts=True,
        enable_f1_plateau_escalation=True,
    )
    with pytest.raises(ValueError, match="requires --enable-f1-live-seed-starts"):
        validate_diagnostic_flags(
            enable_f1_live_seed_starts=False,
            enable_f1_plateau_escalation=True,
        )


def test_f2_plateau_escalation_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--enable-f2-plateau-escalation",
        ]
    )

    assert args.enable_f2_plateau_escalation is True
    assert "enable_f2_plateau_escalation" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_direct_inner_census_flag_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--direct-inner-census",
            "F2_N3",
        ]
    )

    assert args.direct_inner_census == "F2_N3"
    assert "direct_inner_census" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_station_balance_census_flag_is_diagnostic_only() -> None:
    args = build_candidate_census_parser().parse_args(
        [
            "--case",
            "M1",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "diagnostics",
            "--station-balance-census",
        ]
    )

    assert args.station_balance_census is True
    assert "station_balance_census" not in {
        action.dest
        for action in build_formal_parser()._actions
    }


def test_formal_engine_enables_station_balance_repair_without_cli_flag() -> None:
    destinations = {
        action.dest
        for action in build_formal_parser()._actions
    }

    engine = build_formal_engine(
        SimpleNamespace(),
        SimpleNamespace(),
        max_procedures=7,
    )

    assert "station_balance_repair" not in destinations
    assert engine.enable_station_balance_repair is True
    assert engine.scheduler.max_procedures == 7


def test_formal_runner_accepts_target_blind_master_domain_manifest_argument() -> None:
    args = build_formal_parser().parse_args(
        [
            "--case",
            "M8",
            "--runtime-config",
            "runtime.json",
            "--output-dir",
            "formal/M8",
            "--master-domain-manifest",
            "formal/M8/master_domain_v3.json",
        ]
    )

    assert args.master_domain_manifest == "formal/M8/master_domain_v3.json"


def test_formal_config_defaults_to_deterministic_canonical_warm_routing() -> None:
    cfg = global_config_from_policy({})

    assert cfg.warm_start_sp4_guided_local_search is False


def test_serialized_domain_policy_is_hash_verified() -> None:
    policy = sanitize_case_policy("M1", {"config": {}, "diagnostics": {}})
    assert normalize_serialized_case_policy(policy.as_payload()) == policy

    mutated = copy.deepcopy(policy.as_payload())
    mutated["values"]["candidate_stack_topk"] = 1
    with pytest.raises(PolicyError, match="hash mismatch"):
        normalize_serialized_case_policy(mutated)


def test_target_is_used_only_by_postrun_harvest(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    event = FeasibleSolutionEvent(
        run_id="run-1",
        case="M1",
        wall_timestamp_sec=12.5,
        solver_incumbent_timestamp_sec=12.0,
        cycle=1,
        procedure="F1",
        neighborhood="N1",
        manifest_sha256="a" * 64,
        objective_sha256="b" * 64,
        structural_hash="c" * 64,
        solver_objective=582.1,
        solver_cmax=582.0,
        verified_cmax=582.0,
        internal_feasible=True,
        verifier_error_codes=(),
        provenance={"source": "outer_mipsol"},
        snapshot_sha256="d" * 64,
    )
    with EventLedger(event_path) as ledger:
        ledger.append(event)

    result = harvest(
        event_path=event_path,
        case_id="M1",
        target_cmax=582.0,
        run_id="run-1",
    )

    assert result["first_verified_target_time_sec"] == pytest.approx(12.5)
    assert result["cmax_equal"] is True
    assert result["runtime_ok"] is True


def test_search_audit_ledger_is_target_blind_and_append_only(tmp_path: Path) -> None:
    audit_path = tmp_path / "search.jsonl"
    with SearchAuditLedger(audit_path) as ledger:
        ledger.append(
            {
                "run_id": "run-1",
                "case": "M1",
                "stage": "inner",
                "elapsed_sec": 1.25,
                "candidate_count": 2,
                "certified_obj_bound": 580.0,
            }
        )

    rows = list(read_event_rows(audit_path))
    assert rows == [
        {
            "candidate_count": 2,
            "case": "M1",
            "certified_obj_bound": 580.0,
            "elapsed_sec": 1.25,
            "run_id": "run-1",
            "schema_version": 1,
            "stage": "inner",
        }
    ]

    with SearchAuditLedger(audit_path) as ledger:
        with pytest.raises(ValueError, match="target-blind"):
            ledger.append({"stage": "outer", "target_cmax": 582.0})


def test_outer_audit_records_bound_promoted_budget_mode(tmp_path: Path) -> None:
    audit_path = tmp_path / "search.jsonl"
    result = SimpleNamespace(
        runtime_sec=1.0,
        solver_status="TIME_LIMIT",
        solver_status_code=9,
        objective_bound=579.2,
        disposition=OuterDisposition.ACCEPTED,
        resumed_search=False,
        projected_start_cmax=589.0,
        projected_start_wait_sec=0.0,
        installed_start_count=0,
        full_start_complete=False,
        full_start_feasible=False,
        full_start_max_residual=float("nan"),
        full_start_error_codes=(),
        start_projection_error="",
        verified_snapshots=(),
        accepted=None,
        error="",
    )
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)

    with SearchAuditLedger(audit_path) as ledger:
        audit = SearchAuditTrail(
            ledger,
            run_id="run-1",
            case="M1",
            elapsed_sec=lambda: 1.0,
        )
        audit.outer(
            step,
            result,
            submitted_shell_sha256="shell-a",
            reserve_retry=True,
            budget_mode="reserve_bound_promoted",
            stage="reserve_outer",
        )

    row = list(read_event_rows(audit_path))[0]
    assert row["budget_mode"] == "reserve_bound_promoted"
    assert "target" not in row


def test_outer_audit_records_target_blind_candidate_kind(tmp_path: Path) -> None:
    audit_path = tmp_path / "search.jsonl"
    result = SimpleNamespace(
        runtime_sec=1.0,
        solver_status="TIME_LIMIT",
        solver_status_code=9,
        objective_bound=579.2,
        disposition=OuterDisposition.ACCEPTED,
        resumed_search=False,
        projected_start_cmax=589.0,
        projected_start_wait_sec=0.0,
        installed_start_count=0,
        full_start_complete=False,
        full_start_feasible=False,
        full_start_max_residual=float("nan"),
        full_start_error_codes=(),
        start_projection_error="",
        verified_snapshots=(),
        accepted=None,
        error="",
    )
    step = ProcedureStep(9, 3, Procedure.F3, NeighborhoodLevel.N3)

    with SearchAuditLedger(audit_path) as ledger:
        audit = SearchAuditTrail(
            ledger,
            run_id="run-1",
            case="M1",
            elapsed_sec=lambda: 1.0,
        )
        audit.outer(
            step,
            result,
            submitted_shell_sha256="shell-a",
            reserve_retry=False,
            budget_mode="pair_station_workload_repair",
            stage="pair_station_workload_outer",
            candidate_kind="station_swap",
        )

    row = list(read_event_rows(audit_path))[0]
    assert row["candidate_kind"] == "station_swap"
    assert "target" not in row
