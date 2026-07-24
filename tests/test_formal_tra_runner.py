from __future__ import annotations

import copy
from pathlib import Path

import pytest

from Gurobi.tra_events import EventLedger, FeasibleSolutionEvent, SearchAuditLedger, read_event_rows
from Gurobi.tra_templates import global_config_from_policy
from experiments.harvest_m_tra_gurobi import harvest
from experiments.m_tra_policy import (
    PolicyError,
    normalize_serialized_case_policy,
    sanitize_case_policy,
)
from experiments.run_m_tra_gurobi_formal import build_parser


def test_formal_runner_exposes_no_target_replay_export_or_cutoff_argument() -> None:
    destinations = {action.dest.lower() for action in build_parser()._actions}
    forbidden_fragments = ("target", "replay", "export", "cutoff", "best_obj")

    assert not any(fragment in destination for destination in destinations for fragment in forbidden_fragments)


def test_formal_config_uses_deterministic_canonical_warm_routing() -> None:
    cfg = global_config_from_policy({"warm_start_sp4_guided_local_search": True})

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

    assert result["first_verified_target_time_sec"] == pytest.approx(12.0)
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
