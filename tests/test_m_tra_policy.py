from __future__ import annotations

import pytest

from experiments.m_tra_policy import (
    OLD_42_RUNTIME_SEC,
    PolicyError,
    assert_target_blind_payload,
    runtime_budget_for_case,
    sanitize_case_policy,
)


def test_m1_sort_threshold_is_explicitly_disabled_without_solution_export() -> None:
    policy = sanitize_case_policy(
        "M1",
        summary={
            "config": {},
            "diagnostics": {
                "candidate_station_topk_per_stack": 999,
                "route_pickup_neighbor_limit": 0,
                "enable_route_load_interval_arc_prune": True,
                "enable_route_time_window_arc_prune": False,
            },
        },
    )

    assert policy.values["enable_sort_hit_tote_threshold"] is False
    assert policy.provenance["enable_sort_hit_tote_threshold"]["source"] == "legacy_profile"
    assert "solution" not in policy.provenance["enable_sort_hit_tote_threshold"]["reason"].lower()
    assert "export" not in policy.provenance["enable_sort_hit_tote_threshold"]["reason"].lower()


def test_m2_sort_threshold_prefers_effective_diagnostic_over_summary_config() -> None:
    policy = sanitize_case_policy(
        "M2",
        summary={
            "config": {"sort_hit_tote_threshold": 2},
            "diagnostics": {
                "sort_hit_tote_threshold": 3,
                "sort_hit_tote_threshold_count": 108,
            },
        },
    )

    assert policy.values["enable_sort_hit_tote_threshold"] is True
    assert policy.values["sort_hit_tote_threshold"] == 3
    assert policy.provenance["sort_hit_tote_threshold"]["source"] == "summary.diagnostics"


def test_runtime_budget_uses_old_42_runtime_only() -> None:
    budget = runtime_budget_for_case("M9")

    assert budget.baseline_runtime_sec == pytest.approx(OLD_42_RUNTIME_SEC["M9"])
    assert budget.hard_limit_sec == pytest.approx(0.8 * OLD_42_RUNTIME_SEC["M9"])
    assert budget.inner_quota_sec == pytest.approx(0.3 * budget.hard_limit_sec)
    assert budget.outer_quota_sec == pytest.approx(0.55 * budget.hard_limit_sec)
    assert budget.reserve_quota_sec == pytest.approx(0.15 * budget.hard_limit_sec)


@pytest.mark.parametrize(
    "payload",
    [
        {"target_cmax": 582},
        {"solver": {"BestObjStop": 582}},
        {"candidate": {"structure_export": "old/run/gurobi_solution_export"}},
        {"mode": "target-probe"},
        {"replay_path": "old_solution.json"},
    ],
)
def test_formal_payload_rejects_target_or_historical_solution_channels(payload) -> None:
    with pytest.raises(PolicyError, match="target-blind"):
        assert_target_blind_payload(payload)


def test_target_blind_guard_allows_runtime_and_integer_cmax_model_type() -> None:
    assert_target_blind_payload(
        {
            "case": "M1",
            "runtime_sec": OLD_42_RUNTIME_SEC["M1"],
            "integer_cmax": True,
            "manifest_path": "manifest.json",
        }
    )
