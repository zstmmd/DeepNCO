from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping


OLD_42_RUNTIME_SEC = {
    "M1": 360.18,
    "M2": 384.04,
    "M3": 663.53,
    "M4": 567.06,
    "M5": 663.67,
    "M6": 1039.62,
    "M7": 1314.21,
    "M8": 1837.86,
    "M9": 2608.73,
}

OLD_42_GAP = {
    "M1": 0.00541,
    "M2": 0.00907,
    "M3": 0.00985,
    "M4": 0.00932,
    "M5": 0.00950,
    "M6": 0.00921,
    "M7": 0.00940,
    "M8": 0.008794437981997073,
    "M9": 0.00978,
}


class PolicyError(ValueError):
    pass


# Static whitelist: values outside this set never reach a formal model config.
MODEL_POLICY_FIELDS = frozenset(
    {
        "integer_cmax",
        "candidate_stack_topk",
        "max_rank",
        "enable_warm_start",
        "slot_slack_per_order",
        "enable_tight_slot_upper_bound",
        "max_candidate_stacks_per_order",
        "enable_hard_candidate_stack_cap",
        "enable_warm_candidate_stack_prune",
        "candidate_station_topk_per_stack",
        "warm_start_sp4_time_limit_sec",
        "warm_start_sp4_guided_local_search",
        "warm_start_subtask_ordering",
        "warm_start_use_sp2_mip_initial",
        "warm_start_sp2_mip_time_limit_sec",
        "warm_start_refine_sp2_after_sp4",
        "warm_start_use_sp4",
        "u_route_use_mip",
        "big_m_time",
        "integrate_u_route",
        "route_arc_prune",
        "u_same_slot_same_robot",
        "route_big_m_time",
        "route_lazy_constraint",
        "route_lazy_level",
        "bom_arrival_window_sec",
        "enable_order_time_windows",
        "kitting_span_penalty_weight",
        "deadline_penalty_weight",
        "release_time_hard",
        "enable_uz_lb_cuts",
        "enable_sku_cover_cuts",
        "enable_slot_min_arrival_lb",
        "enable_route_incident_travel_lb",
        "enable_route_pair_service_travel_lb",
        "enable_route_slot_stack_count_lb",
        "enable_route_finish_cmax_lb",
        "enable_slot_pair_arrival_lb",
        "enable_slot_sku_arrival_lb",
        "enable_sku_release_workload_lb",
        "enable_station_rank_workload_lb",
        "enable_global_arrival_workload_lb",
        "enable_route_time_window_arc_prune",
        "enable_route_load_interval_arc_prune",
        "enable_route_directional_arc_prune",
        "enable_route_service_sec_cuts",
        "enable_resource_lex_symmetry",
        "enable_slot_lex_symmetry",
        "enable_tote_equivalence_symmetry",
        "enable_station_global_lex_symmetry",
        "enable_robot_finish_lex_symmetry",
        "enable_slot_sku_signature_lex_symmetry",
        "enable_station_load_lex_symmetry",
        "station_slot_count_cap",
        "enable_warm_route_time_upper_bound",
        "warm_route_time_upper_bound_margin_sec",
        "enable_route_time_big_m_linear_cuts",
        "enable_route_load_big_m_linear_cuts",
        "route_constraint_mode",
        "enable_hard_station_topk",
        "enable_slot_specific_warm_station_protection",
        "enable_warm_start_station_projection",
        "enable_anchor_first_order_robot",
        "enable_selected_workload_lbs",
        "enable_route_arrival_slot_linear",
        "enable_station_clock_linear",
        "enable_warm_prune_bound_repair",
        "enable_warm_start_route_repair",
        "enable_scale_adaptive_candidate_prune",
        "enable_sort_hit_tote_threshold",
        "enable_required_slot_active_lb",
        "enable_slot_min_pick_workload_lb",
        "sort_hit_tote_threshold",
        "route_pickup_neighbor_limit",
        "enable_route_delivery_pickup_neighbor_prune",
    }
)


_CASE_PRUNE_PROFILE = {
    "M1": (999, True, False, True, 0),
    "M2": (999, True, False, True, 0),
    "M3": (1, True, False, True, 0),
    "M4": (1, True, False, True, 0),
    "M5": (1, True, False, True, 0),
    "M6": (2, False, False, True, 0),
    "M7": (1, True, True, True, 5),
    "M8": (1, True, False, True, 5),
    "M9": (1, True, False, True, 5),
}


def _legacy_profile(case_id: str) -> Dict[str, Any]:
    try:
        station_topk, route_prune, time_window, load_interval, route_knn = _CASE_PRUNE_PROFILE[case_id]
    except KeyError as exc:
        raise PolicyError(f"unknown M-suite case: {case_id}") from exc
    return {
        "candidate_stack_topk": 999,
        "max_candidate_stacks_per_order": 0,
        "candidate_station_topk_per_stack": station_topk,
        "route_arc_prune": route_prune,
        "enable_route_time_window_arc_prune": time_window,
        "enable_route_load_interval_arc_prune": load_interval,
        "route_pickup_neighbor_limit": route_knn,
        "enable_tight_slot_upper_bound": True,
        "enable_warm_candidate_stack_prune": False,
        "enable_sort_hit_tote_threshold": case_id != "M1",
        # M1 keeps the old report-compatible numeric value, but the constraint family is disabled.
        "sort_hit_tote_threshold": 1 if case_id == "M1" else 3,
        # All accepted M1-M9 baselines predate these unconditional model constraints.
        "enable_required_slot_active_lb": False,
        "enable_slot_min_pick_workload_lb": False,
    }


def _legacy_reason(case_id: str, field_name: str) -> str:
    if case_id == "M1" and field_name == "enable_sort_hit_tote_threshold":
        return "M1 predates the SortByHitThreshold constraint family; the archived model has no such rows."
    if field_name in {
        "enable_required_slot_active_lb",
        "enable_slot_min_pick_workload_lb",
    }:
        return (
            f"{case_id} predates the {field_name} constraint family; "
            "the archived model has no such rows."
        )
    return "Explicit M-suite legacy profile recovered from the archived command and effective diagnostics."


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    return str(value)


@dataclass(frozen=True)
class SanitizedCasePolicy:
    case_id: str
    values: Mapping[str, Any]
    provenance: Mapping[str, Mapping[str, str]]
    policy_sha256: str

    def to_global_config_kwargs(self) -> Dict[str, Any]:
        return dict(self.values)

    def as_payload(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "values": dict(self.values),
            "provenance": {key: dict(value) for key, value in self.provenance.items()},
            "policy_sha256": self.policy_sha256,
        }


def normalize_serialized_case_policy(raw: Mapping[str, Any]) -> SanitizedCasePolicy:
    fields = set(raw)
    expected = {"case_id", "values", "provenance", "policy_sha256"}
    if fields != expected:
        raise PolicyError(f"serialized policy fields differ: missing={sorted(expected - fields)}, unknown={sorted(fields - expected)}")
    case_id = str(raw["case_id"]).upper().replace("GUROBI-", "")
    values = {str(key): _json_value(value) for key, value in dict(raw["values"] or {}).items()}
    unknown = set(values) - MODEL_POLICY_FIELDS
    if unknown:
        raise PolicyError(f"serialized policy contains non-whitelisted fields: {sorted(unknown)}")
    provenance = {
        str(key): {
            "source": str(dict(value or {}).get("source", "")),
            "reason": str(dict(value or {}).get("reason", "")),
        }
        for key, value in dict(raw["provenance"] or {}).items()
    }
    if set(provenance) != set(values):
        raise PolicyError("serialized policy provenance does not cover exactly the policy values")
    canonical = {"case_id": case_id, "values": values, "provenance": provenance}
    text = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    calculated = hashlib.sha256(text.encode("utf-8")).hexdigest()
    supplied = str(raw["policy_sha256"] or "")
    if supplied != calculated:
        raise PolicyError(f"serialized policy hash mismatch: supplied={supplied}, calculated={calculated}")
    return SanitizedCasePolicy(
        case_id=case_id,
        values=values,
        provenance=provenance,
        policy_sha256=calculated,
    )


def sanitize_case_policy(case_id: str, summary: Mapping[str, Any]) -> SanitizedCasePolicy:
    """Resolve model policy without consulting current defaults or historical solutions."""

    normalized_case = str(case_id).upper().replace("GUROBI-", "")
    profile = _legacy_profile(normalized_case)
    diagnostics = dict(summary.get("diagnostics", {}) or {})
    summary_config = dict(summary.get("config", {}) or {})
    candidate_fields = (set(diagnostics) | set(summary_config) | set(profile)) & MODEL_POLICY_FIELDS

    values: Dict[str, Any] = {}
    provenance: Dict[str, Dict[str, str]] = {}
    for field_name in sorted(candidate_fields):
        if field_name in diagnostics and diagnostics[field_name] is not None:
            values[field_name] = _json_value(diagnostics[field_name])
            provenance[field_name] = {
                "source": "summary.diagnostics",
                "reason": "Archived effective diagnostic emitted by the baseline solve.",
            }
        elif field_name in summary_config:
            values[field_name] = _json_value(summary_config[field_name])
            provenance[field_name] = {
                "source": "summary.config",
                "reason": "Archived baseline configuration value.",
            }
        else:
            values[field_name] = _json_value(profile[field_name])
            provenance[field_name] = {
                "source": "legacy_profile",
                "reason": _legacy_reason(normalized_case, field_name),
            }

    # The archived M1 summary has no effective threshold rows. This explicit profile must
    # override any numeric compatibility value that may exist in a regenerated report.
    if normalized_case == "M1":
        values["enable_sort_hit_tote_threshold"] = False
        provenance["enable_sort_hit_tote_threshold"] = {
            "source": "legacy_profile",
            "reason": _legacy_reason(normalized_case, "enable_sort_hit_tote_threshold"),
        }
    else:
        threshold_rows = diagnostics.get("sort_hit_tote_threshold_count")
        values["enable_sort_hit_tote_threshold"] = bool(int(threshold_rows or 0) > 0) if threshold_rows is not None else True
        provenance["enable_sort_hit_tote_threshold"] = {
            "source": "summary.diagnostics" if threshold_rows is not None else "legacy_profile",
            "reason": (
                "Archived effective constraint count for SortByHitThreshold."
                if threshold_rows is not None
                else _legacy_reason(normalized_case, "enable_sort_hit_tote_threshold")
            ),
        }
    for field_name in (
        "enable_required_slot_active_lb",
        "enable_slot_min_pick_workload_lb",
    ):
        count_field = {
            "enable_required_slot_active_lb": "required_slot_active_lb_count",
            "enable_slot_min_pick_workload_lb": "slot_min_pick_workload_lb_count",
        }[field_name]
        archived_count = diagnostics.get(count_field)
        if archived_count is not None:
            values[field_name] = bool(int(archived_count or 0) > 0)
            provenance[field_name] = {
                "source": "summary.diagnostics",
                "reason": f"Archived effective constraint count for {count_field}.",
            }
        else:
            values[field_name] = False
            provenance[field_name] = {
                "source": "legacy_profile",
                "reason": _legacy_reason(normalized_case, field_name),
            }

    if (
        normalized_case in {"M8", "M9"}
        and "warm_start_sp4_guided_local_search" not in values
    ):
        values["warm_start_sp4_guided_local_search"] = True
        provenance["warm_start_sp4_guided_local_search"] = {
            "source": "summary.config",
            "reason": (
                f"Archived {normalized_case} baseline used the default guided "
                "local search warm-route mode."
            ),
        }

    canonical = {
        "case_id": normalized_case,
        "values": values,
        "provenance": provenance,
    }
    text = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return SanitizedCasePolicy(
        case_id=normalized_case,
        values=values,
        provenance=provenance,
        policy_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class RuntimeBudgetPolicy:
    case_id: str
    baseline_runtime_sec: float
    objective_gap_stop: float
    hard_limit_sec: float
    inner_quota_sec: float
    outer_quota_sec: float
    reserve_quota_sec: float


def runtime_budget_for_case(case_id: str) -> RuntimeBudgetPolicy:
    normalized_case = str(case_id).upper().replace("GUROBI-", "")
    try:
        baseline = float(OLD_42_RUNTIME_SEC[normalized_case])
        objective_gap_stop = float(OLD_42_GAP[normalized_case])
    except KeyError as exc:
        raise PolicyError(f"unknown M-suite case: {case_id}") from exc
    hard_limit = 0.8 * baseline
    return RuntimeBudgetPolicy(
        case_id=normalized_case,
        baseline_runtime_sec=baseline,
        objective_gap_stop=objective_gap_stop,
        hard_limit_sec=hard_limit,
        inner_quota_sec=0.30 * hard_limit,
        outer_quota_sec=0.55 * hard_limit,
        reserve_quota_sec=0.15 * hard_limit,
    )


_FORBIDDEN_KEY_FRAGMENTS = (
    "targetcmax",
    "bestobjstop",
    "knowncmax",
    "knownoptimum",
    "structureexport",
    "solutionexport",
    "historicalsolution",
    "targetprobe",
    "targetpolish",
    "replay",
    "cutoff",
)
_FORBIDDEN_EXACT_KEYS = {"cmax", "target", "optimum", "objective_target"}
_FORBIDDEN_VALUE_MARKERS = ("target-probe", "target_probe", "gurobi_solution_export", "target-polish")


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def assert_target_blind_payload(payload: Any, *, _path: str = "payload") -> None:
    """Fail closed when a formal-run input can carry an old objective or solution."""

    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            normalized = _normalized_key(key)
            if normalized in {_normalized_key(item) for item in _FORBIDDEN_EXACT_KEYS} or any(
                marker in normalized for marker in _FORBIDDEN_KEY_FRAGMENTS
            ):
                raise PolicyError(f"target-blind input rejected at {_path}.{key}")
            assert_target_blind_payload(value, _path=f"{_path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_target_blind_payload(value, _path=f"{_path}[{index}]")
        return
    if isinstance(payload, str):
        lowered = payload.lower()
        if any(marker in lowered for marker in _FORBIDDEN_VALUE_MARKERS):
            raise PolicyError(f"target-blind input rejected at {_path}")
