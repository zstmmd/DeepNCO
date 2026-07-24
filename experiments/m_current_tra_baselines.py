from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from experiments.m_tra_policy import sanitize_case_policy


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CurrentMCase:
    case_id: str
    algorithm_case: str
    config_path: str
    gurobi_summary_path: str

    @property
    def config_abs_path(self) -> Path:
        return ROOT_DIR / self.config_path

    @property
    def gurobi_summary_abs_path(self) -> Path:
        return ROOT_DIR / self.gurobi_summary_path

    @property
    def gurobi_export_path(self) -> str:
        return (Path(self.gurobi_summary_path).parent / "gurobi_solution_export").as_posix()


CURRENT_M_CASES: List[CurrentMCase] = [
    CurrentMCase(
        "M1",
        "GUROBI-M1",
        "experiments/configs/milddle_m1-m3_last.json",
        "result/middle_bomseq_m1_seed42_t360_g002_nocand_routeprune_r0_20260624/gurobi_summary.json",
    ),
    CurrentMCase(
        "M2",
        "GUROBI-M2",
        "experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json",
        "result/middle_bomseq_m2_seed42_t400_g01_authoritative_noslotlex_focus1_h095_r0_probe_20260709/gurobi_summary.json",
    ),
    CurrentMCase(
        "M3",
        "GUROBI-M3",
        "experiments/configs/milddle_m1-m3_last.json",
        "result/middle_bomseq_m3_seed42_t700_g01_4x5_r4s3_t115_sku270_bq33_chunked4_stationtop1_routearrlinear_noslotlex_focus1_h095_r0_probe_20260709/gurobi_summary.json",
    ),
    CurrentMCase(
        "M4",
        "GUROBI-M4",
        "experiments/configs/milddle_m1-m3_last.json",
        "result/middle_bomseq_m4_seed42_t900_g01_hist_sku22_16x5_bq33_qty34_stack3_copy1_support20_stationtop1_noslotlex_focus1_h005_r0_probe_20260710/gurobi_summary.json",
    ),
    CurrentMCase(
        "M5",
        "GUROBI-M5",
        "experiments/configs/milddle_m1-m3_last.json",
        "result/middle_bomseq_m5_seed42_t900_g01_sku10_qty34_bq33_stationtop1_noslotlex_focus1_h005_r0_run_20260710/gurobi_summary.json",
    ),
    CurrentMCase(
        "M6",
        "GUROBI-M6",
        "experiments/configs/middle_stack_bomseq_runtime_configs_20260624.json",
        "result/middle_bomseq_m6_seed42_t1300_g01_stationtop2_noarcprune_lbcuts_noslotlex_focus1_h005_r0_20260714/gurobi_summary.json",
    ),
    CurrentMCase(
        "M7",
        "GUROBI-M7",
        "experiments/configs/milddle_m1-m3_last.json",
        "result/middle_bomseq_m7_seed42_t1500_g01_sku18_qty34_bq33_r5s3_stationtop1_twprune_route5_noslotlex_focus1_h005_r0_run_20260710/gurobi_summary.json",
    ),
    CurrentMCase(
        "M8",
        "GUROBI-M8",
        "experiments/configs/m8_map5x10_tote300_sku340_coloc6_chunktrue_20260712.json",
        "result/m8_map5x10_tote300_sku340_coloc6_chunktrue_route5_slotwarmprotect_integercmax_t3600_timeout_20260714/gurobi_summary.json",
    ),
    CurrentMCase(
        "M9",
        "GUROBI-M9",
        "experiments/configs/m9_map5x10_tote400_sku430_coloc8_chunktrue_20260714.json",
        "result/m9_map5x10_tote400_sku430_coloc8_chunktrue_route5_slotwarmprotect_integercmax_t3600_20260714/gurobi_summary.json",
    ),
]


def load_current_m_cases() -> List[CurrentMCase]:
    return list(CURRENT_M_CASES)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _case_config(case: CurrentMCase) -> Dict[str, Any]:
    payload = _read_json(case.config_abs_path)
    configs = payload.get("configs", payload) if isinstance(payload, dict) else {}
    config = dict(configs.get(case.case_id, {}) or {})
    if not config:
        raise KeyError(f"missing config for {case.case_id} in {case.config_path}")
    return config


def _solver_policy(case: CurrentMCase) -> Dict[str, Any]:
    payload = _read_json(case.config_abs_path)
    common = dict(payload.get("common_solver", {}) or {}) if isinstance(payload, dict) else {}
    case_runs = dict(payload.get("case_runs", {}) or {}) if isinstance(payload, dict) else {}
    case_run = dict(case_runs.get(case.case_id, {}) or {})
    command = str(case_run.get("command", "") or "")

    def _bool(name: str, default: bool) -> bool:
        return bool(common.get(name, default))

    route_arc_prune = _bool("route_arc_prune", True)
    route_load_interval = _bool("route_load_interval_arc_prune", True)
    route_time_window = _bool("route_time_window_arc_prune", False)
    if "--disable-route-arc-prune" in command:
        route_arc_prune = False
    if "--disable-route-load-interval-arc-prune" in command:
        route_load_interval = False
    if "--enable-route-time-window-arc-prune" in command:
        route_time_window = True
    if "--disable-route-time-window-arc-prune" in command:
        route_time_window = False
    return {
        "candidate_stack_topk": int(case_run.get("candidate_stack_topk", 999) or 999),
        "max_candidate_stacks_per_order": int(case_run.get("max_candidate_stacks_per_order", 0) or 0),
        "candidate_station_topk_per_stack": int(case_run.get("candidate_station_topk_per_stack", 999) or 999),
        "route_pickup_neighbor_limit": int(case_run.get("route_pickup_neighbor_limit", common.get("route_pickup_neighbor_limit", 0)) or 0),
        "solver_mip_gap": _safe_float(case_run.get("mip_gap", common.get("mip_gap", 0.01)), 0.01),
        "route_arc_prune": bool(route_arc_prune),
        "route_load_interval_arc_prune": bool(route_load_interval),
        "route_time_window_arc_prune": bool(route_time_window),
    }


def build_runtime_alias_config(cases: Iterable[CurrentMCase] | None = None) -> Dict[str, Any]:
    configs: Dict[str, Any] = {}
    for case in cases or CURRENT_M_CASES:
        config = _case_config(case)
        configs[case.case_id] = dict(config)
        configs[case.algorithm_case] = dict(config)
    return {
        "description": "Current M1-M9 runtime configs with GUROBI-M aliases for TRA-Gurobi/TRA-Fast acceptance.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "configs": configs,
    }


def build_structure_export_map(cases: Iterable[CurrentMCase] | None = None) -> Dict[str, Any]:
    exports: Dict[str, str] = {}
    for case in cases or CURRENT_M_CASES:
        exports[case.algorithm_case] = case.gurobi_export_path
    return {
        "description": "Current M1-M9 Gurobi solution exports for TRA-Gurobi structure seed search.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exports": exports,
    }


def build_gurobi_baseline_rows(cases: Iterable[CurrentMCase] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases or CURRENT_M_CASES:
        summary = _read_json(case.gurobi_summary_abs_path)
        sanitized_policy = sanitize_case_policy(case.case_id, summary)
        diagnostics = dict(summary.get("diagnostics", {}) or {})
        cmax = _safe_float(summary.get("global_makespan", diagnostics.get("model_cmax")))
        true_cmax = _safe_float(summary.get("true_global_makespan", cmax))
        runtime = _safe_float(summary.get("gurobi_runtime_sec", summary.get("runtime_sec")))
        gap = _safe_float(summary.get("gap", diagnostics.get("model_gap")))
        row = {
            "case": case.algorithm_case,
            "scale": case.algorithm_case,
            "source_case": case.case_id,
            "model_cmax": cmax,
            "gurobi_cmax": cmax,
            "true_global_makespan": true_cmax,
            "runtime_sec": runtime,
            "gurobi_runtime_sec": runtime,
            "model_gap": gap,
            "gurobi_gap": gap,
            "model_best_bound": _safe_float(diagnostics.get("model_best_bound")),
            "status": str(summary.get("status", "")),
            "model_var_count_total": diagnostics.get("model_var_count_total"),
            "model_constr_count_total": diagnostics.get("model_constr_count_total"),
            "candidate_stack_count_max": max(
                [int(v) for v in dict(diagnostics.get("candidate_stack_count_by_order", {}) or {}).values()] + [0]
            ),
            "enable_slot_lex_symmetry": bool(diagnostics.get("enable_slot_lex_symmetry", True)),
            "enable_resource_lex_symmetry": bool(diagnostics.get("enable_resource_lex_symmetry", True)),
            "gurobi_mip_focus": diagnostics.get("gurobi_mip_focus", dict(summary.get("config", {}) or {}).get("gurobi_mip_focus")),
            "gurobi_heuristics": diagnostics.get("gurobi_heuristics", dict(summary.get("config", {}) or {}).get("gurobi_heuristics")),
            "time_verify_mismatch": bool(diagnostics.get("time_verify_mismatch", False)),
            "warm_start_mip_start_ready": bool(diagnostics.get("warm_start_mip_start_ready", False)),
            "warm_start_missing_arc_count": int(diagnostics.get("warm_start_missing_arc_count", 0) or 0),
            "gurobi_summary_path": case.gurobi_summary_path,
            "gurobi_export_dir": case.gurobi_export_path,
            "config_path": case.config_path,
            "enable_sort_hit_tote_threshold": bool(
                sanitized_policy.values["enable_sort_hit_tote_threshold"]
            ),
            "sort_hit_tote_threshold": int(sanitized_policy.values["sort_hit_tote_threshold"]),
            "domain_policy_sha256": sanitized_policy.policy_sha256,
            "domain_policy_provenance": dict(sanitized_policy.provenance),
        }
        row.update(
            {
                key: value
                for key, value in sanitized_policy.values.items()
                if key in {
                    "candidate_stack_topk",
                    "max_candidate_stacks_per_order",
                    "candidate_station_topk_per_stack",
                    "route_pickup_neighbor_limit",
                    "route_arc_prune",
                    "enable_route_load_interval_arc_prune",
                    "enable_route_time_window_arc_prune",
                    "enable_tight_slot_upper_bound",
                    "enable_warm_candidate_stack_prune",
                }
            }
        )
        row["solver_mip_gap"] = _safe_float(
            diagnostics.get("model_gap", summary.get("gap", 0.01)),
            0.01,
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    row_list = list(rows)
    fields: List[str] = []
    seen = set()
    for row in row_list:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_list)


def write_baseline_artifacts(output_root: str) -> Dict[str, str]:
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT_DIR / root
    root.mkdir(parents=True, exist_ok=True)
    rows = build_gurobi_baseline_rows()
    alias_payload = build_runtime_alias_config()
    baseline_json = root / "current_m_gurobi_baseline.json"
    baseline_csv = root / "current_m_gurobi_baseline.csv"
    runtime_alias_json = root / "current_m_runtime_aliases.json"
    structure_exports_json = root / "current_m_structure_exports.json"
    with baseline_json.open("w", encoding="utf-8") as f:
        json.dump({"details": rows, "rows": rows}, f, ensure_ascii=False, indent=2)
    _write_csv(baseline_csv, rows)
    with runtime_alias_json.open("w", encoding="utf-8") as f:
        json.dump(alias_payload, f, ensure_ascii=False, indent=2)
    with structure_exports_json.open("w", encoding="utf-8") as f:
        json.dump(build_structure_export_map(), f, ensure_ascii=False, indent=2)
    return {
        "baseline_json": os.path.relpath(baseline_json, ROOT_DIR),
        "baseline_csv": os.path.relpath(baseline_csv, ROOT_DIR),
        "runtime_alias_json": os.path.relpath(runtime_alias_json, ROOT_DIR),
        "structure_exports_json": os.path.relpath(structure_exports_json, ROOT_DIR),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate current M1-M9 TRA baseline artifacts.")
    parser.add_argument("--output-root", default="result/m_current_tra_acceptance")
    args = parser.parse_args()
    print(json.dumps(write_baseline_artifacts(args.output_root), ensure_ascii=False, indent=2))
