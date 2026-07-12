#!/usr/bin/env python3
"""Independent M4 time-window and simplification experiments.

All outputs stay under ``test_gurobi_cut/results`` by default. The script does
not modify the production solver; it imports the existing solver and varies
configuration fields for controlled experiments.

Notes on strategy coverage:
* station and stack candidate strategies are native model configurations.
* route-pattern is represented by a route-relaxed proxy
  (``integrate_u_route=False``), because the current production model does not
  expose path/pattern variables.
* fixed-slot uses existing non-CLI ``GlobalXYZUConfig`` fixed decision fields.
* stack inventory aggregation is reported by ``summarize_results.py`` as a
  variable-count estimate, not solved here, because it requires a new
  formulation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from experiments.run_global_xyzu import (  # noqa: E402
    _install_runtime_configs,
    _write_result_files,
    _write_warm_start_export,
)
from problemDto.createInstance import CreateOFSProblem  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "base_config_gap50.json"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


CASE_DESCRIPTIONS: Dict[str, str] = {
    "baseline_no_tw": "Reference: gap50, order time windows disabled.",
    "tw_on_no_cut": "Isolates the cost of enabling order time windows.",
    "s1_station_fixed_top1_tw": "Strategy 1: station choice fixed to top1. This is also the current baseline station setting.",
    "s1_station_relaxed_top2_tw": "Negative control for strategy 1: allow station top2 to quantify the cost avoided by top1.",
    "s2_stack_top2_tw": "Strategy 2: restrict candidate stacks to top2 per order.",
    "s2_stack_top4_tw": "Strategy 2: allow up to top4 stacks; expected to match baseline if effective candidates are already 3.",
    "s3_route_relaxed_proxy_tw": "Strategy 3 proxy: disable integrated route arc decisions. Not equivalent to path-pattern routing.",
    "s4_fixed_slot_order_tw": "Strategy 4: fix work-unit to slot assignment by BOM/SKU order chunks.",
}


DEFAULT_CASES = [
    "baseline_no_tw",
    "tw_on_no_cut",
    "s1_station_relaxed_top2_tw",
    "s2_stack_top2_tw",
    "s3_route_relaxed_proxy_tw",
    "s4_fixed_slot_order_tw",
]


def _base_cfg(time_limit: float, enable_order_time_windows: bool) -> GlobalXYZUConfig:
    return GlobalXYZUConfig(
        time_limit_sec=float(time_limit),
        mip_gap=0.01,
        candidate_stack_topk=999,
        max_rank=0,
        enable_warm_start=True,
        write_lp=False,
        gurobi_output=False,
        integer_cmax=False,
        slot_slack_per_order=1,
        enable_tight_slot_upper_bound=True,
        max_candidate_stacks_per_order=0,
        enable_warm_candidate_stack_prune=False,
        candidate_station_topk_per_stack=1,
        warm_start_sp4_time_limit_sec=15,
        warm_start_subtask_ordering="default",
        warm_start_use_sp2_mip_initial=False,
        warm_start_sp2_mip_time_limit_sec=30.0,
        warm_start_refine_sp2_after_sp4=False,
        u_route_use_mip=True,
        big_m_time=2000.0,
        integrate_u_route=True,
        route_arc_prune=True,
        u_same_slot_same_robot=True,
        route_lazy_constraint=True,
        route_lazy_level=1,
        bom_arrival_window_sec=60.0,
        warm_start_use_sp4=True,
        enable_sp4_fallback=False,
        enable_order_time_windows=bool(enable_order_time_windows),
        kitting_span_penalty_weight=5.0,
        deadline_penalty_weight=1000.0,
        release_time_hard=True,
        gurobi_method=1,
        gurobi_node_method=1,
        gurobi_mip_focus=1,
        gurobi_cuts=1,
        gurobi_cut_passes=1,
        gurobi_presolve=1,
        gurobi_heuristics=0.3,
        enable_uz_lb_cuts=True,
        enable_sku_cover_cuts=True,
        enable_slot_min_arrival_lb=True,
        enable_route_incident_travel_lb=True,
        enable_route_pair_service_travel_lb=True,
        enable_route_slot_stack_count_lb=True,
        enable_route_finish_cmax_lb=True,
        enable_global_arrival_workload_lb=True,
        enable_route_time_window_arc_prune=True,
        enable_route_load_interval_arc_prune=True,
        enable_route_directional_arc_prune=False,
        enable_route_service_sec_cuts=False,
        enable_resource_lex_symmetry=False,
        enable_slot_lex_symmetry=True,
        enable_tote_equivalence_symmetry=True,
        enable_station_global_lex_symmetry=True,
        enable_robot_finish_lex_symmetry=True,
        enable_anchor_first_order_robot=False,
        enable_selected_workload_lbs=True,
        enable_route_arrival_slot_linear=True,
        enable_station_clock_linear=False,
        enable_warm_prune_bound_repair=False,
        enable_warm_start_route_repair=True,
        enable_scale_adaptive_candidate_prune=False,
        sort_hit_tote_threshold=3,
        route_pickup_neighbor_limit=0,
    )


def _case_cfg(case_name: str, time_limit: float) -> GlobalXYZUConfig:
    cfg = _base_cfg(time_limit=time_limit, enable_order_time_windows=(case_name != "baseline_no_tw"))
    if case_name == "baseline_no_tw":
        return cfg
    if case_name in {"tw_on_no_cut", "s1_station_fixed_top1_tw", "s4_fixed_slot_order_tw"}:
        return cfg
    if case_name == "s1_station_relaxed_top2_tw":
        cfg.candidate_station_topk_per_stack = 2
        return cfg
    if case_name == "s2_stack_top2_tw":
        cfg.candidate_stack_topk = 2
        cfg.max_candidate_stacks_per_order = 2
        return cfg
    if case_name == "s2_stack_top4_tw":
        cfg.candidate_stack_topk = 4
        cfg.max_candidate_stacks_per_order = 4
        return cfg
    if case_name == "s3_route_relaxed_proxy_tw":
        cfg.integrate_u_route = False
        cfg.route_lazy_constraint = False
        return cfg
    raise ValueError(f"unknown case: {case_name}")


def _install_config(config_json: Path) -> None:
    _install_runtime_configs(str(config_json))


def _generate_problem(config_json: Path) -> Any:
    _install_config(config_json)
    return CreateOFSProblem.generate_problem_by_scale("M4", seed=42)


def _fixed_slot_units(problem: Any) -> Dict[int, List[List[str]]]:
    cap_limit = 6
    fixed: Dict[int, List[List[str]]] = {}
    for order in getattr(problem, "order_list", []) or []:
        order_id = int(getattr(order, "order_id", -1))
        sku_ids = sorted(int(getattr(sku, "id", sku)) for sku in getattr(order, "unique_sku_list", []) or [])
        rows: List[List[str]] = []
        for start in range(0, len(sku_ids), cap_limit):
            rows.append([f"{order_id}:{sku_id}" for sku_id in sku_ids[start:start + cap_limit]])
        fixed[order_id] = rows
    return fixed


def _copy_case_summary(out_dir: Path, result_root: str, case_name: str, cfg: GlobalXYZUConfig, description: str) -> None:
    summary_path = Path(result_root) / "gurobi_summary.json"
    payload: Dict[str, Any] = {
        "case_name": case_name,
        "description": description,
        "result_root": str(Path(result_root).resolve()),
        "config": {
            "time_limit_sec": float(cfg.time_limit_sec),
            "enable_order_time_windows": bool(cfg.enable_order_time_windows),
            "candidate_station_topk_per_stack": int(cfg.candidate_station_topk_per_stack),
            "candidate_stack_topk": int(cfg.candidate_stack_topk),
            "max_candidate_stacks_per_order": int(cfg.max_candidate_stacks_per_order),
            "integrate_u_route": bool(cfg.integrate_u_route),
            "fixed_slot": bool(cfg.fixed_work_units_by_order_slot),
        },
    }
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        diag = summary.get("diagnostics", {}) or {}
        payload.update(
            {
                "status": summary.get("status"),
                "objective": summary.get("objective"),
                "gap": summary.get("gap"),
                "global_makespan": summary.get("global_makespan"),
                "true_global_makespan": summary.get("true_global_makespan"),
                "runtime_sec": summary.get("runtime_sec"),
                "gurobi_solve_time_sec": summary.get("gurobi_solve_time_sec"),
                "model_best_bound": diag.get("model_best_bound"),
                "model_root_relax_bound": diag.get("model_root_relax_bound"),
                "model_node_count": diag.get("model_node_count"),
                "model_var_count_total": diag.get("model_var_count_total"),
                "model_constr_count_total": diag.get("model_constr_count_total"),
                "u_arc_count": diag.get("u_arc_count"),
                "u_node_count": diag.get("u_node_count"),
                "route_task_count_before_station_prune": diag.get("route_task_count_before_station_prune"),
                "route_task_count_after_station_prune": diag.get("route_task_count_after_station_prune"),
                "total_span_overrun": summary.get("total_span_overrun"),
                "total_deadline_overrun": summary.get("total_deadline_overrun"),
                "total_qty": sum(int(row.get("total_qty", 0) or 0) for row in summary.get("orders", []) or []),
                "model_var_count_by_type": diag.get("model_var_count_by_type", {}),
                "model_constr_count_by_type": diag.get("model_constr_count_by_type", {}),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{case_name}.metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def run_case(case_name: str, config_json: Path, results_dir: Path, time_limit: float, force: bool = False) -> None:
    out_dir = results_dir / case_name
    metrics_path = out_dir / f"{case_name}.metrics.json"
    if metrics_path.exists() and not force:
        print(f"[skip] {case_name}: {metrics_path}")
        return
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = _generate_problem(config_json)
    cfg = _case_cfg(case_name, time_limit=time_limit)
    if case_name == "s4_fixed_slot_order_tw":
        cfg.fixed_work_units_by_order_slot = _fixed_slot_units(problem)
        cfg.fixed_slot_count_by_order = {int(k): len(v) for k, v in cfg.fixed_work_units_by_order_slot.items()}

    solver = GlobalXYZUSolver()
    result = solver.solve(problem, cfg=cfg)
    result_root = _write_result_files(problem, result, scale="M4", seed=42, cfg=cfg, output_root=str(out_dir))
    _write_warm_start_export(result_root=result_root, solver=solver, scale="M4", seed=42)
    _copy_case_summary(out_dir=out_dir, result_root=result_root, case_name=case_name, cfg=cfg, description=CASE_DESCRIPTIONS[case_name])
    print(f"[done] {case_name}: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES, choices=sorted(CASE_DESCRIPTIONS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for case_name in args.cases:
        run_case(
            case_name=case_name,
            config_json=args.config_json,
            results_dir=args.results_dir,
            time_limit=float(args.time_limit),
            force=bool(args.force),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
