import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from experiments.run_gurobi_benchmark18_suite import _install_runtime_configs
from problemDto.createInstance import CreateOFSProblem


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_cases(text: str) -> List[str]:
    return [item.strip().upper() for item in str(text or "").replace(";", ",").split(",") if item.strip()]


def _sum_dict_values(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    total = 0
    for item in value.values():
        try:
            total += int(item)
        except Exception:
            pass
    return int(total)


def _row_from_diag(case: str, cfg_name: str, diag: Dict[str, Any], cfg: GlobalXYZUConfig) -> Dict[str, Any]:
    var_by_type = dict(diag.get("model_var_count_by_type", {}) or {})
    return {
        "case": case,
        "config_name": cfg_name,
        "compile_time_sec": float(diag.get("compile_time_sec", 0.0) or 0.0),
        "slot_count": int(diag.get("slot_count", 0) or 0),
        "work_unit_count": int(diag.get("work_unit_count", 0) or 0),
        "candidate_stack_total": _sum_dict_values(diag.get("candidate_stack_count_by_order")),
        "support_tote_total": _sum_dict_values(diag.get("support_tote_count_by_order")),
        "demand_hit_tote_total": _sum_dict_values(diag.get("demand_hit_tote_count_by_order")),
        "model_var_count_total": int(diag.get("model_var_count_total", 0) or 0),
        "route_arc": int(var_by_type.get("route_arc", 0) or 0),
        "passX": int(var_by_type.get("passX", 0) or 0),
        "x": int(var_by_type.get("x", 0) or 0),
        "y": int(var_by_type.get("y", 0) or 0),
        "sort": int(var_by_type.get("sort", 0) or 0),
        "hit": int(var_by_type.get("hit", 0) or 0),
        "noise": int(var_by_type.get("noise", 0) or 0),
        "slot_robot": int(var_by_type.get("slot_robot", 0) or 0),
        "route_time": int(var_by_type.get("route_time", 0) or 0),
        "route_owner": int(var_by_type.get("route_owner", 0) or 0),
        "route_load": int(var_by_type.get("route_load", 0) or 0),
        "u_candidate_task_count": int(diag.get("u_candidate_task_count", 0) or 0),
        "u_node_count": int(diag.get("u_node_count", 0) or 0),
        "u_arc_count": int(diag.get("u_arc_count", 0) or 0),
        "u_legal_arc_count_before_knn": int(diag.get("u_legal_arc_count_before_knn", 0) or 0),
        "u_knn_pruned_arc_count": int(diag.get("u_knn_pruned_arc_count", 0) or 0),
        "u_time_window_pruned_arc_count": int(diag.get("u_time_window_pruned_arc_count", 0) or 0),
        "u_load_interval_pruned_arc_count": int(diag.get("u_load_interval_pruned_arc_count", 0) or 0),
        "route_pickup_neighbor_limit": int(getattr(cfg, "route_pickup_neighbor_limit", 0) or 0),
        "candidate_stack_topk": int(getattr(cfg, "candidate_stack_topk", 0) or 0),
        "enable_warm_start": bool(getattr(cfg, "enable_warm_start", False)),
        "warm_start_use_sp4": bool(getattr(cfg, "warm_start_use_sp4", False)),
        "enable_order_time_windows": bool(getattr(cfg, "enable_order_time_windows", False)),
        "route_arc_prune": bool(getattr(cfg, "route_arc_prune", False)),
        "enable_route_time_window_arc_prune": bool(getattr(cfg, "enable_route_time_window_arc_prune", False)),
        "enable_route_load_interval_arc_prune": bool(getattr(cfg, "enable_route_load_interval_arc_prune", False)),
        "model_var_count_by_type_json": json.dumps(var_by_type, ensure_ascii=False, sort_keys=True),
    }


def _build_cfg(args, config_name: str) -> GlobalXYZUConfig:
    if config_name == "no_warm_no_tw_wide":
        return GlobalXYZUConfig(
            time_limit_sec=float(args.time_limit),
            mip_gap=float(args.mip_gap),
            candidate_stack_topk=99,
            max_rank=0,
            max_candidate_stacks_per_order=0,
            candidate_station_topk_per_stack=0,
            route_pickup_neighbor_limit=20,
            enable_warm_start=False,
            warm_start_use_sp4=False,
            enable_sp4_fallback=False,
            integrate_u_route=True,
            u_same_slot_same_robot=True,
            route_arc_prune=True,
            enable_route_time_window_arc_prune=True,
            enable_route_load_interval_arc_prune=True,
            enable_order_time_windows=False,
            write_lp=False,
            gurobi_output=False,
        )
    if config_name == "no_warm_no_tw_no_knn":
        cfg = _build_cfg(args, "no_warm_no_tw_wide")
        cfg.route_pickup_neighbor_limit = 0
        return cfg
    if config_name == "no_warm_no_tw_route10":
        cfg = _build_cfg(args, "no_warm_no_tw_wide")
        cfg.route_pickup_neighbor_limit = 10
        return cfg
    if config_name == "current_pruned_no_warm_no_tw":
        cfg = _build_cfg(args, "no_warm_no_tw_wide")
        cfg.candidate_stack_topk = 3
        cfg.route_pickup_neighbor_limit = 5
        return cfg
    raise ValueError(f"unknown config_name={config_name}")


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile GlobalXYZU models and report model size without optimizing.")
    parser.add_argument("--cases", type=str, default="GUROBI-M1")
    parser.add_argument("--configs", type=str, default="no_warm_no_tw_wide,no_warm_no_tw_no_knn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--runtime-config-json", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    _install_runtime_configs(str(args.runtime_config_json or ""))
    out_dir = _ensure_dir(args.output_dir or os.path.join(ROOT_DIR, "result", f"gurobi_model_size_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    rows: List[Dict[str, Any]] = []
    for case in _parse_cases(args.cases):
        problem = CreateOFSProblem.generate_problem_by_scale(case, seed=int(args.seed))
        for config_name in _parse_cases(args.configs):
            cfg = _build_cfg(args, config_name.lower())
            compiled = GlobalXYZUSolver().compile_model(problem, cfg=cfg)
            row = _row_from_diag(case, config_name.lower(), dict(compiled.diagnostics or {}), cfg)
            rows.append(row)
            print(
                f"{case} {config_name}: vars={row['model_var_count_total']} route_arc={row['route_arc']} "
                f"passX={row['passX']} compile={row['compile_time_sec']:.2f}s"
            )
            try:
                compiled.model.dispose()
            except Exception:
                pass
    _write_csv(os.path.join(out_dir, "model_size_probe.csv"), rows)
    with open(os.path.join(out_dir, "model_size_probe.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "output_dir": out_dir}, f, ensure_ascii=False, indent=2)
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
