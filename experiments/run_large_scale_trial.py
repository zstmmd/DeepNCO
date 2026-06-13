import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.run_benchmark import _make_tra_config
from Gurobi.tra import TRAOptimizer
from problemDto.createInstance import CreateOFSProblem


def _stack_counts(order_count: int, high_every: int = 3) -> tuple:
    return tuple(3 if (idx + 1) % max(1, int(high_every)) == 0 else 2 for idx in range(int(order_count)))


def _order_sku_counts(order_count: int, base: int, bump_every: int = 3) -> tuple:
    return tuple(int(base) + (1 if (idx + 1) % max(1, int(bump_every)) == 0 else 0) for idx in range(int(order_count)))


def large_scale_configs() -> Dict[str, Dict[str, Any]]:
    rows = {
        "L1": {"map_size": (5, 8), "resources": (6, 5, 350), "data": (15, 80), "base_lines": 3, "target_stack_count": 60},
        "L2": {"map_size": (6, 8), "resources": (7, 6, 450), "data": (20, 100), "base_lines": 3, "target_stack_count": 75},
        "L3": {"map_size": (6, 9), "resources": (8, 7, 600), "data": (25, 120), "base_lines": 3, "target_stack_count": 95},
        "L4": {"map_size": (7, 9), "resources": (10, 8, 750), "data": (30, 140), "base_lines": 4, "target_stack_count": 120},
        "L5": {"map_size": (8, 10), "resources": (12, 10, 900), "data": (40, 180), "base_lines": 4, "target_stack_count": 145},
        "L6": {"map_size": (9, 10), "resources": (14, 12, 1100), "data": (50, 220), "base_lines": 4, "target_stack_count": 175},
        "L7": {"map_size": (10, 11), "resources": (16, 14, 1300), "data": (60, 330), "base_lines": 5, "target_stack_count": 205},
        "L8": {"map_size": (11, 11), "resources": (18, 14, 1500), "data": (80, 430), "base_lines": 5, "target_stack_count": 235},
        "L9": {"map_size": (12, 12), "resources": (20, 15, 1700), "data": (100, 540), "base_lines": 5, "target_stack_count": 265},
    }
    configs: Dict[str, Dict[str, Any]] = {}
    for case, row in rows.items():
        order_count = int(row["data"][0])
        base_lines = int(row.pop("base_lines"))
        configs[case] = {
            **row,
            "bom_complexity": (8, 1),
            "inventory_cold_filler_probability": 0.25,
            "inventory_initial_unassigned_skus_per_tote": 3,
            "inventory_max_sku_stack_count": 4,
            "exact_order_sku_counts": _order_sku_counts(order_count, base_lines, bump_every=3),
            "exact_order_sku_quantity_range": (1, 1),
            "bom_colocated_inventory": True,
            "bom_colocated_stack_counts": _stack_counts(order_count, high_every=3),
            "bom_colocated_disjoint_stack_groups": False,
            "bom_colocated_support_multiplier": 1.5,
            "bom_colocated_sku_copy_count": 2,
        }
    return configs


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _summarize_case(case: str, result_root: str) -> Dict[str, Any]:
    summary = _read_json(os.path.join(result_root, "tra_summary.json"))
    best = dict(summary.get("best", {}) or {})
    run_stats = dict(summary.get("run_stats", {}) or {})
    objectives = _read_json(os.path.join(result_root, "best_solution_export", "best_solution_objectives.json"))
    config = dict(summary.get("config", {}) or {})
    sp1 = dict(objectives.get("sp1", {}) or {})
    sp2 = dict(objectives.get("sp2", {}) or {})
    best_structure = dict(summary.get("best_structure", {}) or {})
    best_payload = dict(summary.get("best", {}) or {})
    verification = _read_json(os.path.join(result_root, "best_solution_audit.json"))
    if not verification:
        verification = _read_json(os.path.join(result_root, "best_solution_export", "best_solution_audit.json"))
    makespan_verification = _read_json(os.path.join(result_root, "best_solution_export", "tra_makespan_verification.json"))
    cfg_row = large_scale_configs().get(str(case).upper(), {})
    data = tuple(cfg_row.get("data", (0, 0)) or (0, 0))
    resources = tuple(cfg_row.get("resources", (0, 0, 0)) or (0, 0, 0))
    return {
        "case": str(case).upper(),
        "result_root": result_root,
        "best_z": _finite_float(best.get("z", float("nan"))),
        "best_iter": int(best.get("iter_id", best.get("iter", -1)) or -1),
        "runtime_sec": _finite_float(run_stats.get("run_total_time_sec", 0.0), 0.0),
        "global_eval_count": int(run_stats.get("global_eval_count", 0) or 0),
        "iter_count": len(list(summary.get("iters", []) or [])),
        "orders": int(data[0]) if len(data) >= 1 else 0,
        "subtasks": int(sp1.get("subtask_count", 0) or 0),
        "tasks": int(best_payload.get("task_count", best_structure.get("task_count", 0)) or 0),
        "skus": int(data[1]) if len(data) >= 2 else 0,
        "totes": int(resources[2]) if len(resources) >= 3 else 0,
        "robots": int(resources[0]) if len(resources) >= 1 else 0,
        "stations": int(resources[1]) if len(resources) >= 2 else int(sp2.get("station_count", 0) or 0),
        "avg_sku_per_subtask": _finite_float(sp1.get("avg_sku_per_subtask", float("nan"))),
        "audit_status": str(verification.get("status", "")),
        "makespan_verify_status": str(makespan_verification.get("status", "")),
        "stop_reason": str(run_stats.get("stop_reason", "")),
        "search_scheme": str(config.get("search_scheme", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trial large BOM-aware TRA instances L1-L9.")
    parser.add_argument("--cases", type=str, default="L1", help="Comma-separated cases, e.g. L1,L2,L3.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--no-improve-limit", type=int, default=2)
    parser.add_argument("--sp2-time-limit-sec", type=float, default=10.0)
    parser.add_argument("--sp4-lkh-time-limit-seconds", type=int, default=5)
    parser.add_argument("--xz-evaluator-mode", type=str, default="neural")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(large_scale_configs())
    cases = [item.strip().upper() for item in str(args.cases).split(",") if item.strip()]
    result_root = os.path.join(ROOT_DIR, "result", f"large_scale_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(result_root, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def _large_trial_hook(cfg: Any) -> None:
        cfg.fixgurobi_final_validation = False
        cfg.resource_target_cmax = float("nan")
        cfg.target_runtime_sec = 180.0
        cfg.runtime_guard_mode = "soft"
        cfg.layer_operator_budget_x = min(int(getattr(cfg, "layer_operator_budget_x", 4)), 3)
        cfg.layer_operator_budget_y = min(int(getattr(cfg, "layer_operator_budget_y", 6)), 4)
        cfg.layer_operator_budget_z = min(int(getattr(cfg, "layer_operator_budget_z", 3)), 2)
        cfg.layer_operator_budget_u = min(int(getattr(cfg, "layer_operator_budget_u", 1)), 1)
        cfg.x_global_eval_topk = min(int(getattr(cfg, "x_global_eval_topk", 2)), 1)
        cfg.y_global_eval_topk = min(int(getattr(cfg, "y_global_eval_topk", 2)), 1)
        cfg.z_global_eval_topk = min(int(getattr(cfg, "z_global_eval_topk", 1)), 1)

    for case in cases:
        case_root = os.path.join(
            result_root,
            f"{case}_{int(args.seed)}_resource_time_alns_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(case_root, exist_ok=True)
        cfg = _make_tra_config(
            scale=case,
            seed=int(args.seed),
            max_iters=int(args.max_iters),
            no_improve_limit=int(args.no_improve_limit),
            epsilon=0.05,
            sp2_time_limit_sec=float(args.sp2_time_limit_sec),
            sp4_lkh_time_limit_seconds=int(args.sp4_lkh_time_limit_seconds),
            enable_role_vns=False,
            enable_shadow_chain=True,
            shadow_chain_max_depth=3,
        )
        cfg.search_scheme = "resource_time_alns"
        cfg.log_dir = case_root
        cfg.write_iteration_logs = True
        cfg.export_best_solution = True
        cfg.enable_sp1_feedback_analysis = False
        cfg.xz_evaluator_mode = str(args.xz_evaluator_mode).strip().lower() or "neural"
        _large_trial_hook(cfg)

        opt = TRAOptimizer(cfg)
        if bool(args.silent):
            import contextlib

            old_flag = os.environ.get("OFS_BATCH_SILENT")
            os.environ["OFS_BATCH_SILENT"] = "1"
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    opt.run()
            if old_flag is None:
                os.environ.pop("OFS_BATCH_SILENT", None)
            else:
                os.environ["OFS_BATCH_SILENT"] = old_flag
        else:
            opt.run()
        row = _summarize_case(case, case_root)
        rows.append(row)
        print(
            f"{case}: best_z={row['best_z']:.3f}, runtime={row['runtime_sec']:.2f}s, "
            f"orders={row['orders']}, subtasks={row['subtasks']}, tasks={row['tasks']}, "
            f"audit={row['audit_status']}, verify={row['makespan_verify_status']}"
        )

    summary_path = os.path.join(result_root, "large_scale_trial_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
