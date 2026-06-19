import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from experiments.run_global_xyzu import _write_gurobi_solution_export
from problemDto.createInstance import CreateOFSProblem


SMALL_CASES = [f"GUROBI-SM{i}" for i in range(1, 10)]

SMALL_TARGETS: Dict[str, Dict[str, float]] = {}

SUMMARY_COLUMNS = [
    "case",
    "case_class",
    "orders",
    "unique_sku_per_order",
    "skus",
    "totes",
    "stations",
    "robots",
    "stacks",
    "map_size",
    "demanded_sku_count",
    "total_order_qty",
    "min_order_sku_qty",
    "max_order_sku_qty",
    "min_demand_qty_per_unique_sku",
    "max_demand_qty_per_unique_sku",
    "min_distinct_stacks_per_demanded_sku",
    "avg_distinct_stacks_per_demanded_sku",
    "max_distinct_stacks_per_demanded_sku",
    "target_cmax_min",
    "target_cmax_max",
    "upper_bound",
    "lower_bound",
    "gap",
    "model_objective",
    "model_cmax",
    "total_span_overrun",
    "total_deadline_overrun",
    "bom_arrival_window_violating_order_count",
    "model_gap",
    "runtime_sec",
    "status",
    "target_cmax_ok",
    "gap_ok",
    "monotonic_ok",
    "calibration_note",
]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _format_number(value: Any) -> str:
    val = _finite_float(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.6f}"


def _problem_summary(problem: Any, scale: str) -> Dict[str, Any]:
    map_obj = getattr(problem, "map", None)
    map_size = ""
    if map_obj is not None:
        map_size = (
            f"{int(getattr(map_obj, 'warehouse_length_block_number', 0) or 0)}x"
            f"{int(getattr(map_obj, 'warehouse_width_block_number', 0) or 0)}"
        )
    orders = list(getattr(problem, "order_list", []) or [])
    unique_counts = [
        int(len(set(int(sku_id) for sku_id in (getattr(order, "order_product_id_list", []) or []))))
        for order in orders
    ]
    total_quantities = [int(getattr(order, "order_skus_number", 0) or 0) for order in orders]
    demand_quantities_by_sku: List[int] = []
    for order in orders:
        counts_by_sku: Dict[int, int] = {
            int(k): int(v)
            for k, v in dict(getattr(order, "bom_total_quantity_by_sku", {}) or {}).items()
        }
        if not counts_by_sku:
            for sku_id in getattr(order, "order_product_id_list", []) or []:
                counts_by_sku[int(sku_id)] = int(counts_by_sku.get(int(sku_id), 0)) + 1
        demand_quantities_by_sku.extend(int(qty) for qty in counts_by_sku.values())
    unique_sku_per_order = ""
    if unique_counts:
        unique_sku_per_order = str(unique_counts[0]) if len(set(unique_counts)) == 1 else ",".join(str(v) for v in unique_counts)
    return {
        "case": str(scale).upper(),
        "case_class": "small" if str(scale).upper() in SMALL_CASES else "custom",
        "orders": int(getattr(problem, "order_num", len(orders)) or 0),
        "unique_sku_per_order": unique_sku_per_order,
        "skus": int(getattr(problem, "skus_num", len(getattr(problem, "skus_list", []) or [])) or 0),
        "totes": int(getattr(problem, "tote_num", len(getattr(problem, "tote_list", []) or [])) or 0),
        "stations": int(getattr(problem, "station_num", len(getattr(problem, "station_list", []) or [])) or 0),
        "robots": int(getattr(problem, "robot_num", len(getattr(problem, "robot_list", []) or [])) or 0),
        "stacks": int(len(getattr(problem, "stack_list", []) or [])),
        "map_size": map_size,
        "demanded_sku_count": int((getattr(problem, "generator_summary", {}) or {}).get("demanded_sku_count", 0) or 0),
        "total_order_qty": int(sum(total_quantities)),
        "min_order_sku_qty": int(min(total_quantities)) if total_quantities else 0,
        "max_order_sku_qty": int(max(total_quantities)) if total_quantities else 0,
        "min_demand_qty_per_unique_sku": int(min(demand_quantities_by_sku)) if demand_quantities_by_sku else 0,
        "max_demand_qty_per_unique_sku": int(max(demand_quantities_by_sku)) if demand_quantities_by_sku else 0,
        "min_distinct_stacks_per_demanded_sku": int((getattr(problem, "redundancy_summary", {}) or {}).get("min_distinct_stacks_per_demanded_sku", 0) or 0),
        "avg_distinct_stacks_per_demanded_sku": _format_number((getattr(problem, "redundancy_summary", {}) or {}).get("avg_distinct_stacks_per_demanded_sku", float("nan"))),
        "max_distinct_stacks_per_demanded_sku": int((getattr(problem, "redundancy_summary", {}) or {}).get("max_distinct_stacks_per_demanded_sku", 0) or 0),
    }


def _target_row(scale: str) -> Dict[str, Any]:
    target = SMALL_TARGETS.get(str(scale).upper(), {})
    return {
        "target_cmax_min": _format_number(target.get("target_cmax_min", float("nan"))),
        "target_cmax_max": _format_number(target.get("target_cmax_max", float("nan"))),
    }


def _annotate_rows(rows: List[Dict[str, Any]], mip_gap: float) -> None:
    previous_cmax = float("-inf")
    for row in rows:
        if str(row.get("status", "")).upper() == "DRY_RUN":
            row["target_cmax_ok"] = "SKIP"
            row["gap_ok"] = "SKIP"
            row["monotonic_ok"] = "SKIP"
            row["calibration_note"] = "dry_run"
            continue
        cmax = _finite_float(row.get("model_cmax", float("nan")))
        gap = _finite_float(row.get("model_gap", row.get("gap", float("nan"))))
        target_min = _finite_float(row.get("target_cmax_min", float("nan")))
        target_max = _finite_float(row.get("target_cmax_max", float("nan")))
        has_target_band = math.isfinite(target_min) or math.isfinite(target_max)
        cmax_ok = True
        if math.isfinite(target_max):
            cmax_ok = math.isfinite(cmax) and cmax <= target_max + 1e-9
        if math.isfinite(target_min):
            cmax_ok = cmax_ok and cmax >= target_min - 1e-9
        gap_ok = math.isfinite(gap) and gap <= float(mip_gap) + 1e-9
        monotonic_ok = math.isfinite(cmax) and cmax > previous_cmax + 1e-9
        notes: List[str] = []
        if math.isfinite(cmax):
            if math.isfinite(target_min) and cmax < target_min - 1e-9:
                notes.append("below_target_band")
            if math.isfinite(target_max) and cmax > target_max + 1e-9:
                notes.append("above_target_band")
            if not monotonic_ok:
                notes.append("not_strictly_increasing")
            previous_cmax = cmax
        else:
            notes.append("no_cmax")
        if not gap_ok:
            notes.append("gap_above_target")
        row["target_cmax_ok"] = ("PASS" if cmax_ok else "FAIL") if has_target_band else "SKIP"
        row["gap_ok"] = "PASS" if gap_ok else "FAIL"
        row["monotonic_ok"] = "PASS" if monotonic_ok else "FAIL"
        row["calibration_note"] = ",".join(notes)


def _write_outputs(rows: List[Dict[str, Any]], output_dir: str, details: List[Dict[str, Any]], mip_gap: float) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    _annotate_rows(rows, mip_gap=float(mip_gap))
    csv_path = os.path.join(output_dir, "summary.csv")
    md_path = os.path.join(output_dir, "summary.md")
    details_path = os.path.join(output_dir, "run_details.json")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})

    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in SUMMARY_COLUMNS) + " |")
    markdown = "\n".join(lines) + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    return {"csv": csv_path, "markdown": md_path, "details": details_path, "table": markdown}


def _load_runtime_configs(path: str) -> Dict[str, Dict[str, Any]]:
    if not str(path or "").strip():
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"runtime config json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs", payload) if isinstance(payload, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in dict(configs or {}).items():
        if isinstance(cfg, dict):
            out[str(name).upper()] = dict(cfg)
    return out


def _install_runtime_configs(path: str) -> None:
    configs = _load_runtime_configs(path)
    if configs:
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(configs)


def _resolve_cases(case_set: str, scales: str) -> List[str]:
    if scales:
        return [str(item).strip().upper() for item in str(scales).split(",") if str(item).strip()]
    normalized = str(case_set or "small").strip().lower()
    if normalized == "small":
        return list(SMALL_CASES)
    raise ValueError(f"Unsupported case set: {case_set}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated Gurobi benchmark cases.")
    parser.add_argument("--case-set", type=str, default="small")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--quiet-gurobi", action="store_true", default=True)
    parser.add_argument("--show-gurobi", action="store_true", help="Enable Gurobi solver log output.")
    parser.add_argument("--candidate-stack-topk", type=int, default=3)
    parser.add_argument("--max-rank", type=int, default=0)
    parser.add_argument("--max-candidate-stacks-per-order", type=int, default=0)
    parser.add_argument("--candidate-station-topk-per-stack", type=int, default=0)
    parser.add_argument("--route-pickup-neighbor-limit", type=int, default=5)
    parser.add_argument("--big-m-time", type=float, default=2000.0)
    parser.add_argument("--route-big-m-time", type=float, default=0.0)
    parser.add_argument("--gurobi-mip-focus", type=int, default=None)
    parser.add_argument("--gurobi-heuristics", type=float, default=None)
    parser.add_argument("--disable-warm-start", action="store_true", default=False)
    parser.add_argument("--disable-warm-start-sp4", action="store_true", default=False)
    parser.add_argument("--kitting-span-penalty-weight", type=float, default=5.0)
    parser.add_argument("--deadline-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--disable-order-time-windows", action="store_true", default=False)
    parser.add_argument("--disable-all-prune", action="store_true", help="Disable all arc and candidate pruning.")
    parser.add_argument("--dry-run", action="store_true", help="Generate cases and summaries without solving Gurobi.")
    parser.add_argument("--runtime-config-json", type=str, default="", help="Optional runtime scale config JSON.")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()
    _install_runtime_configs(str(args.runtime_config_json or ""))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(ROOT_DIR, "result", f"gurobi_small_calibration_{timestamp}")
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    scales = _resolve_cases(args.case_set, args.scales)

    for scale in scales:
        start = time.perf_counter()
        problem = None
        row: Dict[str, Any] = {"case": scale}
        try:
            print(f">>> Running {scale} seed={int(args.seed)} time_limit={float(args.time_limit):.0f}s dry_run={bool(args.dry_run)}")
            problem = CreateOFSProblem.generate_problem_by_scale(scale, seed=int(args.seed))
            row.update(_problem_summary(problem, scale))
            row.update(_target_row(scale))
            if bool(args.dry_run):
                row.update(
                    {
                        "runtime_sec": _format_number(time.perf_counter() - start),
                        "status": "DRY_RUN",
                    }
                )
                details.append({"scale": scale, "status": "DRY_RUN", "error": ""})
                print(f"<<< {scale} dry-run generated")
            else:
                cfg = GlobalXYZUConfig(
                    time_limit_sec=float(args.time_limit),
                    mip_gap=float(args.mip_gap),
                    candidate_stack_topk=int(args.candidate_stack_topk),
                    max_rank=int(args.max_rank),
                    enable_warm_start=not bool(args.disable_warm_start),
                    write_lp=False,
                    gurobi_output=bool(args.show_gurobi),
                    max_candidate_stacks_per_order=int(args.max_candidate_stacks_per_order),
                    enable_warm_candidate_stack_prune=False,
                    candidate_station_topk_per_stack=int(args.candidate_station_topk_per_stack),
                    route_pickup_neighbor_limit=int(args.route_pickup_neighbor_limit),
                    big_m_time=float(args.big_m_time),
                    route_big_m_time=(float(args.route_big_m_time) if float(args.route_big_m_time or 0.0) > 0.0 else None),
                    gurobi_mip_focus=args.gurobi_mip_focus,
                    gurobi_heuristics=args.gurobi_heuristics,
                    integrate_u_route=True,
                    route_arc_prune=not bool(args.disable_all_prune),
                    enable_route_time_window_arc_prune=not bool(args.disable_all_prune),
                    enable_route_load_interval_arc_prune=not bool(args.disable_all_prune),
                    u_same_slot_same_robot=True,
                    warm_start_use_sp4=not bool(args.disable_warm_start_sp4),
                    enable_sp4_fallback=False,
                    enable_order_time_windows=not bool(args.disable_order_time_windows),
                    kitting_span_penalty_weight=float(args.kitting_span_penalty_weight),
                    deadline_penalty_weight=float(args.deadline_penalty_weight),
                )
                result = GlobalXYZUSolver().solve(problem, cfg=cfg)
                diag = dict(result.diagnostics or {})
                upper = _finite_float(diag.get("model_cmax", result.objective), _finite_float(result.objective))
                lower = _finite_float(diag.get("model_best_bound", float("nan")))
                gap = _finite_float(diag.get("model_gap", result.gap), _finite_float(result.gap))
                runtime = _finite_float(result.runtime_sec, time.perf_counter() - start)
                scale_output_dir = os.path.join(output_dir, scale)
                os.makedirs(scale_output_dir, exist_ok=True)
                solution_export_dir = _write_gurobi_solution_export(
                    result_root=scale_output_dir,
                    problem=problem,
                    result=result,
                    scale=scale,
                    seed=int(args.seed),
                )
                bom_violating_count = int(
                    sum(
                        1
                        for row_item in list(diag.get("order_time_windows", []) or [])
                        if float(row_item.get("span_overrun", 0.0) or 0.0) > 1e-9
                    )
                )
                row.update(
                    {
                        "upper_bound": _format_number(upper),
                        "lower_bound": _format_number(lower),
                        "gap": _format_number(gap),
                        "model_objective": _format_number(diag.get("model_objective", result.objective)),
                        "model_cmax": _format_number(diag.get("model_cmax", upper)),
                        "total_span_overrun": _format_number(diag.get("total_span_overrun", 0.0)),
                        "total_deadline_overrun": _format_number(diag.get("total_deadline_overrun", 0.0)),
                        "bom_arrival_window_violating_order_count": int(bom_violating_count),
                        "model_gap": _format_number(gap),
                        "runtime_sec": _format_number(runtime),
                        "status": str(result.status),
                    }
                )
                details.append(
                    {
                        "scale": scale,
                        "status": str(result.status),
                        "error": "",
                        "model_status_code": int(diag.get("model_status_code", 0) or 0),
                        "model_sol_count": int(diag.get("model_sol_count", 0) or 0),
                        "model_objective": _finite_float(diag.get("model_objective", result.objective)),
                        "model_cmax": _finite_float(diag.get("model_cmax", upper)),
                        "model_best_bound": lower,
                        "model_gap": gap,
                        "runtime_sec": runtime,
                        "u_arc_count": int(diag.get("u_arc_count", 0) or 0),
                        "u_time_window_pruned_arc_count": int(diag.get("u_time_window_pruned_arc_count", 0) or 0),
                        "model_var_count_total": int(diag.get("model_var_count_total", 0) or 0),
                        "model_var_count_by_type": dict(diag.get("model_var_count_by_type", {}) or {}),
                        "model_var_count_by_type_json": str(diag.get("model_var_count_by_type_json", "{}") or "{}"),
                        "solution_export_dir": solution_export_dir,
                    }
                )
                print(f"<<< {scale} status={result.status} cmax={row['model_cmax']} lb={row['lower_bound']} gap={row['model_gap']}")
        except Exception as exc:
            if problem is not None:
                row.update(_problem_summary(problem, scale))
                row.update(_target_row(scale))
            row.update(
                {
                    "runtime_sec": _format_number(time.perf_counter() - start),
                    "status": "ERROR",
                    "calibration_note": str(exc),
                }
            )
            details.append({"scale": scale, "status": "ERROR", "error": str(exc)})
            print(f"<<< {scale} failed: {exc}")
        rows.append({key: row.get(key, "") for key in SUMMARY_COLUMNS})
        _write_outputs(rows, output_dir, details, mip_gap=float(args.mip_gap))

    outputs = _write_outputs(rows, output_dir, details, mip_gap=float(args.mip_gap))
    print("\n=== Gurobi Benchmark Calibration Summary ===")
    print(outputs["table"])
    print(f"summary_csv={outputs['csv']}")
    print(f"summary_md={outputs['markdown']}")
    print(f"run_details_json={outputs['details']}")


if __name__ == "__main__":
    main()
