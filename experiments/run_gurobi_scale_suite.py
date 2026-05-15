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


SCALES = [f"GUROBI-S{i}" for i in range(1, 10)]
SUMMARY_COLUMNS = [
    "算例名称",
    "BOM数目",
    "SKU总数",
    "tote数",
    "工作台数",
    "机器人数",
    "stack数",
    "地图大小",
    "上界",
    "下界",
    "gap",
    "model_objective",
    "model_cmax",
    "total_span_overrun",
    "total_deadline_overrun",
    "bom_arrival_window_violating_order_count",
    "model_gap",
    "运行时间",
]


def _parse_forced_candidate_stacks(text: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid forced candidate stack entry: {item!r}; expected order_id:stack_id")
        order_text, stack_text = item.split(":", 1)
        order_id = int(order_text.strip())
        stack_id = int(stack_text.strip())
        out.setdefault(order_id, []).append(stack_id)
    return {int(k): list(dict.fromkeys(int(v) for v in values)) for k, values in out.items()}


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _problem_summary(problem: Any, scale: str) -> Dict[str, Any]:
    map_obj = getattr(problem, "map", None)
    map_size = ""
    if map_obj is not None:
        map_size = (
            f"{int(getattr(map_obj, 'warehouse_length_block_number', 0) or 0)}x"
            f"{int(getattr(map_obj, 'warehouse_width_block_number', 0) or 0)}"
        )
    return {
        "算例名称": str(scale).upper(),
        "BOM数目": int(getattr(problem, "order_num", len(getattr(problem, "order_list", []) or [])) or 0),
        "SKU总数": int(getattr(problem, "skus_num", len(getattr(problem, "skus_list", []) or [])) or 0),
        "tote数": int(getattr(problem, "tote_num", len(getattr(problem, "tote_list", []) or [])) or 0),
        "工作台数": int(getattr(problem, "station_num", len(getattr(problem, "station_list", []) or [])) or 0),
        "机器人数": int(getattr(problem, "robot_num", len(getattr(problem, "robot_list", []) or [])) or 0),
        "stack数": int(len(getattr(problem, "stack_list", []) or [])),
        "地图大小": map_size,
    }


def _format_number(value: Any) -> str:
    val = _finite_float(value)
    if not math.isfinite(val):
        return ""
    return f"{val:.6f}"


def _write_outputs(rows: List[Dict[str, Any]], output_dir: str, details: List[Dict[str, Any]]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GUROBI-S1..S9 Global XYZU benchmark suite.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--quiet-gurobi", action="store_true", default=True)
    parser.add_argument("--show-gurobi", action="store_true", help="Enable Gurobi solver log output.")
    parser.add_argument("--candidate-stack-topk", type=int, default=3)
    parser.add_argument("--max-rank", type=int, default=0)
    parser.add_argument("--max-candidate-stacks-per-order", type=int, default=24)
    parser.add_argument("--kitting-span-penalty-weight", type=float, default=5.0)
    parser.add_argument("--deadline-penalty-weight", type=float, default=1000.0)
    parser.add_argument("--scales", type=str, default=",".join(SCALES), help="Comma-separated scale list.")
    parser.add_argument(
        "--force-candidate-stacks",
        type=str,
        default="",
        help="Comma/semicolon-separated order_id:stack_id entries to keep in the Gurobi candidate set.",
    )
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(ROOT_DIR, "result", f"gurobi_scale_suite_{timestamp}")
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    forced_candidate_stacks = _parse_forced_candidate_stacks(str(args.force_candidate_stacks or ""))

    scales = [str(item).strip().upper() for item in str(args.scales or "").split(",") if str(item).strip()]
    for scale in scales:
        start = time.perf_counter()
        problem = None
        row: Dict[str, Any] = {"算例名称": scale}
        try:
            print(f">>> Running {scale} seed={int(args.seed)} time_limit={float(args.time_limit):.0f}s")
            problem = CreateOFSProblem.generate_problem_by_scale(scale, seed=int(args.seed))
            row.update(_problem_summary(problem, scale))
            cfg = GlobalXYZUConfig(
                time_limit_sec=float(args.time_limit),
                mip_gap=float(args.mip_gap),
                candidate_stack_topk=int(args.candidate_stack_topk),
                max_rank=int(args.max_rank),
                enable_warm_start=True,
                write_lp=False,
                gurobi_output=bool(args.show_gurobi),
                max_candidate_stacks_per_order=int(args.max_candidate_stacks_per_order),
                integrate_u_route=True,
                route_arc_prune=True,
                u_same_slot_same_robot=True,
                warm_start_use_sp4=True,
                enable_sp4_fallback=False,
                kitting_span_penalty_weight=float(args.kitting_span_penalty_weight),
                deadline_penalty_weight=float(args.deadline_penalty_weight),
                forced_candidate_stacks_by_order=forced_candidate_stacks,
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
                    "上界": _format_number(upper),
                    "下界": _format_number(lower),
                    "gap": _format_number(gap),
                    "model_objective": _format_number(diag.get("model_objective", result.objective)),
                    "model_cmax": _format_number(diag.get("model_cmax", upper)),
                    "total_span_overrun": _format_number(diag.get("total_span_overrun", 0.0)),
                    "total_deadline_overrun": _format_number(diag.get("total_deadline_overrun", 0.0)),
                    "bom_arrival_window_violating_order_count": int(bom_violating_count),
                    "model_gap": _format_number(gap),
                    "运行时间": _format_number(runtime),
                }
            )
            details.append(
                {
                    "scale": scale,
                    "status": str(result.status),
                    "error": "",
                    "model_status_code": int(diag.get("model_status_code", 0) or 0),
                    "model_sol_count": int(diag.get("model_sol_count", 0) or 0),
                    "u_arc_count": int(diag.get("u_arc_count", 0) or 0),
                    "u_time_window_pruned_arc_count": int(diag.get("u_time_window_pruned_arc_count", 0) or 0),
                    "u_time_window_latest_tightened_node_count": int(diag.get("u_time_window_latest_tightened_node_count", 0) or 0),
                    "model_objective": _finite_float(diag.get("model_objective", result.objective)),
                    "model_cmax": _finite_float(diag.get("model_cmax", upper)),
                    "total_span_overrun": _finite_float(diag.get("total_span_overrun", 0.0), 0.0),
                    "total_deadline_overrun": _finite_float(diag.get("total_deadline_overrun", 0.0), 0.0),
                    "bom_arrival_window_violating_order_count": int(bom_violating_count),
                    "solution_export_dir": solution_export_dir,
                }
            )
            print(f"<<< {scale} status={result.status} ub={row['上界']} lb={row['下界']} gap={row['gap']}")
        except Exception as exc:
            if problem is not None:
                row.update(_problem_summary(problem, scale))
            row.setdefault("BOM数目", "")
            row.setdefault("SKU总数", "")
            row.setdefault("tote数", "")
            row.setdefault("工作台数", "")
            row.setdefault("机器人数", "")
            row.setdefault("stack数", "")
            row.setdefault("地图大小", "")
            row.update(
                {
                    "上界": "",
                    "下界": "",
                    "gap": "",
                    "运行时间": _format_number(time.perf_counter() - start),
                }
            )
            details.append({"scale": scale, "status": "ERROR", "error": str(exc)})
            print(f"<<< {scale} failed: {exc}")
        rows.append({key: row.get(key, "") for key in SUMMARY_COLUMNS})
        _write_outputs(rows, output_dir, details)

    outputs = _write_outputs(rows, output_dir, details)
    print("\n=== GUROBI Scale Suite Summary ===")
    print(outputs["table"])
    print(f"summary_csv={outputs['csv']}")
    print(f"summary_md={outputs['markdown']}")
    print(f"run_details_json={outputs['details']}")


if __name__ == "__main__":
    main()
