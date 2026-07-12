#!/usr/bin/env python3
"""Summarize M4 time-window simplification experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_REPORT = Path(__file__).resolve().parent / "time_window_simplification_report.md"
DEFAULT_CSV = Path(__file__).resolve().parent / "summary.csv"


BASELINE_800 = {
    "case_name": "baseline_800_no_tw_external",
    "description": "Known 800s baseline supplied by user / existing result.",
    "status": "TIME_LIMIT",
    "objective": 884.7993662540592,
    "global_makespan": 884.0,
    "true_global_makespan": 884.0,
    "model_best_bound": 874.2267777762222,
    "gap": 0.011949136585164824,
    "gurobi_solve_time_sec": 800.3872454999946,
    "runtime_sec": 816.7919399579987,
    "model_var_count_total": 22993,
    "model_constr_count_total": 63846,
    "u_arc_count": 17531,
    "total_qty": 765,
    "total_span_overrun": 0.0,
    "total_deadline_overrun": 0.0,
}


def _load_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/*.metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["_metrics_path"] = str(path)
        rows.append(row)
    return rows


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _num(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    number = _num(value)
    if number is None:
        return ""
    return f"{100.0 * number:.3f}%"


def _delta(row: Dict[str, Any], key: str, baseline: Dict[str, Any]) -> Optional[float]:
    lhs = _num(row.get(key))
    rhs = _num(baseline.get(key))
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _stack_inventory_estimate(row: Dict[str, Any]) -> Dict[str, Any]:
    var_by_type = row.get("model_var_count_by_type") or {}
    current = int(row.get("model_var_count_total") or 0)
    removed_sku_tote = sum(int(var_by_type.get(name, 0) or 0) for name in ("sku_use", "carry", "hit", "noise", "flip_hit"))
    removed_service = removed_sku_tote + int(var_by_type.get("sort", 0) or 0)
    # Use the current M4 shape directly: 19 slots, 3 stacks/order, SKU counts [22,16x5].
    stack_sku_add = 4 * 3 * 22 + 5 * 3 * 3 * 16
    slot_stack_service = 19 * 3
    conservative_delta = removed_sku_tote - stack_sku_add
    service_delta = removed_service - stack_sku_add - slot_stack_service
    optimistic_delta = removed_service - slot_stack_service
    return {
        "current_total_vars": current,
        "removed_sku_tote_vars": removed_sku_tote,
        "removed_sku_tote_sort_vars": removed_service,
        "stack_sku_add_vars": stack_sku_add,
        "slot_stack_service_add_vars": slot_stack_service,
        "conservative_reduction_vars": conservative_delta,
        "conservative_reduction_pct": conservative_delta / current if current else 0.0,
        "service_approx_reduction_vars": service_delta,
        "service_approx_reduction_pct": service_delta / current if current else 0.0,
        "optimistic_reduction_vars": optimistic_delta,
        "optimistic_reduction_pct": optimistic_delta / current if current else 0.0,
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "case_name",
        "status",
        "global_makespan",
        "objective",
        "model_best_bound",
        "gap",
        "gurobi_solve_time_sec",
        "runtime_sec",
        "model_var_count_total",
        "model_constr_count_total",
        "u_arc_count",
        "total_qty",
        "total_span_overrun",
        "total_deadline_overrun",
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _markdown_table(rows: Iterable[Dict[str, Any]], baseline: Dict[str, Any]) -> str:
    lines = [
        "| Case | cmax | bound | gap | solve(s) | vars | constr | u_arc | span/deadline | Δsolve vs short baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_solve = _delta(row, "gurobi_solve_time_sec", baseline)
        lines.append(
            "| {case} | {cmax} | {bound} | {gap} | {solve} | {vars} | {constr} | {arc} | {overrun} | {delta} |".format(
                case=str(row.get("case_name", "")),
                cmax=_fmt(row.get("global_makespan"), 3),
                bound=_fmt(row.get("model_best_bound"), 3),
                gap=_fmt_pct(row.get("gap")),
                solve=_fmt(row.get("gurobi_solve_time_sec"), 2),
                vars=_fmt(row.get("model_var_count_total"), 0),
                constr=_fmt(row.get("model_constr_count_total"), 0),
                arc=_fmt(row.get("u_arc_count"), 0),
                overrun=f"{_fmt(row.get('total_span_overrun'), 2)}/{_fmt(row.get('total_deadline_overrun'), 2)}",
                delta=_fmt(delta_solve, 2),
            )
        )
    return "\n".join(lines)


def _write_report(rows: List[Dict[str, Any]], report_path: Path) -> None:
    short_baseline = next((row for row in rows if row.get("case_name") == "baseline_no_tw"), rows[0] if rows else BASELINE_800)
    tw_on = next((row for row in rows if row.get("case_name") == "tw_on_no_cut"), None)
    estimate_source = tw_on or short_baseline
    stack_est = _stack_inventory_estimate(estimate_source)

    content = [
        "# M4 Order Time Window Simplification Experiment",
        "",
        "## Fixed 800s Baseline",
        "",
        (
            f"- cmax={BASELINE_800['global_makespan']}, bound={BASELINE_800['model_best_bound']:.3f}, "
            f"gap={100 * BASELINE_800['gap']:.3f}%, solve={BASELINE_800['gurobi_solve_time_sec']:.2f}s, "
            f"vars={BASELINE_800['model_var_count_total']}, constr={BASELINE_800['model_constr_count_total']}, "
            f"u_arc={BASELINE_800['u_arc_count']}, total_qty={BASELINE_800['total_qty']}, "
            "span/deadline overrun=0/0."
        ),
        "",
        "## Short-Horizon Experiment Results",
        "",
        _markdown_table(rows, baseline=short_baseline),
        "",
        "## Strategy 5 Static Estimate",
        "",
        (
            "Stack-level SKU/tote aggregation was not solved because it requires a new model formulation. "
            "The estimate below uses the current variable mix and replaces tote-level SKU selection with "
            "stack-SKU variables."
        ),
        "",
        f"- Conservative stack-SKU formulation: reduces {stack_est['conservative_reduction_vars']} vars ({100 * stack_est['conservative_reduction_pct']:.2f}%).",
        f"- If tote interval `sort` is also approximated at stack service level: reduces {stack_est['service_approx_reduction_vars']} vars ({100 * stack_est['service_approx_reduction_pct']:.2f}%).",
        f"- Optimistic cover-only lower bound: reduces {stack_est['optimistic_reduction_vars']} vars ({100 * stack_est['optimistic_reduction_pct']:.2f}%).",
        "",
        "## Interpretation Template",
        "",
        "- Compare `baseline_no_tw` with `tw_on_no_cut` to isolate order time-window impact.",
        "- Compare `tw_on_no_cut` with `s1_station_relaxed_top2_tw`; if top2 is larger/slower, current top1 is already the useful station simplification.",
        "- Compare `tw_on_no_cut` with stack top2/top4 variants to test candidate stack restriction.",
        "- Treat `s3_route_relaxed_proxy_tw` as an upper-bound proxy for route-pattern ideas, not a valid apples-to-apples full model.",
        "- Treat `s4_fixed_slot_order_tw` as a solve-speed test; it fixes choices but does not remove variables from the production model.",
    ]
    report_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = _load_metrics(args.results_dir)
    rows = sorted(rows, key=lambda row: str(row.get("case_name", "")))
    _write_csv(rows, args.csv)
    _write_report(rows, args.report)
    print(f"wrote {args.csv}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
