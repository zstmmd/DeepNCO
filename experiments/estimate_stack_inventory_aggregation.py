#!/usr/bin/env python3
"""Estimate variable-count impact of stack-level inventory aggregation.

This script is intentionally standalone. It does not import or modify the
Gurobi model. It reads existing ``gurobi_summary.json`` files and estimates how
many variables would remain if the current tote-level inventory selection
variables were replaced by stack-level SKU support variables.

The estimates are scenario based:

* conservative_stack_sku:
  remove ``sku_use + carry + hit + noise + flip_hit`` and add
  ``pick_qty[slot, stack, sku]`` for every order SKU and candidate stack.
  This is the safer formulation because it can still keep stack-SKU quantity
  capacity constraints in the master model.

* service_approx:
  same as conservative_stack_sku, but also removes tote interval ``sort``
  variables and adds one coarse service variable per ``(slot, stack)``. This is
  only safe if tote selection is handled by post-processing plus feasibility
  cuts.

* optimistic_cover_only:
  removes the same variables as service_approx and adds only one coarse service
  variable per ``(slot, stack)``. This is a lower-bound estimate for variable
  count, not a recommended production model by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REMOVED_TOTE_VARS = ("carry", "hit", "noise", "flip_hit")
REMOVED_SKU_TOTE_VARS = ("sku_use",) + REMOVED_TOTE_VARS
REMOVED_SERVICE_APPROX_VARS = ("sort",) + REMOVED_SKU_TOTE_VARS


def _as_int_dict(raw: Any) -> Dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, int] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _summary_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "gurobi_summary.json"
    return p


def _load_summary(path: str) -> Tuple[Path, Dict[str, Any]]:
    p = _summary_path(path)
    with p.open("r", encoding="utf-8") as f:
        return p, json.load(f)


def _var_counts(summary: Dict[str, Any]) -> Dict[str, int]:
    diag = summary.get("diagnostics") or {}
    raw = diag.get("model_var_count_by_type") or {}
    return {str(k): int(v) for k, v in raw.items()}


def _order_sku_counts(summary: Dict[str, Any]) -> Dict[int, int]:
    orders = summary.get("orders") or []
    out: Dict[int, int] = {}
    if isinstance(orders, list):
        for item in orders:
            if not isinstance(item, dict):
                continue
            try:
                out[int(item["order_id"])] = int(item["unique_sku_count"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _sum_vars(var_by_type: Dict[str, int], names: Iterable[str]) -> int:
    return sum(int(var_by_type.get(name, 0)) for name in names)


def _estimate_one(path: str) -> Dict[str, Any]:
    summary_path, summary = _load_summary(path)
    diag = summary.get("diagnostics") or {}
    var_by_type = _var_counts(summary)
    slot_count_by_order = _as_int_dict(diag.get("slot_count_by_order"))
    candidate_stack_count_by_order = _as_int_dict(diag.get("candidate_stack_count_by_order"))
    order_sku_count = _order_sku_counts(summary)

    current_total = int(diag.get("model_var_count_total") or sum(var_by_type.values()))
    slot_stack_pairs = 0
    stack_sku_pick_vars = 0
    missing_orders: List[int] = []

    for order_id, slot_count in sorted(slot_count_by_order.items()):
        candidate_stack_count = int(candidate_stack_count_by_order.get(order_id, 0))
        sku_count = order_sku_count.get(order_id)
        if sku_count is None:
            missing_orders.append(int(order_id))
            sku_count = 0
        slot_stack_pairs += int(slot_count) * int(candidate_stack_count)
        stack_sku_pick_vars += int(slot_count) * int(candidate_stack_count) * int(sku_count)

    removed_tote = _sum_vars(var_by_type, REMOVED_TOTE_VARS)
    removed_sku_tote = _sum_vars(var_by_type, REMOVED_SKU_TOTE_VARS)
    removed_service = _sum_vars(var_by_type, REMOVED_SERVICE_APPROX_VARS)

    conservative_total = current_total - removed_sku_tote + stack_sku_pick_vars
    service_total = current_total - removed_service + stack_sku_pick_vars + slot_stack_pairs
    optimistic_total = current_total - removed_service + slot_stack_pairs

    return {
        "path": str(summary_path),
        "case": summary_path.parent.name,
        "status": summary.get("status"),
        "objective": summary.get("objective"),
        "gap": summary.get("gap"),
        "current_total_vars": current_total,
        "current_tote_level_vars": removed_tote,
        "current_sku_plus_tote_vars": removed_sku_tote,
        "current_sku_tote_sort_vars": removed_service,
        "estimated_stack_sku_pick_vars": stack_sku_pick_vars,
        "estimated_slot_stack_service_vars": slot_stack_pairs,
        "conservative_stack_sku_total_vars": conservative_total,
        "conservative_reduction_vars": current_total - conservative_total,
        "conservative_reduction_pct": _pct(current_total - conservative_total, current_total),
        "service_approx_total_vars": service_total,
        "service_approx_reduction_vars": current_total - service_total,
        "service_approx_reduction_pct": _pct(current_total - service_total, current_total),
        "optimistic_cover_only_total_vars": optimistic_total,
        "optimistic_reduction_vars": current_total - optimistic_total,
        "optimistic_reduction_pct": _pct(current_total - optimistic_total, current_total),
        "slot_count_by_order": slot_count_by_order,
        "candidate_stack_count_by_order": candidate_stack_count_by_order,
        "order_sku_count": order_sku_count,
        "missing_order_sku_counts": missing_orders,
        "var_by_type_subset": {
            name: int(var_by_type.get(name, 0))
            for name in ("sku_use", "carry", "hit", "noise", "flip_hit", "sort", "flip", "x", "y", "route_arc")
        },
    }


def _pct(delta: int, total: int) -> float:
    if int(total) <= 0:
        return 0.0
    return 100.0 * float(delta) / float(total)


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "case",
        "current",
        "removed_sku+tote",
        "stack_sku_add",
        "conservative_delta",
        "conservative_pct",
        "service_delta",
        "service_pct",
        "optimistic_delta",
        "optimistic_pct",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["case"]),
                    _format_int(row["current_total_vars"]),
                    _format_int(row["current_sku_plus_tote_vars"]),
                    _format_int(row["estimated_stack_sku_pick_vars"]),
                    _format_int(row["conservative_reduction_vars"]),
                    _format_pct(row["conservative_reduction_pct"]),
                    _format_int(row["service_approx_reduction_vars"]),
                    _format_pct(row["service_approx_reduction_pct"]),
                    _format_int(row["optimistic_reduction_vars"]),
                    _format_pct(row["optimistic_reduction_pct"]),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate variable reduction from replacing tote-level inventory choice with stack-level aggregation."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result directories or gurobi_summary.json files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON instead of a compact TSV table.",
    )
    args = parser.parse_args()

    rows = [_estimate_one(path) for path in args.paths]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
