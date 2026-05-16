from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List


SUMMARY_COLUMNS = [
    "scale",
    "bpc_status",
    "bpc_cmax",
    "bpc_lb",
    "bpc_gap",
    "bpc_exact",
    "gurobi_status",
    "gurobi_cmax",
    "gurobi_lb",
    "gurobi_gap",
    "bpc_minus_gurobi_cmax",
    "same_objective",
    "exact_conclusion",
]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_gurobi_baseline(baseline_dir: str) -> Dict[str, Dict[str, Any]]:
    details_path = os.path.join(baseline_dir, "run_details.json")
    if not os.path.exists(details_path):
        return {}
    with open(details_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {str(row.get("scale", "")).upper(): dict(row) for row in rows if str(row.get("scale", "")).strip()}


def exact_conclusion(bpc_exact: bool, bpc_cmax: float, gurobi_cmax: float) -> str:
    if bool(bpc_exact):
        if math.isfinite(gurobi_cmax) and bpc_cmax > gurobi_cmax + 1e-9:
            return "inconsistent"
        return "bpc_proved_full_space_optimal"
    if math.isfinite(bpc_cmax) and math.isfinite(gurobi_cmax) and gurobi_cmax + 1e-9 < bpc_cmax:
        return "gurobi_better_incumbent"
    return "not_proven"


def build_comparison_row(scale: str, bpc_result: Any, gurobi_row: Dict[str, Any]) -> Dict[str, Any]:
    bpc_cmax = finite_float(getattr(bpc_result, "objective", float("nan")))
    bpc_lb = finite_float(getattr(bpc_result, "lower_bound", float("nan")))
    bpc_gap = finite_float(getattr(bpc_result, "gap", float("nan")))
    gurobi_cmax = finite_float(gurobi_row.get("model_cmax", float("nan")))
    gurobi_lb = finite_float(gurobi_row.get("model_best_bound", gurobi_row.get("lower_bound", float("nan"))))
    gurobi_gap = finite_float(gurobi_row.get("model_gap", gurobi_row.get("gap", float("nan"))))
    return {
        "scale": str(scale).upper(),
        "bpc_status": str(getattr(bpc_result, "status", "")),
        "bpc_cmax": bpc_cmax,
        "bpc_lb": bpc_lb,
        "bpc_gap": bpc_gap,
        "bpc_exact": bool(getattr(bpc_result, "exact", False)),
        "gurobi_status": str(gurobi_row.get("status", "")),
        "gurobi_cmax": gurobi_cmax,
        "gurobi_lb": gurobi_lb,
        "gurobi_gap": gurobi_gap,
        "bpc_minus_gurobi_cmax": bpc_cmax - gurobi_cmax if math.isfinite(bpc_cmax) and math.isfinite(gurobi_cmax) else float("nan"),
        "same_objective": bool(math.isfinite(bpc_cmax) and math.isfinite(gurobi_cmax) and abs(bpc_cmax - gurobi_cmax) <= 1e-9),
        "exact_conclusion": exact_conclusion(bool(getattr(bpc_result, "exact", False)), bpc_cmax, gurobi_cmax),
    }


def write_outputs(output_dir: str, rows: Iterable[Dict[str, Any]], details: List[Dict[str, Any]]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    rows = list(rows)
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
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(normalize_jsonable(details), f, ensure_ascii=False, indent=2)
    return {"csv": csv_path, "markdown": md_path, "details": details_path}


def write_certificate(output_dir: str, scale: str, result: Any) -> str:
    scale_dir = os.path.join(output_dir, str(scale).upper())
    os.makedirs(scale_dir, exist_ok=True)
    path = os.path.join(scale_dir, "bpc_certificate.json")
    payload = {
        "scale": str(scale).upper(),
        "status": str(getattr(result, "status", "")),
        "certificate": asdict(getattr(result, "certificate")),
        "diagnostics": getattr(result, "diagnostics", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalize_jsonable(payload), f, ensure_ascii=False, indent=2)
    return path
