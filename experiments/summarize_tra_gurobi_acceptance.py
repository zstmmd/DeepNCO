from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List


TARGET_CMAX = {
    "GUROBI-S1": 178.0,
    "GUROBI-S2": 252.0,
    "GUROBI-S3": 266.0,
    "GUROBI-S4": 237.0,
    "GUROBI-S5": 268.0,
    "GUROBI-S6": 318.0,
    "GUROBI-S7": 348.0,
    "GUROBI-S8": 366.0,
    "GUROBI-S9": 438.0,
}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _read_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("details", "rows"):
        rows = payload.get(key, None)
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    return []


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TRA-Gurobi acceptance against no-warm Gurobi.")
    parser.add_argument("--gurobi-details-json", "--gurobi-details", dest="gurobi_details_json", required=True)
    parser.add_argument("--tra-summary-csv", "--tra-summary", dest="tra_summary_csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", default="", help="Optional comma-separated case allowlist.")
    args = parser.parse_args()

    gurobi_by_case = {
        str(row.get("scale", row.get("case", "")) or "").upper(): row
        for row in _read_json_rows(args.gurobi_details_json)
    }
    tra_rows = _read_csv_rows(args.tra_summary_csv)
    allow_cases = {
        str(item).strip().upper()
        for item in str(args.cases or "").split(",")
        if str(item).strip()
    }
    rows: List[Dict[str, Any]] = []
    for tra in tra_rows:
        case = str(tra.get("case", tra.get("scale", "")) or "").upper()
        if allow_cases and case not in allow_cases:
            continue
        gurobi = dict(gurobi_by_case.get(case, {}) or {})
        target = _safe_float(TARGET_CMAX.get(case, float("nan")))
        gurobi_cmax = _safe_float(gurobi.get("model_cmax", float("nan")))
        gurobi_runtime = _safe_float(gurobi.get("runtime_sec", gurobi.get("gurobi_runtime_sec", float("nan"))))
        gurobi_gap = _safe_float(gurobi.get("model_gap", float("nan")))
        tra_cmax = _safe_float(tra.get("tra_gurobi_cmax", tra.get("model_cmax", float("nan"))))
        tra_time_to_opt = _safe_float(tra.get("tra_gurobi_time_to_optimal_sec", float("nan")))
        tra_total_runtime = _safe_float(tra.get("tra_gurobi_total_runtime_sec", tra.get("total_runtime_sec", float("nan"))))
        used_known_target = _safe_bool(tra.get("known_target_guidance", False))
        used_target_probe = _safe_bool(tra.get("global_target_probe_enabled", False)) and _safe_bool(
            tra.get("global_target_probe_accepted", False)
        )
        if not math.isfinite(tra_time_to_opt) and math.isfinite(tra_cmax) and math.isfinite(target) and tra_cmax <= target + 1e-9:
            tra_time_to_opt = tra_total_runtime
        gap_vs_gurobi_pct = (
            (tra_cmax - gurobi_cmax) / max(1e-9, gurobi_cmax)
            if math.isfinite(tra_cmax) and math.isfinite(gurobi_cmax)
            else float("nan")
        )
        time_to_3pct_gap = tra_time_to_opt if math.isfinite(gap_vs_gurobi_pct) and gap_vs_gurobi_pct <= 0.03 else float("nan")
        optimal_pass = bool(math.isfinite(tra_cmax) and math.isfinite(target) and abs(tra_cmax - target) <= 1e-9)
        runtime_pass = bool(math.isfinite(tra_time_to_opt) and math.isfinite(gurobi_runtime) and tra_time_to_opt < gurobi_runtime)
        quality_pass = bool(math.isfinite(gap_vs_gurobi_pct) and gap_vs_gurobi_pct <= 0.03)
        rows.append(
            {
                "case": case,
                "target_cmax": target,
                "gurobi_no_warm_cmax": gurobi_cmax,
                "gurobi_no_warm_runtime_sec": gurobi_runtime,
                "gurobi_no_warm_gap": gurobi_gap,
                "tra_gurobi_cmax": tra_cmax,
                "tra_gurobi_time_to_optimal_sec": tra_time_to_opt,
                "tra_gurobi_total_runtime_sec": tra_total_runtime,
                "gap_vs_gurobi_pct": gap_vs_gurobi_pct,
                "tra_time_to_3pct_gap_sec": time_to_3pct_gap,
                "used_known_target_guidance": used_known_target,
                "used_target_probe": used_target_probe,
                "runtime_pass": runtime_pass,
                "quality_pass": quality_pass,
                "optimal_pass": optimal_pass,
                "acceptance_pass": bool(runtime_pass and quality_pass and optimal_pass),
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "tra_gurobi_acceptance_summary.csv")
    json_path = os.path.join(args.output_dir, "tra_gurobi_acceptance_summary.json")
    _write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"summary_csv={csv_path}")
    print(f"summary_json={json_path}")


if __name__ == "__main__":
    main()
