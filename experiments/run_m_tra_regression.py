from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CASE_PROFILES: Dict[str, Dict[str, Any]] = {
    "GUROBI-M1": {"max_iters": 4, "time_limit": 300, "coarse": 30, "accept_first": True, "layer_order": "AUTO", "yz_scope": "LOCALYZ", "mark_limit": 4},
    "GUROBI-M2": {"max_iters": 4, "time_limit": 300, "coarse": 30, "accept_first": True, "layer_order": "AUTO", "yz_scope": "LOCALYZ", "mark_limit": 4},
    "GUROBI-M3": {"max_iters": 2, "time_limit": 500, "coarse": 50, "accept_first": True, "layer_order": "Y,YZ", "yz_scope": "", "mark_limit": 20},
    "GUROBI-M4": {"max_iters": 7, "time_limit": 700, "coarse": 60, "accept_first": False, "layer_order": "Y,YZ,U,XYZ,XZ", "yz_scope": "", "mark_limit": 20},
    "GUROBI-M5": {"max_iters": 4, "time_limit": 700, "coarse": 60, "accept_first": False, "layer_order": "AUTO", "yz_scope": "", "mark_limit": 20},
    "GUROBI-M6": {"max_iters": 4, "time_limit": 500, "coarse": 40, "accept_first": False, "layer_order": "AUTO", "yz_scope": "", "mark_limit": 20},
    "GUROBI-M7": {"max_iters": 4, "time_limit": 700, "coarse": 60, "accept_first": False, "layer_order": "AUTO", "yz_scope": "", "mark_limit": 20},
    "GUROBI-M8": {"max_iters": 1, "time_limit": 900, "coarse": 40, "accept_first": False, "layer_order": "Y", "yz_scope": "", "mark_limit": 20},
    "GUROBI-M9": {"max_iters": 2, "time_limit": 900, "coarse": 60, "accept_first": False, "layer_order": "XYZ,Y", "yz_scope": "", "mark_limit": 20},
}


LOCKED_BEST_CMAX = {
    "GUROBI-M1": 489.0,
    "GUROBI-M2": 546.0,
    "GUROBI-M3": 558.0,
    "GUROBI-M4": 630.0,
    "GUROBI-M5": 679.0,
    "GUROBI-M6": 687.0,
    "GUROBI-M7": 708.0,
    "GUROBI-M8": 726.0,
    "GUROBI-M9": 720.0,
}


GUROBI_CMAX = {
    "GUROBI-M1": 489.0,
    "GUROBI-M2": 546.0,
    "GUROBI-M3": 558.0,
    "GUROBI-M4": 630.0,
    "GUROBI-M5": 679.0,
    "GUROBI-M6": 687.0,
    "GUROBI-M7": 708.0,
    "GUROBI-M8": 725.0,
    "GUROBI-M9": 731.0,
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _run_case(case: str, args: argparse.Namespace, batch_root: str) -> Dict[str, Any]:
    profile = CASE_PROFILES[case]
    case_root = _ensure_dir(os.path.join(batch_root, case))
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "Gurobi", "tra_gurobi.py"),
        "--cases",
        case,
        "--seed",
        str(args.seed),
        "--max-iters",
        str(profile["max_iters"]),
        "--fixgurobi-time-limit-sec",
        str(profile["time_limit"]),
        "--fixgurobi-coarse-time-limit-sec",
        str(profile["coarse"]),
        "--fixgurobi-mip-gap",
        "0.01",
        "--fixgurobi-candidate-trial-limit",
        "1",
        "--fixgurobi-cache-size",
        str(args.fixgurobi_cache_size),
        "--fixgurobi-compiled-cache-size",
        str(args.fixgurobi_compiled_cache_size),
        "--fixgurobi-enable-compiled-cache",
        "--fixgurobi-enable-two-stage",
        "--fixgurobi-enable-cutoff",
        "--fixgurobi-cheap-gate",
        "--no-fixgurobi-final-validation",
        "--no-known-target-guidance",
        "--no-target-table-fastpath",
        "--no-target-probe-case-presets",
        "--no-global-target-probe",
        "--no-resource-global-decomp-repair",
        "--tra-revolving-mode",
        "--revolving-enable-u-layer",
        "--revolving-layer-order",
        str(profile["layer_order"]),
        "--revolving-mark-limit",
        str(profile["mark_limit"]),
        "--no-resource-candidate-pool-log",
        "--compact-tra-summary-json",
        "--output-root",
        case_root,
    ]
    if profile["accept_first"]:
        cmd.append("--fixgurobi-accept-first-improvement")
    else:
        cmd.append("--no-fixgurobi-accept-first-improvement")
    if str(profile.get("yz_scope", "")).strip():
        cmd.extend(["--revolving-yz-fix-scope", str(profile["yz_scope"])])
    completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True)
    rows = _read_csv_rows(os.path.join(case_root, "tra_gurobi_s1_s9_summary.csv"))
    row = dict(rows[0]) if rows else {"case": case, "status": f"missing_summary_rc_{completed.returncode}"}
    cmax = _safe_float(row.get("tra_gurobi_cmax"))
    locked_best = float(LOCKED_BEST_CMAX[case])
    gurobi = float(GUROBI_CMAX[case])
    row.update(
        {
            "locked_best_cmax": locked_best,
            "gurobi_cmax": gurobi,
            "not_worse_than_locked_best": bool(cmax <= locked_best + 1e-9),
            "within_gurobi_plus_10": bool(cmax <= gurobi + 10.0 + 1e-9),
            "m_regression_accept": bool(cmax <= locked_best + 1e-9 and cmax <= gurobi + 10.0 + 1e-9),
            "profile_json": json.dumps(profile, ensure_ascii=False, sort_keys=True),
            "case_output_root": case_root,
            "returncode": int(completed.returncode),
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay locked no-target M1-M9 TRA regression profiles.")
    parser.add_argument("--cases", nargs="+", default=list(CASE_PROFILES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--fixgurobi-cache-size", type=int, default=256)
    parser.add_argument("--fixgurobi-compiled-cache-size", type=int, default=16)
    parser.add_argument("--fail-on-regression", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [str(case).upper() for case in args.cases]
    unknown = [case for case in cases if case not in CASE_PROFILES]
    if unknown:
        raise SystemExit(f"unknown M cases: {unknown}")
    batch_root = str(args.output_root or os.path.join(ROOT_DIR, "result", f"m_tra_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    batch_root = _ensure_dir(batch_root)
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] replay {case}", flush=True)
        row = _run_case(case, args, batch_root)
        rows.append(row)
        _write_csv(os.path.join(batch_root, "m_tra_regression_summary.csv"), rows)
        print(
            f"  cmax={row.get('tra_gurobi_cmax')} locked={row.get('locked_best_cmax')} "
            f"gurobi={row.get('gurobi_cmax')} accept={row.get('m_regression_accept')}",
            flush=True,
        )
    with open(os.path.join(batch_root, "m_tra_regression_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": cases,
                "seed": int(args.seed),
                "profiles": CASE_PROFILES,
                "locked_best_cmax": LOCKED_BEST_CMAX,
                "gurobi_cmax": GUROBI_CMAX,
                "target_guidance": False,
                "global_target_probe": False,
                "fixgurobi_final_validation": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if bool(args.fail_on_regression) and any(not bool(row.get("m_regression_accept", False)) for row in rows):
        failed = [str(row.get("case", "")) for row in rows if not bool(row.get("m_regression_accept", False))]
        raise SystemExit(f"M TRA regression failed: {', '.join(failed)}")
    print(f"summary={os.path.join(batch_root, 'm_tra_regression_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
