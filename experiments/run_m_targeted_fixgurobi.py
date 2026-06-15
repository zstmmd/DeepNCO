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


GUROBI_CMAX = {
    "GUROBI-M5": 679.0,
    "GUROBI-M6": 687.0,
    "GUROBI-M8": 725.0,
}


# Profiles are intentionally scale-band tuned, not target-solution shortcuts.
# The quality gate is checked after the run: Cmax <= listed Gurobi Cmax + 10.
PROFILES: Dict[str, Dict[str, Any]] = {
    "GUROBI-M5": {
        "max_iters": 4,
        "time_limit": 150,
        "coarse": 20,
        "layer_order": "Y,YZ,XYZ",
        "mark_limit": 12,
        "accept_first": False,
        "candidate_stack_topk": 4,
        "max_candidate_stacks_per_order": 12,
        "candidate_station_topk_per_stack": 2,
        "scale_adaptive_candidate_prune": True,
    },
    "GUROBI-M6": {
        "max_iters": 4,
        "time_limit": 150,
        "coarse": 20,
        "layer_order": "Y,YZ,XYZ",
        "mark_limit": 12,
        "accept_first": False,
        "candidate_stack_topk": 4,
        "max_candidate_stacks_per_order": 12,
        "candidate_station_topk_per_stack": 2,
        "scale_adaptive_candidate_prune": True,
    },
    "GUROBI-M8": {
        "max_iters": 4,
        "time_limit": 180,
        "coarse": 25,
        "layer_order": "Y,YZ,XYZ",
        "mark_limit": 12,
        "accept_first": False,
        "candidate_stack_topk": 8,
        "max_candidate_stacks_per_order": 24,
        "candidate_station_topk_per_stack": 3,
        "scale_adaptive_candidate_prune": False,
    },
}


def _read_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
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


def _run_case(case: str, args: argparse.Namespace, out_root: str) -> Dict[str, Any]:
    profile = dict(PROFILES[case])
    case_root = os.path.join(out_root, case)
    os.makedirs(case_root, exist_ok=True)
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
        str(args.mip_gap),
        "--fixgurobi-candidate-trial-limit",
        "1",
        "--fixgurobi-cache-size",
        str(args.fixgurobi_cache_size),
        "--fixgurobi-compiled-cache-size",
        str(args.fixgurobi_compiled_cache_size),
        "--fixgurobi-candidate-stack-topk",
        str(profile["candidate_stack_topk"]),
        "--fixgurobi-max-candidate-stacks-per-order",
        str(profile["max_candidate_stacks_per_order"]),
        "--fixgurobi-candidate-station-topk-per-stack",
        str(profile["candidate_station_topk_per_stack"]),
        "--fixgurobi-force-candidate-stacks",
        (
            "--fixgurobi-enable-scale-adaptive-candidate-prune"
            if bool(profile.get("scale_adaptive_candidate_prune", True))
            else "--no-fixgurobi-enable-scale-adaptive-candidate-prune"
        ),
        "--no-fixgurobi-enable-compiled-cache",
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
    cmd.append("--fixgurobi-accept-first-improvement" if profile["accept_first"] else "--no-fixgurobi-accept-first-improvement")
    try:
        completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, timeout=float(args.case_timeout_sec))
        timeout_error = ""
    except subprocess.TimeoutExpired:
        completed = subprocess.CompletedProcess(cmd, returncode=124)
        timeout_error = f"case timeout after {float(args.case_timeout_sec):.1f}s"
    rows = _read_rows(os.path.join(case_root, "tra_gurobi_s1_s9_summary.csv"))
    row = dict(rows[0]) if rows else {"case": case, "status": f"missing_summary_rc_{completed.returncode}", "error_text": timeout_error}
    cmax = _safe_float(row.get("tra_gurobi_cmax"))
    runtime = _safe_float(row.get("tra_gurobi_total_runtime_sec", row.get("total_runtime_sec")))
    gate_cmax = float(GUROBI_CMAX[case]) + float(args.cmax_slack)
    row.update(
        {
            "case": case,
            "target_runtime_sec": float(args.runtime_cap_sec),
            "target_cmax_cap": gate_cmax,
            "runtime_le_cap": bool(runtime <= float(args.runtime_cap_sec)),
            "cmax_le_gurobi_plus_slack": bool(cmax <= gate_cmax),
            "targeted_accept": bool(runtime <= float(args.runtime_cap_sec) and cmax <= gate_cmax),
            "profile_json": json.dumps(profile, ensure_ascii=False, sort_keys=True),
            "case_output_root": case_root,
            "returncode": int(completed.returncode),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted no-final-validation TRA-FixGurobi tuning for M5/M6/M8.")
    parser.add_argument("--cases", nargs="+", default=["GUROBI-M5", "GUROBI-M6", "GUROBI-M8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-cap-sec", type=float, default=1600.0)
    parser.add_argument("--cmax-slack", type=float, default=10.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--fixgurobi-cache-size", type=int, default=512)
    parser.add_argument("--fixgurobi-compiled-cache-size", type=int, default=32)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--case-timeout-sec", type=float, default=1700.0)
    parser.add_argument("--fail-on-miss", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cases = [str(case).upper() for case in args.cases]
    unknown = [case for case in cases if case not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown targeted M cases: {unknown}")
    out_root = args.output_root or os.path.join(ROOT_DIR, "result", f"m_targeted_fixgurobi_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_root, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] targeted {case}", flush=True)
        row = _run_case(case, args, out_root)
        rows.append(row)
        _write_rows(os.path.join(out_root, "m_targeted_fixgurobi_summary.csv"), rows)
        print(
            f"  cmax={row.get('tra_gurobi_cmax')} runtime={row.get('tra_gurobi_total_runtime_sec')} "
            f"accept={row.get('targeted_accept')}",
            flush=True,
        )

    with open(os.path.join(out_root, "m_targeted_fixgurobi_config.json"), "w", encoding="utf-8") as f:
        json.dump({"profiles": PROFILES, "gurobi_cmax": GUROBI_CMAX, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    if bool(args.fail_on_miss) and any(not bool(row.get("targeted_accept", False)) for row in rows):
        failed = [str(row.get("case")) for row in rows if not bool(row.get("targeted_accept", False))]
        raise SystemExit(f"targeted M profiles missed gates: {', '.join(failed)}")
    print(f"summary={os.path.join(out_root, 'm_targeted_fixgurobi_summary.csv')}", flush=True)


if __name__ == "__main__":
    main()
