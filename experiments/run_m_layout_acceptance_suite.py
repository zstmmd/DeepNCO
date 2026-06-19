from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M_CASES = [f"GUROBI-M{i}" for i in range(1, 10)]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_command(cmd: List[str], timeout_sec: float, cwd: str = ROOT_DIR) -> Tuple[int, float, str]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=cwd, text=True, timeout=float(timeout_sec))
        return int(completed.returncode), float(time.perf_counter() - start), ""
    except subprocess.TimeoutExpired:
        return 124, float(time.perf_counter() - start), f"timeout after {float(timeout_sec):.1f}s"


def _latest_row(path: str, case: str) -> Dict[str, Any]:
    rows = _read_csv(path)
    for row in reversed(rows):
        if str(row.get("case", row.get("Case", "")) or "").upper() == str(case).upper():
            return dict(row)
    return {}


def _gurobi_row(case: str, output_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = os.path.join(output_dir, "summary.csv")
    existing = _latest_row(summary_path, case)
    if existing and bool(args.resume):
        return {
            "status": str(existing.get("status", "")),
            "cmax": _safe_float(existing.get("model_cmax", existing.get("upper_bound"))),
            "runtime_sec": _safe_float(existing.get("runtime_sec")),
            "gap": _safe_float(existing.get("model_gap", existing.get("gap"))),
            "source": summary_path,
            "returncode": 0,
            "resume_skip": True,
        }
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "experiments", "run_gurobi_benchmark18_suite.py"),
        "--scales",
        case,
        "--time-limit",
        str(args.gurobi_time_limit_sec),
        "--mip-gap",
        str(args.gurobi_mip_gap),
        "--output-dir",
        output_dir,
    ]
    if bool(args.show_gurobi):
        cmd.append("--show-gurobi")
    rc, wall, error = _run_command(cmd, timeout_sec=float(args.gurobi_timeout_sec))
    row = _latest_row(summary_path, case)
    return {
        "status": str(row.get("status", f"missing_summary_rc_{rc}")),
        "cmax": _safe_float(row.get("model_cmax", row.get("upper_bound"))),
        "runtime_sec": _safe_float(row.get("runtime_sec"), wall),
        "gap": _safe_float(row.get("model_gap", row.get("gap"))),
        "source": summary_path,
        "returncode": int(rc),
        "error_text": error,
        "resume_skip": False,
    }


def _tra_fix_profile(case: str, args: argparse.Namespace) -> Dict[str, Any]:
    idx = int(str(case).upper().split("-M")[-1])
    if idx <= 3:
        return {"max_iters": 4, "time_limit": 150, "coarse": 20, "layer_order": "Y,YZ,XYZ", "mark_limit": 10, "stack_topk": 4, "max_stacks": 14, "station_topk": 2}
    if idx <= 6:
        return {"max_iters": 4, "time_limit": 160, "coarse": 20, "layer_order": "Y,YZ,XYZ", "mark_limit": 12, "stack_topk": 5, "max_stacks": 18, "station_topk": 2}
    return {"max_iters": 4, "time_limit": 180, "coarse": 25, "layer_order": "Y,YZ,XYZ", "mark_limit": 12, "stack_topk": 8, "max_stacks": 24, "station_topk": 3}


def _tra_fix_row(case: str, output_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = os.path.join(output_dir, "tra_gurobi_s1_s9_summary.csv")
    existing = _latest_row(summary_path, case)
    if existing and bool(args.resume):
        return {
            "status": str(existing.get("status", "")),
            "cmax": _safe_float(existing.get("tra_gurobi_cmax")),
            "runtime_sec": _safe_float(existing.get("tra_gurobi_total_runtime_sec", existing.get("total_runtime_sec"))),
            "gap": _safe_float(existing.get("gap_vs_gurobi_pct")),
            "source": summary_path,
            "returncode": 0,
            "resume_skip": True,
        }
    profile = _tra_fix_profile(case, args)
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
        str(args.tra_fix_mip_gap),
        "--fixgurobi-candidate-trial-limit",
        "1",
        "--fixgurobi-cache-size",
        "512",
        "--fixgurobi-compiled-cache-size",
        "32",
        "--fixgurobi-candidate-stack-topk",
        str(profile["stack_topk"]),
        "--fixgurobi-max-candidate-stacks-per-order",
        str(profile["max_stacks"]),
        "--fixgurobi-candidate-station-topk-per-stack",
        str(profile["station_topk"]),
        "--fixgurobi-force-candidate-stacks",
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
        "--no-fixgurobi-accept-first-improvement",
        "--no-resource-candidate-pool-log",
        "--compact-tra-summary-json",
        "--output-root",
        output_dir,
    ]
    rc, wall, error = _run_command(cmd, timeout_sec=float(args.tra_fix_timeout_sec))
    row = _latest_row(summary_path, case)
    return {
        "status": str(row.get("status", f"missing_summary_rc_{rc}")),
        "cmax": _safe_float(row.get("tra_gurobi_cmax")),
        "runtime_sec": _safe_float(row.get("tra_gurobi_total_runtime_sec", row.get("total_runtime_sec")), wall),
        "gap": _safe_float(row.get("gap_vs_gurobi_pct")),
        "source": summary_path,
        "returncode": int(rc),
        "error_text": error,
        "resume_skip": False,
    }


def _tra_fast_row(case: str, output_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = os.path.join(output_dir, "tra_fast_summary.csv")
    existing = _latest_row(summary_path, case)
    if existing and bool(args.resume):
        return {
            "status": str(existing.get("status", "")),
            "cmax": _safe_float(existing.get("tra_fast_cmax")),
            "runtime_sec": _safe_float(existing.get("tra_fast_runtime_sec")),
            "gap": _safe_float(existing.get("tra_fast_vs_gurobi_gap")),
            "source": summary_path,
            "returncode": 0,
            "resume_skip": True,
        }
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "experiments", "run_tra_fast.py"),
        "--cases",
        case,
        "--seed",
        str(args.seed),
        "--case-timeout-sec",
        str(args.tra_fast_timeout_sec),
        "--max-iters",
        str(args.tra_fast_max_iters),
        "--no-improve-limit",
        "3",
        "--calibration-mode",
        "off",
        "--output-root",
        output_dir,
    ]
    rc, wall, error = _run_command(cmd, timeout_sec=float(args.tra_fast_timeout_sec) + 60.0)
    row = _latest_row(summary_path, case)
    return {
        "status": str(row.get("status", f"missing_summary_rc_{rc}")),
        "cmax": _safe_float(row.get("tra_fast_cmax")),
        "runtime_sec": _safe_float(row.get("tra_fast_runtime_sec"), wall),
        "gap": _safe_float(row.get("tra_fast_vs_gurobi_gap")),
        "source": summary_path,
        "returncode": int(rc),
        "error_text": error,
        "resume_skip": False,
    }


def _case_acceptance(case: str, row: Dict[str, Any], previous: Dict[str, Any] | None, args: argparse.Namespace) -> Dict[str, Any]:
    gurobi_cmax = _safe_float(row.get("gurobi_cmax"))
    fix_cmax = _safe_float(row.get("tra_fix_cmax"))
    fast_cmax = _safe_float(row.get("tra_fast_cmax"))
    gurobi_runtime = _safe_float(row.get("gurobi_runtime_sec"))
    fix_runtime = _safe_float(row.get("tra_fix_runtime_sec"))
    fast_runtime = _safe_float(row.get("tra_fast_runtime_sec"))
    fix_abs_gap = abs(fix_cmax - gurobi_cmax) if math.isfinite(fix_cmax) and math.isfinite(gurobi_cmax) else float("nan")
    fast_gap = (fast_cmax - gurobi_cmax) / max(1e-9, gurobi_cmax) if math.isfinite(fast_cmax) and math.isfinite(gurobi_cmax) else float("nan")
    cmax_gt_s9 = bool(math.isfinite(gurobi_cmax) and gurobi_cmax > float(args.min_cmax))
    cmax_increasing = True
    runtime_scale_ok = True
    if previous:
        prev_cmax = _safe_float(previous.get("gurobi_cmax"))
        cmax_increasing = bool(math.isfinite(gurobi_cmax) and math.isfinite(prev_cmax) and gurobi_cmax > prev_cmax)
        for prefix in ["gurobi", "tra_fix", "tra_fast"]:
            cur = _safe_float(row.get(f"{prefix}_runtime_sec"))
            prev = _safe_float(previous.get(f"{prefix}_runtime_sec"))
            if math.isfinite(cur) and math.isfinite(prev) and cur < prev * float(args.min_runtime_ratio_vs_previous):
                runtime_scale_ok = False
    return {
        "fix_abs_gap": fix_abs_gap,
        "fast_gap_vs_gurobi": fast_gap,
        "gurobi_gap_ok": bool(_safe_float(row.get("gurobi_gap")) <= float(args.gurobi_mip_gap) + 1e-9),
        "fix_quality_ok": bool(math.isfinite(fix_abs_gap) and fix_abs_gap <= float(args.tra_fix_cmax_abs_tol)),
        "fix_runtime_ok": bool(math.isfinite(fix_runtime) and fix_runtime <= float(args.tra_fix_runtime_cap_sec)),
        "fast_quality_ok": bool(math.isfinite(fast_gap) and fast_gap <= float(args.tra_fast_gap_cap)),
        "fast_runtime_ok": bool(math.isfinite(fast_runtime) and fast_runtime <= float(args.tra_fast_runtime_cap_sec)),
        "cmax_gt_s9_ok": cmax_gt_s9,
        "cmax_increasing_ok": bool(cmax_increasing),
        "runtime_scale_ok": bool(runtime_scale_ok),
    }


def _all_acceptance_ok(row: Dict[str, Any]) -> bool:
    keys = [
        "gurobi_gap_ok",
        "fix_quality_ok",
        "fix_runtime_ok",
        "fast_quality_ok",
        "fast_runtime_ok",
        "cmax_gt_s9_ok",
        "cmax_increasing_ok",
        "runtime_scale_ok",
    ]
    return all(bool(row.get(key, False)) for key in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential M1-M9 layout acceptance suite after warehouseMap changes.")
    parser.add_argument("--cases", nargs="+", default=list(M_CASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-gurobi", action="store_true", default=False)
    parser.add_argument("--gurobi-time-limit-sec", type=float, default=3600.0)
    parser.add_argument("--gurobi-timeout-sec", type=float, default=3900.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=0.01)
    parser.add_argument("--tra-fix-mip-gap", type=float, default=0.01)
    parser.add_argument("--tra-fix-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--tra-fast-timeout-sec", type=float, default=300.0)
    parser.add_argument("--tra-fast-max-iters", type=int, default=50)
    parser.add_argument("--tra-fix-cmax-abs-tol", type=float, default=3.0)
    parser.add_argument("--tra-fix-runtime-cap-sec", type=float, default=1600.0)
    parser.add_argument("--tra-fast-runtime-cap-sec", type=float, default=300.0)
    parser.add_argument("--tra-fast-gap-cap", type=float, default=0.03)
    parser.add_argument("--min-runtime-ratio-vs-previous", type=float, default=0.70)
    parser.add_argument("--min-cmax", type=float, default=438.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.output_root or os.path.join(ROOT_DIR, "result", f"m_layout_acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root = _ensure_dir(out_root)
    summary_path = os.path.join(out_root, "m_layout_acceptance_summary.csv")
    rows = _read_csv(summary_path) if bool(args.resume) else []
    previous = rows[-1] if rows else None
    completed = {str(row.get("case", "")).upper(): dict(row) for row in rows}
    for case in [str(item).upper() for item in args.cases]:
        if case in completed and bool(completed[case].get("acceptance_ok", False)) and bool(args.resume):
            print(f"[resume-skip] {case} acceptance_ok", flush=True)
            previous = completed[case]
            continue
        print(f"[case] {case}", flush=True)
        case_root = _ensure_dir(os.path.join(out_root, case))
        gurobi = _gurobi_row(case, _ensure_dir(os.path.join(case_root, "gurobi")), args)
        print(f"  gurobi status={gurobi['status']} cmax={gurobi['cmax']} runtime={gurobi['runtime_sec']:.2f}s gap={gurobi['gap']}", flush=True)
        tra_fix = _tra_fix_row(case, _ensure_dir(os.path.join(case_root, "tra_fixgurobi")), args)
        print(f"  tra_fix status={tra_fix['status']} cmax={tra_fix['cmax']} runtime={tra_fix['runtime_sec']:.2f}s", flush=True)
        tra_fast = _tra_fast_row(case, _ensure_dir(os.path.join(case_root, "tra_fast")), args)
        print(f"  tra_fast status={tra_fast['status']} cmax={tra_fast['cmax']} runtime={tra_fast['runtime_sec']:.2f}s", flush=True)
        row: Dict[str, Any] = {
            "case": case,
            "gurobi_status": gurobi.get("status", ""),
            "gurobi_cmax": gurobi.get("cmax"),
            "gurobi_gap": gurobi.get("gap"),
            "gurobi_runtime_sec": gurobi.get("runtime_sec"),
            "tra_fix_status": tra_fix.get("status", ""),
            "tra_fix_cmax": tra_fix.get("cmax"),
            "tra_fix_runtime_sec": tra_fix.get("runtime_sec"),
            "tra_fast_status": tra_fast.get("status", ""),
            "tra_fast_cmax": tra_fast.get("cmax"),
            "tra_fast_runtime_sec": tra_fast.get("runtime_sec"),
            "case_root": case_root,
        }
        row.update(_case_acceptance(case, row, previous, args))
        row["acceptance_ok"] = bool(_all_acceptance_ok(row))
        rows = [existing for existing in rows if str(existing.get("case", "")).upper() != case]
        rows.append(row)
        _write_csv(summary_path, rows)
        _write_json(os.path.join(out_root, "m_layout_acceptance_config.json"), {"args": vars(args), "cases": args.cases})
        print(f"  acceptance_ok={row['acceptance_ok']}", flush=True)
        if bool(args.stop_on_fail) and not bool(row["acceptance_ok"]):
            print(f"stop_on_fail: {case} failed acceptance; inspect {summary_path}", flush=True)
            break
        previous = row
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
