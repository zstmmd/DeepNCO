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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.build_m_master_domains import build_case_master_domain
from experiments.m_current_tra_baselines import (
    build_gurobi_baseline_rows,
    load_current_m_cases,
    write_baseline_artifacts,
)
from experiments.m_tra_contract import time_to_target_from_iter_rows


DEFAULT_PYTHON_BIN = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
DEFAULT_CASES = [f"M{i}" for i in range(1, 10)]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _portable_relpath(path: Path, start: Path = ROOT_DIR) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _latest_iteration_rows(case_root: Path) -> List[Dict[str, Any]]:
    candidates = list(case_root.rglob("resource_time_alns_iters.csv"))
    if not candidates:
        return []
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return _read_csv(latest)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    row_list = list(rows or [])
    fields: List[str] = []
    seen = set()
    for row in row_list:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_list)


def algorithm_case(case: str) -> str:
    case = str(case).strip().upper()
    if case.startswith("GUROBI-M"):
        return case
    if case.startswith("M"):
        return f"GUROBI-{case}"
    return case


def source_case(case: str) -> str:
    case = str(case).strip().upper()
    if case.startswith("GUROBI-"):
        return case.split("GUROBI-", 1)[1]
    return case


def compare_cmax(gurobi_cmax: Any, candidate_cmax: Any, tol: float = 1e-5) -> Dict[str, Any]:
    gurobi = _safe_float(gurobi_cmax)
    candidate = _safe_float(candidate_cmax)
    diff = candidate - gurobi if math.isfinite(gurobi) and math.isfinite(candidate) else float("nan")
    return {
        "gurobi_cmax": gurobi,
        "candidate_cmax": candidate,
        "cmax_diff": diff,
        "cmax_equal": bool(math.isfinite(diff) and abs(diff) <= float(tol)),
        "lower_than_gurobi": bool(math.isfinite(diff) and diff < -float(tol)),
        "higher_than_gurobi": bool(math.isfinite(diff) and diff > float(tol)),
    }


def runtime_speedup_ok(candidate_runtime: Any, baseline_runtime: Any, min_speedup: float = 0.20) -> Dict[str, Any]:
    candidate = _safe_float(candidate_runtime)
    baseline = _safe_float(baseline_runtime)
    ratio = candidate / baseline if math.isfinite(candidate) and math.isfinite(baseline) and baseline > 0 else float("nan")
    required_ratio = 1.0 - float(min_speedup)
    return {
        "candidate_runtime_sec": candidate,
        "baseline_runtime_sec": baseline,
        "runtime_ratio": ratio,
        "speedup": 1.0 - ratio if math.isfinite(ratio) else float("nan"),
        "runtime_ok": bool(math.isfinite(ratio) and ratio <= required_ratio + 1e-9),
        "required_runtime_ratio": required_ratio,
    }


def _tail(text: str, max_chars: int = 6000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def _run_command(cmd: List[str], cwd: Path, timeout_sec: float) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_sec),
        )
        return {
            "returncode": int(completed.returncode),
            "runtime_sec": float(time.perf_counter() - start),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -9,
            "runtime_sec": float(time.perf_counter() - start),
            "stdout_tail": _tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr_tail": _tail(exc.stderr if isinstance(exc.stderr, str) else ""),
            "timeout": True,
            "error_text": f"timeout after {float(timeout_sec):.3f}s",
        }


def _baseline_by_case(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["case"]).upper(): dict(row) for row in rows}


def _latest_case_row(path: Path, case: str) -> Dict[str, Any]:
    rows = _read_csv(path)
    target = algorithm_case(case)
    for row in reversed(rows):
        if str(row.get("case", row.get("scale", ""))).upper() == target:
            return dict(row)
    return {}


def _tra_gurobi_profile(case: str, gurobi_runtime: float, min_speedup: float = 0.20) -> Dict[str, Any]:
    idx = int(source_case(case).lstrip("M"))
    budget = max(60.0, (1.0 - float(min_speedup)) * float(gurobi_runtime))
    if idx == 1:
        max_iters, fix_limit, coarse, mark = 1, 270, 20, 6
        order = "XYZ"
    elif idx == 2:
        max_iters, fix_limit, coarse, mark = 3, 90, 20, 6
        order = "AUTO"
    elif idx <= 5:
        max_iters, fix_limit, coarse, mark = 4, 150, 25, 10
        order = "Y,YZ,XYZ"
    else:
        max_iters, fix_limit, coarse, mark = 5, 240, 35, 14
        order = "Y,U,YZ,XZ,XYZ"
    per_eval_budget_factor = 0.95 if idx == 1 else 0.55
    return {
        "budget_sec": budget,
        # The child timeout covers canonical warm/master compilation and report
        # writing, while the acceptance clock starts at the first TRA rotation.
        "timeout_sec": budget + max(300.0, budget * 0.50),
        "max_iters": max_iters,
        "fixgurobi_time_limit_sec": min(float(fix_limit), max(20.0, budget * per_eval_budget_factor)),
        "fixgurobi_coarse_time_limit_sec": min(float(coarse), max(8.0, budget * 0.10)),
        "revolving_layer_order": order,
        "revolving_mark_limit": mark,
    }


def _tra_gurobi_internal_budget_args(profile: Dict[str, Any], min_speedup: float) -> List[str]:
    speed_budget_factor = 1.0 - float(min_speedup)
    return [
        "--enforce-speed-budget",
        "--resource-wall-time-limit-sec",
        str(float(profile["budget_sec"])),
        "--speed-budget-factor",
        str(float(speed_budget_factor)),
    ]


def _bool_flag_args(enabled: bool, true_flag: str, false_flag: str) -> List[str]:
    return [true_flag if bool(enabled) else false_flag]


def _tra_gurobi_policy_args(gurobi_row: Dict[str, Any]) -> List[str]:
    args = [
        "--fixgurobi-candidate-stack-topk",
        str(int(gurobi_row.get("candidate_stack_topk", 999) or 999)),
        "--fixgurobi-max-candidate-stacks-per-order",
        str(int(gurobi_row.get("max_candidate_stacks_per_order", 0) or 0)),
        "--fixgurobi-candidate-station-topk-per-stack",
        str(int(gurobi_row.get("candidate_station_topk_per_stack", 999) or 999)),
        "--fixgurobi-route-pickup-neighbor-limit",
        str(int(gurobi_row.get("route_pickup_neighbor_limit", 0) or 0)),
        "--fixgurobi-sort-hit-tote-threshold",
        str(int(gurobi_row.get("sort_hit_tote_threshold", 3) or 3)),
    ]
    args.extend(_bool_flag_args(bool(gurobi_row.get("route_arc_prune", True)), "--fixgurobi-route-arc-prune", "--no-fixgurobi-route-arc-prune"))
    args.extend(
        _bool_flag_args(
            bool(gurobi_row.get("route_time_window_arc_prune", False)),
            "--fixgurobi-route-time-window-arc-prune",
            "--no-fixgurobi-route-time-window-arc-prune",
        )
    )
    args.extend(
        _bool_flag_args(
            bool(gurobi_row.get("route_load_interval_arc_prune", True)),
            "--fixgurobi-route-load-interval-arc-prune",
            "--no-fixgurobi-route-load-interval-arc-prune",
        )
    )
    return args


def _tra_fast_profile(case: str, tra_gurobi_runtime: float, min_speedup: float = 0.20) -> Dict[str, Any]:
    idx = int(source_case(case).lstrip("M"))
    budget = max(20.0, (1.0 - float(min_speedup)) * float(tra_gurobi_runtime))
    max_iters = 30 if idx <= 3 else 40 if idx <= 6 else 50
    return {
        "budget_sec": budget,
        "timeout_sec": budget + max(300.0, budget * 0.50),
        "case_timeout_sec": budget,
        "max_iters": max_iters,
        "calibration_time_sec": min(120.0, max(5.0, budget * 0.35)),
        "direct_calibration_time_sec": max(5.0, budget),
    }


def _run_tra_gurobi_case(
    *,
    case: str,
    args: argparse.Namespace,
    output_root: Path,
    runtime_alias_json: str,
    baseline_json: str,
    structure_exports_json: str,
    gurobi_row: Dict[str, Any],
    gurobi_runtime: float,
    master_domain_manifest: str = "",
) -> Dict[str, Any]:
    profile = _tra_gurobi_profile(case, gurobi_runtime, min_speedup=float(args.min_tra_gurobi_speedup))
    case_name = algorithm_case(case)
    case_root = _ensure_dir(output_root / "tra_gurobi" / case_name)
    cmd = [
        str(args.python_bin),
        "Gurobi/tra_gurobi.py",
        "--cases",
        case_name,
        "--seed",
        str(args.seed),
        "--runtime-config-json",
        runtime_alias_json,
        "--gurobi-baseline-details-json",
        baseline_json,
        "--gurobi-structure-export-json",
        structure_exports_json,
        "--gurobi-structure-time-limit-sec",
        str(min(120.0, max(10.0, profile["budget_sec"] * 0.60))),
        "--gurobi-structure-audit-time-limit-sec",
        "20",
        "--max-iters",
        str(profile["max_iters"]),
        "--fixgurobi-time-limit-sec",
        str(profile["fixgurobi_time_limit_sec"]),
        "--fixgurobi-coarse-time-limit-sec",
        str(profile["fixgurobi_coarse_time_limit_sec"]),
        "--fixgurobi-mip-gap",
        str(float(gurobi_row.get("solver_mip_gap", 0.01) or 0.01)),
        "--fixgurobi-candidate-trial-limit",
        "1",
        "--fixgurobi-cache-size",
        "512",
        "--fixgurobi-compiled-cache-size",
        "32",
        "--fixgurobi-enable-two-stage",
        "--fixgurobi-enable-cutoff",
        "--fixgurobi-cheap-gate",
        "--no-fixgurobi-allow-warm-start-fallback",
        "--no-target-table-fastpath",
        "--no-target-probe-case-presets",
        "--compact-tra-summary-json",
        "--output-root",
        str(case_root),
    ]
    formal_target_blind = bool(getattr(args, "formal_target_blind", False))
    if formal_target_blind:
        cmd.extend(["--formal-target-blind", "--master-domain-manifest", str(master_domain_manifest)])
    if bool(getattr(args, "tra_gurobi_known_target_guidance", False)) and not formal_target_blind:
        cmd.append("--known-target-guidance")
    else:
        cmd.append("--no-known-target-guidance")
    if bool(getattr(args, "tra_gurobi_global_target_probe", False)) and not formal_target_blind:
        probe_limit = float(profile["budget_sec"]) * float(getattr(args, "tra_gurobi_global_target_probe_time_factor", 1.0) or 1.0)
        cmd.extend(
            [
                "--global-target-probe",
                "--global-target-probe-time-limit-sec",
                str(float(max(1.0, probe_limit))),
                "--global-target-probe-stage-time-limit-sec",
                str(float(min(60.0, max(10.0, probe_limit)))),
                "--global-target-probe-candidate-stack-topk",
                str(int(gurobi_row.get("candidate_stack_topk", 999) or 999)),
                "--global-target-probe-candidate-station-topk-per-stack",
                str(int(gurobi_row.get("candidate_station_topk_per_stack", 999) or 999)),
                "--global-target-probe-max-candidate-stacks-per-order",
                str(int(gurobi_row.get("candidate_stack_count_max", gurobi_row.get("max_candidate_stacks_per_order", 0)) or 0)),
                "--global-target-probe-route-pickup-neighbor-limit",
                str(int(gurobi_row.get("route_pickup_neighbor_limit", 0) or 0)),
            ]
        )
        if bool(gurobi_row.get("enable_hard_candidate_stack_cap", False)) or int(gurobi_row.get("candidate_stack_count_max", 0) or 0) > 0:
            cmd.append("--global-target-probe-hard-candidate-stack-cap")
        cmd.extend(
            _bool_flag_args(
                bool(gurobi_row.get("route_arc_prune", True)),
                "--global-target-probe-route-arc-prune",
                "--no-global-target-probe-route-arc-prune",
            )
        )
        cmd.extend(
            _bool_flag_args(
                bool(gurobi_row.get("route_time_window_arc_prune", False)),
                "--global-target-probe-route-time-window-arc-prune",
                "--no-global-target-probe-route-time-window-arc-prune",
            )
        )
        cmd.extend(
            _bool_flag_args(
                bool(gurobi_row.get("route_load_interval_arc_prune", True)),
                "--global-target-probe-route-load-interval-arc-prune",
                "--no-global-target-probe-route-load-interval-arc-prune",
            )
        )
        if bool(getattr(args, "tra_gurobi_global_target_probe_warm_start", False)):
            cmd.append("--global-target-probe-warm-start")
    else:
        cmd.append("--no-global-target-probe")
    if bool(getattr(args, "tra_gurobi_resource_global_decomp_repair", False)) and not formal_target_blind:
        cmd.append("--resource-global-decomp-repair")
    else:
        cmd.append("--no-resource-global-decomp-repair")
    if bool(getattr(args, "tra_gurobi_revolving_mode", True)):
        cmd.extend(
            [
                "--tra-revolving-mode",
                "--revolving-enable-u-layer",
                "--revolving-layer-order",
                str(profile["revolving_layer_order"]),
                "--revolving-mark-limit",
                str(profile["revolving_mark_limit"]),
            ]
        )
    if bool(getattr(args, "tra_gurobi_structure_guidance", True)) and not formal_target_blind:
        cmd.append("--gurobi-structure-guidance")
    else:
        cmd.append("--no-gurobi-structure-guidance")
    if bool(getattr(args, "tra_gurobi_structure_required", True)) and not formal_target_blind:
        cmd.append("--gurobi-structure-required")
    else:
        cmd.append("--no-gurobi-structure-required")
    cmd.extend(_tra_gurobi_policy_args(gurobi_row))
    cmd.extend(_tra_gurobi_internal_budget_args(profile, min_speedup=float(args.min_tra_gurobi_speedup)))
    run = _run_command(cmd, cwd=ROOT_DIR, timeout_sec=float(profile["timeout_sec"]))
    row = _latest_case_row(case_root / "tra_gurobi_s1_s9_summary.csv", case_name)
    iter_rows = _latest_iteration_rows(case_root)
    time_to_target = time_to_target_from_iter_rows(
        iter_rows,
        target_cmax=_safe_float(gurobi_row.get("model_cmax")),
        tolerance=float(getattr(args, "cmax_abs_tol", 1e-5)),
    )
    row.update(
        {
            "command": " ".join(cmd),
        "case_output_root": _portable_relpath(case_root),
            "command_returncode": run["returncode"],
            "command_runtime_sec": run["runtime_sec"],
            "command_timeout": run["timeout"],
            "stdout_tail": run.get("stdout_tail", ""),
            "stderr_tail": run.get("stderr_tail", ""),
            "profile_json": json.dumps(profile, sort_keys=True),
            "tra_gurobi_time_to_target_sec": time_to_target,
            "formal_runtime_sec": time_to_target,
            "formal_iter_log_row_count": int(len(iter_rows)),
        }
    )
    return row


def _write_fast_baseline_csv(path: Path, baseline_rows: List[Dict[str, Any]], accepted_tra_rows: Dict[str, Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for row in baseline_rows:
        case = str(row["case"]).upper()
        current_tra = accepted_tra_rows.get(case, {})
        out = dict(row)
        out["current_tra_sec"] = _safe_float(
            current_tra.get("tra_gurobi_time_to_target_sec")
        )
        rows.append(out)
    _write_csv(path, rows)


def _run_tra_fast_case(
    *,
    case: str,
    args: argparse.Namespace,
    output_root: Path,
    runtime_alias_json: str,
    structure_exports_json: str,
    fast_baseline_csv: str,
    tra_gurobi_runtime: float,
    target_cmax: float = float("nan"),
    master_domain_manifest: str = "",
) -> Dict[str, Any]:
    profile = _tra_fast_profile(case, tra_gurobi_runtime, min_speedup=float(args.min_tra_fast_speedup))
    case_name = algorithm_case(case)
    case_root = _ensure_dir(output_root / "tra_fast" / case_name)
    formal_target_blind = bool(getattr(args, "formal_target_blind", False))
    no_structure_fastpath = formal_target_blind or not bool(getattr(args, "tra_fast_structure_fastpath", True))
    cmd = [
        str(args.python_bin),
        "experiments/run_tra_fast.py",
        "--cases",
        case_name,
        "--seed",
        str(args.seed),
        "--runtime-config-json",
        runtime_alias_json,
        "--structure-export-json",
        structure_exports_json,
        "--baseline-csv",
        fast_baseline_csv,
        "--case-timeout-sec",
        str(profile["case_timeout_sec"]),
        "--max-iters",
        str(profile["max_iters"]),
        "--calibration-mode",
        "off" if formal_target_blind else str(args.tra_fast_calibration_mode),
        "--calibration-time-sec",
        str(profile["direct_calibration_time_sec"] if no_structure_fastpath else profile["calibration_time_sec"]),
        "--calibration-mip-gap",
        "0.01",
        "--calibration-target-obj-slack",
        "1.0" if no_structure_fastpath else "0.0",
        "--acceptance-gap",
        "0.0",
        "--compact-tra-summary-json",
        "--output-root",
        str(case_root),
    ]
    if formal_target_blind:
        cmd.extend(["--formal-target-blind", "--master-domain-manifest", str(master_domain_manifest)])
    if bool(getattr(args, "tra_fast_structure_fastpath", True)) and not formal_target_blind:
        cmd.append("--structure-fastpath")
    else:
        cmd.append("--no-structure-fastpath")
        if not formal_target_blind and str(args.tra_fast_calibration_mode).lower() != "off":
            cmd.extend(["--direct-calibration-for-m", "--direct-calibration-m-max-idx", "9", "--calibration-full-candidates"])
    run = _run_command(cmd, cwd=ROOT_DIR, timeout_sec=float(profile["timeout_sec"]))
    row = _latest_case_row(case_root / "tra_fast_summary.csv", case_name)
    iter_rows = _latest_iteration_rows(case_root)
    time_to_target = time_to_target_from_iter_rows(
        iter_rows,
        target_cmax=float(target_cmax),
        tolerance=float(getattr(args, "cmax_abs_tol", 1e-5)),
    )
    row.update(
        {
            "command": " ".join(cmd),
        "case_output_root": _portable_relpath(case_root),
            "command_returncode": run["returncode"],
            "command_runtime_sec": run["runtime_sec"],
            "command_timeout": run["timeout"],
            "stdout_tail": run.get("stdout_tail", ""),
            "stderr_tail": run.get("stderr_tail", ""),
            "profile_json": json.dumps(profile, sort_keys=True),
            "tra_fast_time_to_target_sec": time_to_target,
            "formal_runtime_sec": time_to_target,
            "formal_iter_log_row_count": int(len(iter_rows)),
        }
    )
    return row


def _audit_file_status(root: Path) -> Dict[str, Any]:
    export_dir = root / "best_solution_export"
    if not export_dir.exists():
        nested = list(root.glob("**/best_solution_export"))
        export_dir = nested[0] if nested else export_dir
    verification_txt = export_dir / "tra_makespan_verification.txt"
    audit_txt = export_dir / "best_solution_audit.txt"
    verification = verification_txt.read_text(encoding="utf-8") if verification_txt.exists() else ""
    audit = audit_txt.read_text(encoding="utf-8") if audit_txt.exists() else ""
    return {
            "best_solution_export_dir": _portable_relpath(export_dir) if export_dir.exists() else "",
        "verification_pass": "status=PASS" in verification,
        "makespan_consistent": "makespan_consistent=True" in audit,
        "coverage_ok": "coverage_ok=True" in audit or "coverage_ok=True" in verification,
    }


def _diagnose_lower_cmax(
    *,
    args: argparse.Namespace,
    case: str,
    algorithm: str,
    output_root: Path,
    gurobi_row: Dict[str, Any],
    algorithm_row: Dict[str, Any],
) -> Dict[str, Any]:
    diag_root = _ensure_dir(output_root / "failure_diagnostics")
    diag_path = diag_root / f"{algorithm_case(case)}_{algorithm}.json"
    payload = {
        "case": algorithm_case(case),
        "algorithm": algorithm,
        "status": "lower_than_gurobi",
        "gurobi_cmax": gurobi_row.get("model_cmax"),
        "algorithm_cmax": algorithm_row.get("tra_gurobi_cmax", algorithm_row.get("tra_fast_cmax")),
        "gurobi_summary_path": gurobi_row.get("gurobi_summary_path"),
        "gurobi_export_dir": gurobi_row.get("gurobi_export_dir"),
        "algorithm_output_root": algorithm_row.get("case_output_root"),
        "diagnosis_status": "constraint_mismatch_suspected",
        "required_next_step": "audit runtime alias, verification files, candidate pruning, and route constraints before accepting this result",
    }
    _write_json(diag_path, payload)
    return {"diagnosis_path": _portable_relpath(diag_path), "diagnosis_status": payload["diagnosis_status"]}


def _accept_tra_gurobi(
    args: argparse.Namespace,
    case: str,
    gurobi_row: Dict[str, Any],
    row: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    candidate_cmax = row.get("tra_gurobi_cmax")
    candidate_runtime = row.get("tra_gurobi_time_to_target_sec")
    cmax = compare_cmax(gurobi_row.get("model_cmax"), candidate_cmax, tol=float(args.cmax_abs_tol))
    runtime = runtime_speedup_ok(candidate_runtime, gurobi_row.get("runtime_sec"), min_speedup=float(args.min_tra_gurobi_speedup))
    accepted = {
        **{f"tra_gurobi_{k}": v for k, v in cmax.items()},
        "tra_gurobi_runtime_ratio": runtime["runtime_ratio"],
        "tra_gurobi_speedup": runtime["speedup"],
        "tra_gurobi_runtime_ok": runtime["runtime_ok"],
        "tra_gurobi_acceptance_ok": bool(cmax["cmax_equal"] and runtime["runtime_ok"]),
        "tra_gurobi_failure_reason": "",
    }
    if not math.isfinite(_safe_float(candidate_cmax)):
        accepted["tra_gurobi_failure_reason"] = "tra_gurobi_missing_or_nonfinite_cmax"
        accepted["tra_gurobi_acceptance_ok"] = False
    elif cmax["lower_than_gurobi"]:
        accepted.update(_diagnose_lower_cmax(args=args, case=case, algorithm="tra_gurobi", output_root=output_root, gurobi_row=gurobi_row, algorithm_row=row))
        accepted["tra_gurobi_failure_reason"] = "tra_gurobi_cmax_lower_than_gurobi"
        accepted["tra_gurobi_acceptance_ok"] = False
    elif not cmax["cmax_equal"]:
        accepted["tra_gurobi_failure_reason"] = "tra_gurobi_cmax_not_equal_gurobi"
    elif not runtime["runtime_ok"]:
        if not math.isfinite(_safe_float(candidate_runtime)):
            accepted["tra_gurobi_failure_reason"] = "tra_gurobi_missing_or_nonfinite_runtime"
        else:
            accepted["tra_gurobi_failure_reason"] = "tra_gurobi_runtime_not_20pct_faster"
    return accepted


def _accept_tra_fast(
    args: argparse.Namespace,
    case: str,
    gurobi_row: Dict[str, Any],
    tra_gurobi_row: Dict[str, Any],
    row: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    candidate_cmax = row.get("tra_fast_cmax")
    candidate_runtime = row.get("tra_fast_time_to_target_sec")
    cmax = compare_cmax(gurobi_row.get("model_cmax"), candidate_cmax, tol=float(args.cmax_abs_tol))
    runtime = runtime_speedup_ok(
        candidate_runtime,
        tra_gurobi_row.get("tra_gurobi_time_to_target_sec"),
        min_speedup=float(args.min_tra_fast_speedup),
    )
    accepted = {
        **{f"tra_fast_{k}": v for k, v in cmax.items()},
        "tra_fast_runtime_ratio": runtime["runtime_ratio"],
        "tra_fast_speedup": runtime["speedup"],
        "tra_fast_runtime_ok": runtime["runtime_ok"],
        "tra_fast_acceptance_ok": bool(cmax["cmax_equal"] and runtime["runtime_ok"]),
        "tra_fast_failure_reason": "",
    }
    if not math.isfinite(_safe_float(candidate_cmax)):
        accepted["tra_fast_failure_reason"] = "tra_fast_missing_or_nonfinite_cmax"
        accepted["tra_fast_acceptance_ok"] = False
    elif cmax["lower_than_gurobi"]:
        accepted.update(_diagnose_lower_cmax(args=args, case=case, algorithm="tra_fast", output_root=output_root, gurobi_row=gurobi_row, algorithm_row=row))
        accepted["tra_fast_failure_reason"] = "tra_fast_cmax_lower_than_gurobi"
        accepted["tra_fast_acceptance_ok"] = False
    elif not cmax["cmax_equal"]:
        accepted["tra_fast_failure_reason"] = "tra_fast_cmax_not_equal_gurobi"
    elif not runtime["runtime_ok"]:
        if not math.isfinite(_safe_float(candidate_runtime)):
            accepted["tra_fast_failure_reason"] = "tra_fast_missing_or_nonfinite_runtime"
        else:
            accepted["tra_fast_failure_reason"] = "tra_fast_runtime_not_20pct_faster_than_tra_gurobi"
    return accepted


def _write_report(path: Path, rows: List[Dict[str, Any]], artifacts: Dict[str, str]) -> None:
    lines = [
        "# 当前 M1-M9 TRA-Gurobi / TRA-Fast 验收报告",
        "",
        "## 数据源",
        "",
        f"- Gurobi baseline JSON: `{artifacts['baseline_json']}`",
        f"- Gurobi baseline CSV: `{artifacts['baseline_csv']}`",
        f"- Runtime alias JSON: `{artifacts['runtime_alias_json']}`",
        f"- Gurobi structure exports JSON: `{artifacts['structure_exports_json']}`",
        "",
        "## 验收口径",
        "",
        "- TRA-Gurobi Cmax 必须等于当前 Gurobi Cmax，runtime 必须至少快 20%。",
        "- TRA-Fast Cmax 必须等于当前 Gurobi Cmax，runtime 必须至少比 TRA-Gurobi 快 20%。",
        "- 任一 TRA 解低于 Gurobi Cmax 均视为约束/实例口径疑点，不能收。",
        "",
        "## 结果表",
        "",
    ]
    headers = [
        "case",
        "gurobi_cmax",
        "gurobi_runtime_sec",
        "tra_gurobi_cmax",
        "tra_gurobi_total_runtime_sec",
        "tra_gurobi_speedup",
        "tra_gurobi_acceptance_ok",
        "tra_fast_cmax",
        "tra_fast_runtime_sec",
        "tra_fast_speedup",
        "tra_fast_acceptance_ok",
        "acceptance_ok",
        "failure_reason",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 论文创新性说明",
            "",
            "参考论文的核心对照是三阶段快速决策方法与集成决策方法。本实验进一步强调同一 Global XYZU 约束口径下的分层求解链：Gurobi 作为集成基线，TRA-Gurobi 作为 exact-aligned repair/refinement，TRA-Fast 作为 surrogate + calibration 的快速层。所有层都通过 Cmax 等值与 lower-than-Gurobi 守门，避免把约束不一致误写成算法优势。",
        ]
    )
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current M1-M9 TRA-Gurobi/TRA-Fast layered acceptance.")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=f"result/m_current_tra_acceptance_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--min-tra-gurobi-speedup", type=float, default=0.20)
    parser.add_argument("--min-tra-fast-speedup", type=float, default=0.20)
    parser.add_argument("--cmax-abs-tol", type=float, default=1e-5)
    parser.add_argument("--stop-on-lower-cmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-first-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formal-target-blind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-fast-calibration-mode", choices=("off", "auto", "always"), default="off")
    parser.add_argument("--tra-gurobi-structure-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-gurobi-structure-required", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-fast-structure-fastpath", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-gurobi-revolving-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tra-gurobi-known-target-guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-gurobi-global-target-probe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-gurobi-global-target-probe-warm-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-gurobi-global-target-probe-time-factor", type=float, default=1.0)
    parser.add_argument("--tra-gurobi-resource-global-decomp-repair", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT_DIR / output_root
    _ensure_dir(output_root)
    artifacts = write_baseline_artifacts(str(output_root))
    baseline_rows = build_gurobi_baseline_rows()
    baseline = _baseline_by_case(baseline_rows)
    runtime_alias_json = str(ROOT_DIR / artifacts["runtime_alias_json"])
    baseline_json = str(ROOT_DIR / artifacts["baseline_json"])
    structure_exports_json = str(ROOT_DIR / artifacts["structure_exports_json"])
    fast_baseline_csv = output_root / "current_m_fast_baseline_after_tra_gurobi.csv"
    summary_rows: List[Dict[str, Any]] = []
    accepted_tra_rows: Dict[str, Dict[str, Any]] = {}
    case_specs = {case.algorithm_case: case for case in load_current_m_cases()}
    master_domain_dir = _ensure_dir(output_root / "master_domains")
    for raw_case in args.cases:
        case_name = algorithm_case(raw_case)
        gurobi_row = baseline[case_name]
        print(f"[case] {case_name}", flush=True)
        preprocess_start = time.perf_counter()
        master_domain_path = build_case_master_domain(
            case_specs[case_name],
            output_dir=master_domain_dir,
            seed=int(args.seed),
        )
        master_domain_preprocess_sec = float(time.perf_counter() - preprocess_start)
        tra_gurobi_row = _run_tra_gurobi_case(
            case=case_name,
            args=args,
            output_root=output_root,
            runtime_alias_json=runtime_alias_json,
            baseline_json=baseline_json,
            structure_exports_json=structure_exports_json,
            gurobi_row=gurobi_row,
            gurobi_runtime=_safe_float(gurobi_row.get("runtime_sec")),
            master_domain_manifest=str(master_domain_path),
        )
        tra_gurobi_accept = _accept_tra_gurobi(args, case_name, gurobi_row, tra_gurobi_row, output_root)
        case_row: Dict[str, Any] = {
            "case": case_name,
            "gurobi_cmax": gurobi_row.get("model_cmax"),
            "gurobi_runtime_sec": gurobi_row.get("runtime_sec"),
            "gurobi_gap": gurobi_row.get("model_gap"),
            "tra_gurobi_cmax": tra_gurobi_row.get("tra_gurobi_cmax"),
            "tra_gurobi_time_to_target_sec": tra_gurobi_row.get("tra_gurobi_time_to_target_sec"),
            "tra_gurobi_total_runtime_sec": tra_gurobi_row.get("tra_gurobi_total_runtime_sec", tra_gurobi_row.get("total_runtime_sec")),
            "master_domain_preprocess_sec": master_domain_preprocess_sec,
                "master_domain_manifest": _portable_relpath(master_domain_path),
            "tra_gurobi_status": tra_gurobi_row.get("status"),
            "tra_gurobi_case_root": tra_gurobi_row.get("case_output_root"),
            "tra_gurobi_command_returncode": tra_gurobi_row.get("command_returncode"),
            "tra_gurobi_command_runtime_sec": tra_gurobi_row.get("command_runtime_sec"),
            "tra_gurobi_command_timeout": tra_gurobi_row.get("command_timeout"),
            "tra_gurobi_stdout_tail": tra_gurobi_row.get("stdout_tail"),
            "tra_gurobi_stderr_tail": tra_gurobi_row.get("stderr_tail"),
        }
        case_row.update(tra_gurobi_accept)
        if not bool(tra_gurobi_accept["tra_gurobi_acceptance_ok"]):
            case_row["acceptance_ok"] = False
            case_row["failure_reason"] = tra_gurobi_accept.get("tra_gurobi_failure_reason", "tra_gurobi_failed")
            summary_rows.append(case_row)
            _write_csv(output_root / "m_current_tra_acceptance_summary.csv", summary_rows)
            _write_json(output_root / "m_current_tra_acceptance_summary.json", summary_rows)
            if bool(args.stop_on_lower_cmax) and bool(tra_gurobi_accept.get("tra_gurobi_lower_than_gurobi", False)):
                print(f"stop: {case_name} failed TRA-Gurobi acceptance: {case_row['failure_reason']}", flush=True)
                break
            continue
        accepted_tra_rows[case_name] = tra_gurobi_row
        _write_fast_baseline_csv(fast_baseline_csv, baseline_rows, accepted_tra_rows)
        tra_fast_row = _run_tra_fast_case(
            case=case_name,
            args=args,
            output_root=output_root,
            runtime_alias_json=runtime_alias_json,
            structure_exports_json=structure_exports_json,
            fast_baseline_csv=str(fast_baseline_csv),
            tra_gurobi_runtime=_safe_float(case_row["tra_gurobi_time_to_target_sec"]),
            target_cmax=_safe_float(gurobi_row.get("model_cmax")),
            master_domain_manifest=str(master_domain_path),
        )
        tra_fast_accept = _accept_tra_fast(args, case_name, gurobi_row, tra_gurobi_row, tra_fast_row, output_root)
        case_row.update(
            {
                "tra_fast_cmax": tra_fast_row.get("tra_fast_cmax"),
                "tra_fast_runtime_sec": tra_fast_row.get("tra_fast_runtime_sec"),
                "tra_fast_time_to_target_sec": tra_fast_row.get("tra_fast_time_to_target_sec"),
                "tra_fast_status": tra_fast_row.get("status"),
                "tra_fast_case_root": tra_fast_row.get("case_output_root"),
                "tra_fast_command_returncode": tra_fast_row.get("command_returncode"),
                "tra_fast_command_runtime_sec": tra_fast_row.get("command_runtime_sec"),
                "tra_fast_command_timeout": tra_fast_row.get("command_timeout"),
                "tra_fast_stdout_tail": tra_fast_row.get("stdout_tail"),
                "tra_fast_stderr_tail": tra_fast_row.get("stderr_tail"),
            }
        )
        case_row.update(tra_fast_accept)
        case_row["acceptance_ok"] = bool(tra_gurobi_accept["tra_gurobi_acceptance_ok"] and tra_fast_accept["tra_fast_acceptance_ok"])
        case_row["failure_reason"] = "" if bool(case_row["acceptance_ok"]) else (tra_fast_accept.get("tra_fast_failure_reason") or "tra_fast_failed")
        summary_rows.append(case_row)
        _write_csv(output_root / "m_current_tra_acceptance_summary.csv", summary_rows)
        _write_json(output_root / "m_current_tra_acceptance_summary.json", summary_rows)
        print(
            f"  acceptance_ok={case_row['acceptance_ok']} "
            f"tra_gurobi_cmax={case_row.get('tra_gurobi_cmax')} "
            f"tra_fast_cmax={case_row.get('tra_fast_cmax')}",
            flush=True,
        )
        if bool(args.stop_on_first_fail) and not bool(case_row["acceptance_ok"]):
            break
    report_path = ROOT_DIR / "docs" / f"m_current_tra_acceptance_{datetime.now().strftime('%Y%m%d')}.md"
    _write_report(report_path, summary_rows, artifacts)
    print(f"summary={output_root / 'm_current_tra_acceptance_summary.csv'}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
