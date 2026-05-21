from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.tra import TRAOptimizer, TRARunConfig


DEFAULT_CASES = [
    "GUROBI-S1",
    "GUROBI-S2",
    "GUROBI-S3",
    "GUROBI-S4",
    "GUROBI-S5",
    "GUROBI-S6",
    "GUROBI-S7",
    "GUROBI-S8",
    "GUROBI-S9",
]

TARGET_CMAX = {
    "GUROBI-S1": 90.0,
    "GUROBI-S2": 164.0,
    "GUROBI-S3": 222.0,
    "GUROBI-S4": 237.0,
    "GUROBI-S5": 275.0,
    "GUROBI-S6": 299.0,
    "GUROBI-S7": 361.0,
    "GUROBI-S8": 366.0,
    "GUROBI-S9": 438.0,
}

CURRENT_TRA_BASELINE_CMAX = {
    "GUROBI-S1": 94.0,
    "GUROBI-S2": 165.0,
    "GUROBI-S3": 256.0,
    "GUROBI-S4": 266.0,
    "GUROBI-S5": 329.0,
    "GUROBI-S6": 333.0,
    "GUROBI-S7": 362.0,
    "GUROBI-S8": 417.0,
    "GUROBI-S9": 439.0,
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_cfg(args: argparse.Namespace, case_name: str, log_dir: str) -> TRARunConfig:
    cfg = TRARunConfig(
        scale=str(case_name).upper(),
        seed=int(args.seed),
        max_iters=int(args.max_iters),
        no_improve_limit=int(args.no_improve_limit),
        log_dir=str(log_dir),
        export_best_solution=False,
        write_iteration_logs=True,
        search_scheme="resource_time_alns",
    )
    cfg.resource_eval_backend = "fixgurobi_prefix"
    cfg.resource_fixgurobi_skip_ortools_validation = True
    cfg.fixgurobi_time_limit_sec = float(args.fixgurobi_time_limit_sec)
    cfg.fixgurobi_mip_gap = float(args.fixgurobi_mip_gap)
    cfg.fixgurobi_candidate_trial_limit = int(args.fixgurobi_candidate_trial_limit)
    cfg.fixgurobi_cache_size = int(args.fixgurobi_cache_size)
    cfg.fixgurobi_output = bool(args.fixgurobi_output)
    cfg.fixgurobi_fix_used_stack_ids = bool(args.fixgurobi_fix_used_stack_ids)
    cfg.resource_target_cmax = float(TARGET_CMAX.get(str(case_name).upper(), float("nan")))
    cfg.enable_warm_start = False
    cfg.warm_start_use_sp4 = False
    cfg.sp4_use_mip = False
    cfg.exact_sp4_use_mip = False
    cfg.sp4_lkh_time_limit_seconds = 0
    cfg.exact_sp4_lkh_time_limit_seconds = 0
    cfg.resource_assert_sp4_ortools_only = False
    cfg.resource_candidate_pool_size = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_exact_candidate_trial_limit = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_xyz_candidate_pool_size = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_xyz_exact_candidate_trial_limit = max(1, int(args.fixgurobi_candidate_trial_limit))
    cfg.resource_candidate_pool_max_attempts = max(1, int(args.candidate_pool_max_attempts))
    cfg.resource_stop_if_validated_best_no_change_rounds = int(args.stop_if_no_change_rounds)
    cfg.resource_stop_if_best_z_no_change_rounds = int(args.stop_if_no_change_rounds)
    cfg.resource_operator_profile = str(args.operator_profile)
    cfg.resource_enable_best_y_assignment_polish = False
    cfg.resource_enable_best_z_sortify_polish = False
    cfg.resource_enable_best_sortify_polish = False
    cfg.resource_enable_best_rank_sortify_polish = False
    return cfg


def _fix_rows(iter_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in iter_rows if str(row.get("eval_backend", "")) == "fixgurobi_prefix"]


def _solve_time_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    times = [
        _safe_float(row.get("fixgurobi_solve_time", float("nan")))
        for row in rows
    ]
    times = [value for value in times if math.isfinite(value)]
    if not times:
        return {
            "fixgurobi_eval_count": 0,
            "fixgurobi_total_solve_time": 0.0,
            "fixgurobi_avg_solve_time": float("nan"),
            "fixgurobi_max_solve_time": float("nan"),
        }
    return {
        "fixgurobi_eval_count": int(len(times)),
        "fixgurobi_total_solve_time": float(sum(times)),
        "fixgurobi_avg_solve_time": float(sum(times) / len(times)),
        "fixgurobi_max_solve_time": float(max(times)),
    }


def _best_fix_row(iter_rows: List[Dict[str, Any]], best_value: float) -> Dict[str, Any]:
    best_row: Dict[str, Any] = {}
    for row in iter_rows:
        value = _safe_float(row.get("best_z", float("nan")))
        if math.isfinite(value) and math.isfinite(best_value) and abs(value - best_value) <= 1e-9:
            best_row = dict(row)
            break
    return best_row


def run_case(args: argparse.Namespace, case_name: str, batch_root: str) -> Dict[str, Any]:
    case_name = str(case_name).upper()
    case_root = _ensure_dir(os.path.join(batch_root, case_name))
    t0 = time.perf_counter()
    status = "ok"
    error_text = ""
    best_value = float("nan")
    iter_rows: List[Dict[str, Any]] = []
    run_stats: Dict[str, Any] = {}
    best_row_payload: Dict[str, Any] = {}
    try:
        cfg = _build_cfg(args, case_name, case_root)
        opt = TRAOptimizer(cfg)
        opt.initialize()
        best_value = float(opt.run())
        iter_rows = list(getattr(opt, "iter_log", []) or [])
        run_stats = dict(opt._runtime_stats_payload() or {})
        best_row_payload = {
            "z": float(getattr(getattr(opt, "best", None), "z", best_value)),
            "iter_id": int(getattr(getattr(opt, "best", None), "iter_id", -1)),
        }
    except Exception as exc:
        status = f"error:{exc.__class__.__name__}"
        error_text = str(exc)
    runtime_sec = float(time.perf_counter() - t0)
    fix_rows = _fix_rows(iter_rows)
    solve_stats = _solve_time_stats(fix_rows)
    target = float(TARGET_CMAX.get(case_name, float("nan")))
    baseline = float(CURRENT_TRA_BASELINE_CMAX.get(case_name, float("nan")))
    best_iter = _safe_int(best_row_payload.get("iter_id", -1), -1)
    if not math.isfinite(best_value):
        best_value = _safe_float(best_row_payload.get("z", float("nan")))
    if status == "ok" and not math.isfinite(best_value):
        status = "no_feasible"
    best_iter_row = _best_fix_row(iter_rows, best_value)
    last_fix_row = dict(fix_rows[-1]) if fix_rows else {}
    row = {
        "case": case_name,
        "status": status,
        "error_text": error_text,
        "target_cmax": target,
        "current_tra_baseline_cmax": baseline,
        "tra_gurobi_cmax": best_value,
        "gap_vs_global": float(best_value - target) if math.isfinite(best_value) and math.isfinite(target) else float("nan"),
        "gap_vs_current_tra": float(best_value - baseline) if math.isfinite(best_value) and math.isfinite(baseline) else float("nan"),
        "total_runtime_sec": runtime_sec,
        **solve_stats,
        "best_iter": int(best_iter),
        "best_iter_fixgurobi_solve_time": _safe_float(best_iter_row.get("fixgurobi_solve_time", float("nan"))),
        "best_gurobi_status": str(best_iter_row.get("fixgurobi_status", last_fix_row.get("fixgurobi_status", ""))),
        "best_gurobi_gap": _safe_float(best_iter_row.get("fixgurobi_gap", last_fix_row.get("fixgurobi_gap", float("nan")))),
        "best_gurobi_bound": _safe_float(best_iter_row.get("fixgurobi_bound", last_fix_row.get("fixgurobi_bound", float("nan")))),
        "best_fixed_scope": str(best_iter_row.get("fixgurobi_fixed_scope", last_fix_row.get("fixgurobi_fixed_scope", ""))),
        "reached_target": bool(math.isfinite(best_value) and math.isfinite(target) and best_value <= target + 1e-9),
        "not_worse_than_current_tra": bool(math.isfinite(best_value) and math.isfinite(baseline) and best_value <= baseline + 1e-9),
        "stop_reason": str(run_stats.get("stop_reason", "")),
        "iter_count": int(len(iter_rows)),
        "max_iters": int(args.max_iters),
        "fixgurobi_time_limit_sec": float(args.fixgurobi_time_limit_sec),
        "fixgurobi_mip_gap": float(args.fixgurobi_mip_gap),
        "fixgurobi_candidate_trial_limit": int(args.fixgurobi_candidate_trial_limit),
    }
    _write_json(os.path.join(case_root, "tra_gurobi_case_summary.json"), row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRA operators with FixGurobi-only candidate evaluation.")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--no-improve-limit", type=int, default=3)
    parser.add_argument("--fixgurobi-time-limit-sec", type=float, default=300.0)
    parser.add_argument("--fixgurobi-mip-gap", type=float, default=0.01)
    parser.add_argument("--fixgurobi-candidate-trial-limit", type=int, default=1)
    parser.add_argument("--fixgurobi-cache-size", type=int, default=128)
    parser.add_argument("--fixgurobi-fix-used-stack-ids", action="store_true", default=False)
    parser.add_argument("--fixgurobi-output", action="store_true", default=False)
    parser.add_argument("--candidate-pool-max-attempts", type=int, default=24)
    parser.add_argument("--stop-if-no-change-rounds", type=int, default=40)
    parser.add_argument("--operator-profile", type=str, default="baseline_safe")
    parser.add_argument("--output-root", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = str(args.output_root or os.path.join(ROOT_DIR, "result", f"tra_gurobi_{timestamp}"))
    batch_root = _ensure_dir(batch_root)
    rows: List[Dict[str, Any]] = []
    cases = [str(case).upper() for case in (args.cases or DEFAULT_CASES)]
    for idx, case_name in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] case={case_name} seed={int(args.seed)}")
        row = run_case(args, case_name, batch_root)
        rows.append(row)
        print(
            f"  status={row['status']} cmax={row['tra_gurobi_cmax']} "
            f"target={row['target_cmax']} runtime={row['total_runtime_sec']:.3f}s "
            f"fix_time={row['fixgurobi_total_solve_time']:.3f}s"
        )
        _write_csv(os.path.join(batch_root, "tra_gurobi_s1_s9_summary.csv"), rows)
    _write_csv(os.path.join(batch_root, "tra_gurobi_s1_s9_summary.csv"), rows)
    _write_json(
        os.path.join(batch_root, "tra_gurobi_run_config.json"),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cases": cases,
            "seed": int(args.seed),
            "max_iters": int(args.max_iters),
            "fixgurobi_time_limit_sec": float(args.fixgurobi_time_limit_sec),
            "fixgurobi_mip_gap": float(args.fixgurobi_mip_gap),
            "fixgurobi_candidate_trial_limit": int(args.fixgurobi_candidate_trial_limit),
            "output_root": batch_root,
        },
    )
    with open(os.path.join(batch_root, "tra_gurobi_s1_s9_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"batch_root={batch_root}\n")
        f.write(f"cases={cases}\n")
        f.write(f"seed={int(args.seed)}\n")
        f.write(f"max_iters={int(args.max_iters)}\n")
        f.write(f"fixgurobi_time_limit_sec={float(args.fixgurobi_time_limit_sec)}\n\n")
        for row in rows:
            f.write(
                f"case={row['case']}, status={row['status']}, tra_gurobi_cmax={row['tra_gurobi_cmax']}, "
                f"target_cmax={row['target_cmax']}, gap_vs_global={row['gap_vs_global']}, "
                f"runtime={row['total_runtime_sec']:.3f}, fixgurobi_total_solve_time={row['fixgurobi_total_solve_time']:.3f}, "
                f"fixgurobi_eval_count={row['fixgurobi_eval_count']}, best_iter={row['best_iter']}\n"
            )
    print(f"summary={os.path.join(batch_root, 'tra_gurobi_s1_s9_summary.csv')}")


if __name__ == "__main__":
    main()
