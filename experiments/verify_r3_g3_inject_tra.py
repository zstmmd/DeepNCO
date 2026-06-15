from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from entity.calculate import GlobalTimeCalculator
from experiments.run_benchmark import _make_tra_config
from experiments.run_large_scale_trial import large_scale_configs
from Gurobi.sp1 import SP1_BOM_Splitter
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.sp3 import SP3_Bin_Hitter
from Gurobi.sp4 import SP4_Robot_Router
from Gurobi.tra import RankAwareGlobalTimeCalculator, TRAOptimizer
from problemDto.createInstance import CreateOFSProblem


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _build_problem(case: str, seed: int) -> Any:
    case_u = str(case).upper()
    if case_u.startswith("L"):
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(large_scale_configs())
    return CreateOFSProblem.generate_problem_by_scale(case_u, seed=int(seed))


def _solve_r3_g3(case: str, seed: int, algorithm: str, sp4_time_sec: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    problem = _build_problem(case, seed)

    sp1 = SP1_BOM_Splitter(problem)
    subtasks = sp1.solve(use_mip=False)
    if algorithm == "g3":
        subtasks.sort(
            key=lambda st: (
                -len(getattr(st, "unique_sku_list", []) or getattr(st, "sku_list", []) or []),
                int(getattr(st, "id", -1)),
            )
        )
    elif algorithm == "r3":
        subtasks.sort(
            key=lambda st: (
                int(getattr(getattr(st, "parent_order", None), "order_id", -1)),
                int(getattr(st, "id", -1)),
            )
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    problem.subtask_list = list(subtasks)
    problem.subtask_num = len(subtasks)

    sp2 = SP2_Station_Assigner(problem)
    sp2.solve_initial_heuristic()

    sp3 = SP3_Bin_Hitter(problem)
    heuristic = sp3.SP3_Heuristic_Solver(problem)
    physical_tasks, _, _ = heuristic.solve(subtasks, beta_congestion=1.0)
    problem.task_list = list(physical_tasks)
    problem.task_num = len(physical_tasks)

    sp4 = SP4_Robot_Router(problem)
    sp4.sp4_mip_time_limit_seconds = int(max(1, int(sp4_time_sec)))
    sp4._greedy_fallback_route(subtasks, same_subtask_vehicle_mode="conditional")

    layered_cmax = float(GlobalTimeCalculator(problem).calculate())
    rankaware_cmax = float(RankAwareGlobalTimeCalculator(problem).calculate())

    cfg = _make_tra_config(
        str(case).upper(),
        seed=int(seed),
        max_iters=0,
        no_improve_limit=0,
        epsilon=0.05,
        sp2_time_limit_sec=1.0,
        sp4_lkh_time_limit_seconds=int(sp4_time_sec),
    )
    cfg.max_iters = 0
    cfg.export_best_solution = False
    cfg.write_iteration_logs = False
    cfg.compact_tra_summary_json = True
    cfg.log_dir = os.path.join("result", "r3_g3_inject_tra_verify_tmp", str(case).upper(), algorithm)
    opt = TRAOptimizer(cfg)
    opt.problem = problem
    opt.sp1 = sp1
    opt.sp2 = sp2
    opt.sp3 = sp3
    opt.sp4 = sp4
    opt.sim = RankAwareGlobalTimeCalculator(problem)
    opt.best = opt.snapshot(rankaware_cmax, iter_id=0, lightweight=False)
    opt.work = opt.best
    opt.anchor = opt.best
    opt.work_z = rankaware_cmax
    opt.anchor_z = rankaware_cmax

    audit = opt._build_best_solution_audit(rankaware_cmax)
    return {
        "case": str(case).upper(),
        "algorithm": algorithm,
        "orders": int(len(getattr(problem, "order_list", []) or [])),
        "subtasks": int(len(getattr(problem, "subtask_list", []) or [])),
        "tasks": int(len(getattr(problem, "task_list", []) or [])),
        "layered_cmax": layered_cmax,
        "tra_rankaware_cmax": rankaware_cmax,
        "cmax_abs_diff": abs(layered_cmax - rankaware_cmax),
        "coverage_ok": bool(audit.get("coverage_ok", False)),
        "makespan_consistent": bool(audit.get("makespan_consistent", False)),
        "missing_sku_hit": bool(audit.get("missing_sku_hit", False)),
        "unmet_sku_total": int(audit.get("unmet_sku_total", 0) or 0),
        "invalid_station_assignment_count": int(audit.get("invalid_station_assignment_count", 0) or 0),
        "invalid_rank_count": int(audit.get("invalid_rank_count", 0) or 0),
        "invalid_z_task_count": int(audit.get("invalid_z_task_count", 0) or 0),
        "duplicate_tote_use_count": int(audit.get("duplicate_tote_use_count", 0) or 0),
        "unassigned_robot_task_count": int(audit.get("unassigned_robot_task_count", 0) or 0),
        "bom_arrival_window_ok": bool(audit.get("bom_arrival_window_ok", True)),
        "has_unreasonable_solution": bool(audit.get("has_unreasonable_solution", False)),
        "audit_issues_json": json.dumps(list(audit.get("issues", []) or []), ensure_ascii=False),
        "runtime_sec": float(time.perf_counter() - t0),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether R3/G3 solutions can be injected into TRA audit state.")
    parser.add_argument("--cases", nargs="+", default=["L1", "L3", "L9"])
    parser.add_argument("--algorithms", nargs="+", default=["r3", "g3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp4-time-sec", type=int, default=20)
    parser.add_argument("--output", default="result/r3_g3_inject_tra_verify_20260616.csv")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for case in args.cases:
        for algorithm in args.algorithms:
            print(f"[verify] {case} {algorithm}", flush=True)
            try:
                rows.append(_solve_r3_g3(case, int(args.seed), str(algorithm).lower(), int(args.sp4_time_sec)))
            except Exception as exc:
                rows.append({
                    "case": str(case).upper(),
                    "algorithm": str(algorithm).lower(),
                    "status": "error",
                    "error_text": str(exc),
                })
            _write_csv(args.output, rows)
    print(f"summary={args.output}", flush=True)


if __name__ == "__main__":
    main()
