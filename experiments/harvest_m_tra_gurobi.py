from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from Gurobi.tra_events import read_event_rows
from experiments.m_tra_policy import runtime_budget_for_case


def harvest(
    *,
    event_path: Path,
    case_id: str,
    target_cmax: float,
    run_id: str = "",
    tolerance: float = 1e-5,
    baseline_runtime_sec: float | None = None,
    runtime_limit_sec: float | None = None,
) -> dict:
    matching_times = []
    verified_values = []
    for row in read_event_rows(event_path):
        if str(row.get("case", "")) != case_id:
            continue
        if run_id and str(row.get("run_id", "")) != run_id:
            continue
        if not bool(row.get("internal_feasible", False)):
            continue
        value = float(row.get("verified_cmax", float("nan")))
        timestamp = float(
            row.get("solver_incumbent_timestamp_sec", float("nan"))
        )
        if math.isfinite(value):
            verified_values.append(value)
        if math.isfinite(value) and math.isfinite(timestamp) and abs(value - float(target_cmax)) <= float(tolerance):
            matching_times.append(timestamp)
    first_time = min(matching_times) if matching_times else float("nan")
    best_cmax = min(verified_values) if verified_values else float("nan")
    budget = runtime_budget_for_case(case_id)
    baseline_runtime = (
        float(budget.baseline_runtime_sec)
        if baseline_runtime_sec is None
        else float(baseline_runtime_sec)
    )
    runtime_limit = (
        float(budget.hard_limit_sec)
        if runtime_limit_sec is None
        else float(runtime_limit_sec)
    )
    return {
        "case": case_id,
        "run_id": run_id,
        "target_cmax": float(target_cmax),
        "best_verified_cmax": best_cmax,
        "cmax_equal": math.isfinite(best_cmax) and abs(best_cmax - float(target_cmax)) <= float(tolerance),
        "lower_than_target": math.isfinite(best_cmax) and best_cmax < float(target_cmax) - float(tolerance),
        "first_verified_target_time_sec": first_time,
        "baseline_runtime_sec": baseline_runtime,
        "runtime_limit_sec": runtime_limit,
        "runtime_ok": math.isfinite(first_time) and first_time <= runtime_limit + 1e-9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-run target-aware TRA-Gurobi acceptance harvest.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--case", required=True, choices=[f"M{index}" for index in range(1, 10)])
    parser.add_argument("--target-cmax", required=True, type=float)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = harvest(
        event_path=Path(args.events),
        case_id=str(args.case),
        target_cmax=float(args.target_cmax),
        run_id=str(args.run_id),
        tolerance=float(args.tolerance),
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
