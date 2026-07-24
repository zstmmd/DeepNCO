from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.harvest_m_tra_gurobi import harvest
from experiments.m_tra_fast_policy import load_fast_runtime_budget


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-run target-aware TRA-Fast acceptance harvest."
    )
    parser.add_argument("--events", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=[f"M{index}" for index in range(1, 10)],
    )
    parser.add_argument("--target-cmax", required=True, type=float)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--fast-runtime-policy",
        default="experiments/configs/m_tra_fast_runtime_budgets_v1.json",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    policy_path = Path(args.fast_runtime_policy)
    if not policy_path.is_absolute():
        policy_path = ROOT_DIR / policy_path
    budget = load_fast_runtime_budget(policy_path, str(args.case))
    result = harvest(
        event_path=Path(args.events),
        case_id=str(args.case),
        target_cmax=float(args.target_cmax),
        run_id=str(args.run_id),
        tolerance=float(args.tolerance),
        baseline_runtime_sec=budget.source_tra_gurobi_runtime_sec,
        runtime_limit_sec=budget.hard_limit_sec,
    )
    result["algorithm"] = "paper-tra-fast"
    result["runtime_policy_sha256"] = budget.policy_sha256
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
