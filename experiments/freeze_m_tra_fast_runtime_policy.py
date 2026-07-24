from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping


def build_policy(harvest_paths: Mapping[str, Path]) -> dict:
    cases: dict[str, dict[str, float]] = {}
    for case_id in (f"M{index}" for index in range(1, 10)):
        path = harvest_paths.get(case_id)
        if path is None:
            raise ValueError(f"missing TRA-Gurobi harvest for {case_id}")
        with path.open("r", encoding="utf-8") as stream:
            row = json.load(stream)
        hit_time = float(row.get("first_verified_target_time_sec", float("nan")))
        if (
            not bool(row.get("cmax_equal", False))
            or not bool(row.get("runtime_ok", False))
            or not math.isfinite(hit_time)
            or hit_time <= 0.0
        ):
            raise ValueError(f"TRA-Gurobi harvest is not accepted for {case_id}")
        cases[case_id] = {
            "source_tra_gurobi_runtime_sec": hit_time,
            "hard_limit_sec": 0.8 * hit_time,
        }
    return {
        "schema_version": 1,
        "description": "Frozen target-blind TRA-Fast runtime budgets from accepted TRA-Gurobi callback hit times.",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze target-free TRA-Fast budgets from accepted TRA-Gurobi harvests."
    )
    parser.add_argument(
        "--harvest",
        action="append",
        required=True,
        help="CASE=path, repeated for M1 through M9",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    for item in args.harvest:
        case_id, separator, raw_path = str(item).partition("=")
        if not separator:
            raise ValueError(f"invalid --harvest value: {item}")
        paths[case_id.upper()] = Path(raw_path)
    payload = build_policy(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
