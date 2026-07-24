from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.m_current_tra_baselines import load_current_m_cases
from experiments.m_tra_policy import assert_target_blind_payload, sanitize_case_policy


def build_payload() -> dict:
    cases = {}
    for case in load_current_m_cases():
        with case.gurobi_summary_abs_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        policy = sanitize_case_policy(case.case_id, summary)
        cases[case.case_id] = policy.as_payload()
    payload = {
        "schema_version": 1,
        "description": "Target-blind M1-M9 model policies sanitized from archived effective diagnostics.",
        "cases": cases,
    }
    assert_target_blind_payload(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-blind M-suite TRA domain policies.")
    parser.add_argument(
        "--output",
        default="experiments/configs/m_tra_domain_policies_v1.json",
    )
    args = parser.parse_args()
    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(build_payload(), stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
