from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Gurobi.tra_events import read_event_rows
from experiments.tra_structure_gap import (
    compare_candidate_trajectory,
    compare_certified_trajectory,
    compare_structure,
    parse_gurobi_solution_dump,
    select_best_tra_event,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc structural gap audit for Gurobi and TRA incumbents."
    )
    parser.add_argument("--gurobi-dump", required=True)
    parser.add_argument("--tra-events", required=True)
    parser.add_argument("--search-audit")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference = parse_gurobi_solution_dump(args.gurobi_dump)
    event_rows = list(read_event_rows(args.tra_events))
    event = select_best_tra_event(event_rows)
    report = compare_structure(reference, event)
    report["certified_trajectory"] = compare_certified_trajectory(
        reference,
        event_rows,
    )
    if args.search_audit:
        report["candidate_trajectory"] = compare_candidate_trajectory(
            reference,
            read_event_rows(args.search_audit),
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
