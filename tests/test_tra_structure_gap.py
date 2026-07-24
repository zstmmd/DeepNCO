from __future__ import annotations

from pathlib import Path

from experiments.tra_structure_gap import (
    compare_candidate_trajectory,
    compare_certified_trajectory,
    compare_structure,
    parse_gurobi_solution_dump,
)


def test_structure_gap_compares_all_three_paper_blocks(tmp_path: Path) -> None:
    dump = tmp_path / "best_solution_full_dump.txt"
    dump.write_text(
        "\n".join(
                (
                    "global_makespan=582.000000",
                    "[SP1 Decisions]",
                    "subtask_id=10, order_id=0, sku_units=2, sku_list=[0, 1]",
                    "subtask_id=11, order_id=0, sku_units=1, sku_list=[2]",
                    "[SP3 Decisions]",
                    "task_id=0, subtask_id=10, stack_id=43, station_id=0, mode=SORT",
                    "task_id=1, subtask_id=11, stack_id=44, station_id=1, mode=SORT",
                "[SP4 Decisions]",
                "task_id=0, robot_id=2, trip_id=0",
                "task_id=1, robot_id=1, trip_id=0",
            )
        ),
        encoding="utf-8",
    )
    reference = parse_gurobi_solution_dump(dump)
    event = {
        "verified_cmax": 589.0,
        "solver_incumbent_timestamp_sec": 50.0,
        "structural_hash": "shell-a",
        "structural_projection": {
            "x_group": [
                [["atom", "0:0"], 0],
                [["atom", "0:1"], 1],
                [["atom", "0:2"], 1],
            ],
            "s_visit": [
                [["tuple", 0, 43], 0],
                [["tuple", 1, 44], 0],
            ],
            "r_assign": [
                [["atom", 0], 1],
                [["atom", 1], 1],
            ],
        },
    }

    report = compare_structure(reference, event)

    assert report["blocks"]["F1_S_visit"]["changed_carrier_count"] == 1
    assert report["blocks"]["F2_X_group"]["changed_carrier_count"] == 1
    assert report["blocks"]["F3_R_assign"]["changed_carrier_count"] == 1
    assert report["blocks"]["F3_R_assign"]["raw_one_hot_hamming"] == 2

    event.update(
        {
            "internal_feasible": True,
            "procedure": "F2",
            "neighborhood": "N2",
            "snapshot_sha256": "snapshot-a",
        }
    )
    certified = compare_certified_trajectory(reference, [event])
    assert certified[0]["blocks"]["F2_X_group"]["changed_carrier_count"] == 1
    assert "differences" not in certified[0]["blocks"]["F2_X_group"]

    audit = {
        "elapsed_sec": 12.0,
        "stage": "inner",
        "procedure": "F2",
        "neighborhood": "N2",
        "selected_shell_sha256": "shell-a",
        "candidates": [
            {
                "shell_sha256": "shell-a",
                "structural_projection": event["structural_projection"],
                "relaxed_objective": 10.0,
                "comproc": {
                    "projected_cmax": 20.0,
                    "recourse_score": 18.0,
                    "verified_cmax": 20.0,
                },
            }
        ],
    }
    candidates = compare_candidate_trajectory(reference, [audit])
    assert candidates[0]["selected"]
    assert candidates[0]["recourse_score"] == 18.0
