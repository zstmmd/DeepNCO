import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m8_cal", ROOT / "experiments/calibrate_m8_sku_distribution.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_all_candidates_preserve_frozen_fields():
    document = MODULE.load_json(MODULE.DEFAULT_BASELINE)
    base = document["configs"]["M8"]
    for candidate in MODULE.CANDIDATES.values():
        built = MODULE.materialize(base, candidate)
        for field in MODULE.FROZEN_FIELDS:
            assert built[field] == base[field]


def test_rejects_frozen_field_change():
    document = MODULE.load_json(MODULE.DEFAULT_BASELINE)
    base = document["configs"]["M8"]
    changed = dict(base)
    changed["exact_order_sku_counts"] = [21] * 8
    with pytest.raises(ValueError, match="frozen"):
        MODULE.assert_candidate(base, changed)


def test_acceptance_uses_gurobi_runtime_and_gap(tmp_path):
    document = MODULE.load_json(MODULE.DEFAULT_BASELINE)
    config = document["configs"]["M8"]
    summary = {
        "status": "TIME_VERIFY_MISMATCH", "objective": 1600.0, "gap": 0.009,
        "gurobi_runtime_sec": 1800.0, "subtask_count": 1, "task_count": 1,
        "diagnostics": {"model_best_bound": 1585.6, "model_var_count_total": 10, "model_constr_count_total": 20},
        "tasks": [{"stack_id": 7}],
    }
    (tmp_path / "gurobi_summary.json").write_text(__import__("json").dumps(summary), encoding="utf-8")
    row = MODULE.summary_row("candidate", "formal", config, tmp_path)
    assert row["accepted"] is True
