from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from Gurobi.master_domain import (
    MasterDomainError,
    build_master_domain_manifest,
    normalize_master_domain_manifest,
)
from experiments.m_tra_contract import time_to_target_from_iter_rows
from Gurobi.resource_time_alns.hybrid_gate import should_run_hybrid_exact


def _fake_compiled() -> SimpleNamespace:
    warm_task = SimpleNamespace(target_stack_id=12)
    warm_subtask = SimpleNamespace(assigned_station_id=3, execution_tasks=[warm_task])
    warm = SimpleNamespace(subtask_by_order={7: [warm_subtask]})
    route_tasks = {
        0: SimpleNamespace(slot_id=0, stack_id=12, station_id=3),
        1: SimpleNamespace(slot_id=1, stack_id=15, station_id=4),
    }
    problem = SimpleNamespace(scale_name="M1", order_list=[], station_list=[], robot_list=[])
    return SimpleNamespace(
        problem_template=problem,
        warm=warm,
        prepared={
            "slot_ids_by_order": {7: [0, 1]},
            "candidate_stacks_by_order": {7: [15, 12]},
        },
        vars_payload={
            "route_tasks": route_tasks,
            "route_arcs": [(4, 5), (0, 4), (5, 1)],
            "protected_route_arcs": {(4, 5), (0, 4)},
        },
    )


def test_master_domain_manifest_is_canonical_and_self_verifying() -> None:
    first = build_master_domain_manifest(_fake_compiled(), canonical_seed=42)
    second = build_master_domain_manifest(_fake_compiled(), canonical_seed=42)

    assert first == second
    assert first["slot_count_by_order"] == {"7": 2}
    assert first["candidate_stacks_by_order"] == {"7": [12, 15]}
    assert first["route_task_tuples"] == [[0, 12, 3], [1, 15, 4]]
    assert first["protected_route_arcs"] == [[0, 4], [4, 5]]
    assert len(first["manifest_sha256"]) == 64
    assert normalize_master_domain_manifest(first) == first


def test_master_domain_manifest_rejects_mutation() -> None:
    manifest = build_master_domain_manifest(_fake_compiled(), canonical_seed=42)
    mutated = copy.deepcopy(manifest)
    mutated["route_arcs"].append([99, 100])

    with pytest.raises(MasterDomainError, match="hash mismatch"):
        normalize_master_domain_manifest(mutated)


def test_time_to_target_is_retrospective_and_requires_internal_validation() -> None:
    rows = [
        {"iter_runtime_sec": 1.5, "best_z": 900, "incumbent_internal_feasible": True},
        {"iter_runtime_sec": 2.0, "best_z": 805, "incumbent_internal_feasible": False},
        {"iter_runtime_sec": 0.5, "best_z": 805, "validated_makespan": 805},
        {"iter_runtime_sec": 10.0, "best_z": 800, "validated_makespan": 800},
    ]

    assert time_to_target_from_iter_rows(rows, target_cmax=805) == pytest.approx(4.0)
    assert math.isnan(time_to_target_from_iter_rows(rows, target_cmax=804))


def test_time_to_target_does_not_use_unvalidated_best_z_by_default() -> None:
    rows = [{"iter_runtime_sec": 3.0, "best_z": 582}]

    assert math.isnan(time_to_target_from_iter_rows(rows, target_cmax=582))


def test_hybrid_exact_gate_is_periodic_target_blind_and_one_per_iteration() -> None:
    kwargs = {
        "iter_id": 4,
        "last_exact_iter": -1,
        "layer": "U",
        "allowed_layers": ["U", "XYZ"],
        "period": 4,
        "margin_ratio": 0.08,
        "current_best": 805.0,
        "proxy_value": 840.0,
        "revolving_lb": 790.0,
    }
    assert should_run_hybrid_exact(**kwargs)
    assert not should_run_hybrid_exact(**{**kwargs, "layer": "Z"})
    assert not should_run_hybrid_exact(**{**kwargs, "last_exact_iter": 4})
    assert not should_run_hybrid_exact(**{**kwargs, "iter_id": 5})
