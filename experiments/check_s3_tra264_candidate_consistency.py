import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.run_fixgurobi_replay import build_fixed_payload, parse_gurobi_export
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from problemDto.createInstance import CreateOFSProblem


CASE = "GUROBI-S3"
SEED = 42
TRA_EXPORT_DIR = os.path.join(ROOT_DIR, "result", "replay_check_s3_tra264_source", "gurobi_solution_export")
GUROBI_BASELINE_DIR = os.path.join(
    ROOT_DIR,
    "result",
    "gurobi_s1_s3_rebalanced_original_no_warm_no_prune_1200s_gap001",
)
GUROBI_EXPORT_DIR = os.path.join(GUROBI_BASELINE_DIR, CASE, "gurobi_solution_export")
OUT_DIR = os.path.join(ROOT_DIR, "result", "s3_tra264_candidate_consistency_check")


def _ints(values: Iterable[Any]) -> List[int]:
    return sorted({int(v) for v in values})


def _descriptor_totes(descriptor: Dict[str, Any]) -> Set[int]:
    tote_ids: Set[int] = set()
    for key in ("target_tote_ids", "hit_tote_ids", "noise_tote_ids"):
        tote_ids.update(int(v) for v in descriptor.get(key, []) or [])
    return tote_ids


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_summary_row(path: str, case_name: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row_case = str(row.get("case", row.get("scale", row.get("算例名称", "")))).upper()
            if row_case == case_name.upper():
                return dict(row)
    return {}


def _first_float(row: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in row:
            continue
        try:
            return float(row.get(key))
        except Exception:
            continue
    return float("nan")


def _collect_gurobi_export_cmax_candidates(export_dir: str) -> Dict[str, Any]:
    parsed = parse_gurobi_export(export_dir)
    payload = build_fixed_payload(parsed)
    objective = parsed.get("objectives", {}) or {}
    header = parsed.get("header", {}) or {}
    return {
        "objectives": objective,
        "header": header,
        "fixed_used_stack_ids_by_order": payload.get("fixed_used_stack_ids_by_order", {}),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    parsed = parse_gurobi_export(TRA_EXPORT_DIR)
    payload = build_fixed_payload(parsed)
    used_stack_ids_by_order = {
        int(order_id): _ints(stack_ids)
        for order_id, stack_ids in (payload.get("fixed_used_stack_ids_by_order", {}) or {}).items()
    }
    z_descriptors_by_order_slot = payload.get("fixed_z_descriptors_by_order_slot", {}) or {}

    problem = CreateOFSProblem.generate_problem_by_scale(CASE, seed=SEED)
    cfg = GlobalXYZUConfig(
        time_limit_sec=1200.0,
        mip_gap=0.01,
        candidate_stack_topk=999,
        candidate_station_topk_per_stack=999,
        max_candidate_stacks_per_order=0,
        enable_warm_candidate_stack_prune=False,
        enable_warm_start=False,
        warm_start_use_sp4=False,
        fixgurobi_no_warm_start=True,
        fixgurobi_allow_warm_start_fallback=False,
        integrate_u_route=True,
        route_arc_prune=False,
        enable_route_time_window_arc_prune=False,
        enable_scale_adaptive_candidate_prune=False,
        route_pickup_neighbor_limit=0,
        gurobi_output=False,
    )
    compiled = GlobalXYZUSolver().compile_model(problem, cfg)
    prepared = compiled.prepared

    candidate_stacks_by_order = {
        int(order_id): _ints(stack_ids)
        for order_id, stack_ids in (prepared.get("candidate_stacks_by_order", {}) or {}).items()
    }
    support_totes_by_order = {
        int(order_id): _ints(tote_ids)
        for order_id, tote_ids in (prepared.get("support_totes_by_order", {}) or {}).items()
    }
    demand_hit_totes_by_order = {
        int(order_id): _ints(tote_ids)
        for order_id, tote_ids in (prepared.get("demand_hit_totes_by_order", {}) or {}).items()
    }
    tote_to_stack = {int(k): int(v) for k, v in (prepared.get("tote_to_stack", {}) or {}).items()}

    order_checks: List[Dict[str, Any]] = []
    descriptor_checks: List[Dict[str, Any]] = []
    missing_stack_count = 0
    missing_tote_count = 0

    for order_id, used_stacks in sorted(used_stack_ids_by_order.items()):
        candidate_stacks = set(candidate_stacks_by_order.get(order_id, []))
        missing_stacks = [sid for sid in used_stacks if sid not in candidate_stacks]
        missing_stack_count += len(missing_stacks)
        order_checks.append(
            {
                "order_id": order_id,
                "used_stack_ids": used_stacks,
                "candidate_stack_count": len(candidate_stacks),
                "candidate_stack_min": min(candidate_stacks) if candidate_stacks else None,
                "candidate_stack_max": max(candidate_stacks) if candidate_stacks else None,
                "missing_used_stack_ids": missing_stacks,
                "all_used_stacks_in_candidates": not missing_stacks,
            }
        )

    for order_id_raw, slot_rows in sorted((z_descriptors_by_order_slot or {}).items(), key=lambda kv: int(kv[0])):
        order_id = int(order_id_raw)
        support_totes = set(support_totes_by_order.get(order_id, []))
        demand_hit_totes = set(demand_hit_totes_by_order.get(order_id, []))
        for local_slot_index, descriptors in enumerate(slot_rows or []):
            for descriptor_index, descriptor in enumerate(descriptors or []):
                stack_id = int(descriptor.get("stack_id", -1))
                tote_ids = _ints(_descriptor_totes(descriptor))
                missing_totes = [tid for tid in tote_ids if tid not in support_totes]
                hit_missing = [tid for tid in _ints(descriptor.get("hit_tote_ids", []) or []) if tid not in demand_hit_totes]
                stack_mismatch = [
                    tid for tid in tote_ids if int(tote_to_stack.get(tid, -1)) != stack_id
                ]
                missing_tote_count += len(missing_totes)
                descriptor_checks.append(
                    {
                        "order_id": order_id,
                        "local_slot_index": local_slot_index,
                        "descriptor_index": descriptor_index,
                        "stack_id": stack_id,
                        "mode": str(descriptor.get("mode", "")),
                        "tote_ids": tote_ids,
                        "missing_from_support_totes": missing_totes,
                        "hit_missing_from_demand_hit_totes": hit_missing,
                        "tote_stack_mismatch": stack_mismatch,
                        "all_totes_in_support": not missing_totes,
                    }
                )

    baseline_summary = _read_summary_row(os.path.join(GUROBI_BASELINE_DIR, "summary.csv"), CASE)
    baseline_objectives = _read_json(os.path.join(GUROBI_EXPORT_DIR, "best_solution_objectives.json"))
    baseline_audit = _read_json(os.path.join(GUROBI_EXPORT_DIR, "best_solution_audit.json"))
    baseline_verification = _read_json(os.path.join(GUROBI_EXPORT_DIR, "tra_makespan_verification.json"))
    baseline_export = _collect_gurobi_export_cmax_candidates(GUROBI_EXPORT_DIR)

    tra_objectives = parsed.get("objectives", {}) or {}
    tra_model_cmax = float(tra_objectives.get("model_cmax", math.nan))
    tra_objective = float(tra_objectives.get("model_objective", math.nan))
    gurobi_model_cmax = float(baseline_objectives.get("model_cmax", math.nan))
    gurobi_objective = float(baseline_objectives.get("model_objective", math.nan))
    gurobi_bound = _first_float(
        baseline_summary,
        "model_best_bound",
        "下界",
    )
    if not math.isfinite(gurobi_bound):
        gurobi_bound = _first_float(baseline_objectives, "model_best_bound")
    gurobi_gap = _first_float(
        baseline_summary,
        "model_gap",
        "gap",
    )
    if not math.isfinite(gurobi_gap):
        gurobi_gap = _first_float(baseline_objectives, "model_gap")

    conclusion = {
        "case": CASE,
        "seed": SEED,
        "tra_export_dir": TRA_EXPORT_DIR,
        "gurobi_baseline_dir": GUROBI_BASELINE_DIR,
        "checked_original_prepare_config": {
            "disable_warm_start": True,
            "candidate_stack_topk": 999,
            "candidate_station_topk_per_stack": 999,
            "max_candidate_stacks_per_order": 0,
            "route_arc_prune": False,
            "enable_route_time_window_arc_prune": False,
            "enable_scale_adaptive_candidate_prune": False,
            "mip_gap": 0.01,
        },
        "missing_stack_count": missing_stack_count,
        "missing_tote_count": missing_tote_count,
        "all_used_stacks_in_original_candidates": missing_stack_count == 0,
        "all_descriptor_totes_in_original_support_totes": missing_tote_count == 0,
        "tra_model_cmax": tra_model_cmax,
        "tra_model_objective": tra_objective,
        "gurobi_model_cmax": gurobi_model_cmax,
        "gurobi_model_objective": gurobi_objective,
        "gurobi_model_best_bound": gurobi_bound,
        "gurobi_model_gap": gurobi_gap,
        "gurobi_gap_allows_tra264_objective": (
            math.isfinite(gurobi_bound)
            and math.isfinite(tra_objective)
            and tra_objective >= gurobi_bound - 1e-6
        ),
        "gurobi_export_used_stack_ids_by_order": baseline_export.get("fixed_used_stack_ids_by_order", {}),
        "gurobi_baseline_audit_summary": baseline_audit,
        "gurobi_baseline_verification": baseline_verification,
    }

    output = {
        "conclusion": conclusion,
        "order_candidate_stack_checks": order_checks,
        "descriptor_tote_checks": descriptor_checks,
        "candidate_stacks_by_order": candidate_stacks_by_order,
        "support_totes_by_order": support_totes_by_order,
        "demand_hit_totes_by_order": demand_hit_totes_by_order,
    }
    with open(os.path.join(OUT_DIR, "check.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(conclusion, indent=2, ensure_ascii=False))
    print("ORDER_CHECKS")
    for row in order_checks:
        print(json.dumps(row, ensure_ascii=False))
    print("DESCRIPTOR_CHECK_FAILURES")
    for row in descriptor_checks:
        if row["missing_from_support_totes"] or row["hit_missing_from_demand_hit_totes"] or row["tote_stack_mismatch"]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
