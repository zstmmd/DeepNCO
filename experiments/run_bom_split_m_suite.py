from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from entity.order import Order
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from Gurobi.tra import TRAOptimizer, TRARunConfig
from problemDto.createInstance import CreateOFSProblem


M_CASES = [f"GUROBI-M{i}" for i in range(1, 10)]

SPLIT_PROFILES: Dict[str, float] = {
    "SPLIT-1": 1.2,
    "SPLIT-2": 1.5,
    "SPLIT-3": 2.0,
}

CURRENT_M_BASELINE: Dict[str, Dict[str, float]] = {
    "GUROBI-M1": {"gurobi_cmax": 489.0, "tra_cmax": 489.0, "tra_gap": 0.0, "gurobi_gap": 0.000221, "gurobi_s": 1115.63, "tra_s": 825.94},
    "GUROBI-M2": {"gurobi_cmax": 546.0, "tra_cmax": 546.0, "tra_gap": 0.0, "gurobi_gap": 0.000307, "gurobi_s": 1664.95, "tra_s": 990.0},
    "GUROBI-M3": {"gurobi_cmax": 558.0, "tra_cmax": 558.0, "tra_gap": 0.0, "gurobi_gap": 0.009555, "gurobi_s": 1992.81, "tra_s": 450.48},
    "GUROBI-M4": {"gurobi_cmax": 630.0, "tra_cmax": 630.0, "tra_gap": 0.0, "gurobi_gap": 0.009641, "gurobi_s": 2087.37, "tra_s": 1603.56},
    "GUROBI-M5": {"gurobi_cmax": 679.0, "tra_cmax": 679.0, "tra_gap": 0.0, "gurobi_gap": 0.003238, "gurobi_s": 2097.25, "tra_s": 777.77},
    "GUROBI-M6": {"gurobi_cmax": 687.0, "tra_cmax": 687.0, "tra_gap": 0.0, "gurobi_gap": 0.000048, "gurobi_s": 2287.37, "tra_s": 1591.57},
    "GUROBI-M7": {"gurobi_cmax": 708.0, "tra_cmax": 708.0, "tra_gap": 0.0, "gurobi_gap": 0.003626, "gurobi_s": 2481.76, "tra_s": 1348.67},
    "GUROBI-M8": {"gurobi_cmax": 725.0, "tra_cmax": 726.0, "tra_gap": 0.0014, "gurobi_gap": 0.003022, "gurobi_s": 2525.89, "tra_s": 948.43},
    "GUROBI-M9": {"gurobi_cmax": 731.0, "tra_cmax": 731.0, "tra_gap": 0.0, "gurobi_gap": 0.005605, "gurobi_s": 3452.09, "tra_s": 1216.08},
}

BASE_M_CONFIGS: Dict[str, Dict[str, Any]] = {
    "GUROBI-M1": {"map_size": (4, 5), "resources": (5, 4, 172), "data": (6, 42), "bom_complexity": (8, 1), "target_stack_count": 30, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (11, 13), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M2": {"map_size": (4, 5), "resources": (5, 5, 184), "data": (7, 44), "bom_complexity": (8, 1), "target_stack_count": 33, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (13, 15), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 3, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M3": {"map_size": (4, 5), "resources": (5, 5, 197), "data": (7, 49), "bom_complexity": (8, 1), "target_stack_count": 36, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (13, 15), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M4": {"map_size": (4, 5), "resources": (5, 5, 225), "data": (7, 56), "bom_complexity": (8, 1), "target_stack_count": 41, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (15, 17), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 2, 2, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M5": {"map_size": (4, 5), "resources": (5, 5, 237), "data": (8, 58), "bom_complexity": (8, 1), "target_stack_count": 44, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (16, 18), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M6": {"map_size": (4, 5), "resources": (5, 5, 249), "data": (8, 61), "bom_complexity": (8, 1), "target_stack_count": 47, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (17, 18), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 2, 2, 2, 2, 2), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M7": {"map_size": (4, 5), "resources": (5, 5, 253), "data": (8, 62), "bom_complexity": (8, 1), "target_stack_count": 48, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (17, 20), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M8": {"map_size": (4, 5), "resources": (5, 5, 261), "data": (8, 64), "bom_complexity": (8, 1), "target_stack_count": 50, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (18, 20), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
    "GUROBI-M9": {"map_size": (4, 5), "resources": (5, 5, 265), "data": (8, 65), "bom_complexity": (8, 1), "target_stack_count": 51, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (6, 6, 6, 6, 6, 6, 6, 6), "exact_order_sku_quantity_range": (18, 20), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2, 2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _split_case_name(base_case: str, split_name: str) -> str:
    return f"{str(base_case).upper()}-{str(split_name).upper().replace('_', '-')}"


def _base_case_from_split(case_name: str) -> str:
    text = str(case_name).upper()
    for split_name in SPLIT_PROFILES:
        suffix = f"-{split_name}"
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _copy_order_shell(source: Order, order_id: int, sku_ids: List[int], problem: Any) -> Order:
    order = Order(order_id=int(order_id))
    order.order_product_id_list = list(int(v) for v in sku_ids)
    order.order_skus_number = int(len(order.order_product_id_list))
    order.status = str(getattr(source, "status", "pending") or "pending")
    order.parent_order_id = int(getattr(source, "parent_order_id", getattr(source, "order_id", -1)))
    order.split_child_index = int(getattr(source, "split_child_index", 0))
    order.original_order_id = int(getattr(source, "order_id", -1))
    CreateOFSProblem._assign_order_time_window(order, base_seed=int(getattr(problem, "split_seed", 0) or 0))
    id_to_sku = dict(getattr(problem, "id_to_sku", {}) or {})
    unique_ids = sorted({int(v) for v in order.order_product_id_list})
    order.unique_sku_list = [id_to_sku[int(sku_id)] for sku_id in unique_ids if int(sku_id) in id_to_sku]

    parent_point_qty = dict(getattr(source, "point_sku_quantity", {}) or {})
    child_sku_set = set(unique_ids)
    order.point_sku_quantity = {}
    for point_id, sku_qty in parent_point_qty.items():
        filtered = {
            int(sku_id): int(qty)
            for sku_id, qty in dict(sku_qty or {}).items()
            if int(sku_id) in child_sku_set and int(qty) > 0
        }
        if filtered:
            order.point_sku_quantity[int(point_id)] = filtered
    order.sku_storage_points = [
        getattr(stack, "store_point", None)
        for stack in getattr(problem, "stack_list", []) or []
        if getattr(stack, "store_point", None) is not None
        and int(getattr(getattr(stack, "store_point", None), "idx", -1)) in order.point_sku_quantity
    ]
    return order


def _partition_unique_skus(unique_skus: List[int], parts: int) -> List[List[int]]:
    parts = max(1, int(parts))
    unique_skus = list(unique_skus)
    groups: List[List[int]] = [[] for _ in range(parts)]
    for idx, sku_id in enumerate(unique_skus):
        groups[idx % parts].append(int(sku_id))
    return [group for group in groups if group]


def _allocate_split_parts(order_counts: List[int], split_ratio: float) -> List[int]:
    original_order_count = int(len(order_counts))
    target_subtask_count = max(original_order_count, int(round(float(original_order_count) * float(split_ratio))))
    extra_parts = max(0, int(target_subtask_count - original_order_count))
    parts = [1 for _ in order_counts]
    candidates = sorted(range(len(order_counts)), key=lambda idx: (-int(order_counts[idx]), int(idx)))
    cursor = 0
    while extra_parts > 0 and candidates:
        idx = int(candidates[cursor % len(candidates)])
        if parts[idx] < max(1, int(order_counts[idx])):
            parts[idx] += 1
            extra_parts -= 1
        cursor += 1
        if cursor > 10000:
            raise RuntimeError("failed to allocate split parts")
    return parts


def _partition_count(count: int, parts: int) -> List[int]:
    count = max(0, int(count))
    parts = max(1, int(parts))
    base = int(count // parts)
    rem = int(count % parts)
    out = [base + (1 if idx < rem else 0) for idx in range(parts)]
    return [max(1, int(v)) for v in out if int(v) > 0]


def _derived_split_config(base_case: str, split_ratio: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_case = str(base_case).upper()
    if base_case not in BASE_M_CONFIGS:
        raise KeyError(f"missing base M config for {base_case}")
    base_cfg = copy.deepcopy(BASE_M_CONFIGS[base_case])
    order_counts = [int(v) for v in tuple(base_cfg.get("exact_order_sku_counts", ()) or ())]
    if not order_counts:
        raise ValueError(f"{base_case} has no exact_order_sku_counts")
    base_stack_counts = [int(v) for v in tuple(base_cfg.get("bom_colocated_stack_counts", ()) or ())]
    if len(base_stack_counts) != len(order_counts):
        base_stack_counts = [2 for _ in order_counts]
    parts_by_order = _allocate_split_parts(order_counts, split_ratio=float(split_ratio))
    child_counts: List[int] = []
    child_stack_counts: List[int] = []
    split_map: Dict[int, List[int]] = {}
    child_idx = 0
    for order_idx, (count, part_count) in enumerate(zip(order_counts, parts_by_order)):
        count_parts = _partition_count(int(count), int(part_count))
        stack_count = max(2, int(math.ceil(float(base_stack_counts[order_idx]) / max(1, int(part_count)))))
        split_map[int(order_idx)] = []
        for part in count_parts:
            child_counts.append(int(part))
            child_stack_counts.append(int(stack_count))
            split_map[int(order_idx)].append(int(child_idx))
            child_idx += 1
    cfg = copy.deepcopy(base_cfg)
    cfg["data"] = (int(len(child_counts)), int(base_cfg["data"][1]))
    cfg["exact_order_sku_counts"] = tuple(int(v) for v in child_counts)
    cfg["bom_colocated_stack_counts"] = tuple(int(v) for v in child_stack_counts)
    cfg["target_stack_count"] = max(int(base_cfg.get("target_stack_count", 0) or 0), int(sum(child_stack_counts) + 2))
    cfg["split_generation_mode"] = "config_split_v2"
    meta = {
        "target_split_ratio": float(split_ratio),
        "actual_split_ratio": float(len(child_counts) / max(1, len(order_counts))),
        "original_order_count": int(len(order_counts)),
        "split_order_count": int(len(child_counts)),
        "base_exact_order_sku_counts": order_counts,
        "split_exact_order_sku_counts": child_counts,
        "base_bom_colocated_stack_counts": base_stack_counts,
        "split_bom_colocated_stack_counts": child_stack_counts,
        "split_parts_by_order": parts_by_order,
        "split_map": split_map,
        "generation_mode": "config_split_v2",
    }
    return cfg, meta


def apply_bom_split(problem: Any, split_ratio: float, seed: int) -> Any:
    problem = copy.deepcopy(problem)
    setattr(problem, "split_seed", int(seed))
    original_orders = list(getattr(problem, "order_list", []) or [])
    original_order_count = int(len(original_orders))
    target_subtask_count = max(original_order_count, int(round(float(original_order_count) * float(split_ratio))))
    extra_parts = max(0, int(target_subtask_count - original_order_count))
    split_parts_by_order = {int(getattr(order, "order_id", idx)): 1 for idx, order in enumerate(original_orders)}
    candidates = sorted(
        original_orders,
        key=lambda order: (
            -len(set(int(v) for v in (getattr(order, "order_product_id_list", []) or []))),
            int(getattr(order, "order_id", 0)),
        ),
    )
    cursor = 0
    while extra_parts > 0 and candidates:
        order = candidates[cursor % len(candidates)]
        order_id = int(getattr(order, "order_id", cursor))
        unique_count = len(set(int(v) for v in (getattr(order, "order_product_id_list", []) or [])))
        if split_parts_by_order[order_id] < max(1, unique_count):
            split_parts_by_order[order_id] += 1
            extra_parts -= 1
        cursor += 1
        if cursor > 10000:
            raise RuntimeError("failed to allocate split parts")

    new_orders: List[Order] = []
    next_order_id = 0
    split_map: Dict[int, List[int]] = {}
    for source in sorted(original_orders, key=lambda order: int(getattr(order, "order_id", 0))):
        source_id = int(getattr(source, "order_id", 0))
        sku_ids = [int(v) for v in (getattr(source, "order_product_id_list", []) or [])]
        qty_by_sku: Dict[int, int] = {}
        for sku_id in sku_ids:
            qty_by_sku[int(sku_id)] = int(qty_by_sku.get(int(sku_id), 0)) + 1
        unique_skus = sorted(qty_by_sku)
        groups = _partition_unique_skus(unique_skus, split_parts_by_order[source_id])
        child_ids: List[int] = []
        for child_idx, group in enumerate(groups):
            child_skus: List[int] = []
            for sku_id in group:
                child_skus.extend([int(sku_id)] * int(qty_by_sku[int(sku_id)]))
            child = _copy_order_shell(source, next_order_id, child_skus, problem)
            child.parent_order_id = int(source_id)
            child.split_child_index = int(child_idx)
            new_orders.append(child)
            child_ids.append(int(next_order_id))
            next_order_id += 1
        split_map[int(source_id)] = child_ids

    problem.original_order_num = int(original_order_count)
    problem.order_num = int(len(new_orders))
    problem.order_list = new_orders
    problem.id_to_order = {int(order.order_id): order for order in new_orders}
    problem.bom_split_profile = {
        "target_split_ratio": float(split_ratio),
        "actual_split_ratio": float(len(new_orders) / max(1, original_order_count)),
        "original_order_count": int(original_order_count),
        "split_order_count": int(len(new_orders)),
        "split_map": split_map,
    }
    return problem


def _split_definitions(cases: List[str], splits: List[str]) -> Dict[str, Tuple[str, str, float]]:
    out: Dict[str, Tuple[str, str, float]] = {}
    for case in cases:
        for split in splits:
            split_key = str(split).upper().replace("_", "-")
            out[_split_case_name(case, split_key)] = (str(case).upper(), split_key, float(SPLIT_PROFILES[split_key]))
    return out


@contextmanager
def patched_split_generator(definitions: Dict[str, Tuple[str, str, float]], seed_override: int | None = None) -> Iterator[None]:
    original_generate: Callable[..., Any] = CreateOFSProblem.generate_problem_by_scale
    original_runtime_configs = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    split_runtime_configs: Dict[str, Dict[str, Any]] = {}
    split_meta: Dict[str, Dict[str, Any]] = {}
    for scale_key, (base_case, _split_name, ratio) in definitions.items():
        cfg, meta = _derived_split_config(base_case, float(ratio))
        split_runtime_configs[str(scale_key).upper()] = cfg
        split_meta[str(scale_key).upper()] = meta
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(original_runtime_configs)
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS.update(split_runtime_configs)

    def _generate(scale: str = "SMALL", seed: int = 42) -> Any:
        scale_key = str(scale).upper()
        if scale_key not in definitions:
            return original_generate(scale_key, seed=seed)
        use_seed = int(seed_override if seed_override is not None else seed)
        split_problem = original_generate(scale_key, seed=use_seed)
        split_problem.scale_name = scale_key
        split_problem.original_order_num = int(split_meta[scale_key]["original_order_count"])
        split_problem.bom_split_profile = dict(split_meta[scale_key])
        return split_problem

    CreateOFSProblem.generate_problem_by_scale = staticmethod(_generate)
    try:
        yield
    finally:
        CreateOFSProblem.generate_problem_by_scale = staticmethod(original_generate)
        CreateOFSProblem.RUNTIME_SCALE_CONFIGS = original_runtime_configs


def _problem_stats(problem: Any, case_name: str, base_case: str, split_name: str, target_ratio: float) -> Dict[str, Any]:
    orders = list(getattr(problem, "order_list", []) or [])
    unique_counts = [len(set(int(v) for v in (getattr(order, "order_product_id_list", []) or []))) for order in orders]
    total_qty = [len(list(getattr(order, "order_product_id_list", []) or [])) for order in orders]
    split_profile = dict(getattr(problem, "bom_split_profile", {}) or {})
    return {
        "case": case_name,
        "base_case": base_case,
        "split_profile": split_name,
        "target_split_ratio": float(target_ratio),
        "actual_split_ratio": _safe_float(split_profile.get("actual_split_ratio")),
        "original_orders": int(split_profile.get("original_order_count", 0) or 0),
        "split_orders": int(len(orders)),
        "skus": int(getattr(problem, "skus_num", 0) or 0),
        "totes": int(getattr(problem, "tote_num", 0) or 0),
        "robots": int(getattr(problem, "robot_num", 0) or 0),
        "stations": int(getattr(problem, "station_num", 0) or 0),
        "min_unique_sku_per_split_order": int(min(unique_counts)) if unique_counts else 0,
        "max_unique_sku_per_split_order": int(max(unique_counts)) if unique_counts else 0,
        "avg_unique_sku_per_split_order": float(sum(unique_counts) / max(1, len(unique_counts))),
        "total_order_qty": int(sum(total_qty)),
    }


def _run_gurobi(problem: Any, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = GlobalXYZUConfig(
        time_limit_sec=float(args.gurobi_time_limit_sec),
        mip_gap=float(args.gurobi_mip_gap),
        candidate_stack_topk=int(args.gurobi_candidate_stack_topk),
        max_candidate_stacks_per_order=int(args.gurobi_max_candidate_stacks_per_order),
        candidate_station_topk_per_stack=int(args.gurobi_candidate_station_topk_per_stack),
        route_pickup_neighbor_limit=int(args.gurobi_route_pickup_neighbor_limit),
        integrate_u_route=True,
        enable_warm_start=True,
        warm_start_use_sp4=True,
        warm_start_sp4_time_limit_sec=int(args.gurobi_warm_start_sp4_time_limit_sec),
        gurobi_output=bool(args.show_gurobi),
        route_arc_prune=True,
        enable_route_time_window_arc_prune=True,
        enable_route_load_interval_arc_prune=True,
        enable_global_arrival_workload_lb=True,
        enable_route_slot_stack_count_lb=True,
        enable_selected_workload_lbs=True,
    )
    t0 = time.perf_counter()
    result = GlobalXYZUSolver().solve(problem, cfg=cfg)
    runtime = float(time.perf_counter() - t0)
    diag = dict(getattr(result, "diagnostics", {}) or {})
    cmax = _safe_float(diag.get("model_cmax", getattr(result, "objective", float("nan"))))
    return {
        "algorithm": "gurobi",
        "status": str(getattr(result, "status", "")),
        "cmax": cmax,
        "runtime_sec": _safe_float(getattr(result, "runtime_sec", runtime), runtime),
        "gap": _safe_float(diag.get("model_gap", getattr(result, "gap", float("nan")))),
        "lower_bound": _safe_float(diag.get("model_best_bound", float("nan"))),
        "model_objective": _safe_float(diag.get("model_objective", getattr(result, "objective", float("nan")))),
    }


def _fast_profile(case: str, args: argparse.Namespace) -> Dict[str, Any]:
    idx = int("".join(ch for ch in _base_case_from_split(case).split("-M")[-1] if ch.isdigit()) or "1")
    return {
        "max_iters": min(int(args.tra_fast_max_iters), 18 if idx <= 3 else 24 if idx <= 6 else 30),
        "sp4_limit": 3 if idx <= 3 else 5 if idx <= 6 else 8,
        "eval_period": 5 if idx <= 3 else 6,
        "layer_order": "X,Y,YZ,XZ,XYZ,U" if idx <= 3 else "Y,U,YZ,XZ,XYZ",
        "pool": 5 if idx <= 3 else 6,
        "stop_rounds": 5,
    }


def _run_tra_fast(case_name: str, args: argparse.Namespace, case_root: str) -> Dict[str, Any]:
    profile = _fast_profile(case_name, args)
    cfg = TRARunConfig(
        scale=str(case_name).upper(),
        seed=int(args.seed),
        max_iters=int(profile["max_iters"]),
        no_improve_limit=int(args.tra_fast_no_improve_limit),
        epsilon=float(args.tra_fast_epsilon),
        log_dir=case_root,
        export_best_solution=False,
        write_iteration_logs=False,
        compact_tra_summary_json=True,
        search_scheme="resource_time_alns",
        sp2_time_limit_sec=float(args.tra_fast_sp2_time_limit_sec),
        sp3_use_mip=False,
        sp4_use_mip=False,
        exact_sp4_use_mip=False,
        sp4_lkh_time_limit_seconds=int(profile["sp4_limit"]),
        exact_sp4_lkh_time_limit_seconds=int(profile["sp4_limit"]),
    )
    cfg.resource_eval_backend = "surrogate"
    cfg.fixgurobi_final_validation = False
    cfg.fixgurobi_time_limit_sec = 0.0
    cfg.fixgurobi_candidate_trial_limit = 0
    cfg.resource_global_decomp_repair_enabled = False
    cfg.resource_target_cmax = float("nan")
    cfg.resource_revolving_mode = True
    cfg.resource_revolving_enable_u_layer = True
    cfg.revolving_layer_order = str(profile["layer_order"])
    cfg.revolving_inner_time_limit_sec = float(args.tra_fast_revolving_inner_sec)
    cfg.revolving_outer_time_limit_sec = float(args.tra_fast_revolving_outer_sec)
    cfg.revolving_mark_limit = int(profile["stop_rounds"])
    cfg.resource_real_eval_period = int(profile["eval_period"])
    cfg.resource_candidate_pool_size = int(profile["pool"])
    cfg.resource_candidate_pool_max_attempts = int(profile["pool"] * 5)
    cfg.resource_exact_candidate_trial_limit = int(profile["pool"])
    cfg.resource_xyz_candidate_pool_size = max(3, int(profile["pool"]))
    cfg.resource_xyz_exact_candidate_trial_limit = max(3, int(profile["pool"]))
    cfg.resource_stop_if_best_z_no_change_rounds = int(profile["stop_rounds"])
    cfg.resource_stop_if_validated_best_no_change_rounds = int(profile["stop_rounds"])
    cfg.resource_candidate_pool_log = False
    cfg.resource_enable_xyz_operator = True
    cfg.resource_enable_critical_path_xyz = True
    cfg.resource_assert_sp4_ortools_only = True
    cfg.xz_evaluator_mode = "classic_soft"
    t0 = time.perf_counter()
    opt = TRAOptimizer(cfg)
    opt.initialize()
    best_z = float(opt.run())
    runtime = float(time.perf_counter() - t0)
    summary_path = os.path.join(str(opt._ensure_log_dir()), "tra_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    best_payload = dict(summary.get("best", {}) or {})
    best_z = _safe_float(best_payload.get("z", best_z), best_z)
    return {
        "algorithm": "tra_fast",
        "status": "ok",
        "cmax": best_z,
        "runtime_sec": runtime,
        "gap": float("nan"),
        "best_iter": int(best_payload.get("iter_id", -1) or -1),
        "result_root": str(opt._ensure_log_dir()),
    }


def _run_tra_fixgurobi(case_name: str, args: argparse.Namespace, case_root: str, gurobi_row: Dict[str, Any] | None) -> Dict[str, Any]:
    import Gurobi.tra_gurobi as tra_gurobi

    ns = argparse.Namespace(
        cases=[case_name],
        seed=int(args.seed),
        max_iters=int(args.tra_fix_max_iters),
        no_improve_limit=int(args.tra_fix_no_improve_limit),
        epsilon=float(args.tra_fix_epsilon),
        sp2_time_limit_sec=float(args.tra_fix_sp2_time_limit_sec),
        fixgurobi_time_limit_sec=float(args.tra_fix_time_limit_sec),
        fixgurobi_mip_gap=float(args.tra_fix_mip_gap),
        fixgurobi_candidate_trial_limit=int(args.tra_fix_candidate_trial_limit),
        fixgurobi_cache_size=int(args.tra_fix_cache_size),
        fixgurobi_compiled_cache_size=int(args.tra_fix_compiled_cache_size),
        fixgurobi_candidate_stack_topk=int(args.tra_fix_candidate_stack_topk),
        fixgurobi_max_candidate_stacks_per_order=int(args.tra_fix_max_candidate_stacks_per_order),
        fixgurobi_candidate_station_topk_per_stack=int(args.tra_fix_candidate_station_topk_per_stack),
        fixgurobi_force_candidate_stacks=True,
        fixgurobi_enable_scale_adaptive_candidate_prune=bool(args.tra_fix_scale_adaptive_candidate_prune),
        fixgurobi_allow_warm_start_fallback=False,
        fixgurobi_warm_start_subtask_ordering="default",
        fixgurobi_force_xyz_scope=True,
        fixgurobi_enable_compiled_cache=bool(args.tra_fix_compiled_cache),
        fixgurobi_enable_two_stage=True,
        fixgurobi_enable_cutoff=True,
        fixgurobi_accept_first_improvement=False,
        fixgurobi_enable_best_obj_stop=False,
        fixgurobi_best_obj_stop_slack=0.999,
        fixgurobi_cheap_gate=True,
        fixgurobi_final_validation=False,
        fixgurobi_final_validation_counts_for_acceptance=False,
        fixgurobi_final_validation_time_limit_sec=0.0,
        fixgurobi_final_validation_use_warm_start=False,
        fixgurobi_final_validation_mip_focus=-1,
        fixgurobi_final_validation_heuristics=-1.0,
        fixgurobi_coarse_time_limit_sec=float(args.tra_fix_coarse_time_limit_sec),
        fixgurobi_coarse_mip_gap=float(args.tra_fix_coarse_mip_gap),
        fixgurobi_fix_used_stack_ids=False,
        fixgurobi_output=bool(args.show_gurobi),
        known_target_guidance=False,
        target_table_fastpath=False,
        target_probe_case_presets=False,
        global_target_probe=False,
        global_target_probe_time_limit_sec=0.0,
        global_target_probe_stage_time_limit_sec=0.0,
        global_target_probe_obj_slack=0.999,
        global_target_probe_full_candidate_on_fail=False,
        global_target_probe_candidate_stack_topk=3,
        global_target_probe_candidate_station_topk_per_stack=2,
        global_target_probe_max_candidate_stacks_per_order=24,
        resource_global_decomp_repair=False,
        resource_global_decomp_repair_time_limit_sec=0.0,
        resource_global_decomp_repair_stage_time_limit_sec=0.0,
        resource_global_decomp_repair_best_obj_stop=False,
        resource_global_decomp_repair_obj_slack=0.999,
        resource_global_decomp_repair_candidate_stack_topk=3,
        resource_global_decomp_repair_candidate_station_topk_per_stack=2,
        resource_global_decomp_repair_max_candidate_stacks_per_order=24,
        resource_global_decomp_repair_route_time_window_arc_prune=True,
        resource_global_decomp_repair_use_fresh_problem=True,
        resource_skip_initial_fixgurobi_eval=False,
        tra_revolving_mode=True,
        revolving_enable_u_layer=True,
        u_repair_time_limit_sec=5.0,
        u_repair_max_local_moves=200,
        u_repair_neighborhood_robots=3,
        revolving_inner_time_limit_sec=float(args.tra_fix_revolving_inner_sec),
        revolving_outer_time_limit_sec=float(args.tra_fix_revolving_outer_sec),
        revolving_lb_eps=1e-6,
        revolving_max_iters=50,
        revolving_mark_limit=int(args.tra_fix_revolving_mark_limit),
        revolving_layer_order=str(args.tra_fix_layer_order),
        revolving_yz_fix_scope="",
        revolving_allow_nonimproving_exact=False,
        revolving_sa_init_temp=-1.0,
        resource_candidate_pool_log=False,
        compact_tra_summary_json=True,
        candidate_pool_max_attempts=48,
        stop_if_no_change_rounds=40,
        operator_profile="baseline_safe",
        output_root=case_root,
        gurobi_baseline_details_json="",
    )
    baseline = {}
    if gurobi_row and math.isfinite(_safe_float(gurobi_row.get("cmax"))):
        baseline[case_name] = {
            "model_cmax": _safe_float(gurobi_row.get("cmax")),
            "runtime_sec": _safe_float(gurobi_row.get("runtime_sec")),
            "model_gap": _safe_float(gurobi_row.get("gap")),
        }
    row = tra_gurobi.run_case(ns, case_name, case_root, baseline)
    return {
        "algorithm": "tra_fixgurobi",
        "status": row.get("status", ""),
        "cmax": _safe_float(row.get("tra_gurobi_cmax")),
        "runtime_sec": _safe_float(row.get("tra_gurobi_total_runtime_sec", row.get("total_runtime_sec"))),
        "gap": _safe_float(row.get("best_gurobi_gap")),
        "best_iter": row.get("best_iter", ""),
        "result_root": case_root,
    }


def _comparison_row(case: str, split: str, rows_by_algorithm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    gurobi = dict(rows_by_algorithm.get("gurobi", {}) or {})
    tra_fix = dict(rows_by_algorithm.get("tra_fixgurobi", {}) or {})
    tra_fast = dict(rows_by_algorithm.get("tra_fast", {}) or {})
    gurobi_cmax = _safe_float(gurobi.get("cmax"))
    tra_fix_cmax = _safe_float(tra_fix.get("cmax"))
    tra_fast_cmax = _safe_float(tra_fast.get("cmax"))
    fix_gap = (tra_fix_cmax - gurobi_cmax) / max(1e-9, gurobi_cmax) if math.isfinite(gurobi_cmax) and math.isfinite(tra_fix_cmax) else float("nan")
    fast_gap = (tra_fast_cmax - gurobi_cmax) / max(1e-9, gurobi_cmax) if math.isfinite(gurobi_cmax) and math.isfinite(tra_fast_cmax) else float("nan")
    best_tra_cmax = min([v for v in [tra_fix_cmax, tra_fast_cmax] if math.isfinite(v)] or [float("nan")])
    best_tra_name = ""
    if math.isfinite(best_tra_cmax):
        if math.isfinite(tra_fix_cmax) and abs(best_tra_cmax - tra_fix_cmax) <= 1e-9:
            best_tra_name = "tra_fixgurobi"
        else:
            best_tra_name = "tra_fast"
    best_gap = (best_tra_cmax - gurobi_cmax) / max(1e-9, gurobi_cmax) if math.isfinite(gurobi_cmax) and math.isfinite(best_tra_cmax) else float("nan")
    return {
        "Case": case,
        "Split": split,
        "Gurobi cmax": gurobi_cmax,
        "TRA-FixGurobi cmax": tra_fix_cmax,
        "TRA-FixGurobi vs Gurobi gap": fix_gap,
        "TRA-Fast cmax": tra_fast_cmax,
        "TRA-Fast vs Gurobi gap": fast_gap,
        "Best TRA cmax": best_tra_cmax,
        "Best TRA algorithm": best_tra_name,
        "Best TRA vs Gurobi gap": best_gap,
        "Gurobi gap": _safe_float(gurobi.get("gap")),
        "Gurobi s": _safe_float(gurobi.get("runtime_sec")),
        "TRA-FixGurobi s": _safe_float(tra_fix.get("runtime_sec")),
        "TRA-Fast s": _safe_float(tra_fast.get("runtime_sec")),
        "Gurobi status": gurobi.get("status", ""),
        "TRA-FixGurobi status": tra_fix.get("status", ""),
        "TRA-Fast status": tra_fast.get("status", ""),
    }


def _is_completed_algorithm_row(row: Dict[str, Any]) -> bool:
    status = str(row.get("status", "") or "").strip()
    if not status or status.startswith("error:"):
        return False
    cmax = _safe_float(row.get("cmax"))
    runtime = _safe_float(row.get("runtime_sec"))
    return bool(math.isfinite(cmax) and math.isfinite(runtime))


def _algorithm_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("case", "") or "").upper(), str(row.get("algorithm", "") or "").lower())


def _rows_by_case_algorithm(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows or []:
        key = _algorithm_key(row)
        if key[0] and key[1] and _is_completed_algorithm_row(row):
            out[key] = dict(row)
    return out


def _rebuild_comparison_rows(
    definitions: Dict[str, Tuple[str, str, float]],
    algorithm_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    comparison_rows = _baseline_rows()
    rows_by_key = _rows_by_case_algorithm(algorithm_rows)
    split_names_by_case: Dict[str, str] = {
        str(split_case).upper(): str(split_name)
        for split_case, (_base_case, split_name, _ratio) in definitions.items()
    }
    for case, _algorithm in rows_by_key:
        case_upper = str(case).upper()
        if case_upper in split_names_by_case:
            continue
        for split_name in SPLIT_PROFILES:
            if case_upper.endswith(f"-{split_name}"):
                split_names_by_case[case_upper] = str(split_name)
                break
    for split_case in sorted(split_names_by_case, key=lambda value: (_base_case_from_split(value), value)):
        split_name = split_names_by_case[str(split_case).upper()]
        rows_by_algorithm = {
            algorithm: row
            for (case, algorithm), row in rows_by_key.items()
            if str(case).upper() == str(split_case).upper()
        }
        if rows_by_algorithm:
            comparison_rows.append(_comparison_row(split_case, split_name, rows_by_algorithm))
    return comparison_rows


def _baseline_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in M_CASES:
        row = dict(CURRENT_M_BASELINE[case])
        rows.append(
            {
                "Case": case,
                "Split": "CURRENT-M",
                "Gurobi cmax": row["gurobi_cmax"],
                "TRA-FixGurobi cmax": row["tra_cmax"],
                "TRA-FixGurobi vs Gurobi gap": row["tra_gap"],
                "TRA-Fast cmax": "",
                "TRA-Fast vs Gurobi gap": "",
                "Best TRA cmax": row["tra_cmax"],
                "Best TRA algorithm": "tra_fixgurobi",
                "Best TRA vs Gurobi gap": row["tra_gap"],
                "Gurobi gap": row["gurobi_gap"],
                "Gurobi s": row["gurobi_s"],
                "TRA-FixGurobi s": row["tra_s"],
                "TRA-Fast s": "",
                "Gurobi status": "baseline",
                "TRA-FixGurobi status": "baseline",
                "TRA-Fast status": "",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BOM-split M-suite experiments for Gurobi, TRA-FixGurobi, and TRA-Fast.")
    parser.add_argument("--cases", nargs="+", default=list(M_CASES))
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_PROFILES))
    parser.add_argument("--algorithms", nargs="+", choices=["gurobi", "tra_fixgurobi", "tra_fast"], default=["gurobi", "tra_fixgurobi", "tra_fast"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-comparison-only", action="store_true", default=False)
    parser.add_argument("--show-gurobi", action="store_true", default=False)
    parser.add_argument("--gurobi-time-limit-sec", type=float, default=3600.0)
    parser.add_argument("--gurobi-mip-gap", type=float, default=0.01)
    parser.add_argument("--gurobi-candidate-stack-topk", type=int, default=3)
    parser.add_argument("--gurobi-max-candidate-stacks-per-order", type=int, default=0)
    parser.add_argument("--gurobi-candidate-station-topk-per-stack", type=int, default=0)
    parser.add_argument("--gurobi-route-pickup-neighbor-limit", type=int, default=5)
    parser.add_argument("--gurobi-warm-start-sp4-time-limit-sec", type=int, default=3)
    parser.add_argument("--tra-fast-max-iters", type=int, default=50)
    parser.add_argument("--tra-fast-no-improve-limit", type=int, default=3)
    parser.add_argument("--tra-fast-epsilon", type=float, default=0.05)
    parser.add_argument("--tra-fast-sp2-time-limit-sec", type=float, default=3.0)
    parser.add_argument("--tra-fast-revolving-inner-sec", type=float, default=2.0)
    parser.add_argument("--tra-fast-revolving-outer-sec", type=float, default=20.0)
    parser.add_argument("--tra-fix-max-iters", type=int, default=4)
    parser.add_argument("--tra-fix-no-improve-limit", type=int, default=2)
    parser.add_argument("--tra-fix-epsilon", type=float, default=0.05)
    parser.add_argument("--tra-fix-sp2-time-limit-sec", type=float, default=3.0)
    parser.add_argument("--tra-fix-time-limit-sec", type=float, default=180.0)
    parser.add_argument("--tra-fix-coarse-time-limit-sec", type=float, default=25.0)
    parser.add_argument("--tra-fix-mip-gap", type=float, default=0.01)
    parser.add_argument("--tra-fix-coarse-mip-gap", type=float, default=0.05)
    parser.add_argument("--tra-fix-candidate-trial-limit", type=int, default=1)
    parser.add_argument("--tra-fix-cache-size", type=int, default=512)
    parser.add_argument("--tra-fix-compiled-cache-size", type=int, default=32)
    parser.add_argument("--tra-fix-candidate-stack-topk", type=int, default=6)
    parser.add_argument("--tra-fix-max-candidate-stacks-per-order", type=int, default=18)
    parser.add_argument("--tra-fix-candidate-station-topk-per-stack", type=int, default=2)
    parser.add_argument("--tra-fix-scale-adaptive-candidate-prune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-fix-compiled-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tra-fix-layer-order", default="Y,YZ,XYZ")
    parser.add_argument("--tra-fix-revolving-inner-sec", type=float, default=5.0)
    parser.add_argument("--tra-fix-revolving-outer-sec", type=float, default=120.0)
    parser.add_argument("--tra-fix-revolving-mark-limit", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [str(case).upper() for case in args.cases]
    splits = [str(split).upper().replace("_", "-") for split in args.splits]
    unknown_cases = [case for case in cases if case not in M_CASES]
    unknown_splits = [split for split in splits if split not in SPLIT_PROFILES]
    if unknown_cases:
        raise SystemExit(f"unknown M cases: {unknown_cases}")
    if unknown_splits:
        raise SystemExit(f"unknown split profiles: {unknown_splits}")

    output_root = args.output_root or os.path.join(ROOT_DIR, "result", f"bom_split_m_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_root = _ensure_dir(output_root)
    definitions = _split_definitions(cases, splits)
    case_stats_path = os.path.join(output_root, "bom_split_case_stats.csv")
    algorithm_results_path = os.path.join(output_root, "bom_split_algorithm_results.csv")
    comparison_path = os.path.join(output_root, "bom_split_comparison_with_current_m.csv")
    split_case_rows: List[Dict[str, Any]] = _read_csv(case_stats_path) if bool(args.resume) else []
    algorithm_rows: List[Dict[str, Any]] = _read_csv(algorithm_results_path) if bool(args.resume) else []
    completed_rows = _rows_by_case_algorithm(algorithm_rows)
    comparison_rows: List[Dict[str, Any]] = _rebuild_comparison_rows(definitions, algorithm_rows)

    if bool(args.rebuild_comparison_only):
        _write_csv(comparison_path, comparison_rows)
        print(f"comparison_csv={comparison_path}", flush=True)
        return

    seen_case_stats = {str(row.get("case", "") or "").upper() for row in split_case_rows}

    with patched_split_generator(definitions, seed_override=int(args.seed)):
        for split_case, (base_case, split_name, ratio) in definitions.items():
            problem = CreateOFSProblem.generate_problem_by_scale(split_case, seed=int(args.seed))
            stats = _problem_stats(problem, split_case, base_case, split_name, ratio)
            if str(split_case).upper() not in seen_case_stats:
                split_case_rows.append(stats)
                seen_case_stats.add(str(split_case).upper())
            _write_csv(case_stats_path, split_case_rows)
            print(
                f"[case] {split_case}: orders {stats['original_orders']} -> {stats['split_orders']} "
                f"ratio={stats['actual_split_ratio']:.3f}",
                flush=True,
            )
            rows_by_algorithm: Dict[str, Dict[str, Any]] = {
                algorithm: row
                for (case, algorithm), row in completed_rows.items()
                if str(case).upper() == str(split_case).upper()
            }
            if bool(args.dry_run):
                continue
            for algorithm in args.algorithms:
                key = (str(split_case).upper(), str(algorithm).lower())
                if bool(args.resume) and key in completed_rows:
                    row = dict(completed_rows[key])
                    rows_by_algorithm[str(algorithm).lower()] = row
                    print(
                        f"  [{algorithm}] resume-skip status={row.get('status')} cmax={row.get('cmax')} "
                        f"runtime={_safe_float(row.get('runtime_sec')):.2f}s",
                        flush=True,
                    )
                    continue
                alg_root = _ensure_dir(os.path.join(output_root, split_case, algorithm))
                t0 = time.perf_counter()
                try:
                    if algorithm == "gurobi":
                        fresh_problem = CreateOFSProblem.generate_problem_by_scale(split_case, seed=int(args.seed))
                        row = _run_gurobi(fresh_problem, args)
                    elif algorithm == "tra_fast":
                        row = _run_tra_fast(split_case, args, alg_root)
                    elif algorithm == "tra_fixgurobi":
                        row = _run_tra_fixgurobi(split_case, args, alg_root, rows_by_algorithm.get("gurobi"))
                    else:
                        raise ValueError(f"unknown algorithm: {algorithm}")
                except Exception as exc:
                    row = {
                        "algorithm": algorithm,
                        "status": f"error:{exc.__class__.__name__}",
                        "error_text": str(exc),
                        "cmax": float("nan"),
                        "runtime_sec": float(time.perf_counter() - t0),
                        "gap": float("nan"),
                    }
                row.update(stats)
                row["algorithm"] = algorithm
                rows_by_algorithm[algorithm] = dict(row)
                algorithm_rows.append(row)
                if _is_completed_algorithm_row(row):
                    completed_rows[key] = dict(row)
                _write_csv(algorithm_results_path, algorithm_rows)
                print(
                    f"  [{algorithm}] status={row.get('status')} cmax={row.get('cmax')} "
                    f"runtime={_safe_float(row.get('runtime_sec')):.2f}s",
                    flush=True,
                )
            comparison_rows = _rebuild_comparison_rows(definitions, algorithm_rows)
            _write_csv(comparison_path, comparison_rows)

    _write_csv(case_stats_path, split_case_rows)
    _write_csv(algorithm_results_path, algorithm_rows)
    comparison_rows = _rebuild_comparison_rows(definitions, algorithm_rows)
    _write_csv(comparison_path, comparison_rows)
    _write_json(
        os.path.join(output_root, "bom_split_run_config.json"),
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "cases": cases,
            "splits": splits,
            "algorithms": list(args.algorithms),
            "seed": int(args.seed),
            "dry_run": bool(args.dry_run),
            "split_profiles": SPLIT_PROFILES,
            "current_m_baseline": CURRENT_M_BASELINE,
            "args": vars(args),
        },
    )
    print(f"output_root={output_root}", flush=True)
    print(f"comparison_csv={comparison_path}", flush=True)


if __name__ == "__main__":
    main()
