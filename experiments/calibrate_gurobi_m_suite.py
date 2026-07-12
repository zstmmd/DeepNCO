import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver
from problemDto.createInstance import CreateOFSProblem


DEFAULT_S9 = {
    "case": "GUROBI-S9",
    "orders": 5,
    "skus": 36,
    "totes": 145,
    "stations": 4,
    "robots": 4,
    "stacks": 26,
    "cmax": 438.0,
    "runtime_sec": 937.097453,
    "gap": 0.007479,
}


FIXED_M1 = {
    "case": "GUROBI-M1",
    "status": "OPTIMAL",
    "orders": 6,
    "skus": 42,
    "totes": 172,
    "stations": 4,
    "robots": 5,
    "stacks": 30,
    "slot_count": 6,
    "candidate_stack_count_total": 15,
    "route_task_count_before_station_prune": 60,
    "route_task_count_after_station_prune": 60,
    "runtime_sec": 1118.5147872001398,
    "solver_runtime_sec": 1116.3672776999883,
    "gurobi_runtime_sec": 1115.601000070572,
    "gurobi_solve_time_sec": 1115.6312587999273,
    "model_cmax": 489.0,
    "model_objective": 489.6200000000001,
    "model_best_bound": 489.51158761528336,
    "model_gap": 0.0002214214793447112,
    "model_status_code": 2,
    "model_sol_count": 10,
    "config": {
        "map_size": (4, 5),
        "resources": (5, 4, 172),
        "data": (6, 42),
        "bom_complexity": (8, 1),
        "target_stack_count": 30,
        "inventory_cold_filler_probability": 0.25,
        "exact_order_sku_counts": (6, 6, 6, 6, 6, 6),
        "exact_order_sku_quantity_range": (11, 13),
        "bom_colocated_inventory": True,
        "bom_colocated_stack_counts": (2, 2, 3, 3, 3, 3),
        "bom_colocated_disjoint_stack_groups": True,
        "bom_colocated_support_multiplier": 1.2,
    },
    "spec": {
        "map_size": [4, 5],
        "order_count": 6,
        "qty_range": [11, 13],
        "robot_count": 5,
        "sku_count": 42,
        "stack_counts": [2, 2, 3, 3, 3, 3],
        "station_count": 4,
        "target_stack_count": 30,
        "tote_count": 172,
    },
    "fixed_source": "result/gurobi_m1_design_route_arc_load_only_1200s_gap001_v9",
}


@dataclass(frozen=True)
class CandidateSpec:
    case: str
    order_count: int
    station_count: int
    robot_count: int
    sku_count: int
    tote_count: int
    target_stack_count: int
    stack_counts: Tuple[int, ...]
    qty_range: Tuple[int, int]
    map_size: Tuple[int, int] = (4, 5)

    def scale_key(self) -> Tuple[int, int, int, int, int, int]:
        return (
            int(self.order_count),
            int(self.sku_count),
            int(self.tote_count),
            int(self.station_count),
            int(self.robot_count),
            int(self.target_stack_count),
        )

    def route_complexity_key(self) -> Tuple[int, int, int, int]:
        return (
            int(sum(self.stack_counts) * self.station_count),
            int(sum(self.stack_counts)),
            int(self.station_count),
            int(self.order_count),
        )

    def to_problem_config(self) -> Dict[str, Any]:
        return {
            "map_size": tuple(int(v) for v in self.map_size),
            "resources": (int(self.robot_count), int(self.station_count), int(self.tote_count)),
            "data": (int(self.order_count), int(self.sku_count)),
            "bom_complexity": (8, 1),
            "target_stack_count": int(self.target_stack_count),
            "inventory_cold_filler_probability": 0.25,
            "exact_order_sku_counts": tuple(int(v) for v in self.stack_counts_to_sku_counts()),
            "exact_order_sku_quantity_range": tuple(int(v) for v in self.qty_range),
            "bom_colocated_inventory": True,
            "bom_colocated_stack_counts": tuple(int(v) for v in self.stack_counts),
            "bom_colocated_disjoint_stack_groups": True,
            "bom_colocated_support_multiplier": 1.2,
        }

    def stack_counts_to_sku_counts(self) -> Tuple[int, ...]:
        # Keep per-order SKU count stable while stack-count patterns control SP4 route candidates.
        return tuple(6 for _ in range(int(self.order_count)))


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _base_solver_cfg(args: argparse.Namespace, time_limit: float) -> GlobalXYZUConfig:
    return GlobalXYZUConfig(
        time_limit_sec=float(time_limit),
        mip_gap=float(args.mip_gap),
        candidate_stack_topk=999,
        candidate_station_topk_per_stack=999,
        max_candidate_stacks_per_order=0,
        enable_warm_start=False,
        warm_start_use_sp4=False,
        write_lp=False,
        gurobi_output=bool(args.show_gurobi),
        integrate_u_route=True,
        route_arc_prune=True,
        enable_route_time_window_arc_prune=False,
        enable_route_load_interval_arc_prune=True,
        enable_route_directional_arc_prune=False,
        route_pickup_neighbor_limit=0,
        enable_scale_adaptive_candidate_prune=False,
        enable_sp4_fallback=False,
    )


def _problem_summary(problem: Any) -> Dict[str, int]:
    return {
        "orders": int(getattr(problem, "order_num", len(getattr(problem, "order_list", []) or [])) or 0),
        "skus": int(getattr(problem, "skus_num", len(getattr(problem, "skus_list", []) or [])) or 0),
        "totes": int(getattr(problem, "tote_num", len(getattr(problem, "tote_list", []) or [])) or 0),
        "stations": int(getattr(problem, "station_num", len(getattr(problem, "station_list", []) or [])) or 0),
        "robots": int(getattr(problem, "robot_num", len(getattr(problem, "robot_list", []) or [])) or 0),
        "stacks": int(len(getattr(problem, "stack_list", []) or [])),
    }


def _register_candidate(spec: CandidateSpec) -> None:
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS = dict(getattr(CreateOFSProblem, "RUNTIME_SCALE_CONFIGS", {}) or {})
    CreateOFSProblem.RUNTIME_SCALE_CONFIGS[str(spec.case).upper()] = spec.to_problem_config()


def _compile_route_stats(args: argparse.Namespace, spec: CandidateSpec) -> Dict[str, Any]:
    _register_candidate(spec)
    problem = CreateOFSProblem.generate_problem_by_scale(spec.case, seed=int(args.seed))
    compiled = GlobalXYZUSolver().compile_model(problem, _base_solver_cfg(args, time_limit=1.0))
    diag = dict(compiled.diagnostics or {})
    prepared = dict(compiled.prepared or {})
    return {
        **_problem_summary(problem),
        "slot_count": int(len(prepared.get("slots", []) or [])),
        "candidate_stack_count_total": int(
            sum(len(v) for v in (prepared.get("candidate_stacks_by_order", {}) or {}).values())
        ),
        "route_task_count_before_station_prune": int(diag.get("route_task_count_before_station_prune", 0) or 0),
        "route_task_count_after_station_prune": int(diag.get("route_task_count_after_station_prune", 0) or 0),
        "u_arc_count": int(diag.get("u_arc_count", 0) or 0),
        "u_load_interval_pruned_arc_count": int(diag.get("u_load_interval_pruned_arc_count", 0) or 0),
        "u_time_window_pruned_arc_count": int(diag.get("u_time_window_pruned_arc_count", 0) or 0),
        "u_directional_pruned_arc_count": int(diag.get("u_directional_pruned_arc_count", 0) or 0),
        "u_knn_pruned_arc_count": int(diag.get("u_knn_pruned_arc_count", 0) or 0),
    }


def _solve_candidate(args: argparse.Namespace, spec: CandidateSpec, out_dir: str) -> Dict[str, Any]:
    _register_candidate(spec)
    problem = CreateOFSProblem.generate_problem_by_scale(spec.case, seed=int(args.seed))
    route_stats = _compile_route_stats(args, spec)
    t0 = time.perf_counter()
    result = GlobalXYZUSolver().solve(problem, _base_solver_cfg(args, time_limit=float(args.time_limit_sec)))
    wall = float(time.perf_counter() - t0)
    diag = dict(result.diagnostics or {})
    row = {
        "case": spec.case,
        "status": str(result.status),
        "runtime_sec": wall,
        "solver_runtime_sec": _finite_float(getattr(result, "runtime_sec", float("nan"))),
        "gurobi_runtime_sec": _finite_float(diag.get("gurobi_runtime_sec", float("nan"))),
        "gurobi_solve_time_sec": _finite_float(diag.get("gurobi_solve_time_sec", float("nan"))),
        "model_cmax": _finite_float(diag.get("model_cmax", getattr(result, "objective", float("nan")))),
        "model_objective": _finite_float(diag.get("model_objective", getattr(result, "objective", float("nan")))),
        "model_best_bound": _finite_float(diag.get("model_best_bound", float("nan"))),
        "model_gap": _finite_float(diag.get("model_gap", getattr(result, "gap", float("nan")))),
        "model_status_code": int(diag.get("model_status_code", 0) or 0),
        "model_sol_count": int(diag.get("model_sol_count", 0) or 0),
        "config": spec.to_problem_config(),
        "spec": {
            "order_count": spec.order_count,
            "station_count": spec.station_count,
            "robot_count": spec.robot_count,
            "sku_count": spec.sku_count,
            "tote_count": spec.tote_count,
            "target_stack_count": spec.target_stack_count,
            "stack_counts": list(spec.stack_counts),
            "qty_range": list(spec.qty_range),
            "map_size": list(spec.map_size),
        },
        **route_stats,
    }
    _write_json(os.path.join(out_dir, f"{spec.case.lower()}_last_result.json"), row)
    return row


def _scale_bigger(row: Dict[str, Any], prev: Dict[str, Any]) -> bool:
    scale_keys = ["orders", "skus", "totes", "stations", "robots", "stacks"]
    return all(int(row.get(k, 0) or 0) >= int(prev.get(k, 0) or 0) for k in scale_keys) and any(
        int(row.get(k, 0) or 0) > int(prev.get(k, 0) or 0) for k in scale_keys
    )


def _calibration_status_ok(row: Dict[str, Any]) -> bool:
    # Gurobi may stop on TIME_LIMIT after already reaching the configured gap.
    return str(row.get("status", "")).upper() in {"OPTIMAL", "USER_OBJ_LIMIT", "TIME_LIMIT"}


def _row_accepts_stage(args: argparse.Namespace, stage: int, row: Dict[str, Any], prev: Dict[str, Any]) -> Tuple[bool, str]:
    if not _calibration_status_ok(row):
        return False, "status"
    if _finite_float(row.get("model_gap")) > float(args.mip_gap) + 1e-9:
        return False, "gap"
    runtime = _finite_float(row.get("runtime_sec"))
    if not math.isfinite(runtime) or runtime >= float(args.time_limit_sec):
        return False, "runtime_max"
    if stage < int(args.stages):
        stage_runtime_ceiling = float(args.time_limit_sec) - (
            int(args.stages) - int(stage)
        ) * float(args.reserve_runtime_step_sec)
        if runtime >= stage_runtime_ceiling:
            return False, "runtime_reserve"
    if stage == 1:
        if runtime < float(args.m1_min_runtime_sec) or runtime > float(args.m1_max_runtime_sec):
            return False, "m1_runtime_window"
    elif runtime <= _finite_float(prev.get("runtime_sec")) + float(args.min_runtime_step_sec):
        return False, "runtime_not_increasing"
    if _finite_float(row.get("model_cmax")) <= _finite_float(prev.get("model_cmax", prev.get("cmax"))) + float(args.min_cmax_step):
        return False, "cmax_not_increasing"
    if not _scale_bigger(row, prev):
        return False, "scale_not_increasing"
    return True, "accepted"


def _stack_patterns(order_count: int, min_sum: int, max_sum: int) -> Iterable[Tuple[int, ...]]:
    seeds = [
        tuple([2] * order_count),
        tuple([2] + [3] * (order_count - 1)),
        tuple([2, 2] + [3] * max(0, order_count - 2)),
        tuple([3] * order_count),
    ]
    seen = set()
    for pattern in seeds:
        if len(pattern) == order_count and min_sum <= sum(pattern) <= max_sum and pattern not in seen:
            seen.add(pattern)
            yield pattern
    for threes in range(0, order_count + 1):
        pattern = tuple([2] * (order_count - threes) + [3] * threes)
        if min_sum <= sum(pattern) <= max_sum and pattern not in seen:
            seen.add(pattern)
            yield pattern


def _generate_stage_candidates(args: argparse.Namespace, stage: int, prev: Dict[str, Any]) -> List[CandidateSpec]:
    prev_orders = int(prev.get("orders", 5) or 5)
    prev_skus = int(prev.get("skus", 36) or 36)
    prev_totes = int(prev.get("totes", 145) or 145)
    prev_stacks = int(prev.get("stacks", 26) or 26)
    prev_runtime = _finite_float(prev.get("runtime_sec", 0.0), 0.0)

    order_options = sorted(set([max(6, prev_orders), max(6, prev_orders + 1)]))
    if stage == 1:
        order_options = [6]
    prev_stations = int(prev.get("stations", 4) or 4)
    prev_robots = int(prev.get("robots", 5) or 5)
    station_options = sorted(set([max(4, prev_stations), max(4, min(5, prev_stations + 1))]))
    robot_options = [max(5, prev_robots)]
    base_qty = 11 + min(stage - 1, 5)
    qty_ranges = sorted(set([
        (base_qty, base_qty + 2),
        (base_qty + 1, base_qty + 2),
        (base_qty + 1, base_qty + 3),
    ]))
    candidates: List[CandidateSpec] = []

    priority_specs: List[CandidateSpec] = []
    if stage == 1:
        priority_specs.append(
            CandidateSpec(
                case="GUROBI-M1_KNOWN_0000",
                order_count=6,
                station_count=4,
                robot_count=5,
                sku_count=42,
                tote_count=172,
                target_stack_count=30,
                stack_counts=(2, 2, 3, 3, 3, 3),
                qty_range=(11, 13),
            )
        )
    if stage == 2:
        priority_specs.append(
            CandidateSpec(
                case="GUROBI-M2_KNOWN_0000",
                order_count=7,
                station_count=5,
                robot_count=5,
                sku_count=44,
                tote_count=184,
                target_stack_count=33,
                stack_counts=(2, 2, 3, 3, 3, 3, 3),
                qty_range=(13, 15),
            )
        )
    if stage >= 7:
        late_stations = max(5, prev_stations)
        late_robots = max(5, prev_robots)
        late_qty_base = max(17, int(base_qty))
        focus_order_options = sorted(set([max(8, prev_orders), max(8, prev_orders + 1)]))
        idx = 0
        if stage == 9:
            # M9 is calibrated from the fixed M8 row.  The broader pattern
            # sweep hits several 3600s time-limit structures before reaching
            # useful variants, so try the known-solvable route-heavy pattern
            # first and nudge only scale / quantity.
            m9_pattern = tuple([2, 2, 2, 2, 3, 3, 3, 3])
            m9_variants = [
                (8, 64, 261, 50, (18, 20)),
                (8, 65, 265, 51, (18, 20)),
                (8, 66, 269, 52, (18, 20)),
                (8, 64, 269, 51, (18, 20)),
                (8, 65, 269, 52, (18, 20)),
                (8, 66, 273, 53, (18, 20)),
                (8, 64, 261, 50, (19, 21)),
                (8, 65, 265, 51, (19, 21)),
                (8, 66, 269, 52, (19, 21)),
                (8, 65, 265, 51, (18, 21)),
                (8, 66, 269, 52, (18, 21)),
                (8, 67, 273, 53, (18, 21)),
            ]
            for order_count, sku_count, tote_count, stack_count, qty_range in m9_variants:
                priority_specs.append(
                    CandidateSpec(
                        case=f"GUROBI-M{stage}_TARGET_{idx:04d}",
                        order_count=int(order_count),
                        station_count=int(late_stations),
                        robot_count=int(late_robots),
                        sku_count=int(max(sku_count, prev_skus + 1)),
                        tote_count=int(max(tote_count, prev_totes + 4)),
                        target_stack_count=int(max(stack_count, prev_stacks + 1)),
                        stack_counts=m9_pattern,
                        qty_range=(int(qty_range[0]), int(qty_range[1])),
                    )
                )
                idx += 1
        for late_orders in focus_order_options:
            late_sku_base = max(prev_skus + 1, 58 + 2 * (stage - 7), late_orders * 6)
            late_tote_base = max(prev_totes + 4, 237 + 8 * (stage - 7))
            late_stack_base = max(prev_stacks + 1, 44 + 2 * (stage - 7))
            pattern_specs = [
                (late_orders - 4, 4, 0),
                (late_orders - 3, 2, 1),
                (late_orders - 3, 3, 0),
                (late_orders - 2, 2, 0),
                (late_orders - 4, 3, 1),
            ]
            late_patterns = []
            for twos, threes, fours in pattern_specs:
                if twos < 0 or threes < 0 or fours < 0:
                    continue
                pattern = tuple([2] * twos + [3] * threes + [4] * fours)
                if len(pattern) == late_orders and pattern not in late_patterns:
                    late_patterns.append(pattern)
            qty_focus = [
                (late_qty_base + 1, late_qty_base + 3),
                (late_qty_base + 2, late_qty_base + 4),
                (late_qty_base, late_qty_base + 3),
                (late_qty_base + 1, late_qty_base + 2),
                (late_qty_base, late_qty_base + 2),
            ]
            for sku_delta, tote_delta, stack_delta, qty_range, stack_counts in itertools.product(
                [0, 2, 4],
                [0, 8, 16],
                [0, 1, 2],
                qty_focus,
                late_patterns,
            ):
                priority_specs.append(
                    CandidateSpec(
                        case=f"GUROBI-M{stage}_LATE_{idx:04d}",
                        order_count=int(late_orders),
                        station_count=int(late_stations),
                        robot_count=int(late_robots),
                        sku_count=int(late_sku_base + sku_delta),
                        tote_count=int(late_tote_base + tote_delta),
                        target_stack_count=int(late_stack_base + stack_delta),
                        stack_counts=tuple(int(v) for v in stack_counts),
                        qty_range=(int(qty_range[0]), int(qty_range[1])),
                    )
                )
                idx += 1

    for orders, stations, robots, qty_range in itertools.product(order_options, station_options, robot_options, qty_ranges):
        sku_base = max(prev_skus + 1, 36 + 2 * stage + orders)
        tote_base = max(prev_totes + 4, 145 + 8 * stage + 4 * orders)
        target_base = max(prev_stacks + 1, 26 + stage + orders)
        for sku_delta in [0, 1, 2]:
            for tote_delta in [0, 4, 8]:
                sku_count = int(sku_base + sku_delta)
                tote_count = int(tote_base + tote_delta)
                target_stack_count = int(target_base + (tote_delta // 4))
                min_stack_sum = max(orders * 2, int(prev.get("candidate_stack_count_total", orders * 2) or orders * 2))
                max_stack_sum = min(orders * 3, min_stack_sum + 4)
                for stack_counts in _stack_patterns(orders, min_sum=min_stack_sum, max_sum=max_stack_sum):
                    spec = CandidateSpec(
                        case=f"GUROBI-M{stage}_CAND_{len(candidates):04d}",
                        order_count=int(orders),
                        station_count=int(stations),
                        robot_count=int(robots),
                        sku_count=int(max(sku_count, sum([6] * orders))),
                        tote_count=int(tote_count),
                        target_stack_count=int(target_stack_count),
                        stack_counts=tuple(int(v) for v in stack_counts),
                        qty_range=(int(qty_range[0]), int(qty_range[1])),
                    )
                    prev_scale = {
                        "orders": int(prev.get("orders", 0) or 0),
                        "skus": int(prev.get("skus", 0) or 0),
                        "totes": int(prev.get("totes", 0) or 0),
                        "stations": int(prev.get("stations", 0) or 0),
                        "robots": int(prev.get("robots", 0) or 0),
                        "stacks": int(prev.get("stacks", 0) or 0),
                    }
                    spec_scale = {
                        "orders": int(spec.order_count),
                        "skus": int(spec.sku_count),
                        "totes": int(spec.tote_count),
                        "stations": int(spec.station_count),
                        "robots": int(spec.robot_count),
                        "stacks": int(spec.target_stack_count),
                    }
                    if not (
                        all(spec_scale[key] >= prev_scale[key] for key in prev_scale)
                        and any(spec_scale[key] > prev_scale[key] for key in prev_scale)
                    ):
                        continue
                    candidates.append(spec)

    # Prefer candidates whose route task count should increase gently from the previous accepted row.
    def score(spec: CandidateSpec) -> Tuple[float, int, Tuple[int, int, int, int]]:
        target_runtime = (
            (float(args.m1_min_runtime_sec) + float(args.m1_max_runtime_sec)) / 2.0
            if stage == 1
            else min(float(args.time_limit_sec) * 0.92, prev_runtime + float(args.target_runtime_step_sec))
        )
        route_score = spec.route_complexity_key()[0]
        prev_route = int(prev.get("route_task_count_after_station_prune", prev.get("route_tasks", 56)) or 56)
        desired_route = prev_route + max(4, int((target_runtime - prev_runtime) / 80.0))
        return (abs(route_score - desired_route), route_score, spec.scale_key())

    candidates.sort(key=score)
    merged: List[CandidateSpec] = []
    seen_configs = set()
    for spec in list(priority_specs) + candidates:
        key = json.dumps(spec.to_problem_config(), sort_keys=True)
        if key in seen_configs:
            continue
        seen_configs.add(key)
        merged.append(spec)
    return merged[: int(args.max_candidates_per_stage)]


def _select_chain_from_rows(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_rows = [
        dict(row)
        for row in rows
        if _calibration_status_ok(row)
        and _finite_float(row.get("model_gap"), float("inf")) <= float(args.mip_gap) + 1e-9
        and _finite_float(row.get("runtime_sec"), float("inf")) < float(args.time_limit_sec)
    ]
    valid_rows.sort(
        key=lambda row: (
            _finite_float(row.get("runtime_sec"), float("inf")),
            _finite_float(row.get("model_cmax"), float("inf")),
            int(row.get("orders", 0) or 0),
            int(row.get("skus", 0) or 0),
            int(row.get("totes", 0) or 0),
            int(row.get("stacks", 0) or 0),
        )
    )

    fixed_m1 = dict(FIXED_M1)
    best: List[Dict[str, Any]] = [dict(fixed_m1)]

    def better(candidate: List[Dict[str, Any]], incumbent: List[Dict[str, Any]]) -> bool:
        if len(candidate) != len(incumbent):
            return len(candidate) > len(incumbent)
        if len(candidate) <= 1:
            return False
        cand_final = _finite_float(candidate[-1].get("runtime_sec"), float("inf"))
        inc_final = _finite_float(incumbent[-1].get("runtime_sec"), float("inf"))
        if abs(cand_final - inc_final) > 1e-9:
            return cand_final < inc_final
        cand_cmax = _finite_float(candidate[-1].get("model_cmax"), float("inf"))
        inc_cmax = _finite_float(incumbent[-1].get("model_cmax"), float("inf"))
        return cand_cmax < inc_cmax

    def dfs(start_idx: int, stage: int, prev: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
        nonlocal best
        if better(chain, best):
            best = [dict(row) for row in chain]
        if stage > int(args.stages):
            return
        if len(chain) + (len(valid_rows) - start_idx) <= len(best):
            return
        for idx in range(start_idx, len(valid_rows)):
            row = valid_rows[idx]
            ok, _reason = _row_accepts_stage(args, stage, row, prev)
            if not ok:
                continue
            next_row = dict(row)
            chain.append(next_row)
            dfs(idx + 1, stage + 1, next_row, chain)
            chain.pop()

    dfs(0, 2, fixed_m1, [dict(fixed_m1)])
    for idx, row in enumerate(best):
        row["case"] = f"GUROBI-M{idx + 1}"
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate GUROBI-M1..M9 medium benchmark cases.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", type=int, default=9)
    parser.add_argument("--time-limit-sec", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--m1-min-runtime-sec", type=float, default=950.0)
    parser.add_argument("--m1-max-runtime-sec", type=float, default=1200.0)
    parser.add_argument("--min-runtime-step-sec", type=float, default=1.0)
    parser.add_argument("--target-runtime-step-sec", type=float, default=220.0)
    parser.add_argument("--reserve-runtime-step-sec", type=float, default=120.0)
    parser.add_argument("--min-cmax-step", type=float, default=0.0)
    parser.add_argument("--max-candidates-per-stage", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--show-gurobi", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    out_dir = _ensure_dir(args.output_dir or os.path.join(ROOT_DIR, "result", f"gurobi_m_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    results_path = os.path.join(out_dir, "candidate_results.jsonl")
    existing_rows = _read_jsonl(results_path)
    tested_keys = {str(row.get("candidate_key", "")) for row in existing_rows}
    selected = _select_chain_from_rows(args, existing_rows)
    prev = dict(selected[-1]) if selected else dict(DEFAULT_S9)

    for stage in range(len(selected) + 1, int(args.stages) + 1):
        stage_candidates = _generate_stage_candidates(args, stage, prev)
        stage_accepted: Optional[Dict[str, Any]] = None
        for spec in stage_candidates:
            key = json.dumps(spec.to_problem_config(), sort_keys=True)
            if key in tested_keys:
                continue
            route_stats = _compile_route_stats(args, spec)
            preview = {"stage": stage, "candidate_key": key, "case": spec.case, "config": spec.to_problem_config(), **route_stats}
            if args.dry_run:
                print(json.dumps(preview, ensure_ascii=False))
                continue
            row = _solve_candidate(args, spec, out_dir)
            row["stage"] = int(stage)
            row["candidate_key"] = key
            ok, reason = _row_accepts_stage(args, stage, row, prev)
            row["acceptance_ok"] = bool(ok)
            row["acceptance_reason"] = str(reason)
            _append_jsonl(results_path, row)
            existing_rows.append(row)
            tested_keys.add(key)
            print(
                f"stage=M{stage} case={spec.case} status={row['status']} cmax={row['model_cmax']} "
                f"runtime={row['runtime_sec']:.3f}s gap={row['model_gap']} accept={ok}:{reason}"
            )
            if ok:
                stage_accepted = dict(row)
                stage_accepted["case"] = f"GUROBI-M{stage}"
                break
        if args.dry_run:
            continue
        if stage_accepted is None:
            print(f"blocked: no acceptable candidate found for GUROBI-M{stage}")
            break
        selected.append(stage_accepted)
        prev = dict(stage_accepted)
        _write_json(os.path.join(out_dir, "selected_chain.json"), selected)
        _write_csv(os.path.join(out_dir, "selected_chain.csv"), selected)
        _write_json(
            os.path.join(out_dir, "selected_problem_configs.json"),
            {f"GUROBI-M{idx + 1}": row.get("config", {}) for idx, row in enumerate(selected)},
        )

    selected = _select_chain_from_rows(args, existing_rows)
    _write_json(os.path.join(out_dir, "selected_chain.json"), selected)
    _write_csv(os.path.join(out_dir, "selected_chain.csv"), selected)
    _write_json(
        os.path.join(out_dir, "selected_problem_configs.json"),
        {f"GUROBI-M{idx + 1}": row.get("config", {}) for idx, row in enumerate(selected)},
    )
    print(f"selected={len(selected)}/{int(args.stages)}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
