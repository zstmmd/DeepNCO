from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from config.ofs_config import OFSConfig

from .state import ResourceConfig, ResourceSubtask
from .utils import pick_ranked_candidate, pick_soft_greedy_min


def _station_count(opt) -> int:
    return max(1, len(getattr(getattr(opt, "problem", None), "station_list", []) or []))


def _station_loads(config: ResourceConfig) -> Dict[int, float]:
    loads: Dict[int, float] = defaultdict(float)
    for row in config.subtasks.values():
        if int(row.station_id) < 0:
            continue
        loads[int(row.station_id)] += float(_subtask_station_work(row))
    return dict(loads)


def _subtask_station_work(subtask: ResourceSubtask) -> float:
    total = 0.0
    for task in subtask.z_tasks or []:
        total += float(max(1, int(task.sku_pick_count or 0))) * float(getattr(OFSConfig, "PICKING_TIME", 1.0))
        total += float(task.station_service_time)
    return float(total)


def _arrival_proxy(opt, subtask: ResourceSubtask, station_id: int) -> float:
    problem = getattr(opt, "problem", None)
    if problem is None or not (0 <= int(station_id) < len(getattr(problem, "station_list", []) or [])):
        return 1e6
    rows = []
    for task in subtask.z_tasks or []:
        stack = problem.point_to_stack.get(int(task.stack_id))
        if stack is None:
            continue
        station = problem.station_list[int(station_id)]
        rows.append(abs(float(stack.store_point.x) - float(station.point.x)) + abs(float(stack.store_point.y) - float(station.point.y)))
    return float(sum(rows) / max(1, len(rows))) if rows else 0.0


def _normalize_ranks(config: ResourceConfig) -> None:
    config.normalize_station_ranks()


def _release_rows(rows: List[ResourceSubtask], move_n: int) -> Dict[int, Tuple[int, int]]:
    released = {}
    for row in list(rows or [])[: max(0, int(move_n))]:
        released[int(row.subtask_id)] = (int(row.station_id), int(row.station_rank))
        row.station_id = -1
        row.station_rank = -1
    return released


def _preview_release_rows(rows: List[ResourceSubtask], move_n: int) -> Dict[int, Tuple[int, int]]:
    released = {}
    for row in list(rows or [])[: max(0, int(move_n))]:
        released[int(row.subtask_id)] = (int(row.station_id), int(row.station_rank))
    return released


def _normalize_station_ranks_subset(config: ResourceConfig, station_ids: List[int]) -> None:
    touched = {int(x) for x in (station_ids or []) if int(x) >= 0}
    if not touched:
        return
    for station_id in sorted(touched):
        rows = [row for row in config.subtasks.values() if int(row.station_id) == int(station_id)]
        rows.sort(key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)))
        for rank, row in enumerate(rows):
            row.station_rank = int(rank)


def y_destroy_congested_station_block(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    loads = _station_loads(config)
    if not loads:
        return {"success": False}
    ranked = sorted(
        [((-float(loads[sid]), int(sid)), int(sid)) for sid in loads.keys()],
        key=lambda item: item[0],
    )
    picked = pick_ranked_candidate(rng, ranked, opt.cfg)
    if picked is None:
        return {"success": False}
    _, heavy_station = picked
    rows = sorted(config.station_subtasks(int(heavy_station)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    if not rows:
        return {"success": False}
    move_n = max(1, min(int(degree), len(rows)))
    chosen = list(reversed(rows))[:move_n]
    released = _release_rows(chosen, move_n)
    return {"success": True, "released_subtasks": released}


def y_destroy_cross_station_fragment(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    order_station_map: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        order_station_map[int(row.order_id)].append(row)
    candidate_rows = []
    for order_id, rows in order_station_map.items():
        stations = {int(row.station_id) for row in rows if int(row.station_id) >= 0}
        if len(stations) >= 2:
            candidate_rows.append(((-len(stations), -len(rows), int(order_id)), int(order_id)))
    if not candidate_rows:
        return {"success": False}
    picked = pick_ranked_candidate(rng, sorted(candidate_rows, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False}
    _, target_order = picked
    rows = sorted(order_station_map[int(target_order)], key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    move_n = max(1, min(int(degree), len(rows)))
    released = _release_rows(rows[:move_n], move_n)
    return {"success": True, "released_subtasks": released}


def y_destroy_rank_window_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) >= 0],
        key=lambda row: (-int(row.station_rank), int(row.subtask_id)),
    )
    if not ranked:
        return {"success": False}
    candidates = [((float(-int(row.station_rank)), int(row.subtask_id)), int(row.subtask_id)) for row in ranked]
    picked = pick_ranked_candidate(rng, candidates, opt.cfg)
    if picked is None:
        return {"success": False}
    _, center_subtask_id = picked
    center_row = config.subtasks.get(int(center_subtask_id))
    if center_row is None or int(center_row.station_id) < 0:
        return {"success": False}
    station_rows = sorted(config.station_subtasks(int(center_row.station_id)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    center_idx = next((idx for idx, row in enumerate(station_rows) if int(row.subtask_id) == int(center_subtask_id)), 0)
    move_n = max(1, min(int(degree), len(station_rows)))
    start = max(0, int(center_idx) - move_n // 2)
    end = min(len(station_rows), start + move_n)
    start = max(0, end - move_n)
    released = _release_rows(station_rows[start:end], move_n)
    return {"success": True, "released_subtasks": released}


def y_destroy_load_skew_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return y_destroy_congested_station_block(opt, config, rng, degree)


def y_plan_destroy_congested_station_block(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    loads = _station_loads(config)
    if not loads:
        return {"success": False}
    ranked = sorted([((-float(loads[sid]), int(sid)), int(sid)) for sid in loads.keys()], key=lambda item: item[0])
    picked = pick_ranked_candidate(rng, ranked, opt.cfg)
    if picked is None:
        return {"success": False}
    _, heavy_station = picked
    rows = sorted(config.station_subtasks(int(heavy_station)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    if not rows:
        return {"success": False}
    move_n = max(1, min(int(degree), len(rows)))
    chosen = list(reversed(rows))[:move_n]
    released = _preview_release_rows(chosen, move_n)
    return {
        "success": True,
        "released_subtasks": released,
        "source_station_ids": [int(heavy_station)],
    }


def y_plan_destroy_cross_station_fragment(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    order_station_map: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        order_station_map[int(row.order_id)].append(row)
    candidate_rows = []
    for order_id, rows in order_station_map.items():
        stations = {int(row.station_id) for row in rows if int(row.station_id) >= 0}
        if len(stations) >= 2:
            candidate_rows.append(((-len(stations), -len(rows), int(order_id)), int(order_id)))
    if not candidate_rows:
        return {"success": False}
    picked = pick_ranked_candidate(rng, sorted(candidate_rows, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False}
    _, target_order = picked
    rows = sorted(order_station_map[int(target_order)], key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    move_n = max(1, min(int(degree), len(rows)))
    released = _preview_release_rows(rows[:move_n], move_n)
    return {
        "success": True,
        "released_subtasks": released,
        "source_station_ids": sorted({int(station_id) for station_id, _ in released.values() if int(station_id) >= 0}),
    }


def y_plan_destroy_rank_window_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked = sorted(
        [row for row in config.subtasks.values() if int(row.station_id) >= 0],
        key=lambda row: (-int(row.station_rank), int(row.subtask_id)),
    )
    if not ranked:
        return {"success": False}
    candidates = [((float(-int(row.station_rank)), int(row.subtask_id)), int(row.subtask_id)) for row in ranked]
    picked = pick_ranked_candidate(rng, candidates, opt.cfg)
    if picked is None:
        return {"success": False}
    _, center_subtask_id = picked
    center_row = config.subtasks.get(int(center_subtask_id))
    if center_row is None or int(center_row.station_id) < 0:
        return {"success": False}
    station_rows = sorted(config.station_subtasks(int(center_row.station_id)), key=lambda row: (int(row.station_rank), int(row.subtask_id)))
    center_idx = next((idx for idx, row in enumerate(station_rows) if int(row.subtask_id) == int(center_subtask_id)), 0)
    move_n = max(1, min(int(degree), len(station_rows)))
    start = max(0, int(center_idx) - move_n // 2)
    end = min(len(station_rows), start + move_n)
    start = max(0, end - move_n)
    released = _preview_release_rows(station_rows[start:end], move_n)
    return {
        "success": True,
        "released_subtasks": released,
        "source_station_ids": [int(center_row.station_id)],
    }


def y_plan_destroy_load_skew_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    return y_plan_destroy_congested_station_block(opt, config, rng, degree)


def _assign_station_rank(config: ResourceConfig, subtask_id: int, station_id: int) -> None:
    row = config.subtasks.get(int(subtask_id))
    if row is None:
        return
    row.station_id = int(station_id)
    current_rows = [item for item in config.subtasks.values() if int(item.station_id) == int(station_id) and int(item.subtask_id) != int(subtask_id)]
    row.station_rank = int(len(current_rows))


def _choose_station_earliest_finish(opt, config: ResourceConfig, subtask: ResourceSubtask, rng=None) -> int:
    loads = _station_loads(config)
    choices = []
    for station_id in range(_station_count(opt)):
        projected_finish = float(loads.get(int(station_id), 0.0) + _subtask_station_work(subtask))
        arrival = float(_arrival_proxy(opt, subtask, station_id))
        choices.append((projected_finish + 0.1 * arrival, int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _choose_station_load_balance(opt, config: ResourceConfig, subtask: ResourceSubtask, rng=None) -> int:
    loads = _station_loads(config)
    choices = []
    for station_id in range(_station_count(opt)):
        projected = dict(loads)
        projected[int(station_id)] = projected.get(int(station_id), 0.0) + _subtask_station_work(subtask)
        vals = list(projected.values()) or [0.0]
        mean = sum(vals) / max(1, len(vals))
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
        choices.append((float(variance), int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _station_load_std_from_map(loads: Dict[int, float]) -> float:
    vals = list(loads.values()) or [0.0]
    mean = sum(vals) / max(1, len(vals))
    variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    return float(math.sqrt(max(0.0, variance)))


def _y_base_maps(config: ResourceConfig, released_ids: List[int]) -> Tuple[Dict[int, float], Dict[int, int]]:
    released = {int(x) for x in (released_ids or [])}
    loads: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    for row in config.subtasks.values():
        if int(row.subtask_id) in released or int(row.station_id) < 0:
            continue
        loads[int(row.station_id)] += float(_subtask_station_work(row))
        counts[int(row.station_id)] += 1
    return dict(loads), dict(counts)


def _choose_station_earliest_finish_from_maps(opt, loads: Dict[int, float], subtask: ResourceSubtask, rng=None) -> int:
    choices = []
    for station_id in range(_station_count(opt)):
        projected_finish = float(loads.get(int(station_id), 0.0) + _subtask_station_work(subtask))
        arrival = float(_arrival_proxy(opt, subtask, station_id))
        choices.append((projected_finish + 0.1 * arrival, int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _choose_station_load_balance_from_maps(opt, loads: Dict[int, float], subtask: ResourceSubtask, rng=None) -> int:
    choices = []
    for station_id in range(_station_count(opt)):
        projected = dict(loads)
        projected[int(station_id)] = projected.get(int(station_id), 0.0) + _subtask_station_work(subtask)
        vals = list(projected.values()) or [0.0]
        mean = sum(vals) / max(1, len(vals))
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
        choices.append((float(variance), int(station_id)))
    picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
    return int((picked or choices[0])[1])


def _build_y_action_signature(destroy_name: str, repair_name: str, released_subtasks: Dict[int, Tuple[int, int]], assignments: Dict[int, Dict[str, int]]) -> Tuple[object, ...]:
    release_sig = tuple(sorted((int(subtask_id), int(station_id), int(rank)) for subtask_id, (station_id, rank) in (released_subtasks or {}).items()))
    assign_sig = tuple(sorted((int(subtask_id), int(meta["station_id"]), int(meta["station_rank"])) for subtask_id, meta in (assignments or {}).items()))
    return ("Y", str(destroy_name), release_sig, str(repair_name), assign_sig)


def _build_y_rough_features(opt, config: ResourceConfig, released_subtasks: Dict[int, Tuple[int, int]], assignments: Dict[int, Dict[str, int]]) -> Dict[str, float]:
    released_ids = [int(x) for x in (released_subtasks or {}).keys()]
    base_loads, base_counts = _y_base_maps(config, released_ids)
    old_loads = _station_loads(config)
    old_std = _station_load_std_from_map(old_loads)
    new_loads = dict(base_loads)
    old_arrival = 0.0
    new_arrival = 0.0
    old_rank_sum = 0.0
    new_rank_sum = 0.0
    for subtask_id, meta in (assignments or {}).items():
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        work = float(_subtask_station_work(row))
        target_station = int(meta["station_id"])
        new_loads[target_station] = new_loads.get(target_station, 0.0) + work
        new_arrival += float(_arrival_proxy(opt, row, target_station))
        new_rank_sum += float(meta["station_rank"])
        origin_station = int(released_subtasks.get(int(subtask_id), (-1, -1))[0])
        origin_rank = int(released_subtasks.get(int(subtask_id), (-1, -1))[1])
        if origin_station >= 0:
            old_arrival += float(_arrival_proxy(opt, row, origin_station))
        if origin_rank >= 0:
            old_rank_sum += float(origin_rank)
    new_std = _station_load_std_from_map(new_loads)
    count = max(1, len(assignments))
    sy_delta = float((new_std - old_std) / max(1.0, float(getattr(opt, "work_z", 1.0) or 1.0)))
    sy_delta += 0.01 * float((new_arrival - old_arrival) / count)
    sy_delta += 0.02 * float((new_rank_sum - old_rank_sum) / count)
    return {
        "sy_delta": float(sy_delta),
        "affected_count": float(len(assignments)),
        "load_std_before": float(old_std),
        "load_std_after": float(new_std),
    }


def _plan_y_assignments(opt, config: ResourceConfig, released_subtasks: Dict[int, Tuple[int, int]], strategy: str, rng=None) -> Dict[str, object]:
    released_ids = [int(x) for x in (released_subtasks or {}).keys()]
    if not released_ids:
        return {"success": False}
    loads, counts = _y_base_maps(config, released_ids)
    assignments: Dict[int, Dict[str, int]] = {}
    remaining = list(released_ids)
    if str(strategy) == "y_repair_regret2_station":
        while remaining:
            best = None
            for subtask_id in list(remaining):
                row = config.subtasks.get(int(subtask_id))
                if row is None:
                    continue
                choices = []
                for station_id in range(_station_count(opt)):
                    projected_finish = float(loads.get(int(station_id), 0.0) + _subtask_station_work(row))
                    arrival = float(_arrival_proxy(opt, row, station_id))
                    choices.append((projected_finish + 0.1 * arrival, int(station_id)))
                if not choices:
                    continue
                ranked_choices = sorted(choices, key=lambda item: (item[0], item[1]))
                regret = float(ranked_choices[1][0] - ranked_choices[0][0]) if len(ranked_choices) >= 2 else float(ranked_choices[0][0])
                candidate = {
                    "regret_score": -regret,
                    "best_cost": float(ranked_choices[0][0]),
                    "subtask_id": int(subtask_id),
                    "station_choice": pick_soft_greedy_min(rng, ranked_choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1]))),
                }
                if best is None or (float(candidate["regret_score"]), float(candidate["best_cost"]), int(candidate["subtask_id"])) < (
                    float(best["regret_score"]),
                    float(best["best_cost"]),
                    int(best["subtask_id"]),
                ):
                    best = candidate
            if best is None:
                break
            subtask_id = int(best["subtask_id"])
            station_id = int(best["station_choice"][1])
            assignments[int(subtask_id)] = {"station_id": int(station_id), "station_rank": int(counts.get(int(station_id), 0))}
            row = config.subtasks.get(int(subtask_id))
            if row is not None:
                loads[int(station_id)] = loads.get(int(station_id), 0.0) + _subtask_station_work(row)
            counts[int(station_id)] = counts.get(int(station_id), 0) + 1
            remaining.remove(int(subtask_id))
    else:
        chooser = {
            "y_repair_earliest_finish": lambda row: _choose_station_earliest_finish_from_maps(opt, loads, row, rng),
            "y_repair_arrival_aware_rank": lambda row: int(
                (
                    pick_soft_greedy_min(
                        rng,
                        [
                            (float(_arrival_proxy(opt, row, station_id)) + 0.1 * float(loads.get(int(station_id), 0.0)), int(station_id))
                            for station_id in range(_station_count(opt))
                        ],
                        opt.cfg,
                        score_getter=lambda item: (float(item[0]), int(item[1])),
                    )
                    or (0.0, 0)
                )[1]
            ),
            "y_repair_load_balance": lambda row: _choose_station_load_balance_from_maps(opt, loads, row, rng),
        }.get(str(strategy), lambda row: _choose_station_earliest_finish_from_maps(opt, loads, row, rng))
        for subtask_id in released_ids:
            row = config.subtasks.get(int(subtask_id))
            if row is None:
                continue
            station_id = int(chooser(row))
            assignments[int(subtask_id)] = {"station_id": int(station_id), "station_rank": int(counts.get(int(station_id), 0))}
            loads[int(station_id)] = loads.get(int(station_id), 0.0) + _subtask_station_work(row)
            counts[int(station_id)] = counts.get(int(station_id), 0) + 1
    if not assignments:
        return {"success": False}
    return {"success": True, "assignments": assignments}


def plan_y_candidate(opt, config: ResourceConfig, destroy_name: str, repair_name: str, rng, degree: int) -> Dict[str, object]:
    destroy_planners = {
        "y_destroy_congested_station_block": y_plan_destroy_congested_station_block,
        "y_destroy_cross_station_fragment": y_plan_destroy_cross_station_fragment,
        "y_destroy_rank_window_release": y_plan_destroy_rank_window_release,
        "y_destroy_load_skew_release": y_plan_destroy_load_skew_release,
    }
    destroy_ctx = destroy_planners[str(destroy_name)](opt, config, rng, degree)
    if not bool(destroy_ctx.get("success", False)):
        return {"success": False}
    repair_plan = _plan_y_assignments(opt, config, destroy_ctx.get("released_subtasks", {}), str(repair_name), rng)
    fallback_used = False
    if not bool(repair_plan.get("success", False)):
        repair_plan = _plan_y_assignments(opt, config, destroy_ctx.get("released_subtasks", {}), "y_repair_earliest_finish", rng)
        fallback_used = bool(repair_plan.get("success", False))
    if not bool(repair_plan.get("success", False)):
        return {"success": False}
    assignments = dict(repair_plan.get("assignments", {}) or {})
    return {
        "success": True,
        "destroy_ctx": destroy_ctx,
        "assignments": assignments,
        "fallback_used": bool(fallback_used),
        "action_signature": _build_y_action_signature(str(destroy_name), str(repair_name), destroy_ctx.get("released_subtasks", {}), assignments),
        "rough_features": _build_y_rough_features(opt, config, destroy_ctx.get("released_subtasks", {}), assignments),
    }


def apply_exact_y_plan(opt, config: ResourceConfig, plan: Dict[str, object], rng=None) -> Dict[str, object]:
    del rng
    destroy_ctx = dict(plan.get("destroy_ctx", {}) or {})
    assignments = dict(plan.get("assignments", {}) or {})
    released_subtasks = dict(destroy_ctx.get("released_subtasks", {}) or {})
    if not released_subtasks or not assignments:
        return {"success": False}
    source_station_ids = [int(station_id) for station_id, _ in released_subtasks.values() if int(station_id) >= 0]
    target_station_ids = [int(meta.get("station_id", -1)) for meta in assignments.values() if int(meta.get("station_id", -1)) >= 0]
    touched_station_ids = sorted(set(source_station_ids + target_station_ids))
    touched_subtask_ids = set(int(released_id) for released_id in released_subtasks.keys())
    for row in config.subtasks.values():
        if int(row.station_id) in set(touched_station_ids):
            touched_subtask_ids.add(int(row.subtask_id))
    candidate = config.clone_for_layer("Y", sorted(touched_subtask_ids))
    for subtask_id in released_subtasks.keys():
        row = candidate.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = -1
        row.station_rank = -1
    for subtask_id, meta in assignments.items():
        row = candidate.subtasks.get(int(subtask_id))
        if row is None:
            continue
        row.station_id = int(meta["station_id"])
        row.station_rank = int(meta["station_rank"])
    _normalize_station_ranks_subset(candidate, touched_station_ids)
    return {
        "success": True,
        "config": candidate,
        "score_cache": None,
        "affected_ids": set(int(x) for x in touched_subtask_ids),
        "fallback_used": bool(plan.get("fallback_used", False)),
        "projection_mode": "",
        "projection_repaired_subtask_count": 0,
        "validation_signature": candidate.validation_signature(),
    }


def y_repair_earliest_finish(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        _assign_station_rank(config, int(subtask_id), _choose_station_earliest_finish(opt, config, row, rng))
    _normalize_ranks(config)
    return {"success": True, "affected_subtask_ids": set(released)}


def y_repair_regret2_station(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    remaining = set(released)
    affected = set()
    while remaining:
        best_row = None
        for subtask_id in list(remaining):
            row = config.subtasks.get(int(subtask_id))
            if row is None:
                remaining.remove(int(subtask_id))
                continue
            choices = []
            for station_id in range(_station_count(opt)):
                loads = _station_loads(config)
                projected_finish = float(loads.get(int(station_id), 0.0) + _subtask_station_work(row))
                arrival = float(_arrival_proxy(opt, row, station_id))
                choices.append((projected_finish + 0.1 * arrival, int(station_id)))
            if not choices:
                continue
            ranked_choices = sorted(choices, key=lambda item: (item[0], item[1]))
            regret = float(ranked_choices[1][0] - ranked_choices[0][0]) if len(ranked_choices) >= 2 else float(ranked_choices[0][0])
            station_choice = pick_soft_greedy_min(rng, ranked_choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
            candidate = {
                "regret_score": -regret,
                "best_cost": float(ranked_choices[0][0]),
                "subtask_id": int(subtask_id),
                "station_choice": station_choice or ranked_choices[0],
            }
            if best_row is None or (float(candidate["regret_score"]), float(candidate["best_cost"]), int(candidate["subtask_id"])) < (
                float(best_row["regret_score"]),
                float(best_row["best_cost"]),
                int(best_row["subtask_id"]),
            ):
                best_row = candidate
        if best_row is None:
            break
        chosen_subtask_id = int(best_row["subtask_id"])
        chosen_station_id = int(best_row["station_choice"][1])
        _assign_station_rank(config, int(chosen_subtask_id), int(chosen_station_id))
        affected.add(int(chosen_subtask_id))
        remaining.remove(int(chosen_subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_arrival_aware_rank(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    affected = set()
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        choices = []
        for station_id in range(_station_count(opt)):
            arrival = float(_arrival_proxy(opt, row, station_id))
            load = float(_station_loads(config).get(int(station_id), 0.0))
            choices.append((arrival + 0.1 * load, int(station_id)))
        picked = pick_soft_greedy_min(rng, choices, opt.cfg, score_getter=lambda item: (float(item[0]), int(item[1])))
        _assign_station_rank(config, int(subtask_id), int((picked or choices[0])[1]))
        affected.add(int(subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_load_balance(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    affected = set()
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        _assign_station_rank(config, int(subtask_id), _choose_station_load_balance(opt, config, row, rng))
        affected.add(int(subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


def y_repair_greedy_fallback(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    released = [int(x) for x in (ctx.get("released_subtasks", {}) or {}).keys()]
    if not released:
        return {"success": False}
    affected = set()
    for subtask_id in released:
        row = config.subtasks.get(int(subtask_id))
        if row is None:
            continue
        station_id = _choose_station_earliest_finish(opt, config, row, rng)
        _assign_station_rank(config, int(subtask_id), int(station_id))
        affected.add(int(subtask_id))
    _normalize_ranks(config)
    return {"success": bool(affected), "affected_subtask_ids": affected}


Y_DESTROY_OPERATORS = {
    "y_destroy_congested_station_block": y_destroy_congested_station_block,
    "y_destroy_cross_station_fragment": y_destroy_cross_station_fragment,
    "y_destroy_rank_window_release": y_destroy_rank_window_release,
    "y_destroy_load_skew_release": y_destroy_load_skew_release,
}

Y_REPAIR_OPERATORS = {
    "y_repair_earliest_finish": y_repair_earliest_finish,
    "y_repair_regret2_station": y_repair_regret2_station,
    "y_repair_arrival_aware_rank": y_repair_arrival_aware_rank,
    "y_repair_load_balance": y_repair_load_balance,
}

Y_FALLBACK_OPERATOR = "y_repair_greedy_fallback"
