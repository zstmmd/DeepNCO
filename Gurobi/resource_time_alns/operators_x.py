from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

from .state import ResourceConfig, ResourceSubtask
from .utils import pick_ranked_candidate


def _order_subtasks(config: ResourceConfig, order_id: int) -> List[ResourceSubtask]:
    return sorted(
        [row for row in config.subtasks.values() if int(row.order_id) == int(order_id)],
        key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)),
    )


def _capacity_limit(config: ResourceConfig, order_id: int) -> int:
    return max(1, int(config.capacity_limits.get(int(order_id), 1)))


def _sku_diversity(config: ResourceConfig, subtask: ResourceSubtask) -> int:
    return len({int(config.work_units[str(work_unit_id)].sku_id) for work_unit_id in (subtask.work_unit_ids or ()) if str(work_unit_id) in config.work_units})


def _stack_span(subtask: ResourceSubtask) -> int:
    return len({int(task.stack_id) for task in (subtask.z_tasks or []) if int(task.stack_id) >= 0})


def _remove_work_units(config: ResourceConfig, subtask_id: int, chosen_units: List[str]) -> None:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return
    keep_units = [str(work_unit_id) for work_unit_id in (subtask.work_unit_ids or ()) if str(work_unit_id) not in set(chosen_units)]
    subtask.work_unit_ids = tuple(sorted(keep_units))
    if not keep_units:
        config.subtasks.pop(int(subtask_id), None)


def _score_insert_affinity(config: ResourceConfig, candidate: ResourceSubtask, work_unit_id: str) -> float:
    target_sku = int(config.work_units[str(work_unit_id)].sku_id)
    sku_counts = Counter(int(config.work_units[str(unit_id)].sku_id) for unit_id in (candidate.work_unit_ids or ()) if str(unit_id) in config.work_units)
    return float(-sku_counts.get(target_sku, 0) + 0.2 * _stack_span(candidate))


def _score_insert_route_span(config: ResourceConfig, candidate: ResourceSubtask, work_unit_id: str) -> float:
    del work_unit_id
    return float(_stack_span(candidate) + 0.5 * _sku_diversity(config, candidate))


def _score_insert_template(config: ResourceConfig, candidate: ResourceSubtask, work_unit_id: str, origin_station: int) -> float:
    del work_unit_id
    return float(0.0 if int(candidate.station_id) == int(origin_station) else 1.0) + 0.1 * float(_stack_span(candidate))


def x_finalize_insert_or_new_group(
    config: ResourceConfig,
    order_id: int,
    work_unit_id: str,
    scorer: Callable[[ResourceConfig, ResourceSubtask, str], float],
    origin_group_ids: Tuple[str, ...],
    origin_station: int = -1,
    prefer_new_group: bool = False,
) -> int:
    candidates = []
    limit = _capacity_limit(config, int(order_id))
    for row in _order_subtasks(config, int(order_id)):
        if str(work_unit_id) in set(str(x) for x in (row.work_unit_ids or ())):
            continue
        if len(row.work_unit_ids) >= int(limit):
            continue
        if scorer is _score_insert_template:
            score = float(_score_insert_template(config, row, work_unit_id, int(origin_station)))
        else:
            score = float(scorer(config, row, work_unit_id))
        candidates.append((score, int(row.subtask_id)))
    if candidates and not bool(prefer_new_group):
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen_id = int(candidates[0][1])
        chosen = config.subtasks[chosen_id]
        chosen.work_unit_ids = tuple(sorted(list(chosen.work_unit_ids) + [str(work_unit_id)]))
        chosen.origin_group_ids = tuple(sorted(set(chosen.origin_group_ids + tuple(origin_group_ids))))
        return int(chosen_id)

    new_id = int(config.next_subtask_id)
    config.next_subtask_id += 1
    config.subtasks[new_id] = ResourceSubtask(
        subtask_id=new_id,
        order_id=int(order_id),
        work_unit_ids=(str(work_unit_id),),
        station_id=-1,
        station_rank=-1,
        z_tasks=[],
        origin_group_ids=tuple(origin_group_ids),
    )
    return int(new_id)


def _destroy_generic(config: ResourceConfig, ranked_rows: List[Tuple[Tuple[float, ...], int]], degree: int, rng, cfg) -> Dict[str, object]:
    budget_remaining = max(1, int(degree))
    removed_units: List[str] = []
    affected_old_ids = set()
    origin_ids = set()
    order_id = -1
    remaining = list(ranked_rows or [])
    while budget_remaining > 0 and remaining:
        live_rows = []
        for score, subtask_id in remaining:
            subtask = config.subtasks.get(int(subtask_id))
            if subtask is None or len(subtask.work_unit_ids) <= 1:
                continue
            live_rows.append((score, int(subtask_id)))
        if not live_rows:
            break
        picked = pick_ranked_candidate(rng, live_rows, cfg)
        if picked is None:
            break
        _, chosen_subtask_id = picked
        subtask = config.subtasks.get(int(chosen_subtask_id))
        if subtask is None or len(subtask.work_unit_ids) <= 1:
            remaining = [row for row in remaining if int(row[1]) != int(chosen_subtask_id)]
            continue
        move_n = max(1, min(int(budget_remaining), len(subtask.work_unit_ids) - 1))
        chosen_units = list(subtask.work_unit_ids[-move_n:])
        order_id = int(subtask.order_id)
        affected_old_ids.add(int(chosen_subtask_id))
        origin_ids.update(str(x) for x in subtask.origin_keys())
        removed_units.extend(chosen_units)
        budget_remaining -= len(chosen_units)
        _remove_work_units(config, int(chosen_subtask_id), chosen_units)
        config.rebuild_indices()
        remaining = [row for row in remaining if int(row[1]) != int(chosen_subtask_id)]
    if removed_units:
        return {
            "success": True,
            "order_id": int(order_id),
            "removed_units": list(removed_units),
            "affected_old_ids": set(affected_old_ids),
            "origin_group_ids": tuple(sorted(origin_ids)),
        }
    return {"success": False, "removed_units": []}


def x_destroy_spatial_outliers(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked_rows = sorted(
        [((float(-_stack_span(row)), float(-len(row.work_unit_ids)), float(-_sku_diversity(config, row))), int(row.subtask_id)) for row in config.subtasks.values()],
        key=lambda item: item[0],
    )
    return _destroy_generic(config, ranked_rows, degree, rng, opt.cfg)


def x_destroy_low_consolidation(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked_rows = sorted(
        [((float(-_sku_diversity(config, row)), float(-len(row.work_unit_ids))), int(row.subtask_id)) for row in config.subtasks.values()],
        key=lambda item: item[0],
    )
    return _destroy_generic(config, ranked_rows, degree, rng, opt.cfg)


def x_destroy_group_boundary_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    orders = defaultdict(list)
    for row in config.subtasks.values():
        orders[int(row.order_id)].append(row)
    candidates = []
    for order_id, rows in orders.items():
        rows = sorted(rows, key=lambda row: int(row.subtask_id))
        for idx in range(len(rows) - 1):
            left = rows[idx]
            right = rows[idx + 1]
            if len(left.work_unit_ids) <= 1 and len(right.work_unit_ids) <= 1:
                continue
            candidates.append(((-len(rows), idx, int(order_id)), int(order_id), int(left.subtask_id), int(right.subtask_id)))
    if not candidates:
        return {"success": False, "removed_units": []}
    picked = pick_ranked_candidate(rng, sorted(candidates, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False, "removed_units": []}
    _, order_id, left_id, right_id = picked
    removed_units = []
    affected_old_ids = set()
    origin_ids = set()
    budget_remaining = max(1, int(degree))
    for subtask_id in [int(left_id), int(right_id)]:
        if budget_remaining <= 0:
            break
        row = config.subtasks.get(int(subtask_id))
        if row is None or len(row.work_unit_ids) <= 1:
            continue
        take_n = max(1, min(int(budget_remaining), len(row.work_unit_ids) - 1))
        chosen = [str(x) for x in row.work_unit_ids[-take_n:]]
        removed_units.extend(chosen)
        affected_old_ids.add(int(row.subtask_id))
        origin_ids.update(str(x) for x in row.origin_keys())
        _remove_work_units(config, int(row.subtask_id), chosen)
        budget_remaining -= len(chosen)
    if removed_units:
        config.rebuild_indices()
        return {
            "success": True,
            "order_id": int(order_id),
            "removed_units": list(removed_units),
            "affected_old_ids": set(affected_old_ids),
            "origin_group_ids": tuple(sorted(origin_ids)),
        }
    return {"success": False, "removed_units": []}


def x_destroy_over_capacity_release(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    ranked_rows = []
    for row in config.subtasks.values():
        limit = _capacity_limit(config, int(row.order_id))
        overflow = max(0, len(row.work_unit_ids) - limit)
        ranked_rows.append(((float(-overflow), float(-len(row.work_unit_ids))), int(row.subtask_id)))
    ranked_rows.sort(key=lambda item: item[0])
    return _destroy_generic(config, ranked_rows, degree, rng, opt.cfg)


def _repair_generic(
    config: ResourceConfig,
    ctx: Dict[str, object],
    scorer: Callable[[ResourceConfig, ResourceSubtask, str], float],
    prefer_new_group: bool = False,
) -> Dict[str, object]:
    removed_units = [str(x) for x in (ctx.get("removed_units", []) or [])]
    if not removed_units:
        return {"success": False}
    order_id = int(ctx.get("order_id", -1))
    origin_group_ids = tuple(str(x) for x in (ctx.get("origin_group_ids", ()) or ()))
    affected_ids = set(int(x) for x in (ctx.get("affected_old_ids", set()) or set()))
    origin_station = int(ctx.get("origin_station", -1))
    for work_unit_id in removed_units:
        chosen_id = x_finalize_insert_or_new_group(
            config=config,
            order_id=order_id,
            work_unit_id=str(work_unit_id),
            scorer=scorer,
            origin_group_ids=origin_group_ids,
            origin_station=origin_station,
            prefer_new_group=bool(prefer_new_group),
        )
        affected_ids.add(int(chosen_id))
    config.rebuild_indices()
    return {"success": True, "affected_subtask_ids": affected_ids}


def x_repair_affinity_pack(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del opt, rng
    return _repair_generic(config, ctx, _score_insert_affinity, prefer_new_group=False)


def x_repair_route_span_min(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del opt, rng
    return _repair_generic(config, ctx, _score_insert_route_span, prefer_new_group=False)


def x_repair_template_preserve(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del rng
    rows = _order_subtasks(config, int(ctx.get("order_id", -1)))
    if rows:
        ctx = copy.deepcopy(ctx)
        ctx["origin_station"] = int(rows[0].station_id)
    return _repair_generic(config, ctx, _score_insert_template, prefer_new_group=False)


def x_repair_regret2_new_group(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del opt, rng
    return _repair_generic(config, ctx, _score_insert_affinity, prefer_new_group=True)


def x_repair_greedy_fallback(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del opt, rng
    return _repair_generic(config, ctx, _score_insert_route_span, prefer_new_group=False)


X_DESTROY_OPERATORS = {
    "x_destroy_spatial_outliers": x_destroy_spatial_outliers,
    "x_destroy_low_consolidation": x_destroy_low_consolidation,
    "x_destroy_group_boundary_release": x_destroy_group_boundary_release,
    "x_destroy_over_capacity_release": x_destroy_over_capacity_release,
}

X_REPAIR_OPERATORS = {
    "x_repair_affinity_pack": x_repair_affinity_pack,
    "x_repair_route_span_min": x_repair_route_span_min,
    "x_repair_template_preserve": x_repair_template_preserve,
    "x_repair_regret2_new_group": x_repair_regret2_new_group,
}

X_FALLBACK_OPERATOR = "x_repair_greedy_fallback"
