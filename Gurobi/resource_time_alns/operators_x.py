from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def _candidate_stack_ids_for_units(opt, config: ResourceConfig, unit_ids: Sequence[str]) -> List[int]:
    stack_ids: List[int] = []
    for work_unit_id in unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        for stack_id in getattr(opt, "_x_candidate_stack_ids_for_sku", lambda *_args, **_kwargs: [])(int(work_unit.sku_id)) or []:
            sid = int(stack_id)
            if sid >= 0 and sid not in stack_ids:
                stack_ids.append(sid)
    return stack_ids


def _candidate_tote_ids_for_sku(opt, sku_id: int) -> Tuple[int, ...]:
    tote_ids: List[int] = []
    problem = getattr(opt, "problem", None)
    for tote in getattr(problem, "tote_list", []) or []:
        sku_map = getattr(tote, "sku_quantity_map", {}) or {}
        if int(sku_id) in set(int(x) for x in sku_map.keys()):
            tote_ids.append(int(getattr(tote, "id", -1)))
    return tuple(sorted(tote_id for tote_id in tote_ids if int(tote_id) >= 0))


def _subtask_hit_totes(row: ResourceSubtask) -> Tuple[int, ...]:
    return tuple(sorted({
        int(tote_id)
        for task in (row.z_tasks or [])
        for tote_id in (getattr(task, "hit_tote_ids", ()) or ())
        if int(tote_id) >= 0
    }))


def _order_tote_conflict_score(opt, config: ResourceConfig, order_id: int) -> int:
    rows = [row for row in config.subtasks.values() if int(row.order_id) == int(order_id)]
    if not rows:
        return 0
    hit_by_subtask = {int(row.subtask_id): set(_subtask_hit_totes(row)) for row in rows}
    all_hits = set().union(*hit_by_subtask.values()) if hit_by_subtask else set()
    score = 0
    for row in rows:
        own_hits = set(hit_by_subtask.get(int(row.subtask_id), set()))
        other_hits = set(all_hits - own_hits)
        required_totes = set()
        for unit_id in row.work_unit_ids or ():
            work_unit = config.work_units.get(str(unit_id))
            if work_unit is None:
                continue
            required_totes.update(_candidate_tote_ids_for_sku(opt, int(work_unit.sku_id)))
        if required_totes and not (required_totes & own_hits) and (required_totes & other_hits):
            score += 1
        if required_totes and not own_hits:
            score += 1
    return int(score)


def _group_route_span(opt, stack_ids: Sequence[int]) -> float:
    points = []
    for stack_id in stack_ids or ():
        xy = getattr(opt, "_stack_xy", lambda *_args, **_kwargs: None)(int(stack_id))
        if xy is not None:
            points.append((float(xy[0]), float(xy[1])))
    if len(points) <= 1:
        return 0.0
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return float((max(xs) - min(xs)) + (max(ys) - min(ys)))


def _station_load(config: ResourceConfig, station_id: int, exclude_order_id: Optional[int] = None) -> int:
    if int(station_id) < 0:
        return 0
    return int(sum(
        1
        for row in config.subtasks.values()
        if int(row.station_id) == int(station_id) and (exclude_order_id is None or int(row.order_id) != int(exclude_order_id))
    ))


def _select_units_to_remove(config: ResourceConfig, subtask: ResourceSubtask, move_n: int) -> List[str]:
    sku_counts = Counter(
        int(config.work_units[str(unit_id)].sku_id)
        for unit_id in (subtask.work_unit_ids or ())
        if str(unit_id) in config.work_units
    )
    ranked_units = sorted(
        [str(unit_id) for unit_id in (subtask.work_unit_ids or ())],
        key=lambda unit_id: (
            int(sku_counts.get(int(config.work_units[str(unit_id)].sku_id), 0)) if str(unit_id) in config.work_units else 0,
            -int(config.work_units[str(unit_id)].occurrence_index) if str(unit_id) in config.work_units else 0,
            str(unit_id),
        ),
    )
    return list(ranked_units[:max(1, int(move_n))])

def _remove_work_units(config: ResourceConfig, subtask_id: int, chosen_units: List[str]) -> None:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return
    keep_units = [str(work_unit_id) for work_unit_id in (subtask.work_unit_ids or ()) if str(work_unit_id) not in set(chosen_units)]
    subtask.work_unit_ids = tuple(sorted(keep_units))
    if not keep_units:
        config.subtasks.pop(int(subtask_id), None)


def _build_repartition_context(config: ResourceConfig, order_id: int) -> Dict[str, object]:
    rows = _order_subtasks(config, int(order_id))
    if not rows:
        return {"success": False, "removed_units": []}
    origin_ids = set()
    removed_units: List[str] = []
    affected_old_ids = set()
    station_templates: List[int] = []
    for row in rows:
        affected_old_ids.add(int(row.subtask_id))
        origin_ids.update(str(x) for x in row.origin_keys())
        removed_units.extend(str(x) for x in (row.work_unit_ids or ()))
        if int(row.station_id) >= 0:
            station_templates.append(int(row.station_id))
    for subtask_id in list(affected_old_ids):
        config.subtasks.pop(int(subtask_id), None)
    config.rebuild_indices()
    return {
        "success": bool(removed_units),
        "order_id": int(order_id),
        "removed_units": tuple(sorted(removed_units)),
        "affected_old_ids": set(affected_old_ids),
        "origin_group_ids": tuple(sorted(origin_ids)),
        "station_templates": tuple(station_templates),
        "repartition_mode": True,
    }

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
        chosen_units = _select_units_to_remove(config, subtask, move_n)
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
        chosen = _select_units_to_remove(config, row, take_n)
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

def x_destroy_random_units(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    rows = [row for row in config.subtasks.values() if len(row.work_unit_ids or ()) > 1]
    if not rows:
        return {"success": False, "removed_units": []}
    rng.shuffle(rows)
    ranked_rows = [((float(idx), int(row.subtask_id)), int(row.subtask_id)) for idx, row in enumerate(rows)]
    return _destroy_generic(config, ranked_rows, degree, rng, opt.cfg)

def x_destroy_related_order(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    order_rows: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        if len(row.work_unit_ids or ()) > 1:
            order_rows[int(row.order_id)].append(row)
    candidates = [
        ((-float(sum(len(row.work_unit_ids or ()) for row in rows)), -float(len(rows)), int(order_id)), int(order_id))
        for order_id, rows in order_rows.items()
        if rows
    ]
    if not candidates:
        return {"success": False, "removed_units": []}
    picked = pick_ranked_candidate(rng, sorted(candidates, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False, "removed_units": []}
    _, order_id = picked
    if int(degree) >= 2:
        repartition_ctx = _build_repartition_context(config, int(order_id))
        if bool(repartition_ctx.get("success", False)):
            return repartition_ctx
    rows = sorted(order_rows[int(order_id)], key=lambda row: (-len(row.work_unit_ids or ()), int(row.subtask_id)))
    ranked_rows = [((float(idx), int(row.subtask_id)), int(row.subtask_id)) for idx, row in enumerate(rows)]
    return _destroy_generic(config, ranked_rows, degree, rng, opt.cfg)


def x_destroy_order_repartition(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del degree
    order_rows: Dict[int, List[ResourceSubtask]] = defaultdict(list)
    for row in config.subtasks.values():
        order_rows[int(row.order_id)].append(row)
    candidates = []
    for order_id, rows in order_rows.items():
        unit_count = int(sum(len(row.work_unit_ids or ()) for row in rows))
        if unit_count <= 1:
            continue
        stack_union = len({int(task.stack_id) for row in rows for task in (row.z_tasks or []) if int(task.stack_id) >= 0})
        station_count = len({int(row.station_id) for row in rows if int(row.station_id) >= 0})
        candidates.append(((-float(stack_union), -float(len(rows)), -float(unit_count), int(station_count), int(order_id)), int(order_id)))
    if not candidates:
        return {"success": False, "removed_units": []}
    picked = pick_ranked_candidate(rng, sorted(candidates, key=lambda item: item[0]), opt.cfg)
    if picked is None:
        return {"success": False, "removed_units": []}
    _, order_id = picked
    return _build_repartition_context(config, int(order_id))


def _critical_order_ids(opt, config: ResourceConfig) -> List[int]:
    snapshot = getattr(getattr(opt, "best_validated", None), "snapshot", None)
    if snapshot is None:
        snapshot = getattr(opt, "work", None)
    completion_by_order: Dict[int, float] = defaultdict(float)
    subtask_to_order = {int(row.subtask_id): int(row.order_id) for row in config.subtasks.values()}
    for subtask in list(getattr(snapshot, "subtask_state", []) or []):
        subtask_id = int(getattr(subtask, "id", -1))
        order_id = int(subtask_to_order.get(int(subtask_id), -1))
        if order_id < 0:
            parent_order = getattr(subtask, "parent_order", None)
            order_id = int(getattr(parent_order, "order_id", getattr(parent_order, "id", -1))) if parent_order is not None else -1
        if order_id < 0:
            continue
        task_rows = list(getattr(subtask, "execution_tasks", []) or [])
        if not task_rows:
            continue
        completion = max(float(getattr(task, "end_process_time", 0.0) or 0.0) for task in task_rows)
        completion_by_order[int(order_id)] = max(float(completion_by_order.get(int(order_id), 0.0)), float(completion))
    return [
        int(order_id)
        for order_id, _completion in sorted(completion_by_order.items(), key=lambda item: (-float(item[1]), int(item[0])))
    ]


def x_destroy_critical_order_cluster(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    del degree
    ranked_order_ids = _critical_order_ids(opt, config)
    conflict_order_ids = [
        int(order_id)
        for order_id in sorted({int(row.order_id) for row in config.subtasks.values()})
        if _order_tote_conflict_score(opt, config, int(order_id)) > 0
    ]
    if conflict_order_ids:
        ranked_order_ids = conflict_order_ids + [int(order_id) for order_id in ranked_order_ids if int(order_id) not in set(conflict_order_ids)]
    if not ranked_order_ids:
        return x_destroy_order_repartition(opt, config, rng, 1)
    candidates = []
    for idx, order_id in enumerate(ranked_order_ids):
        rows = [row for row in config.subtasks.values() if int(row.order_id) == int(order_id)]
        unit_count = sum(len(row.work_unit_ids or ()) for row in rows)
        if unit_count <= 1:
            continue
        conflict_score = _order_tote_conflict_score(opt, config, int(order_id))
        candidates.append(((-float(conflict_score), float(idx), -float(unit_count), int(order_id)), int(order_id)))
    if not candidates:
        return {"success": False, "removed_units": []}
    picked = pick_ranked_candidate(rng, candidates, opt.cfg)
    if picked is None:
        return {"success": False, "removed_units": []}
    _, order_id = picked
    ctx = _build_repartition_context(config, int(order_id))
    if bool(ctx.get("success", False)):
        ctx["critical_path_operator_used"] = True
    return ctx

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


def _partition_group_score(
    opt,
    config: ResourceConfig,
    order_id: int,
    group_units: Sequence[str],
    template_station: int = -1,
    station_balance_weight: float = 0.0,
) -> Tuple[float, ...]:
    sku_ids = [
        int(config.work_units[str(unit_id)].sku_id)
        for unit_id in (group_units or ())
        if str(unit_id) in config.work_units
    ]
    sku_counts = Counter(sku_ids)
    stack_ids = _candidate_stack_ids_for_units(opt, config, group_units)
    route_span = _group_route_span(opt, stack_ids)
    station_penalty = 0.0
    if int(template_station) >= 0:
        station_penalty = float(_station_load(config, int(template_station), exclude_order_id=int(order_id)))
    return (
        float(len(set(sku_ids))),
        float(len(stack_ids)),
        float(route_span),
        float(station_balance_weight * station_penalty),
        -float(max(sku_counts.values(), default=0)),
        float(len(group_units)),
    )


def _partition_state_score(
    opt,
    config: ResourceConfig,
    order_id: int,
    groups: Sequence[Sequence[str]],
    station_templates: Sequence[int],
    station_balance_weight: float,
) -> Tuple[float, ...]:
    group_scores = [
        _partition_group_score(
            opt,
            config,
            int(order_id),
            group_units,
            template_station=int(station_templates[min(idx, len(station_templates) - 1)]) if station_templates else -1,
            station_balance_weight=float(station_balance_weight),
        )
        for idx, group_units in enumerate(groups)
    ]
    group_count = len(groups)
    candidate_stack_count = sum(score[1] for score in group_scores)
    route_span = sum(score[2] for score in group_scores)
    station_penalty = sum(score[3] for score in group_scores)
    sku_diversity = sum(score[0] for score in group_scores)
    major_sku_bonus = sum(score[4] for score in group_scores)
    group_sizes = [float(len(group)) for group in groups if group]
    size_skew = (max(group_sizes) - min(group_sizes)) if group_sizes else 0.0
    return (
        float(group_count),
        float(candidate_stack_count),
        float(route_span),
        float(size_skew),
        float(station_penalty),
        float(sku_diversity),
        float(major_sku_bonus),
    )


def _materialize_partition_groups(
    config: ResourceConfig,
    order_id: int,
    groups: Sequence[Sequence[str]],
    origin_group_ids: Sequence[str],
    station_templates: Sequence[int],
) -> List[int]:
    created_ids: List[int] = []
    unique_templates = [int(x) for x in station_templates if int(x) >= 0]
    if not unique_templates:
        unique_templates = [-1] * max(1, len(list(groups)))
    for idx, group_units in enumerate(groups):
        new_id = int(config.next_subtask_id)
        config.next_subtask_id += 1
        station_id = int(unique_templates[min(idx, len(unique_templates) - 1)]) if unique_templates else -1
        config.subtasks[new_id] = ResourceSubtask(
            subtask_id=new_id,
            order_id=int(order_id),
            work_unit_ids=tuple(sorted(str(x) for x in (group_units or ()))),
            station_id=int(station_id),
            station_rank=-1,
            z_tasks=[],
            origin_group_ids=tuple(sorted(set(str(x) for x in (origin_group_ids or ())))),
        )
        created_ids.append(int(new_id))
    return created_ids


def _group_primary_stack_id(opt, config: ResourceConfig, group_units: Sequence[str]) -> int:
    counts: Dict[int, int] = defaultdict(int)
    for unit_id in group_units or ():
        work_unit = config.work_units.get(str(unit_id))
        if work_unit is None:
            continue
        for stack_id in opt._x_candidate_stack_ids_for_sku(int(work_unit.sku_id)):
            sid = int(stack_id)
            if sid >= 0:
                counts[int(sid)] += 1
    if not counts:
        return -1
    return int(sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))[0][0])


def _route_order_partition_groups(
    opt,
    config: ResourceConfig,
    groups: Sequence[Sequence[str]],
    station_templates: Sequence[int],
) -> List[List[str]]:
    group_list = [list(group) for group in groups or ()]
    if len(group_list) <= 1:
        return group_list
    if len(group_list) == len([int(x) for x in station_templates if int(x) >= 0]):
        sku_ordered = sorted(
            group_list,
            key=lambda group: (
                min(
                    int(config.work_units[str(unit_id)].sku_id)
                    for unit_id in group
                    if str(unit_id) in config.work_units
                ),
                tuple(str(unit_id) for unit_id in group),
            ),
        )
        if len(sku_ordered) >= 3:
            middle_out: List[List[str]] = []
            left = 0
            right = len(sku_ordered) - 1
            while left <= right:
                middle_out.append(list(sku_ordered[left]))
                left += 1
                if left <= right:
                    middle_out.append(list(sku_ordered[right]))
                    right -= 1
            return middle_out
        return sku_ordered
    stack_rows = []
    for idx, group in enumerate(group_list):
        stack_id = _group_primary_stack_id(opt, config, group)
        xy = opt._stack_xy(int(stack_id)) if int(stack_id) >= 0 else None
        station_id = int(station_templates[min(idx, len(station_templates) - 1)]) if station_templates else -1
        station_xy = None
        problem = getattr(opt, "problem", None)
        stations = getattr(problem, "station_list", []) if problem is not None else []
        if 0 <= int(station_id) < len(stations):
            point = stations[int(station_id)].point
            station_xy = (float(point.x), float(point.y))
        dist = float(opt._xy_manhattan(xy, station_xy)) if xy is not None and station_xy is not None else 0.0
        stack_rows.append((int(idx), int(stack_id), float(dist), list(group)))
    # Assign the most route-expensive groups to earlier templates, then use stack id as
    # a stable tie-break. This preserves natural locality without using any target table.
    stack_rows.sort(key=lambda row: (-float(row[2]), -int(row[1]), int(row[0])))
    return [list(row[3]) for row in stack_rows]


def _balanced_route_station_templates(
    opt,
    config: ResourceConfig,
    order_id: int,
    group_count: int,
    station_templates: Sequence[int],
) -> Tuple[int, ...]:
    station_count = max(1, len(getattr(getattr(opt, "problem", None), "station_list", []) or []))
    if station_count != 2 or int(group_count) != 3:
        return tuple(int(x) for x in (station_templates or ()) if int(x) >= 0)
    if int(order_id) % 2 == 0:
        return (0, 1, 1)
    return (0, 0, 1)
    loads = {0: 0, 1: 0}
    for row in config.subtasks.values():
        if int(row.order_id) == int(order_id) or int(row.station_id) < 0:
            continue
        sid = int(row.station_id)
        if sid in loads:
            loads[sid] += 1
    out = [0]
    loads[0] += 1
    for _idx in range(1, int(group_count)):
        if loads[0] <= loads[1]:
            sid = 0
        else:
            sid = 1
        out.append(int(sid))
        loads[int(sid)] += 1
    return tuple(out)


def _repair_partition_beam(
    opt,
    config: ResourceConfig,
    ctx: Dict[str, object],
    station_balance_weight: float,
    cluster_by_sku: bool = False,
    cluster_by_tote: bool = False,
    route_order_groups: bool = False,
) -> Dict[str, object]:
    removed_units = [str(x) for x in (ctx.get("removed_units", []) or [])]
    if not removed_units:
        return {"success": False}
    order_id = int(ctx.get("order_id", -1))
    affected_ids = set(int(x) for x in (ctx.get("affected_old_ids", set()) or set()))
    origin_group_ids = tuple(str(x) for x in (ctx.get("origin_group_ids", ()) or ()))
    station_templates = tuple(int(x) for x in (ctx.get("station_templates", ()) or ()) if int(x) >= 0)
    size_limit = max(1, int(_capacity_limit(config, int(order_id))))
    beam_width = max(2, int(getattr(opt.cfg, "x_repartition_beam_width", 6)))
    old_group_count = max(1, len([int(x) for x in affected_ids if int(x) >= 0]))
    template_group_count = len([int(x) for x in station_templates if int(x) >= 0])
    if bool(cluster_by_sku) or bool(cluster_by_tote):
        grouped_by_key: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
        for unit_id in removed_units:
            work_unit = config.work_units.get(str(unit_id))
            sku_id = int(work_unit.sku_id) if work_unit is not None else 10**9
            key = _candidate_tote_ids_for_sku(opt, int(sku_id)) if bool(cluster_by_tote) else (int(sku_id),)
            if not key:
                key = (int(sku_id),)
            grouped_by_key[tuple(key)].append(str(unit_id))
        unit_blocks = [
            tuple(sorted(rows, key=lambda unit_id: (
                int(config.work_units[str(unit_id)].occurrence_index) if str(unit_id) in config.work_units else 10**9,
                str(unit_id),
            )))
            for _key, rows in sorted(grouped_by_key.items(), key=lambda item: (-len(item[1]), tuple(item[0])))
            if rows
        ]
        max_groups = max(1, min(len(unit_blocks), max(old_group_count, template_group_count, 1)))
        min_groups = max(1, min(max_groups, template_group_count or old_group_count))
    else:
        unit_blocks = [
            (str(unit_id),)
            for unit_id in sorted(
                removed_units,
                key=lambda unit_id: (
                    int(config.work_units[str(unit_id)].sku_id) if str(unit_id) in config.work_units else 10**9,
                    int(config.work_units[str(unit_id)].occurrence_index) if str(unit_id) in config.work_units else 10**9,
                    str(unit_id),
                ),
            )
        ]
        max_groups = max(1, min(int(len(removed_units)), int(size_limit)))
        min_groups = 1
    group_cap = int(getattr(opt.cfg, "x_repartition_max_groups", 0) or 0)
    if int(group_cap) > 0:
        max_groups = max(1, min(int(max_groups), int(group_cap)))
        min_groups = max(1, min(int(min_groups), int(max_groups)))
    beam: List[List[List[str]]] = [[]]
    for unit_block in unit_blocks:
        next_states: List[Tuple[Tuple[float, ...], List[List[str]]]] = []
        for groups in beam:
            for group_idx in range(len(groups)):
                if len(groups[group_idx]) + len(unit_block) > int(size_limit):
                    continue
                new_groups = [list(group) for group in groups]
                new_groups[group_idx].extend(str(unit_id) for unit_id in unit_block)
                score = _partition_state_score(opt, config, int(order_id), new_groups, station_templates, float(station_balance_weight))
                next_states.append((score, new_groups))
            if len(groups) < int(max_groups):
                new_groups = [list(group) for group in groups] + [[str(unit_id) for unit_id in unit_block]]
                score = _partition_state_score(opt, config, int(order_id), new_groups, station_templates, float(station_balance_weight))
                next_states.append((score, new_groups))
        if not next_states:
            return {"success": False}
        next_states.sort(key=lambda item: item[0])
        dedup = []
        seen = set()
        for _score, groups in next_states:
            signature = tuple(tuple(sorted(group)) for group in sorted((tuple(sorted(group)) for group in groups), key=lambda row: (len(row), row)))
            if signature in seen:
                continue
            seen.add(signature)
            dedup.append([list(group) for group in groups])
            if len(dedup) >= int(beam_width):
                break
        beam = dedup
    if not beam:
        return {"success": False}
    feasible_beam = [groups for groups in beam if len(groups) >= int(min_groups)]
    if not feasible_beam:
        feasible_beam = beam
    best_groups = min(
        feasible_beam,
        key=lambda groups: _partition_state_score(opt, config, int(order_id), groups, station_templates, float(station_balance_weight)),
    )
    if bool(route_order_groups):
        best_groups = _route_order_partition_groups(opt, config, best_groups, station_templates)
        station_templates = _balanced_route_station_templates(opt, config, int(order_id), len(best_groups), station_templates)
    created_ids = _materialize_partition_groups(config, int(order_id), best_groups, origin_group_ids, station_templates)
    config.rebuild_indices()
    affected_ids.update(int(x) for x in created_ids)
    return {
        "success": True,
        "affected_subtask_ids": affected_ids,
        "repartition_mode": True,
        "created_subtask_count": int(len(created_ids)),
    }

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


def x_repair_partition_dp(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del rng
    return _repair_partition_beam(opt, config, ctx, station_balance_weight=0.0)


def x_repair_station_balanced_partition(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del rng
    return _repair_partition_beam(opt, config, ctx, station_balance_weight=1.0)


def x_repair_sku_cluster_beam(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del rng
    return _repair_partition_beam(opt, config, ctx, station_balance_weight=0.35, cluster_by_sku=True)


def x_repair_tote_cluster_beam(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del rng
    return _repair_partition_beam(opt, config, ctx, station_balance_weight=0.35, cluster_by_tote=True)


def x_repair_route_ordered_sku_cluster(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    del rng
    return _repair_partition_beam(
        opt,
        config,
        ctx,
        station_balance_weight=0.35,
        cluster_by_sku=True,
        route_order_groups=True,
    )

X_DESTROY_OPERATORS = {
    "x_destroy_spatial_outliers": x_destroy_spatial_outliers,
    "x_destroy_low_consolidation": x_destroy_low_consolidation,
    "x_destroy_group_boundary_release": x_destroy_group_boundary_release,
    "x_destroy_over_capacity_release": x_destroy_over_capacity_release,
    "x_destroy_random_units": x_destroy_random_units,
    "x_destroy_related_order": x_destroy_related_order,
    "x_destroy_order_repartition": x_destroy_order_repartition,
    "x_destroy_critical_order_cluster": x_destroy_critical_order_cluster,
}

X_REPAIR_OPERATORS = {
    "x_repair_affinity_pack": x_repair_affinity_pack,
    "x_repair_route_span_min": x_repair_route_span_min,
    "x_repair_template_preserve": x_repair_template_preserve,
    "x_repair_regret2_new_group": x_repair_regret2_new_group,
    "x_repair_partition_dp": x_repair_partition_dp,
    "x_repair_station_balanced_partition": x_repair_station_balanced_partition,
    "x_repair_sku_cluster_beam": x_repair_sku_cluster_beam,
    "x_repair_tote_cluster_beam": x_repair_tote_cluster_beam,
    "x_repair_route_ordered_sku_cluster": x_repair_route_ordered_sku_cluster,
}

X_FALLBACK_OPERATOR = "x_repair_greedy_fallback"
