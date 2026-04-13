from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config.ofs_config import OFSConfig
from entity.subTask import SubTask
from entity.task import Task

from .state import ResourceConfig, ResourceSubtask, ZTaskDescriptor
from .utils import global_used_totes, pick_ranked_candidate, pick_soft_greedy_min


def _demand_counts(config: ResourceConfig, subtask: ResourceSubtask) -> Dict[int, int]:
    demand: Dict[int, int] = defaultdict(int)
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        demand[int(work_unit.sku_id)] += 1
    return dict(demand)


def _descriptor_to_task(subtask: ResourceSubtask, descriptor: ZTaskDescriptor) -> Task:
    return Task(
        task_id=int(descriptor.task_id),
        sub_task_id=int(subtask.subtask_id),
        target_stack_id=int(descriptor.stack_id),
        target_station_id=int(subtask.station_id),
        operation_mode=str(descriptor.mode).upper(),
        station_sequence_rank=int(subtask.station_rank),
        target_tote_ids=list(int(x) for x in (descriptor.target_tote_ids or ())),
        hit_tote_ids=list(int(x) for x in (descriptor.hit_tote_ids or ())),
        noise_tote_ids=list(int(x) for x in (descriptor.noise_tote_ids or ())),
        sort_layer_range=None if descriptor.sort_layer_range is None else (int(descriptor.sort_layer_range[0]), int(descriptor.sort_layer_range[1])),
        station_service_time=float(descriptor.station_service_time),
        robot_service_time=float(descriptor.robot_service_time),
        sku_pick_count=int(descriptor.sku_pick_count),
    )


def _build_temp_subtask(opt, config: ResourceConfig, subtask: ResourceSubtask, descriptors: Sequence[ZTaskDescriptor]) -> SubTask:
    order_map = {int(getattr(order, "order_id", -1)): order for order in getattr(opt.problem, "order_list", []) or []}
    sku_map = {int(getattr(sku, "id", -1)): sku for sku in getattr(opt.problem, "skus_list", []) or []}
    order_obj = order_map.get(int(subtask.order_id))
    sku_list = [
        sku_map[int(config.work_units[str(work_unit_id)].sku_id)]
        for work_unit_id in (subtask.work_unit_ids or ())
        if str(work_unit_id) in config.work_units and int(config.work_units[str(work_unit_id)].sku_id) in sku_map
    ]
    temp_subtask = SubTask(id=int(subtask.subtask_id), parent_order=order_obj, sku_list=sku_list)
    temp_subtask.assigned_station_id = int(subtask.station_id)
    temp_subtask.station_sequence_rank = int(subtask.station_rank)
    for descriptor in descriptors:
        task = _descriptor_to_task(subtask, descriptor)
        stack = opt.problem.point_to_stack.get(int(task.target_stack_id))
        if stack is not None:
            temp_subtask.add_execution_detail(task, stack)
    return temp_subtask


def _coverage_gain(opt, remaining: Dict[int, int], hit_tote_ids: Sequence[int]) -> int:
    gain = 0
    local = dict(remaining)
    for tote_id in hit_tote_ids or ():
        tote = getattr(opt.problem, "id_to_tote", {}).get(int(tote_id))
        if tote is None:
            continue
        for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
            sku_id = int(sku_id)
            use = min(int(local.get(sku_id, 0)), int(qty))
            if use <= 0:
                continue
            local[sku_id] = int(local.get(sku_id, 0)) - int(use)
            gain += int(use)
    return int(gain)


def _consume_coverage(opt, remaining: Dict[int, int], hit_tote_ids: Sequence[int]) -> Dict[int, int]:
    updated = dict(remaining)
    for tote_id in hit_tote_ids or ():
        tote = getattr(opt.problem, "id_to_tote", {}).get(int(tote_id))
        if tote is None:
            continue
        for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
            sku_id = int(sku_id)
            use = min(int(updated.get(sku_id, 0)), int(qty))
            if use <= 0:
                continue
            updated[sku_id] = int(updated.get(sku_id, 0)) - int(use)
    return updated


def _candidate_centroid_xy(opt, config: ResourceConfig, subtask: ResourceSubtask) -> Optional[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    seen: Set[Tuple[float, float]] = set()
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        for stack_id in opt._x_candidate_stack_ids_for_sku(int(work_unit.sku_id)):
            xy = opt._stack_xy(int(stack_id))
            if xy is None or xy in seen:
                continue
            seen.add(xy)
            points.append(xy)
    if not points:
        return None
    return (
        float(sum(pt[0] for pt in points) / len(points)),
        float(sum(pt[1] for pt in points) / len(points)),
    )


def _candidate_stack_ids(opt, config: ResourceConfig, subtask: ResourceSubtask, seed_stack_ids: Optional[Sequence[int]] = None) -> List[int]:
    primary_stack_ids: List[int] = []
    extra_stack_ids: List[int] = []
    for stack_id in seed_stack_ids or ():
        if int(stack_id) >= 0 and int(stack_id) not in primary_stack_ids:
            primary_stack_ids.append(int(stack_id))
    for descriptor in subtask.z_tasks or []:
        if int(descriptor.stack_id) >= 0 and int(descriptor.stack_id) not in primary_stack_ids:
            primary_stack_ids.append(int(descriptor.stack_id))
    for work_unit_id in subtask.work_unit_ids or ():
        work_unit = config.work_units.get(str(work_unit_id))
        if work_unit is None:
            continue
        for stack_id in opt._x_candidate_stack_ids_for_sku(int(work_unit.sku_id)):
            sid = int(stack_id)
            if sid < 0 or sid in primary_stack_ids or sid in extra_stack_ids:
                continue
            extra_stack_ids.append(sid)
    centroid = _candidate_centroid_xy(opt, config, subtask)
    if centroid is not None and extra_stack_ids:
        extra_stack_ids.sort(
            key=lambda sid: (
                float(opt._xy_manhattan(centroid, opt._stack_xy(int(sid)))) if opt._stack_xy(int(sid)) is not None else float("inf"),
                float(opt._z_best_insertion_detour(int(sid))),
                int(sid),
            )
        )
    topk = max(0, int(getattr(opt.cfg, "resource_z_candidate_stack_topk", 5)))
    if topk > 0:
        extra_stack_ids = extra_stack_ids[:topk]
    return list(primary_stack_ids) + list(extra_stack_ids)


def _estimate_wait_overflow(config: ResourceConfig, station_id: int) -> float:
    if int(station_id) < 0:
        return 1e9
    count = len([row for row in config.subtasks.values() if int(row.station_id) == int(station_id)])
    return float(count * getattr(OFSConfig, "PICKING_TIME", 1.0))


def _guard_reason(opt, config: ResourceConfig, subtask: ResourceSubtask, plan: Dict[str, object]) -> str:
    detour = float(opt._z_best_insertion_detour(int(plan.get("target_stack_id", -1))))
    arrival_shift = float(detour / max(1.0, float(getattr(OFSConfig, "ROBOT_SPEED", 1.0))))
    wait_overflow = float(_estimate_wait_overflow(config, int(subtask.station_id)))
    route_tail_delta = float(arrival_shift + 0.5 * len(list(plan.get("target_tote_ids", []) or [])))
    if arrival_shift > float(getattr(opt.cfg, "z_arrival_shift_soft_cap", 140.0)) + 1e-9:
        return "z_arrival_shift_soft_cap"
    if wait_overflow > float(getattr(opt.cfg, "z_wait_overflow_soft_cap", 180.0)) + 1e-9:
        return "z_wait_overflow_soft_cap"
    if route_tail_delta > float(getattr(opt.cfg, "z_route_tail_soft_cap", 90.0)) + 1e-9:
        return "z_route_tail_soft_cap"
    if detour > float(getattr(opt.cfg, "z_route_gap_soft_cap", 25.0)) + 1e-9:
        return "z_route_gap_soft_cap"
    return ""


def _descriptor_from_plan(opt, subtask: ResourceSubtask, plan: Dict[str, object], task_id: int, sku_pick_count: int) -> ZTaskDescriptor:
    robot_service = max(float(plan.get("robot_service_time", 0.0) or 0.0), 0.5 * float(len(list(plan.get("target_tote_ids", []) or []))))
    return ZTaskDescriptor(
        task_id=int(task_id),
        stack_id=int(plan.get("target_stack_id", -1)),
        mode=str(plan.get("operation_mode", "FLIP")).upper(),
        target_tote_ids=tuple(int(x) for x in (plan.get("target_tote_ids", []) or [])),
        hit_tote_ids=tuple(int(x) for x in (plan.get("hit_tote_ids", []) or [])),
        noise_tote_ids=tuple(int(x) for x in (plan.get("noise_tote_ids", []) or [])),
        sort_layer_range=None if plan.get("sort_layer_range", None) is None else (
            int(plan.get("sort_layer_range", (0, 0))[0]),
            int(plan.get("sort_layer_range", (0, 0))[1]),
        ),
        station_service_time=float(plan.get("station_service_time", 0.0)),
        robot_service_time=float(robot_service),
        sku_pick_count=int(max(1, sku_pick_count)),
    )


def _is_joint_sort_strategy(strategy: str) -> bool:
    return str(strategy) == "z_repair_joint_sort_colocated_flip"


def _sort_plan_within_capacity(plan: Dict[str, object]) -> bool:
    if str(plan.get("operation_mode", "")).upper() != "SORT":
        return True
    capacity = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)))
    return len(list(plan.get("target_tote_ids", []) or [])) <= int(capacity)


def _joint_sort_seed_hit_map(removed_window: Sequence[ZTaskDescriptor]) -> Dict[int, List[int]]:
    stack_to_rows: Dict[int, List[ZTaskDescriptor]] = defaultdict(list)
    for descriptor in removed_window or ():
        if str(getattr(descriptor, "mode", "")).upper() != "FLIP":
            continue
        stack_to_rows[int(getattr(descriptor, "stack_id", -1))].append(descriptor)
    seed_hits: Dict[int, List[int]] = {}
    for stack_id, rows in stack_to_rows.items():
        if int(stack_id) < 0 or len(rows) < 2:
            continue
        dedup: List[int] = []
        for row in rows:
            for tote_id in (getattr(row, "hit_tote_ids", ()) or ()):
                tid = int(tote_id)
                if tid >= 0 and tid not in dedup:
                    dedup.append(tid)
        if dedup:
            seed_hits[int(stack_id)] = list(dedup)
    return seed_hits


def validate_z_assignment(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    descriptors: Sequence[ZTaskDescriptor],
    external_used_totes: Optional[Set[int]] = None,
) -> bool:
    used_totes: Set[int] = set()
    blocked = {int(x) for x in (external_used_totes or set())}
    for descriptor in descriptors:
        target_ids = [int(x) for x in (descriptor.target_tote_ids or ())]
        hit_ids = [int(x) for x in (descriptor.hit_tote_ids or ())]
        if str(descriptor.mode).upper() == "FLIP" and tuple(target_ids) != tuple(hit_ids):
            return False
        if descriptor.sort_layer_range is not None:
            stack = opt.problem.point_to_stack.get(int(descriptor.stack_id))
            if stack is None:
                return False
            lo, hi = descriptor.sort_layer_range
            expected = [int(getattr(tote, "id", -1)) for tote in (getattr(stack, "totes", []) or [])[int(lo):int(hi) + 1]]
            if tuple(expected) != tuple(target_ids):
                return False
        for tote_id in target_ids:
            if int(tote_id) in used_totes or int(tote_id) in blocked:
                return False
            used_totes.add(int(tote_id))
    demand = _demand_counts(config, subtask)
    remaining = dict(demand)
    for descriptor in descriptors:
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)
    return all(int(qty) <= 0 for qty in remaining.values())


def build_full_z_assignment(
    opt,
    config: ResourceConfig,
    subtask_id: int,
    preferred_stack_ids: Optional[Sequence[int]] = None,
    strategy: str = "fallback",
    allow_fallback: bool = True,
    external_used_totes: Optional[Set[int]] = None,
    rng=None,
) -> Tuple[bool, List[ZTaskDescriptor], Dict[str, object]]:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return False, [], {"reason": "subtask_missing"}
    preserved: List[ZTaskDescriptor] = []
    if external_used_totes is None:
        external_used_totes = global_used_totes(config, exclude_subtask_ids={int(subtask_id)})
    return _rebuild_window(
        opt=opt,
        config=config,
        subtask=subtask,
        preserved_before=preserved,
        preserved_after=preserved,
        seed_stack_ids=list(preferred_stack_ids or ()),
        strategy=str(strategy),
        allow_fallback=bool(allow_fallback),
        external_used_totes=set(int(x) for x in (external_used_totes or set())),
        rng=rng,
    )


def _rebuild_window(
    opt,
    config: ResourceConfig,
    subtask: ResourceSubtask,
    preserved_before: Sequence[ZTaskDescriptor],
    preserved_after: Sequence[ZTaskDescriptor],
    seed_stack_ids: Sequence[int],
    strategy: str,
    allow_fallback: bool,
    removed_window: Optional[Sequence[ZTaskDescriptor]] = None,
    external_used_totes: Optional[Set[int]] = None,
    rng=None,
) -> Tuple[bool, List[ZTaskDescriptor], Dict[str, object]]:
    preserved_all = list(preserved_before) + list(preserved_after)
    temp_subtask = _build_temp_subtask(opt, config, subtask, preserved_all)
    demand = _demand_counts(config, subtask)
    remaining = dict(demand)
    for descriptor in preserved_all:
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)
    created: List[ZTaskDescriptor] = []
    candidate_stack_ids = _candidate_stack_ids(opt, config, subtask, seed_stack_ids)
    fallback_used = False
    blocked_totes = {int(x) for x in (external_used_totes or set())}
    local_used_totes = {
        int(tote_id)
        for descriptor in preserved_all
        for tote_id in (descriptor.target_tote_ids or ())
        if int(tote_id) >= 0
    }
    joint_seed_hits = _joint_sort_seed_hit_map(removed_window or []) if _is_joint_sort_strategy(str(strategy)) else {}

    while any(int(qty) > 0 for qty in remaining.values()):
        candidate_rows = []
        if joint_seed_hits:
            for stack_id, seed_hits in joint_seed_hits.items():
                hit_ids = [
                    int(tote_id)
                    for tote_id in (seed_hits or [])
                    if int(tote_id) not in blocked_totes and int(tote_id) not in local_used_totes
                ]
                if not hit_ids:
                    continue
                dummy_task = Task(
                    task_id=-1,
                    sub_task_id=int(subtask.subtask_id),
                    target_stack_id=int(stack_id),
                    target_station_id=int(subtask.station_id),
                    operation_mode="SORT",
                )
                plan = opt._z_build_plan_from_hits(temp_subtask, dummy_task, int(stack_id), hit_ids, "SORT", {-1})
                if not bool(plan.get("valid", False)) or not _sort_plan_within_capacity(plan):
                    continue
                target_ids = [int(x) for x in (plan.get("target_tote_ids", []) or [])]
                if any(int(tid) in blocked_totes or int(tid) in local_used_totes for tid in target_ids):
                    continue
                coverage_gain = int(_coverage_gain(opt, remaining, list(plan.get("hit_tote_ids", []) or [])))
                if coverage_gain <= 0:
                    continue
                guard_reason = _guard_reason(opt, config, subtask, plan)
                if guard_reason and not bool(fallback_used):
                    continue
                detour = float(opt._z_best_insertion_detour(int(stack_id)))
                target_len = len(list(plan.get("target_tote_ids", []) or []))
                noise_len = len(list(plan.get("noise_tote_ids", []) or []))
                score = (
                    -2.0,
                    -float(coverage_gain),
                    float(detour),
                    float(noise_len),
                    float(target_len),
                    int(stack_id),
                    "SORT",
                )
                candidate_rows.append({"score": score, "plan": plan, "coverage_gain": coverage_gain})

        for stack_id in candidate_stack_ids:
            dummy_task = Task(
                task_id=-1,
                sub_task_id=int(subtask.subtask_id),
                target_stack_id=int(stack_id),
                target_station_id=int(subtask.station_id),
                operation_mode="FLIP",
            )
            summary = opt._z_stack_summary(temp_subtask, int(stack_id), {-1})
            hit_ids = [
                int(tote_id)
                for tote_id in (summary.get("hit_tote_ids", []) or [])
                if int(tote_id) not in blocked_totes and int(tote_id) not in local_used_totes
            ]
            if not hit_ids:
                continue
            modes = ["FLIP", "SORT"]
            if str(strategy) == "z_repair_sort_range_shrink_first":
                modes = ["SORT", "FLIP"]
            elif _is_joint_sort_strategy(str(strategy)) and int(stack_id) in set(int(x) for x in joint_seed_hits.keys()):
                modes = ["SORT", "FLIP"]
            for mode in modes:
                plan = opt._z_build_plan_from_hits(temp_subtask, dummy_task, int(stack_id), hit_ids, str(mode).upper(), {-1})
                if not bool(plan.get("valid", False)):
                    continue
                if not _sort_plan_within_capacity(plan):
                    continue
                target_ids = [int(x) for x in (plan.get("target_tote_ids", []) or [])]
                if any(int(tid) in blocked_totes or int(tid) in local_used_totes for tid in target_ids):
                    continue
                coverage_gain = int(_coverage_gain(opt, remaining, list(plan.get("hit_tote_ids", []) or [])))
                if coverage_gain <= 0:
                    continue
                guard_reason = _guard_reason(opt, config, subtask, plan)
                if guard_reason and not bool(fallback_used):
                    continue
                detour = float(opt._z_best_insertion_detour(int(stack_id)))
                target_len = len(list(plan.get("target_tote_ids", []) or []))
                noise_len = len(list(plan.get("noise_tote_ids", []) or []))
                same_stack_bonus = 0.0 if int(stack_id) in set(int(x) for x in seed_stack_ids) else 1.0
                if _is_joint_sort_strategy(str(strategy)) and int(stack_id) in set(int(x) for x in joint_seed_hits.keys()):
                    same_stack_bonus -= 0.5
                score = (
                    -float(coverage_gain),
                    float(same_stack_bonus),
                    float(detour),
                    float(noise_len),
                    float(target_len),
                    int(stack_id),
                    str(mode).upper(),
                )
                candidate_rows.append({"score": score, "plan": plan, "coverage_gain": coverage_gain})
        chosen_row = pick_soft_greedy_min(rng, candidate_rows, opt.cfg, score_getter=lambda item: item["score"])
        if chosen_row is None:
            if not bool(allow_fallback) or bool(fallback_used):
                break
            fallback_used = True
            continue

        chosen_plan = chosen_row["plan"]
        coverage_gain = int(chosen_row["coverage_gain"])
        next_task_id = int(config.next_task_id)
        config.next_task_id += 1
        descriptor = _descriptor_from_plan(opt, subtask, chosen_plan, next_task_id, coverage_gain)
        created.append(descriptor)
        local_used_totes.update(int(tid) for tid in (descriptor.target_tote_ids or ()) if int(tid) >= 0)
        temp_task = _descriptor_to_task(subtask, descriptor)
        stack_obj = opt.problem.point_to_stack.get(int(temp_task.target_stack_id))
        if stack_obj is not None:
            temp_subtask.add_execution_detail(temp_task, stack_obj)
        remaining = _consume_coverage(opt, remaining, descriptor.hit_tote_ids)

    full_assignment = list(preserved_before) + list(created) + list(preserved_after)
    if validate_z_assignment(opt, config, subtask, full_assignment, external_used_totes=blocked_totes):
        return True, full_assignment, {"fallback_used": bool(fallback_used)}
    return False, list(full_assignment), {"reason": "invalid_assignment", "fallback_used": bool(fallback_used)}


def _expand_window(descriptors: Sequence[ZTaskDescriptor], center_idx: int, base_size: int, max_size: int, mode_sensitive: bool = False) -> Tuple[int, int]:
    if not descriptors:
        return 0, 0
    size = max(1, min(int(base_size), len(descriptors)))
    start = max(0, int(center_idx) - size // 2)
    end = min(len(descriptors), start + size)
    start = max(0, end - size)
    if bool(mode_sensitive):
        base_mode = str(descriptors[int(center_idx)].mode).upper()
        while start > 0 and str(descriptors[start - 1].mode).upper() == base_mode and (end - start) < int(max_size):
            start -= 1
        while end < len(descriptors) and str(descriptors[end].mode).upper() == base_mode and (end - start) < int(max_size):
            end += 1
    return int(start), int(end)


def _destroy_window(config: ResourceConfig, subtask_id: int, start: int, end: int) -> Dict[str, object]:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return {"success": False}
    before = list(subtask.z_tasks[:int(start)])
    removed = list(subtask.z_tasks[int(start):int(end)])
    after = list(subtask.z_tasks[int(end):])
    if not removed:
        return {"success": False}
    subtask.z_tasks = list(before) + list(after)
    return {
        "success": True,
        "subtask_id": int(subtask_id),
        "window_start": int(start),
        "window_end": int(end),
        "preserved_before": before,
        "removed_window": removed,
        "preserved_after": after,
        "seed_stack_ids": [int(row.stack_id) for row in removed],
    }


def _preview_destroy_window(config: ResourceConfig, subtask_id: int, start: int, end: int) -> Dict[str, object]:
    subtask = config.subtasks.get(int(subtask_id))
    if subtask is None:
        return {"success": False}
    before = list(subtask.z_tasks[:int(start)])
    removed = list(subtask.z_tasks[int(start):int(end)])
    after = list(subtask.z_tasks[int(end):])
    if not removed:
        return {"success": False}
    return {
        "success": True,
        "subtask_id": int(subtask_id),
        "window_start": int(start),
        "window_end": int(end),
        "preserved_before": before,
        "removed_window": removed,
        "preserved_after": after,
        "seed_stack_ids": [int(row.stack_id) for row in removed],
    }


def _destroy_windows(opt, config: ResourceConfig, rng, degree: int, candidate_builder, mode_sensitive: bool) -> Dict[str, object]:
    budget_remaining = max(1, int(degree))
    windows = []
    touched_subtasks: Set[int] = set()
    removed_total = 0
    while budget_remaining > 0:
        candidates = list(candidate_builder(config, touched_subtasks))
        if not candidates:
            break
        picked = pick_ranked_candidate(rng, candidates, opt.cfg)
        if picked is None:
            break
        _, subtask_id, center_idx = picked
        row = config.subtasks.get(int(subtask_id))
        if row is None or not row.z_tasks:
            touched_subtasks.add(int(subtask_id))
            continue
        window_size = max(int(getattr(opt.cfg, "resource_z_window_size", 3)), min(int(budget_remaining), 5))
        start, end = _expand_window(row.z_tasks, int(center_idx), int(window_size), 5, mode_sensitive=bool(mode_sensitive))
        ctx = _destroy_window(config, int(subtask_id), int(start), int(end))
        touched_subtasks.add(int(subtask_id))
        if not bool(ctx.get("success", False)):
            continue
        removed_len = len(list(ctx.get("removed_window", []) or []))
        removed_total += int(removed_len)
        budget_remaining -= int(removed_len)
        windows.append(ctx)
    if not windows:
        return {"success": False}
    payload = {"success": True, "windows": windows, "released_task_count": int(removed_total)}
    if len(windows) == 1:
        payload.update(dict(windows[0]))
    return payload


def _plan_destroy_windows(opt, config: ResourceConfig, rng, degree: int, candidate_builder, mode_sensitive: bool) -> Dict[str, object]:
    budget_remaining = max(1, int(degree))
    windows = []
    touched_subtasks: Set[int] = set()
    removed_total = 0
    while budget_remaining > 0:
        candidates = list(candidate_builder(config, touched_subtasks))
        if not candidates:
            break
        picked = pick_ranked_candidate(rng, candidates, opt.cfg)
        if picked is None:
            break
        _, subtask_id, center_idx = picked
        row = config.subtasks.get(int(subtask_id))
        if row is None or not row.z_tasks:
            touched_subtasks.add(int(subtask_id))
            continue
        window_size = max(int(getattr(opt.cfg, "resource_z_window_size", 3)), min(int(budget_remaining), 5))
        start, end = _expand_window(row.z_tasks, int(center_idx), int(window_size), 5, mode_sensitive=bool(mode_sensitive))
        ctx = _preview_destroy_window(config, int(subtask_id), int(start), int(end))
        touched_subtasks.add(int(subtask_id))
        if not bool(ctx.get("success", False)):
            continue
        removed_len = len(list(ctx.get("removed_window", []) or []))
        removed_total += int(removed_len)
        budget_remaining -= int(removed_len)
        windows.append(ctx)
    if not windows:
        return {"success": False}
    payload = {"success": True, "windows": windows, "released_task_count": int(removed_total)}
    if len(windows) == 1:
        payload.update(dict(windows[0]))
    return payload


def z_destroy_noise_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            noise_scores = [len(task.noise_tote_ids) for task in row.z_tasks]
            center_idx = max(range(len(noise_scores)), key=lambda idx: (noise_scores[idx], -idx))
            candidates.append(((-max(noise_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_noise_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            noise_scores = [len(task.noise_tote_ids) for task in row.z_tasks]
            center_idx = max(range(len(noise_scores)), key=lambda idx: (noise_scores[idx], -idx))
            candidates.append(((-max(noise_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_multistack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or len(row.z_tasks) < 2:
                continue
            change_scores = []
            for idx in range(len(row.z_tasks)):
                left = int(row.z_tasks[idx - 1].stack_id) if idx > 0 else int(row.z_tasks[idx].stack_id)
                right = int(row.z_tasks[idx + 1].stack_id) if idx + 1 < len(row.z_tasks) else int(row.z_tasks[idx].stack_id)
                score = int(left != int(row.z_tasks[idx].stack_id)) + int(right != int(row.z_tasks[idx].stack_id))
                change_scores.append(score)
            center_idx = max(range(len(change_scores)), key=lambda idx: (change_scores[idx], -idx))
            candidates.append(((-max(change_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_multistack_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or len(row.z_tasks) < 2:
                continue
            change_scores = []
            for idx in range(len(row.z_tasks)):
                left = int(row.z_tasks[idx - 1].stack_id) if idx > 0 else int(row.z_tasks[idx].stack_id)
                right = int(row.z_tasks[idx + 1].stack_id) if idx + 1 < len(row.z_tasks) else int(row.z_tasks[idx].stack_id)
                score = int(left != int(row.z_tasks[idx].stack_id)) + int(right != int(row.z_tasks[idx].stack_id))
                change_scores.append(score)
            center_idx = max(range(len(change_scores)), key=lambda idx: (change_scores[idx], -idx))
            candidates.append(((-max(change_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_detour_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            detours = [float(opt._z_best_insertion_detour(int(task.stack_id))) for task in row.z_tasks]
            center_idx = max(range(len(detours)), key=lambda idx: (detours[idx], -idx))
            candidates.append(((-max(detours), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_plan_destroy_detour_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            detours = [float(opt._z_best_insertion_detour(int(task.stack_id))) for task in row.z_tasks]
            center_idx = max(range(len(detours)), key=lambda idx: (detours[idx], -idx))
            candidates.append(((-max(detours), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=False)


def z_destroy_mode_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            mode_scores = []
            for idx in range(len(row.z_tasks)):
                left = str(row.z_tasks[idx - 1].mode).upper() if idx > 0 else str(row.z_tasks[idx].mode).upper()
                right = str(row.z_tasks[idx + 1].mode).upper() if idx + 1 < len(row.z_tasks) else str(row.z_tasks[idx].mode).upper()
                cur = str(row.z_tasks[idx].mode).upper()
                mode_scores.append(int(left != cur) + int(right != cur) + int(cur == "SORT"))
            center_idx = max(range(len(mode_scores)), key=lambda idx: (mode_scores[idx], -idx))
            candidates.append(((-max(mode_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _destroy_windows(opt, config, rng, degree, _build, mode_sensitive=True)


def z_plan_destroy_mode_window(opt, config: ResourceConfig, rng, degree: int) -> Dict[str, object]:
    def _build(config_obj: ResourceConfig, touched_subtasks: Set[int]):
        candidates = []
        for row in config_obj.subtasks.values():
            if int(row.subtask_id) in touched_subtasks or not row.z_tasks:
                continue
            mode_scores = []
            for idx in range(len(row.z_tasks)):
                left = str(row.z_tasks[idx - 1].mode).upper() if idx > 0 else str(row.z_tasks[idx].mode).upper()
                right = str(row.z_tasks[idx + 1].mode).upper() if idx + 1 < len(row.z_tasks) else str(row.z_tasks[idx].mode).upper()
                cur = str(row.z_tasks[idx].mode).upper()
                mode_scores.append(int(left != cur) + int(right != cur) + int(cur == "SORT"))
            center_idx = max(range(len(mode_scores)), key=lambda idx: (mode_scores[idx], -idx))
            candidates.append(((-max(mode_scores), int(row.subtask_id)), int(row.subtask_id), int(center_idx)))
        return sorted(candidates, key=lambda item: item[0])

    return _plan_destroy_windows(opt, config, rng, degree, _build, mode_sensitive=True)


def _build_z_action_signature(destroy_name: str, repair_name: str, windows: Sequence[Dict[str, object]], target_stack_ids: Sequence[int], mode_summary: Sequence[str]) -> Tuple[object, ...]:
    window_sig = tuple(
        sorted(
            (
                int(window_ctx.get("subtask_id", -1)),
                tuple(int(task.task_id) for task in (window_ctx.get("removed_window", []) or [])),
            )
            for window_ctx in (windows or [])
        )
    )
    return (
        "Z",
        str(destroy_name),
        window_sig,
        str(repair_name),
        tuple(int(x) for x in (target_stack_ids or ())),
        tuple(str(x) for x in (mode_summary or ())),
    )


def _build_z_rough_features(opt, windows: Sequence[Dict[str, object]], target_stack_ids: Sequence[int], mode_summary: Sequence[str]) -> Dict[str, float]:
    removed_noise = 0
    removed_target = 0
    removed_stacks = set()
    for window_ctx in windows or []:
        for descriptor in window_ctx.get("removed_window", []) or []:
            removed_noise += len(getattr(descriptor, "noise_tote_ids", []) or [])
            removed_target += len(getattr(descriptor, "target_tote_ids", []) or [])
            removed_stacks.add(int(getattr(descriptor, "stack_id", -1)))
    detour_proxy = 0.0
    if target_stack_ids:
        detour_proxy = float(sum(float(opt._z_best_insertion_detour(int(stack_id))) for stack_id in target_stack_ids) / max(1, len(list(target_stack_ids))))
    noise_ratio = float(removed_noise / max(1, removed_target))
    stack_delta = float(max(0, len(set(int(x) for x in (target_stack_ids or []))) - max(0, len(removed_stacks) - 1)))
    mode_penalty = 0.2 * float(sum(1 for mode in (mode_summary or []) if str(mode).upper() == "SORT"))
    sz_delta = float(0.35 * noise_ratio + 0.25 * stack_delta + 0.20 * detour_proxy / 100.0 + 0.20 * mode_penalty - 0.15 * len(target_stack_ids))
    return {
        "sz_delta": float(sz_delta),
        "affected_count": float(sum(len(window_ctx.get("removed_window", []) or []) for window_ctx in (windows or []))),
    }


def plan_z_candidate(opt, config: ResourceConfig, destroy_name: str, repair_name: str, rng, degree: int) -> Dict[str, object]:
    destroy_planners = {
        "z_destroy_noise_window": z_plan_destroy_noise_window,
        "z_destroy_multistack_window": z_plan_destroy_multistack_window,
        "z_destroy_detour_window": z_plan_destroy_detour_window,
        "z_destroy_mode_window": z_plan_destroy_mode_window,
    }
    destroy_ctx = destroy_planners[str(destroy_name)](opt, config, rng, degree)
    if not bool(destroy_ctx.get("success", False)):
        return {"success": False}
    windows = list(destroy_ctx.get("windows", []) or [])
    target_stack_ids: List[int] = []
    mode_summary: List[str] = []
    for window_ctx in windows:
        subtask_id = int(window_ctx.get("subtask_id", -1))
        subtask = config.subtasks.get(int(subtask_id))
        if subtask is None:
            continue
        candidate_stacks = _candidate_stack_ids(opt, config, subtask, window_ctx.get("seed_stack_ids", []) or [])
        for stack_id in candidate_stacks[:2]:
            if int(stack_id) not in target_stack_ids:
                target_stack_ids.append(int(stack_id))
        removed_modes = [str(descriptor.mode).upper() for descriptor in (window_ctx.get("removed_window", []) or [])]
        mode_summary.append(
            "SORT"
            if str(repair_name) in {"z_repair_sort_range_shrink_first", "z_repair_joint_sort_colocated_flip"}
            else (removed_modes[0] if removed_modes else "FLIP")
        )
    return {
        "success": True,
        "destroy_ctx": destroy_ctx,
        "strategy": str(repair_name),
        "target_stack_ids": list(target_stack_ids),
        "mode_summary": list(mode_summary),
        "fallback_used": False,
        "action_signature": _build_z_action_signature(str(destroy_name), str(repair_name), windows, target_stack_ids, mode_summary),
        "rough_features": _build_z_rough_features(opt, windows, target_stack_ids, mode_summary),
    }


def apply_exact_z_plan(opt, config: ResourceConfig, plan: Dict[str, object], rng=None) -> Dict[str, object]:
    destroy_ctx = dict(plan.get("destroy_ctx", {}) or {})
    windows = list(destroy_ctx.get("windows", []) or [])
    if not windows and int(destroy_ctx.get("subtask_id", -1)) >= 0:
        windows = [destroy_ctx]
    touched_subtask_ids = sorted({int(window_ctx.get("subtask_id", -1)) for window_ctx in windows if int(window_ctx.get("subtask_id", -1)) >= 0})
    if not touched_subtask_ids:
        return {"success": False}
    candidate = config.clone_for_layer("Z", touched_subtask_ids)
    exact_windows = []
    for window_ctx in windows:
        subtask_id = int(window_ctx.get("subtask_id", -1))
        exact_ctx = _destroy_window(candidate, subtask_id, int(window_ctx.get("window_start", 0)), int(window_ctx.get("window_end", 0)))
        if not bool(exact_ctx.get("success", False)):
            return {"success": False}
        exact_windows.append(exact_ctx)
    exact_ctx = {"success": True, "windows": exact_windows}
    repair_result = _repair_window(opt, candidate, exact_ctx, str(plan.get("strategy", "z_repair_same_stack_window")), allow_fallback=False, rng=rng)
    fallback_used = False
    if not bool(repair_result.get("success", False)):
        repair_result = _repair_window(opt, candidate, exact_ctx, "z_repair_greedy_fallback", allow_fallback=True, rng=rng)
        fallback_used = bool(repair_result.get("success", False))
    if not bool(repair_result.get("success", False)):
        return {"success": False}
    candidate.rebuild_indices()
    return {
        "success": True,
        "config": candidate,
        "score_cache": None,
        "affected_ids": set(int(x) for x in (repair_result.get("affected_subtask_ids", set()) or set())),
        "fallback_used": bool(fallback_used or repair_result.get("fallback_used", False)),
        "projection_mode": "",
        "projection_repaired_subtask_count": 0,
        "validation_signature": candidate.validation_signature(),
    }


def _repair_window(opt, config: ResourceConfig, ctx: Dict[str, object], strategy: str, allow_fallback: bool, rng=None) -> Dict[str, object]:
    if not bool(ctx.get("success", False)):
        return {"success": False}
    windows = list(ctx.get("windows", []) or [])
    if not windows and int(ctx.get("subtask_id", -1)) >= 0:
        windows = [ctx]
    if _is_joint_sort_strategy(str(strategy)) and len(windows) != 1:
        return {"success": False, "reason": "joint_sort_requires_single_subtask_window", "fallback_used": False}
    affected_subtasks: Set[int] = set()
    fallback_used = False
    original_assignments = {
        int(window_ctx.get("subtask_id", -1)): list((config.subtasks.get(int(window_ctx.get("subtask_id", -1))).z_tasks if config.subtasks.get(int(window_ctx.get("subtask_id", -1))) is not None else []))
        for window_ctx in windows
        if int(window_ctx.get("subtask_id", -1)) >= 0
    }
    for window_ctx in windows:
        subtask_id = int(window_ctx.get("subtask_id", -1))
        subtask = config.subtasks.get(int(subtask_id))
        if subtask is None:
            return {"success": False, "reason": "subtask_missing", "fallback_used": bool(fallback_used)}
        external_used = global_used_totes(config, exclude_subtask_ids={int(subtask_id)})
        success, assignment, meta = _rebuild_window(
            opt=opt,
            config=config,
            subtask=subtask,
            preserved_before=list(window_ctx.get("preserved_before", []) or []),
            preserved_after=list(window_ctx.get("preserved_after", []) or []),
            seed_stack_ids=list(window_ctx.get("seed_stack_ids", []) or []),
            strategy=str(strategy),
            allow_fallback=bool(allow_fallback),
            removed_window=list(window_ctx.get("removed_window", []) or []),
            external_used_totes=external_used,
            rng=rng,
        )
        if not success:
            for restore_subtask_id, restore_assignment in original_assignments.items():
                restore_row = config.subtasks.get(int(restore_subtask_id))
                if restore_row is not None:
                    restore_row.z_tasks = list(restore_assignment)
            return {"success": False, "reason": str(meta.get("reason", "repair_fail")), "fallback_used": bool(fallback_used or meta.get("fallback_used", False))}
        subtask.z_tasks = list(assignment)
        affected_subtasks.add(int(subtask_id))
        fallback_used = bool(fallback_used or meta.get("fallback_used", False))
    return {
        "success": bool(affected_subtasks),
        "affected_subtask_ids": affected_subtasks,
        "fallback_used": bool(fallback_used),
    }


def z_repair_same_stack_window(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_same_stack_window", allow_fallback=False, rng=rng)


def z_repair_bounded_detour_window(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_bounded_detour_window", allow_fallback=False, rng=rng)


def z_repair_sort_range_shrink_first(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_sort_range_shrink_first", allow_fallback=False, rng=rng)


def z_repair_mode_toggle_contextual(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_mode_toggle_contextual", allow_fallback=False, rng=rng)


def z_repair_joint_sort_colocated_flip(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_joint_sort_colocated_flip", allow_fallback=False, rng=rng)


def z_repair_greedy_fallback(opt, config: ResourceConfig, ctx: Dict[str, object], rng) -> Dict[str, object]:
    return _repair_window(opt, config, ctx, "z_repair_greedy_fallback", allow_fallback=True, rng=rng)


Z_DESTROY_OPERATORS = {
    "z_destroy_noise_window": z_destroy_noise_window,
    "z_destroy_multistack_window": z_destroy_multistack_window,
    "z_destroy_detour_window": z_destroy_detour_window,
    "z_destroy_mode_window": z_destroy_mode_window,
}

Z_REPAIR_OPERATORS = {
    "z_repair_same_stack_window": z_repair_same_stack_window,
    "z_repair_bounded_detour_window": z_repair_bounded_detour_window,
    "z_repair_sort_range_shrink_first": z_repair_sort_range_shrink_first,
    "z_repair_mode_toggle_contextual": z_repair_mode_toggle_contextual,
    "z_repair_joint_sort_colocated_flip": z_repair_joint_sort_colocated_flip,
}

Z_FALLBACK_OPERATOR = "z_repair_greedy_fallback"
