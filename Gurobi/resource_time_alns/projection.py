from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .operators_y import _assign_station_rank, _choose_station_load_balance, _normalize_ranks
from .operators_z import build_full_z_assignment
from .state import ResourceConfig, ResourceSubtask, ScoreCache
from .utils import global_used_totes


def _origin_to_subtask_ids(previous_config: ResourceConfig) -> Dict[str, int]:
    rows = {}
    for subtask_id in previous_config.subtasks.keys():
        rows[f"st_{int(subtask_id)}"] = int(subtask_id)
    return rows


def _seed_candidates(previous_config: ResourceConfig, current_subtask: ResourceSubtask) -> Tuple[Optional[ResourceSubtask], List[ResourceSubtask]]:
    origin_map = _origin_to_subtask_ids(previous_config)
    explicit_ids = [origin_map[key] for key in current_subtask.origin_keys() if key in origin_map and int(origin_map[key]) in previous_config.subtasks]
    if not explicit_ids:
        overlap_rows = []
        current_units = set(current_subtask.work_unit_signature())
        for row in previous_config.subtasks.values():
            overlap = len(current_units & set(row.work_unit_signature()))
            if overlap > 0:
                overlap_rows.append((-overlap, int(row.subtask_id)))
        overlap_rows.sort()
        explicit_ids = [int(row[1]) for row in overlap_rows]
    primary = previous_config.subtasks.get(int(explicit_ids[0])) if explicit_ids else None
    secondary = [previous_config.subtasks[int(subtask_id)] for subtask_id in explicit_ids[1:] if int(subtask_id) in previous_config.subtasks]
    return primary, secondary


def _station_load_counts(config: ResourceConfig) -> Dict[int, int]:
    loads: Dict[int, int] = {}
    for row in config.subtasks.values():
        if int(row.station_id) < 0:
            continue
        loads[int(row.station_id)] = int(loads.get(int(row.station_id), 0)) + 1
    return loads


def _choose_station(opt, config: ResourceConfig, subtask: ResourceSubtask, primary_seed: Optional[ResourceSubtask], secondary_seeds: Sequence[ResourceSubtask]) -> int:
    station_count = max(1, len(getattr(getattr(opt, "problem", None), "station_list", []) or []))
    load_counts = _station_load_counts(config)
    candidate_station_ids: List[int] = []
    if primary_seed is not None and int(primary_seed.station_id) >= 0:
        candidate_station_ids.append(int(primary_seed.station_id))
    for row in secondary_seeds:
        if int(row.station_id) >= 0 and int(row.station_id) not in candidate_station_ids:
            candidate_station_ids.append(int(row.station_id))
    for station_id in range(station_count):
        if int(station_id) not in candidate_station_ids:
            candidate_station_ids.append(int(station_id))
    rows = []
    for station_id in candidate_station_ids:
        seed_penalty = 0.0 if primary_seed is None or int(primary_seed.station_id) == int(station_id) else 1.0
        load_penalty = float(load_counts.get(int(station_id), 0))
        rank_penalty = float(load_counts.get(int(station_id), 0))
        rows.append((seed_penalty + 0.35 * load_penalty + 0.15 * rank_penalty, int(station_id)))
    rows.sort(key=lambda item: (item[0], item[1]))
    return int(rows[0][1]) if rows else 0


def _preferred_stack_ids(primary_seed: Optional[ResourceSubtask], secondary_seeds: Sequence[ResourceSubtask]) -> List[int]:
    stack_ids: List[int] = []
    for row in ([primary_seed] if primary_seed is not None else []) + list(secondary_seeds):
        if row is None:
            continue
        for descriptor in row.z_tasks or []:
            if int(descriptor.stack_id) >= 0 and int(descriptor.stack_id) not in stack_ids:
                stack_ids.append(int(descriptor.stack_id))
    return list(stack_ids)


def _full_y_refresh(opt, config: ResourceConfig) -> None:
    ordered_rows = sorted(
        list(config.subtasks.values()),
        key=lambda row: (
            int(row.station_id if row.station_id >= 0 else 10**9),
            int(row.station_rank if row.station_rank >= 0 else 10**9),
            int(row.subtask_id),
        ),
    )
    for row in ordered_rows:
        row.station_id = -1
        row.station_rank = -1
    for row in ordered_rows:
        station_id = _choose_station_load_balance(opt, config, row)
        _assign_station_rank(config, int(row.subtask_id), int(station_id))
    _normalize_ranks(config)


def apply_projection_repair(
    opt,
    previous_config: ResourceConfig,
    candidate_config: ResourceConfig,
    previous_eval,
    affected_subtask_ids: Sequence[int],
    iter_id: int,
    rng,
) -> Tuple[ResourceConfig, ScoreCache, Dict[str, object]]:
    del iter_id
    affected_ids = {int(x) for x in (affected_subtask_ids or [])}
    frozen_ids = set(int(subtask_id) for subtask_id in previous_config.subtasks.keys() if int(subtask_id) not in affected_ids)
    projection_repaired = 0
    fallback_used = False
    full_refresh = bool(affected_ids) and float(rng.random()) < float(getattr(opt.cfg, "resource_projection_full_y_refresh_prob", 0.10))
    if full_refresh:
        _full_y_refresh(opt, candidate_config)
    for subtask_id in sorted(affected_ids):
        subtask = candidate_config.subtasks.get(int(subtask_id))
        if subtask is None:
            continue
        primary_seed, secondary_seeds = _seed_candidates(previous_config, subtask)
        if not full_refresh:
            chosen_station = _choose_station(opt, candidate_config, subtask, primary_seed, secondary_seeds)
            subtask.station_id = int(chosen_station)
            subtask.station_rank = int(len(candidate_config.station_subtasks(int(chosen_station))))
        preferred_stacks = _preferred_stack_ids(primary_seed, secondary_seeds)
        success, assignment, meta = build_full_z_assignment(
            opt=opt,
            config=candidate_config,
            subtask_id=int(subtask.subtask_id),
            preferred_stack_ids=preferred_stacks,
            strategy="projection",
            allow_fallback=True,
            external_used_totes=global_used_totes(candidate_config, exclude_subtask_ids={int(subtask.subtask_id)}),
            rng=rng,
        )
        if success:
            subtask.z_tasks = list(assignment)
            fallback_used = bool(fallback_used or meta.get("fallback_used", False))
        projection_repaired += 1
    candidate_config.rebuild_indices()
    score_cache = ScoreCache(
        frozen_subtask_ids=frozenset() if full_refresh else frozenset(int(x) for x in frozen_ids if int(x) in previous_eval.subtask_y_contribs),
        Sy_frozen=0.0 if full_refresh else float(sum(previous_eval.subtask_y_contribs.get(int(x), 0.0) for x in frozen_ids)),
        Sz_frozen=0.0 if full_refresh else float(sum(previous_eval.subtask_z_contribs.get(int(x), 0.0) for x in frozen_ids)),
        dirty=bool(not full_refresh and affected_ids),
        last_validation_iter=int(getattr(previous_eval, "metadata", {}).get("last_validation_iter", 0)),
        last_validation_f_raw=float(getattr(previous_eval, "metadata", {}).get("last_validation_f_raw", 0.0)),
        recent_validated_makespans=list(getattr(previous_eval, "metadata", {}).get("recent_validated_makespans", []) or []),
    )
    return candidate_config, score_cache, {
        "projection_mode": "full_y_refresh" if full_refresh else ("greedy_repair" if affected_ids else "frozen_only"),
        "projection_repaired_subtask_count": int(projection_repaired),
        "fallback_used": bool(fallback_used),
    }
