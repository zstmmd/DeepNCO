from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

from .state import ResourceConfig

T = TypeVar("T")


def _pool_weights(cfg, pool_len: int) -> List[float]:
    raw = getattr(cfg, "resource_destroy_candidate_pool_weights", ())
    try:
        weights = [max(0.0, float(x)) for x in tuple(raw)]
    except Exception:
        weights = []
    if not weights:
        weights = [1.0 / float(idx + 1) for idx in range(int(pool_len))]
    elif len(weights) < int(pool_len):
        weights.extend([float(weights[-1])] * (int(pool_len) - len(weights)))
    weights = weights[: int(pool_len)]
    total = sum(weights)
    if total <= 1e-9:
        return [1.0 / max(1, int(pool_len))] * int(pool_len)
    return [float(x / total) for x in weights]


def pick_ranked_candidate(rng, ranked_items: Sequence[T], cfg) -> Optional[T]:
    items = list(ranked_items or [])
    if not items:
        return None
    if rng is None:
        return items[0]
    pool_size = max(1, min(int(getattr(cfg, "resource_destroy_candidate_pool_size", 3)), len(items)))
    pool = items[:pool_size]
    weights = _pool_weights(cfg, len(pool))
    draw = float(rng.random())
    acc = 0.0
    for item, weight in zip(pool, weights):
        acc += float(weight)
        if draw <= acc + 1e-12:
            return item
    return pool[-1]


def pick_soft_greedy_min(rng, scored_items: Sequence[T], cfg, score_getter) -> Optional[T]:
    items = list(scored_items or [])
    if not items:
        return None
    ranked = sorted(items, key=lambda item: score_getter(item))
    topk = max(1, min(int(getattr(cfg, "resource_soft_greedy_topk", 3)), len(ranked)))
    noise = max(0.0, float(getattr(cfg, "resource_soft_greedy_noise", 0.05)))
    pool = ranked[:topk]
    if rng is None or noise <= 1e-12:
        return pool[0]
    best = None
    for item in pool:
        raw_score = score_getter(item)
        if isinstance(raw_score, (tuple, list)) and raw_score:
            base_numeric = float(raw_score[0])
            stable_tail = tuple(raw_score[1:])
        else:
            base_numeric = float(raw_score)
            stable_tail = tuple()
        scale = 1.0 + float(rng.uniform(-noise, noise))
        noisy_score = float(base_numeric * scale)
        candidate = (noisy_score, base_numeric, stable_tail)
        if best is None or candidate < best[0]:
            best = (candidate, item)
    return best[1] if best is not None else pool[0]


def tote_to_task_rows(
    config: ResourceConfig,
    exclude_subtask_ids: Optional[Iterable[int]] = None,
    exclude_task_ids: Optional[Iterable[int]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    excluded_subtasks = {int(x) for x in (exclude_subtask_ids or set())}
    excluded_tasks = {int(x) for x in (exclude_task_ids or set())}
    tote_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for subtask in config.subtasks.values():
        if int(subtask.subtask_id) in excluded_subtasks:
            continue
        for descriptor in subtask.z_tasks or []:
            if int(descriptor.task_id) in excluded_tasks:
                continue
            for tote_id in descriptor.target_tote_ids or ():
                tote_id = int(tote_id)
                if tote_id < 0:
                    continue
                tote_rows[tote_id].append(
                    {
                        "task_id": int(descriptor.task_id),
                        "subtask_id": int(subtask.subtask_id),
                    }
                )
    return dict(tote_rows)


def duplicate_tote_rows(
    config: ResourceConfig,
    exclude_subtask_ids: Optional[Iterable[int]] = None,
    exclude_task_ids: Optional[Iterable[int]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    rows = tote_to_task_rows(
        config=config,
        exclude_subtask_ids=exclude_subtask_ids,
        exclude_task_ids=exclude_task_ids,
    )
    return {
        int(tote_id): list(items)
        for tote_id, items in rows.items()
        if len(list(items)) >= 2
    }


def duplicate_tote_count(
    config: ResourceConfig,
    exclude_subtask_ids: Optional[Iterable[int]] = None,
    exclude_task_ids: Optional[Iterable[int]] = None,
) -> int:
    rows = duplicate_tote_rows(
        config=config,
        exclude_subtask_ids=exclude_subtask_ids,
        exclude_task_ids=exclude_task_ids,
    )
    return int(sum(max(0, len(items) - 1) for items in rows.values()))


def global_used_totes(
    config: ResourceConfig,
    exclude_subtask_ids: Optional[Iterable[int]] = None,
    exclude_task_ids: Optional[Iterable[int]] = None,
) -> Set[int]:
    tote_rows = tote_to_task_rows(
        config=config,
        exclude_subtask_ids=exclude_subtask_ids,
        exclude_task_ids=exclude_task_ids,
    )
    return {int(tote_id) for tote_id in tote_rows.keys()}
