from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


@dataclass
class WorkUnitInfo:
    work_unit_id: str
    order_id: int
    sku_id: int
    occurrence_index: int
    origin_subtask_id: int


@dataclass
class ZTaskDescriptor:
    task_id: int
    stack_id: int
    mode: str
    target_tote_ids: Tuple[int, ...]
    hit_tote_ids: Tuple[int, ...]
    noise_tote_ids: Tuple[int, ...]
    sort_layer_range: Optional[Tuple[int, int]] = None
    station_service_time: float = 0.0
    robot_service_time: float = 0.0
    sku_pick_count: int = 0

    def clone(self) -> "ZTaskDescriptor":
        return ZTaskDescriptor(
            task_id=int(self.task_id),
            stack_id=int(self.stack_id),
            mode=str(self.mode),
            target_tote_ids=tuple(int(x) for x in (self.target_tote_ids or ())),
            hit_tote_ids=tuple(int(x) for x in (self.hit_tote_ids or ())),
            noise_tote_ids=tuple(int(x) for x in (self.noise_tote_ids or ())),
            sort_layer_range=None if self.sort_layer_range is None else (int(self.sort_layer_range[0]), int(self.sort_layer_range[1])),
            station_service_time=float(self.station_service_time),
            robot_service_time=float(self.robot_service_time),
            sku_pick_count=int(self.sku_pick_count),
        )

    def signature(self) -> Tuple[Any, ...]:
        return (
            int(self.stack_id),
            str(self.mode).upper(),
            tuple(int(x) for x in self.target_tote_ids),
            tuple(int(x) for x in self.hit_tote_ids),
            tuple(int(x) for x in self.noise_tote_ids),
            None if self.sort_layer_range is None else (int(self.sort_layer_range[0]), int(self.sort_layer_range[1])),
        )

    def validation_signature(self) -> Tuple[Any, ...]:
        return (
            int(self.stack_id),
            str(self.mode).upper(),
            tuple(int(x) for x in self.target_tote_ids),
            None if self.sort_layer_range is None else (int(self.sort_layer_range[0]), int(self.sort_layer_range[1])),
        )

    def exact_eval_signature(self) -> Tuple[Any, ...]:
        return self.validation_signature()


@dataclass
class ResourceSubtask:
    subtask_id: int
    order_id: int
    work_unit_ids: Tuple[str, ...]
    station_id: int
    station_rank: int
    z_tasks: List[ZTaskDescriptor] = field(default_factory=list)
    origin_group_ids: Tuple[str, ...] = field(default_factory=tuple)

    def clone(self) -> "ResourceSubtask":
        return copy.deepcopy(self)

    def clone_for_y(self) -> "ResourceSubtask":
        return ResourceSubtask(
            subtask_id=int(self.subtask_id),
            order_id=int(self.order_id),
            work_unit_ids=tuple(str(x) for x in (self.work_unit_ids or ())),
            station_id=int(self.station_id),
            station_rank=int(self.station_rank),
            z_tasks=list(self.z_tasks or []),
            origin_group_ids=tuple(str(x) for x in (self.origin_group_ids or ())),
        )

    def clone_for_z(self) -> "ResourceSubtask":
        return ResourceSubtask(
            subtask_id=int(self.subtask_id),
            order_id=int(self.order_id),
            work_unit_ids=tuple(str(x) for x in (self.work_unit_ids or ())),
            station_id=int(self.station_id),
            station_rank=int(self.station_rank),
            z_tasks=[task.clone() for task in (self.z_tasks or [])],
            origin_group_ids=tuple(str(x) for x in (self.origin_group_ids or ())),
        )

    def work_unit_signature(self) -> Tuple[str, ...]:
        return tuple(sorted(str(x) for x in (self.work_unit_ids or ())))

    def signature(self) -> Tuple[Any, ...]:
        return (
            int(self.order_id),
            self.work_unit_signature(),
            int(self.station_id),
            int(self.station_rank),
            tuple(task.signature() for task in (self.z_tasks or [])),
        )

    def validation_signature(self) -> Tuple[Any, ...]:
        return (
            int(self.order_id),
            self.work_unit_signature(),
            int(self.station_id),
            int(self.station_rank),
            tuple(task.validation_signature() for task in (self.z_tasks or [])),
        )

    def exact_eval_signature(self) -> Tuple[Any, ...]:
        return (
            int(self.order_id),
            self.work_unit_signature(),
            int(self.station_id),
            int(self.station_rank),
            tuple(task.exact_eval_signature() for task in (self.z_tasks or [])),
        )

    def origin_keys(self) -> Tuple[str, ...]:
        if self.origin_group_ids:
            return tuple(str(x) for x in self.origin_group_ids)
        return (f"st_{int(self.subtask_id)}",)


@dataclass
class ResourceConfig:
    work_units: Dict[str, WorkUnitInfo]
    subtasks: Dict[int, ResourceSubtask]
    capacity_limits: Dict[int, int]
    next_subtask_id: int
    next_task_id: int

    def clone(self) -> "ResourceConfig":
        return copy.deepcopy(self)

    def clone_for_layer(self, layer: str, touched_subtask_ids: Optional[List[int]] = None) -> "ResourceConfig":
        layer_name = str(layer).upper()
        if layer_name == "X":
            return self.clone()
        touched = {int(x) for x in (touched_subtask_ids or [])}
        subtasks = dict(self.subtasks)
        for subtask_id in list(touched):
            row = self.subtasks.get(int(subtask_id))
            if row is None:
                continue
            subtasks[int(subtask_id)] = row.clone_for_y() if layer_name == "Y" else row.clone_for_z()
        return ResourceConfig(
            work_units=self.work_units,
            subtasks=subtasks,
            capacity_limits=self.capacity_limits,
            next_subtask_id=int(self.next_subtask_id),
            next_task_id=int(self.next_task_id),
        )

    def rebuild_indices(self) -> "ResourceConfig":
        normalized: Dict[int, ResourceSubtask] = {}
        for subtask_id, subtask in list((self.subtasks or {}).items()):
            if not getattr(subtask, "work_unit_ids", None):
                continue
            subtask.subtask_id = int(subtask_id)
            subtask.work_unit_ids = tuple(sorted(str(x) for x in (subtask.work_unit_ids or ())))
            normalized[int(subtask_id)] = subtask
        self.subtasks = normalized
        self.normalize_station_ranks()
        self.next_subtask_id = max([int(self.next_subtask_id)] + [int(x) + 1 for x in self.subtasks.keys()])
        next_task_candidates = [int(self.next_task_id)]
        for subtask in self.subtasks.values():
            next_task_candidates.extend(int(task.task_id) + 1 for task in (subtask.z_tasks or []))
        self.next_task_id = max(next_task_candidates)
        return self

    def subtasks_by_order(self, order_id: int) -> List[ResourceSubtask]:
        rows = [row for row in self.subtasks.values() if int(row.order_id) == int(order_id)]
        rows.sort(key=lambda row: (int(row.station_rank), int(row.subtask_id)))
        return rows

    def station_subtasks(self, station_id: int) -> List[ResourceSubtask]:
        rows = [row for row in self.subtasks.values() if int(row.station_id) == int(station_id)]
        rows.sort(key=lambda row: (int(row.station_rank), int(row.subtask_id)))
        return rows

    def normalize_station_ranks(self) -> None:
        station_rows: Dict[int, List[ResourceSubtask]] = {}
        for subtask in self.subtasks.values():
            if int(subtask.station_id) < 0:
                subtask.station_rank = -1
                continue
            station_rows.setdefault(int(subtask.station_id), []).append(subtask)
        for rows in station_rows.values():
            rows.sort(key=lambda row: (int(row.station_rank if row.station_rank >= 0 else 10**9), int(row.subtask_id)))
            for rank, row in enumerate(rows):
                row.station_rank = int(rank)

    def signature(self) -> Tuple[Any, ...]:
        return tuple(
            (int(subtask_id), self.subtasks[int(subtask_id)].signature())
            for subtask_id in sorted(self.subtasks.keys())
        )

    def validation_signature(self) -> Tuple[Any, ...]:
        return tuple(
            (int(subtask_id), self.subtasks[int(subtask_id)].validation_signature())
            for subtask_id in sorted(self.subtasks.keys())
        )

    def exact_eval_signature(self) -> Tuple[Any, ...]:
        return tuple(
            (int(subtask_id), self.subtasks[int(subtask_id)].exact_eval_signature())
            for subtask_id in sorted(self.subtasks.keys())
        )

    def coverage_summary(self, tote_map: Dict[int, Any]) -> Dict[str, Any]:
        unmet_total = 0
        unmet_subtasks = 0
        subtask_rows: List[Dict[str, Any]] = []
        for subtask in self.subtasks.values():
            req: Dict[int, int] = defaultdict(int)
            for work_unit_id in (subtask.work_unit_ids or ()):
                work_unit = self.work_units.get(str(work_unit_id))
                if work_unit is None:
                    continue
                req[int(work_unit.sku_id)] += 1
            prov: Dict[int, int] = defaultdict(int)
            for descriptor in (subtask.z_tasks or []):
                for tote_id in (descriptor.target_tote_ids or ()):
                    tote = tote_map.get(int(tote_id))
                    if tote is None:
                        continue
                    for sku_id, qty in getattr(tote, "sku_quantity_map", {}).items():
                        sku_id = int(sku_id)
                        if sku_id in req:
                            prov[sku_id] += int(qty)
            unmet = {
                int(sku_id): int(max(0, int(req[sku_id]) - int(prov.get(sku_id, 0))))
                for sku_id in req
                if int(req[sku_id]) - int(prov.get(sku_id, 0)) > 0
            }
            unmet_units = int(sum(unmet.values()))
            if unmet_units > 0:
                unmet_total += int(unmet_units)
                unmet_subtasks += 1
            subtask_rows.append(
                {
                    "subtask_id": int(subtask.subtask_id),
                    "order_id": int(subtask.order_id),
                    "required_sku_units": int(sum(int(v) for v in req.values())),
                    "provided_sku_units": int(sum(min(int(req.get(sku_id, 0)), int(prov.get(sku_id, 0))) for sku_id in req)),
                    "unmet_sku_units": int(unmet_units),
                    "unmet_skus": dict(sorted(unmet.items())),
                    "coverage_ok": bool(unmet_units == 0),
                }
            )
        return {
            "coverage_ok": bool(unmet_total == 0),
            "unmet_sku_total": int(unmet_total),
            "unmet_subtask_count": int(unmet_subtasks),
            "subtasks": subtask_rows,
        }


@dataclass
class ScoreCache:
    frozen_subtask_ids: FrozenSet[int] = frozenset()
    Sy_frozen: float = 0.0
    Sz_frozen: float = 0.0
    dirty: bool = False
    last_validation_iter: int = 0
    last_validation_f_raw: float = 0.0
    recent_validated_makespans: List[float] = field(default_factory=list)


@dataclass
class UpperEvalResult:
    Sx: float
    Sy: float
    Sz: float
    F_raw: float
    F_cal: float
    Sy_frozen: float = 0.0
    Sy_affected: float = 0.0
    Sz_frozen: float = 0.0
    Sz_affected: float = 0.0
    fallback_penalty: float = 0.0
    feasibility_penalty: float = 0.0
    duplicate_tote_count: int = 0
    duplicate_tote_penalty: float = 0.0
    coverage_feasible: bool = True
    unmet_sku_total: int = 0
    residual_hat: float = 0.0
    residual_std: float = 0.0
    residual_decay_alpha: float = 0.0
    residual_conf_alpha: float = 0.0
    uncertainty: float = 0.0
    subtask_y_contribs: Dict[int, float] = field(default_factory=dict)
    subtask_z_contribs: Dict[int, float] = field(default_factory=dict)
    affected_subtask_ids: FrozenSet[int] = frozenset()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    validated_makespan: float
    trigger: str
    snapshot: Any
    improved_best: bool = False
    catastrophic_rollback: bool = False
    lkh_call_count: int = 1
    lkh_budget_consumed_by_rollback: int = 0
    validated_cv: float = 0.0


@dataclass
class ValidatedIncumbent:
    config: ResourceConfig
    makespan: float
    iter_id: int
    snapshot: Any


@dataclass
class OperatorArm:
    name: str
    weight: float = 1.0
    pending_rewards: List[float] = field(default_factory=list)
    execution_count: int = 0
    since_update_count: int = 0
    last_update_iter: int = 0

    def record(self, reward: float, iter_id: int) -> None:
        self.pending_rewards.append(float(reward))
        self.execution_count += 1
        self.since_update_count += 1
        self.last_update_iter = int(max(self.last_update_iter, iter_id))
