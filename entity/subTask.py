from dataclasses import dataclass, field
from typing import List, Optional

from entity.SKUs import SKUs
from entity.order import Order
from entity.point import Point
from entity.stack import Stack
from entity.task import Task


@dataclass
class SubTask:
    """Sub-task created by SP1."""

    id: int
    parent_order: Order
    sku_list: List[SKUs]

    sku_quantity: int = field(init=False)
    unique_sku_list: List[SKUs] = field(init=False)

    assigned_station_id: int = -1
    station_sequence_rank: int = -1
    estimated_process_start_time: float = 0.0
    completion_time: float = 0.0
    est_sec: float = 0.0
    kitting_span_limit_sec: float = 0.0
    lst_sec: float = 0.0
    order_anchor_start_sec: float = 0.0
    order_time_window_lb_sec: float = 0.0
    order_time_window_ub_sec: float = 0.0

    assigned_robot_id: int = -1
    assigned_tote_ids: List[int] = field(default_factory=list)
    execution_tasks: List[Task] = field(default_factory=list)
    involved_stacks: List[Stack] = field(default_factory=list)
    visit_points: List[Point] = field(default_factory=list)

    _cached_start_pt: Optional[Point] = field(default=None, init=False)
    _cached_end_pt: Optional[Point] = field(default=None, init=False)
    _cached_duration: float = field(default=0.0, init=False)

    def add_execution_detail(self, task_obj: Task, stack_obj: Stack):
        self.execution_tasks.append(task_obj)
        if not any(int(getattr(s, "stack_id", -1)) == int(getattr(stack_obj, "stack_id", -1)) for s in self.involved_stacks):
            self.involved_stacks.append(stack_obj)
            if getattr(stack_obj, "store_point", None):
                self.visit_points.append(stack_obj.store_point)
        current_totes = set(self.assigned_tote_ids)
        new_totes = set(getattr(task_obj, "target_tote_ids", []) or [])
        self.assigned_tote_ids = list(current_totes.union(new_totes))

    def __post_init__(self):
        seen_ids = set()
        self.unique_sku_list = []
        for sku in self.sku_list:
            if int(getattr(sku, "id", -1)) not in seen_ids:
                self.unique_sku_list.append(sku)
                seen_ids.add(int(getattr(sku, "id", -1)))
        self.sku_quantity = len(self.unique_sku_list)
        self.est_sec = float(getattr(self.parent_order, "est_sec", 0.0) or 0.0)
        self.kitting_span_limit_sec = float(getattr(self.parent_order, "kitting_span_limit_sec", 0.0) or 0.0)
        self.lst_sec = float(getattr(self.parent_order, "lst_sec", 0.0) or 0.0)

    def reset_execution_details(self):
        self.assigned_tote_ids = []
        self.execution_tasks = []
        self.involved_stacks = []
        self.visit_points = []
        self.completion_time = 0.0
        self.assigned_robot_id = -1
        self._cached_start_pt = None
        self._cached_end_pt = None
        self._cached_duration = 0.0

    @property
    def capacity_usage(self) -> int:
        return len(set(int(getattr(s, "id", -1)) for s in self.sku_list))

    def confirm_allocation(self, tote_ids: List[int]):
        self.assigned_tote_ids = list(tote_ids)

    def __str__(self):
        return f"SubTask(id={self.id}, unique_types={self.sku_quantity})"
