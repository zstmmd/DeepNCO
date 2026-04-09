from __future__ import annotations

from typing import Dict, List

from entity.subTask import SubTask
from entity.task import Task

from .state import ResourceConfig, ResourceSubtask, ZTaskDescriptor


class ResourceValidator:
    def __init__(self, opt):
        self.opt = opt

    def _unassigned_robot_summary(self) -> Dict[str, object]:
        subtask_rows: List[Dict[str, int]] = []
        task_rows: List[Dict[str, int]] = []
        for subtask in getattr(getattr(self, "opt", None), "problem", None).subtask_list or []:
            execution_tasks = list(getattr(subtask, "execution_tasks", []) or [])
            if execution_tasks and int(getattr(subtask, "assigned_robot_id", -1)) < 0:
                subtask_rows.append(
                    {
                        "subtask_id": int(getattr(subtask, "id", -1)),
                        "task_count": int(len(execution_tasks)),
                    }
                )
            for task in execution_tasks:
                if int(getattr(task, "robot_id", -1)) < 0:
                    task_rows.append(
                        {
                            "task_id": int(getattr(task, "task_id", -1)),
                            "subtask_id": int(getattr(task, "sub_task_id", -1)),
                        }
                    )
        return {
            "task_count": int(len(task_rows)),
            "subtask_count": int(len(subtask_rows)),
            "tasks": task_rows,
            "subtasks": subtask_rows,
        }

    def materialize(self, config: ResourceConfig) -> None:
        problem = self.opt.problem
        assert problem is not None
        config.rebuild_indices()
        order_by_id = {int(getattr(order, "order_id", -1)): order for order in getattr(problem, "order_list", []) or []}
        sku_by_id = {int(getattr(sku, "id", -1)): sku for sku in getattr(problem, "skus_list", []) or []}
        subtasks: List[SubTask] = []
        all_tasks: List[Task] = []

        for station in getattr(problem, "station_list", []) or []:
            try:
                station.processed_tasks = []
                station.total_idle_time = 0.0
                station.next_available_time = 0.0
            except Exception:
                pass

        ordered_rows = sorted(
            list(config.subtasks.values()),
            key=lambda row: (
                int(row.station_id if row.station_id >= 0 else 10**9),
                int(row.station_rank if row.station_rank >= 0 else 10**9),
                int(row.subtask_id),
            ),
        )
        for row in ordered_rows:
            order_obj = order_by_id.get(int(row.order_id))
            if order_obj is None:
                continue
            sku_list = [
                sku_by_id[int(config.work_units[str(work_unit_id)].sku_id)]
                for work_unit_id in (row.work_unit_ids or ())
                if str(work_unit_id) in config.work_units and int(config.work_units[str(work_unit_id)].sku_id) in sku_by_id
            ]
            subtask = SubTask(id=int(row.subtask_id), parent_order=order_obj, sku_list=sku_list)
            subtask.assigned_station_id = int(row.station_id)
            subtask.station_sequence_rank = int(row.station_rank)
            for descriptor in row.z_tasks or []:
                task = self._build_task(row, descriptor)
                stack_obj = problem.point_to_stack.get(int(task.target_stack_id))
                if stack_obj is None:
                    continue
                subtask.add_execution_detail(task, stack_obj)
                all_tasks.append(task)
            subtasks.append(subtask)

        problem.subtask_list = subtasks
        problem.subtask_num = len(subtasks)
        problem.task_list = all_tasks
        problem.task_num = len(all_tasks)
        self.opt._rebuild_solvers()
        self.opt._sync_task_assignments_from_subtasks()
        self.opt._rebuild_problem_task_list()

    def _build_task(self, subtask: ResourceSubtask, descriptor: ZTaskDescriptor) -> Task:
        task = Task(
            task_id=int(descriptor.task_id),
            sub_task_id=int(subtask.subtask_id),
            target_stack_id=int(descriptor.stack_id),
            target_station_id=int(subtask.station_id),
            operation_mode=str(descriptor.mode).upper(),
            station_sequence_rank=int(subtask.station_rank),
            target_tote_ids=list(int(x) for x in (descriptor.target_tote_ids or ())),
            hit_tote_ids=list(int(x) for x in (descriptor.hit_tote_ids or ())),
            noise_tote_ids=list(int(x) for x in (descriptor.noise_tote_ids or ())),
            sort_layer_range=None if descriptor.sort_layer_range is None else (
                int(descriptor.sort_layer_range[0]),
                int(descriptor.sort_layer_range[1]),
            ),
            robot_service_time=float(descriptor.robot_service_time),
            station_service_time=float(descriptor.station_service_time),
            sku_pick_count=int(descriptor.sku_pick_count),
        )
        task.sp3_station_service_source = "RESOURCE_TIME_ALNS"
        task.sp3_station_service_inputs = "materialized_from_resource_config"
        return task

    def validate(self, config: ResourceConfig, iter_id: int) -> Dict[str, object]:
        coverage = config.coverage_summary(dict(getattr(getattr(self.opt, "problem", None), "id_to_tote", {}) or {}))
        if not bool(coverage.get("coverage_ok", False)):
            return {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": True,
                "hard_reject_reason": "coverage_hard_reject",
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": 0,
                "unassigned_robot_tasks": [],
                "lkh_call_count": 0,
            }
        self.materialize(config)
        self.opt._run_sp4()
        unassigned = self._unassigned_robot_summary()
        if int(unassigned.get("task_count", 0) or 0) > 0:
            return {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "unassigned_robot_task_hard_reject",
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": int(unassigned.get("task_count", 0) or 0),
                "unassigned_robot_tasks": list(unassigned.get("tasks", []) or []),
                "lkh_call_count": 1,
            }
        makespan = float(self.opt.evaluate())
        self.opt._harvest_station_start_times()
        self.opt._update_beta_from_station()
        snapshot = self.opt.snapshot(makespan, iter_id=int(iter_id), lightweight=True)
        return {
            "makespan": float(makespan),
            "snapshot": snapshot,
            "coverage_hard_reject": False,
            "hard_reject_reason": "",
            "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
            "unassigned_robot_task_count": 0,
            "unassigned_robot_tasks": [],
            "lkh_call_count": 1,
        }
