from __future__ import annotations

import copy
from collections import OrderedDict
import math
from typing import Dict, List

from entity.subTask import SubTask
from entity.task import Task
from Gurobi.sp4 import SP4RoutingInfeasibleError

from .state import ResourceConfig, ResourceSubtask, ZTaskDescriptor


class ResourceValidator:
    def __init__(self, opt):
        self.opt = opt
        self.validation_cache: OrderedDict[str, Dict[str, object]] = OrderedDict()
        self.validation_cache_hit_count = 0
        self.validation_cache_miss_count = 0

    def _validation_cache_enabled(self) -> bool:
        cfg = getattr(getattr(self, "opt", None), "cfg", None)
        return bool(getattr(cfg, "resource_validation_cache_enabled", False))

    def _validation_cache_size(self) -> int:
        cfg = getattr(getattr(self, "opt", None), "cfg", None)
        return max(1, int(getattr(cfg, "resource_validation_cache_size", 1024) or 1024))

    def _validation_cache_key(self, config: ResourceConfig) -> str:
        return repr(config.validation_signature())

    def _lookup_validation_cache(self, config: ResourceConfig) -> Dict[str, object] | None:
        if not self._validation_cache_enabled():
            return None
        key = self._validation_cache_key(config)
        entry = self.validation_cache.get(key)
        if entry is None:
            return None
        self.validation_cache.move_to_end(key)
        self.validation_cache_hit_count = int(self.validation_cache_hit_count) + 1
        cached = copy.deepcopy(entry)
        snapshot = cached.get("snapshot", None)
        if snapshot is not None:
            self.opt.restore_snapshot(snapshot)
        cached["lkh_call_count"] = 0
        cached["validation_call_count"] = 0
        cached["validation_cache_hit"] = True
        return cached

    def _store_validation_cache(self, config: ResourceConfig, result: Dict[str, object]) -> None:
        if not self._validation_cache_enabled():
            return
        key = self._validation_cache_key(config)
        cached = copy.deepcopy(dict(result or {}))
        cached["validation_cache_hit"] = False
        self.validation_cache[key] = cached
        self.validation_cache.move_to_end(key)
        while len(self.validation_cache) > self._validation_cache_size():
            self.validation_cache.popitem(last=False)

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

    def _build_conflict_summary(self, config: ResourceConfig, unassigned: Dict[str, object]) -> Dict[str, object]:
        cfg = getattr(getattr(self, "opt", None), "cfg", None)
        bucket_sec = max(1.0, float(getattr(cfg, "resource_conflict_time_bucket_sec", 20.0)))
        sp4_summary = dict(getattr(getattr(self.opt, "sp4", None), "last_conflict_summary", {}) or {})
        failed_task_ids = [int(row.get("task_id", -1)) for row in (unassigned.get("tasks", []) or []) if int(row.get("task_id", -1)) >= 0]
        failed_subtask_ids = [int(row.get("subtask_id", -1)) for row in (unassigned.get("subtasks", []) or []) if int(row.get("subtask_id", -1)) >= 0]
        if not failed_subtask_ids:
            failed_subtask_ids = sorted({int(row.get("subtask_id", -1)) for row in (unassigned.get("tasks", []) or []) if int(row.get("subtask_id", -1)) >= 0})
        station_ids = sorted({
            int(config.subtasks[int(subtask_id)].station_id)
            for subtask_id in failed_subtask_ids
            if int(subtask_id) in config.subtasks and int(config.subtasks[int(subtask_id)].station_id) >= 0
        })
        stack_ids = sorted({
            int(descriptor.stack_id)
            for subtask_id in failed_subtask_ids
            if int(subtask_id) in config.subtasks
            for descriptor in (config.subtasks[int(subtask_id)].z_tasks or [])
            if int(descriptor.stack_id) >= 0
        })
        target_tote_ids = sorted({
            int(tote_id)
            for subtask_id in failed_subtask_ids
            if int(subtask_id) in config.subtasks
            for descriptor in (config.subtasks[int(subtask_id)].z_tasks or [])
            for tote_id in (descriptor.target_tote_ids or ())
            if int(tote_id) >= 0
        })
        time_bucket_ids = sorted({
            int(math.floor(float(max(0, int(config.subtasks[int(subtask_id)].station_rank))) * bucket_sec / bucket_sec))
            for subtask_id in failed_subtask_ids
            if int(subtask_id) in config.subtasks
        })
        summary = {
            "reason_code": str(sp4_summary.get("reason_code", "unassigned_robot_task_hard_reject") or "unassigned_robot_task_hard_reject"),
            "failed_task_ids": sorted(set(failed_task_ids) | set(int(x) for x in (sp4_summary.get("failed_task_ids", []) or []) if int(x) >= 0)),
            "failed_subtask_ids": sorted(set(failed_subtask_ids) | set(int(x) for x in (sp4_summary.get("failed_subtask_ids", []) or []) if int(x) >= 0)),
            "station_ids": sorted(set(station_ids) | set(int(x) for x in (sp4_summary.get("station_ids", []) or []) if int(x) >= 0)),
            "stack_ids": sorted(set(stack_ids) | set(int(x) for x in (sp4_summary.get("stack_ids", []) or []) if int(x) >= 0)),
            "target_tote_ids": list(target_tote_ids),
            "time_bucket_ids": list(time_bucket_ids),
            "time_bucket": sp4_summary.get("time_bucket", time_bucket_ids[0] if time_bucket_ids else None),
            "resource_type": str(sp4_summary.get("resource_type", "robot_capacity") or "robot_capacity"),
        }
        return summary

    def _unreasonable_solution_summary(self, problem=None) -> Dict[str, object]:
        problem = problem if problem is not None else getattr(self.opt, "problem", None)
        failures: List[str] = []
        station_task_rows: List[Dict[str, float | int]] = []
        all_tasks: List[Task] = []
        station_tasks_by_id: Dict[int, List[Task]] = {}
        for subtask in getattr(problem, "subtask_list", []) or []:
            station_id = int(getattr(subtask, "assigned_station_id", -1))
            for task in list(getattr(subtask, "execution_tasks", []) or []):
                all_tasks.append(task)
                station_tasks_by_id.setdefault(int(station_id), []).append(task)
        station_ids = [int(getattr(station, "id", -1)) for station in getattr(problem, "station_list", []) or []]
        for station_id in station_ids:
            station_obj = next(
                (station for station in getattr(problem, "station_list", []) or [] if int(getattr(station, "id", -1)) == int(station_id)),
                None,
            )
            processed_seq = list(getattr(station_obj, "processed_tasks", []) or []) if station_obj is not None else []
            seq_source = processed_seq if processed_seq else list(station_tasks_by_id.get(int(station_id), []) or [])
            seq = sorted(
                seq_source,
                key=lambda task: (
                    float(getattr(task, "start_process_time", 0.0) or 0.0),
                    int(getattr(task, "task_id", -1)),
                ),
            )
            prev_end = 0.0
            for task in seq:
                extra = float(getattr(task, "extra_service_used", 0.0) or 0.0)
                start = float(getattr(task, "start_process_time", 0.0) or 0.0)
                pick = float(getattr(task, "picking_duration", 0.0) or 0.0)
                end = float(getattr(task, "end_process_time", 0.0) or 0.0)
                expected_end = start + pick + extra
                if abs(expected_end - end) > 1e-6:
                    failures.append(
                        f"Task {int(getattr(task, 'task_id', -1))}: end mismatch expected={expected_end:.6f}, actual={end:.6f}"
                    )
                if start + 1e-6 < prev_end:
                    failures.append(
                        f"Station {int(station_id)}: FCFS violation at task {int(getattr(task, 'task_id', -1))}, start={start:.6f} < prev_end={prev_end:.6f}"
                    )
                prev_end = end
                station_task_rows.append(
                    {
                        "station_id": int(station_id),
                        "task_id": int(getattr(task, "task_id", -1)),
                        "start": start,
                        "end": end,
                        "pick": pick,
                        "extra": extra,
                    }
                )
        max_end = max((float(getattr(task, "end_process_time", 0.0) or 0.0) for task in all_tasks), default=0.0)
        global_makespan = float(getattr(problem, "global_makespan", 0.0) or 0.0)
        if abs(max_end - global_makespan) > 1e-6:
            failures.append(f"Global makespan mismatch: max_task_end={max_end:.6f}, global_makespan={global_makespan:.6f}")
        return {
            "has_unreasonable_solution": bool(failures),
            "verification_failures": failures,
            "station_task_rows": station_task_rows,
            "max_task_end": float(max_end),
            "global_makespan": float(global_makespan),
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
        suffix = ""
        config_metadata = getattr(getattr(self, "current_config", None), "metadata", {}) or {}
        if bool(config_metadata.get("joint_colocated_sort_postprocess", False)):
            suffix = " + joint_colocated_sort_postprocess"
        task.sp3_station_service_inputs = f"materialized_from_resource_config{suffix}"
        return task

    def validate(self, config: ResourceConfig, iter_id: int) -> Dict[str, object]:
        self.current_config = config
        cached = self._lookup_validation_cache(config)
        if cached is not None:
            return cached
        self.validation_cache_miss_count = int(self.validation_cache_miss_count) + 1
        coverage = config.coverage_summary(dict(getattr(getattr(self.opt, "problem", None), "id_to_tote", {}) or {}))
        if not bool(coverage.get("coverage_ok", False)):
            result = {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": True,
                "hard_reject_reason": "coverage_hard_reject",
                "conflict_summary": {},
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": 0,
                "unassigned_robot_tasks": [],
                "lkh_call_count": 0,
                "validation_call_count": 1,
            }
            self._store_validation_cache(config, result)
            return result
        self.materialize(config)
        try:
            try:
                self.opt._run_sp4(allow_greedy_fallback=False, raise_on_failure=True)
            except TypeError:
                self.opt._run_sp4()
        except SP4RoutingInfeasibleError as exc:
            conflict_summary = dict(exc.summary or {})
            if not conflict_summary:
                conflict_summary = dict(getattr(getattr(self.opt, "sp4", None), "last_conflict_summary", {}) or {})
            result = {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "unassigned_robot_task_hard_reject",
                "conflict_summary": conflict_summary,
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": int(len(list(conflict_summary.get("failed_task_ids", []) or []))),
                "unassigned_robot_tasks": [{"task_id": int(task_id), "subtask_id": -1} for task_id in (conflict_summary.get("failed_task_ids", []) or [])],
                "lkh_call_count": 1,
                "validation_call_count": 1,
            }
            self._store_validation_cache(config, result)
            return result
        unassigned = self._unassigned_robot_summary()
        if int(unassigned.get("task_count", 0) or 0) > 0:
            conflict_summary = self._build_conflict_summary(config, unassigned)
            result = {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "unassigned_robot_task_hard_reject",
                "conflict_summary": conflict_summary,
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": int(unassigned.get("task_count", 0) or 0),
                "unassigned_robot_tasks": list(unassigned.get("tasks", []) or []),
                "lkh_call_count": 1,
                "validation_call_count": 1,
            }
            self._store_validation_cache(config, result)
            return result
        bom_arrival_window = self.opt._evaluate_bom_arrival_window()
        time_window_metrics = self.opt._evaluate_order_time_window_metrics()
        if not bool(bom_arrival_window.get("feasible", True)):
            result = {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "bom_arrival_window_hard_reject",
                "conflict_summary": {},
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": 0,
                "unassigned_robot_tasks": [],
                "bom_arrival_window_violating_order_count": int(bom_arrival_window.get("violating_order_count", 0) or 0),
                "bom_arrival_window_violations": list(bom_arrival_window.get("violations", []) or []),
                "order_time_window_metrics": dict(time_window_metrics or {}),
                "lkh_call_count": 1,
                "validation_call_count": 1,
            }
            self._store_validation_cache(config, result)
            return result
        span_overrun = float(time_window_metrics.get("span_overrun_total", 0.0) or 0.0)
        deadline_overrun = float(time_window_metrics.get("deadline_overrun_total", 0.0) or 0.0)
        if span_overrun > 1e-9 or deadline_overrun > 1e-9:
            result = {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "kitting_span_hard_reject" if span_overrun > 1e-9 else "order_time_window_hard_reject",
                "conflict_summary": {},
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": 0,
                "unassigned_robot_tasks": [],
                "bom_arrival_window_violating_order_count": int(bom_arrival_window.get("violating_order_count", 0) or 0),
                "bom_arrival_window_violations": list(bom_arrival_window.get("violations", []) or []),
                "order_time_window_metrics": dict(time_window_metrics or {}),
                "lkh_call_count": 1,
                "validation_call_count": 1,
            }
            self._store_validation_cache(config, result)
            return result
        makespan = float(self.opt.evaluate())
        unreasonable_summary = self._unreasonable_solution_summary()
        if bool(unreasonable_summary.get("has_unreasonable_solution", False)) and bool(
            getattr(getattr(self.opt, "cfg", None), "resource_hard_reject_unreasonable_solution", True)
        ):
            result = {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "unreasonable_solution_hard_reject",
                "conflict_summary": {},
                "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
                "unassigned_robot_task_count": 0,
                "unassigned_robot_tasks": [],
                "bom_arrival_window_violating_order_count": int(bom_arrival_window.get("violating_order_count", 0) or 0),
                "bom_arrival_window_violations": list(bom_arrival_window.get("violations", []) or []),
                "order_time_window_metrics": dict(time_window_metrics or {}),
                "has_unreasonable_solution": True,
                "verification_failures": list(unreasonable_summary.get("verification_failures", []) or []),
                "unreasonable_solution_summary": dict(unreasonable_summary or {}),
                "lkh_call_count": 1,
                "validation_call_count": 1,
            }
            self._store_validation_cache(config, result)
            return result
        self.opt._harvest_station_start_times()
        self.opt._update_beta_from_station()
        snapshot = self.opt.snapshot(makespan, iter_id=int(iter_id), lightweight=True)
        result = {
            "makespan": float(makespan),
            "validated_true_makespan": float(makespan),
            "snapshot": snapshot,
            "coverage_hard_reject": False,
            "hard_reject_reason": "",
            "conflict_summary": {},
            "unmet_sku_total": int(coverage.get("unmet_sku_total", 0) or 0),
            "unassigned_robot_task_count": 0,
            "unassigned_robot_tasks": [],
            "bom_arrival_window_violating_order_count": int(bom_arrival_window.get("violating_order_count", 0) or 0),
            "bom_arrival_window_violations": list(bom_arrival_window.get("violations", []) or []),
            "order_time_window_metrics": dict(time_window_metrics or {}),
            "has_unreasonable_solution": False,
            "verification_failures": [],
            "unreasonable_solution_summary": dict(unreasonable_summary or {}),
            "lkh_call_count": 1,
            "validation_call_count": 1,
        }
        self._store_validation_cache(config, result)
        return result
