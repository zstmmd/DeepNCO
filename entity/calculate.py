import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class GlobalTimeCalculator:
    def __init__(self, problem: OFSProblemDTO):
        self.problem = problem
        self.t_pick = OFSConfig.PICKING_TIME

    def calculate(self) -> float:
        """
        纯计算版本：不落盘，只更新 problem/subtask/task 的时间字段并返回全局 makespan。

        返回值:
            z = problem.global_makespan
        """
        self._reset_station_runtime_state()
        all_tasks = self._collect_all_execution_tasks()
        self._compute_trip_arrival_times(all_tasks)
        self._simulate_station_fcfs(all_tasks)
        self._aggregate_times()
        return float(self.problem.global_makespan)

    def calculate_and_export(self, output_dir: str):
        """
        核心逻辑：计算所有时间节点并导出结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._reset_station_runtime_state()

        all_tasks = self._collect_all_execution_tasks()
        self._compute_trip_arrival_times(all_tasks)
        self._simulate_station_fcfs(all_tasks)

        self._aggregate_times()
        self._write_robot_paths(output_dir)
        self._write_summary(output_dir)
        self._write_task_field_checklist(output_dir)
        self._write_time_breakdown(output_dir, all_tasks)
        self._write_station_fcfs_queue(output_dir)
        self._write_calculation_example(output_dir)
        self._write_consistency_checks(output_dir, all_tasks)

    def _reset_station_runtime_state(self):
        # 初始化工作站状态
        for station in self.problem.station_list:
            station.next_available_time = 0.0
            station.total_idle_time = 0.0
            station.processed_tasks = []

    def _collect_all_execution_tasks(self):
        all_tasks = []
        for subtask in self.problem.subtask_list:
            if not getattr(subtask, "execution_tasks", None):
                continue
            all_tasks.extend(subtask.execution_tasks)
        return all_tasks

    def _compute_trip_arrival_times(self, all_tasks: List):
        # 若 SP4 已回填 delivery 到达站点时间，则优先保留。
        pending_tasks = [t for t in all_tasks if float(getattr(t, "arrival_time_at_station", 0.0)) <= 1e-9]
        if not pending_tasks:
            return

        # 收集所有物理任务并按 Trip 分组
        # 结构: { (robot_id, trip_id): [Task, Task, ...] }
        trip_groups: Dict[Tuple[int, int], List] = defaultdict(list)
        for task in pending_tasks:
            r_id = getattr(task, 'robot_id', -1)
            t_id = getattr(task, 'trip_id', 0)
            trip_groups[(r_id, t_id)].append(task)

        # 计算每个 Trip 的统一到达时间
        for (_r_id, _t_id), tasks_in_trip in trip_groups.items():
            if not tasks_in_trip:
                continue

            last_task = max(tasks_in_trip, key=lambda t: t.arrival_time_at_stack)
            last_stack_obj = self.problem.point_to_stack[last_task.target_stack_id]
            target_station_id = last_task.target_station_id
            station_obj = self.problem.station_list[target_station_id]

            dist = abs(last_stack_obj.store_point.x - station_obj.point.x) + \
                   abs(last_stack_obj.store_point.y - station_obj.point.y)
            return_travel_time = dist / OFSConfig.ROBOT_SPEED

            # 到达时间 = (最后堆垛到达时间) + (最后堆垛作业时间) + (返回路程时间)
            trip_arrival_at_station = last_task.arrival_time_at_stack + \
                                      last_task.robot_service_time + \
                                      return_travel_time

            # 将这个时间赋值给该 Trip 下的所有任务
            for task in tasks_in_trip:
                task.arrival_time_at_station = trip_arrival_at_station

    def _simulate_station_fcfs(self, all_tasks: List):
        # 按到达时间排序
        # 如果时间相同（同一 Trip），可以按 subtask 优先级或 task id 二次排序
        all_tasks.sort(key=lambda t: (
            t.arrival_time_at_station,
            getattr(t, "target_station_id", 0),
            getattr(t, "station_sequence_rank", 0),
            t.task_id,
        ))

        # 模拟工作站流 (计算 start, end, wait, idle)
        for task in all_tasks:
            station = self.problem.station_list[task.target_station_id]

            # A. 计算拣选时长
            sku_count_in_task = self._calculate_sku_count(task)
            task.picking_duration = sku_count_in_task * self.t_pick

            # B. 计算额外服务时长（理货/剔除噪音）
            extra_service = task.station_service_time if getattr(task, "noise_tote_ids", None) else 0.0
            total_process_duration = task.picking_duration + extra_service

            # FCFS 逻辑
            start_time = max(task.arrival_time_at_station, station.next_available_time)

            # 记录空闲
            if start_time > station.next_available_time:
                station.total_idle_time += (start_time - station.next_available_time)

            # 记录料箱等待时间
            task.tote_wait_time = start_time - task.arrival_time_at_station

            # 更新时间
            task.start_process_time = start_time
            task.end_process_time = start_time + total_process_duration
            task.extra_service_used = float(extra_service)
            task.total_process_duration = float(total_process_duration)

            # 更新工作站状态
            station.next_available_time = task.end_process_time
            station.processed_tasks.append(task)

    def _calculate_sku_count(self, task) -> int:
        """
        计算该物理 Task 在工作站需要拣选的 SKU 单位数。

        约定优先使用 SP3 回填的 task.sku_pick_count（更接近“实际贡献”）。
        若缺失，则使用一个保守兜底值，避免仿真崩溃。
        """
        val = getattr(task, "sku_pick_count", 0)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)

        # 兜底：至少拣 1 个，或者按命中料箱数粗略估计
        hit = getattr(task, "hit_tote_ids", None) or []
        return max(1, len(hit))


    def _aggregate_times(self):
        max_end_time = 0.0

        # 更新 SubTask
        for st in self.problem.subtask_list:
            if not st.execution_tasks:
                continue
            st.completion_time = max(t.end_process_time for t in st.execution_tasks)
            max_end_time = max(max_end_time, st.completion_time)

        # 更新 Order
        for order in self.problem.order_list:
            # 找到该 Order 关联的所有 SubTasks
            related_subtasks = [st for st in self.problem.subtask_list if st.parent_order.order_id == order.order_id]
            if related_subtasks:
                order.bom_completion_time = max(st.completion_time for st in related_subtasks)

        self.problem.global_makespan = max_end_time

    def _write_robot_paths(self, output_dir):
        """输出机器人路径到 txt"""
        filename = os.path.join(output_dir, "robot_paths.txt")
        has_task_timeline = False
        for st in getattr(self.problem, "subtask_list", []) or []:
            for t in getattr(st, "execution_tasks", []) or []:
                if float(getattr(t, "arrival_time_at_stack", 0.0)) > 0.0 or float(getattr(t, "arrival_time_at_station", 0.0)) > 0.0:
                    has_task_timeline = True
                    break
            if has_task_timeline:
                break

        # 优先使用 SP4 的原始路由日志，保证时间线连续且与求解器一致。
        sp4_result_path = os.path.join(os.path.dirname(output_dir), "SP4_result.txt")
        if os.path.exists(sp4_result_path) and not has_task_timeline:
            with open(sp4_result_path, "r", encoding="utf-8") as src, open(filename, "w", encoding="utf-8") as dst:
                current_robot = None
                for raw_line in src:
                    line = raw_line.rstrip("\n")
                    if line.startswith("=== Robot ") and line.endswith(" Route ==="):
                        robot_label = line.replace("=== ", "").replace(" Route ===", "")
                        current_robot = robot_label
                        dst.write(f"{current_robot} Path:\n")
                        dst.write("Time, X, Y, Action\n")
                        continue

                    if current_robot is None:
                        continue

                    if line.startswith("  ["):
                        m = re.search(r"\[\s*([0-9.]+)s\]\s*(.*)", line)
                        if m:
                            dst.write(f"{m.group(1)}, -, -, {m.group(2)}\n")
                    elif line.startswith("  -> Total tasks visited"):
                        dst.write("------------------------------\n")
                        current_robot = None
            return

        # 兜底导出：若没有 SP4_result.txt，则按任务时间字段输出。
        with open(filename, 'w') as f:
            for robot in self.problem.robot_list:
                f.write(f"Robot {robot.id} Path:\n")
                f.write("Time, X, Y, Action\n")

                # 收集该机器人所有任务的路径片段
                # 需按时间排序
                robot_tasks = []
                for st in self.problem.subtask_list:
                    if st.assigned_robot_id == robot.id:
                        robot_tasks.extend(st.execution_tasks)

                robot_tasks.sort(key=lambda t: t.arrival_time_at_stack)

                for t in robot_tasks:
                    # 模拟输出：起点 -> 堆垛 -> 工作站
                    # 实际应使用 t.detailed_path
                    stack = self.problem.point_to_stack[t.target_stack_id]
                    station = self.problem.station_list[t.target_station_id]

                    f.write(
                        f"{t.arrival_time_at_stack:.2f}, {stack.store_point.x}, {stack.store_point.y}, Arrive_Stack_{t.target_stack_id}\n")
                    f.write(
                        f"{t.arrival_time_at_station:.2f}, {station.point.x}, {station.point.y}, Arrive_Station_{t.target_station_id}\n")
                f.write("-" * 30 + "\n")

    def _write_summary(self, output_dir):
        filename = os.path.join(output_dir, "simulation_summary.txt")
        with open(filename, 'w') as f:
            f.write(f"Global Makespan: {self.problem.global_makespan:.2f}s\n")
            f.write("=" * 30 + "\n")

            f.write("\n[Station Statistics]\n")
            for s in self.problem.station_list:
                f.write(f"Station {s.id}: Idle Time = {s.total_idle_time:.2f}s\n")

            f.write("\n[Order Completion Times]\n")
            for o in self.problem.order_list:
                f.write(f"Order {o.order_id}: Completed at {o.bom_completion_time:.2f}s\n")

            f.write("\n[Detailed Task Log]\n")
            f.write(
                f"{'TaskID':<8} | {'SubTask':<8} | {'Station':<8} | {'Arrive(S)':<10} | {'StartPick':<10} | {'EndPick':<10} | {'WaitTime':<10} | {'PickDur':<10} | {'ExtraDur':<10} | {'TotalDur':<10}\n")

            # 获取所有任务并排序
            all_tasks = []
            for st in self.problem.subtask_list:
                all_tasks.extend(st.execution_tasks)
            all_tasks.sort(key=lambda x: x.end_process_time)

            for t in all_tasks:
                f.write(f"{t.task_id:<8} | {t.sub_task_id:<8} | {t.target_station_id:<8} | "
                        f"{t.arrival_time_at_station:<10.2f} | {t.start_process_time:<10.2f} | "
                        f"{t.end_process_time:<10.2f} | {t.tote_wait_time:<10.2f} | {t.picking_duration:<10.2f} | "
                        f"{getattr(t, 'extra_service_used', 0.0):<10.2f} | {getattr(t, 'total_process_duration', 0.0):<10.2f}\n")

    def _write_task_field_checklist(self, output_dir: str):
        filename = os.path.join(output_dir, "task_field_checklist.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[Task Fields For TRA Validation]\n")
            f.write("1. task_id\n")
            f.write("2. sub_task_id\n")
            f.write("3. target_station_id\n")
            f.write("4. robot_id\n")
            f.write("5. trip_id\n")
            f.write("6. arrival_time_at_stack\n")
            f.write("7. robot_service_time\n")
            f.write("8. arrival_time_at_station\n")
            f.write("9. sku_pick_count\n")
            f.write("10. picking_duration\n")
            f.write("11. station_service_time\n")
            f.write("12. noise_tote_ids (count)\n")
            f.write("13. start_process_time\n")
            f.write("14. end_process_time\n")
            f.write("15. tote_wait_time\n")
            f.write("16. extra_service_used\n")
            f.write("17. total_process_duration\n")
            f.write("18. sp3_station_service_source\n")
            f.write("19. sp3_station_service_inputs\n")

    def _write_time_breakdown(self, output_dir: str, all_tasks: List[Any]):
        filename = os.path.join(output_dir, "task_time_breakdown.txt")
        rows = sorted(all_tasks, key=lambda t: (t.task_id, t.sub_task_id))
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[Task Time Breakdown]\n")
            f.write(
                "TaskID | SubTask | Station | Robot | Trip | ArriveStack | RobotSrv | ArriveStation | "
                "Start | PickDur | ExtraDur | End | Wait | TotalDur | NoiseCnt | SP3Source\n"
            )
            for t in rows:
                noise_cnt = len(getattr(t, "noise_tote_ids", None) or [])
                f.write(
                    f"{t.task_id:6d} | {t.sub_task_id:7d} | {t.target_station_id:7d} | "
                    f"{int(getattr(t, 'robot_id', -1)):5d} | {int(getattr(t, 'trip_id', 0)):4d} | "
                    f"{float(getattr(t, 'arrival_time_at_stack', 0.0)):11.2f} | {float(getattr(t, 'robot_service_time', 0.0)):8.2f} | "
                    f"{float(getattr(t, 'arrival_time_at_station', 0.0)):13.2f} | {float(getattr(t, 'start_process_time', 0.0)):5.2f} | "
                    f"{float(getattr(t, 'picking_duration', 0.0)):7.2f} | {float(getattr(t, 'extra_service_used', 0.0)):8.2f} | "
                    f"{float(getattr(t, 'end_process_time', 0.0)):5.2f} | {float(getattr(t, 'tote_wait_time', 0.0)):4.2f} | "
                    f"{float(getattr(t, 'total_process_duration', 0.0)):8.2f} | {noise_cnt:8d} | "
                    f"{getattr(t, 'sp3_station_service_source', '')}\n"
                )
                f.write(f"  SP3Inputs: {getattr(t, 'sp3_station_service_inputs', '')}\n")

    def _write_station_fcfs_queue(self, output_dir: str):
        filename = os.path.join(output_dir, "station_fcfs_queue.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[Station FCFS Queue]\n")
            for station in self.problem.station_list:
                f.write(f"\n=== Station {station.id} ===\n")
                f.write(
                    "Seq | TaskID | SubTask | Arrive | Start | End | Wait | PickDur | ExtraDur | TotalDur | NoiseCnt\n"
                )
                seq_tasks = sorted(getattr(station, "processed_tasks", []) or [], key=lambda t: t.start_process_time)
                for idx, t in enumerate(seq_tasks, start=1):
                    noise_cnt = len(getattr(t, "noise_tote_ids", None) or [])
                    f.write(
                        f"{idx:3d} | {t.task_id:6d} | {t.sub_task_id:7d} | {t.arrival_time_at_station:6.2f} | "
                        f"{t.start_process_time:5.2f} | {t.end_process_time:5.2f} | {t.tote_wait_time:4.2f} | "
                        f"{t.picking_duration:7.2f} | {getattr(t, 'extra_service_used', 0.0):8.2f} | "
                        f"{getattr(t, 'total_process_duration', 0.0):8.2f} | {noise_cnt:8d}\n"
                    )

    def _write_calculation_example(self, output_dir: str):
        filename = os.path.join(output_dir, "time_calculation_example.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[Simple Example For Manual Validation]\n")
            f.write("Config: PICKING_TIME = t_pick\n")
            f.write("Case A (with noise):\n")
            f.write("  sku_pick_count = 4, t_pick = 3.00, station_service_time = 6.00, noise_cnt > 0\n")
            f.write("  picking_duration = 4 * 3.00 = 12.00\n")
            f.write("  extra_service_used = 6.00\n")
            f.write("  total_process_duration = 12.00 + 6.00 = 18.00\n")
            f.write("  if arrival=100, station_next_available=110 -> start=max(100,110)=110\n")
            f.write("  end = 110 + 18.00 = 128.00\n")
            f.write("\nCase B (without noise):\n")
            f.write("  sku_pick_count = 4, t_pick = 3.00, station_service_time = 6.00, noise_cnt = 0\n")
            f.write("  picking_duration = 12.00\n")
            f.write("  extra_service_used = 0.00\n")
            f.write("  total_process_duration = 12.00\n")
            f.write("  if arrival=100, station_next_available=110 -> start=110, end=122\n")

    def _write_consistency_checks(self, output_dir: str, all_tasks: List[Any]):
        filename = os.path.join(output_dir, "time_consistency_checks.txt")
        failures = []
        max_end = 0.0
        for t in all_tasks:
            has_noise = len(getattr(t, "noise_tote_ids", None) or []) > 0
            expected_extra = float(getattr(t, "station_service_time", 0.0)) if has_noise else 0.0
            used_extra = float(getattr(t, "extra_service_used", 0.0))
            expected_end = float(getattr(t, "start_process_time", 0.0)) + float(getattr(t, "picking_duration", 0.0)) + used_extra
            actual_end = float(getattr(t, "end_process_time", 0.0))
            max_end = max(max_end, actual_end)

            if abs(expected_extra - used_extra) > 1e-6:
                failures.append(f"Task {t.task_id}: extra mismatch expected={expected_extra:.6f} used={used_extra:.6f}")
            if abs(expected_end - actual_end) > 1e-6:
                failures.append(f"Task {t.task_id}: end mismatch expected={expected_end:.6f} actual={actual_end:.6f}")

        if abs(max_end - float(getattr(self.problem, "global_makespan", 0.0))) > 1e-6:
            failures.append(
                "global_makespan mismatch "
                f"max_end={max_end:.6f} global={float(getattr(self.problem, 'global_makespan', 0.0)):.6f}"
            )

        with open(filename, "w", encoding="utf-8") as f:
            f.write("[Time Consistency Checks]\n")
            if not failures:
                f.write("PASS\n")
            else:
                f.write("FAIL\n")
                for item in failures:
                    f.write(f"- {item}\n")
