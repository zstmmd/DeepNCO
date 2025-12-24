import os
from collections import defaultdict
from typing import List, Tuple, Dict
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class GlobalTimeCalculator:
    def __init__(self, problem: OFSProblemDTO):
        self.problem = problem
        self.t_pick = OFSConfig.PICKING_TIME

    def calculate_and_export(self, output_dir: str):
        """
        核心逻辑：计算所有时间节点并导出结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化工作站状态
        for station in self.problem.station_list:
            station.next_available_time = 0.0
            station.total_idle_time = 0.0
            station.processed_tasks = []

        # 收集所有物理任务并按 Trip 分组
        # 结构: { (robot_id, trip_id): [Task, Task, ...] }
        trip_groups: Dict[Tuple[int, int], List] = defaultdict(list)

        all_tasks = []

        #  遍历并分组
        for subtask in self.problem.subtask_list:
            if not subtask.execution_tasks:
                continue
            for task in subtask.execution_tasks:

                r_id = getattr(task, 'robot_id', -1)

                t_id = getattr(task, 'trip_id', 0)

                trip_groups[(r_id, t_id)].append(task)
                all_tasks.append(task)

        # 计算每个 Trip 的统一到达时间
        for (r_id, t_id), tasks_in_trip in trip_groups.items():
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

        # 按到达时间排序
        # 如果时间相同（同一 Trip），可以按 subtask 优先级或 task id 二次排序
        all_tasks.sort(key=lambda t: (t.arrival_time_at_station, t.task_id))

        #  模拟工作站流 (计算 start, end, wait, idle)
        for task in all_tasks:
            station = self.problem.station_list[task.target_station_id]

            # A. 计算捡货时长
            sku_count_in_task = self._calculate_sku_count(task)
            task.picking_duration = sku_count_in_task * self.t_pick

            # B. 计算额外服务时长
            extra_service = 0.0
            if task.noise_tote_ids:
                extra_service = task.station_service_time

            total_process_duration = task.picking_duration + extra_service

            # FCFS 逻辑
            # 开始时间 = max(任务到达时间, 工作站就绪时间)
            start_time = max(task.arrival_time_at_station, station.next_available_time)

            # 记录空闲
            if start_time > station.next_available_time:
                station.total_idle_time += (start_time - station.next_available_time)

            # 记录料箱等待时间
            task.tote_wait_time = start_time - task.arrival_time_at_station

            # 更新时间
            task.start_process_time = start_time
            task.end_process_time = start_time + total_process_duration

            # 更新工作站状态
            station.next_available_time = task.end_process_time
            station.processed_tasks.append(task)

        self._aggregate_times()
        self._write_robot_paths(output_dir)
        self._write_summary(output_dir)


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
        # 注意：这里的 detailed_path 需要在 SP4 解析解的时候填充
        filename = os.path.join(output_dir, "robot_paths.txt")
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
                f"{'TaskID':<8} | {'SubTask':<8} | {'Station':<8} | {'Arrive(S)':<10} | {'StartPick':<10} | {'EndPick':<10} | {'WaitTime':<10} | {'Duration':<10}\n")

            # 获取所有任务并排序
            all_tasks = []
            for st in self.problem.subtask_list:
                all_tasks.extend(st.execution_tasks)
            all_tasks.sort(key=lambda x: x.end_process_time)

            for t in all_tasks:
                f.write(f"{t.task_id:<8} | {t.sub_task_id:<8} | {t.target_station_id:<8} | "
                        f"{t.arrival_time_at_station:<10.2f} | {t.start_process_time:<10.2f} | "
                        f"{t.end_process_time:<10.2f} | {t.tote_wait_time:<10.2f} | {t.picking_duration:<10.2f}\n")