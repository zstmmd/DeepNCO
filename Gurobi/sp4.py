import math
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import os
import sys
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from entity.subTask import SubTask
from entity.task import Task
from entity.robot import Robot
from entity.point import Point
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class SP4_Robot_Router:
    """
    SP4 子问题求解器：任务-机器人分配与路径规划

    核心逻辑：
    1. 基于 SP3 确定的堆垛访问需求，为每个 SubTask 分配机器人
    2. 规划机器人访问堆垛的顺序（TSP with Capacity）
    3. 计算到达时间并反馈给 SP2
    """

    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem = problem_dto
        self.robot_capacity = OFSConfig.ROBOT_CAPACITY
        self.robot_speed = OFSConfig.ROBOT_SPEED
        self.t_shift = OFSConfig.PACKING_TIME
        self.t_lift = OFSConfig.LIFTING_TIME
        self.lkh_time_limit_seconds = 100
        # --- 初始化 Logger ---
        log_dir = os.path.join(ROOT_DIR, 'log')
        # 实例化 logger
        self.logger = SP4Logger(log_dir, filename="sp4_debug.txt")

    def _extract_sequence(self, x, y, T, trip, nodes_map, N, R, depot_layer_nodes, robot_start_nodes,
                          stack_nodes_indices):
        """
        提取机器人路径（使用二维时间变量）
        """
        for r in R:
            print(f"\n  === Robot {r} Routes ===")

            visited_nodes = []
            for i in stack_nodes_indices:  # 只遍历 Stack 节点
                if y[i, r].X > 0.5:
                    # 🔧 修复：使用二维 T 变量
                    arrival_time = T[i, r].X
                    trip_idx = int(trip[i, r].X) if (i, r) in trip else 0
                    pt, subtask, task_obj, _, _ = nodes_map[i]
                    visited_nodes.append((arrival_time, i, task_obj, trip_idx))

            if not visited_nodes:
                print(f"  No tasks assigned")
                continue

            visited_nodes.sort(key=lambda x: (x[3], x[0]))

            trips = defaultdict(list)
            for time, node_id, task_obj, trip_num in visited_nodes:
                trips[trip_num].append((time, node_id, task_obj))

            for trip_idx in sorted(trips.keys()):
                trip_nodes = trips[trip_idx]
                if trip_nodes:
                    start_time = trip_nodes[0][0]
                    end_time = trip_nodes[-1][0]
                    total_load = sum(task_obj.total_load_count for _, _, task_obj in trip_nodes)

                    print(f"  Trip {trip_idx}: {len(trip_nodes)} tasks, "
                          f"load={total_load}/{self.robot_capacity}, "
                          f"time [{start_time:.1f}s, {end_time:.1f}s]")

                    for seq, (time, node_id, task_obj) in enumerate(trip_nodes):
                        task_obj.robot_visit_sequence = seq
                        print(f"    [{seq}] Stack {task_obj.target_stack_id} @ {time:.1f}s "
                              f"(SubTask {task_obj.sub_task_id}, Load={task_obj.total_load_count})")

    def _solve_LKH(self, sub_tasks: List, soft_time_windows: Optional[Dict[int, float]] = None, mu: float = 0.0) -> Tuple[Dict[int, float], Dict[int, int]]:
        print(f"  >>> [SP4] Starting LKH-based Routing (OR-Tools)...")
        quiet_mode = os.environ.get("OFS_BATCH_SILENT", "0") == "1"

        # 1. 提取有效任务
        valid_tasks = [st for st in sub_tasks if st.execution_tasks]
        if not valid_tasks:
            return {}, {}
        num_vehicles = len(self.problem.robot_list)
        depot_pt = self.problem.robot_list[0].start_point
        # 2. 构建节点映射 (Node Mapping)
        # Index 0: Depot (起点和终点)
        # Index 1 ~ K: Pickups (取货点 P)
        # Index K+1 ~ 2K: Deliveries (卸货点 D)

        nodes_info = []  # 存储 (point_obj, task_obj, n_type)
        nodes_info.append((depot_pt, None, 'depot'))  # Index 0

        pickups = []
        deliveries = []
        pair_map = []  # [(pickup_index, delivery_index), ...]
        subtask_groups = defaultdict(list)  # subtask_id -> [pickup_indices]

        # 解析任务生成 P 和 D
        for st in valid_tasks:
            for task in st.execution_tasks:
                stack_pt = self.problem.point_to_stack[task.target_stack_id].store_point
                station_pt = self.problem.station_list[st.assigned_station_id].point

                # 暂存 P 和 D 信息
                pickups.append((stack_pt, task, 'pickup', st.id))
                deliveries.append((station_pt, task, 'delivery', st.id))

        # 拼装 nodes_info: [Depot] + [All Pickups] + [All Deliveries]
        num_tasks = len(pickups)
        for p in pickups:
            nodes_info.append((p[0], p[1], p[2]))
        for d in deliveries:
            nodes_info.append((d[0], d[1], d[2]))

        # 构建配对关系
        for i in range(num_tasks):
            p_idx = i + 1
            d_idx = num_tasks + i + 1
            pair_map.append((p_idx, d_idx))

            st_id = pickups[i][3]
            subtask_groups[st_id].append(p_idx)

        # 3. 构建距离矩阵和需求数组
        num_nodes = len(nodes_info)
        distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        demands = [0] * num_nodes
        service_times = [0] * num_nodes

        for i in range(num_nodes):
            pt_i, task_i, type_i = nodes_info[i]

            # 提取需求和耗时
            if type_i == 'pickup':
                demands[i] = task_i.total_load_count
                service_times[i] = int(task_i.robot_service_time)
            elif type_i == 'delivery':
                demands[i] = -task_i.total_load_count
                service_times[i] = 0  # 卸货耗时

            for j in range(num_nodes):
                pt_j, _, _ = nodes_info[j]
                # 计算曼哈顿距离耗时 (放大10倍转为整数，OR-Tools 需要整数)
                dist_time = (abs(pt_i.x - pt_j.x) + abs(pt_i.y - pt_j.y)) / self.robot_speed
                distance_matrix[i][j] = int(dist_time * 10)

        # 4. 初始化 OR-Tools Routing Model
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        # =====================================================================
        # 准备工作：在前面循环构建 deliveries 列表时，记录 task_id -> delivery_index
        # =====================================================================
        task_to_delivery_index = {}
        # 假设 deliveries 列表存的是卸货点 D 节点
        for offset, (pt, task_obj, n_type, st_id) in enumerate(deliveries):
            # 根据 OR-Tools 的节点编号规则：0是起点，1~K是P，K+1~2K是D
            d_node = num_tasks + offset + 1
            task_to_delivery_index[task_obj.task_id] = manager.NodeToIndex(d_node)


        # 5.1 距离/时间 回调
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # 时间 = 行驶时间 + 节点服务时间
            return distance_matrix[from_node][to_node] + (service_times[from_node] * 10)

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 5.2 容量 回调
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # 6. 添加维度约束 (Dimensions)
        # 6.1 容量约束
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.robot_capacity] *num_vehicles,  # 每个机器人的最大容量
            True,  # 起点容量为0
            'Capacity'
        )

        # 6.2 时间维度 (用于记录到达时间和满足先后顺序)
        time = "Time"
        routing.AddDimension(
            transit_callback_index,
            30000,  # 允许的最大等待时间 slack
            300000,  # 每天最大时间 (极大值)
            True,  # 不强制起始时间为 0，但通常是 0
            time
        )
        time_dimension = routing.GetDimensionOrDie(time)
        time_dimension.SetGlobalSpanCostCoefficient(100)

        # --- Soft upper bounds for deliveries (t_latest) ---
        if soft_time_windows:
            # Build task -> delivery index map (already below for logs; reconstruct here for safety)
            try:
                # reconstruct mapping for this scope
                num_tasks = len([t for st in valid_tasks for t in st.execution_tasks])
                # We rebuild pair_map as in original code
                # Note: nodes_info: [0]=Depot, [1..K]=Pickups, [K+1..2K]=Deliveries
                # We'll map by scanning deliveries constructed above
                # We replicate minimal part to map task_id -> delivery index in manager space
                task_to_delivery_index = {}
                offset = 0
                # Rebuild deliveries list like above
                deliveries_tmp = []
                for st in valid_tasks:
                    for task in st.execution_tasks:
                        station_pt = self.problem.station_list[st.assigned_station_id].point
                        deliveries_tmp.append((station_pt, task, 'delivery', st.id))
                for idx, (_pt, task_obj, _nt, _sid) in enumerate(deliveries_tmp):
                    d_node = num_tasks + idx + 1
                    task_to_delivery_index[task_obj.task_id] = manager.NodeToIndex(d_node)
                # apply soft UB
                penalty = max(0, int(round(mu * 10)))
                for tid, latest in soft_time_windows.items():
                    if tid in task_to_delivery_index:
                        idx = task_to_delivery_index[tid]
                        time_dimension.SetCumulVarSoftUpperBound(idx, int(round(float(latest) * 10)), penalty)
            except Exception:
                pass
    
        solver = routing.solver()

        # 7.1 取送货成对，且 P 在 D 前面
        for p_idx, d_idx in pair_map:
            p_index = manager.NodeToIndex(p_idx)
            d_index = manager.NodeToIndex(d_idx)

            # 必须由同一台车完成
            routing.AddPickupAndDelivery(p_idx, d_idx)
            solver.Add(routing.VehicleVar(p_index) == routing.VehicleVar(d_index))
            # P 必须在 D 之前
            solver.Add(time_dimension.CumulVar(p_index) <= time_dimension.CumulVar(d_index))

        # 4.2 同 SubTask 同车约束
        for st_id, p_indices in subtask_groups.items():
            if len(p_indices) > 1:
                base_index = manager.NodeToIndex(p_indices[0])
                for other_p in p_indices[1:]:
                    other_index = manager.NodeToIndex(other_p)
                    solver.Add(routing.ActiveVar(base_index) == routing.ActiveVar(other_index))
                    solver.Add(routing.VehicleVar(base_index) == routing.VehicleVar(other_index))

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = int(self.lkh_time_limit_seconds)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time_dimension.SetGlobalSpanCostCoefficient(10000)
        # 9. 求解
        solution = routing.SolveWithParameters(search_parameters)
        verbose_console = bool(getattr(self.problem, "runtime_verbose_sp4", False)) and not bool(quiet_mode)


        # 诊断输出
        if verbose_console:
            print(f"  >>> [Diag] num_nodes={num_nodes}, num_vehicles={num_vehicles}")
            print(f"  >>> [Diag] num_tasks={num_tasks}, pairs={len(pair_map)}")
            print(f"  >>> [Diag] total demand={sum(d for d in demands if d > 0)}, capacity={self.robot_capacity}")
            print(f"  >>> [Diag] subtask_groups={dict(subtask_groups)}")
            print(f"  >>> [Diag] solution={solution}")
        if solution is None:
            if verbose_console:
                print(f"  >>> [Diag] routing status={routing.status()}")
            # 0=NOT_SOLVED, 1=SUCCESS, 2=PARTIAL_SUCCESS, 3=FAIL, 4=FAIL_TIMEOUT, 5=INVALID
        result_times = {}
        result_assign = {}

        if solution:
            obj_val = solution.ObjectiveValue() / 10.0
            if verbose_console:
                print(f"\n" + "=" * 50)
                print(f"[OK] LKH Engine Solved. Total Objective: {obj_val:.1f}s")
                print("=" * 50)

            # --- 新增：准备日志文件 ---
            runtime_result_dir = getattr(self.problem, "runtime_result_dir", None)
            log_dir = str(runtime_result_dir) if runtime_result_dir else os.path.join(ROOT_DIR, 'log')
            write_result_log = bool(getattr(self.problem, "runtime_write_sp4_result", False)) and not bool(quiet_mode)
            if write_result_log and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            result_log_path = os.path.join(log_dir, 'SP4_result.txt') if write_result_log else os.devnull

            with open(result_log_path, 'w', encoding='utf-8') as f_log:
                f_log.write(f"=== SP4 LKH Routing Final Results ===\n")
                f_log.write(f"Total Objective (Makespan/Cost): {obj_val:.1f}s\n\n")

                # 获取容量维度，用于打印实时载重
                capacity_dimension = routing.GetDimensionOrDie('Capacity')

                for vehicle_id in range(num_vehicles):
                    robot_id = self.problem.robot_list[vehicle_id].id
                    header = f"\n=== Robot {robot_id} Route ==="
                    if verbose_console:
                        print(header)
                    f_log.write(header + "\n")

                    index = routing.Start(vehicle_id)
                    step_count = 0

                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        pt, task_obj, n_type = nodes_info[node_index]

                        # 从解中提取时间和载重 (除以 10 还原真实时间)
                        arr_time = solution.Value(time_dimension.CumulVar(index)) / 10.0
                        current_load = solution.Value(capacity_dimension.CumulVar(index))

                        # 提取坐标
                        coord = f"({pt.x:.1f}, {pt.y:.1f})"

                        if n_type == 'depot':
                            line = f"  [{arr_time:>5.1f}s] Start @ Depot {coord} | Load: {current_load}/{self.robot_capacity}"

                        elif n_type == 'pickup':
                            demand_val = task_obj.total_load_count
                            line = (
                                f"  [{arr_time:>5.1f}s] PICKUP  (SubTask: {task_obj.sub_task_id}, Task: {task_obj.task_id}) "
                                f"@ Stack {task_obj.target_stack_id} {coord} | Load: +{demand_val} -> {current_load}/{self.robot_capacity}")

                            # 回填结果
                            result_times[pt.idx] = arr_time
                            task_obj.arrival_time_at_stack = arr_time
                            task_obj.robot_id = robot_id
                            result_assign[task_obj.sub_task_id] = robot_id

                        elif n_type == 'delivery':
                            demand_val = task_obj.total_load_count
                            rank_val = getattr(task_obj, 'station_sequence_rank', 'N/A')
                            line = (
                                f"  [{arr_time:>5.1f}s] DELIVER (SubTask: {task_obj.sub_task_id}, Task: {task_obj.task_id}, Rank: {rank_val}) "
                                f"@ Station {task_obj.target_station_id} {coord} | Load: -{demand_val} -> {current_load}/{self.robot_capacity}")
                            # 回填站点到达时间（优先使用 SP4 的实际到达时间）
                            task_obj.arrival_time_at_station = arr_time

                        if verbose_console:
                            print(line)
                        f_log.write(line + "\n")

                        # 步进到下一个节点
                        index = solution.Value(routing.NextVar(index))
                        step_count += 1

                    # 打印终点信息 (返回 Depot)
                    node_index = manager.IndexToNode(index)
                    pt, _, _ = nodes_info[node_index]
                    coord = f"({pt.x:.1f}, {pt.y:.1f})"
                    end_time = solution.Value(time_dimension.CumulVar(index)) / 10.0
                    final_load = solution.Value(capacity_dimension.CumulVar(index))

                    end_line = f"  [{end_time:>5.1f}s] End @ Depot {coord}   | Load: {final_load}/{self.robot_capacity}"
                    summary_line = f"  -> Total tasks visited: {step_count - 1}, Return Time: {end_time:.1f}s\n"

                    if verbose_console:
                        print(end_line)
                        print(summary_line)
                    f_log.write(end_line + "\n")
                    f_log.write(summary_line + "\n")

            if verbose_console:
                print(f"  >>> [Log] Final results written to {result_log_path}")
            return result_times, result_assign
        else:
            print("  >>> [Error] LKH Engine Failed to find a solution (Infeasible).")
            # 可能是容量限制太死，或者有些点根本无法到达
            return {}, {}

    def solve(self,
              sub_tasks: List[SubTask],
              use_mip: bool = True,
              lkh_time_limit_seconds: Optional[int] = None,
              soft_time_windows: Optional[Dict[int, float]] = None,
              mu: float = 0.0) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        执行求解

        :param sub_tasks: SP3 已完成选箱的子任务列表
        :param use_mip: 是否使用 MIP 精确求解
        :return: (robot_arrival_times, subtask_robot_assignment)
                 - robot_arrival_times: {point_idx: arrival_time}
                 - subtask_robot_assignment: {subtask_id: robot_id}
        """
        print(f"  >>> [SP4] Starting Robot Routing (MIP={use_mip})...")

        if use_mip:
            return self._solve_mip_pdp_v2(sub_tasks)
        else:
            if lkh_time_limit_seconds is not None:
                self.lkh_time_limit_seconds = int(lkh_time_limit_seconds)
            return self._solve_LKH(sub_tasks, soft_time_windows=soft_time_windows, mu=mu)

    def _solve_mip_pdp_v2(self, sub_tasks: List[SubTask]) -> Tuple[Dict[int, float], Dict[int, int]]:
        r"""
        PDP-VRP 建模

        节点定义:
          - 1个 robot_start (所有机器人共享起点)
          - 每个 Task -> 1个 P节点 (取货点, 对应Stack位置)
          - 每个 Task -> 1个 D节点 (卸货点, 对应Station位置)
          - 1个 robot_end  (所有机器人共享终点, 位置=起点)

        核心约束:
          - 每个 (P_i, D_i) 对必须由同一机器人服务
          - P_i 必须在 D_i 之前被访问 (T[P_i] < T[D_i])
          - 同一 SubTask 的所有 P 必须由同一机器人服务
          - 可以是P1-P2-P3-D1-D2-D3,只要容量允许
          - 假设task $i$ 和 $j$ 都去往同一个工站 $s$，且 $i$ 的 rank 小于 $j$ 的 rank，到达时间约束：$T_{j, delivery} \ge T_{i, delivery}$ (任务 $j$ 的交付时间必须晚于任务 $i$)路径修剪（极大幅度减小空间）：如果在同一个机器人 $r$ 上，不可能出现从 $j$ 走到 $i$ 的路径，直接令 $x_{j, i, r} = 0$。即Di的路径在Dj之前
          - D1D2D3可以在不同SubTask之间交叉，只要不违反优先级和容量约束
        """

        # ============================================================
        # 1. 数据准备
        # ============================================================
        valid_tasks = [st for st in sub_tasks if st.execution_tasks]
        if not valid_tasks:
            return {}, {}
        station_task_ranks = defaultdict(list)
        for st in valid_tasks:
            for task in st.execution_tasks:
                # 使用 sub_task 的 rank 或者 task 自带的 rank
                rank = getattr(task, 'station_sequence_rank', getattr(st, 'station_sequence_rank', 0))
                station_task_ranks[st.assigned_station_id].append((rank, task.task_id))

        # 记录前置依赖: precedes_map[task_a] = [task_b, task_c] 表示 A 必须在 B 和 C 之前完成
        precedes_map = defaultdict(list)
        for sid, tasks_in_station in station_task_ranks.items():
            # 按 rank 升序排序
            tasks_in_station.sort(key=lambda x: x[0])
            # 构建依赖链
            for i in range(len(tasks_in_station)):
                for j in range(i + 1, len(tasks_in_station)):
                    rank_i, tid_i = tasks_in_station[i]
                    rank_j, tid_j = tasks_in_station[j]
                    if rank_i < rank_j:  # 严格小于才有先后驱
                        precedes_map[tid_i].append(tid_j)
        # ============================================================
        # 2. 节点构建
        # ============================================================
        nodes_map = {}  # {node_id: (point, subtask, task_obj, task.total,type, extra)}
        node_id = 0

        # (A) 起点
        start_pt = self.problem.robot_list[0].start_point
        start_node = node_id
        nodes_map[node_id] = (start_pt, None, None, None,'start', -1)
        node_id += 1

        # (B) 终点 (位置与起点相同)
        end_node = node_id
        nodes_map[node_id] = (start_pt, None, None,None, 'end', -1)
        node_id += 1

        pair_map = {}
        p_nodes = []
        d_nodes = []
        subtask_p_nodes = defaultdict(list)
        task_station_map = {}
        node_to_task_id = {}  # 方便查找节点对应的原始 task_id

        for st in valid_tasks:
            station_id = st.assigned_station_id
            station_pt = self.problem.station_list[station_id].point

            for task in st.execution_tasks:
                stack_obj = self.problem.point_to_stack[task.target_stack_id]
                stack_pt = stack_obj.store_point

                # P 节点
                p_id = node_id
                nodes_map[node_id] = (stack_pt, st, task, task.total_load_count, 'pickup', station_id)
                p_nodes.append(p_id)
                subtask_p_nodes[st.id].append(p_id)
                task_station_map[p_id] = station_id
                node_to_task_id[p_id] = task.task_id
                node_id += 1

                # D 节点
                d_id = node_id
                nodes_map[node_id] = (station_pt, st, task, task.total_load_count, 'delivery', station_id)
                d_nodes.append(d_id)
                task_station_map[d_id] = station_id
                node_to_task_id[d_id] = task.task_id
                node_id += 1

                pair_map[task.task_id] = (p_id, d_id)

        N = list(range(node_id))
        R = list(range(len(self.problem.robot_list)))
        num_robots = len(R)
        self.logger.log_node_definitions(nodes_map)
        # ============================================================
        # 3. 距离矩阵 & 弧集合 (剪枝)
        # ============================================================
        tau = {}  # {(i,j): travel_time}

        def get_travel_time(n1, n2):
            p1 = nodes_map[n1][0]
            p2 = nodes_map[n2][0]
            return (abs(p1.x - p2.x) + abs(p1.y - p2.y)) / self.robot_speed

        def is_rank_violation(node_from, node_to):
            """
            检查是否违反了 Rank 优先级（只限制 Delivery 的顺序，放开 Pickup 的顺序）。
            目标：保证对于同一工站的任务，Rank小的（紧急）的 Delivery 必须在 Rank大的 Delivery 之前。
            """
            # 过滤起点和终点
            if node_from not in node_to_task_id or node_to not in node_to_task_id:
                return False

            t_from = node_to_task_id[node_from]
            t_to = node_to_task_id[node_to]

            # 如果是同一个任务内部的连线，不涉及跨任务比较
            if t_from == t_to:
                return False

            type_from = nodes_map[node_from][4]  # 'pickup' 或 'delivery'
            type_to = nodes_map[node_to][4]  # 'pickup' 或 'delivery'

            # precedes_map[A] = [B, C] 代表 A 比 B,C 紧急，A 的 Delivery 必须先发生

            # 场景 1: t_to 是前置任务(紧急，如A), t_from 是后置任务(次急，如B)
            # 即规定必须 Da < Db
            if t_from in precedes_map.get(t_to, []):
                # 我们正在尝试从 B 走向 A。
                # 只有当离开的节点是 B 的卸货点 (Db) 时，才违规。
                # (即拦截了 Db -> Pa 和 Db -> Da)
                # 允许了 Pb -> Pa 和 Pb -> Da
                if type_from == 'delivery':
                    return True

            # 场景 2: t_from 是前置任务(紧急，如A), t_to 是后置任务(次急，如B)
            # 即规定必须 Da < Db
            if t_to in precedes_map.get(t_from, []):
                # 我们正在尝试从 A 走向 B。
                # 如果从 A 的取货点 (Pa) 直接走向 B 的卸货点 (Db)，
                # 意味着 A 还没卸货，就先把 B 卸了，违规！
                # (即拦截了 Pa -> Db)
                if type_from == 'pickup' and type_to == 'delivery':
                    return True

            return False
        for i in N:
            type_i = nodes_map[i][4]
            station_i = task_station_map.get(i, -1)

            for j in N:
                if i == j:
                    continue
                type_j = nodes_map[j][4]
                station_j = task_station_map.get(j, -1)

                # ---- 剪枝规则 ----

                # 终点没有出边
                if type_i == 'end':
                    continue

                if is_rank_violation(i, j):
                    continue
                # 起点 -> P 节点
                if type_i == 'start' and type_j == 'pickup':
                    tau[i, j] = get_travel_time(i, j)

                # 起点 -> 终点
                elif type_i == 'start' and type_j == 'end':
                    tau[i, j] = 0.0

                # P -> D
                elif type_i == 'pickup' and type_j == 'delivery':
                    tau[i, j] = get_travel_time(i, j)

                # P -> P
                elif type_i == 'pickup' and type_j == 'pickup':
                    # 必须两个task需要的容量加起来不超过机器人容量（同一趟内）
                    demand_i = nodes_map[i][3]
                    demand_j = nodes_map[j][3]
                    if demand_i + demand_j <= self.robot_capacity:
                        tau[i, j] = get_travel_time(i, j)

                # D -> P
                elif type_i == 'delivery' and type_j == 'pickup':
                    #必须是不同的stack的pd
                    task_i=nodes_map[i][2].task_id
                    task_j= nodes_map[j][2].task_id
                    if task_i != task_j:
                        tau[i, j] = get_travel_time(i, j)

                # D -> D
                elif type_i == 'delivery' and type_j == 'delivery':
                
                    tau[i, j] = get_travel_time(i, j)

                # D -> 终点 (允许)
                elif type_i == 'delivery' and type_j == 'end':
                    tau[i, j] = get_travel_time(i, j)

        arcs = list(tau.keys())

        # ============================================================
        # 4. 参数提取
        # ============================================================
        service_time = {}  # 节点服务时间
        demand = {}  # 节点需求量（P为正，D为负）

        for i in N:
            _, _, task_obj, _,n_type, _ = nodes_map[i]
            if n_type == 'pickup':
                service_time[i] = task_obj.robot_service_time
                demand[i] = task_obj.total_load_count  # 装货
            elif n_type == 'delivery':
                service_time[i] = 0.0
                demand[i] = -task_obj.total_load_count  # 卸货
            else:
                service_time[i] = 0.0
                demand[i] = 0

        M_time = 10000
        C = self.robot_capacity
        with gp.Model("SP4_PDP_v2") as model:
            model.Params.OutputFlag = 1
            model.Params.MIPGap = 0.02
            model.Params.TimeLimit = 180
            model.Params.LazyConstraints = 1

            # --- 决策变量 ---
            # x[i,j,r]: 机器人r走边(i->j)
            x = model.addVars(
                [(i, j, r) for (i, j) in arcs for r in R],
                vtype=GRB.BINARY, name="x"
            )
            # y[i,r]: 机器人r访问节点i
            y = model.addVars(N, R, vtype=GRB.BINARY, name="y")
            # T[i,r]: 机器人r到达节点i的时间
            T = model.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="T")
            # Q[i,r]: 机器人r离开节点i时的负载
            Q = model.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, ub=C, name="Q")
            # Z: Makespan
            Z = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Z")

            # ============================================================
            # 6. 约束
            # ============================================================

            # --- 约束1: 起点/终点 ---
            for r in R:
                # 每个机器人必须从起点出发
                model.addConstr(y[start_node, r] == 1, name=f"StartVisit_{r}")
                model.addConstr(T[start_node, r] == 0, name=f"StartTime_{r}")
                model.addConstr(Q[start_node, r] == 0, name=f"StartLoad_{r}")
                # 起点出度 = 1
                model.addConstr(
                    gp.quicksum(x[start_node, j, r] for j in N if (start_node, j) in tau) == 1,
                    name=f"StartOut_{r}"
                )
                # 终点入度 = 1
                model.addConstr(
                    gp.quicksum(x[i, end_node, r] for i in N if (i, end_node) in tau) == 1,
                    name=f"EndIn_{r}"
                )
                # 终点出度 = 0（终点没有出边，由剪枝保证）

            # --- 约束2: 覆盖约束（每个P节点被恰好一个机器人访问）---
            for p in p_nodes:
                model.addConstr(
                    gp.quicksum(y[p, r] for r in R) == 1,
                    name=f"CoverP_{p}"
                )

            # --- 约束3: P-D配对约束（同一机器人服务）---
            for tasks, (p_id, d_id) in pair_map.items():
                for r in R:
                    # P和D必须由同一机器人服务
                    model.addConstr(y[p_id, r] == y[d_id, r], name=f"PairSameRobot_{p_id}_{d_id}_{r}")

            # --- 约束4: 流守恒（中间节点入度=出度）---
            for r in R:
                for i in p_nodes + d_nodes:
                    in_flow = gp.quicksum(x[j, i, r] for j in N if (j, i) in tau)
                    out_flow = gp.quicksum(x[i, j, r] for j in N if (i, j) in tau)
                    model.addConstr(in_flow == y[i, r], name=f"FlowIn_{i}_{r}")
                    model.addConstr(out_flow == y[i, r], name=f"FlowOut_{i}_{r}")

            # --- 约束5: 优先级约束（P必须在D之前）---
            # 使用时间变量直接保证: T[P] + service[P] + travel(P,D) <= T[D]
            for tasks, (p_id, d_id) in pair_map.items():
                for r in R:
                    # 若机器人r服务该对，则T[D] >= T[P] + service[P]
                    # 使用 Big-M 松弛
                    model.addConstr(
                        T[d_id, r] >= T[p_id, r] + service_time[p_id] - M_time * (1 - y[p_id, r]),
                        name=f"PrecedenceTime_{p_id}_{d_id}_{r}"
                    )

            # --- 约束6: 同SubTask同机器人 ---
            for st_id, p_node_list in subtask_p_nodes.items():
                if len(p_node_list) > 1:
                    base = p_node_list[0]
                    for other in p_node_list[1:]:
                        for r in R:
                            model.addConstr(
                                y[base, r] == y[other, r],
                                name=f"SameRobotST_{st_id}_{base}_{other}_{r}"
                            )

            # --- 约束7: 容量约束（MTZ风格）---
            for r in R:
                # 起点负载=0
                model.addConstr(Q[start_node, r] == 0)

                for i in N:
                    model.addConstr(T[i, r] <= M_time * y[i, r], name=f"GhostTimeBind_{i}_{r}")
                for i, j in arcs:
                    if (i, j, r) not in x:
                        continue
                    d_i = demand[i]  # P节点>0, D节点<0

                    # Q[j] = Q[i] + demand[j] (线性化)
                    # Big-M: Q[j] >= Q[i] + demand[j] - C*(1-x[i,j,r])
                    #        Q[j] <= Q[i] + demand[j] + C*(1-x[i,j,r])
                    model.addConstr(
                        Q[j, r] >= Q[i, r] + demand[j] - C * (1 - x[i, j, r]),
                        name=f"LoadLB_{i}_{j}_{r}"
                    )
                    model.addConstr(
                        Q[j, r] <= Q[i, r] + demand[j] + C * (1 - x[i, j, r]),
                        name=f"LoadUB_{i}_{j}_{r}"
                    )

                # 容量上限（P节点负载不能超过C）
                for p in p_nodes:
                    model.addConstr(Q[p, r] <= C, name=f"CapP_{p}_{r}")

            # --- 约束8: 时间连续性（Big-M）---
            for r in R:
                model.addConstr(T[start_node, r] == 0)
                for i, j in arcs:
                    if (i, j, r) not in x:
                        continue
                    model.addConstr(
                        T[j, r] >= T[i, r] + service_time[i] + tau[i, j] - M_time * (1 - x[i, j, r]),
                        name=f"TimeCont_{i}_{j}_{r}"
                    )
                # Makespan
                model.addConstr(Z >= T[end_node, r], name=f"Makespan_{r}")


            # --- 约束10: 对称破缺 ---
            for r in range(1, num_robots):
                model.addConstr(
                    gp.quicksum(y[p, r] for p in p_nodes) <=
                    gp.quicksum(y[p, r - 1] for p in p_nodes),
                    name=f"SymBreak_{r}"
                )

            # ============================================================
            # 7. 目标函数
            # ============================================================
            total_travel = gp.quicksum(tau[i, j] * x[i, j, r] for (i, j) in arcs for r in R)
            model.setObjective(Z + 0.01 * total_travel, GRB.MINIMIZE)

            model.write("model.lp")
            # ============================================================
            # 9. 分阶段求解
            # ============================================================
            model._x_vars = x
            model._nodes_map = nodes_map
            model._p_nodes = set(p_nodes)
            model._d_nodes = set(d_nodes)


            print("\n  >>> [Phase 1] Quick feasibility (60s)...")
            model.Params.TimeLimit = 3600
            model.Params.MIPFocus = 1
            model.Params.Heuristics = 0.3

            model.optimize()

            # ============================================================
            # 10. 结果提取
            # ============================================================
            result_times = {}
            result_assign = {}

            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
                print(f"  >>> PDP-v2 Solved. Obj={model.objVal:.2f}")
                self.logger.log_solution(x, y, T, Q, nodes_map, R, p_nodes, d_nodes)
                for task, (p_id, d_id) in pair_map.items():
                    _, subtask, task_obj,_, _, _ = nodes_map[p_id]
                    for r in R:
                        if y[p_id, r].X > 0.5:
                            arr_time = T[p_id, r].X
                            pt = nodes_map[p_id][0]

                            print(f"      [路线验证] 任务 {task_obj.task_id} (Rank {task_obj.station_sequence_rank}): "
                                  f"取货时间 = {T[p_id, r].X:.1f}s, 卸货时间 = {T[d_id, r].X:.1f}s")
                            result_times[pt.idx] = arr_time
                            result_assign[subtask.id] = self.problem.robot_list[r].id
                            # 回填
                            task_obj.arrival_time_at_stack = arr_time
                            task_obj.robot_id = self.problem.robot_list[r].id
            else:
                print("  >>> PDP-v2 Infeasible or no solution found.")
                print("  >>> Falling back to heuristic.")
                return {}, {}

        return result_times, result_assign
    
    @staticmethod
    def _find_connected_components(edges):
        """返回节点列表的列表，例如 [[1,2,3], [4,5]]"""
        if not edges: return []
        adj = defaultdict(list)
        nodes = set()
        for i, j in edges:
            adj[i].append(j)
            nodes.add(i)
            nodes.add(j)

        visited = set()
        components = []
        for n in nodes:
            if n not in visited:
                comp = []
                q = [n]
                visited.add(n)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for nxt in adj[curr]:
                        # 无向化处理以找到连通块，或者仅根据出边
                        # 为防止 x->y 但 y->x 没被识别为同一组，建议视为无向图做连通性检查
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)
                components.append(comp)
        return components

    @staticmethod
    def _find_cycles_dfs(edges):
        """
        使用 DFS 检测有向图中的所有环
        返回: List[List[int]] - 每个环的节点列表
        """
        if not edges:
            return []

        # 构建邻接表（有向边）
        adj = defaultdict(list)
        nodes = set()
        for i, j in edges:
            adj[i].append(j)
            nodes.add(i)
            nodes.add(j)

        visited = set()
        rec_stack = set()  # 递归栈，用于检测环
        cycles = []

        def dfs(node, path):
            """DFS 搜索，path 记录当前路径"""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # 发现环！提取环路
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    cycles.append(cycle)

            rec_stack.remove(node)
            path.pop()

        # 从每个未访问的节点开始 DFS
        for start_node in nodes:
            if start_node not in visited:
                dfs(start_node, [])

        return cycles

    def _cb_lazy_subtour_optimized(self, model, where):
        """优化版子回路检测"""
        if where != GRB.Callback.MIPSOL:
            return

        x_vals = model.cbGetSolution(model._vars)

        # 只检查活跃的边（阈值提高到 0.9 以减少误判）
        edges_per_robot = defaultdict(list)
        for (i, j, r), val in x_vals.items():
            if val > 0.9:  # 提高阈值
                edges_per_robot[r].append((i, j))

        cuts_added = 0
        for r, edges in edges_per_robot.items():
            if len(edges) < 2:  # 少于 2 条边不可能成环
                continue

            # 使用并查集快速检测连通性
            components = self._find_components_union_find(edges)

            # 只切割最小的子回路（切割力最强）
            min_subtour = min(
                (comp for comp in components if self._is_illegal_subtour(comp, r)),
                key=len,
                default=None
            )

            if min_subtour:
                model.cbLazy(
                    gp.quicksum(model._vars[i, j, r]
                                for i in min_subtour
                                for j in min_subtour
                                if (i, j, r) in model._vars)
                    <= len(min_subtour) - 1
                )
                cuts_added += 1
                break  # 每次只加一个最强的割

        if cuts_added > 0:
            print(f"  [CUT] Added {cuts_added} lazy cut(s)")

    def _find_components_union_find(self, edges):
        """使用并查集快速查找连通分量（比 DFS 快）"""
        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # 路径压缩
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 初始化
        nodes = set()
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)
            parent.setdefault(i, i)
            parent.setdefault(j, j)

        # 合并
        for i, j in edges:
            union(i, j)

        # 分组
        components = defaultdict(list)
        for node in nodes:
            components[find(node)].append(node)

        return list(components.values())

    def _is_illegal_subtour(self, component, robot_id):
        """判断是否为非法子回路"""
        # 包含起点或终点的是合法路径
        has_start = any(self.nodes_map_ref[n][3] == 'robot_start' for n in component)
        has_depot = any(self.nodes_map_ref[n][3] == 'depot' for n in component)

        # 纯 Stack 环是非法的
        if not has_start and not has_depot:
            return True

        return False

    def _reconstruct_path(self, edges):
        """从边列表重建有序路径"""
        if not edges:
            return []

        # 构建邻接表
        adj = {i: j for i, j in edges}

        # 找起点（出度>0 但入度=0 的节点）
        out_nodes = set(i for i, _ in edges)
        in_nodes = set(j for _, j in edges)
        start = list(out_nodes - in_nodes)[0] if out_nodes - in_nodes else edges[0][0]

        # 重建路径
        path = [start]
        curr = start
        while curr in adj:
            curr = adj[curr]
            path.append(curr)
            if len(path) > 1000:  # 防止死循环
                break

        return path

    @staticmethod
    def _find_weak_components(edges):
        """辅助函数：找弱连通分量（将有向图视为无向）"""
        if not edges:
            return []

        # 双向邻接表
        adj = defaultdict(set)
        nodes = set()
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)  # 无向化
            nodes.add(i)
            nodes.add(j)

        visited = set()
        components = []

        for start in nodes:
            if start not in visited:
                comp = []
                queue = [start]
                visited.add(start)

                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                components.append(comp)

        return components


import os
from typing import Dict, List


class SP4Logger:
    def __init__(self, log_dir: str, filename: str = "sp4_debug.txt"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.file_path = os.path.join(log_dir, filename)
        # 初始化时清空文件，避免追加混乱
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== SP4 Solver Debug Log ===\n")

    def _get_node_desc(self, n_id: int, nodes_map: Dict) -> str:
        """内部辅助函数：将 MIP node_id 转为人类可读字符串"""
        if n_id not in nodes_map:
            return f"Unknown_Node_{n_id}"

        # nodes_map 结构: (point_obj, subtask, task_obj, type, layer)
        pt, subtask, task_obj,_, n_type, layer = nodes_map[n_id]

        if n_type == 'start':
            return f"Start @ Point:{pt.idx} ({pt.x},{pt.y})"
        elif n_type == 'end':
            return f"End @ Point:{pt.idx} ({pt.x},{pt.y})"
        elif n_type == 'pickup':
            stack_id = task_obj.target_stack_id if task_obj else "?"
            return f"Pickup @ Stack{stack_id}, Point:{pt.idx} ({pt.x},{pt.y}), Task:{task_obj.task_id}, Stack_total_tote:{task_obj.total_load_count}"
        elif n_type == 'delivery':
            station_id = layer if layer != -1 else (subtask.assigned_station_id if subtask else "?")
            return f"Delivery @ Station{station_id}, Point:{pt.idx} ({pt.x},{pt.y}), Task:{task_obj.task_id}, Stack_total_tote:{task_obj.total_load_count}"

        return f"Node_{n_id} ({n_type})"

    def log_node_definitions(self, nodes_map: Dict):
        """功能 1: 记录节点定义 (ID -> 物理含义)"""
        print(f"  >>> [Log] Writing node definitions to {self.file_path} ...")
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("PART 1: Node Definitions (MIP Graph Mapping)\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Node ID':<10} | {'Type':<12} | {'Description'}\n")
            f.write("-" * 60 + "\n")

            # 按 ID 排序输出
            for n_id in sorted(nodes_map.keys()):
                stack_pt, st, task, total_load_count, n_type, station_id=nodes_map[n_id]

                desc = self._get_node_desc(n_id, nodes_map)
                f.write(f"{n_id:<10} | {n_type:<12} | {desc}\n")
            f.write("\n")

    def log_mip_step(self, step_name: str, data: Dict, nodes_map: Dict = None):
        """通用的 MIP 步骤日志记录"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"MIP Step: {step_name}\n")
            f.write(f"{'=' * 60}\n")

            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (list, tuple)) and len(value) < 20:
                        f.write(f"{key}: {value}\n")
                    elif isinstance(value, dict) and len(value) < 50:
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {type(value)} (size={len(value) if hasattr(value, '__len__') else 'N/A'})\n")
            else:
                f.write(f"{data}\n")
            f.write("\n")

    def log_node_graph(self, nodes_map: Dict, arcs: List, tau: Dict):
        """记录节点图结构"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write("PART 4: Graph Structure (Nodes + Arcs)\n")
            f.write(f"{'=' * 80}\n\n")

            # 1. 节点信息
            f.write(f"[Nodes Summary] Total: {len(nodes_map)}\n")
            node_types = {}
            for nid, (pt, st, task, _, ntype, _) in nodes_map.items():
                node_types[ntype] = node_types.get(ntype, 0) + 1
            f.write(f"Node Types: {node_types}\n\n")

            # 2. 边信息
            f.write(f"[Arcs Summary] Total: {len(arcs)}\n")
            arc_types = {}
            for i, j in arcs:
                type_i = nodes_map[i][4]
                type_j = nodes_map[j][4]
                key = f"{type_i}->{type_j}"
                arc_types[key] = arc_types.get(key, 0) + 1

            for arc_type, count in sorted(arc_types.items()):
                f.write(f"  {arc_type}: {count}\n")

            # 3. 边的详细信息（可选，仅输出部分）
            f.write(f"\n[Sample Arcs] (First 20)\n")
            for idx, (i, j) in enumerate(arcs[:20]):
                type_i = nodes_map[i][4]
                type_j = nodes_map[j][4]
                time = tau.get((i, j), 0)
                f.write(f"  Arc {idx}: Node{i}({type_i}) -> Node{j}({type_j}), travel={time:.2f}s\n")

            if len(arcs) > 20:
                f.write(f"  ... (and {len(arcs) - 20} more arcs)\n")
            f.write("\n")

    def log_constraint_summary(self, model):
        """记录约束统计"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write("PART 5: Model Statistics\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Variables: {model.NumVars}\n")
            f.write(f"Binary Variables: {model.NumBinVars}\n")
            f.write(f"Continuous Variables: {model.NumVars - model.NumBinVars}\n")
            f.write(f"Constraints: {model.NumConstrs}\n")
            f.write(f"Non-zeros: {model.NumNZs}\n\n")

    def log_solution(self, x, y, T, Q, nodes_map: Dict, R: List, p_nodes: List, d_nodes: List):
        """记录最优解"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write("PART 6: Optimal Solution\n")
            f.write(f"{'=' * 80}\n\n")

            for r in R:
                f.write(f"\n--- Robot {r} Route ---\n")

                # 提取该机器人的路径
                edges = [(i, j) for (i, j, rr) in x.keys() if rr == r and x[i, j, rr].X > 0.5]

                if not edges:
                    f.write("  No tasks assigned\n")
                    continue

                # 重建路径
                adj = {i: j for i, j in edges}
                # 找起点
                all_i = set(i for i, _ in edges)
                all_j = set(j for _, j in edges)
                start = list(all_i - all_j)[0] if (all_i - all_j) else edges[0][0]

                path = [start]
                curr = start
                while curr in adj:
                    curr = adj[curr]
                    path.append(curr)
                    if len(path) > 100:  # 防止死循环
                        f.write("  ERROR: Path reconstruction failed (loop detected)\n")
                        break

                f.write(f"  Path ({len(path)} nodes): {path}\n\n")

                # 详细节点信息
                for seq, nid in enumerate(path):
                    ntype = nodes_map[nid][4]
                    time_val = T[nid, r].X if (nid, r) in T else 0
                    load_val = Q[nid, r].X if (nid, r) in Q else 0

                    desc = self._get_node_desc(nid, nodes_map)
                    f.write(f"  [{seq}] Node {nid} ({ntype}): T={time_val:.2f}s, Q={load_val:.1f}\n")
                    f.write(f"       {desc}\n")

                f.write("\n")

    def log_heuristic_solution(self, injected: Dict, nodes_map: Dict):
        """功能 2: 记录启发式解的变量详情"""
        print(f"  >>> [Log] Writing heuristic variables to {self.file_path} ...")
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("PART 2: Heuristic Warm Start Variables\n")
            f.write("=" * 60 + "\n")

            # 1. 写入 X 变量
            f.write("\n[Variables: x(i, j, r)]\n")
            f.write(f"{'Variable':<25} | {'Val':<3} | {'Description (From -> To)'}\n")
            f.write("-" * 80 + "\n")

            # 排序：按机器人 -> 起点ID
            sorted_x = sorted(injected['x'].items(), key=lambda item: (item[0][2], item[0][0]))

            for (i, j, r), val in sorted_x:
                desc_from = self._get_node_desc(i, nodes_map)
                desc_to = self._get_node_desc(j, nodes_map)
                f.write(f"x[{i}, {j}, {r}] = {val}     # Robot_{r}: {desc_from} --> {desc_to}\n")

            # 2. 写入 Y 变量
            f.write("\n[Variables: y(i, r)]\n")
            sorted_y = sorted(injected['y'].items(), key=lambda item: (item[0][1], item[0][0]))
            for (i, r), val in sorted_y:
                desc = self._get_node_desc(i, nodes_map)
                f.write(f"y[{i}, {r}] = {val}        # Robot_{r} visits {desc}\n")

            # 3. 写入 T 变量
            f.write("\n[Variables: T(i, r)]\n")
            sorted_T = sorted(injected['T'].items(), key=lambda item: (item[0][1], item[1]))
            for (i, r), val in sorted_T:
                desc = self._get_node_desc(i, nodes_map)
                f.write(f"T[{i}, {r}] = {val:.2f}s    # Robot_{r} at {desc}\n")

            # 4. 写入 L 变量
            f.write("\n[Variables: L(i, r)]\n")
            sorted_L = sorted(injected['L'].items(), key=lambda item: (item[0][1], item[1]))
            for (i, r), val in sorted_L:
                desc = self._get_node_desc(i, nodes_map)
                f.write(f"L[{i}, {r}] = {val}       # Robot_{r} load at {desc}\n")
                # 5. 写入 trip 变量
            f.write("\n[Variables: trip(i, r)]\n")
            sorted_trip = sorted(injected['trip'].items(), key=lambda item: (item[0][1], item[1]))
            for (i, r), val in sorted_trip:
                desc = self._get_node_desc(i, nodes_map)
                f.write(f"trip[{i}, {r}] = {val}     # Robot_{r} trip layer at {desc}\n")

    def log_validation(self, message: str):
        """功能 3: 记录验证信息"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(message + "\n")


def checksp3hit(
        sub_tasks: List[SubTask], problem, logger: SP4Logger = None):
    header = f"  >>> [Check] SP3 结果验证：检查料箱命中是否满足 SubTask 的 SKU 需求 (含冗余检查) ..."
    print(header)
    if logger:
        logger.log_validation("\n" + "=" * 60 + "\nPART 3: SP3 Hit Validation\n" + "=" * 60)
        logger.log_validation(header)

    for st in sub_tasks:
        if st.assigned_station_id == -1:
            print(f" >>>warning1！！！！")

        # 1. 统计 SubTask 的 SKU 需求
        required_skus = {}  # {sku_id: required_quantity}
        for sku in st.sku_list:
            required_skus[sku.id] = required_skus.get(sku.id, 0) + 1

        # 2. 统计 execution_tasks 中所有 hit_tote_ids 提供的 SKU
        provided_skus = {}  # {sku_id: provided_quantity}

        # ✅ 新增：记录每个料箱提供的 SKU 详情
        tote_sku_details = []  # [(tote_id, stack_id, sku_map)]

        # ✅ 关键修改：使用动态剩余需求追踪
        remaining_req = required_skus.copy()  # {sku_id: remaining_quantity}
        redundant_totes_info = []

        for task in st.execution_tasks:
            for tote_id in task.hit_tote_ids:
                tote = problem.id_to_tote.get(tote_id)
                if not tote:
                    print(f"  [ERR] [SubTask {st.id}] Tote {tote_id} not found in problem.id_to_tote")
                    continue

                # ✅ 关键修改：计算该料箱实际贡献的 SKU
                actual_contribution = {}  # 该料箱真正满足的 SKU
                noise_skus = {}  # 该料箱中多余的 SKU

                for sku_id, qty in tote.sku_quantity_map.items():
                    if remaining_req.get(sku_id, 0) > 0:
                        # 计算实际使用量（不超过剩余需求）
                        used = min(remaining_req[sku_id], qty)
                        actual_contribution[sku_id] = used
                        remaining_req[sku_id] -= used

                        # 如果该料箱中该SKU数量超出需求，超出部分算噪音
                        if qty > used:
                            noise_skus[sku_id] = qty - used
                    else:
                        # 该 SKU 已经满足，全部算噪音
                        noise_skus[sku_id] = qty

                # ✅ 记录该料箱的实际贡献（而非原始内容）
                tote_sku_details.append((
                    tote_id,
                    task.target_stack_id,
                    actual_contribution,  # ✅ 只记录实际贡献的部分
                    noise_skus  # ✅ 单独记录噪音
                ))

                # 累加总供给（用于最终检查）
                for sku_id, qty in tote.sku_quantity_map.items():
                    provided_skus[sku_id] = provided_skus.get(sku_id, 0) + qty

                # 判断该料箱是否有贡献
                if not actual_contribution:
                    redundant_totes_info.append(f"Tote {tote_id} (Stack {task.target_stack_id})")

        # 3. 检查覆盖性
        missing_skus = []
        excess_skus = []
        validation_passed = True

        for sku_id, required_qty in required_skus.items():
            provided_qty = provided_skus.get(sku_id, 0)

            if provided_qty < required_qty:
                missing_skus.append((sku_id, required_qty - provided_qty))
                validation_passed = False
            elif provided_qty > required_qty:
                excess_skus.append((sku_id, provided_qty - required_qty))

        # 4. ✅ 输出验证结果（修正版）
        log_lines = []

        msg = f"\n  [SubTask {st.id}] SKU Overview:"
        print(msg)
        log_lines.append(msg)

        msg = f"      Required SKUs: {required_skus}"
        print(msg)
        log_lines.append(msg)

        msg = f"      Provided SKUs: {provided_skus}"
        print(msg)
        log_lines.append(msg)

        # ✅ 核心修改：显示每个料箱的实际贡献（扣除已满足的SKU）
        msg = f"      Tote-Level SKU Breakdown ({len(tote_sku_details)} totes):"
        print(msg)
        log_lines.append(msg)

        for tote_id, stack_id, needed_skus, noise_skus in tote_sku_details:
            msg = f"         Tote {tote_id} @ Stack {stack_id}:"
            print(msg)
            log_lines.append(msg)

            if needed_skus:
                msg = f"           Needed: {needed_skus}"
                print(msg)
                log_lines.append(msg)

            if noise_skus:
                msg = f"           Noise: {noise_skus}"
                print(msg)
                log_lines.append(msg)

            # 如果两者都为空，说明是完全冗余的料箱
            if not needed_skus and not noise_skus:
                msg = f"           [WARN] Completely Redundant (all SKUs already satisfied)"
                print(msg)
                log_lines.append(msg)

        # 输出验证结果
        if missing_skus:
            msg = f"\n  [FAIL] [SubTask {st.id}] Validation FAILED:"
            print(msg)
            log_lines.append(msg)

            msg = f"      Missing SKUs:"
            print(msg)
            log_lines.append(msg)
            for sku_id, shortage in missing_skus:
                msg = f"         - SKU {sku_id}: Need {shortage} more"
                print(msg)
                log_lines.append(msg)
        else:
            msg = f"  [OK] [SubTask {st.id}] Validation PASSED ({len(required_skus)} SKU types, {sum(required_skus.values())} units)"
            print(msg)
            log_lines.append(msg)

        # 输出多余 SKU 信息
        if excess_skus:
            msg = f"      Excess SKUs (over-supply):"
            print(msg)
            log_lines.append(msg)
            for sku_id, excess in excess_skus:
                msg = f"         - SKU {sku_id}: +{excess} extra"
                print(msg)
                log_lines.append(msg)

        # 输出冗余料箱信息
        if redundant_totes_info:
            msg = f"      [WARN] Redundant Totes (not contributing to required SKUs): {len(redundant_totes_info)}"
            print(msg)
            log_lines.append(msg)
            for info in redundant_totes_info:
                msg = f"         - {info}"
                print(msg)
                log_lines.append(msg)

        if logger:
            for line in log_lines:
                logger.log_validation(line)

    final_msg = f"  >>> [OK] SP3 Validation Complete. All SubTasks have sufficient tote coverage.\n"
    print(final_msg)
    if logger:
        logger.log_validation(final_msg)


if __name__ == "__main__":
    from Gurobi.sp1 import SP1_BOM_Splitter
    from Gurobi.sp2 import SP2_Station_Assigner
    from Gurobi.sp3 import SP3_Bin_Hitter
    from problemDto.createInstance import CreateOFSProblem
    import random
    import numpy as np

    # ✅ 在任何导入和计算之前固定种子
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    print("\n" + "=" * 60)
    print("=== Integrated SP1-SP2-SP3-SP4 Pipeline Test ===")
    print("=" * 60)
    print("\n[Phase 0] Generating Problem Instance...")
    problem_dto = CreateOFSProblem.generate_problem_by_scale('SMALL', seed=SEED)
    print(f"  - Orders: {len(problem_dto.order_list)}")
    print(f"  - Robots: {len(problem_dto.robot_list)}")
    print(f"  - Stations: {len(problem_dto.station_list)}")
    print(f"  - Stacks: {len(problem_dto.stack_list)}")
    print(f"  - Totes: {len(problem_dto.tote_list)}")
    # 2. SP1: 拆分订单
    sp1 = SP1_BOM_Splitter(problem_dto)
    sub_tasks = sp1.solve(use_mip=False)
    # ✅ 回填到 ProblemDTO
    problem_dto.subtask_list = sub_tasks
    problem_dto.subtask_num = len(sub_tasks)
    print(f"  ✓ Generated {len(sub_tasks)} sub-tasks")
    print(f"  ✓ Bound to problem_dto.subtask_list")

    # 验证覆盖性
    from collections import defaultdict

    order_coverage = defaultdict(list)
    for task in sub_tasks:
        order_coverage[task.parent_order.order_id].extend([sku.id for sku in task.sku_list])

    for order in problem_dto.order_list:
        original = sorted(order.order_product_id_list)
        generated = sorted(order_coverage[order.order_id])
        assert original == generated, f"Order {order.order_id} coverage mismatch!"
    print(f"  ✓ Verification passed: All orders fully covered")

    # 3. SP2: 初始工作站分配
    sp2 = SP2_Station_Assigner(problem_dto)
    sp2.solve_initial_heuristic()
    # ✅ 结果已在 solve_initial_heuristic() 中直接回填到 SubTask 对象
    # 验证分配结果
    assigned_count = sum(1 for t in sub_tasks if t.assigned_station_id != -1)
    print(f"  ✓ Assigned {assigned_count}/{len(sub_tasks)} tasks to stations")

    # 统计每个工作站的负载
    station_loads = defaultdict(int)
    for task in sub_tasks:
        if task.assigned_station_id != -1:
            station_loads[task.assigned_station_id] += 1

    print(f"  ✓ Station load distribution:")
    for s_id, count in sorted(station_loads.items()):
        print(f"      Station {s_id}: {count} tasks")
    # 输出每个subtask被分配到的工作站
    for task in sub_tasks:
        print(f"    SubTask {task.id} assigned to Station {task.assigned_station_id}")
    # 4. SP3: 选箱决策
    sp3 = SP3_Bin_Hitter(problem_dto)
    physical_tasks, tote_selection, sorting_costs = sp3.SP3_Heuristic_Solver(problem_dto).solve(
        sub_tasks,
        beta_congestion=1.0
    )
    # ✅ 回填结果
    # (1) 物理任务列表 -> ProblemDTO
    # 注意：这里可以选择存储到 problem_dto 的新字段，或者通过 SubTask.execution_tasks 访问
    problem_dto.task_num = len(physical_tasks)
    problem_dto.task_list= physical_tasks
    # (2) 记录每个 SubTask 的选箱信息（已在 SP3 内部通过 task.add_execution_detail() 完成）
    # 验证：
    print(f"  ✓ Generated {len(physical_tasks)} physical tasks")
    print(f"  ✓ Total sorting cost: {sum(sorting_costs.values()):.2f}s")
    # 验证每个 SubTask 的执行细节
    for task in sub_tasks:
        if task.execution_tasks:
            print(f"    SubTask {task.id}: {len(task.execution_tasks)} tasks, "
                  f"{len(task.involved_stacks)} stacks, "
                  f"{len(task.assigned_tote_ids)} totes")

    print(f"\n=== SP3 Results ===")
    print(f"Generated {len(physical_tasks)} physical tasks")
    print(f"Total sorting cost: {sum(sorting_costs.values()):.2f}")
    sum_load = 0
    # 验证每个task的选箱结果
    for task in physical_tasks:
        sum_load += task.total_load_count
        print(f"Physical Task {task.task_id}: SubTask {task.sub_task_id}, "
              f"Stack {task.target_stack_id}, Tote {task.hit_tote_ids}, noise {task.noise_tote_ids}"
              f"Load {task.total_load_count}, Service Time {task.robot_service_time}s")
    print(f"[OK] Total load across all physical tasks: {sum_load}")
    # # 5. SP4: 机器人路径规划
    sp4 = SP4_Robot_Router(problem_dto)
    checksp3hit(sub_tasks, problem_dto, logger=sp4.logger)
    arrival_times, robot_assign = sp4.solve(sub_tasks, use_mip=False)
    # ✅ 回填结果
    # (1) 到达时间已在 _solve_mip() 中回填到 Task.arrival_time_at_stack
    # (2) 机器人分配已回填到 SubTask.assigned_robot_id

    print(f"  ✓ Computed arrival times for {len(arrival_times)} points")
    print(f"  ✓ Assigned {len(robot_assign)} sub-tasks to robots")

    print(f"\n=== SP4 Results ===")
    print(f"Arrival times computed for {len(arrival_times)} points")
    print(f"SubTask-Robot assignments: {len(robot_assign)}")
    # 统计机器人负载
    robot_loads = defaultdict(int)
    robot_tasks = defaultdict(list)
    for st_id, r_id in robot_assign.items():
        robot_loads[r_id] += 1
        robot_tasks[r_id].append(st_id)

    print(f"  ✓ Robot workload distribution:")
    for r_id, count in sorted(robot_loads.items()):
        print(f"      Robot {r_id}: {count} sub-tasks -> {robot_tasks[r_id]}")

    # 验证结果
    for st_id, r_id in robot_assign.items():
        st = next(t for t in sub_tasks if t.id == st_id)
        print(f"SubTask {st_id} -> Robot {r_id} | Tasks: {len(st.execution_tasks)}")

