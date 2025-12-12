import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import os
import sys

# 假设 sp4.py 存在于 DeepNCO/Gurobi/sp4.py
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
        # --- 初始化 Logger ---
        log_dir = os.path.join(ROOT_DIR, 'log')
        # 实例化 logger
        self.logger = SP4Logger(log_dir, filename="sp4_debug.txt")
    def _apply_warm_start_layered(self,
                                  model: gp.Model,
                                  x: gp.tupledict,
                                  y: gp.tupledict,
                                  T: gp.tupledict,
                                  L: gp.tupledict,
                                  trip: gp.tupledict,
                                  heu_robot_assign: Dict[int, int],
                                  heu_arrival_times: Dict[int, float],
                                  nodes_map: Dict,

                                  depot_layer_nodes: Dict,
                                  robot_start_nodes: Dict,
                                  stack_nodes_indices: List[int],
                                  tau: Dict,
                                  demand: Dict,
                                  service_time: Dict,
                                  max_trips: int):
        """
        [完全重构版] 启发式路径 -> MIP 分层图映射，并记录日志
        """
        print(f"  >>> [SP4] Applying Layered Warm Start (Fixed Version)...")

        # ===== 第一步：建立物理位置 -> MIP 节点的映射 =====
        point_to_stack_nodes = defaultdict(list)
        for node_id in stack_nodes_indices:
            pt_obj, subtask, task_obj, _, _ = nodes_map[node_id]
            point_to_stack_nodes[pt_obj.idx].append({
                'node_id': node_id,
                'subtask_id': subtask.id,
                'stack_id': task_obj.target_stack_id,
                'task_obj': task_obj
            })

        # ===== 第二步：从启发式结果中提取机器人路径 =====
        robot_physical_routes = defaultdict(list)

        for subtask in self.problem.subtask_list:
            if not subtask.execution_tasks:
                continue

            r_id = heu_robot_assign.get(subtask.id)
            if r_id is None:
                continue

            for task in subtask.execution_tasks:
                arrival_time = getattr(task, 'arrival_time_at_stack', None)
                trip_idx = getattr(task, 'robot_visit_sequence', 0)

                if arrival_time is None:
                    continue

                stack_obj = self.problem.point_to_stack[task.target_stack_id]
                point_idx = stack_obj.store_point.idx

                robot_physical_routes[r_id].append({
                    'time': arrival_time,
                    'trip': trip_idx,
                    'point_idx': point_idx,
                    'stack_id': task.target_stack_id,
                    'subtask': subtask,
                    'task_obj': task,
                    'demand': task.total_load_count,
                    'service_time': task.robot_service_time
                })

        for r_id in robot_physical_routes:
            robot_physical_routes[r_id].sort(key=lambda x: (x['trip'], x['time']))

        # ===== 第三步：映射到 MIP 图并注入 =====
        injected = {'x': {}, 'y': {}, 'T': {}, 'L': {}, 'trip': {}}

        for r_id, route in robot_physical_routes.items():
            if not route:
                continue

            # 起点：机器人起始节点
            current_node = robot_start_nodes[r_id]
            current_time = 0.0
            current_load = 0.0
            current_trip = 1

            # 注入起点
            if (current_node, r_id) in y:
                y[current_node, r_id].Start = 1
                T[current_node, r_id].Start = 0.0
                injected['y'][(current_node, r_id)] = 1
                injected['T'][(current_node, r_id)] = 0.0

            last_subtask = None
            last_station_id = None

            for idx, visit in enumerate(route):
                point_idx = visit['point_idx']
                stack_id = visit['stack_id']
                visit_trip = visit['trip']
                visit_demand = visit['demand']
                subtask = visit['subtask']

                # 检测是否需要回 Depot
                need_depot_return = False
                target_station_id = subtask.assigned_station_id

                if idx > 0:
                    prev_visit = route[idx - 1]
                    if current_load + visit_demand > self.robot_capacity + 0.001:
                        need_depot_return = True
                    if subtask.id != last_subtask.id:
                        need_depot_return = True
                    if visit_trip != prev_visit['trip']:
                        need_depot_return = True

                # 执行 Depot 返回逻辑
                if need_depot_return:
                    prev_station = last_station_id
                    depot_node = depot_layer_nodes[prev_station][current_trip]

                    if (current_node, depot_node, r_id) in x:
                        x[current_node, depot_node, r_id].Start = 1
                        injected['x'][(current_node, depot_node, r_id)] = 1

                    travel_time = tau.get((current_node, depot_node), 0)
                    prev_service = service_time.get(current_node, 0)
                    current_time += prev_service + travel_time

                    if (depot_node, r_id) in y:
                        y[depot_node, r_id].Start = 1
                        T[depot_node, r_id].Start = current_time
                        injected['y'][(depot_node, r_id)] = 1
                        injected['T'][(depot_node, r_id)] = current_time

                    current_load = 0.0
                    current_node = depot_node
                    current_trip += 1

                    if current_trip > max_trips:
                        break

                # 访问 Stack 节点
                candidates = point_to_stack_nodes.get(point_idx, [])
                target_node = None
                for cand in candidates:
                    if cand['subtask_id'] == subtask.id and cand['stack_id'] == stack_id:
                        target_node = cand['node_id']
                        break

                if target_node is None:
                    continue

                # 注入边
                if (current_node, target_node, r_id) in x:
                    x[current_node, target_node, r_id].Start = 1
                    injected['x'][(current_node, target_node, r_id)] = 1

                travel_time = tau.get((current_node, target_node), 0)
                prev_service = service_time.get(current_node, 0)
                current_time += prev_service + travel_time
                current_load += visit_demand

                if (target_node, r_id) in y:
                    y[target_node, r_id].Start = 1
                    T[target_node, r_id].Start = current_time
                    L[target_node, r_id].Start = current_load

                    if (target_node, r_id) in trip:
                        trip[target_node, r_id].Start = current_trip
                        injected['trip'][(target_node, r_id)] = current_trip

                    injected['y'][(target_node, r_id)] = 1
                    injected['T'][(target_node, r_id)] = current_time
                    injected['L'][(target_node, r_id)] = current_load

                current_node = target_node
                last_subtask = subtask
                last_station_id = target_station_id

            # 路径结束
            if last_station_id is not None:
                final_depot = depot_layer_nodes[last_station_id][current_trip]
                if (current_node, final_depot, r_id) in x:
                    x[current_node, final_depot, r_id].Start = 1
                    injected['x'][(current_node, final_depot, r_id)] = 1

                travel_time = tau.get((current_node, final_depot), 0)
                prev_service = service_time.get(current_node, 0)
                current_time += prev_service + travel_time

                if (final_depot, r_id) in y:
                    y[final_depot, r_id].Start = 1
                    T[final_depot, r_id].Start = current_time
                    injected['y'][(final_depot, r_id)] = 1
                    injected['T'][(final_depot, r_id)] = current_time

        self.logger.log_heuristic_solution(injected, nodes_map)

        # 验证注入解的可行性 (保留原有逻辑)
        self._verify_warm_start_solution(injected, nodes_map, depot_layer_nodes, tau, demand, service_time)
    def _verify_warm_start_solution(self,
                                    vals: Dict,
                                    nodes_map: Dict,
                                    depot_layer_nodes: Dict,
                                    tau: Dict,
                                    demand: Dict,
                                    service_time: Dict):
        """
        验证注入解的逻辑正确性
        """
        print(f"  >>> [SP4] Verifying Warm Start Solution...")
        
        x_s = vals.get('x', {})
        y_s = vals.get('y', {})
        trip_s = vals.get('trip', {})
        T_s = vals.get('T', {})
        L_s = vals.get('L', {})
        
        violations = []
        
        # 1. 流守恒检查
        node_flow = defaultdict(lambda: {'in': 0, 'out': 0})
        for (i, j, r), val in x_s.items():
            if val > 0.5:
                node_flow[(i, r)]['out'] += 1
                node_flow[(j, r)]['in'] += 1
                
                # 检查端点是否激活
                if y_s.get((i, r), 0) < 0.5:
                    violations.append(f"Flow error: x[{i},{j},{r}]=1 but y[{i},{r}]=0")
                if y_s.get((j, r), 0) < 0.5:
                    violations.append(f"Flow error: x[{i},{j},{r}]=1 but y[{j},{r}]=0")
        
        # 检查度数平衡（除起点外）
        for (node, r), flow in node_flow.items():
            node_type = nodes_map[node][3]
            if node_type != 'robot_start' and flow['in'] != flow['out']:
                violations.append(f"Flow imbalance at node {node} (r={r}): in={flow['in']}, out={flow['out']}")
        
        # 2. Trip 逻辑检查
        for (i, j, r), val in x_s.items():
            if val < 0.5:
                continue
            
            type_i = nodes_map[i][3]
            type_j = nodes_map[j][3]
            
            trip_i = trip_s.get((i, r))
            trip_j = trip_s.get((j, r))
            
            # Stack -> Stack: trip 必须相同
            if type_i == 'stack' and type_j == 'stack':
                if trip_i is not None and trip_j is not None and trip_i != trip_j:
                    violations.append(f"Stack->Stack trip jump: {i}(trip={trip_i}) -> {j}(trip={trip_j})")
            
            # Stack -> Depot: Stack.trip 必须等于 Depot.layer
            if type_i == 'stack' and type_j == 'depot':
                depot_layer = nodes_map[j][4]
                if trip_i is not None and trip_i != depot_layer:
                    violations.append(f"Stack->Depot mismatch: Stack {i}(trip={trip_i}) -> Depot {j}(layer={depot_layer})")
            
            # Depot -> Stack: Stack.trip 必须等于 Depot.layer + 1
            if type_i == 'depot' and type_j == 'stack':
                depot_layer = nodes_map[i][4]
                if trip_j is not None and trip_j != depot_layer + 1:
                    violations.append(f"Depot->Stack mismatch: Depot {i}(layer={depot_layer}) -> Stack {j}(trip={trip_j})")
        
        # 3. 容量检查
        for (node, r), load in L_s.items():
            if load > self.robot_capacity + 0.01:
                violations.append(f"Capacity violation at node {node} (r={r}): load={load:.2f}")
        
        # 4. 时间单调性检查（沿路径）
        for (i, j, r), val in x_s.items():
            if val < 0.5:
                continue
            
            t_i = T_s.get((i, r))
            t_j = T_s.get((j, r))
            
            if t_i is not None and t_j is not None:
                expected_t_j = t_i + service_time.get(i, 0) + tau.get((i, j), 0)
                if t_j < expected_t_j - 0.01:
                    violations.append(f"Time violation: {i}->{j}, T[{j}]={t_j:.2f} < expected {expected_t_j:.2f}")
        
        if violations:
            print(f"  ❌ Verification Failed ({len(violations)} errors):")
            for v in violations[:10]:  # 只显示前 10 个
                print(f"     - {v}")
        else:
            print(f"  ✅ Warm Start Solution Verified.")
    def _extract_sequence(self, x, y, T, trip, nodes_map, N, R, depot_layer_nodes, robot_start_nodes,
                          stack_nodes_indices):
        """
        [修复版] 提取机器人路径（使用二维时间变量）
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

    def _solve_heuristic(self, sub_tasks: List[SubTask]) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        修正后的启发式：
        路径逻辑：上一单Station -> 本单Stack -> ... -> 本单Station
        包含完整的结果解析输出。
        """
        print(f"  >>> [SP4] Using Heuristic Solver (Direct Routing A->Stack->B)...")

        valid_tasks = [t for t in sub_tasks if t.execution_tasks]
        if not valid_tasks:
            return {}, {}

        robot_arrival_times = {}
        subtask_robot_assignment = {}

        # 初始化状态
        robot_times = {r.id: 0.0 for r in self.problem.robot_list}
        # 初始位置都在 StartPoint
        robot_positions = {r.id: r.start_point for r in self.problem.robot_list}
        robot_routes = {r.id: [] for r in self.problem.robot_list}
        robot_trip_counter = {r.id: 0 for r in self.problem.robot_list}

        # 贪婪分配
        for st in valid_tasks:
            total_demand = sum(task.total_load_count for task in st.execution_tasks)
            target_station_pt = self.problem.station_list[st.assigned_station_id].point

            # --- 1. 选车阶段 (Cost Estimation) ---
            best_robot = None
            best_cost = float('inf')

            for robot in self.problem.robot_list:
                r_id = robot.id
                current_pos = robot_positions[r_id]
                first_stack = self.problem.point_to_stack[st.execution_tasks[0].target_stack_id]

                # 关键修正：无论是不是第一趟，都直接计算 CurrentPos -> FirstStack
                # 如果是 Trip 0，CurrentPos 是起点
                # 如果是 Trip > 0，CurrentPos 是上一单的 Station
                dist_to_first = abs(current_pos.x - first_stack.store_point.x) + \
                                abs(current_pos.y - first_stack.store_point.y)
                start_overhead = dist_to_first / self.robot_speed

                # 估算总时间
                trips_needed = (total_demand + self.robot_capacity - 1) // self.robot_capacity
                station_to_stack_dist = abs(target_station_pt.x - first_stack.store_point.x) + \
                                        abs(target_station_pt.y - first_stack.store_point.y)

                # 第一趟：Current -> Stack -> Station
                # 后续趟：Station -> Stack -> Station
                # 近似估算后续趟次
                subsequent_trips_cost = 0
                if trips_needed > 1:
                    avg_cycle = (2 * station_to_stack_dist / self.robot_speed)
                    subsequent_trips_cost = (trips_needed - 1) * avg_cycle

                service_cost = sum(t.robot_service_time for t in st.execution_tasks)

                estimated_completion_time = robot_times[r_id] + start_overhead + subsequent_trips_cost + service_cost

                if estimated_completion_time < best_cost:
                    best_cost = estimated_completion_time
                    best_robot = r_id

            # --- 2. 执行阶段 ---
            r_id = best_robot
            st.assigned_robot_id = r_id
            subtask_robot_assignment[st.id] = r_id

            current_time = robot_times[r_id]
            current_pos = robot_positions[r_id]
            trip_sequence = robot_trip_counter[r_id]

            remaining_tasks = list(st.execution_tasks)

            # 处理多趟搬运
            while remaining_tasks:
                current_trip_tasks = []
                trip_load = 0

                # 如果不是该子任务的第一趟（即同一子任务内的第二、三趟），
                # 起点是该子任务的目标 Station，而不是上一单的 Station
                if len(current_trip_tasks) == 0 and len(remaining_tasks) < len(st.execution_tasks):
                    # 检查是否是刚送完上一趟回来（即 st 内的多趟搬运）
                    # 判断逻辑：如果 trip_sequence > robot_trip_counter[r_id] 说明已经在循环里增加过趟次了
                    if trip_sequence > robot_trip_counter[r_id]:
                        current_pos = target_station_pt

                # 贪婪装载
                while remaining_tasks:
                    best_task = None
                    best_dist = float('inf')

                    for task in remaining_tasks:
                        if trip_load + task.total_load_count > self.robot_capacity:
                            continue
                        stack = self.problem.point_to_stack[task.target_stack_id]
                        dist = abs(current_pos.x - stack.store_point.x) + \
                               abs(current_pos.y - stack.store_point.y)

                        if dist < best_dist:
                            best_dist = dist
                            best_task = task

                    if best_task is None: break

                    # 移动到 Stack
                    stack = self.problem.point_to_stack[best_task.target_stack_id]
                    travel_time = best_dist / self.robot_speed
                    current_time += travel_time

                    # 记录时间
                    best_task.robot_id = r_id
                    best_task.arrival_time_at_stack = current_time
                    best_task.robot_visit_sequence = trip_sequence
                    robot_arrival_times[stack.store_point.idx] = current_time

                    current_time += best_task.robot_service_time
                    trip_load += best_task.total_load_count
                    current_pos = stack.store_point
                    current_trip_tasks.append(best_task)
                    remaining_tasks.remove(best_task)

                # 本趟结束，去往当前单的 Target Station
                return_dist = abs(current_pos.x - target_station_pt.x) + \
                              abs(current_pos.y - target_station_pt.y)
                current_time += return_dist / self.robot_speed
                current_pos = target_station_pt  # 更新位置为 Station B

                robot_routes[r_id].append({
                    'trip': trip_sequence + 1,
                    'start_time': current_trip_tasks[0].arrival_time_at_stack if current_trip_tasks else current_time,
                    'end_time': current_time,
                    'tasks': current_trip_tasks,
                    'depot_used': target_station_pt,  # ✅ 新增
                    'depot_layer': trip_sequence,
                    'load': trip_load  # <--- [FIXED] 添加 load 字段，供后续打印使用
                })
                trip_sequence += 1

            # 任务结束状态更新
            robot_times[r_id] = current_time
            robot_positions[r_id] = current_pos  # 停留在 Station B
            robot_trip_counter[r_id] = trip_sequence

        # --- 3. 结果解析与打印 ---
        print(f"\n  >>> [SP4] Heuristic Solved.")
        print(f"  - Total arrival times: {len(robot_arrival_times)}")
        print(f"  - SubTask assignments: {len(subtask_robot_assignment)}")

        for r_id in sorted(robot_routes.keys()):
            routes = robot_routes[r_id]
            if not routes:
                continue

            print(f"\n  === Robot {r_id} Routes (Heuristic) ===")
            for route in routes:
                print(f"  Trip {route['trip']}: {len(route['tasks'])} tasks, "
                      f"load={route['load']}/{self.robot_capacity}, "
                      f"time [{route['start_time']:.1f}s, {route['end_time']:.1f}s]，depot use {route['depot_used']},depot layer {route['depot_layer']}  ")

                for seq, task in enumerate(route['tasks']):
                    print(f"    [{seq}] Stack {task.target_stack_id} @ {task.arrival_time_at_stack:.1f}s "
                          f"(SubTask {task.sub_task_id}, Load={task.total_load_count})")

        total_trips = sum(len(routes) for routes in robot_routes.values())
        max_time = max(robot_times.values()) if robot_times else 0

        print(f"\n  === Heuristic Summary ===")
        print(f"  - Total trips: {total_trips}")
        print(f"  - Makespan: {max_time:.2f}s")
        print(
            f"  - Active robots: {sum(1 for routes in robot_routes.values() if routes)}/{len(self.problem.robot_list)}")

        return robot_arrival_times, subtask_robot_assignment

    def solve(self,
              sub_tasks: List[SubTask],
              use_mip: bool = True) -> Tuple[Dict[int, float], Dict[int, int]]:
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
            return self._solve_mip(sub_tasks)
        else:
            return self._solve_heuristic(sub_tasks)

    def _solve_mip(self, sub_tasks: List[SubTask]) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        修复版 MIP：引入分层 Depot 节点（Layered Depots）以支持多趟次访问
        """
        # 1. 数据预处理
        valid_tasks = [t for t in sub_tasks if t.execution_tasks]
        if not valid_tasks:
            return {}, {}

        # --- 构建节点 ---
        nodes_map = {}
        node_id = 0
        max_trips = 6  # 限制最大趟次以减少变量规模，可根据需要调整

        # (A) 机器人起点
        robot_start_nodes = {}
        for robot in self.problem.robot_list:
            robot_start_nodes[robot.id] = node_id
            nodes_map[node_id] = (robot.start_point, None, None, 'robot_start', 0)  # 0表示trip层级
            node_id += 1

        # (B) Stack 节点 (每个任务一个节点)
        stack_nodes_indices = []
        for st in valid_tasks:
            for task in st.execution_tasks:
                stack = self.problem.point_to_stack[task.target_stack_id]
                stack_nodes_indices.append(node_id)
                nodes_map[node_id] = (stack.store_point, st, task, 'stack', -1)  # -1表示不绑定特定层级
                node_id += 1

        # (C) [核心修复] 分层 Depot 节点
        # depot_nodes[station_id][trip_k] = node_id
        depot_layer_nodes = defaultdict(dict)

        for k in range(1, max_trips + 1):  # Trip 1 到 Trip max
            for station in self.problem.station_list:
                depot_layer_nodes[station.id][k] = node_id
                # 记录这是一个属于第 k 趟结束的 Depot 节点
                nodes_map[node_id] = (station.point, None, None, 'depot', k)
                node_id += 1

        self.logger.log_node_definitions(nodes_map)

        N = range(node_id)
        R = range(len(self.problem.robot_list))

        # 辅助映射
        subtask_nodes = defaultdict(list)
        for i in stack_nodes_indices:
            _, subtask, _, _, _ = nodes_map[i]
            subtask_nodes[subtask.id].append(i)

        # 2. 计算距离矩阵 (Tau)
        tau = {}  # 使用字典稀疏存储，减少内存
        for i in N:
            pt_i = nodes_map[i][0]
            for j in N:
                if i == j: continue
                # 剪枝：不同 Station 的 Depot 之间不需要连接
                type_i = nodes_map[i][3]
                type_j = nodes_map[j][3]
                if type_i == 'depot' and type_j == 'depot': continue

                pt_j = nodes_map[j][0]
                dist = abs(pt_i.x - pt_j.x) + abs(pt_i.y - pt_j.y)
                tau[i, j] = dist / self.robot_speed

        # 参数提取
        service_time = {}
        demand = {}
        for i in N:
            _, _, task_obj, _, _ = nodes_map[i]
            if task_obj:
                service_time[i] = task_obj.robot_service_time
                demand[i] = task_obj.total_load_count
            else:
                service_time[i] = 0.0
                demand[i] = 0

        # 3. 建模
        m = gp.Model("SP4_Layered_VRP")
        m.Params.OutputFlag = 1
        m.Params.MIPGap = 0.01

        # 变量
        # x[i,j,r]: 弧流量
        x = m.addVars([(i, j, r) for i in N for j in N if (i, j) in tau for r in R],
                      vtype=GRB.BINARY, name="x")
        # y[i,r]: 节点访问
        y = m.addVars(N, R, vtype=GRB.BINARY, name="y")
        # T[i,r]: 到达时间 (因为 Depot 已经分层，每个节点只会被访问一次，不需要三维 T)
        T = m.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="T")
        # L[i,r]: 负载
        L = m.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, ub=self.robot_capacity, name="L")
        # trip[i,r]: 记录 Stack 属于哪一趟 (Depot 节点不需要此变量，因为自带层级)
        trip = m.addVars(stack_nodes_indices, R, vtype=GRB.INTEGER, lb=1, ub=max_trips, name="trip")

        M = 2000

        # --- 约束 ---

        # 1. 覆盖约束 (Stack 必须被访问一次)
        for i in stack_nodes_indices:
            m.addConstr(gp.quicksum(y[i, r] for r in R) == 1, name=f"Cover_{i}")

        # 2. 流守恒
        for r in R:
            # 2.1 起点约束
            start_node = robot_start_nodes[self.problem.robot_list[r].id]
            m.addConstr(y[start_node, r] == 1)  # 起点必须激活
            m.addConstr(T[start_node, r] == 0)  # ✅ 显式设置起始时间
            m.addConstr(L[start_node, r] == 0)  # ✅ 显式设置起始负载
            m.addConstr(gp.quicksum(x[start_node, j, r] for j in N if (start_node, j) in tau) == 1)
            m.addConstr(gp.quicksum(x[j, start_node, r] for j in N if (j, start_node) in tau) == 0)

            # 2.2 普通节点 (Stack) 流守恒
            for i in stack_nodes_indices:
                m.addConstr(
                    gp.quicksum(x[j, i, r] for j in N if (j, i) in tau) == y[i, r],
                    name=f"FlowIn_{i}_{r}"
                )
                m.addConstr(
                    gp.quicksum(x[i, j, r] for j in N if (i, j) in tau) == y[i, r],
                    name=f"FlowOut_{i}_{r}"
                )

            # 2.3 分层 Depot 流守恒 (允许不访问，访问则进出平衡)
            # 关键路径逻辑：Stack (Trip k) -> Depot (Layer k) -> Stack (Trip k+1)
            for s_id, layer_dict in depot_layer_nodes.items():
                for k in range(1, max_trips + 1):
                    d_node = layer_dict[k]

                    # 入度：只能来自 Stack 或 起点 (Trip 1)
                    in_arcs = gp.quicksum(x[i, d_node, r] for i in N if (i, d_node) in tau)
                    # 出度：只能去往 Stack
                    out_arcs = gp.quicksum(x[d_node, j, r] for j in N if (d_node, j) in tau)

                    m.addConstr(in_arcs == y[d_node, r])
                    m.addConstr(out_arcs == y[d_node, r])

                    # 2.4 强制 Depot 连接逻辑 (防止乱序)
                    # 如果是从 Depot(k) 出去到 Stack j，则 Stack j 必须属于 Trip k+1
                    # 如果是从 Stack i 进来 Depot(k)，则 Stack i 必须属于 Trip k
                    for i in stack_nodes_indices:
                        if (i, d_node) in tau:
                            # Stack i -> Depot k implies trip[i] == k
                            m.addGenConstrIndicator(x[i, d_node, r], True, trip[i, r] == k)
                        if (d_node, i) in tau:
                            # Depot k -> Stack i implies trip[i] == k + 1
                            m.addGenConstrIndicator(x[d_node, i, r], True, trip[i, r] == k + 1)

        # 3. Stack 之间的直接连接 (同趟次)
        for r in R:
            for i in stack_nodes_indices:
                for j in stack_nodes_indices:
                    if i != j and (i, j) in tau:
                        # Stack -> Stack 意味着 trip 序号不变
                        m.addGenConstrIndicator(x[i, j, r], True, trip[i, r] == trip[j, r])

        # 4. Depot 必须回访约束 (SubTask 指定的 Station)
        for st_id, nodes in subtask_nodes.items():
            st = next(t for t in valid_tasks if t.id == st_id)
            target_station_id = st.assigned_station_id

            # 这一组 Stack 的任何流出到 Depot 的边，必须连向 target_station 对应的 Depot 节点
            # 或者 Stack -> Stack
            for i in nodes:
                for r in R:
                    # 禁止流向错误的 Station Depot
                    for s_id, layer_dict in depot_layer_nodes.items():
                        if s_id != target_station_id:
                            for k in range(1, max_trips + 1):
                                wrong_depot = layer_dict[k]
                                if (i, wrong_depot) in tau:
                                    m.addConstr(x[i, wrong_depot, r] == 0)

        # 5. 时间和容量约束 (标准 VRP)
        for r in R:
            for i in N:
                for j in N:
                    if (i, j) in tau:
                        # 时间推演
                        m.addConstr(
                            T[j, r] >= T[i, r] + service_time[i] + tau[i, j] - M * (1 - x[i, j, r]),
                            name=f"Time_{i}_{j}"
                        )

                        # 容量推演 (仅针对 Stack -> Stack)
                        # 如果 j 是 Stack，增加负载
                        if j in stack_nodes_indices:
                            m.addConstr(
                                L[j, r] >= L[i, r] + demand[j] - M * (1 - x[i, j, r]),
                                name=f"LoadInc_{i}_{j}"
                            )
                        # 如果 j 是 Depot，清空负载 (Reset)
                        elif nodes_map[j][3] == 'depot':
                            m.addConstr(
                                L[j, r] <= M * (1 - x[i, j, r]),  # L[depot] 必须为 0
                                name=f"LoadReset_{i}_{j}"
                            )

        # 6. 同 SubTask 同机器人约束
        for st_id, nodes in subtask_nodes.items():
            if len(nodes) > 1:
                base = nodes[0]
                for other in nodes[1:]:
                    for r in R:
                        m.addConstr(y[base, r] == y[other, r])
        robot_subtask_groups = defaultdict(list)
        for st_id, nodes in subtask_nodes.items():
            st = next(t for t in valid_tasks if t.id == st_id)
            if st.station_sequence_rank >= 0:  # 只处理有排序信息的任务
                # 获取该 SubTask 的代表节点（取第一个）
                repr_node = nodes[0]
                robot_subtask_groups[st.assigned_station_id].append((st, repr_node))
        # 为每个机器人添加约束
        for r in R:
            # 收集该机器人可能执行的 SubTask（按 station_sequence_rank 排序）
            candidate_subtasks = []
            for station_id, st_nodes_list in robot_subtask_groups.items():
                for st, repr_node in st_nodes_list:
                    # 如果该节点可能被机器人 r 访问
                    candidate_subtasks.append((st, repr_node, st.station_sequence_rank))
            
            if len(candidate_subtasks) < 2:
                continue  # 少于 2 个任务不需要排序约束
            
            # 按 station_sequence_rank 排序
            candidate_subtasks.sort(key=lambda x: x[2])
            
            # 添加时间序约束：如果两个 SubTask 都被机器人 r 执行，
            # 则 rank 小的必须在时间上早于 rank 大的
            for idx in range(len(candidate_subtasks) - 1):
                st_early, node_early, rank_early = candidate_subtasks[idx]
                st_late, node_late, rank_late = candidate_subtasks[idx + 1]
                if st_early.assigned_station_id != st_late.assigned_station_id:
                    early_nodes = subtask_nodes[st_early.id]
                    late_nodes = subtask_nodes[st_late.id]
                    
                    # 对于每一对 early-late 节点
                    for i in early_nodes:
                        for j in late_nodes:
                            # 如果两者都被 r 访问，则 T[i] + service[i] <= T[j]
                            both_flag = m.addVar(vtype=GRB.BINARY)
                            m.addConstr(both_flag <= y[i, r])
                            m.addConstr(both_flag <= y[j, r])
                            m.addConstr(both_flag >= y[i, r] + y[j, r] - 1)
                            
                            # Indicator 约束
                            m.addGenConstrIndicator(
                                both_flag, True, 
                                T[i, r] + service_time[i] <= T[j, r],
                                name=f"SeqRank_{i}_{j}_{r}"
                            )
               
        # 7. 对称性破缺约束 (防止机器人互换产生等价解)
        m.addConstrs(
            gp.quicksum(y[i, r] for i in stack_nodes_indices) >=  # ✅ 修复
            gp.quicksum(y[i, r + 1] for i in stack_nodes_indices)  # ✅ 修复
            for r in range(len(R) - 1)
        )
        # 1. 定义 Makespan 变量 Z
        Z = m.addVar(vtype=GRB.CONTINUOUS, name="Makespan")

        # 2. 收集所有的 Depot 节点索引
        # depot_layer_nodes 结构是: {station_id: {trip_layer_k: node_id}}
        all_depot_nodes = []
        for station_dict in depot_layer_nodes.values():
            for node_id in station_dict.values():
                all_depot_nodes.append(node_id)

        # 3. 添加 Makespan 约束
        # 逻辑：对于每一个机器人 r，如果它访问了某个 Depot 节点 d，那么 Z 必须大于该节点的到达时间
        for r in R:
            for d in all_depot_nodes:
                # 使用 Indicator Constraint: if y[d, r] == 1, then Z >= T[d, r]
                # 注意：如果有卸货时间(t_drop), 应该是 Z >= T[d, r] + t_drop
                m.addGenConstrIndicator(y[d, r], True, Z >= T[d, r])

        # 4. 设置目标函数：最小化 Makespan
        # 加上一点点总距离惩罚(epsilon)，用于在时间相同时选择路程更短的方案
        epsilon = 0.01
        total_dist = gp.quicksum(tau[i, j] * x[i, j, r] for i, j, r in x)

        m.setObjective(Z + epsilon * total_dist, GRB.MINIMIZE)

        print("  >>> [SP4] Generating heuristic warm start...")
        # 1. 运行启发式获取物理路径
        heu_arrival_times, heu_robot_assign = self._solve_heuristic(sub_tasks)

        # 2. 映射到分层图并注入
        self._apply_warm_start_layered(
            m, x, y, T, L, trip,
            heu_robot_assign,
            heu_arrival_times,
            nodes_map,
            depot_layer_nodes,  # 需确保在 solve_mip 作用域内可用
            robot_start_nodes,
            stack_nodes_indices,  # 需确保在 solve_mip 作用域内可用
            tau,
            demand,
            service_time,
            max_trips
        )
        m.optimize()

        # --- 结果解析 ---
        robot_arrival_times = {}
        subtask_robot_assign = {}

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"  >>> Solved. Obj: {m.objVal:.2f}")

            # 🔧 修复：传入正确参数
            self._extract_sequence(x, y, T, trip, nodes_map, N, R,
                                   depot_layer_nodes, robot_start_nodes, stack_nodes_indices)

            # 提取结果
            for i in stack_nodes_indices:
                pt, subtask, task, _, _ = nodes_map[i]
                for r in R:
                    if y[i, r].X > 0.5:
                        arr_time = T[i, r].X
                        robot_arrival_times[pt.idx] = arr_time
                        subtask_robot_assign[subtask.id] = self.problem.robot_list[r].id

                        task.robot_id = r
                        task.arrival_time_at_stack = arr_time

        else:
            print("  >>> MIP Infeasible or Failed.")

        return robot_arrival_times, subtask_robot_assign


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
        pt, subtask, task_obj, n_type, layer = nodes_map[n_id]

        if n_type == 'robot_start':
            return f"StackPoint:{pt.idx}（x,y):({pt.x},{pt.y}) (Robot_Start)"

        elif n_type == 'stack':
            stack_id = task_obj.target_stack_id if task_obj else "Unknown"
            st_id = subtask.id if subtask else "?"
            return f"Stack_{stack_id}，StackPoint:{pt.idx}（x,y):({pt.x},{pt.y}),task_id:{task_obj.task_id} ，task_service_time：{task_obj.robot_service_time},task_mode:{task_obj.operation_mode},(SubTask_{st_id})"

        elif n_type == 'depot':
            # Depot 节点包含层级信息 (Trip)
            return f"Station_Point_{pt.idx} (Trip_Layer_{layer})"

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
                point, _, _, n_type, _ = nodes_map[n_id]
                desc = self._get_node_desc(n_id, nodes_map)
                f.write(f"{n_id:<10} | {n_type:<12} | {desc}\n")
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

    def log_validation(self, message: str):
        """功能 3: 记录验证信息"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
def checksp3hit(
        sub_tasks: List[SubTask], problem, logger: SP4Logger = None):
    header = f"  >>>🔍 SP3 结果验证：检查料箱命中是否满足 SubTask 的 SKU 需求 (含冗余检查) ..."
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

        # --- 新增：冗余检查逻辑 ---
        remaining_req = required_skus.copy()
        redundant_totes_info = []
        # -----------------------

        for task in st.execution_tasks:
            for tote_id in task.hit_tote_ids:
                tote = problem.id_to_tote.get(tote_id)
                if not tote:
                    print(f"  ❌ [SubTask {st.id}] Tote {tote_id} not found in problem.id_to_tote")
                    continue

                # 累加该料箱提供的 SKU 数量 (用于总覆盖检查)
                for sku_id, qty in tote.sku_quantity_map.items():
                    provided_skus[sku_id] = provided_skus.get(sku_id, 0) + qty

                # --- 冗余判断 ---
                is_useful = False
                for sku_id, qty in tote.sku_quantity_map.items():
                    if remaining_req.get(sku_id, 0) > 0:
                        is_useful = True
                        # 扣减需求（贪婪扣减）
                        take = min(remaining_req[sku_id], qty)
                        remaining_req[sku_id] -= take

                if not is_useful:
                    redundant_totes_info.append(f"Tote {tote_id} (Stack {task.target_stack_id})")
                # ----------------

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

        # 4. 检查是否有不需要的 SKU
        unexpected_skus = []
        for sku_id in provided_skus:
            if sku_id not in required_skus:
                unexpected_skus.append((sku_id, provided_skus[sku_id]))

        # 5. 输出验证结果
        log_lines = []
        if missing_skus:
            msg = f"\n  ❌ [SubTask {st.id}] Validation FAILED:"
            print(msg)
            log_lines.append(msg)

            msg = f"      Required SKUs: {required_skus}"
            print(msg)
            log_lines.append(msg)

            msg = f"      Provided SKUs: {provided_skus}"
            print(msg)
            log_lines.append(msg)

            if missing_skus:
                msg = f"      ⚠️ Missing SKUs:"
                print(msg)
                log_lines.append(msg)
                for sku_id, shortage in missing_skus:
                    msg = f"         - SKU {sku_id}: Need {shortage} more"
                    print(msg)
                    log_lines.append(msg)

            # 详细列出涉及的料箱
            msg = f"      📦 Hit Totes ({len(st.assigned_tote_ids)} total):"
            print(msg)
            log_lines.append(msg)
            for task_idx, task in enumerate(st.execution_tasks):
                msg = f"         Task {task_idx} @ Stack {task.target_stack_id}:"
                print(msg)
                log_lines.append(msg)
                msg = f"           - Hit: {task.hit_tote_ids}"
                print(msg)
                log_lines.append(msg)
                msg = f"           - Noise: {task.noise_tote_ids}"
                print(msg)
                log_lines.append(msg)
                for tote_id in task.hit_tote_ids:
                    tote = problem.id_to_tote.get(tote_id)
                    if tote:
                        msg = f"             Tote {tote_id}: {tote.sku_quantity_map}"
                        print(msg)
                        log_lines.append(msg)

        else:
            msg = f"  ✅ [SubTask {st.id}] Validation PASSED ({len(required_skus)} SKU types, {sum(required_skus.values())} units)"
            print(msg)
            log_lines.append(msg)

        # --- 输出冗余信息 ---
        if redundant_totes_info:
            msg = f"      ⚠️ Redundant Totes Found ({len(redundant_totes_info)}): {redundant_totes_info}"
            print(msg)
            log_lines.append(msg)

        if logger:
            for line in log_lines:
                logger.log_validation(line)


    final_msg = f"  >>> ✅ SP3 Validation Complete. All SubTasks have sufficient tote coverage.\n"
    print(final_msg)
    if logger:
        logger.log_validation(final_msg)

if __name__ == "__main__":
    from Gurobi.sp1 import SP1_BOM_Splitter
    from Gurobi.sp2 import SP2_Station_Assigner
    from Gurobi.sp3 import SP3_Bin_Hitter
    from problemDto.createInstance import CreateOFSProblem

    print("\n" + "=" * 60)
    print("=== Integrated SP1-SP2-SP3-SP4 Pipeline Test ===")
    print("=" * 60)
    print("\n[Phase 0] Generating Problem Instance...")
    problem_dto = CreateOFSProblem.generate_problem_by_scale('SMALL')
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
    #输出每个subtask被分配到的工作站
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
    #验证每个task的选箱结果
    for task in physical_tasks:
        print(f"Physical Task {task.task_id}: SubTask {task.sub_task_id}, "
              f"Stack {task.target_stack_id}, Tote {task.hit_tote_ids}, noise {task.noise_tote_ids}"
              f"Load {task.total_load_count}, Service Time {task.robot_service_time}s")

    # # 5. SP4: 机器人路径规划
    sp4 = SP4_Robot_Router(problem_dto)
    checksp3hit(sub_tasks,problem_dto,logger=sp4.logger)
    arrival_times, robot_assign = sp4.solve(sub_tasks, use_mip=True)
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

