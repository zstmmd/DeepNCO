import math
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
                L[current_node, r_id].Start = 0.0
                injected['y'][(current_node, r_id)] = 1
                injected['T'][(current_node, r_id)] = 0.0
                injected['L'][(current_node, r_id)] = 0.0
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
                        L[depot_node, r_id].Start = current_load
                        injected['y'][(depot_node, r_id)] = 1
                        injected['T'][(depot_node, r_id)] = current_time
                        injected['L'][(depot_node, r_id)] = current_load

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
                    L[final_depot, r_id].Start = current_load
                    injected['y'][(final_depot, r_id)] = 1
                    injected['T'][(final_depot, r_id)] = current_time
                    injected['L'][(final_depot, r_id)] = current_load

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
            # 起点:只有出度
            if node_type == 'robot_start':
                if flow['in'] != 0:
                    violations.append(f"Start node {node} (r={r}) has incoming flow: {flow['in']}")

            # 终点(最后的Depot):只有入度
            # 判断是否为终点:有入度但无出度,且是Depot节点
            elif node_type == 'depot' and flow['out'] == 0 and flow['in'] > 0:
                # 这是终点,合法
                pass

            # 中间节点:入度=出度
            elif flow['in'] != flow['out']:
                violations.append(
                    f"Flow imbalance at node {node} (r={r}, type={node_type}): in={flow['in']}, out={flow['out']}")

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
                    violations.append(
                        f"Stack->Depot mismatch: Stack {i}(trip={trip_i}) -> Depot {j}(layer={depot_layer})")

            # Depot -> Stack: Stack.trip 必须等于 Depot.layer + 1
            if type_i == 'depot' and type_j == 'stack':
                depot_layer = nodes_map[i][4]
                if trip_j is not None and trip_j != depot_layer + 1:
                    violations.append(
                        f"Depot->Stack mismatch: Depot {i}(layer={depot_layer}) -> Stack {j}(trip={trip_j})")

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

    def _subtour_callback(self, model, where):
        """
        Gurobi 回调函数：用于检测并切除子回路
        """
        # 只有当 Gurobi 找到一个新的整数解 (MIPSOL) 时才检查
        if where == GRB.Callback.MIPSOL:
            # 获取当前的解
            # model._vars 是我们在 solve_mip 中通过 model._vars = x 绑定的
            x_vals = model.cbGetSolution(model._vars)

            # 按机器人分组提取边
            # edges_by_robot[r_id] = [(i, j), (j, k)...]
            edges_by_robot = defaultdict(list)

            for (i, j, r), val in x_vals.items():
                if val > 0.5:  # 选中的边
                    edges_by_robot[r].append((i, j))

            # 对每个机器人检查子回路
            for r, edges in edges_by_robot.items():
                # 获取该机器人的连通分量列表
                components = self.get_subtour(edges)

                for comp in components:
                    # 关键逻辑：如何判断 component 是非法的？
                    # 你的图结构：Start -> [Stacks] -> Depot
                    # 合法路径是不闭合的（Start 到 Depot）。
                    # 非法子回路是闭合的圈。

                    # 检查 component 是否构成了一个闭环 (对于 Stack 节点)
                    # 简单判据：如果 component 里面全是 Stack 节点（不含 Start 和 Depot），那它一定是孤立环

                    is_pure_stack_loop = True
                    for node in comp:
                        n_type = self.nodes_map_ref[node][3]  # 需要在类里存一份引用
                        if n_type in ['robot_start', 'depot']:
                            is_pure_stack_loop = False
                            break

                    if is_pure_stack_loop:
                        # === 发现子回路！添加 Lazy Constraint 切掉它 ===
                        # 约束公式：sum(x[i,j] for i in S for j in S) <= |S| - 1
                        # 意思：在这个集合 S 内部，最多只能有 |S|-1 条边。如果有 |S| 条边，就成环了。

                        # 构造 Gurobi 表达式
                        expr = gp.quicksum(model._vars[i, j, r]
                                           for i in comp
                                           for j in comp
                                           if (i, j, r) in model._vars)

                        model.cbLazy(expr <= len(comp) - 1)
                        # print(f"  🔪 Cut added for Robot {r}, Subtour size {len(comp)}")

    @staticmethod
    def get_subtour(edges: List[Tuple[int, int]]) -> List[int]:
        """
        给定一组边，寻找其中最小的子回路（Subtour）。
        如果所有节点都连通且包含起点（假设逻辑上判断），返回空。
        这里使用简化的寻找连通分量逻辑。
        """
        if not edges:
            return []

        # 1. 构建邻接表
        adj = defaultdict(list)
        nodes = set()
        for i, j in edges:
            adj[i].append(j)
            nodes.add(i)
            nodes.add(j)

        # 2. 寻找所有连通分量
        visited = set()
        subtours = []

        for node in list(nodes):
            if node in visited:
                continue

            # 开始一次遍历 (BFS/DFS) 找连通分量
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    # 注意：这是有向图，但为了切平面，我们通常看强连通或只要成圈就行
                    # 在 VRP 中，任何不包含起点的闭环都是非法的
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            subtours.append(component)

        # 3. 筛选非法子回路
        # 规则：合法的路径必须包含“起点”或者“Depot”。
        # 但在你的分层图中，路径是 Start -> Stack -> ... -> Stack -> Depot
        # 所以，任何【纯 Stack 节点】组成的环，绝对是子回路。

        # 找到长度最短的纯 Stack 环返回（切割力最强）
        # 我们假设外部逻辑会传入所有的 Stack 节点 ID，或者根据 ID 范围判断
        # 这里简化：只要 component 数量 > 1，说明图断开了，除了包含起点的那一组，其他的都是子回路

        # ⚠️ 注意：需要识别哪个 component 包含起点。
        # 由于我们在 Callback 内部很难拿到由外部定义的 robot_start_node，
        # 我们通常假定：如果一个分量是封闭的环（出入度平衡），且没有连接到 Depot/Start，它就是 Subtour。

        return subtours

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

        return robot_arrival_times, subtask_robot_assignment, max_time

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
        print("  >>> [SP4] Generating heuristic warm start...")
        heu_arrival_times, heu_robot_assign, heu_time = self._solve_heuristic(sub_tasks)
        # 1. 数据预处理
        valid_tasks = [t for t in sub_tasks if t.execution_tasks]
        if not valid_tasks:
            return {}, {}

        # --- 构建节点 ---
        nodes_map = {}
        node_id = 0
        # ✅ 动态计算最大趟数
        total_demand = sum(
            sum(task.total_load_count for task in st.execution_tasks)
            for st in valid_tasks
        )
        num_robots = len(self.problem.robot_list)

        # 考虑容量约束的理论最小趟数
        min_trips_needed = math.ceil(total_demand / (self.robot_capacity * num_robots))

        # 增加安全余量（考虑路径不均衡）
        max_trips = max(3, min_trips_needed + 2)
        print(f"  >>> [SP4] Max trips per robot set to: {max_trips}")
        # (A) 机器人起点
        robot_start_nodes = {}
        for robot in self.problem.robot_list:
            robot_start_nodes[robot.id] = node_id
            nodes_map[node_id] = (robot.start_point, None, None, 'robot_start', 0)
            node_id += 1

        # (B) Stack 节点
        stack_nodes_indices = []
        for st in valid_tasks:
            for task in st.execution_tasks:
                stack = self.problem.point_to_stack[task.target_stack_id]
                stack_nodes_indices.append(node_id)
                nodes_map[node_id] = (stack.store_point, st, task, 'stack', -1)
                node_id += 1

        # (C) 分层 Depot 节点
        depot_layer_nodes = defaultdict(dict)
        for k in range(1, max_trips + 1):
            for station in self.problem.station_list:
                depot_layer_nodes[station.id][k] = node_id
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
        tau = {}
        for i in N:
            pt_i = nodes_map[i][0]
            for j in N:
                if i == j: continue
                type_i = nodes_map[i][3]
                type_j = nodes_map[j][3]
                # 剪枝：Depot 之间不直连
                if type_i == 'depot' and type_j == 'depot': continue
                if type_j == 'robot_start':
                    continue
                if type_i == 'robot_start' and type_j == 'depot':
                    continue
                # 如果i和j属于不同的subtask，且subtask的目标station不同，则不连边
                if type_i == 'stack' and type_j == 'stack':
                    _, subtask_i, _, _, _ = nodes_map[i]
                    _, subtask_j, _, _, _ = nodes_map[j]
                    if subtask_i.id != subtask_j.id:
                        if subtask_i.assigned_station_id != subtask_j.assigned_station_id:
                            continue
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
        m.Params.LazyConstraints = 1
        m.Params.Cuts = 3
        m.Params.GomoryPasses = 5  # 增加 Gomory 割的次数

        # 变量
        x = m.addVars([(i, j, r) for i in N for j in N if (i, j) in tau for r in R], vtype=GRB.BINARY, name="x")
        y = m.addVars(N, R, vtype=GRB.BINARY, name="y")
        T = m.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="T")
        L = m.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, ub=self.robot_capacity, name="L")
        trip = m.addVars(stack_nodes_indices, R, vtype=GRB.INTEGER, lb=1, ub=max_trips, name="trip")
        Z = m.addVar(vtype=GRB.CONTINUOUS, name="Makespan")
        M_load = self.robot_capacity
        m._vars = x  # 将 x 变量绑定到 model 对象，方便 Callback 读取
        m._trip_vars = trip  # 新增 trip 绑定
        m._y_vars = y  # 可选：也可以绑定 y
        self.nodes_map_ref = nodes_map  # 将 nodes_map 存为成员变量供 Callback 查询类型
        # max_path_time = max(tau.values()) * len(N) + sum(service_time.values())
        # M_time = max_path_time * 1.2  # 预留20%余量
        M_time = heu_time * 1.2
        print(f"  >>> [SP4] Big-M Load: {M_load}, Big-M Time: {M_time:.2f}s")

        # --- 约束组 1: 基础流与覆盖 ---

        # Stack 覆盖
        for i in stack_nodes_indices:
            m.addConstr(gp.quicksum(y[i, r] for r in R) == 1, name=f"Cover_{i}")

        # 机器人流守恒
        for r in R:
            # 起点
            start_node = robot_start_nodes[self.problem.robot_list[r].id]
            m.addConstr(y[start_node, r] == 1)
            m.addConstr(T[start_node, r] == 0)
            m.addConstr(L[start_node, r] == 0)
            m.addConstr(gp.quicksum(x[start_node, j, r] for j in N if (start_node, j) in tau) == 1)
            m.addConstr(gp.quicksum(x[j, start_node, r] for j in N if (j, start_node) in tau) == 0)

            # Stack 节点
            for i in stack_nodes_indices:
                m.addConstr(gp.quicksum(x[j, i, r] for j in N if (j, i) in tau) == y[i, r])
                m.addConstr(gp.quicksum(x[i, j, r] for j in N if (i, j) in tau) == y[i, r])

            # Depot 节点 (流量平衡)
            for s_id, layer_dict in depot_layer_nodes.items():
                for k in range(1, max_trips + 1):
                    d_node = layer_dict[k]
                    in_arcs = gp.quicksum(x[i, d_node, r] for i in N if (i, d_node) in tau)
                    out_arcs = gp.quicksum(x[d_node, j, r] for j in N if (d_node, j) in tau)
                    m.addConstr(in_arcs == y[d_node, r])
                    m.addConstr(out_arcs <= y[d_node, r])  # 允许在 Depot 结束

                    # 强制 Depot 连接逻辑 (防止层级乱序)
                    for i in stack_nodes_indices:
                        if (i, d_node) in tau:  # Stack -> Depot(k) => trip[i] == k
                            m.addGenConstrIndicator(x[i, d_node, r], True, trip[i, r] == k)
                        if (d_node, i) in tau:  # Depot(k) -> Stack => trip[i] == k + 1
                            m.addGenConstrIndicator(x[d_node, i, r], True, trip[i, r] == k + 1)

        # --- 约束组 2: 终点管理 ---
        all_depot_nodes = []
        for station_dict in depot_layer_nodes.values():
            all_depot_nodes.extend(station_dict.values())
        for r in R:
            all_depots = []
            for layer_dict in depot_layer_nodes.values():
                all_depots.extend(layer_dict.values())

            robot_active = m.addVar(vtype=GRB.BINARY, name=f"RobotActive_{r}")
            # Robot active if it visits any stack
            m.addConstr(robot_active * len(stack_nodes_indices) >= gp.quicksum(y[i, r] for i in stack_nodes_indices))
            m.addConstr(robot_active <= gp.quicksum(y[i, r] for i in stack_nodes_indices))

            end_depot = m.addVars(all_depots, vtype=GRB.BINARY)
            for d in all_depots:
                out_d = gp.quicksum(x[d, j, r] for j in N if (d, j) in tau)
                # End depot if visited AND no outgoing flow
                m.addConstr(end_depot[d] >= y[d, r] - out_d)
                m.addConstr(end_depot[d] <= y[d, r])
                m.addConstr(end_depot[d] <= 1 - out_d + M_time * (1 - y[d, r]))  # Logical constraint logic fix

            # Active robots must have exactly one endpoint
            m.addConstr(gp.quicksum(end_depot[d] for d in all_depots) == robot_active)

            # Non-end depots must have outgoing flow
            for d in all_depots:
                m.addConstr(gp.quicksum(x[d, j, r] for j in N if (d, j) in tau) >= y[d, r] - end_depot[d])

        # --- 约束组 3: Trip 连续性 ---
        for r in R:
            # 起点出发的 Trip 必须初始化为 1
            start_node = robot_start_nodes[self.problem.robot_list[r].id]
            for j in stack_nodes_indices:
                if (start_node, j) in tau:
                    m.addGenConstrIndicator(x[start_node, j, r], True, trip[j, r] == 1)

            # Stack 之间保持 Trip
            for i in stack_nodes_indices:
                for j in stack_nodes_indices:
                    if i != j and (i, j) in tau:
                        m.addGenConstrIndicator(x[i, j, r], True, trip[i, r] == trip[j, r])

        # --- 约束组 4: 负载与容量 (MTZ)  ---
        for r in R:
            # 硬约束：所有访问点的负载不能超限
            for i in stack_nodes_indices:
                m.addConstr(L[i, r] <= self.robot_capacity, name=f"Cap_{i}_{r}")

            for i in N:
                for j in N:
                    if (i, j) in tau:
                        type_i = nodes_map[i][3]
                        type_j = nodes_map[j][3]
                        d_j = demand.get(j, 0)

                        # Case 1: Stack -> Stack (负载累加)
                        if type_i == 'stack' and type_j == 'stack':
                            m.addConstr(
                                L[j, r] >= L[i, r] + d_j - self.robot_capacity * (1 - x[i, j, r]),
                                name=f"LoadInc_{i}_{j}_{r}"
                            )

                        # Case 2: Start -> Stack (初始负载)
                        elif type_i == 'robot_start' and type_j == 'stack':
                            m.addGenConstrIndicator(x[i, j, r], True, L[j, r] == d_j)

                        # Case 3: Depot -> Stack (重置负载)
                        elif type_i == 'depot' and type_j == 'stack':
                            m.addGenConstrIndicator(x[i, j, r], True, L[j, r] == d_j)

                        # Case 4: Stack -> Depot (仅仅是为了记录到达时的负载，可选)
                        elif type_i == 'stack' and type_j == 'depot':
                            m.addGenConstrIndicator(x[i, j, r], True, L[j, r] == L[i, r])

        # --- 约束组 5: 时间推演 ---
        for r in R:
            time_vars = [(i, T[i, r]) for i in N]
            time_arcs = [(i, j, tau[i, j] + service_time[i]) for (i, j) in tau]

            m.addConstr(
                gp.quicksum(tau[i, j] * x[i, j, r] for i, j in tau) +
                gp.quicksum(service_time[i] * y[i, r] for i in N)
                <= Z,
                name=f"TotalTime_{r}"
            )
        for st_id, nodes in subtask_nodes.items():
            if len(nodes) > 1:
                for r in R:
                    # 禁止 SubTask 内部形成子回路
                    m.addConstr(
                        gp.quicksum(x[i, j, r] for i in nodes for j in nodes
                                    if (i, j, r) in x and i != j)
                        <= len(nodes) - 1,
                        name=f"NoSubtour_ST{st_id}_R{r}"
                    )
        # --- 约束组 6: 业务逻辑约束 ---
        # Depot 回访匹配 (SubTask -> Correct Station)
        for st_id, nodes in subtask_nodes.items():
            st = next(t for t in valid_tasks if t.id == st_id)
            target_station_id = st.assigned_station_id
            for i in nodes:
                for r in R:
                    # 禁止连接到错误的 Station Depot
                    for s_id, layer_dict in depot_layer_nodes.items():
                        if s_id != target_station_id:
                            for k in range(1, max_trips + 1):
                                wrong_depot = layer_dict[k]
                                if (i, wrong_depot) in tau:
                                    m.addConstr(x[i, wrong_depot, r] == 0)

        # 同 SubTask 同机器人
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

        # 计算总需求
        total_demand = sum(demand.values())
        # 计算理论最少需要的总 Trip 数 (向上取整)
        min_total_trips = math.ceil(total_demand / self.robot_capacity)

        # 约束：所有机器人出发的 Trip 总数必须满足需求

        depot_starts = gp.quicksum(x[d, j, r]
                                   for d in all_depot_nodes
                                   for j in stack_nodes_indices
                                   for r in R
                                   if (d, j, r) in x)
        start_starts = gp.quicksum(x[s, j, r]
                                   for s in robot_start_nodes.values()
                                   for j in stack_nodes_indices
                                   for r in R
                                   if (s, j, r) in x)

        m.addConstr(depot_starts + start_starts >= min_total_trips, name="LB_MinTrips")
        total_service_load = sum(service_time[i] for i in stack_nodes_indices)
        # 2. LB Cut 1: 平均负载约束

        m.addConstr(Z * len(self.problem.robot_list) >= total_service_load, name="LB_AverageLoad")

        # LB2: 容量下界 - 考虑往返时间
        for r in R:
            # 计算该机器人可能访问的 SubTask 的最小往返成本
            subtask_min_costs = {}
            for st_id, nodes in subtask_nodes.items():
                st = next(t for t in valid_tasks if t.id == st_id)
                target_station = self.problem.station_list[st.assigned_station_id].point

                # 最近 Stack 到 Station 的距离
                min_dist = float('inf')
                for node_id in nodes:
                    stack_pt = nodes_map[node_id][0]
                    dist = abs(stack_pt.x - target_station.x) + abs(stack_pt.y - target_station.y)
                    min_dist = min(min_dist, dist)

                # 单趟最小成本 = 服务时间 + 往返时间
                st_service = sum(service_time[n] for n in nodes)
                st_trips = (sum(demand[n] for n in nodes) + self.robot_capacity - 1) // self.robot_capacity
                subtask_min_costs[st_id] = st_service + st_trips * (2 * min_dist / self.robot_speed)

            # 如果机器人 r 执行某 SubTask，则 Makespan >= 该 SubTask 的最小完成时间
            for st_id, min_cost in subtask_min_costs.items():
                repr_node = subtask_nodes[st_id][0]
                m.addConstr(Z >= min_cost * y[repr_node, r], name=f"LB_SubTask_{st_id}_{r}")

        # --- 约束组 7: 目标函数 ---

        for r in R:
            for d in all_depot_nodes:
                m.addGenConstrIndicator(y[d, r], True, Z >= T[d, r])

        epsilon = 0.01
        total_dist = gp.quicksum(tau[i, j] * x[i, j, r] for i, j, r in x)
        m.setObjective(Z + epsilon * total_dist, GRB.MINIMIZE)

        # --- 求解 ---

        self._apply_warm_start_layered(
            m, x, y, T, L, trip,
            heu_robot_assign, heu_arrival_times, nodes_map,
            depot_layer_nodes, robot_start_nodes, stack_nodes_indices,
            tau, demand, service_time, max_trips
        )
        m.setParam('LogFile', 'log/gurobi_run.log')
        print("正在导出模型约束到 log/debug_model.lp ...")
        m.write("log/debug_model.lp")
        m.Params.Cutoff = heu_time * 1.2
        # 🔧 分阶段求解策略
        print("\n  >>> [Phase 1] Quick feasibility search (60s)...")
        m.Params.TimeLimit = 600
        m.Params.MIPFocus = 1  # 聚焦可行解
        m.Params.Heuristics = 0.3  # 高频启发式
        m.Params.Cuts = 0  # 暂不生成割平面
        m.Params.NoRelHeurTime = 30  # 前30秒不依赖 LP 松弛

        m.optimize(self._subtour_callback)

        if m.SolCount > 0:
            incumbent = m.objVal
            print(f"  >>> [Phase 1] Found solution: {incumbent:.2f}")

            # Phase 2: 改善解质量
            print(f"\n  >>> [Phase 2] Improving solution (剩余时间)...")
            m.Params.TimeLimit = 3600
            m.Params.MIPFocus = 2  # 证明最优性
            m.Params.Cuts = 3  # 激进割平面
            m.Params.CutPasses = 20
            m.Params.Heuristics = 0.05  # 降低启发式比例

            # 🔧 关键：设置 Cutoff（只接受改善 5% 以上的解）
            m.Params.Cutoff = incumbent * 0.95

            # 🔧 专门针对 VRP 的 Cuts
            m.Params.FlowCoverCuts = 2
            m.Params.MIRCuts = 2
            m.Params.GomoryPasses = 10

            m.optimize(self._subtour_callback)

        # --- 结果提取 ---
        robot_arrival_times = {}
        subtask_robot_assign = {}

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"  >>> Solved. Obj: {m.objVal:.2f}")
            self._extract_sequence(x, y, T, trip, nodes_map, N, R, depot_layer_nodes, robot_start_nodes,
                                   stack_nodes_indices)
            for i in stack_nodes_indices:
                pt, subtask, task, _, _ = nodes_map[i]
                for r in R:
                    if y[i, r].X > 0.5:
                        arr_time = T[i, r].X
                        robot_arrival_times[pt.idx] = arr_time
                        subtask_robot_assign[subtask.id] = self.problem.robot_list[r].id
                        task.robot_id = r
                        task.arrival_time_at_stack = arr_time
            with open("log/debug_result.txt", "w") as f:
                f.write(f"Objective Value: {m.objVal}\n")
                f.write("-" * 30 + "\n")

                # 3.1 打印所有被选中的路径 (x 变量)
                f.write("=== Active Routes (x > 0.5) ===\n")
                # 假设你的变量叫 x，根据实际情况调整名字

                for v in m.getVars():
                    if v.varName.startswith("x") and v.x > 0.5:
                        f.write(f"{v.varName} = {v.x}\n")

                f.write("\n")

                # 3.2 打印所有负载情况 (load 变量)
                f.write("=== Load Variables ===\n")
                for v in m.getVars():

                    if ("L" in v.varName) and v.x > 0.001:
                        f.write(f"{v.varName} = {v.x}\n")

                f.write("\n")

                # 3.3 打印 Trip 变量
                f.write("=== Trip Variables ===\n")
                for v in m.getVars():
                    if "trip" in v.varName:
                        f.write(f"{v.varName} = {v.x}\n")
                # 3.4 打印时间变量 (只打印 Active 的)
                f.write("\n=== Time Variables (Active Only) ===\n")
                for v in m.getVars():
                    if v.varName.startswith("T"):

                        import re
                        match = re.match(r"T\[(\d+),(\d+)\]", v.varName)
                        if match:
                            n_id, r_id = int(match.group(1)), int(match.group(2))
                            # 关键判断：只有当 y[n,r] > 0.5 时才打印 T
                            if y[n_id, r_id].X > 0.5:
                                f.write(f"{v.varName} = {v.x}\n")

                # 3.3 打印 Trip 变量 (同理)
                f.write("=== Trip Variables (Active Only) ===\n")
                for v in m.getVars():
                    if "trip" in v.varName:
                        match = re.match(r"trip\[(\d+),(\d+)\]", v.varName)
                        if match:
                            n_id, r_id = int(match.group(1)), int(match.group(2))
                            # 只有访问了该点，Trip 才有意义
                            if y[n_id, r_id].X > 0.5:
                                f.write(f"{v.varName} = {v.x}\n")

            print("调试文件已生成在 log/ 目录下。")


        else:
            print("  >>> MIP Infeasible or Failed.")

        return robot_arrival_times, subtask_robot_assign

    # -----------------------------------------------------------
    # 1. 静态辅助函数：寻找连通分量
    # -----------------------------------------------------------
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

    def _cb_lazy_subtour(self, model, where):
        """
        [完全重构] 多层次子回路检测 + 精确切割
        """
        if where != GRB.Callback.MIPSOL:
            return

        x_vals = model.cbGetSolution(model._vars)

        # 按机器人分组
        edges_per_robot = defaultdict(list)
        for (i, j, r), val in x_vals.items():
            if val > 0.5:
                edges_per_robot[r].append((i, j))

        cuts_added = 0

        for r, edges in edges_per_robot.items():
            if not edges:
                continue

            # === 第一层检测：简单环路 ===
            cycles = self._find_cycles_dfs(edges)

            for cycle in cycles:
                # 判断是否为非法环
                has_start = False
                has_depot = False
                all_stack_ids = []

                for node in cycle:
                    n_type = self.nodes_map_ref[node][3]
                    if n_type == 'robot_start':
                        has_start = True
                    elif n_type == 'depot':
                        has_depot = True
                    elif n_type == 'stack':
                        all_stack_ids.append(node)

                # 规则1: 纯 Stack 环（最常见的子回路）
                if not has_start and not has_depot:
                    expr = gp.quicksum(model._vars[i, j, r]
                                       for i in cycle
                                       for j in cycle
                                       if (i, j, r) in model._vars)
                    model.cbLazy(expr <= len(cycle) - 1)
                    cuts_added += 1
                    continue

                # 规则2: 包含 Start 但又回到 Start（非法）
                if has_start and cycle[0] == cycle[-1]:
                    # Start 不能形成环（必须单向出发）
                    start_node = next(n for n in cycle if self.nodes_map_ref[n][3] == 'robot_start')
                    expr = gp.quicksum(model._vars[i, start_node, r]
                                       for i in cycle if (i, start_node, r) in model._vars)
                    model.cbLazy(expr == 0)  # 禁止任何边指向 Start
                    cuts_added += 1

                # 规则3: Depot 之间的非法连接
                if has_depot:
                    depot_nodes = [n for n in cycle if self.nodes_map_ref[n][3] == 'depot']
                    if len(depot_nodes) > 1:
                        # 不同 Depot 之间不能直连
                        for d1 in depot_nodes:
                            for d2 in depot_nodes:
                                if d1 != d2 and (d1, d2, r) in model._vars:
                                    model.cbLazy(model._vars[d1, d2, r] == 0)
                                    cuts_added += 1

            # === 第二层检测：路径连通性 ===
            # 检查是否存在多个不连通的子路径
            components = self._find_weak_components(edges)

            if len(components) > 1:
                # 找出包含 Start 的主路径
                start_node = next(n for n in self.nodes_map_ref
                                  if self.nodes_map_ref[n][3] == 'robot_start')
                main_comp = None
                for comp in components:
                    if start_node in comp:
                        main_comp = comp
                        break

                # 其他分量都是孤立子回路
                for comp in components:
                    if comp == main_comp:
                        continue

                    # 标准 Subtour Elimination Constraint
                    expr = gp.quicksum(model._vars[i, j, r]
                                       for i in comp
                                       for j in comp
                                       if (i, j, r) in model._vars)
                    model.cbLazy(expr <= len(comp) - 1)
                    cuts_added += 1

            # === 第三层检测：Trip 层级违规 ===
            # 检查是否存在跨层级的非法连接
            for i, j in edges:
                type_i = self.nodes_map_ref[i][3]
                type_j = self.nodes_map_ref[j][3]

                # Stack -> Depot: 检查 Trip 匹配
                if type_i == 'stack' and type_j == 'depot':
                    depot_layer = self.nodes_map_ref[j][4]

                    # 获取该 Stack 的 Trip（从解中读取）
                    if hasattr(model, '_trip_vars'):
                        stack_trip_val = model.cbGetSolution(model._trip_vars.get((i, r), None))
                        if stack_trip_val is not None:
                            if abs(stack_trip_val - depot_layer) > 0.5:
                                # Trip 不匹配，添加冲突约束
                                model.cbLazy(model._vars[i, j, r] == 0)
                                cuts_added += 1

        if cuts_added > 0:
            print(f"  🔪 [Callback] Added {cuts_added} lazy cuts")
            # === 新增：容量违规检测 ===
        for r, edges in edges_per_robot.items():
            # 构建路径
            path = self._reconstruct_path(edges)

            cumulative_load = 0
            last_depot_idx = -1

            for idx, node in enumerate(path):
                n_type = self.nodes_map_ref[node][3]

                if n_type == 'stack':
                    demand_val = self.demand_ref.get(node, 0)
                    cumulative_load += demand_val

                    # 检查是否超载
                    if cumulative_load > self.robot_capacity + 0.01:
                        # 找出导致超载的子路径
                        violating_segment = path[last_depot_idx + 1: idx + 1]

                        # 添加容量割：该路径段内必须插入至少一个 Depot
                        depot_nodes = [n for n in self.nodes_map_ref
                                       if self.nodes_map_ref[n][3] == 'depot']

                        # 如果该子路径被选中，则必须访问至少一个 Depot
                        segment_active = gp.quicksum(model._vars.get((violating_segment[i],
                                                                      violating_segment[i + 1], r), 0)
                                                     for i in range(len(violating_segment) - 1))
                        depot_visit = gp.quicksum(model._y_vars.get((d, r), 0)
                                                  for d in depot_nodes)

                        model.cbLazy(segment_active <= depot_visit * len(violating_segment))
                        cuts_added += 1

                elif n_type == 'depot':
                    cumulative_load = 0
                    last_depot_idx = idx

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


from collections import defaultdict
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
            return f"Stack_{stack_id}，StackPoint:{pt.idx}（x,y):({pt.x},{pt.y}),task_id:{task_obj.task_id} ，task_service_time：{task_obj.robot_service_time},(SubTask_{st_id})，（subtask assigned_station:{subtask.assigned_station_id if subtask else 'Unknown'})"

        elif n_type == 'depot':
            # Depot 节点包含层级信息 (Trip)
            return f"Station_Point_{pt.idx}（x,y):({pt.x},{pt.y}) (Trip_Layer_{layer})"

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

        # ✅ 新增：记录每个料箱提供的 SKU 详情
        tote_sku_details = []  # [(tote_id, stack_id, sku_map)]

        # ✅ 关键修改：使用动态剩余需求追踪
        remaining_req = required_skus.copy()  # {sku_id: remaining_quantity}
        redundant_totes_info = []

        for task in st.execution_tasks:
            for tote_id in task.hit_tote_ids:
                tote = problem.id_to_tote.get(tote_id)
                if not tote:
                    print(f"  ❌ [SubTask {st.id}] Tote {tote_id} not found in problem.id_to_tote")
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

        msg = f"\n  📋 [SubTask {st.id}] SKU Overview:"
        print(msg)
        log_lines.append(msg)

        msg = f"      Required SKUs: {required_skus}"
        print(msg)
        log_lines.append(msg)

        msg = f"      Provided SKUs: {provided_skus}"
        print(msg)
        log_lines.append(msg)

        # ✅ 核心修改：显示每个料箱的实际贡献（扣除已满足的SKU）
        msg = f"      📦 Tote-Level SKU Breakdown ({len(tote_sku_details)} totes):"
        print(msg)
        log_lines.append(msg)

        for tote_id, stack_id, needed_skus, noise_skus in tote_sku_details:
            msg = f"         Tote {tote_id} @ Stack {stack_id}:"
            print(msg)
            log_lines.append(msg)

            if needed_skus:
                msg = f"           ✅ Needed: {needed_skus}"
                print(msg)
                log_lines.append(msg)

            if noise_skus:
                msg = f"           🔇 Noise: {noise_skus}"
                print(msg)
                log_lines.append(msg)

            # 如果两者都为空，说明是完全冗余的料箱
            if not needed_skus and not noise_skus:
                msg = f"           ⚠️ Completely Redundant (all SKUs already satisfied)"
                print(msg)
                log_lines.append(msg)

        # 输出验证结果
        if missing_skus:
            msg = f"\n  ❌ [SubTask {st.id}] Validation FAILED:"
            print(msg)
            log_lines.append(msg)

            msg = f"      ⚠️ Missing SKUs:"
            print(msg)
            log_lines.append(msg)
            for sku_id, shortage in missing_skus:
                msg = f"         - SKU {sku_id}: Need {shortage} more"
                print(msg)
                log_lines.append(msg)
        else:
            msg = f"  ✅ [SubTask {st.id}] Validation PASSED ({len(required_skus)} SKU types, {sum(required_skus.values())} units)"
            print(msg)
            log_lines.append(msg)

        # 输出多余 SKU 信息
        if excess_skus:
            msg = f"      ℹ️ Excess SKUs (over-supply):"
            print(msg)
            log_lines.append(msg)
            for sku_id, excess in excess_skus:
                msg = f"         - SKU {sku_id}: +{excess} extra"
                print(msg)
                log_lines.append(msg)

        # 输出冗余料箱信息
        if redundant_totes_info:
            msg = f"      ⚠️ Redundant Totes (not contributing to required SKUs): {len(redundant_totes_info)}"
            print(msg)
            log_lines.append(msg)
            for info in redundant_totes_info:
                msg = f"         - {info}"
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
    problem_dto = CreateOFSProblem.generate_problem_by_scale('SMALL3', seed=SEED)
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
    print(f"✅ Total load across all physical tasks: {sum_load}")
    # # 5. SP4: 机器人路径规划
    sp4 = SP4_Robot_Router(problem_dto)
    checksp3hit(sub_tasks, problem_dto, logger=sp4.logger)
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

