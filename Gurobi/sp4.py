import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Set
from collections import defaultdict

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
        基于 Gurobi MIP 的精确求解
        """
        # 1. 数据预处理
        # 过滤出已完成 SP3 选箱的任务
        valid_tasks = [t for t in sub_tasks if t.execution_tasks]
        if not valid_tasks:
            print("  [SP4] No valid tasks with execution details.")
            return {}, {}
        
        # 构建节点集合 N
        # 节点0: Depot (工作站)
        # 节点1~n: 需要访问的堆垛
        nodes_map = {}  # {node_id: (point, subtask, task_obj)}
        node_id = 0
        
        # Depot nodes (每个工作站作为起点/终点)
        depot_nodes = {}  # {station_id: node_id}
        for station in self.problem.station_list:
            depot_nodes[station.id] = node_id
            nodes_map[node_id] = (station.point, None, None)
            node_id += 1
        
        # Stack nodes (每个 SubTask 的执行任务对应的堆垛)
        task_to_node = {}  # {(subtask_id, stack_id): node_id}
        for st in valid_tasks:
            for task in st.execution_tasks:
                stack = self.problem.point_to_stack[task.target_stack_id]
                task_to_node[(st.id, task.target_stack_id)] = node_id
                nodes_map[node_id] = (stack.store_point, st, task)
                node_id += 1
        
        N = range(node_id)
        R = range(len(self.problem.robot_list))
        
        # 2. 计算参数
        # 距离矩阵 tau[i][j]
        tau = [[0.0] * node_id for _ in range(node_id)]
        for i in N:
            for j in N:
                if i != j:
                    pt_i = nodes_map[i][0]
                    pt_j = nodes_map[j][0]
                    dist = abs(pt_i.x - pt_j.x) + abs(pt_i.y - pt_j.y)
                    tau[i][j] = dist / self.robot_speed
        
        # 服务时间 service_time[i]
        service_time = {}
        for i in N:
            _, subtask, task_obj = nodes_map[i]
            if task_obj:  # Stack node
                service_time[i] = task_obj.robot_service_time
            else:  # Depot
                service_time[i] = 0.0
        
        # 需求量 demand[i]
        demand = {}
        for i in N:
            _, subtask, task_obj = nodes_map[i]
            if task_obj:
                demand[i] = task_obj.total_load_count
            else:
                demand[i] = 0
        
        # SubTask 所需的节点集合
        subtask_nodes = defaultdict(list)
        for i in N:
            _, subtask, _ = nodes_map[i]
            if subtask:
                subtask_nodes[subtask.id].append(i)
        
        # 3. 建模
        m = gp.Model("SP4_CVRP")
        m.Params.OutputFlag = 1
        m.Params.TimeLimit = 120
        
        # 决策变量
        x = m.addVars(N, N, R, vtype=GRB.BINARY, name="x")
        y = m.addVars(N, R, vtype=GRB.BINARY, name="y")
        T = m.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="T")
        L = m.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="L")
        
        M = 100000
        
        # 4. 约束
        
        # (1) 每个堆垛节点必须被恰好一个机器人服务
        for i in N:
            if i not in depot_nodes.values():  # 非 Depot 节点
                m.addConstr(gp.quicksum(y[i, r] for r in R) == 1, 
                           name=f"Coverage_{i}")
        
        # (2) 流守恒：入度 = 出度 = y[i,r]
        for i in N:
            if i not in depot_nodes.values():
                for r in R:
                    m.addConstr(
                        gp.quicksum(x[i, j, r] for j in N if j != i) == y[i, r],
                        name=f"Flow_out_{i}_{r}"
                    )
                    m.addConstr(
                        gp.quicksum(x[j, i, r] for j in N if j != i) == y[i, r],
                        name=f"Flow_in_{i}_{r}"
                    )
         # (2.5) 新增：Depot 流守恒（支持多次往返）
        # 每个机器人在每个 Depot 的出度 = 入度
        for depot in depot_nodes.values():
            for r in R:
                m.addConstr(
                    gp.quicksum(x[depot, j, r] for j in N if j != depot) == 
                    gp.quicksum(x[j, depot, r] for j in N if j != depot),
                    name=f"Depot_Balance_{depot}_{r}"
                )
        
        
        for st_id, node_list in subtask_nodes.items():
            subtask_obj = next(t for t in valid_tasks if t.id == st_id)
            target_depot = depot_nodes[subtask_obj.assigned_station_id]
            
            if len(node_list) > 1:
                # 所有节点必须由同一机器人服务（但可以分多次）
                first_node = node_list[0]
                for r in R:
                    for other_node in node_list[1:]:
                        m.addConstr(
                            y[other_node, r] == y[first_node, r],
                            name=f"SubTask_SameRobot_{st_id}_{r}"
                        )
        
        # (5) 容量约束
        for i in N:
            if i not in depot_nodes.values():
                for j in N:
                    if j != i:
                        for r in R:
                            if j in depot_nodes.values():
                                # 返回 Depot 时卸货，负载清零
                                m.addConstr(
                                    L[j, r] <= M * (1 - x[i, j, r]),
                                    name=f"Unload_{i}_{j}_{r}"
                                )
                            else:
                                # 堆垛间移动，累加负载
                                m.addConstr(
                                    L[j, r] >= L[i, r] + demand[j] - M * (1 - x[i, j, r]),
                                    name=f"Load_{i}_{j}_{r}"
                                )
    
        for i in N:
            if i not in depot_nodes.values():
                for r in R:
                    m.addConstr(L[i, r] >= demand[i] * y[i, r],
                            name=f"MinLoad_{i}_{r}")
                    m.addConstr(L[i, r] <= self.robot_capacity,
                            name=f"MaxLoad_{i}_{r}")
        
        # (6) 时间窗约束
        for i in N:
            for j in N:
                if i != j:
                    for r in R:
                        if i in depot_nodes.values() and j in depot_nodes.values():
                            # Depot 到 Depot：禁止直接连接
                            m.addConstr(x[i, j, r] == 0, name=f"NoDepotDirect_{i}_{j}_{r}")
                        
                        elif i in depot_nodes.values() and j not in depot_nodes.values():
                            # 从 Depot 出发到堆垛：时间继承 Depot 的到达时间
                            m.addConstr(
                                T[j, r] >= T[i, r] + tau[i][j] - M * (1 - x[i, j, r]),
                                name=f"Time_FromDepot_{i}_{j}_{r}"
                            )
                        
                        elif i not in depot_nodes.values() and j in depot_nodes.values():
                            # 从堆垛返回 Depot：记录到达时间
                            m.addConstr(
                                T[j, r] >= T[i, r] + service_time[i] + tau[i][j] 
                                        - M * (1 - x[i, j, r]),
                                name=f"Time_Return_{i}_{j}_{r}"
                            )
                        
                        else:
                            # 堆垛到堆垛：正常时间传递
                            m.addConstr(
                                T[j, r] >= T[i, r] + service_time[i] + tau[i][j] 
                                        - M * (1 - x[i, j, r]),
                                name=f"Time_{i}_{j}_{r}"
                            )
                
        # 5. 目标函数
        obj = gp.quicksum(tau[i][j] * x[i, j, r] 
                         for i in N for j in N if i != j for r in R)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # 6. 求解
        m.optimize()
        
        # 7. 结果解析
        robot_arrival_times = {}
        subtask_robot_assignment = {}
        
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"  >>> [SP4] MIP Solved. Objective: {m.objVal:.2f}")
            
            # 提取路径和时间
            for i in N:
                pt, subtask, task_obj = nodes_map[i]
                if task_obj:  # Stack node
                    for r in R:
                        if y[i, r].X > 0.5:
                            arrival = T[i, r].X
                            robot_arrival_times[pt.idx] = arrival
                            
                            # 更新 Task 对象
                            task_obj.robot_id = r
                            task_obj.arrival_time_at_stack = arrival
                            
                            # 计算到达工作站时间
                            station_pt = self.problem.station_list[task_obj.target_station_id].point
                            travel_back = abs(pt.x - station_pt.x) + abs(pt.y - station_pt.y) / self.robot_speed
                            task_obj.arrival_time_at_station = arrival + task_obj.robot_service_time + travel_back
                            
                            # SubTask 分配
                            if subtask.id not in subtask_robot_assignment:
                                subtask_robot_assignment[subtask.id] = r
                                subtask.assigned_robot_id = r
            
            # 提取访问顺序
            self._extract_sequence(x, y,T, nodes_map, N, R,depot_nodes)
            
        else:
            print(f"  [SP4] MIP Failed with status {m.status}")
            
        return robot_arrival_times, subtask_robot_assignment

    def _extract_sequence(self, x, y, T, nodes_map, N, R, depot_nodes):
        """
        提取并记录机器人的访问顺序（支持多趟次）

        参数:
            x: 边决策变量
            y: 节点访问决策变量
            T: 时间变量 ← 新增
            nodes_map: 节点映射
            N: 节点集合
            R: 机器人集合
            depot_nodes: Depot 节点映射
        """
        for r in R:
            print(f"\n  === Robot {r} Routes ===")

            # 收集所有访问节点并按时间排序
            visited_nodes = []
            for i in N:
                _, subtask, task_obj = nodes_map[i]
                if task_obj and y[i, r].X > 0.5:
                    visited_nodes.append((T[i, r].X, i, task_obj))

            if not visited_nodes:
                print(f"  No tasks assigned")
                continue

            visited_nodes.sort(key=lambda x: x[0])  # 按时间排序

            # 通过路径追踪提取真实的往返趟次
            trips = self._reconstruct_trips(x, y, T, visited_nodes, nodes_map, N, r, depot_nodes)

            # 输出结果
            for trip_idx, trip_nodes in enumerate(trips, start=1):
                if trip_nodes:
                    start_time = trip_nodes[0][0]
                    end_time = trip_nodes[-1][0]
                    print(f"  Trip {trip_idx}: {len(trip_nodes)} tasks, time [{start_time:.1f}s, {end_time:.1f}s]")

                    for seq, (time, node_id, task_obj) in enumerate(trip_nodes):
                        task_obj.robot_visit_sequence = seq
                        print(
                            f"    [{seq}] Node {node_id} @ {time:.1f}s (SubTask {task_obj.sub_task_id}, Stack {task_obj.target_stack_id})")

    def _reconstruct_trips(self, x, y, T, visited_nodes, nodes_map, N, r, depot_nodes):
        """
        通过追踪边关系重建真实的往返趟次

        返回: List[List[Tuple[time, node_id, task_obj]]]
        """
        trips = []
        visited_set = set()

        # 找到所有从 Depot 出发的起始边
        depot_starts = []
        for depot in depot_nodes.values():
            for j in N:
                if j not in depot_nodes.values() and x[depot, j, r].X > 0.5:
                    depot_starts.append((depot, j, T[j, r].X))

        # 按出发时间排序
        depot_starts.sort(key=lambda x: x[2])

        for start_depot, first_node, _ in depot_starts:
            if first_node in visited_set:
                continue

            # 追踪这一趟的完整路径
            current_trip = []
            current = first_node

            while current is not None and current not in visited_set:
                _, subtask, task_obj = nodes_map[current]

                if task_obj:
                    current_trip.append((T[current, r].X, current, task_obj))
                    visited_set.add(current)

                # 找下一个节点
                next_node = None
                for j in N:
                    if x[current, j, r].X > 0.5:
                        if j in depot_nodes.values():
                            # 返回 Depot，结束本趟
                            break
                        else:
                            next_node = j
                            break

                current = next_node

            if current_trip:
                trips.append(current_trip)

        return trips

    def _solve_heuristic(self, sub_tasks: List[SubTask]) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        启发式求解：贪婪最近邻 + 容量检查
        """
        print(f"  >>> [SP4] Using Heuristic Solver...")

        valid_tasks = [t for t in sub_tasks if t.execution_tasks]
        if not valid_tasks:
            return {}, {}

        robot_arrival_times = {}
        subtask_robot_assignment = {}

        # 初始化机器人状态
        robot_loads = {r.id: 0 for r in self.problem.robot_list}
        robot_times = {r.id: 0.0 for r in self.problem.robot_list}
        robot_positions = {r.id: self.problem.station_list[0].point for r in self.problem.robot_list}  # 初始在第一个工作站
        robot_routes = {r.id: [] for r in self.problem.robot_list}  # 记录每个机器人的路径

        # 为每个 SubTask 分配机器人并规划路径
        for st in valid_tasks:
            # 计算 SubTask 总需求
            total_demand = sum(task.total_load_count for task in st.execution_tasks)
            target_station = self.problem.station_list[st.assigned_station_id]

            # 找最优机器人（考虑当前位置、负载、时间）
            best_robot = None
            best_cost = float('inf')

            for robot in self.problem.robot_list:
                r_id = robot.id

                # 检查容量约束（可能需要多次往返）
                trips_needed = (total_demand + self.robot_capacity - 1) // self.robot_capacity

                # 估算完成时间
                current_pos = robot_positions[r_id]
                first_stack = self.problem.point_to_stack[st.execution_tasks[0].target_stack_id]

                # 从当前位置到目标工作站
                to_station_dist = abs(current_pos.x - target_station.point.x) + abs(
                    current_pos.y - target_station.point.y)
                to_station_time = to_station_dist / self.robot_speed

                # 从工作站到第一个堆垛
                station_to_stack = abs(target_station.point.x - first_stack.store_point.x) + abs(
                    target_station.point.y - first_stack.store_point.y)

                # 简单估算：(往返次数 * 平均往返时间) + 服务时间
                avg_trip_time = (2 * station_to_stack / self.robot_speed) + sum(
                    task.robot_service_time for task in st.execution_tasks) / trips_needed
                estimated_time = robot_times[r_id] + to_station_time + trips_needed * avg_trip_time

                if estimated_time < best_cost:
                    best_cost = estimated_time
                    best_robot = r_id

            # 分配机器人
            st.assigned_robot_id = best_robot
            subtask_robot_assignment[st.id] = best_robot

            # 详细路径规划（支持多次往返）
            current_load = robot_loads[best_robot]
            current_time = robot_times[best_robot]
            current_pos = robot_positions[best_robot]
            trip_sequence = 0

            # 如果当前不在目标工作站，先移动过去
            if current_pos.idx != target_station.point.idx:
                travel_dist = abs(current_pos.x - target_station.point.x) + abs(current_pos.y - target_station.point.y)
                current_time += travel_dist / self.robot_speed
                current_pos = target_station.point

            # 按堆垛分组（贪婪最近邻）
            remaining_tasks = list(st.execution_tasks)

            while remaining_tasks:
                # 当前趟次的任务列表
                current_trip_tasks = []
                trip_load = 0

                # 贪婪选择：按距离和容量选择任务
                while remaining_tasks and trip_load < self.robot_capacity:
                    # 找最近的堆垛
                    best_task = None
                    best_dist = float('inf')

                    for task in remaining_tasks:
                        stack = self.problem.point_to_stack[task.target_stack_id]
                        dist = abs(current_pos.x - stack.store_point.x) + abs(current_pos.y - stack.store_point.y)

                        # 检查容量
                        if trip_load + task.total_load_count <= self.robot_capacity:
                            if dist < best_dist:
                                best_dist = dist
                                best_task = task

                    if best_task is None:
                        break

                    # 执行移动和服务
                    stack = self.problem.point_to_stack[best_task.target_stack_id]
                    travel_time = best_dist / self.robot_speed
                    current_time += travel_time

                    # 更新任务信息
                    best_task.robot_id = best_robot
                    best_task.arrival_time_at_stack = current_time
                    best_task.robot_visit_sequence = trip_sequence
                    robot_arrival_times[stack.store_point.idx] = current_time

                    # 服务时间
                    current_time += best_task.robot_service_time

                    # 更新状态
                    trip_load += best_task.total_load_count
                    current_pos = stack.store_point
                    current_trip_tasks.append(best_task)
                    remaining_tasks.remove(best_task)

                    trip_sequence += 1

                # 返回工作站
                return_dist = abs(current_pos.x - target_station.point.x) + abs(current_pos.y - target_station.point.y)
                return_time = return_dist / self.robot_speed
                current_time += return_time
                current_pos = target_station.point

                # 计算到达工作站时间
                for task in current_trip_tasks:
                    stack = self.problem.point_to_stack[task.target_stack_id]
                    travel_back = abs(stack.store_point.x - target_station.point.x) + abs(
                        stack.store_point.y - target_station.point.y)
                    task.arrival_time_at_station = task.arrival_time_at_stack + task.robot_service_time + travel_back / self.robot_speed

                # 记录路径
                robot_routes[best_robot].append({
                    'trip': len([r for r in robot_routes[best_robot] if r]) + 1,
                    'tasks': current_trip_tasks,
                    'load': trip_load,
                    'start_time': current_trip_tasks[0].arrival_time_at_stack if current_trip_tasks else current_time,
                    'end_time': current_time
                })

            # 更新机器人状态
            robot_loads[best_robot] = 0  # 卸货完成
            robot_times[best_robot] = current_time
            robot_positions[best_robot] = current_pos

        # ========================================
        # 结果解析和输出（类似 MIP 求解器）
        # ========================================
        print(f"\n  >>> [SP4] Heuristic Solved.")
        print(f"  - Total arrival times: {len(robot_arrival_times)}")
        print(f"  - SubTask assignments: {len(subtask_robot_assignment)}")

        # 输出每个机器人的路径
        for r_id in sorted(robot_routes.keys()):
            routes = robot_routes[r_id]
            if not routes:
                continue

            print(f"\n  === Robot {r_id} Routes (Heuristic) ===")
            for route in routes:
                print(f"  Trip {route['trip']}: {len(route['tasks'])} tasks, "
                      f"load={route['load']}/{self.robot_capacity}, "
                      f"time [{route['start_time']:.1f}s, {route['end_time']:.1f}s]")

                for seq, task in enumerate(route['tasks']):
                    print(f"    [{seq}] Stack {task.target_stack_id} @ {task.arrival_time_at_stack:.1f}s "
                          f"(SubTask {task.sub_task_id}, Load={task.total_load_count})")

        # 统计信息
        total_distance = sum(
            sum(abs(route['tasks'][i].target_stack_id - route['tasks'][i - 1].target_stack_id)
                if i > 0 else 0
                for i in range(len(route['tasks'])))
            for routes in robot_routes.values()
            for route in routes
        )

        total_trips = sum(len(routes) for routes in robot_routes.values())
        max_time = max(robot_times.values()) if robot_times else 0

        print(f"\n  === Heuristic Summary ===")
        print(f"  - Total trips: {total_trips}")
        print(f"  - Makespan: {max_time:.2f}s")
        print(
            f"  - Active robots: {sum(1 for routes in robot_routes.values() if routes)}/{len(self.problem.robot_list)}")

        return robot_arrival_times, subtask_robot_assignment


if __name__ == "__main__":
    from Gurobi.sp1 import SP1_BOM_Splitter
    from Gurobi.sp2 import SP2_Station_Assigner
    from Gurobi.sp3 import SP3_Bin_Hitter
    from problemDto.createInstance import CreateOFSProblem
    
    print("\n" + "="*60)
    print("=== Integrated SP1-SP2-SP3-SP4 Pipeline Test ===")
    print("="*60)
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
    
    # 5. SP4: 机器人路径规划
    sp4 = SP4_Robot_Router(problem_dto)
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