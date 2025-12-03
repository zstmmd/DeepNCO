import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Optional

from Gurobi.sp1 import SP1_BOM_Splitter
from entity.subTask import SubTask
from entity.station import Station
from entity.point import Point
from problemDto.createInstance import CreateOFSProblem
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class SP2_Station_Assigner:
    """
    SP2 子问题求解器：任务指派与排序 (Task-Station Assignment & Sequencing)

    核心逻辑：
    1. 任务开始时间受限于：工作站上一任务完成时间 & 物料到达时间。
    2. 物料到达时间计算：基于上一轮 SP3 选定的具体 Tote 位置 + SP4 计算的机器人到达时间。
    """

    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem = problem_dto
        self.stations = problem_dto.station_list
        self.picking_time = OFSConfig.PICKING_TIME
        self.robot_speed = OFSConfig.ROBOT_SPEED
        # 固定返回时间 (Fixed Return Time): 卸货、输送线对接等时间
        self.fixed_return_time = 15.0

    def solve_initial_heuristic(self):
        """
        基于贪婪负载均衡的启发式算法生成初始解。
        不依赖 SP4 的反馈（假设物料即时可达或平均时间）。
        """
        print(f"  >>> [SP2] Generating Initial Solution (Heuristic)...")

        # 记录每个工作站当前的预计完工时间 (Available Time)
        station_makespan = {s.id: 0.0 for s in self.stations}
        # 记录每个工作站已分配的任务数 (用于生成 sequence rank)
        station_counts = {s.id: 0 for s in self.stations}

        # 简单的负载均衡：将任务分配给当前完工时间最早的站点
        for task in self.problem.subtask_list:
            # 估算任务处理时间
            proc_time = len(task.sku_list) * self.picking_time
            # 贪婪选择：找到负荷最小的站
            best_station_id = min(station_makespan, key=station_makespan.get)

            # 执行分配
            start_time = station_makespan[best_station_id]
            rank = station_counts[best_station_id]

            # 更新 Task 对象
            task.assigned_station_id = best_station_id
            task.station_sequence_rank = rank
            task.estimated_process_start_time = start_time

            # 更新站点状态
            station_makespan[best_station_id] += proc_time
            station_counts[best_station_id] += 1

        print(f"  >>> [SP2] Initial Heuristic Done. Max Makespan: {max(station_makespan.values()):.2f}")

    def solve_mip_with_feedback(self,
                                tasks: List[SubTask],
                                sp4_robot_arrival_times: Dict[int, float],
                                sp3_tote_selection: Optional[Dict[int, List[int]]] = None,
                                sp3_sorting_costs: Optional[Dict[int, float]] = None):
        """
        基于 MIP 优化任务指派

        :param sp3_sorting_costs:  {sub_task_id: extra_time}
               来自 SP3 的理货惩罚时间 (Sorting Cost)。
               仅当 SP3 选择 Sort 模式时，工作站需要额外时间处理无效箱子。
        """
        print(f"  >>> [SP2] Optimizing with MIP (Feedback: RobotTime, ToteSel, SortingCost)...")

        if not tasks: return

        # 1. 数据准备
        K = range(len(tasks))
        S = range(len(self.stations))

        # 估算最大位次
        avg_load = len(tasks) / len(self.stations)
        max_ranks = math.ceil(avg_load * 1.2) + 2
        P = range(max_ranks)

        # --- 核心修改：计算任务处理时长 P_k ---
        # P_k = (SKU拣选时间) + (无效箱子理货时间，来自SP3反馈)
        p_times = {}
        for k_idx, task in enumerate(tasks):
            # 基础拣选时间
            base_time = len(task.sku_list) * self.picking_time
            # 额外理货时间 (SC_u)
            extra_time = sp3_sorting_costs.get(task.id, 0.0) if sp3_sorting_costs else 0.0

            p_times[k_idx] = base_time + extra_time
        # 预计算：物料到达时间矩阵 Arrival[k][s]
        # 核心：利用 SP3 选定的 Tote 和 SP4 的时间计算
        mat_arrival = self._compute_arrival_matrix(tasks, sp4_robot_arrival_times, sp3_tote_selection)

        # 2. 建模
        m = gp.Model("SP2_RHMA_Iter")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = 60

        # 决策变量
        y = m.addVars(K, P, S, vtype=GRB.BINARY, name="y")  # 指派 y[任务, 位次, 站点]
        t = m.addVars(P, S, vtype=GRB.CONTINUOUS, name="t")  # 开始时间 t[位次, 站点]
        C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max")  # Makespan

        M = 100000

        # 3. 约束

        # (1) 任务覆盖与位次唯一
        for k in K:
            m.addConstr(gp.quicksum(y[k, p, s] for p in P for s in S) == 1)
        for s in S:
            for p in P:
                m.addConstr(gp.quicksum(y[k, p, s] for k in K) <= 1)

        # (2) 紧凑性约束
        for s in S:
            for p in range(max_ranks - 1):
                m.addConstr(gp.quicksum(y[k, p + 1, s] for k in K) <= gp.quicksum(y[k, p, s] for k in K))
        # (4) 时间流约束 (使用更新后的 p_times)
        for s in S:
            # 第 0 位
            expr_arrival_0 = gp.quicksum(y[k, 0, s] * mat_arrival[k][s] for k in K)
            m.addConstr(t[0, s] >= expr_arrival_0)

            for p in range(max_ranks - 1):
                # 这里的 p_times[k] 已经包含了理货时间
                proc_time_p = gp.quicksum(y[k, p, s] * p_times[k] for k in K)

                # A. 资源约束 (上一任务完成)
                m.addConstr(t[p + 1, s] >= t[p, s] + proc_time_p)

                # B. 物料约束
                for k in K:
                    m.addConstr(t[p + 1, s] >= mat_arrival[k][s] - M * (1 - y[k, p + 1, s]))

                # C. Makespan
                m.addConstr(C_max >= t[p, s] + proc_time_p)

        # 4. 求解
        m.setObjective(C_max, GRB.MINIMIZE)
        m.optimize()

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"  >>> [SP2] MIP Solved. Makespan: {C_max.X:.2f}")
            self._apply_solution(tasks, y, t, K, P, S)
        else:
            print("  >>> [SP2] MIP Failed.")

    def _compute_arrival_matrix(self,
                                tasks: List[SubTask],
                                robot_arrival_map: Dict[int, float],
                                sp3_selection: Optional[Dict[int, List[int]]]) -> List[List[float]]:
        """
        预计算物料到达时间矩阵。
        T_arrival = Max( T_robot_at_stack + T_stack_to_station ) + Fixed
        """
        K = len(tasks)
        S = len(self.stations)
        matrix = [[0.0] * S for _ in range(K)]

        for k_idx, task in enumerate(tasks):
            # 获取该任务所需的特定 Tote ID 列表 (来自 SP3 反馈)
            selected_totes = sp3_selection.get(task.id) if sp3_selection else None

            # 获取涉及的物理存储点
            related_points = self._get_task_related_points(task, selected_totes)

            for s_idx, station in enumerate(self.stations):
                max_arrival = 0.0

                for pt in related_points:
                    # 1. 机器人到达堆垛的时间 (SP4 反馈)
                    # 如果没有反馈数据 (如第一轮)，给一个默认估算值 (30s)
                    t_robot = robot_arrival_map.get(pt.idx, 30.0)

                    # 2. 堆垛搬运到工作站的时间 (物理距离/速度)
                    dist = abs(pt.x - station.point.x) + abs(pt.y - station.point.y)
                    t_travel = dist / self.robot_speed

                    # 总时间
                    total = t_robot + t_travel + self.fixed_return_time

                    # 取最晚到达的那个 Tote 的时间 (齐套)
                    if total > max_arrival:
                        max_arrival = total

                matrix[k_idx][s_idx] = max_arrival

        return matrix

    def _get_task_related_points(self, task: SubTask, selected_tote_ids: List[int] = None) -> List[Point]:
        """
        获取任务涉及的物理存储点。
        策略：优先使用 SP3 选定的 Tote；若无，使用启发式兜底。
        """
        points = set()

        # 场景 A: 有 SP3 反馈 (准确位置)
        if selected_tote_ids:
            for tote_id in selected_tote_ids:
                tote = self.problem.id_to_tote.get(tote_id)
                if tote and tote.store_point:
                    points.add(tote.store_point)

        # 场景 B: 无反馈 (初始阶段，模糊估算)
        else:
            for sku in task.sku_list:
                # 简单启发式：取第一个可用的 Tote 位置作为参考
                if sku.storeToteList:
                    tote_id = sku.storeToteList[0]
                    tote = self.problem.id_to_tote.get(tote_id)
                    if tote and tote.store_point:
                        points.add(tote.store_point)

        return list(points)

    def _apply_solution(self, tasks, y, t, K, P, S):
        """回填结果"""
        for k in K:
            for s in S:
                for p in P:
                    if y[k, p, s].X > 0.5:
                        tasks[k].assigned_station_id = self.stations[s].id
                        tasks[k].station_sequence_rank = p
                        tasks[k].estimated_process_start_time = t[p, s].X
                        break
if __name__ == "__main__":
    # 初始化问题和求解器
    scales = ["SMALL", "MEDIUM"]
    problem_dto = CreateOFSProblem.generate_problem_by_scale('SMALL')
    sp1_solver = SP1_BOM_Splitter(problem_dto)

    #  默认生成（基于空间聚类，使用全局容量限制）
    initial_tasks = sp1_solver.solve(use_mip=False)
    #验证是否覆盖order的所有sku
    order_to_skus: Dict[int, List[int]] = defaultdict(list)
    for task in initial_tasks:
        order_id = task.parent_order.order_id
        sku_ids = [sku.id for sku in task.sku_list]
        order_to_skus[order_id].extend(sku_ids)

    for order in problem_dto.order_list:
        original_skus = sorted(order.order_product_id_list)
        generated_skus = sorted(order_to_skus[order.order_id])
        assert original_skus == generated_skus, f"Order {order.order_id} SKU mismatch!"
    problem_dto.subtask_list = initial_tasks
    print("Initial task generation verified: all orders covered correctly.")
    # 2. SP2 初始启发式解
    sp2_solver = SP2_Station_Assigner(problem_dto)
    sp2_solver.solve_initial_heuristic()
    print('1')