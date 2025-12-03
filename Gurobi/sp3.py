import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from entity.subTask import SubTask
from entity.task import Task
from entity.tote import Tote
from entity.stack import Stack
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class SP3_Bin_Hitter:
    """
    SP3 子问题求解器：料箱命中 (Bin Hitting)

    基于 Gurobi MIP 求解。
    Sort 模式逻辑：搬运区间 [Min_Index, Max_Index] 内的所有料箱。
    """

    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem = problem_dto

        # --- 成本参数 ---
        self.t_shift = OFSConfig.PACKING_TIME  # 机器人抓取时间
        self.t_lift = OFSConfig.LIFTING_TIME  # 机器人移位/挖掘时间
        self.t_move = OFSConfig.MOVE_EXTRA_TOTE_TIME  # 工作站单箱理货时间

        self.alpha = 1.0  # 机器人成本权重
        self.w_routing = 0.5  # 路径成本权重
        self.BigM = 10000

    def solve(self,
              sub_tasks: List[SubTask],
              beta_congestion: float = 1.0,
              sp4_routing_costs: Dict[int, float] = None
              ) -> Tuple[List[Task], Dict[int, List[int]], Dict[int, float]]:

        print(f"  >>> [SP3] Solving Per-SubTask (No Pooling, Beta={beta_congestion:.2f})...")

        physical_tasks: List[Task] = []
        final_tote_selection = defaultdict(list)
        final_sorting_costs = defaultdict(float)
        self._global_task_id = 0

        # 遍历每个子任务，独立求解
        for task in sub_tasks:
            # 如果尚未分配工作站，无法计算距离成本，跳过或由SP2处理
            if task.assigned_station_id == -1: continue

            # 为该任务求解最优选箱策略
            p_tasks, totes, cost = self._solve_single_subtask_mip(
                task, beta_congestion, sp4_routing_costs
            )

            # 记录结果
            physical_tasks.extend(p_tasks)
            final_tote_selection[task.id] = totes
            final_sorting_costs[task.id] = cost

        return physical_tasks, final_tote_selection, final_sorting_costs

    def _solve_single_subtask_mip(self,
                                  task: SubTask,
                                  beta: float,
                                  routing_costs: Dict[int, float]) -> Tuple[List[Task], List[int], float]:

        # --- A. 数据准备 ---
        # 1. 需求统计
        demand = defaultdict(int)
        for sku in task.sku_list:
            demand[sku.id] += 1

        # 2. 候选堆垛与料箱
        relevant_stack_ids = set()
        stack_bin_map = defaultdict(list)

        for sku_id in demand:
            sku_obj = self.problem.id_to_sku[sku_id]
            for tote_id in sku_obj.storeToteList:
                tote = self.problem.id_to_tote[tote_id]
                if tote.store_point:
                    u_idx = tote.store_point.idx
                    relevant_stack_ids.add(u_idx)
                    if tote not in stack_bin_map[u_idx]:
                        stack_bin_map[u_idx].append(tote)

        U = list(relevant_stack_ids)
        if not U: return [], [], 0.0

        # --- B. 构建 MIP ---
        m = gp.Model(f"SP3_Task_{task.id}")
        m.Params.OutputFlag = 0

        # 变量
        m_flip = m.addVars(U, vtype=GRB.BINARY, name="m_flip")
        m_sort = m.addVars(U, vtype=GRB.BINARY, name="m_sort")

        all_bins = [b for u in U for b in stack_bin_map[u]]
        bin_ids = [b.id for b in all_bins]
        u_use = m.addVars(bin_ids, vtype=GRB.BINARY, name="u_use")

        # 辅助变量
        idx_high = m.addVars(U, vtype=GRB.INTEGER, name="idx_h")
        idx_low = m.addVars(U, vtype=GRB.INTEGER, name="idx_l")

        cost_rc = m.addVars(U, vtype=GRB.CONTINUOUS, name="c_rc")
        cost_sc = m.addVars(U, vtype=GRB.CONTINUOUS, name="c_sc")  # 这是我们要回传的重点
        cost_fc = m.addVars(U, vtype=GRB.CONTINUOUS, name="c_fc")

        # --- C. 约束条件 ---

        # 1. 需求覆盖
        for sku_id, qty in demand.items():
            lhs = gp.LinExpr()
            sku_obj = self.problem.id_to_sku[sku_id]
            for tote_id in sku_obj.storeToteList:
                if tote_id in u_use:
                    tote = self.problem.id_to_tote[tote_id]
                    q = tote.sku_quantity_map.get(sku_id, 0)
                    lhs += u_use[tote_id] * q
            m.addConstr(lhs >= qty)

        total_load = gp.LinExpr()

        for u in U:
            bins = stack_bin_map[u]
            stack_obj = self.problem.point_to_stack[u]
            max_h = stack_obj.current_height  # 物理高度
            top_tote = max(bins, key=lambda b: b.layer)  # 顶层箱

            # 模式互斥与激活 (保持不变)
            bin_sum = gp.quicksum(u_use[b.id] for b in bins)
            m.addConstr(m_flip[u] + m_sort[u] <= 1)
            m.addConstr(bin_sum <= self.BigM * (m_flip[u] + m_sort[u]))
            m.addConstr(m_flip[u] + m_sort[u] <= bin_sum)

            # --- Range 约束 (Sort 模式核心) ---
            # idx_high: 选中箱子的最大层级 (Active时有效)
            # idx_low: 选中箱子的最小层级 (Active时有效)
            for b in bins:
                l = b.layer
                m.addConstr(idx_high[u] >= l * u_use[b.id])
                # 如果选中，low <= l；如果不选中，low <= M (不约束)
                # 配合最小化 Range 的逻辑，low 会逼近最小值
                m.addConstr(idx_low[u] <= l * u_use[b.id] + max_h * (1 - u_use[b.id]))

            # --- 容量计算 (Load Calculation) ---
            load_u = m.addVar(vtype=GRB.CONTINUOUS, name=f"load_{u}")

            # Case 1: Flip Mode -> Load = bin_sum (选中几个占几个)
            m.addConstr(load_u >= bin_sum - self.BigM * (1 - m_flip[u]))

            # Case 2: Sort Mode -> Load = Range Size (idx_high - idx_low + 1)
            # 【修正点】不再是 max_h，而是动态计算的区间跨度
            range_size = idx_high[u] - idx_low[u] + 1
            m.addConstr(load_u >= range_size - self.BigM * (1 - m_sort[u]))

            total_load += load_u

            # --- 成本计算 (保持不变) ---
            # Flip Cost
            flip_val = gp.LinExpr()
            for b in bins:
                is_deep = 1 if b.layer < (max_h - 1) else 0
                c = self.alpha * (self.t_shift + is_deep * self.t_lift)
                flip_val += u_use[b.id] * c
            m.addConstr(cost_fc[u] >= flip_val - self.BigM * (1 - m_flip[u]))

            # Robot Sort Cost
            # 如果是 Sort 模式，是否需要移开顶层？
            # 逻辑：如果 Range 包含顶层 (idx_high == max_h-1)，则不需要额外 Lift
            # 或者沿用之前的逻辑：如果 top_tote 被选中，则不需要 Lift。
            # 但现在的 Sort 是区间搬运，如果区间本身就包含顶层（比如拿 Layer 3-5, 5是顶），那么就没有阻挡。
            # 如果区间是 Layer 1-2，顶层是 5，那么其实需要把 3-5 都搬走或者移开。
            # 这里假设 Sort 模式下，机器人把 [idx_low, idx_high] 这一段搬走。
            # 如果 idx_high < max_h - 1 (没包含顶层)，物理上是无法直接抽出来的，通常需要先把顶层移开。
            # 因此这里保留 Lift 惩罚：如果 idx_high != 顶层，则产生 Lift 成本。
            # 简化实现：沿用之前的 u_use[top] 判断

            is_top_used = u_use[top_tote.id] if top_tote.id in u_use else 0
            rc_expr = self.alpha * (self.t_shift + self.t_lift * (1 - is_top_used))
            m.addConstr(cost_rc[u] >= rc_expr - self.BigM * (1 - m_sort[u]))

            # Station Sort Cost (SC_u)
            # Noise = Range - Useful
            noise_expr = range_size - bin_sum
            sc_raw = beta * self.t_move * noise_expr
            m.addConstr(cost_sc[u] >= sc_raw - self.BigM * (1 - m_sort[u]))

            # 非负约束
            m.addConstr(cost_fc[u] >= 0)
            m.addConstr(cost_rc[u] >= 0)
            m.addConstr(cost_sc[u] >= 0)



        # 3. 添加容量硬约束
        m.addConstr(total_load <= self.robot_capacity, name="Capacity_Limit")

        # --- D. 目标函数 ---
        obj = gp.LinExpr()
        obj += gp.quicksum(cost_rc[u] + cost_sc[u] + cost_fc[u] for u in U)

        # 路径成本软耦合
        if routing_costs:
            for u in U:
                r_cost = routing_costs.get(u, 0.0)
                obj += self.w_routing * r_cost * (m_flip[u] + m_sort[u])

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        # --- E. 结果解析 ---
        p_tasks = []
        selected_totes = []
        total_sc_cost = 0.0

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            for u in U:
                if m_flip[u].X > 0.5 or m_sort[u].X > 0.5:
                    mode = 'SORT' if m_sort[u].X > 0.5 else 'FLIP'

                    target_totes = [b.id for b in stack_bin_map[u] if u_use[b.id].X > 0.5]
                    selected_totes.extend(target_totes)

                    # 累计 SC_u
                    if mode == 'SORT':
                        total_sc_cost += cost_sc[u].X

                    # 计算服务时间
                    svc_time = cost_rc[u].X if mode == 'SORT' else cost_fc[u].X

                    # 创建物理任务 (1对1)
                    new_task = Task(
                        task_id=self._global_task_id,
                        sub_task_id=task.id,  # 精准对应
                        target_stack_id=u,
                        target_station_id=task.assigned_station_id,
                        operation_mode=mode,
                        target_tote_ids=target_totes
                    )
                    new_task.estimated_service_time = svc_time
                    p_tasks.append(new_task)
                    self._global_task_id += 1
        else:
            # 处理无解情况 (容量不足)
            print(f"  [SP3] Task {task.id} Infeasible (Likely Capacity). Regret Feedback needed.")
            # 这里可以返回空的 selected_totes，触发外层 Capacity Feedback
            pass

        return p_tasks, selected_totes, total_sc_cost

    class SP3_Heuristic_Solver:
        """
        SP3 启发式求解器 (Heuristic Solver)
        功能：快速生成 SP3 的初始解，不依赖 Gurobi。
        策略：贪婪覆盖 (Greedy Set Cover) + 成本对比 (Cost Benefit Analysis)。
        """

        def __init__(self, problem_dto: OFSProblemDTO):
            self.problem = problem_dto

            # --- 成本参数 ---
            self.t_shift = OFSConfig.PACKING_TIME
            self.t_lift = OFSConfig.LIFTING_TIME
            self.t_move = OFSConfig.PLACE_TOTE_TIME  # Station sorting move time
            self.robot_capacity = OFSConfig.ROBOT_CAPACITY
            self.alpha = 1.0

        def solve(self,
                  sub_tasks: List[SubTask],
                  beta_congestion: float = 1.0
                  ) -> Tuple[List[Task], Dict[int, List[int]], Dict[int, float]]:

            print(f"  >>> [SP3 Heuristic] Generating Initial Solution (Beta={beta_congestion:.2f})...")

            physical_tasks: List[Task] = []
            final_tote_selection = defaultdict(list)
            final_sorting_costs = defaultdict(float)
            self._global_task_id = 0

            # 遍历每个子任务独立求解
            for task in sub_tasks:
                # 1. 贪婪选箱：决定去哪些堆垛拿哪些箱子
                # 返回: {stack_idx: [tote_obj, ...]}
                stack_plan = self._greedy_tote_selection(task)

                # 2. 模式决策：对每个涉及的堆垛决定 Flip 还是 Sort
                for stack_idx, target_totes in stack_plan.items():
                    stack = self.problem.point_to_stack[stack_idx]

                    # 决策模式并计算成本
                    mode, cost_sc, svc_time = self._decide_operation_mode(
                        stack, target_totes, beta_congestion
                    )

                    # 创建物理任务
                    new_task = Task(
                        task_id=self._global_task_id,
                        related_subtask_ids=[task.id],  # 独立求解，只关联自己
                        target_stack_id=stack_idx,
                        target_station_id=task.assigned_station_id,
                        operation_mode=mode,
                        target_tote_ids=[t.id for t in target_totes]
                    )
                    new_task.estimated_service_time = svc_time
                    physical_tasks.append(new_task)
                    self._global_task_id += 1

                    # 记录反馈数据
                    final_tote_selection[task.id].extend([t.id for t in target_totes])
                    if cost_sc > 0:
                        final_sorting_costs[task.id] += cost_sc

            return physical_tasks, final_tote_selection, final_sorting_costs

        def _greedy_tote_selection(self, task: SubTask) -> Dict[int, List[Tote]]:
            """
            贪婪策略：寻找最少的堆垛覆盖所有 SKU
            """
            # 待满足的 SKU 集合
            pending_skus = set(sku.id for sku in task.sku_list)
            # SKU -> {tote_id: tote_obj} 缓存
            sku_availability = defaultdict(list)

            # 1. 构建索引：SKU 在哪里？
            for sku_id in pending_skus:
                sku_obj = self.problem.id_to_sku[sku_id]
                for tote_id in sku_obj.storeToteList:
                    tote = self.problem.id_to_tote[tote_id]
                    if tote.store_point:
                        sku_availability[sku_id].append(tote)

            selected_stacks_map = defaultdict(list)  # stack_idx -> list[Tote]

            # 2. 贪婪循环
            while pending_skus:
                # 评分：每个堆垛包含多少个"当前还需要的" SKU
                stack_score = defaultdict(int)
                stack_candidate_totes = defaultdict(list)  # stack -> [tote_to_pick]

                # 遍历所有待满足 SKU 的候选位置
                for sku_id in pending_skus:
                    candidates = sku_availability[sku_id]
                    for tote in candidates:
                        s_idx = tote.store_point.idx

                        # 关键逻辑：如果这个 Stack 已经被选中过，优先复用（成本极低）
                        score_bonus = 100 if s_idx in selected_stacks_map else 1

                        # 简单评分：包含一个需要的 SKU +1分
                        # 注意：这里简化处理，假设一个 Tote 只满足一个 SKU，或者被选中一次就把所有需要的都带走
                        if tote not in stack_candidate_totes[s_idx]:
                            stack_score[s_idx] += score_bonus
                            stack_candidate_totes[s_idx].append(tote)

                if not stack_score:
                    # 理论上不应发生，除非无库存
                    print(f"Error: Cannot find totes for remaining SKUs: {pending_skus}")
                    break

                # 选择得分最高的堆垛
                best_stack_idx = max(stack_score, key=stack_score.get)
                chosen_totes = stack_candidate_totes[best_stack_idx]

                # 将选中的 Tote 加入结果
                for t in chosen_totes:
                    if t not in selected_stacks_map[best_stack_idx]:
                        selected_stacks_map[best_stack_idx].append(t)

                    # 标记这些 Tote 覆盖了哪些 SKU
                    for s_in_tote in t.skus_list:
                        if s_in_tote.id in pending_skus:
                            pending_skus.remove(s_in_tote.id)

            return selected_stacks_map

        def _decide_operation_mode(self,
                                   stack: Stack,
                                   target_totes: List[Tote],
                                   beta: float) -> Tuple[str, float, float]:
            """
            成本对比与模式决策
            Returns: (Mode, SC_cost, Service_Time)
            """
            # --- 准备数据 ---
            # 目标箱子的层级索引 (0=Bottom, N=Top)
            target_indices = [t.layer for t in target_totes]
            top_index = stack.current_height - 1

            # --- 1. 计算 FLIP 成本 ---
            # Flip Load = 选中箱子数
            flip_load = len(target_totes)

            cost_flip = 0.0
            for idx in target_indices:
                is_deep = 1 if idx < top_index else 0
                cost_flip += self.alpha * (self.t_shift + is_deep * self.t_lift)

            # --- 2. 计算 SORT 成本 ---
            deepest_index = min(target_indices)
            highest_index = max(target_indices)  # 选中箱子中的最高层

            # Sort Load = 区间跨度
            range_size = highest_index - deepest_index + 1

            # 容量校验：如果区间太大，背不动，只能 Flip
            if range_size > self.robot_capacity:
                return 'FLIP', 0.0, cost_flip

            # Robot Cost (Sort):
            # 如果区间包含顶层 (highest_index == top_index)，则无需 Lift，直接背走
            # 否则需要先移开顶层
            is_top_included = (highest_index == top_index)
            cost_robot_sort = self.alpha * (self.t_shift + self.t_lift * (0 if is_top_included else 1))

            # Station Cost (Sort): SC = Beta * move * Noise
            useful_count = len(target_totes)
            noise_count = range_size - useful_count
            cost_sc = beta * self.t_move * noise_count

            total_sort_cost = cost_robot_sort + cost_sc

            # --- 3. 决策 ---
            if total_sort_cost < cost_flip:
                return 'SORT', cost_sc, cost_robot_sort
            else:
                return 'FLIP', 0.0, cost_flip