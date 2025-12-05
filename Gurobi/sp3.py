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
    SP3 子问题求解器：料箱命中

    基于 Gurobi MIP 求解。
    FLIP 模式 (挖掘/翻箱)：带回来的全是需要的箱子，不占用额外容量，工作站不需要剔除杂箱。
    Sort 模式逻辑：搬运区间 [Min_Index, Max_Index] 内的所有料箱。
    """

    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem = problem_dto

        # --- 成本参数 ---
        self.t_shift = OFSConfig.PACKING_TIME  # 机器人抓取时间
        self.t_lift = OFSConfig.LIFTING_TIME  # 机器人移位/挖掘时间
        self.t_move = OFSConfig.MOVE_EXTRA_TOTE_TIME  # 工作站单箱理货时间
        self.robot_capacity = OFSConfig.ROBOT_CAPACITY  # 机器人容量
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

        # 1. 需求统计
        demand = defaultdict(int)
        for sku in task.sku_list:
            demand[sku.id] += 1

        # 2. 候选堆垛与料箱
        stack_tote_set = defaultdict(set)

        for sku in task.unique_sku_list:
            for tote_id in sku.storeToteList:
                # 利用 problem_dto 的字典直接获取 Tote 对象
                tote = self.problem.id_to_tote.get(tote_id)

                # 确保料箱存在且当前在库位上 (store_point 不为空)
                if tote and tote.store_point:
                    stack_idx = tote.store_point.idx
                    stack_tote_set[stack_idx].add(tote)

        U = list(stack_tote_set.keys())

        stack_bin_map = {u: list(totes) for u, totes in stack_tote_set.items()}

        if not U: return [], [], 0.0

        # --- B. 构建 MIP ---
        m = gp.Model(f"SP3_Task_{task.id}")
        m.Params.OutputFlag = 0

        # 变量
        m_flip = m.addVars(U, vtype=GRB.BINARY, name="m_flip")
        m_sort = m.addVars(U, vtype=GRB.BINARY, name="m_sort")

        all_bins = [b for u in U for b in stack_bin_map[u]]
        bin_ids = [b.id for b in all_bins]
        u_use = m.addVars(bin_ids, vtype=GRB.BINARY, name="u_use")  #u_use[tote_id]

        # 辅助变量
        idx_high = m.addVars(U, vtype=GRB.INTEGER, name="idx_h")
        idx_low = m.addVars(U, vtype=GRB.INTEGER, name="idx_l")
        # is_top_inc[u] = 1 表示对于堆垛 u，Sort 区间包含了物理顶层
        is_top_inc = m.addVars(U, vtype=GRB.BINARY, name="is_top_inc")
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
            top_layer_idx = max_h - 1
            # 模式互斥与激活 (保持不变)
            bin_sum = gp.quicksum(u_use[b.id] for b in bins)
            m.addConstr(m_flip[u] + m_sort[u] <= 1)
            m.addConstr(bin_sum <= self.BigM * (m_flip[u] + m_sort[u]))
            m.addConstr(m_flip[u] + m_sort[u] <= bin_sum)
            m.addConstr(idx_high[u] >= top_layer_idx * is_top_inc[u] - self.BigM * (1 - m_sort[u]))
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
            rc_expr = self.alpha * (self.t_shift + self.t_lift * (1 - is_top_inc[u]))
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
        selected_totes_for_feedback = []  # 用于返回给外部记录
        total_sc_cost = 0.0

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:

            # 遍历所有堆垛 U
            for u in U:
                # 检查该堆垛是否被激活 (Flip 或 Sort)
                if m_flip[u].X > 0.5 or m_sort[u].X > 0.5:

                    # 1. 确定模式
                    is_sort_mode = (m_sort[u].X > 0.5)
                    mode = 'SORT' if is_sort_mode else 'FLIP'

                    # 准备临时列表
                    physical_carried_totes = []  # 物理带走的 (Target)
                    hit_totes = []  # 命中的 (Hit)
                    noise_totes = []  # 噪音 (Noise)
                    layer_range = None

                    # 获取该堆垛所有的 Tote 对象 (需确保 stack_bin_map 里的 tote 有 layer 属性)
                    all_totes_in_stack = stack_bin_map[u]

                    # --- 核心逻辑分支 ---

                    if is_sort_mode:
                        # === SORT 模式逻辑 ===
                        # 读取 Gurobi 决策出的物理层级区间
                        # 注意：Gurobi 可能会输出 3.9999 或 4.0001，需 round
                        val_low = int(round(idx_low[u].X))
                        val_high = int(round(idx_high[u].X))
                        layer_range = (val_low, val_high)

                        # 物理遍历：只要层级在 [low, high] 之间，就被机器人带走
                        for tote in all_totes_in_stack:
                            if val_low <= tote.layer <= val_high:
                                physical_carried_totes.append(tote.id)

                                # 判断是否为有用箱子 (u_use > 0.5)
                                # 也可以通过需求反查，这里直接信赖 u_use 变量
                                if u_use[tote.id].X > 0.5:
                                    hit_totes.append(tote.id)
                                else:
                                    noise_totes.append(tote.id)

                        # 记录 Station Cost (基于实际产生的噪音)
                        # 虽然 cost_sc[u] 变量里有值，但重新计算更保险
                        #这里检查一下cost_sc[u]的值是否等于beta * self.t_move * len(noise_totes)
                        if cost_sc[u].X != beta * self.t_move * len(noise_totes):
                            print(f"Warning: Negative SC cost for Task {task.id}, Stack {u}")
                            print(f"  Computed SC: {cost_sc[u].X}, Recalculated SC: {beta * self.t_move * len(noise_totes)}")
                        total_sc_cost += beta * self.t_move * len(noise_totes)

                        # 记录 Service Time (Sort 模式下是 Robot Cost)
                        task_svc_time = cost_rc[u].X

                    else:
                        # === FLIP 模式逻辑 ===
                        # Flip 模式下，物理带走的 = 选中的
                        for tote in all_totes_in_stack:
                            if u_use[tote.id].X > 0.5:
                                physical_carried_totes.append(tote.id)
                                hit_totes.append(tote.id)

                        # Flip 模式没有 Noise
                        noise_totes = []
                        layer_range = None

                        # 记录 Service Time (Flip 模式下是 Flip Cost)
                        task_svc_time = cost_fc[u].X

                    # --- 3. 生成 Task 对象 ---
                    # 仅当有箱子被搬运时才生成任务 (防止空解)
                    if physical_carried_totes:
                        new_task = Task(
                            task_id=self._global_task_id,  # 注意：需确保 global_id 在类级别正确自增
                            sub_task_id=task.id,
                            target_stack_id=u,
                            target_station_id=task.assigned_station_id,
                            operation_mode=mode,

                            # 填充新字段
                            target_tote_ids=physical_carried_totes,
                            hit_tote_ids=hit_totes,
                            noise_tote_ids=noise_totes,
                            sort_layer_range=layer_range
                        )

                        new_task.estimated_service_time = task_svc_time
                        p_tasks.append(new_task)

                        # 更新反馈数据
                        selected_totes_for_feedback.extend(physical_carried_totes)
                        self._global_task_id += 1

        elif m.status == GRB.INFEASIBLE:
            # 处理无解 (通常是 Robot Capacity 不足带不回必须的箱子)
            print(f"  [SP3] Task {task.id} INFEASIBLE. Capacity constraints likely violated.")
            # 可以在这里做一些特定的 Log 或者返回空列表让上层处理
            m.computeIIS()  # 调试用：计算不可行集
            # m.write("model.ilp")

        return p_tasks, selected_totes_for_feedback, total_sc_cost


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
            self.t_move = OFSConfig.PLACE_TOTE_TIME
            self.robot_capacity = OFSConfig.ROBOT_CAPACITY
            self.alpha = 1.0

            # 全局ID生成器，防止多次调用solve重复
            self._global_task_id = 0

        def solve(self,
                  sub_tasks: List[SubTask],
                  beta_congestion: float = 1.0
                  ) -> Tuple[List[Task], Dict[int, List[int]], Dict[int, float]]:

            print(f"  >>> [SP3 Heuristic] Generating Initial Solution (Beta={beta_congestion:.2f})...")

            physical_tasks: List[Task] = []
            final_tote_selection = defaultdict(list)
            final_sorting_costs = defaultdict(float)

            # 遍历每个子任务独立求解
            for task in sub_tasks:
                if task.assigned_station_id == -1: continue

                # 1. 贪婪选箱：决定去哪些堆垛拿哪些箱子
                # 返回: {stack_idx: [tote_obj, ...]} (这些是 Hit Totes)
                stack_plan = self._greedy_tote_selection(task)

                # 2. 模式决策：对每个涉及的堆垛决定 Flip 还是 Sort
                for stack_idx, needed_totes in stack_plan.items():
                    stack = self.problem.point_to_stack[stack_idx]

                    # 决策模式并计算成本
                    # Returns: (Mode, RangeTuple, SC_Cost, Svc_Time)
                    mode, layer_range, cost_sc, svc_time = self._decide_operation_mode(
                        stack, needed_totes, beta_congestion
                    )

                    # --- 3. 解析结果并构建 Task ---
                    physical_totes_ids = []
                    hit_totes_ids = []
                    noise_totes_ids = []

                    needed_tote_ids_set = set(t.id for t in needed_totes)

                    if mode == 'SORT':
                        low, high = layer_range
                        # 找出堆垛中位于 [low, high] 区间内的所有物理箱子
                        # 注意：这里假设 stack.totes 是有序的或者包含所有箱子信息
                        for tote in stack.totes:
                            if low <= tote.layer <= high:
                                physical_totes_ids.append(tote.id)
                                if tote.id in needed_tote_ids_set:
                                    hit_totes_ids.append(tote.id)
                                else:
                                    noise_totes_ids.append(tote.id)
                    else:
                        # FLIP 模式：物理 = 命中
                        physical_totes_ids = [t.id for t in needed_totes]
                        hit_totes_ids = [t.id for t in needed_totes]
                        noise_totes_ids = []
                        layer_range = None

                    # 创建物理任务
                    new_task = Task(
                        task_id=self._global_task_id,
                        sub_task_id=task.id,  # 修正为 int, 这里 sub_tasks 循环的是单个 SubTask 对象
                        target_stack_id=stack_idx,
                        target_station_id=task.assigned_station_id,
                        operation_mode=mode,

                        # 填充详细字段
                        target_tote_ids=physical_totes_ids,
                        hit_tote_ids=hit_totes_ids,
                        noise_tote_ids=noise_totes_ids,
                        sort_layer_range=layer_range
                    )
                    new_task.estimated_service_time = svc_time
                    physical_tasks.append(new_task)
                    self._global_task_id += 1

                    # 记录反馈数据
                    # 记录物理带走的以便计算库存扣减。
                    final_tote_selection[task.id].extend(physical_totes_ids)
                    if cost_sc > 0:
                        final_sorting_costs[task.id] += cost_sc

            return physical_tasks, final_tote_selection, final_sorting_costs

        def _greedy_tote_selection(self, task: SubTask) -> Dict[int, List[Tote]]:
            """
            贪婪策略：寻找最少的堆垛覆盖所有 SKU
            (保持原有逻辑不变)
            """
            pending_skus = set(sku.id for sku in task.sku_list)
            sku_availability = defaultdict(list)

            for sku_id in pending_skus:
                sku_obj = self.problem.id_to_sku[sku_id]
                for tote_id in sku_obj.storeToteList:
                    tote = self.problem.id_to_tote.get(tote_id)
                    # 确保 tote 存在且在库位上
                    if tote and tote.store_point:
                        sku_availability[sku_id].append(tote)

            selected_stacks_map = defaultdict(list)

            while pending_skus:
                stack_score = defaultdict(int)
                stack_candidate_totes = defaultdict(list)

                for sku_id in pending_skus:
                    candidates = sku_availability[sku_id]
                    for tote in candidates:
                        s_idx = tote.store_point.idx
                        score_bonus = 100 if s_idx in selected_stacks_map else 1

                        # 只有当这个箱子还没被选入该 Stack 的候选列表时才添加
                        if tote not in stack_candidate_totes[s_idx]:
                            stack_score[s_idx] += score_bonus
                            stack_candidate_totes[s_idx].append(tote)

                if not stack_score:
                    print(f"Error: Cannot find totes for remaining SKUs: {pending_skus}")
                    break

                best_stack_idx = max(stack_score, key=stack_score.get)
                chosen_totes = stack_candidate_totes[best_stack_idx]

                for t in chosen_totes:
                    if t not in selected_stacks_map[best_stack_idx]:
                        selected_stacks_map[best_stack_idx].append(t)

                    # 移除已覆盖的 SKU
                    for s_in_tote in t.skus_list:
                        if s_in_tote.id in pending_skus:
                            pending_skus.remove(s_in_tote.id)

            return selected_stacks_map

        def _decide_operation_mode(self,
                                   stack: Stack,
                                   target_totes: List[Tote],
                                   beta: float) -> Tuple[str, Optional[Tuple[int, int]], float, float]:
            """
            成本对比与模式决策
            Returns: (Mode, (Layer_Low, Layer_High), SC_cost, Service_Time)
            """
            # --- 基础数据 ---
            # 目标箱子的层级索引 (0=Bottom)
            target_indices = [t.layer for t in target_totes]
            top_layer_idx = stack.current_height - 1

            # 1. === 计算 FLIP 成本 ===
            cost_flip = 0.0
            for idx in target_indices:
                # 如果不是顶层，需要 Lift
                is_deep = 1 if idx < top_layer_idx else 0
                cost_flip += self.alpha * (self.t_shift + is_deep * self.t_lift)

            # Flip 的候选结果包
            res_flip = ('FLIP', None, 0.0, cost_flip)

            # 2. === 计算 SORT 成本 (比较两种策略) ===
            deepest_index = min(target_indices)
            highest_needed_index = max(target_indices)

            candidates = []  # 存储 (Total_Cost, Mode, Range, SC, RC)

            # 策略 A: Strict Range (仅搬运到最高的需求层)
            # 范围: [deepest, highest_needed]
            # 如果 highest_needed < top_layer，则会有 Lift 成本
            range_a = (deepest_index, highest_needed_index)
            size_a = highest_needed_index - deepest_index + 1

            if size_a <= self.robot_capacity:
                # Robot Cost: 如果不包含顶层，需要 Lift
                has_lift_a = (highest_needed_index < top_layer_idx)
                rc_a = self.alpha * (self.t_shift + self.t_lift * (1 if has_lift_a else 0))

                # Station Cost
                noise_a = size_a - len(target_totes)
                sc_a = beta * self.t_move * noise_a

                candidates.append((rc_a + sc_a, 'SORT', range_a, sc_a, rc_a))

            # 策略 B: Top Included (直接搬运到物理顶层)
            # 范围: [deepest, top_layer]
            # 前提: 必须最高需求层本身不是顶层，否则策略B和A一样
            if highest_needed_index < top_layer_idx:
                range_b = (deepest_index, top_layer_idx)
                size_b = top_layer_idx - deepest_index + 1

                if size_b <= self.robot_capacity:
                    # Robot Cost: 包含顶层，无 Lift
                    rc_b = self.alpha * (self.t_shift + 0)  # No Lift

                    # Station Cost: 噪音变大 (需计算这个区间内所有的箱子数 - 有用的)
                    # 简化计算：区间大小 - 有用的 = 噪音
                    noise_b = size_b - len(target_totes)
                    sc_b = beta * self.t_move * noise_b

                    candidates.append((rc_b + sc_b, 'SORT', range_b, sc_b, rc_b))

            # 3. === 最终决策 ===
            # 如果 Sort 没有可行解 (都超重)，直接选 Flip
            if not candidates:
                return res_flip

            # 选出最好的 Sort 策略
            best_sort = min(candidates, key=lambda x: x[0])
            best_sort_total_cost = best_sort[0]

            # 对比 Flip 和 Best Sort
            if best_sort_total_cost < cost_flip:
                # Unpack sort result: (Total, Mode, Range, SC, RC)
                return best_sort[1], best_sort[2], best_sort[3], best_sort[4]
            else:
                return res_flip