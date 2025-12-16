import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from entity.SKUs import SKUs
from entity.point import Point
from entity.subTask import SubTask
from entity.task import Task
from entity.tote import Tote
from entity.stack import Stack
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig
import copy


class SP3_Bin_Hitter:
    """
    SP3 子问题求解器：料箱命中

    基于 Gurobi MIP 求解。
    使用虚拟 layer 管理堆垛状态
    """

    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem = problem_dto

        # --- 成本参数 ---
        self.t_shift = OFSConfig.PACKING_TIME
        self.t_lift = OFSConfig.LIFTING_TIME
        self.t_move = OFSConfig.MOVE_EXTRA_TOTE_TIME
        self.robot_capacity = OFSConfig.ROBOT_CAPACITY
        self.alpha = 1.0
        self.w_routing = 0.5
        self.stack_allocation: Dict[int, int] = {}
        self.BigM = 10000

        # ✅ 虚拟层级管理
        self.stack_snapshots: Dict[int, List[Tote]] = {}  # {stack_id: [current_totes]}
        self.layer_mapping: Dict[int, int] = {}  # {tote_id: virtual_layer}

    def solve(self,
              sub_tasks: List[SubTask],
              beta_congestion: float = 1.0,
              sp4_routing_costs: Dict[int, float] = None
              ) -> Tuple[List[Task], Dict[int, List[int]], Dict[int, float]]:

        print(f"  >>> [SP3] Solving with Virtual Layer (Beta={beta_congestion:.2f})...")
        self._initialize_stack_snapshots()
        self.stack_allocation = {}
        physical_tasks: List[Task] = []
        final_tote_selection = defaultdict(list)
        final_sorting_costs = defaultdict(float)
        self._global_task_id = 0

        sorted_tasks = sorted(sub_tasks, key=lambda t: (
            len(t.unique_sku_list),
            t.station_sequence_rank if t.station_sequence_rank >= 0 else 999
        ))

        for task in sorted_tasks:
            task.reset_execution_details()
            if task.assigned_station_id == -1:
                continue

            p_tasks, totes, cost = self._solve_single_subtask_mip(
                task, beta_congestion, sp4_routing_costs
            )

            # 先占用堆垛
            for pt in p_tasks:
                self.stack_allocation[pt.target_stack_id] = task.id

            # 再模拟出库（更新虚拟层级）
            for pt in p_tasks:
                self._apply_stack_modification(pt)

            physical_tasks.extend(p_tasks)
            final_tote_selection[task.id] = totes
            final_sorting_costs[task.id] = cost

        return physical_tasks, final_tote_selection, final_sorting_costs

    def _initialize_stack_snapshots(self):
        """
        初始化堆垛快照和虚拟层级映射
        关键：只拷贝列表结构，Tote 对象保持引用
        """
        for stack in self.problem.stack_list:
            # 浅拷贝列表（Tote 对象共享）
            self.stack_snapshots[stack.stack_id] = list(stack.totes)

            # ✅ 初始化虚拟层级（基于原始位置）
            for i, tote in enumerate(stack.totes):
                self.layer_mapping[tote.id] = i

    def _get_virtual_layer(self, tote_id: int) -> int:
        """获取 Tote 的虚拟层级"""
        return self.layer_mapping.get(tote_id, -1)

    def _apply_stack_modification(self, task: Task):
        """
        模拟出库并更新虚拟层级
        关键：不修改 tote.layer 和 stack.totes
        """
        stack_id = task.target_stack_id

        if stack_id not in self.stack_snapshots:
            print(f"  [Warning] Stack {stack_id} not in snapshots.")
            return

        current_totes = self.stack_snapshots[stack_id]

        # 1. 根据操作模式计算剩余 Tote
        if task.operation_mode == 'FLIP':
            removed_ids = set(task.target_tote_ids)
            remaining_totes = [t for t in current_totes if t.id not in removed_ids]

            print(f"  [SP3] Stack {stack_id} FLIP: Removed {len(removed_ids)} totes, {len(remaining_totes)} remain.")

        elif task.operation_mode == 'SORT':
            if task.sort_layer_range is None:
                return

            low, high = task.sort_layer_range
            # ✅ 使用虚拟层级判断
            remaining_totes = [
                t for t in current_totes
                if not (low <= self._get_virtual_layer(t.id) <= high)
            ]

            print(
                f"  [SP3] Stack {stack_id} SORT [{low}, {high}]: Removed {len(current_totes) - len(remaining_totes)} totes.")

        else:
            return

        # 2. ✅ 更新虚拟层级（不修改 Tote 对象）
        for i, tote in enumerate(remaining_totes):
            self.layer_mapping[tote.id] = i  # 只更新映射表

        # 3. 更新快照列表
        self.stack_snapshots[stack_id] = remaining_totes

    def _solve_single_subtask_mip(self,
                                  task: SubTask,
                                  beta: float,
                                  routing_costs: Dict[int, float]) -> Tuple[List[Task], List[int], float]:

        # 1. 需求统计
        demand = defaultdict(int)
        for sku in task.sku_list:
            demand[sku.id] += 1

        # 2. 候选堆垛与料箱
        candidate_stack_indices = set()

        for sku in task.unique_sku_list:
            for tote_id in sku.storeToteList:
                tote = self.problem.id_to_tote.get(tote_id)
                if not (tote and tote.store_point):
                    continue

                stack_idx = tote.store_point.idx

                if stack_idx not in self.stack_snapshots:
                    continue

                current_totes_in_stack = self.stack_snapshots[stack_idx]
                tote_still_exists = any(t.id == tote_id for t in current_totes_in_stack)

                if not tote_still_exists:
                    continue

                candidate_stack_indices.add(stack_idx)

        U = sorted(list(candidate_stack_indices))  # 原本是 set
        if not U:
            return [], [], 0.0

        stack_bin_map = {}
        for u in U:
            if u in self.stack_snapshots:
                stack_bin_map[u] = list(self.stack_snapshots[u])
            else:
                stack_bin_map[u] = []

        # --- B. 构建 MIP ---
        m = gp.Model(f"SP3_Task_{task.id}")
        m.Params.OutputFlag = 0

        m_flip = m.addVars(U, vtype=GRB.BINARY, name="m_flip")
        m_sort = m.addVars(U, vtype=GRB.BINARY, name="m_sort")

        all_bins = [b for u in U for b in stack_bin_map[u]]
        bin_ids = [b.id for b in all_bins]
        u_use = m.addVars(bin_ids, vtype=GRB.BINARY, name="u_use")
        needed_sku_ids = set(demand.keys())

        for u in U:
            for b in stack_bin_map[u]:
                has_needed_sku = any(s_id in needed_sku_ids for s_id in b.sku_quantity_map)
                if not has_needed_sku:
                    m.addConstr(u_use[b.id] == 0, name=f"Block_Useless_{b.id}")

        idx_high = m.addVars(U, vtype=GRB.INTEGER, name="idx_h")
        idx_low = m.addVars(U, vtype=GRB.INTEGER, name="idx_l")
        is_top_inc = m.addVars(U, vtype=GRB.BINARY, name="is_top_inc")
        cost_rc = m.addVars(U, vtype=GRB.CONTINUOUS, name="c_rc")
        cost_sc = m.addVars(U, vtype=GRB.CONTINUOUS, name="c_sc")
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
            current_totes = self.stack_snapshots.get(u, [])
            max_h = len(current_totes)
            top_layer_idx = max_h - 1

            bin_sum = gp.quicksum(u_use[b.id] for b in bins)
            m.addConstr(m_flip[u] + m_sort[u] <= 1)
            m.addConstr(bin_sum <= self.BigM * (m_flip[u] + m_sort[u]))
            m.addConstr(m_flip[u] + m_sort[u] <= bin_sum)
            m.addConstr(idx_high[u] >= top_layer_idx * is_top_inc[u] - self.BigM * (1 - m_sort[u]))

            # ✅ 使用虚拟层级构建约束
            for b in bins:
                virtual_layer = self._get_virtual_layer(b.id)
                m.addConstr(idx_high[u] >= virtual_layer * u_use[b.id])
                m.addConstr(idx_low[u] <= virtual_layer * u_use[b.id] + max_h * (1 - u_use[b.id]))

            load_u = m.addVar(vtype=GRB.CONTINUOUS, name=f"load_{u}")
            m.addConstr(load_u >= bin_sum - self.BigM * (1 - m_flip[u]))
            range_size = idx_high[u] - idx_low[u] + 1
            m.addConstr(load_u >= range_size - self.BigM * (1 - m_sort[u]))
            total_load += load_u

            # 成本计算
            flip_val = gp.LinExpr()
            for b in bins:
                virtual_layer = self._get_virtual_layer(b.id)
                is_deep = 1 if virtual_layer < top_layer_idx else 0
                c = self.alpha * (self.t_shift + is_deep * self.t_lift)
                flip_val += u_use[b.id] * c
            m.addConstr(cost_fc[u] >= flip_val - self.BigM * (1 - m_flip[u]))

            rc_expr = self.alpha * (self.t_shift + self.t_lift * (1 - is_top_inc[u]))
            m.addConstr(cost_rc[u] >= rc_expr - self.BigM * (1 - m_sort[u]))

            noise_expr = range_size - bin_sum
            sc_raw = beta * self.t_move * noise_expr
            m.addConstr(cost_sc[u] >= sc_raw - self.BigM * (1 - m_sort[u]))

            m.addConstr(cost_fc[u] >= 0)
            m.addConstr(cost_rc[u] >= 0)
            m.addConstr(cost_sc[u] >= 0)

        m.addConstr(total_load <= self.robot_capacity, name="Capacity_Limit")

        # --- D. 目标函数 ---
        obj = gp.LinExpr()
        obj += gp.quicksum(cost_rc[u] + cost_sc[u] + cost_fc[u] for u in U)
        penalty_per_tote = beta * self.t_move + 0.5
        obj += penalty_per_tote * gp.quicksum(u_use)
        if routing_costs:
            for u in U:
                r_cost = routing_costs.get(u, 0.0)
                obj += self.w_routing * r_cost * (m_flip[u] + m_sort[u])

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        # --- E. 结果解析 ---
        p_tasks = []
        selected_totes_for_feedback = []
        total_sc_cost = 0.0

        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            for u in U:
                if m_flip[u].X > 0.5 or m_sort[u].X > 0.5:
                    is_sort_mode = (m_sort[u].X > 0.5)
                    mode = 'SORT' if is_sort_mode else 'FLIP'

                    physical_carried_totes = []
                    hit_totes = []
                    noise_totes = []
                    layer_range = None
                    robot_time_val = 0.0
                    station_time_val = 0.0

                    all_totes_in_stack = stack_bin_map[u]

                    if is_sort_mode:
                        val_low = int(round(idx_low[u].X))
                        val_high = int(round(idx_high[u].X))
                        layer_range = (val_low, val_high)
                        station_time_val = cost_sc[u].X / beta
                        robot_time_val = cost_rc[u].X / self.alpha

                        # ✅ 使用虚拟层级判断
                        for tote in all_totes_in_stack:
                            virtual_layer = self._get_virtual_layer(tote.id)
                            if val_low <= virtual_layer <= val_high:
                                physical_carried_totes.append(tote.id)
                                if u_use[tote.id].X > 0.5:
                                    hit_totes.append(tote.id)
                                else:
                                    noise_totes.append(tote.id)

                        total_sc_cost += beta * self.t_move * len(noise_totes)

                    else:
                        for tote in all_totes_in_stack:
                            if u_use[tote.id].X > 0.5:
                                physical_carried_totes.append(tote.id)
                                hit_totes.append(tote.id)

                        noise_totes = []
                        layer_range = None
                        station_time_val = cost_sc[u].X / beta
                        robot_time_val = cost_rc[u].X / self.alpha

                    if physical_carried_totes:
                        new_task = Task(
                            task_id=self._global_task_id,
                            sub_task_id=task.id,
                            target_stack_id=u,
                            target_station_id=task.assigned_station_id,
                            operation_mode=mode,
                            robot_service_time=robot_time_val,
                            station_service_time=station_time_val,
                            target_tote_ids=physical_carried_totes,
                            hit_tote_ids=hit_totes,
                            noise_tote_ids=noise_totes,
                            sort_layer_range=layer_range
                        )

                        target_stack_obj = self.problem.point_to_stack.get(u)
                        if target_stack_obj:
                            task.add_execution_detail(new_task, target_stack_obj)

                        p_tasks.append(new_task)
                        selected_totes_for_feedback.extend(physical_carried_totes)
                        self._global_task_id += 1

        elif m.status == GRB.INFEASIBLE:
            print(f"  [SP3] Task {task.id} INFEASIBLE.")

        return p_tasks, selected_totes_for_feedback, total_sc_cost

    class SP3_Heuristic_Solver:
        """SP3 启发式求解器（使用虚拟层级）"""

        def __init__(self, problem_dto: OFSProblemDTO):
            self.problem = problem_dto
            self.t_shift = OFSConfig.PACKING_TIME
            self.t_lift = OFSConfig.LIFTING_TIME
            self.t_move = OFSConfig.PLACE_TOTE_TIME
            self.robot_capacity = OFSConfig.ROBOT_CAPACITY
            self.alpha = 2.0
            self.stack_allocation: Dict[int, int] = {}
            self._global_task_id = 0

            # ✅ 虚拟层级管理
            self.stack_snapshots: Dict[int, List[Tote]] = {}
            self.layer_mapping: Dict[int, int] = {}

        def solve(self,
                  sub_tasks: List[SubTask],
                  beta_congestion: float = 1.0
                  ) -> Tuple[List[Task], Dict[int, List[int]], Dict[int, float]]:

            print(f"  >>> [SP3 Heuristic] Using Virtual Layer (Beta={beta_congestion:.2f})...")
            self._initialize_stack_snapshots()
            physical_tasks: List[Task] = []
            final_tote_selection = defaultdict(list)
            final_sorting_costs = defaultdict(float)
            self.stack_allocation = {}

            sorted_tasks = sorted(sub_tasks, key=lambda t: (
                len(t.unique_sku_list),
                t.station_sequence_rank if t.station_sequence_rank >= 0 else 999
            ))

            for task in sorted_tasks:
                task.reset_execution_details()
                if task.assigned_station_id == -1:
                    continue

                stack_plan = self._greedy_tote_selection_v2(task)

                # accumulated_load = 0
                current_batch_tasks = []

                

                for stack_idx, needed_totes in stack_plan.items():
                    stack = self.problem.point_to_stack[stack_idx]
                    pending_totes = list(needed_totes)

                    while pending_totes:
                        pending_totes.sort(key=lambda t: self._get_virtual_layer(t.id))
                        if len(pending_totes) <= self.robot_capacity:
                            current_batch = pending_totes
                        else:
                            # 计算最大连续层级数
                            layers = [self._get_virtual_layer(t.id) for t in pending_totes]
                            max_len = 0
                            max_start_idx = 0
                            curr_len = 1
                            curr_start = 0

                            # 遍历寻找最长连续序列
                            for i in range(1, len(layers)):
                                if layers[i] == layers[i - 1] + 1:
                                    curr_len += 1
                                else:
                                    if curr_len > max_len:
                                        max_len = curr_len
                                        max_start_idx = curr_start
                                    curr_len = 1
                                    curr_start = i
                            # 检查最后一段
                            if curr_len > max_len:
                                max_len = curr_len
                                max_start_idx = curr_start

                            # --- 策略分支 ---
                            if max_len > self.robot_capacity:
                                # 策略 A: 最大连续层级 > 容量
                                # -> 取该连续段的最低层开始，长度为 Capacity 的一段
                                batch_indices = set(range(max_start_idx, max_start_idx + self.robot_capacity))
                                current_batch = [pending_totes[i] for i in range(len(pending_totes)) if
                                                     i in batch_indices]
                            elif max_len < 2:
                                # 策略 B: 离散分布 (最大连续数 < 2)
                                # -> 均分 (取前一半，但不超过容量)
                                take_count = min(len(pending_totes) // 2, self.robot_capacity)
                                take_count = max(take_count, 1)  # 至少取1个防止死循环
                                current_batch = pending_totes[:take_count]

                            else:
                                # 策略 C: 存在连续层级 (>=2)
                                # -> 取最大连续段 (传入连续层级，剩余的留给下一轮)
                                end_idx = max_start_idx + max_len
                                batch_indices = set(range(max_start_idx, end_idx))
                                current_batch = [pending_totes[i] for i in range(len(pending_totes)) if
                                                 i in batch_indices]


                        mode, layer_range, sc_time, rc_time = self._decide_operation_mode(
                            stack, current_batch, beta_congestion
                        )

                        # 计算物理负载
                        physical_totes_ids = []
                        hit_totes_ids = []
                        noise_totes_ids = []
                        needed_tote_ids_set = set(t.id for t in needed_totes)

                        if mode == 'SORT':
                            low, high = layer_range
                            current_snapshot = self.stack_snapshots.get(stack_idx, [])

                            for tote in current_snapshot:
                                virtual_layer = self._get_virtual_layer(tote.id)
                                if low <= virtual_layer <= high:
                                    physical_totes_ids.append(tote.id)

                        else:
                            physical_totes_ids = [t.id for t in current_batch]

                            layer_range = None

                        # 5. 区分 Hit 和 Noise (基于该 Stack 的所有需求)
                        # 注意：这里使用 all_needed_ids 而不是 current_batch，
                        # 因为如果操作范围意外覆盖了 pending_totes 中其他的箱子，也应算作 Hit
                        hit_totes_ids = []
                        noise_totes_ids = []
                        needed_tote_ids_set = set(t.id for t in needed_totes)

                        for tid in physical_totes_ids:
                            if tid in needed_tote_ids_set:
                                hit_totes_ids.append(tid)
                            else:
                                noise_totes_ids.append(tid)
                        # 6. 生成 Task
                        new_task = Task(
                            task_id=self._global_task_id,
                            sub_task_id=task.id,
                            target_stack_id=stack_idx,
                            target_station_id=task.assigned_station_id,
                            operation_mode=mode,
                            target_tote_ids=physical_totes_ids,
                            hit_tote_ids=hit_totes_ids,
                            robot_service_time=rc_time,
                            station_service_time=sc_time,
                            noise_tote_ids=noise_totes_ids,
                            sort_layer_range=layer_range
                        )

                        task.add_execution_detail(new_task, stack)
                        physical_tasks.append(new_task)
                        current_batch_tasks.append(new_task)

                        self._global_task_id += 1
                        self.stack_allocation[stack_idx] = task.id
                        # 7. 模拟出库 (更新虚拟层级)
                        self._apply_stack_modification(new_task)

                        final_tote_selection[task.id].extend(physical_totes_ids)
                        if sc_time > 0:
                            final_sorting_costs[task.id] += sc_time

                        # 8. ✅ 关键：从 pending_totes 中移除已经被带走的箱子 (Hits)
                        # 这样下一轮循环时，pending_totes 就是剩余未命中的箱子
                        carried_hits_set = set(hit_totes_ids)
                        pending_totes = [t for t in pending_totes if t.id not in carried_hits_set]

                        # 安全检查：如果生成的任务没有带走任何东西（异常情况），强制退出防止死循环
                        if not physical_totes_ids:
                            break
            return physical_tasks, final_tote_selection, final_sorting_costs

        def _initialize_stack_snapshots(self):
            """初始化堆垛快照和虚拟层级"""
            for stack in self.problem.stack_list:
                self.stack_snapshots[stack.stack_id] = list(stack.totes)
                for i, tote in enumerate(stack.totes):
                    self.layer_mapping[tote.id] = i

        def _get_virtual_layer(self, tote_id: int) -> int:
            """获取 Tote 的虚拟层级"""
            return self.layer_mapping.get(tote_id, -1)

        def _apply_stack_modification(self, task: Task):
            """模拟出库并更新虚拟层级"""
            stack_id = task.target_stack_id

            if stack_id not in self.stack_snapshots:
                return

            current_totes = self.stack_snapshots[stack_id]

            if task.operation_mode == 'FLIP':
                removed_ids = set(task.target_tote_ids)
                remaining_totes = [t for t in current_totes if t.id not in removed_ids]
            elif task.operation_mode == 'SORT':
                if task.sort_layer_range is None:
                    return
                low, high = task.sort_layer_range
                remaining_totes = [
                    t for t in current_totes
                    if not (low <= self._get_virtual_layer(t.id) <= high)
                ]
            else:
                return

            # ✅ 只更新虚拟层级
            for i, tote in enumerate(remaining_totes):
                self.layer_mapping[tote.id] = i

            self.stack_snapshots[stack_id] = remaining_totes

        def _greedy_tote_selection_v2(self, task: SubTask) -> Dict[int, List[Tote]]:
            """
            改进的贪婪选箱策略（确定性版本）
            """
            # ✅ 修复 1: 使用排序后的 SKU 集合
            pending_skus = sorted([sku.id for sku in task.sku_list])  # 转为有序列表
            pending_skus_set = set(pending_skus)  # 用于快速查找

            selected_stacks_map = defaultdict(list)

            while pending_skus_set:
                sku_availability = defaultdict(list)

                # ✅ 修复 2: 按固定顺序遍历 SKU
                for sku_id in sorted(pending_skus_set):  # 排序确保顺序
                    sku_obj = self.problem.id_to_sku[sku_id]
                    
                    # ✅ 修复 3: 排序 tote 列表
                    for tote_id in sorted(sku_obj.storeToteList):
                        tote = self.problem.id_to_tote.get(tote_id)
                        if not (tote and tote.store_point):
                            continue

                        stack_idx = tote.store_point.idx

                        # 检查 1: Stack 是否已被其他 SubTask 占用
                        if stack_idx in self.stack_allocation:
                            allocated_to = self.stack_allocation[stack_idx]
                            if allocated_to != task.id:
                                continue

                        # 检查 2: Tote 是否还在虚拟堆垛中
                        if stack_idx not in self.stack_snapshots:
                            continue

                        current_totes_in_stack = self.stack_snapshots[stack_idx]
                        if not any(t.id == tote_id for t in current_totes_in_stack):
                            continue

                        sku_availability[sku_id].append(tote)

                if not sku_availability:
                    print(f"  ⚠️ [Heuristic] Cannot find totes for remaining SKUs: {pending_skus_set}")
                    break

                stack_score = {}
                stack_candidate_totes = defaultdict(list)

                # ✅ 修复 4: 按固定顺序遍历 SKU
                for sku_id in sorted(pending_skus_set):
                    candidates = sku_availability.get(sku_id, [])
                    
                    # ✅ 修复 5: 料箱按 ID 排序
                    candidates.sort(key=lambda t: t.id)
                    
                    for tote in candidates:
                        s_idx = tote.store_point.idx

                        # Bundle 效应
                        bundle_bonus = 100 if s_idx in selected_stacks_map else 1

                        # 层级惩罚
                        virtual_layer = self._get_virtual_layer(tote.id)
                        current_snapshot = self.stack_snapshots.get(s_idx, [])
                        stack_height = len(current_snapshot)

                        if stack_height > 1:
                            normalized_layer = virtual_layer / (stack_height - 1)
                        else:
                            normalized_layer = 0

                        layer_bonus = 10 * (1 - normalized_layer)

                        total_score = bundle_bonus + layer_bonus
                        if s_idx not in stack_score:
                            stack_score[s_idx] = 0
                        stack_score[s_idx] += total_score

                        if tote not in stack_candidate_totes[s_idx]:
                            stack_candidate_totes[s_idx].append(tote)

                if not stack_score:
                    print(f"  ⚠️ [Heuristic] No valid stack candidates")
                    break

                # ✅ 修复 6: 分数相同时按 stack_id 排序
                best_stack_idx = max(
                    stack_score.items(), 
                    key=lambda x: (x[1], -x[0])  # 分数降序，ID 升序
                )[0]
                
                chosen_totes = stack_candidate_totes[best_stack_idx]

                # ✅ 修复 7: 完整的排序逻辑
                chosen_totes.sort(key=lambda t: (
                    -len(set(s.id for s in t.skus_list) & pending_skus_set),  # 匹配 SKU 数降序
                    self._get_virtual_layer(t.id),                            # 层级升序
                    t.id                                                       # Tote ID 升序
                ))

                for t in chosen_totes:
                    tote_sku_ids = set(s.id for s in t.skus_list)
                    contributes = not pending_skus_set.isdisjoint(tote_sku_ids)

                    if contributes:
                        if t not in selected_stacks_map[best_stack_idx]:
                            selected_stacks_map[best_stack_idx].append(t)

                        for s_in_tote in t.skus_list:
                            if s_in_tote.id in pending_skus_set:
                                pending_skus_set.remove(s_in_tote.id)

            return selected_stacks_map
        def _decide_operation_mode(self,
                                   stack: Stack,
                                   target_totes: List[Tote],
                                   beta: float) -> Tuple[str, Optional[Tuple[int, int]], float, float]:
            """模式决策（使用虚拟层级）"""
            stack_id = stack.stack_id

            current_snapshot = self.stack_snapshots.get(stack_id, [])
            current_height = len(current_snapshot)

            if current_height == 0:
                return ('FLIP', None, 0.0, 0.0)

            # ✅ 使用虚拟层级
            target_indices = [self._get_virtual_layer(t.id) for t in target_totes]
            top_layer_idx = current_height - 1

            # FLIP 成本
            time_flip = 0.0
            for idx in target_indices:
                is_deep = 1 if idx < top_layer_idx else 0
                time_flip += (self.t_shift + is_deep * self.t_lift)
            cost_flip = self.alpha * time_flip

            res_flip = ('FLIP', None, 0.0, time_flip)

            # SORT 成本
            deepest_index = min(target_indices)
            highest_needed_index = max(target_indices)
            candidates = []

            # 策略 A: Strict Range
            range_a = (deepest_index, highest_needed_index)
            size_a = highest_needed_index - deepest_index + 1

            if size_a <= self.robot_capacity:
                has_lift_a = (highest_needed_index < top_layer_idx)
                time_rc_a = (self.t_shift + self.t_lift * (1 if has_lift_a else 0))
                cost_rc_a = self.alpha * time_rc_a

                noise_a = size_a - len(target_totes)
                cost_sc_a = beta * self.t_move * noise_a

                candidates.append((cost_rc_a + cost_sc_a, 'SORT', range_a, self.t_move * noise_a, time_rc_a))

            # 策略 B: Top Included
            if highest_needed_index < top_layer_idx:
                range_b = (deepest_index, top_layer_idx)
                size_b = top_layer_idx - deepest_index + 1

                if size_b <= self.robot_capacity:
                    time_rc_b = self.t_shift
                    cost_rc_b = self.alpha * time_rc_b

                    noise_b = size_b - len(target_totes)
                    cost_sc_b = beta * self.t_move * noise_b

                    candidates.append((cost_rc_b + cost_sc_b, 'SORT', range_b, self.t_move * noise_b, time_rc_b))


            if not candidates:
                return res_flip

            best_sort = min(candidates, key=lambda x: x[0])
            best_sort_total_cost = best_sort[0]

            if best_sort_total_cost < cost_flip:
                return best_sort[1], best_sort[2], best_sort[3], best_sort[4]
            else:
                return res_flip


if __name__ == "__main__":

    # ==========================================
    # 1. Mock Data Construction (构建测试数据)
    # ==========================================

    def create_mock_problem():
        problem = OFSProblemDTO()

        # ✅ 先创建空的 SKU
        sku1 = SKUs(sku_id=1, storeToteList=[])
        sku2 = SKUs(sku_id=2, storeToteList=[])
        sku3 = SKUs(sku_id=3, storeToteList=[])
        problem.id_to_sku = {1: sku1, 2: sku2, 3: sku3}

        # 创建 Stack
        stack_point = Point(0, 10, 0)
        stack = Stack(stack_id=0, store_point=stack_point, max_height=8)
        problem.point_to_stack = {0: stack}
        problem.stack_list = [stack]  # ✅ 必须添加
        problem.id_to_tote = {}

        # 创建 Tote 并手动建立双向关联
        t0 = Tote(100)
        t0.skus_list = [sku1]
        t0.sku_quantity_map = {1: 1}
        sku1.storeToteList.append(100)  # ✅ 关键

        t1 = Tote(101)
        t1.skus_list = []
        t1.sku_quantity_map = {}

        t2 = Tote(102)
        t2.skus_list = [sku2]
        t2.sku_quantity_map = {2: 1}
        sku2.storeToteList.append(102)  # ✅ 关键

        t3 = Tote(103)
        t3.skus_list = [sku3]
        t3.sku_quantity_map = {3: 1}
        sku3.storeToteList.append(103)  # ✅ 关键

        t4 = Tote(104)
        t4.skus_list = []
        t4.sku_quantity_map = {}

        totes = [t0, t1, t2, t3, t4]
        for t in totes:
            stack.add_tote(t)
            problem.id_to_tote[t.id] = t

        return problem, [sku1, sku2, sku3]


    def run_test():
        # 1. 准备环境
        problem, target_skus = create_mock_problem()

        # 修改配置以符合测试预期
        OFSConfig.ROBOT_CAPACITY = 5
        OFSConfig.PACKING_TIME = 10.0  # t_shift
        OFSConfig.LIFTING_TIME = 20.0  # t_lift (挖掘成本高)
        OFSConfig.PLACE_TOTE_TIME = 5.0  # t_move (理货成本低)

        # 创建 SubTask
        # 由于 SubTask 是 dataclass 且 unique_sku_list 是 init=False
        # 我们需要先创建对象，然后手动触发 post_init 或手动赋值
        from entity.order import Order
        dummy_order = Order(order_id=999)

        task = SubTask(id=1, parent_order=dummy_order, sku_list=target_skus)
        # 手动触发 post_init 逻辑 (如果 dataclass 自动生成了 __init__，它会自动调用 __post_init__)
        # 但为了保险起见，或者如果是在非标准环境下，我们可以手动确保字段正确
        # task.__post_init__() # 通常不需要手动调用，除非被覆盖

        task.assigned_station_id = 1  # 必须分配站点
        sub_tasks = [task]

        print("=== Test Scenario ===")
        print(f"Stack Height: 5")
        print(f"Needed Layers: 0, 2, 3")
        print(f"Noise Layers: 1, 4 (Top)")
        print(f"Params: Lift Cost=20, Noise Cost=5, Capacity=5")
        print("Expectation: Algorithm should choose SORT [0, 4] (Include Top) because Lift(20) > Noise(5).")
        print("=====================\n")

        # --- Run MIP Solver ---
        print("--- Running MIP Solver ---")
        mip_solver = SP3_Bin_Hitter(problem)

        # 强制修改内部参数以防 Config 不生效
        mip_solver.t_lift = 10.0
        mip_solver.t_move = 10
        mip_solver.robot_capacity = 5

        mip_tasks, _, _ = mip_solver.solve(sub_tasks, beta_congestion=2.0)

        verify_result("MIP", mip_tasks)
        print(f'robot_servicetime : {mip_tasks[0].robot_service_time if mip_tasks else "N/A"}')
        print(f'station_servicetime : {mip_tasks[0].station_service_time if mip_tasks else "N/A"}')
        print("\n--- Running Heuristic Solver ---")
        heu_solver = mip_solver.SP3_Heuristic_Solver(problem)
        # 强制修改内部参数
        heu_solver.t_lift = 20.0
        heu_solver.t_move = 5.0
        heu_solver.robot_capacity = 5

        heu_tasks, _, _ = heu_solver.solve(sub_tasks, beta_congestion=2.0)
        print(f'robot_servicetime : {heu_tasks[0].robot_service_time if mip_tasks else "N/A"}')
        print(f'station_servicetime : {heu_tasks[0].station_service_time if mip_tasks else "N/A"}')
        verify_result("Heuristic", heu_tasks)


    # ==========================================
    # 2. Test Execution
    # ==========================================

    def verify_result(solver_name, tasks):
        if not tasks:
            print(f"[{solver_name}] FAIL: No tasks generated!")
            return

        t = tasks[0]
        print(f"[{solver_name}] Result:")
        print(f"  Mode: {t.operation_mode}")
        print(f"  Target Totes (Physical): {t.target_tote_ids}")
        print(f"  Hit Totes: {t.hit_tote_ids}")
        print(f"  Noise Totes: {t.noise_tote_ids}")
        print(f"  Sort Range: {t.sort_layer_range}")

        # 验证逻辑
        passed = True
        if t.operation_mode != 'SORT':
            print(f"  [X] FAIL: Expected SORT, got {t.operation_mode}")
            passed = False

        # 关键验证：是否包含了 ID 104 (Top Noise)
        if 104 in t.target_tote_ids:
            print(f"  [√] SUCCESS: Top noise tote (104) included. Lift cost avoided.")
        else:
            if 100 in t.target_tote_ids and 103 in t.target_tote_ids:
                print(f"  [!] INFO: Solved valid range but excluded top. Lift cost incurred.")
                print(f"      Check if Lift Cost (20) > Noise Cost (5) logic holds.")
                passed = False
            else:
                print(f"  [X] FAIL: Needed totes missing.")
                passed = False

        # 验证 Hit 列表是否正确
        expected_hits = {100, 102, 103}
        if set(t.hit_tote_ids) == expected_hits:
            print(f"  [√] SUCCESS: Hit totes match exactly.")
        else:
            print(f"  [X] FAIL: Hit totes mismatch. Expected {expected_hits}, got {set(t.hit_tote_ids)}")


    run_test()