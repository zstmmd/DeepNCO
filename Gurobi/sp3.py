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
        # --- 计算 SKU 的稀缺度 (Scarcity) ---
        # 统计每个 SKU 在当前快照中出现在多少个堆垛里
        sku_stack_count = defaultdict(int)
        for stack_id, totes in self.stack_snapshots.items():
            seen_skus_in_stack = set()
            for t in totes:
                for sku in t.skus_list:
                    seen_skus_in_stack.add(sku.id)
            for sku_id in seen_skus_in_stack:
                sku_stack_count[sku_id] += 1

        def get_task_scarcity_score(t: SubTask):
            # 分数越小越稀缺。取任务中所有 SKU 最小的堆垛覆盖数。
            # 如果某个 SKU 只在 1 个堆垛里有，这个任务优先级极高。
            min_availability = 9999
            for sku in t.unique_sku_list:
                count = sku_stack_count.get(sku.id, 0)
                if count < min_availability:
                    min_availability = count
            return (min_availability, -len(t.unique_sku_list))

        sorted_tasks = sorted(sub_tasks, key=get_task_scarcity_score)

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
                pt.station_sequence_rank = task.station_sequence_rank
            # 再模拟出库（更新虚拟层级）
            for pt in p_tasks:
                self._apply_stack_modification(pt)

            physical_tasks.extend(p_tasks)
            final_tote_selection[task.id] = totes
            final_sorting_costs[task.id] = cost
            for pt in physical_tasks:
                pt.efficiency_score = self._calculate_efficiency_score(pt)

            # 2. 执行双层排序
            # 第一关键字：SubTask Rank (升序，硬约束)
            # 第二关键字：Efficiency Score (降序，软优化)
            physical_tasks.sort(key=lambda t: (t.station_sequence_rank, -t.efficiency_score))

            # 3. 赋值 Priority (1 是最高)
            # 直接使用当前的列表索引 + 1 作为 priority
            for idx, pt in enumerate(physical_tasks):
                pt.priority = idx + 1
                # 重新编号 Task ID 以保持一致性 (可选)
                pt.task_id = idx
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

    def _calculate_efficiency_score(self, task: Task) -> float:
        """
        [新增] 计算任务的检索效率分数
        Score = 有效料箱数 / (机器人动作耗时 + 往返路程耗时)
        """
        # 1. 分子：有效料箱数量
        valid_tote_count = len(task.hit_tote_ids)
        if valid_tote_count == 0:
            valid_tote_count = 0.1

        # 2. 分母：预估耗时 = 动作时间 + 路程时间
        action_time = task.robot_service_time

        # 计算路程耗时
        travel_time = 30.0  # 默认兜底值

        # 获取 Stack 对象
        stack_obj = self.problem.point_to_stack.get(task.target_stack_id)

        # 获取 Station 对象
        station_obj = None
        for s in self.problem.station_list:
            if s.id == task.target_station_id:
                station_obj = s
                break

        if stack_obj and station_obj:
            p_stack = stack_obj.store_point
            p_station = station_obj.point

            # 计算曼哈顿距离
            dist = abs(p_stack.x - p_station.x) + abs(p_stack.y - p_station.y)

            # 获取机器人速度
            speed = getattr(OFSConfig, 'ROBOT_SPEED', 1.5)  # 防止未定义
            if speed <= 0: speed = 1.5

            # 计算往返时间 (Station -> Stack -> Station)
            travel_time = (dist / speed) * 2.0

        estimated_total_time = action_time + travel_time

        if estimated_total_time < 1.0:
            estimated_total_time = 1.0

        score = valid_tote_count / estimated_total_time

        return score

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
            # print(f"  [SP3] Stack {stack_id} FLIP: Removed {len(removed_ids)} totes, {len(remaining_totes)} remain.")

        elif task.operation_mode == 'SORT':
            if task.sort_layer_range is None:
                return

            low, high = task.sort_layer_range
            # ✅ 使用虚拟层级判断
            remaining_totes = [
                t for t in current_totes
                if not (low <= self._get_virtual_layer(t.id) <= high)
            ]
            # print(f"  [SP3] Stack {stack_id} SORT [{low}, {high}]: Removed {len(current_totes) - len(remaining_totes)} totes.")

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

        U = sorted(list(candidate_stack_indices))
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

        # 追踪该 SubTask 的剩余需求，用于计算每个 Task 的 sku_pick_count
        remaining_demand = demand.copy()

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
                        # ✅ [新增] 计算该 Task 实际命中的 SKU 数量
                        current_task_pick_count = 0
                        for t_id in hit_totes:
                            tote = self.problem.id_to_tote.get(t_id)
                            if not tote: continue

                            # 遍历该料箱内的 SKU
                            for s_id, qty in tote.sku_quantity_map.items():
                                if s_id in remaining_demand and remaining_demand[s_id] > 0:
                                    # 实际贡献量 = min(料箱库存, 剩余需求)
                                    used = min(qty, remaining_demand[s_id])
                                    current_task_pick_count += used
                                    remaining_demand[s_id] -= used

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
                            sort_layer_range=layer_range,
                            sku_pick_count=current_task_pick_count  # ✅ 传入计算结果
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

            # 统计每个 SKU 在当前快照中出现在多少个堆垛里
            sku_stack_count = defaultdict(int)
            for stack_id, totes in self.stack_snapshots.items():
                seen_skus_in_stack = set()
                for t in totes:
                    for sku in t.skus_list:
                        seen_skus_in_stack.add(sku.id)
                for sku_id in seen_skus_in_stack:
                    sku_stack_count[sku_id] += 1

            def get_task_scarcity_score(t: SubTask):
                min_availability = 9999
                for sku in t.unique_sku_list:
                    count = sku_stack_count.get(sku.id, 0)
                    if count < min_availability:
                        min_availability = count
                return (min_availability, -len(t.unique_sku_list))

            sorted_tasks = sorted(sub_tasks, key=get_task_scarcity_score)

            for task in sorted_tasks:
                task.reset_execution_details()
                if task.assigned_station_id == -1:
                    continue

                # ✅ [新增] 追踪该 SubTask 的剩余需求
                task_remaining_demand = defaultdict(int)
                for sku in task.sku_list:
                    task_remaining_demand[sku.id] += 1

                stack_plan = self._greedy_tote_selection_v2(task)

                # 验证：检查选中的 Tote 是否仍然可用
                for stack_idx, needed_totes in list(stack_plan.items()):
                    current_available = set(t.id for t in self.stack_snapshots.get(stack_idx, []))
                    valid_totes = [t for t in needed_totes if t.id in current_available]
                    if valid_totes:
                        stack_plan[stack_idx] = valid_totes
                    else:
                        del stack_plan[stack_idx]

                used_stacks_in_task = set()

                for stack_idx, needed_totes in stack_plan.items():
                    stack = self.problem.point_to_stack[stack_idx]
                    pending_totes = list(needed_totes)
                    used_stacks_in_task.add(stack_idx)
                    self.stack_allocation[stack_idx] = task.id

                    while pending_totes:
                        # --- 批次划分逻辑 ---
                        if len(pending_totes) <= self.robot_capacity:
                            current_batch = pending_totes
                        else:
                            # 计算最大连续层级数
                            layers = [self._get_virtual_layer(t.id) for t in pending_totes]
                            max_len = 0
                            max_start_idx = 0
                            curr_len = 1
                            curr_start = 0
                            for i in range(1, len(layers)):
                                if layers[i] == layers[i - 1] + 1:
                                    curr_len += 1
                                else:
                                    if curr_len > max_len:
                                        max_len = curr_len
                                        max_start_idx = curr_start
                                    curr_len = 1
                                    curr_start = i
                            if curr_len > max_len:
                                max_len = curr_len
                                max_start_idx = curr_start

                            if max_len > self.robot_capacity:
                                batch_indices = set(range(max_start_idx, max_start_idx + self.robot_capacity))
                                current_batch = [pending_totes[i] for i in range(len(pending_totes)) if
                                                 i in batch_indices]
                            elif max_len < 2:
                                take_count = min(len(pending_totes), self.robot_capacity)
                                take_count = max(take_count, 1)
                                current_batch = pending_totes[:take_count]
                            else:
                                end_idx = max_start_idx + max_len
                                batch_indices = set(range(max_start_idx, end_idx))
                                current_batch = [pending_totes[i] for i in range(len(pending_totes)) if
                                                 i in batch_indices]

                        mode, layer_range, sc_time, rc_time = self._decide_operation_mode(
                            stack, current_batch, beta_congestion
                        )

                        # 计算物理负载
                        physical_totes_ids = []
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

                        # 区分 Hit 和 Noise
                        hit_totes_ids = []
                        noise_totes_ids = []
                        needed_tote_ids_set = set(t.id for t in needed_totes)

                        for tid in physical_totes_ids:
                            if tid in needed_tote_ids_set:
                                hit_totes_ids.append(tid)
                            else:
                                noise_totes_ids.append(tid)

                        # ✅ [新增] 计算当前 Task 的 sku_pick_count
                        current_pick_count = 0
                        for t_id in hit_totes_ids:
                            tote = self.problem.id_to_tote.get(t_id)
                            if not tote: continue
                            for s_id, qty in tote.sku_quantity_map.items():
                                if s_id in task_remaining_demand and task_remaining_demand[s_id] > 0:
                                    contribution = min(qty, task_remaining_demand[s_id])
                                    current_pick_count += contribution
                                    task_remaining_demand[s_id] -= contribution

                        # 生成 Task
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
                            sort_layer_range=layer_range,
                            sku_pick_count=current_pick_count  # ✅ 传入计算结果
                        )
                        new_task.station_sequence_rank = task.station_sequence_rank

                        task.add_execution_detail(new_task, stack)
                        physical_tasks.append(new_task)

                        self._global_task_id += 1
                        self.stack_allocation[stack_idx] = task.id
                        self._apply_stack_modification(new_task)

                        final_tote_selection[task.id].extend(physical_totes_ids)
                        if sc_time > 0:
                            final_sorting_costs[task.id] += sc_time

                        carried_hits_set = set(hit_totes_ids)
                        pending_totes = [t for t in pending_totes if t.id not in carried_hits_set]

                        if not physical_totes_ids:
                            break

                for used_stack_idx in used_stacks_in_task:
                    if used_stack_idx in self.stack_allocation:
                        del self.stack_allocation[used_stack_idx]

            print(f"  >>> [SP3 Heuristic] Applying Priority Sorting...")

            for pt in physical_tasks:
                pt.efficiency_score = self._calculate_efficiency_score(pt)

            physical_tasks.sort(key=lambda t: (t.station_sequence_rank, -t.efficiency_score))

            for idx, pt in enumerate(physical_tasks):
                pt.priority = idx + 1
                pt.task_id = idx

            return physical_tasks, final_tote_selection, final_sorting_costs

        def _calculate_efficiency_score(self, task: Task) -> float:
            # ... (保持原样) ...
            valid_tote_count = len(task.hit_tote_ids)
            if valid_tote_count == 0:
                valid_tote_count = 0.1
            action_time = task.robot_service_time
            travel_time = 30.0

            stack_obj = self.problem.point_to_stack.get(task.target_stack_id)
            station_obj = None
            for s in self.problem.station_list:
                if s.id == task.target_station_id:
                    station_obj = s
                    break

            if stack_obj and station_obj:
                p_stack = stack_obj.store_point
                p_station = station_obj.point
                dist = abs(p_stack.x - p_station.x) + abs(p_stack.y - p_station.y)
                speed = getattr(OFSConfig, 'ROBOT_SPEED', 1.5)
                if speed <= 0: speed = 1.5
                travel_time = (dist / speed) * 2.0

            estimated_total_time = action_time + travel_time
            if estimated_total_time < 1.0:
                estimated_total_time = 1.0
            score = valid_tote_count / estimated_total_time
            return score

        def _initialize_stack_snapshots(self):
            for stack in self.problem.stack_list:
                self.stack_snapshots[stack.stack_id] = list(stack.totes)
                for i, tote in enumerate(stack.totes):
                    self.layer_mapping[tote.id] = i

        def _get_virtual_layer(self, tote_id: int) -> int:
            return self.layer_mapping.get(tote_id, -1)

        def _apply_stack_modification(self, task: Task):
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

            for i, tote in enumerate(remaining_totes):
                self.layer_mapping[tote.id] = i
            self.stack_snapshots[stack_id] = remaining_totes

        def _greedy_tote_selection_v2(self, task: SubTask) -> Dict[int, List[Tote]]:
            # ... (保持原样) ...
            pending_skus = sorted([sku.id for sku in task.sku_list])
            pending_skus_set = set(pending_skus)
            selected_stacks_map = defaultdict(list)

            while pending_skus_set:
                sku_availability = defaultdict(list)
                for sku_id in sorted(pending_skus_set):
                    sku_obj = self.problem.id_to_sku[sku_id]
                    for tote_id in sorted(sku_obj.storeToteList):
                        tote = self.problem.id_to_tote.get(tote_id)
                        if not (tote and tote.store_point): continue
                        stack_idx = tote.store_point.idx
                        if stack_idx not in self.stack_snapshots: continue
                        current_totes_in_stack = self.stack_snapshots[stack_idx]
                        if not any(t.id == tote_id for t in current_totes_in_stack): continue
                        sku_availability[sku_id].append(tote)

                if not sku_availability:
                    print(f"  ⚠️ [Heuristic] Cannot find totes for remaining SKUs: {pending_skus_set}")
                    break

                stack_score = {}
                stack_candidate_totes = defaultdict(list)
                for sku_id in sorted(pending_skus_set):
                    candidates = sku_availability.get(sku_id, [])
                    candidates.sort(key=lambda t: t.id)
                    for tote in candidates:
                        s_idx = tote.store_point.idx
                        bundle_bonus = 100 if s_idx in selected_stacks_map else 1
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
                    break

                best_stack_idx = max(stack_score.items(), key=lambda x: (x[1], -x[0]))[0]
                chosen_totes = stack_candidate_totes[best_stack_idx]
                chosen_totes.sort(key=lambda t: (
                    -len(set(s.id for s in t.skus_list) & pending_skus_set),
                    self._get_virtual_layer(t.id),
                    t.id
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
            # ... (保持原样) ...
            stack_id = stack.stack_id
            current_snapshot = self.stack_snapshots.get(stack_id, [])
            current_height = len(current_snapshot)
            if current_height == 0:
                return ('FLIP', None, 0.0, 0.0)

            target_indices = [self._get_virtual_layer(t.id) for t in target_totes]
            top_layer_idx = current_height - 1

            time_flip = 0.0
            for idx in target_indices:
                is_deep = 1 if idx < top_layer_idx else 0
                time_flip += (self.t_shift + is_deep * self.t_lift)
            cost_flip = self.alpha * time_flip
            res_flip = ('FLIP', None, 0.0, time_flip)

            deepest_index = min(target_indices)
            highest_needed_index = max(target_indices)
            candidates = []

            range_a = (deepest_index, highest_needed_index)
            size_a = highest_needed_index - deepest_index + 1
            if size_a <= self.robot_capacity:
                has_lift_a = (highest_needed_index < top_layer_idx)
                time_rc_a = (self.t_shift + self.t_lift * (1 if has_lift_a else 0))
                cost_rc_a = self.alpha * time_rc_a
                noise_a = size_a - len(target_totes)
                cost_sc_a = beta * self.t_move * noise_a
                candidates.append((cost_rc_a + cost_sc_a, 'SORT', range_a, self.t_move * noise_a, time_rc_a))

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
            if best_sort[0] < cost_flip:
                return best_sort[1], best_sort[2], best_sort[3], best_sort[4]
            else:
                return res_flip


