# solver/sp1_bom_splitter.py

import math
from typing import List, Dict, Tuple,DefaultDict
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from entity.SKUs import SKUs
from problemDto.createInstance import CreateOFSProblem
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig
from entity.subTask import SubTask
from entity.point import Point


class SP1_BOM_Splitter:
    """
    SP1 子问题求解器：BOM 拆分 (Sub-task Creation)
    功能：将大订单拆分为符合容量限制的子任务。
    特性：支持空间聚类生成初始解，支持软耦合动态容量限制。
    """

    def __init__(self, problem: OFSProblemDTO):
        self.problem = problem
        self._global_task_id = 0

        # --- 软耦合状态 ---
        # 存储每个订单允许的最大子任务容量
        # key: order_id, value: max_skus (默认 = 机器人最大容量)
        self.order_capacity_limits: Dict[int, int] = {}
        self._init_capacity_limits()

    def _init_capacity_limits(self):
        """初始化容量限制为全局默认值"""
        default_cap = OFSConfig.ROBOT_CAPACITY-2
        for order in self.problem.order_list:
            self.order_capacity_limits[order.order_id] = default_cap

    def update_capacity_feedback(self, order_id: int, new_limit: int):
        """
        [软耦合接口] 接收来自 SP3/SP4 的反馈，降低特定订单的容量上限
        以便后续能凑出 Sort Mode (整垛搬运)
        """
        if order_id in self.order_capacity_limits:
            current = self.order_capacity_limits[order_id]
            # 只能降低，不能超过物理极限
            valid_limit = max(1, min(new_limit, OFSConfig.ROBOT_CAPACITY))
            self.order_capacity_limits[order_id] = valid_limit
            print(f"[SP1 Feedback] Order {order_id} capacity limit updated: {current} -> {valid_limit}")

    def solve(self, use_mip: bool = False) -> List[SubTask]:
        """
        执行拆分
        :param use_mip: 是否使用 MIP 进行精确求解 (通常启发式已足够)
        :return: 生成的子任务列表
        """
        self._global_task_id = 0  # 重置 ID 计数器
        all_sub_tasks = []

        print(f">>> [SP1] Starting BOM Splitting (MIP={use_mip})...")

        # SP1 的特性是：订单之间是独立的。
        # 因此我们可以遍历每个订单，单独进行拆分，无需构建全局大模型。
        for order in self.problem.order_list:
            if use_mip:
                tasks = self._split_order_by_mip(order)
            else:
                tasks = self._split_order_by_stack_clustering(order)

            all_sub_tasks.extend(tasks)

        print(f">>> [SP1] Finished. Generated {len(all_sub_tasks)} tasks.")
        return all_sub_tasks

    # =========================================================================
    # 基于空间聚类的启发式拆分
    # =========================================================================
    def _split_order_by_spatial_clustering(self, order) -> List[SubTask]:
        """
        高质量初始解生成器：
        1. 获取订单中每个 SKU 的参考物理坐标。
        2. 根据坐标对 SKU 进行空间排序 (Spatial Sorting)。
        3. 按顺序切分，保证生成的子任务内的 SKU 在物理上是相近的。
        """
        # 1. 获取当前订单的动态容量限制
        cap_limit = self.order_capacity_limits.get(order.order_id, OFSConfig.ROBOT_CAPACITY)

        # 2. 构建 SKU 位置信息列表
        # list of dict: {'sku': sku_obj, 'x': int, 'y': int}
        sku_locations = []

        for sku_id in order.order_product_id_list:
            sku_obj = self.problem.id_to_sku.get(sku_id)
            if not sku_obj: continue

            # 确定参考坐标：
            # 策略：取该 SKU 库存量最大的那个料箱的位置，或者离默认中心点最近的料箱
            # 这里简化为：取第一个可用存储点
            ref_x, ref_y = 0, 0
            if sku_obj.storeToteList:
                first_tote_id = sku_obj.storeToteList[0]
                tote = self.problem.id_to_tote.get(first_tote_id)
                if tote and tote.store_point:
                    ref_x = tote.store_point.x
                    ref_y = tote.store_point.y

            sku_locations.append({
                'sku': sku_obj,
                'x': ref_x,
                'y': ref_y
            })

        # 3. 空间排序 (Spatial Sort)
        # 简单的策略：按 Y 轴主序，X 轴次序 (或者 X+Y, 或者 Hilbert Curve)
        # 这样可以将物理上相邻的 SKU 排在一起
        sku_locations.sort(key=lambda k: (k['y'], k['x']))

        # 4. 提取排序后的 SKU 对象
        sorted_skus = [item['sku'] for item in sku_locations]

        # 5. 切分 (Slicing)
        sub_tasks = []
        total_items = len(sorted_skus)

        for i in range(0, total_items, cap_limit):
            # 切片
            chunk = sorted_skus[i: i + cap_limit]

            # 创建子任务
            task = SubTask(
                id=self._global_task_id,
                parent_order=order,
                sku_list=chunk
            )
            sub_tasks.append(task)
            self._global_task_id += 1

        return sub_tasks
    def _split_order_by_stack_clustering(self, order) -> List[SubTask]:
        """
        [新增方法] 基于堆垛的强聚类拆分
        目标：让同一 SubTask 的 SKU 尽量来自同一组 Stack
        """
        cap_limit = self.order_capacity_limits.get(order.order_id, OFSConfig.ROBOT_CAPACITY)
        
        # 1. 聚合 SKU 实例
        sku_groups: DefaultDict[int, List[SKUs]] = defaultdict(list)
        for sku_id in order.order_product_id_list:
            sku_obj = self.problem.id_to_sku.get(sku_id)
            if sku_obj:
                sku_groups[sku_id].append(sku_obj)
        
        unique_sku_ids = list(sku_groups.keys())
        
        # 2. 构建 SKU -> Stack 关系图
        sku_to_stacks: Dict[int, Set[int]] = {}
        stack_to_skus: Dict[int, Set[int]] = defaultdict(set)
        
        for sku_id in unique_sku_ids:
            sku_obj = sku_groups[sku_id][0]
            available_stacks = set()
            
            for tote_id in sku_obj.storeToteList:
                tote = self.problem.id_to_tote.get(tote_id)
                if tote and tote.store_point:
                    stack_idx = tote.store_point.idx
                    available_stacks.add(stack_idx)
                    stack_to_skus[stack_idx].add(sku_id)
            
            sku_to_stacks[sku_id] = available_stacks
        
        # 3. 贪婪聚类：优先选择能覆盖最多 SKU 的 Stack 组合
        sub_tasks = []
        remaining_skus = set(unique_sku_ids)
        
        while remaining_skus:
            # 当前 SubTask 的 SKU 集合
            current_task_skus = []
            used_stacks = set()
            
            while len(current_task_skus) < cap_limit and remaining_skus:
                # 选择能覆盖最多剩余 SKU 的 Stack
                best_stack = None
                best_coverage = 0
                
                for stack_idx, sku_set in stack_to_skus.items():
                    if stack_idx in used_stacks:
                        continue
                    
                    # 计算该 Stack 能覆盖多少剩余 SKU
                    coverage = len(sku_set & remaining_skus)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_stack = stack_idx
                
                if best_stack is None:
                    break  # 无法继续聚类，退出
                
                # 将该 Stack 上的 SKU 加入当前 SubTask
                covered_skus = stack_to_skus[best_stack] & remaining_skus
                for sku_id in covered_skus:
                    if len(current_task_skus) < cap_limit:
                        current_task_skus.append(sku_id)
                        remaining_skus.remove(sku_id)
                
                used_stacks.add(best_stack)
            
            # 4. 生成 SubTask
            if current_task_skus:
                task_full_sku_list = []
                for sid in current_task_skus:
                    task_full_sku_list.extend(sku_groups[sid])
                
                task = SubTask(
                    id=self._global_task_id,
                    parent_order=order,
                    sku_list=task_full_sku_list
                )
                sub_tasks.append(task)
                self._global_task_id += 1
        
        return sub_tasks
    def _split_order_by_unique_types(self, order) -> List[SubTask]:
        """
        核心逻辑修正：
        1. 聚合：Order [A, A, B, C, C, C] -> {A: [A,A], B: [B], C: [C,C,C]}
        2. 空间排序：根据 A, B, C 的物理位置排序 -> [A, C, B] (假设位置关系)
        3. 切分：容量=2 -> Task1(A, C), Task2(B)
        """
        # 1. 获取容量限制 (Max Unique SKUs per Task)
        cap_limit = self.order_capacity_limits.get(order.order_id, OFSConfig.ROBOT_CAPACITY)

        # 2. 聚合 SKU 实例
        # sku_groups: SKU_ID -> List[SKUs objects]
        sku_groups: DefaultDict[int, List[SKUs]] = defaultdict(list)

        for sku_id in order.order_product_id_list:
            sku_obj = self.problem.id_to_sku.get(sku_id)
            if sku_obj:
                sku_groups[sku_id].append(sku_obj)

        unique_sku_ids = list(sku_groups.keys())

        # 3. 准备空间排序数据
        # location_info: {'sku_id': int, 'x': int, 'y': int}
        location_info_list = []

        for sku_id in unique_sku_ids:
            sku_obj = sku_groups[sku_id][0]  # 取第一个对象查位置即可

            # 确定参考坐标 (取第一个有库存的料箱位置)
            ref_x, ref_y = 0, 0
            if sku_obj.storeToteList:
                first_tote_id = sku_obj.storeToteList[0]
                tote = self.problem.id_to_tote.get(first_tote_id)
                if tote and tote.store_point:
                    ref_x = tote.store_point.x
                    ref_y = tote.store_point.y

            location_info_list.append({
                'sku_id': sku_id,
                'x': ref_x,
                'y': ref_y
            })

        # 4. 空间排序 (Spatial Sort)
        # 确保物理相近的 SKU 种类被分在同一组
        location_info_list.sort(key=lambda k: (k['y'], k['x']))

        sorted_unique_ids = [item['sku_id'] for item in location_info_list]

        # 5. 切分并生成 SubTask
        sub_tasks = []
        total_types = len(sorted_unique_ids)

        # 按“种类”步进切分
        for i in range(0, total_types, cap_limit):
            # 这一组包含的 SKU 种类 ID
            chunk_ids = sorted_unique_ids[i: i + cap_limit]

            # 还原为完整的 SKU 对象列表 (包含数量)
            # Task1: 种类 [A, C] -> 列表 [A, A, C, C, C]
            task_full_sku_list = []
            for sid in chunk_ids:
                task_full_sku_list.extend(sku_groups[sid])

            # 创建子任务
            task = SubTask(
                id=self._global_task_id,
                parent_order=order,
                sku_list=task_full_sku_list
            )
            sub_tasks.append(task)
            self._global_task_id += 1

        return sub_tasks
    # =========================================================================
    #  基于 MIP 的精确覆盖
    # =========================================================================
    def _split_order_by_mip(self, order) -> List[SubTask]:

        # 1. 获取容量限制 (Max Unique SKUs per Task)
        cap_limit = self.order_capacity_limits.get(order.order_id, OFSConfig.ROBOT_CAPACITY)

        # 2. 聚合 SKU：找出该订单包含的所有唯一 SKU 对象
        # sku_groups: SKU_ID -> List[SKUs objects] (即该 SKU 的所有实例)
        sku_groups: Dict[int, List[SKUs]] = defaultdict(list)
        for sku_id in order.order_product_id_list:
            sku_obj = self.problem.id_to_sku.get(sku_id)
            if sku_obj:
                sku_groups[sku_id].append(sku_obj)

        # 唯一 SKU 列表 (MIP 的操作对象)
        unique_skus = [self.problem.id_to_sku[sid] for sid in sku_groups.keys()]
        n_types = len(unique_skus)

        # 估算最大任务数 K (最坏情况：每个种类一个任务)
        max_k = math.ceil(n_types / 1.0)
        K_range = range(max_k)

        # --- 构建模型 ---
        m = gp.Model(f"SP1_Order_{order.order_id}")
        m.Params.OutputFlag = 0

        # 决策变量
        # x[j, k]: 第 j 种 SKU 是否放入 第 k 个任务
        x = m.addVars(n_types, max_k, vtype=GRB.BINARY, name="x")
        # z[k]: 第 k 个任务是否启用
        z = m.addVars(max_k, vtype=GRB.BINARY, name="z")

        # 约束 1: 覆盖 (每个种类必须且只能分配给一个任务)
        for j in range(n_types):
            m.addConstr(gp.quicksum(x[j, k] for k in K_range) == 1, name=f"Cover_{j}")

        # 约束 2: 容量 (限制每个任务包含的种类数量)
        for k in K_range:
            # sum(x[j,k]) <= Limit * z[k]
            m.addConstr(gp.quicksum(x[j, k] for j in range(n_types)) <= cap_limit * z[k], name=f"Cap_{k}")

        # 目标: 最小化任务数 (及潜在的软耦合惩罚)
        # 如果有软耦合(SKU互斥)，可以在这里加二次项 obj += x[a,k]*x[b,k]*penalty
        m.setObjective(gp.quicksum(z[k] for k in K_range), GRB.MINIMIZE)

        m.optimize()

        # --- 结果解析 ---
        generated = []
        if m.status == GRB.OPTIMAL:
            for k in K_range:
                if z[k].X > 0.5:
                    # 这个任务包含哪些种类？
                    task_full_sku_list = []

                    for j in range(n_types):
                        if x[j, k].X > 0.5:
                            sku_id = unique_skus[j].id
                            # 【关键】把该种类的所有实例全部加进去
                            # 保证同一种 SKU 不会被拆散到两个任务
                            task_full_sku_list.extend(sku_groups[sku_id])

                    if task_full_sku_list:
                        task = SubTask(
                            id=self._global_task_id,
                            parent_order=order,
                            sku_list=task_full_sku_list
                        )
                        generated.append(task)
                        self._global_task_id += 1
        else:
            print(f"Warning: SP1 MIP failed for order {order.order_id}, falling back to heuristic.")
            return self._split_order_by_spatial_clustering(order)  # 这里的启发式也需要确保是按种类切分的逻辑

        return generated
if __name__ == "__main__":
    # 初始化问题和求解器
    scales = ["SMALL", "MEDIUM"]
    problem_dto = CreateOFSProblem.generate_problem_by_scale('SMALL')
    sp1_solver = SP1_BOM_Splitter(problem_dto)

    # 1. 默认生成（基于空间聚类，使用全局容量限制）
    initial_tasks = sp1_solver.solve(use_mip=True)
    #验证是否覆盖order的所有sku
    order_to_skus: Dict[int, List[int]] = defaultdict(list)
    order_unique_count_check: Dict[int, int] = defaultdict(int)
    for task in initial_tasks:
        order_id = task.parent_order.order_id
        sku_ids = [sku.id for sku in task.sku_list]
        order_to_skus[order_id].extend(sku_ids)
        order_unique_count_check[order_id] += task.sku_quantity

    for order in problem_dto.order_list:
        # 1. 验证所有 SKU 实例是否都存在
        original_skus = sorted(order.order_product_id_list)
        generated_skus = sorted(order_to_skus[order.order_id])
        assert original_skus == generated_skus, f"Order {order.order_id} SKU mismatch!"

        # 2. 验证 Unique SKU 数量统计
        original_unique_count = len(set(order.order_product_id_list))
        task_unique_sum = order_unique_count_check[order.order_id]

        print(f"Order {order.order_id}: Original Unique={original_unique_count}, Sum from Tasks={task_unique_sum}")
        assert original_unique_count == task_unique_sum, f"Order {order.order_id} Unique SKU count mismatch!"

    print("Initial task generation verified: all orders covered correctly and unique counts match.")
    #  模拟软耦合反馈：SP3 发现 Order 5 的某些任务因为容量太满无法使用 Sort 模式
    # 反馈：将 Order 5 的子任务容量限制降为 4
    sp1_solver.update_capacity_feedback(order_id=5, new_limit=4)

    #  重新生成（Order 5 将被拆得更细，其余保持不变）
    refined_tasks = sp1_solver.solve(use_mip=False)