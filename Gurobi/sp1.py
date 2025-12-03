# solver/sp1_bom_splitter.py

import math
from typing import List, Dict, Tuple
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

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
                tasks = self._split_order_by_spatial_clustering(order)

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

    # =========================================================================
    #  基于 MIP 的精确覆盖
    # =========================================================================
    def _split_order_by_mip(self, order) -> List[SubTask]:
        """
        使用 Gurobi 求解 Set Partitioning 问题。
        仅当需要极度优化任务数量，且不在乎 SKUs 空间离散度时使用。
        """
        cap_limit = self.order_capacity_limits.get(order.order_id, OFSConfig.ROBOT_CAPACITY)
        skus = [self.problem.id_to_sku[sid] for sid in order.order_product_id_list]
        n_items = len(skus)

        # 估算最大任务数 K
        max_k = math.ceil(n_items / 1.0)

        m = gp.Model(f"SP1_Order_{order.order_id}")
        m.Params.OutputFlag = 0

        # y[i, k]: 第 i 个 SKU 是否放入 第 k 个任务
        y = m.addVars(n_items, max_k, vtype=GRB.BINARY, name="y")
        # z[k]: 第 k 个任务是否启用
        z = m.addVars(max_k, vtype=GRB.BINARY, name="z")

        # 约束 1: 覆盖 (Cover)
        for i in range(n_items):
            m.addConstr(gp.quicksum(y[i, k] for k in range(max_k)) == 1)

        # 约束 2: 容量 (Capacity)
        for k in range(max_k):
            m.addConstr(gp.quicksum(y[i, k] for i in range(n_items)) <= cap_limit * z[k])

        # 目标: 最小化任务数
        m.setObjective(gp.quicksum(z[k] for k in range(max_k)), GRB.MINIMIZE)

        m.optimize()

        generated = []
        if m.status == GRB.OPTIMAL:
            for k in range(max_k):
                if z[k].X > 0.5:
                    task_skus = []
                    for i in range(n_items):
                        if y[i, k].X > 0.5:
                            task_skus.append(skus[i])

                    if task_skus:
                        task = SubTask(
                            id=self._global_task_id,
                            parent_order=order,
                            sku_list=task_skus
                        )
                        generated.append(task)
                        self._global_task_id += 1
        else:
            # Fallback if MIP fails
            return self._split_order_by_spatial_clustering(order)

        return generated
if __name__ == "__main__":
    # 测试代码可以放在这里
    # 初始化问题和求解器
    scales = ["SMALL", "MEDIUM"]
    problem_dto = CreateOFSProblem.generate_problem_by_scale('SMALL')
    sp1_solver = SP1_BOM_Splitter(problem_dto)

    # 1. 默认生成（基于空间聚类，使用全局容量限制）
    initial_tasks = sp1_solver.solve(use_mip=True)
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
    print("Initial task generation verified: all orders covered correctly.")
    # 2. 模拟软耦合反馈：SP3 发现 Order 5 的某些任务因为容量太满无法使用 Sort 模式
    # 反馈：将 Order 5 的子任务容量限制降为 4
    sp1_solver.update_capacity_feedback(order_id=5, new_limit=4)

    # 3. 重新生成（Order 5 将被拆得更细，其余保持不变）
    refined_tasks = sp1_solver.solve(use_mip=False)