import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional
from entity.MainBatch import MainBatch
from entity.order import Order
from problemDto.ofs_problem_dto import OFSProblemDTO

class SP1Variable:
    """
    变量容器：用于保存 SP1 的结果
    """
    def __init__(self, O_size: int, B_size: int):
        # 决策变量
        self.z_ob = np.zeros((O_size, B_size), dtype=int)  # order → sub-task assignment
        self.u_b = np.zeros((B_size,), dtype=int)          # sub-task active

        # 额外保存 order_id 映射（方便后续用）
        self.order_index_map: Dict[int, int] = {}  # order_id → index

    def set_solution(self, orders: List[Order], z_vars, u_vars):
        """根据 Gurobi 求解结果加载变量"""
        # 建立订单索引映射
        self.order_index_map = {orders[i].order_id: i for i in range(len(orders))}
        for i, o in enumerate(orders):
            for b in range(self.u_b.shape[0]):
                self.z_ob[i, b] = int(z_vars[o, b].X > 0.5)
        for b in range(self.u_b.shape[0]):
            self.u_b[b] = int(u_vars[b].X > 0.5)


class SolveSP1:
    """
    求解子问题1：Bin Packing
    """
    def solve(self,
              main_batch: MainBatch,
              problem_dto: OFSProblemDTO,
              MAX_TASK_WEIGHT: float) -> Optional[SP1Variable]:

        print(f"开始求解 SP1，目标: MainBatch {main_batch.id}")

        O_objects: List[Order] = main_batch.orders
        order_weights: Dict[int, float] = {}
        has_infeasible_order = False

        try:
            for order in O_objects:
                current_order_weight = 0.0
                for sku_id in order.order_product_id_list:
                    current_order_weight += problem_dto.id_to_sku[sku_id].weight

                order_weights[order.order_id] = current_order_weight

                if current_order_weight > MAX_TASK_WEIGHT:
                    print(f"[错误] SP1 无解：订单 {order.order_id} 的总重 "
                          f"({current_order_weight:.2f}) 超过了 "
                          f"MAX_TASK_WEIGHT ({MAX_TASK_WEIGHT})。")
                    has_infeasible_order = True

        except KeyError as e:
            print(f"[错误] SP1失败：订单中的SKU ID {e} 在 problem_dto.id_to_sku 中未找到。")
            return None

        if has_infeasible_order:
            return None

        if not O_objects:
            print("警告: 此 MainBatch 中没有需要处理的 Orders。")
            return None

        num_potential_tasks = len(O_objects)
        B_indices = range(num_potential_tasks)

        print(f"问题规模: {len(O_objects)} 个 Orders, {num_potential_tasks} 个潜在子任务。")
        print(f"容量限制 C_max: {MAX_TASK_WEIGHT}")

        # 创建 Gurobi 模型
        m = gp.Model("SP1_Order_Clustering")
        m.setParam('OutputFlag', 0)

        # 决策变量: z_ob 和 u_b
        z = m.addVars(O_objects, B_indices, vtype=GRB.BINARY, name="z")
        u = m.addVars(B_indices, vtype=GRB.BINARY, name="u")

        # 目标函数：最少子任务个数
        m.setObjective(gp.quicksum(u[b] for b in B_indices), GRB.MINIMIZE)

        # 约束 (1) 每个订单分配到唯一子任务
        m.addConstrs((z.sum(o, '*') == 1 for o in O_objects), name="order_assign")

        # 约束 (2) 订单只能分配到已激活子任务
        m.addConstrs((z[o, b] <= u[b] for o in O_objects for b in B_indices), name="linking")

        # 约束 (3) 子任务容量限制
        m.addConstrs(
            (gp.quicksum(order_weights[o.order_id] * z[o, b] for o in O_objects)
             <= MAX_TASK_WEIGHT * u[b]
             for b in B_indices),
            name="capacity"
        )

        print("Gurobi 开始求解 SP1...")
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"求解成功, 最小子任务数量: {int(m.ObjVal)}")
            # 创建变量容器并保存结果
            sp1_var = SP1Variable(len(O_objects), len(B_indices))
            sp1_var.set_solution(O_objects, z, u)
            return sp1_var

        elif m.Status == GRB.INFEASIBLE:
            print("[错误] SP1 模型无解 (Infeasible)。")
            m.computeIIS()
            m.write("sp1_infeasible_model.ilp")
            return None
        else:
            print(f"[警告] Gurobi 求解结束, 状态: {m.Status}")
            return None
