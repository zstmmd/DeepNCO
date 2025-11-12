import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional

from entity.MainBatch import MainBatch
from entity.Order import Order
from entity.SKUs import SKUs
from problemDto.ofs_problem_dto import OFSProblemDTO


class SolveSP1:
    """
    求解子问题1： Bin Packing
    """

    def solve(self,
              main_batch: MainBatch,
              problem_dto: OFSProblemDTO,
              MAX_TASK_WEIGHT: float) -> Optional[List[List[Order]]]:

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

                # 检查单个订单是否就超过了容量
                if current_order_weight > MAX_TASK_WEIGHT:
                    print(f"[错误] SP1 无解：订单 {order.order_id} 的总重 "
                          f"({current_order_weight:.2f}) 超过了 "
                          f"MAX_TASK_WEIGHT ({MAX_TASK_WEIGHT})。")
                    has_infeasible_order = True

        except KeyError as e:
            print(f"[错误] SP1失败：订单中的SKU ID {e} 在 problem_dto.id_to_sku 中未找到。")
            return None

        if has_infeasible_order:
            return None  # 如果有单个订单超重，则无法求解

        if not O_objects:
            print("警告: 此MainBatch中没有需要处理的Orders。")
            return []

        num_potential_tasks = len(O_objects)
        B_indices = range(num_potential_tasks)

        print(f"问题规模: {len(O_objects)} 个Orders, {num_potential_tasks} 个潜在子任务。")
        print(f"容量限制 C_max: {MAX_TASK_WEIGHT}")

        # 2. 创建 Gurobi 模型
        m = gp.Model("SP1_Order_Clustering")
        m.setParam('OutputFlag', 0)  # 减少冗余输出


        # z_ob (z[o, b]): Order o 分配给 子任务 b
        z = m.addVars(O_objects, B_indices, vtype=GRB.BINARY, name="z")

        # u_b (u[b]): 子任务 b 被使用
        u = m.addVars(B_indices, vtype=GRB.BINARY, name="u")

        # 4. 设置目标函数 (eq:objective)
        # min sum(u_b)
        m.setObjective(gp.quicksum(u[b] for b in B_indices), GRB.MINIMIZE)

        # 5. 添加约束

        # (1) 订单分配约束
        # 每个Order必须且只能被分配到一个子任务
        # sum(z_ob over b) = 1, for all o in O
        m.addConstrs(
            (z.sum(o, '*') == 1 for o in O_objects),
            name="order_assign"
        )

        # (2) 链接约束 (eq:linking)
        # Order o 只能被分配到 *已激活* 的子任务 b
        # z_ob <= u_b, for all o, b
        m.addConstrs(
            (z[o, b] <= u[b] for o in O_objects for b in B_indices),
            name="linking"
        )

        # (3) 子任务容量约束
        # 分配到子任务b的所有Order的总重量
        # 不得超过 C_max * u_b
        # sum(Order_Weight(o) * z_ob over o) <= C_max * u_b, for all b
        m.addConstrs(
            (gp.quicksum(order_weights[o.order_id] * z[o, b] for o in O_objects)
             <= MAX_TASK_WEIGHT * u[b]
             for b in B_indices),
            name="capacity"
        )
        # 6. 运行求解器
        print("Gurobi 开始求解...")
        m.optimize()

        # 7. 解析并返回结果
        # 现在的返回类型是 List[List[Order]]
        solution_tasks: List[List[Order]] = []

        if m.Status == GRB.OPTIMAL:
            print(f"求解成功。状态: OPTIMAL")
            print(f"最小子任务数量: {int(m.ObjVal)}")

            # 提取结果
            for b in B_indices:
                if u[b].X > 0.5:  # 如果这个子任务(bin)被使用了

                    # 创建一个新的子任务 (订单列表)
                    current_sub_task: List[Order] = []

                    for o in O_objects:
                        if z[o, b].X > 0.5:  # 如果Order o被分配到了这个子任务b
                            current_sub_task.append(o)

                    if current_sub_task:
                        solution_tasks.append(current_sub_task)

            print(f"成功创建 {len(solution_tasks)} 个子任务。")
            return solution_tasks

        elif m.Status == GRB.INFEASIBLE:
            print("[错误] SP1 模型无解 (Infeasible)。")
            print("  (已在预检中排除了单个订单超重的情况，这不应该发生)")
            m.computeIIS()
            m.write("sp1_infeasible_model.ilp")
            print("已写入 'sp1_infeasible_model.ilp' 供分析。")
            return None

        else:
            print(f"Gurobi 求解结束，但未找到最优解。状态: {m.Status}")
            return None