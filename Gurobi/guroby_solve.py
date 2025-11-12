'''
定价子问题
'''
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Set
import numpy as np

from problemDto.ofs_problem_dto import OFSProblemDTO
from entity.order import Order
from entity.tote import Tote
from entity.robot import Robot
from entity.station import Station
from entity.TaskBatch import TaskBatch
from entity.MainBatch import MainBatch
from config.ofs_config import OFSConfig


class PricingProblemSolver:
    """
    子问题求解器 - 生成reduced cost为负的新任务批次

    子问题本质上是一个带时间窗和容量约束的拣货路径规划问题
    """

    def __init__(self, problem: OFSProblemDTO, dual_variables: Dict):
        """
        初始化子问题求解器

        :param problem: 原始问题实例
        :param dual_variables: 主问题的对偶变量
            - 'order_duals': Dict[order_id, float] 订单覆盖约束的对偶值
            - 'bin_duals': Dict[(location, layer, sku_id), float] 料箱库存约束的对偶值
        """
        self.problem = problem
        self.dual_variables = dual_variables

        # 提取对偶变量
        self.order_duals = dual_variables.get('order_duals', {})
        self.bin_duals = dual_variables.get('bin_duals', {})

        # 预计算距离矩阵
        self._compute_distances()

    def _compute_distances(self):
        """预计算料箱位置之间的曼哈顿距离"""
        self.distance_matrix = {}

        # 获取所有料箱位置
        tote_locations = set()
        for tote in self.problem.tote_list:
            if tote.store_point:
                tote_locations.add(tote.store_point.idx)

        # 计算距离
        for loc1 in tote_locations:
            for loc2 in tote_locations:
                if loc1 != loc2:
                    point1 = self.problem.map.point_list[loc1]
                    point2 = self.problem.map.point_list[loc2]
                    # 曼哈顿距离
                    dist = abs(point1.x - point2.x) + abs(point1.y - point2.y)
                    self.distance_matrix[(loc1, loc2)] = dist
                else:
                    self.distance_matrix[(loc1, loc2)] = 0

    def solve(self, main_batch: MainBatch, robot: Robot,
              max_orders_per_batch: int = 5) -> List[TaskBatch]:
        """
        求解子问题，生成新的任务批次

        :param main_batch: 目标主批次
        :param robot: 可用机器人
        :param max_orders_per_batch: 单个任务批次最多包含的订单数
        :return: 生成的新任务批次列表(reduced cost < 0)
        """
        new_batches = []

        # 获取主批次中的订单
        candidate_orders = [o for o in main_batch.orders
                          if o.status == "pending"]

        if not candidate_orders:
            return new_batches

        # 尝试不同的订单组合
        for k in range(1, min(max_orders_per_batch + 1, len(candidate_orders) + 1)):
            # 使用贪心策略选择订单子集
            order_subset = self._select_order_subset(candidate_orders, k)

            if not order_subset:
                continue

            # 为该订单子集构建并求解子问题
            task_batch = self._build_and_solve_subproblem(
                order_subset, main_batch, robot
            )

            if task_batch and task_batch.reduced_cost < -1e-6:
                new_batches.append(task_batch)

                # 限制生成数量
                if len(new_batches) >= 10:
                    break

        return new_batches

    def _select_order_subset(self, orders: List[Order], k: int) -> List[Order]:
        """
        选择k个订单的子集(基于对偶值的贪心策略)

        :param orders: 候选订单列表
        :param k: 要选择的订单数量
        :return: 选中的订单列表
        """
        # 按对偶值降序排序(对偶值高的订单优先)
        sorted_orders = sorted(
            orders,
            key=lambda o: self.order_duals.get(o.order_id, 0),
            reverse=True
        )

        return sorted_orders[:k]

    def _build_and_solve_subproblem(self, orders: List[Order],
                                    main_batch: MainBatch,
                                    robot: Robot) -> TaskBatch:
        """
        为给定订单子集构建并求解子问题的MIP模型

        决策变量:
        - x[i]: 订单i是否被选中
        - s[l,j,u]: 从位置l第j层料箱取SKU u的数量
        - phi[l,j]: 是否使用位置l第j层的料箱
        - psi[l]: 是否访问位置l
        - delta[l1,l2]: 是否从l1直接前往l2
        - pi[l]: 位置l的访问顺序
        """
        try:
            model = gp.Model("PricingSubproblem")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 30)  # 限制求解时间

            # === 1. 汇总订单需求 ===
            total_demand = {}  # {sku_id: quantity}
            for order in orders:
                for sku_id in order.order_product_id_list:
                    total_demand[sku_id] = total_demand.get(sku_id, 0) + 1

            # === 2. 构建料箱-SKU映射 ===
            tote_sku_map = {}  # {(location, layer, sku_id): available_qty}
            location_set = set()

            for tote in self.problem.tote_list:
                if not tote.store_point:
                    continue

                loc = tote.store_point.idx
                layer = tote.layer
                location_set.add(loc)

                for sku_id, qty in tote.sku_quantity_map.items():
                    if sku_id in total_demand and qty > 0:
                        tote_sku_map[(loc, layer, sku_id)] = qty

            if not tote_sku_map:
                return None

            locations = sorted(list(location_set))

            # === 3. 决策变量 ===

            # 订单选择变量
            x = {}
            for order in orders:
                x[order.order_id] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{order.order_id}"
                )

            # SKU拣选变量
            s = {}
            for (loc, layer, sku_id), max_qty in tote_sku_map.items():
                s[(loc, layer, sku_id)] = model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=max_qty,
                    name=f"s_{loc}_{layer}_{sku_id}"
                )

            # 料箱使用变量
            phi = {}
            psi = {}
            for tote in self.problem.tote_list:
                if not tote.store_point:
                    continue
                loc = tote.store_point.idx
                layer = tote.layer
                phi[(loc, layer)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"phi_{loc}_{layer}"
                )

            for loc in locations:
                psi[loc] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"psi_{loc}"
                )

            # 路径变量
            delta = {}
            pi = {}
            for i, loc1 in enumerate(locations):
                pi[loc1] = model.addVar(
                    vtype=GRB.INTEGER,
                    lb=1,
                    ub=len(locations),
                    name=f"pi_{loc1}"
                )
                for loc2 in locations:
                    if loc1 != loc2:
                        delta[(loc1, loc2)] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"delta_{loc1}_{loc2}"
                        )

            model.update()

            # === 4. 目标函数: 最小化 (成本 - 对偶值) ===

            # 行驶时间
            travel_cost = gp.quicksum(
                self.distance_matrix.get((loc1, loc2), 0) / robot.speed * delta[(loc1, loc2)]
                for loc1 in locations for loc2 in locations if loc1 != loc2
            )

            # 挖掘时间(简化: 每使用一个料箱计dig_time)
            dig_cost = gp.quicksum(
                OFSConfig.DIG_TIME * phi[(loc, layer)]
                for loc, layer in phi.keys()
            )

            # 对偶值(订单)
            dual_benefit = gp.quicksum(
                self.order_duals.get(order.order_id, 0) * x[order.order_id]
                for order in orders
            )

            # Reduced cost = cost - dual_benefit
            model.setObjective(
                travel_cost + dig_cost - dual_benefit,
                GRB.MINIMIZE
            )

            # === 5. 约束 ===

            # 5.1 SKU需求满足
            for sku_id in total_demand.keys():
                model.addConstr(
                    gp.quicksum(
                        s[(loc, layer, sku_id)]
                        for (loc, layer, sid) in s.keys()
                        if sid == sku_id
                    ) >= total_demand[sku_id],
                    name=f"demand_{sku_id}"
                )

            # 5.2 料箱库存约束
            for (loc, layer, sku_id), max_qty in tote_sku_map.items():
                model.addConstr(
                    s[(loc, layer, sku_id)] <= max_qty * phi[(loc, layer)],
                    name=f"bin_cap_{loc}_{layer}_{sku_id}"
                )

            # 5.3 料箱使用定义
            for (loc, layer) in phi.keys():
                model.addConstr(
                    gp.quicksum(
                        s[(loc, layer, sku_id)]
                        for (l, ly, sku_id) in s.keys()
                        if l == loc and ly == layer
                    ) >= phi[(loc, layer)],
                    name=f"bin_use_{loc}_{layer}"
                )

            # 5.4 位置访问定义
            for loc in locations:
                bins_at_loc = [(l, ly) for (l, ly) in phi.keys() if l == loc]
                if bins_at_loc:
                    model.addConstr(
                        gp.quicksum(phi[(l, ly)] for (l, ly) in bins_at_loc)
                        <= len(bins_at_loc) * psi[loc],
                        name=f"loc_visit_{loc}"
                    )

            # 5.5 容量约束
            total_weight = gp.quicksum(
                s[(loc, layer, sku_id)] * self.problem.id_to_sku[sku_id].weight
                for (loc, layer, sku_id) in s.keys()
            )
            model.addConstr(
                total_weight <= robot.max_weight,
                name="weight_cap"
            )

            # 5.6 堆叠高度约束
            model.addConstr(
                gp.quicksum(phi[(loc, layer)] for (loc, layer) in phi.keys())
                <= robot.max_stack_height,
                name="stack_height"
            )

            # 5.7 路径流守恒
            for loc in locations:
                # 流入 = 流出
                if len(locations) > 1:
                    model.addConstr(
                        gp.quicksum(delta[(l1, loc)] for l1 in locations if l1 != loc)
                        == gp.quicksum(delta[(loc, l2)] for l2 in locations if l2 != loc),
                        name=f"flow_{loc}"
                    )

                    # 访问关联
                    model.addConstr(
                        gp.quicksum(delta[(l1, loc)] for l1 in locations if l1 != loc)
                        == psi[loc],
                        name=f"visit_flow_{loc}"
                    )

            # 5.8 MTZ子回路消除
            M = len(locations)
            for loc1 in locations:
                for loc2 in locations:
                    if loc1 != loc2:
                        model.addConstr(
                            pi[loc1] - pi[loc2] + 1 <= M * (1 - delta[(loc1, loc2)]),
                            name=f"mtz_{loc1}_{loc2}"
                        )

            # === 6. 求解 ===
            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                if model.objVal < -1e-6:  # Reduced cost为负
                    # 提取解
                    selected_orders = [o for o in orders if x[o.order_id].X > 0.5]

                    selected_bins = {}
                    for (loc, layer), var in phi.items():
                        if var.X > 0.5:
                            sku_picks = {
                                sku_id: int(s[(loc, layer, sku_id)].X)
                                for (l, ly, sku_id) in s.keys()
                                if l == loc and ly == layer and s[(loc, layer, sku_id)].X > 0.5
                            }
                            if sku_picks:
                                selected_bins[(loc, layer)] = sku_picks

                    # 构建路径
                    route = []
                    for loc in locations:
                        if psi[loc].X > 0.5:
                            route.append((loc, pi[loc].X))
                    route.sort(key=lambda x: x[1])
                    route = [loc for loc, _ in route]

                    # 创建TaskBatch
                    task_batch = TaskBatch(
                        task_id=f"GENERATED_{len(selected_orders)}",
                        main_batch=main_batch
                    )
                    task_batch.orders = selected_orders
                    task_batch.selected_bins = selected_bins
                    task_batch.robot_route = route
                    task_batch.reduced_cost = model.objVal

                    return task_batch

            return None

        except Exception as e:
            print(f"子问题求解出错: {e}")
            return None