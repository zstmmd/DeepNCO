import sys

from config.ofs_config import OFSConfig

sys.path.append('..')
import gurobipy as gp
import numpy as np
from gurobipy import GRB


class OrderBatchingGurobiModel:
   '''优化订单组批和加工排序'''
    def __init__(self, instance, variable, time_limit=None, init_flag=False):
        """
        子问题1：订单组批（固定 y' 和 X'）
        """
        self.instance = instance  # OFSProblemDTO
        self.variable = variable  # Variable 类实例
        self.time_limit = time_limit
        self.init_flag = init_flag
        self.bigM = 1e5

        # 固定参数
        self.order_list = instance.order_list
        self.main_batches = instance.main_batch_list
        self.skus_list = instance.skus_list
        self.robot_list = instance.robot_list
        self.station_list = instance.station_list

        # 批次数量（从 main_batches 来）
        self.B = len(self.main_batches)
        # 订单数量
        self.O = len(self.order_list)
        # 工作台数量
        self.P = len(self.station_list)

        # 固定y'（批次→机器人&工作台分配）
        self.y_fixed = variable.y
        # 固定的 tote 集合 X'_b
        self.X_fixed = variable.X_fixed
        # 计算到达时间参数
        self.arrival_fixed = self._compute_arrival_times()


    def _compute_arrival_times(self):
        """
        根据固定 y' 和固定料箱集合 X'_b 计算每个批次的到达时间参数
        """
        arrival_times = []

        # 路径用曼哈顿距离 × robot速度
        robot_speed = OFSConfig.ROBOT_SPEED

        for b in range(self.B):
            # 找到分配到该批次的机器人和工作台
            k, p = self._get_robot_station_for_batch(b)

            # 机器人起点位置
            start_point = self.instance.robot_list[k].start_point
            start_idx = start_point.idx

            # 料箱集合
            totes = [self.instance.id_to_tote[t] for t in self.X_fixed[b]]

            # 计算 travel 时间（机器人起点到第一个料箱，再到其他料箱，最后到工作台）
            travel_time = 0.0
            visited_points = []

            # 从起点到第一个tote位置（简单最近点）
            if totes:
                first_tote = totes[0]
                dist = self.instance.map.distance_matrix[start_idx][first_tote.store_point.idx]
                travel_time += dist / robot_speed
                visited_points.append(first_tote.store_point.idx)

                # tote之间的距离
                for t_idx in range(1, len(totes)):
                    prev_point = totes[t_idx - 1].store_point.idx
                    cur_point = totes[t_idx].store_point.idx
                    dist = self.instance.map.distance_matrix[prev_point][cur_point]
                    travel_time += dist / robot_speed
                    visited_points.append(cur_point)

                # 最后到工作台
                station_point_idx = self.instance.station_list[p].point.idx
                dist = self.instance.map.distance_matrix[totes[-1].store_point.idx][station_point_idx]
                travel_time += dist / robot_speed

            # dig 时间
            dig_time = sum((t.max_layer - t.layer) * T_move_bin for t in totes)

            # 假设机器人立即执行任务（没有 δ）→ robot_start_time = 0
            robot_start_time = 0.0
            arrival_times.append(robot_start_time + travel_time + dig_time)

        return arrival_times

    def _get_robot_station_for_batch(self, b):
        """
        从变量 y_fixed 获取批次b分配的机器人k和工作台p
        y_fixed[b][p] = 1 表示批次b分配给工作台p
        """
        p = None
        k = None
        # 变量y是二维(n, P)，这里假设第一维是batch
        for station_idx in range(self.P):
            if self.y_fixed[b, station_idx] == 1:
                p = station_idx
                break
        # 找机器人
        # 这里你的y_fixed可能不是完全的batch→robot映射，需要补充逻辑
        # 暂时假设机器人索引等于batch id（实际需要结合 δ 来确定）
        k = b % len(self.instance.robot_list)
        return k, p

    def build_model(self, model):
        M = self.bigM

        # 决策变量
        z = model.addVars(self.O, self.B, vtype=GRB.BINARY, name="z")
        beta = model.addVars(self.B, self.B, self.P, vtype=GRB.BINARY, name="beta")
        T_start = model.addVars(self.B, vtype=GRB.CONTINUOUS, name="T_start")
        T_end = model.addVars(self.B, vtype=GRB.CONTINUOUS, name="T_end")
        Tmax = model.addVar(vtype=GRB.CONTINUOUS, name="Tmax")

        # 目标函数：最小化最大完工时间
        model.modelSense = GRB.MINIMIZE
        model.setObjective(Tmax)

        # 1. Makespan约束
        for b in range(self.B):
            model.addConstr(Tmax >= T_end[b], name=f"makespan_{b}")

        # 2. 每个主批次内的订单必须且只能属于一个任务批次
        for o in range(self.O):
            model.addConstr(gp.quicksum(z[o, b] for b in range(self.B)) == 1, name=f"order_assign_{o}")

        # 3. 库存满足约束
        for b in range(self.B):
            for sku in self.skus_list:
                demand_sum = gp.quicksum(
                    z[o, b] * self.instance.id_to_order[o].order_product_id_list.count(sku.id)
                    for o in range(self.O)
                )
                supply_sum = gp.quicksum(
                    self.instance.id_to_tote[t].sku_quantity_map.get(sku.id, 0)
                    for t in self.X_fixed[b]
                )
                model.addConstr(demand_sum <= supply_sum, name=f"inv_{b}_{sku.id}")

        # 4. 机器人载重约束
        for b in range(self.B):
            assigned_robot_idx = np.argmax(self.y_fixed[b])  # 找固定分配的机器人id
            max_height = self.instance.robot_list[assigned_robot_idx].max_stack_height
            model.addConstr(len(self.X_fixed[b]) <= max_height, name=f"stack_{b}")

        # 5. 时间链接（到达时间）
        for b in range(self.B):
            model.addConstr(T_start[b] >= self.arrival_fixed[b], name=f"arrival_{b}")

        # 6. 加工时间约束（固定拆垛+变量拣选时间+固定码垛）
        for b in range(self.B):
            # 简化：处理时间 = 固定常数 + SKU总需求数量（依赖z）
            fixed_disassembly = 5
            fixed_assembly = 5
            pick_time = gp.quicksum(
                z[o, b] * len(self.instance.id_to_order[o].order_product_id_list) * 0.5
                for o in range(self.O)
            )
            model.addConstr(T_end[b] >= T_start[b] + fixed_disassembly + pick_time + fixed_assembly,
                            name=f"proc_time_{b}")

        # 7. 工作台加工顺序约束
        for p in range(self.P):
            for b1 in range(self.B):
                for b2 in range(self.B):
                    if b1 != b2:
                        model.addConstr(T_start[b2] >= T_end[b1] - M * (1 - beta[b1, b2, p]),
                                        name=f"seq_{b1}_{b2}_p{p}")

        # 8. 序列流约束
        for p in range(self.P):
            for b in range(self.B):
                model.addConstr(gp.quicksum(beta[b1, b, p] for b1 in range(self.B)) == 1, name=f"flow_in_{b}_p{p}")
                model.addConstr(gp.quicksum(beta[b, b2, p] for b2 in range(self.B)) == 1, name=f"flow_out_{b}_p{p}")

        model.update()
        return z

    def run(self):
        model = gp.Model("OrderBatching")
        z_var = self.build_model(model)

        if self.time_limit:
            model.setParam("TimeLimit", self.time_limit)
        model.setParam("OutputFlag", 1)

        model.optimize()

        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            solution_z = np.array([[z_var[o, b].X for b in range(self.B)] for o in range(self.O)])
            return model.ObjVal, model.ObjBound, solution_z
        else:
            return None, None, None
