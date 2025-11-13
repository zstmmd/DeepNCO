"""
File: solve_sp4.py
Description: 
    SP4 子问题求解器：Task-Robot Assignment & Sequence Routing
    包含 SP4Variable (变量容器) 和 SolveSP4 (求解器)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Optional
from problemDto.ofs_problem_dto import OFSProblemDTO


class SP4Variable:
    """
    变量容器：用于保存 SP4 (Task-Robot Assignment & Sequence Routing) 的结果
    """
    def __init__(self, B_size: int, K_size: int, Node_size: int, P_size: int):
        """
        Args:
            B_size: 子任务数量
            K_size: 机器人数量
            Node_size: 节点数量 (包含 bins, start, end)
            P_size: 工作站数量
        """
        # 固定输入 (从其他 SP 传入)
        self.u_b = np.zeros((B_size,), dtype=int)            # from SP1
        self.y_bp = np.zeros((B_size, P_size), dtype=int)    # from SP2
        self.x_ikb = {}  # from SP3, dict[(i,k,b)] -> 0/1

        # SP4 决策变量
        self.y_bk = np.zeros((B_size, K_size), dtype=int)            # task-robot assignment
        self.delta_b1b2k = np.zeros((B_size, B_size, K_size), dtype=int)  # sequencing
        self.gamma_ijkb = np.zeros((Node_size, Node_size, K_size, B_size), dtype=int)  # routing
        
        self.T_robot_start = np.zeros((B_size,), dtype=float)
        self.T_robot_end = np.zeros((B_size,), dtype=float)

        # 辅助变量
        self.w_bkp = np.zeros((B_size, K_size, P_size), dtype=int)

    def fix_from_sp1(self, u_b_array):
        """从 SP1 固定 u_b"""
        self.u_b = np.array(u_b_array, dtype=int)

    def fix_from_sp2(self, y_bp_array):
        """从 SP2 固定工作站分配"""
        self.y_bp = np.array(y_bp_array, dtype=int)

    def fix_from_sp3(self, x_ikb_dict):
        """从 SP3 固定料箱选择"""
        self.x_ikb = dict(x_ikb_dict)

    def set_solution(self, y_bk_vars, delta_vars, gamma_vars, 
                     T_start_vars, T_end_vars, w_vars,
                     B_indices, K_indices, Node_indices, P_indices):
        """从 Gurobi 解中提取结果"""
        # y_bk
        for b in B_indices:
            for k in K_indices:
                self.y_bk[b, k] = int(y_bk_vars[b, k].X > 0.5)

        # delta
        for b1 in B_indices:
            for b2 in B_indices:
                for k in K_indices:
                    self.delta_b1b2k[b1, b2, k] = int(delta_vars[b1, b2, k].X > 0.5)

        # gamma
        for i in Node_indices:
            for j in Node_indices:
                for k in K_indices:
                    for b in B_indices:
                        self.gamma_ijkb[i, j, k, b] = int(gamma_vars[i, j, k, b].X > 0.5)

        # 时间
        for b in B_indices:
            self.T_robot_start[b] = T_start_vars[b].X
            self.T_robot_end[b] = T_end_vars[b].X

        # w_bkp
        for b in B_indices:
            for k in K_indices:
                for p in P_indices:
                    self.w_bkp[b, k, p] = int(w_vars[b, k, p].X > 0.5)


class SolveSP4:
    """
    求解子问题4：Task-Robot Assignment & Sequence Routing
    """
    def __init__(self, problem_dto: OFSProblemDTO):
        self.problem_dto = problem_dto
        self.M = 100000  # Big-M

    def solve(self, 
              sp1_var,  # SP1Variable 实例
              sp2_var,  # SP2Variable 实例
              sp3_var,  # SP3Variable 实例
              time_limit: int = 3600) -> Optional[SP4Variable]:
        """
        求解 SP4
        
        Args:
            sp1_var: SP1 的变量对象
            sp2_var: SP2 的变量对象
            sp3_var: SP3 的变量对象
            time_limit: 求解时间限制(秒)
            
        Returns:
            SP4Variable 实例，包含求解结果；失败返回 None
        """
        print("=" * 60)
        print("开始求解 SP4: Task-Robot Assignment & Routing")
        print("=" * 60)

        # 提取集合
        B_size = len(sp1_var.u_b)
        B_indices = [b for b in range(B_size) if sp1_var.u_b[b] > 0.5]  # 只考虑激活的任务
        
        K_size = len(self.problem_dto.robot_list)
        K_indices = list(range(K_size))

        I_size = len(self.problem_dto.tote_list)
        I_indices = list(range(I_size))

        P_size = len(self.problem_dto.station_list)
        P_indices = list(range(P_size))

        # 构造节点集合 Nodes = I ∪ {Start_k} ∪ {End_p}
        # Start_k: robot k 的起点
        # End_p: 工作站 p 的终点
        Nodes_indices = list(I_indices)  # bins
        start_node_offset = I_size
        end_node_offset = I_size + K_size

        for k in K_indices:
            Nodes_indices.append(start_node_offset + k)  # Start_k
        for p in P_indices:
            Nodes_indices.append(end_node_offset + p)    # End_p

        Node_size = len(Nodes_indices)

        print(f"问题规模: {len(B_indices)} 个激活子任务, {K_size} 个机器人, "
              f"{I_size} 个料箱, {P_size} 个工作站")
        print(f"节点总数: {Node_size} (包含 {I_size} bins + {K_size} starts + {P_size} ends)")

        # 创建 Gurobi 模型
        m = gp.Model("SP4_Task_Robot_Routing")
        m.setParam('OutputFlag', 1)
        m.setParam('TimeLimit', time_limit)

        # ==================== 决策变量 ====================
        # y_bk: 子任务 b 分配给机器人 k
        y_bk = m.addVars(B_indices, K_indices, vtype=GRB.BINARY, name="y_bk")

        # delta: 机器人 k 上任务 b1 在 b2 之前
        delta = m.addVars(B_indices, B_indices, K_indices, vtype=GRB.BINARY, name="delta")

        # gamma: 机器人 k 执行任务 b 时从节点 i 到 j
        gamma = m.addVars(Nodes_indices, Nodes_indices, K_indices, B_indices, 
                         vtype=GRB.BINARY, name="gamma")

        # 时间变量
        T_start = m.addVars(B_indices, vtype=GRB.CONTINUOUS, lb=0, name="T_start")
        T_end = m.addVars(B_indices, vtype=GRB.CONTINUOUS, lb=0, name="T_end")

        # 辅助变量 w_bkp (线性化用)
        w_bkp = m.addVars(B_indices, K_indices, P_indices, vtype=GRB.BINARY, name="w_bkp")

        # ==================== 目标函数 ====================
        # 最小化最大完成时间
        FT = m.addVar(vtype=GRB.CONTINUOUS, name="FT")
        m.addConstrs((FT >= T_end[b] for b in B_indices), name="max_finish_time")
        m.setObjective(FT, GRB.MINIMIZE)

        # ==================== 约束条件 ====================
        
        # (1) 每个激活任务必须分配给一个机器人 (eq:task_assign_robot)
        m.addConstrs(
            (gp.quicksum(y_bk[b, k] for k in K_indices) == sp1_var.u_b[b] 
             for b in B_indices),
            name="task_assign_robot"
        )

        # (2) 任务顺序约束 (eq:robot_schedule)
        m.addConstrs(
            (T_start[b2] >= T_end[b1] - self.M * (1 - delta[b1, b2, k])
             for k in K_indices for b1 in B_indices for b2 in B_indices if b1 != b2),
            name="robot_schedule"
        )

        # (3) 流平衡约束 - 出度 (eq:robot_flow_out)
        # 每个任务 b 必须有唯一后继（或到虚拟终点）
        virtual_end_nodes = [end_node_offset + p for p in P_indices]
        m.addConstrs(
            (gp.quicksum(delta[b, b2, k] for b2 in B_indices if b2 != b) 
             + gp.quicksum(delta[b, -1, k] for _ in range(1))  # 虚拟终点
             == y_bk[b, k]
             for b in B_indices for k in K_indices),
            name="robot_flow_out"
        )

        # (4) 流平衡约束 - 入度 (eq:robot_flow_in)
        m.addConstrs(
            (gp.quicksum(delta[b1, b, k] for b1 in B_indices if b1 != b)
             + gp.quicksum(delta[-1, b, k] for _ in range(1))  # 虚拟起点
             == y_bk[b, k]
             for b in B_indices for k in K_indices),
            name="robot_flow_in"
        )

        # (5) 路线起点约束 (eq:route_start)
        # 机器人 k 从 Start_k 出发
        m.addConstrs(
            (gp.quicksum(gamma[start_node_offset + k, j, k, b] 
                        for j in Nodes_indices if j != start_node_offset + k)
             == y_bk[b, k]
             for k in K_indices for b in B_indices),
            name="route_start"
        )

        # (6) 路线料箱入度 (eq:route_in)
        m.addConstrs(
            (gp.quicksum(gamma[j, i, k, b] for j in Nodes_indices if j != i)
             == sp3_var.x_ikb.get((i, k, b), 0)
             for i in I_indices for k in K_indices for b in B_indices),
            name="route_in"
        )

        # (7) 路线料箱出度 (eq:route_out)
        m.addConstrs(
            (gp.quicksum(gamma[i, j, k, b] for j in Nodes_indices if j != i)
             == sp3_var.x_ikb.get((i, k, b), 0)
             for i in I_indices for k in K_indices for b in B_indices),
            name="route_out"
        )

        # (8) 路线终点约束的线性化 (eq:route_end_linear_a, b)
        # w_bkp = y_bk[b,k] AND y_bp[b,p]
        m.addConstrs(
            (w_bkp[b, k, p] <= y_bk[b, k]
             for b in B_indices for k in K_indices for p in P_indices),
            name="w_link_a"
        )
        m.addConstrs(
            (w_bkp[b, k, p] <= sp2_var.y_bp[b, p]
             for b in B_indices for k in K_indices for p in P_indices),
            name="w_link_b"
        )
        m.addConstrs(
            (w_bkp[b, k, p] >= y_bk[b, k] + sp2_var.y_bp[b, p] - 1
             for b in B_indices for k in K_indices for p in P_indices),
            name="w_link_c"
        )

        # (9) 路线到达工作站 (eq:route_end)
        m.addConstrs(
            (gp.quicksum(gamma[i, end_node_offset + p, k, b] 
                        for i in Nodes_indices if i != end_node_offset + p)
             == w_bkp[b, k, p]
             for p in P_indices for k in K_indices for b in B_indices),
            name="route_end"
        )

        # (10) 机器人完成时间约束 (eq:robot_time)
        # T_end >= T_start + 路径时间 + 挖掘时间(ET)
        t_matrix = self.problem_dto.map.time_matrix  # 节点之间的旅行时间矩阵
        excavation_time = self._get_excavation_time_dict()    # 料箱挖取时间字典: ET(i)

        m.addConstrs(
            (T_end[b] >= T_start[b] 
             + gp.quicksum(gamma[i, j, k, b] * t_matrix[i][j]
                           for i in Nodes_indices for j in Nodes_indices if i != j)
             + gp.quicksum(sp3_var.x_ikb.get((i, k, b), 0) * excavation_time.get(i, 0)
                           for i in I_indices)
             for b in B_indices for k in K_indices),
            name="robot_time"
        )

        # ================== 模型构建完成 ==================
        print("SP4 模型构建完成，开始调用 Gurobi 求解...")
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"SP4 求解成功，最优目标值: {m.ObjVal}")
            # 创建变量容器
            sp4_var = SP4Variable(B_size, K_size, Node_size, P_size)
            # 固定输入部分
            sp4_var.fix_from_sp1(sp1_var.u_b)
            sp4_var.fix_from_sp2(sp2_var.y_bp)
            sp4_var.fix_from_sp3(sp3_var.x_ikb)
            # 记录解
            sp4_var.set_solution(
                y_bk_vars=y_bk,
                delta_vars=delta,
                gamma_vars=gamma,
                T_start_vars=T_start,
                T_end_vars=T_end,
                w_vars=w_bkp,
                B_indices=B_indices,
                K_indices=K_indices,
                Node_indices=Nodes_indices,
                P_indices=P_indices
            )
            return sp4_var

        elif m.Status == GRB.INFEASIBLE:
            print("[错误] SP4 模型无解 (Infeasible)")
            m.computeIIS()
            m.write("sp4_infeasible_model.ilp")
            return None
        else:
            print(f"[警告] SP4 求解结束，但未找到最优解。状态: {m.Status}")
            return None

    def _get_excavation_time_dict(self) -> Dict[int, float]:
        """
        根据地图料箱堆叠信息计算每个料箱的挖取时间ET(i)
        公式: min( (MaxLayer(i) - Layer(i)) * T_move_bin, 2 * T_move_bin )
        """
        ET_dict = {}
        T_move_bin = self.problem_dto.config.T_move_bin

        for tote in self.problem_dto.tote_list:
            move_above_cost = (tote.max_layer - tote.layer) * T_move_bin
            move_self_cost = 2 * T_move_bin
            ET_dict[tote.id] = min(move_above_cost, move_self_cost)

        return ET_dict
