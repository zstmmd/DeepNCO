import numpy as np


class Variable:
    def __init__(self, instance=None) -> None:
        """
        存储综合模型各子问题的所有决策变量与固定参数.

        Args:
            instance: OFSProblemDTO 或等价的实例，提供维度信息
        """
        # 主要决策变量
        self.x = np.zeros((instance.node_num, instance.node_num), dtype=int)  # 路径  point i -point j
        self.y = np.zeros((instance.order_num, instance.station_num), dtype=int)  # tote-工作台
        self.z = np.zeros((instance.order_num, instance.station_num), dtype=int) # 订单-工作台

        # 固定参数（用于子问题）
        self.X_fixed = [[] for _ in range(len(instance.main_batch_list))]  # 每个批次固定料箱集合

        # 辅助决策变量
        self.Q = np.zeros((instance.node_num))    #通常用于车辆路径问题（VRP）的子回路消除约束中
        self.passX = np.zeros((instance.node_num, instance.robot_num), dtype=int)   #passX[i, k] = 1 表示第 k 个机器人访问（经过）了节点 i；否则为 0
        self.a1 = np.zeros((instance.order_num, instance.order_num), dtype=int)   #处理订单ij在工作台的加工顺序的变量
        self.b1 = np.zeros((instance.order_num, instance.order_num), dtype=int)
        self.c1 = np.zeros((instance.order_num, instance.order_num), dtype=int)
        self.d1 = np.zeros((instance.order_num, instance.order_num), dtype=int)
        self.f = np.zeros((instance.order_num, instance.order_num, instance.station_num), dtype=int)    #订单在工作台的服务顺序变量

        # 时间决策变量
        self.tos = np.zeros((instance.order_num))  #订单开始处理时间
        self.toe = np.zeros((instance.order_num))  #订单完成处理时间
        self.T = np.zeros((instance.node_num))  #机器人到达节点的时间
        self.I = np.zeros(instance.n)
        self.Ta = np.zeros((instance.n, instance.P))   #到达
        self.Ts = np.zeros((instance.n, instance.P))
        self.Te = np.zeros((instance.n, instance.P))
        self.FT = 0

        # 状态记录
        self.input_st = [0, 0, 0]
        self.T_list = [0, 0, 0]
        self.spn = 3

    def set_x_variable(self, variable_update_dict):
        """更新车辆路径变量"""
        if 'x' in variable_update_dict:
            self.x = np.array(variable_update_dict['x'])

    def set_y_variable(self, variable_update_dict):
        """更新批次->资源分配变量"""
        if 'y' in variable_update_dict:
            self.y = np.array(variable_update_dict['y'])

    def set_z_variable(self, variable_update_dict):
        """更新订单->批次变量"""
        if 'z' in variable_update_dict:
            self.z = np.array(variable_update_dict['z'])

    def set_passX_variable(self, variable_update_dict):
        """更新passX路径辅助变量"""
        if 'passX' in variable_update_dict:
            self.passX = np.array(variable_update_dict['passX'])

    def set_X_fixed(self, X_fixed_list):
        """
        设置固定料箱集合 X'_b
        X_fixed_list: 列表，长度 = 批次数，每个元素是料箱ID列表
        """
        self.X_fixed = X_fixed_list

    def set_auxiliary_variable(self, variable_update_dict):
        """统一更新所有辅助变量"""
        for key in ['a1', 'b1', 'c1', 'd1', 'Q', 'passX', 'f']:
            if key in variable_update_dict:
                setattr(self, key, np.array(variable_update_dict[key]))

    def set_time_variable(self, variable_update_dict):
        """更新时间相关变量"""
        for key in ['tos', 'toe', 'I', 'Ta', 'Ts', 'Te', 'T', 'FT']:
            if key in variable_update_dict:
                setattr(self, key, np.array(variable_update_dict[key])
                if not isinstance(variable_update_dict[key], (float, int))
                else variable_update_dict[key])

    def __repr__(self):
        return f"<Variable: x={self.x.shape}, y={self.y.shape}, z={self.z.shape}, X_fixed={len(self.X_fixed)}批次>"
