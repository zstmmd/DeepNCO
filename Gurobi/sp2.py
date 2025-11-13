'''
目的：最小化所有子任务的完成时间
y{b,p}.  子任务b是否分配给工作站p
β{b1,b2,p}. 工作站p上任务 b1是否先于 b2 执行
辅助时间变量：Ta,Ts,Te
'''
import numpy as np
from typing import List, Dict, Optional, Tuple
from entity.order import Order
from entity.tote import Tote
from entity.robot import Robot
from entity.station import Station

import numpy as np
from typing import List, Dict, Optional, Tuple


class SP2Variable:
    """
    SP2 (工作站分配与调度) 的决策变量容器
    """
    
    def __init__(self, B_size: int, P_size: int, O_size: int):
        """
        Args:
            B_size: 子任务数量
            P_size: 工作站数量
            O_size: 订单数量
        """
        # 主要决策变量
        self.y_bp = np.zeros((B_size, P_size), dtype=int)  # task-station assignment
        self.beta_b1b2p = np.zeros((B_size, B_size, P_size), dtype=int)  # task precedence at station
        
        # 时间变量
        self.T_b_arrival = np.zeros(B_size, dtype=float)   # 任务到达工作站时间
        self.T_b_start = np.zeros(B_size, dtype=float)     # 任务开始处理时间
        self.T_b_end = np.zeros(B_size, dtype=float)       # 任务结束处理时间
        
        # 辅助决策变量
        self.u_ob = np.zeros((O_size, B_size), dtype=int)  # 订单o是否包含在任务b中
        self.w_o_b1b2 = np.zeros((O_size, B_size, B_size), dtype=int)  # 订单o的任务b1和b2关系
        
        # 解的质量指标
        self.objective_value: float = 0.0  # 目标函数值 (makespan)
        self.max_completion_time: float = 0.0  # 最大完成时间
        self.is_feasible: bool = False  # 是否可行解
        
        # 规模信息
        self.B_size = B_size
        self.P_size = P_size
        self.O_size = O_size
        
        # 统计信息
        self.station_workloads: List[int] = [0] * P_size  # 各工作站任务数
        self.station_utilization: List[float] = [0.0] * P_size  # 各工作站利用率

    def set_solution(self,
                     y_vars,
                     beta_vars,
                     T_arrival_vars,
                     T_start_vars,
                     T_end_vars,
                     u_ob_vars,
                     w_o_b1b2_vars,
                     obj_value: float,
                     active_tasks: List[int]):
        """
        从 Gurobi 求解结果加载变量
        
        Args:
            y_vars: Gurobi y[b,p] 变量
            beta_vars: Gurobi beta[b1,b2,p] 变量
            T_arrival_vars: 到达时间变量
            T_start_vars: 开始时间变量
            T_end_vars: 结束时间变量
            u_ob_vars: u[o,b] 辅助变量
            w_o_b1b2_vars: w[o,b1,b2] 辅助变量
            obj_value: 目标函数值
            active_tasks: 激活的任务列表
        """
        # 提取 y_bp (任务-工作站分配)
        for b in range(self.B_size):
            for p in range(self.P_size):
                try:
                    self.y_bp[b, p] = int(y_vars[b, p].X > 0.5)
                except:
                    self.y_bp[b, p] = 0
        
        # 提取 beta (任务优先级)
        for b1 in range(self.B_size):
            for b2 in range(self.B_size):
                if b1 != b2:
                    for p in range(self.P_size):
                        try:
                            self.beta_b1b2p[b1, b2, p] = int(beta_vars[b1, b2, p].X > 0.5)
                        except:
                            self.beta_b1b2p[b1, b2, p] = 0
        
        # 提取时间变量
        for b in active_tasks:
            try:
                self.T_b_arrival[b] = float(T_arrival_vars[b].X)
                self.T_b_start[b] = float(T_start_vars[b].X)
                self.T_b_end[b] = float(T_end_vars[b].X)
            except:
                self.T_b_arrival[b] = 0.0
                self.T_b_start[b] = 0.0
                self.T_b_end[b] = 0.0
        
        # 提取辅助变量
        for o in range(self.O_size):
            for b in range(self.B_size):
                try:
                    self.u_ob[o, b] = int(u_ob_vars[o, b].X > 0.5)
                except:
                    pass
        
        for o in range(self.O_size):
            for b1 in range(self.B_size):
                for b2 in range(self.B_size):
                    if b1 != b2:
                        try:
                            self.w_o_b1b2[o, b1, b2] = int(w_o_b1b2_vars[o, (b1, b2)].X > 0.5)
                        except:
                            pass
        
        # 设置解的质量
        self.objective_value = obj_value
        self.max_completion_time = float(np.max(self.T_b_end[active_tasks])) if active_tasks else 0.0
        self.is_feasible = True
        
        # 计算统计信息
        self._compute_statistics()

    def _compute_statistics(self):
        """计算统计信息"""
        # 工作站工作量
        for p in range(self.P_size):
            self.station_workloads[p] = int(np.sum(self.y_bp[:, p]))
        
        # 工作站利用率 (工作时间 / 总时间)
        if self.max_completion_time > 0:
            for p in range(self.P_size):
                busy_time = 0.0
                for b in range(self.B_size):
                    if self.y_bp[b, p] > 0:
                        busy_time += (self.T_b_end[b] - self.T_b_start[b])
                self.station_utilization[p] = busy_time / self.max_completion_time

    def get_task_station(self, task_id: int) -> Optional[int]:
        """
        获取任务分配的工作站
        
        Args:
            task_id: 任务ID
            
        Returns:
            工作站ID，如果未分配返回 None
        """
        for p in range(self.P_size):
            if self.y_bp[task_id, p] > 0:
                return p
        return None

    def get_station_schedule(self, station_id: int) -> List[Tuple[int, float, float, float]]:
        """
        获取某工作站的任务调度序列
        
        Args:
            station_id: 工作站ID
            
        Returns:
            [(task_id, arrival_time, start_time, end_time), ...] 按开始时间排序
        """
        schedule = []
        for b in range(self.B_size):
            if self.y_bp[b, station_id] > 0:
                schedule.append((
                    b,
                    self.T_b_arrival[b],
                    self.T_b_start[b],
                    self.T_b_end[b]
                ))
        schedule.sort(key=lambda x: x[2])  # 按开始时间排序
        return schedule

    def get_task_wait_time(self, task_id: int) -> float:
        """
        获取任务的等待时间（开始时间 - 到达时间）
        
        Args:
            task_id: 任务ID
            
        Returns:
            等待时间
        """
        return self.T_b_start[task_id] - self.T_b_arrival[task_id]

    def get_task_processing_time(self, task_id: int) -> float:
        """
        获取任务的处理时间（结束时间 - 开始时间）
        
        Args:
            task_id: 任务ID
            
        Returns:
            处理时间
        """
        return self.T_b_end[task_id] - self.T_b_start[task_id]

    def get_order_completion_span(self, order_id: int) -> Tuple[float, float]:
        """
        获取订单的完成时间跨度（最早开始 - 最晚结束）
        
        Args:
            order_id: 订单ID
            
        Returns:
            (earliest_start, latest_end)
        """
        order_tasks = [b for b in range(self.B_size) if self.u_ob[order_id, b] > 0]
        if not order_tasks:
            return (0.0, 0.0)
        
        earliest_start = min(self.T_b_start[b] for b in order_tasks)
        latest_end = max(self.T_b_end[b] for b in order_tasks)
        return (earliest_start, latest_end)

    def validate_kitting_window(self, order_id: int, max_window: float) -> bool:
        """
        验证订单是否满足齐套窗口约束
        
        Args:
            order_id: 订单ID
            max_window: 最大允许时间窗口
            
        Returns:
            是否满足约束
        """
        earliest, latest = self.get_order_completion_span(order_id)
        return (latest - earliest) <= max_window

    def get_station_idle_time(self, station_id: int) -> float:
        """
        获取工作站的空闲时间
        
        Args:
            station_id: 工作站ID
            
        Returns:
            空闲时间
        """
        if self.max_completion_time == 0:
            return 0.0
        
        busy_time = 0.0
        for b in range(self.B_size):
            if self.y_bp[b, station_id] > 0:
                busy_time += self.get_task_processing_time(b)
        
        return self.max_completion_time - busy_time

    def summary(self) -> str:
        """返回变量摘要信息"""
        avg_utilization = np.mean(self.station_utilization) if self.P_size > 0 else 0.0
        
        summary_str = f"""
                SP2Variable Summary:
                ===================
                Makespan: {self.max_completion_time:.2f}
                Objective Value: {self.objective_value:.2f}
                Feasible: {self.is_feasible}

                Station Statistics:
                -------------------
                """
        for p in range(self.P_size):
            summary_str += f"  Station {p}: Tasks={self.station_workloads[p]}, Utilization={self.station_utilization[p]:.2%}\n"
        
        summary_str += f"\nAverage Station Utilization: {avg_utilization:.2%}\n"
        
        return summary_str

    def export_schedule(self) -> Dict[str, any]:
        """
        导出调度结果为字典格式
        
        Returns:
            包含所有调度信息的字典
        """
        schedule_dict = {
            'makespan': self.max_completion_time,
            'objective': self.objective_value,
            'stations': []
        }
        
        for p in range(self.P_size):
            station_info = {
                'station_id': p,
                'workload': self.station_workloads[p],
                'utilization': self.station_utilization[p],
                'schedule': self.get_station_schedule(p)
            }
            schedule_dict['stations'].append(station_info)
        
        return schedule_dict
'''
File: solve_sp2.py
Project: OFS_Integrated_Model
Description: 
----------
求解子问题2: 工作站分配与调度
----------
'''

import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional, Tuple
from problemDto.ofs_problem_dto import OFSProblemDTO
from solver.sp2_variable import SP2Variable
from config.ofs_config import OFSConfig
from entity.order import Order


class SolveSP2:
    """
    求解子问题2: 工作站分配与调度 (Workstation Assignment & Scheduling)
    
    目标: 最小化所有子任务的最大完成时间
    
    输入:
        - SP1: u_b (任务激活状态), z_{o,s,b} (SKU分配)
        - SP4: T_b^{robot_end} (机器人完成时间)
    
    输出:
        - y_{b,p}: 任务-工作站分配
        - T_b^a, T_b^s, T_b^e: 任务时间调度
    """

    def __init__(self, problem_dto: OFSProblemDTO, config: OFSConfig = None):
        """
        初始化求解器
        
        Args:
            problem_dto: 问题实例
            config: 配置参数
        """
        self.problem_dto = problem_dto
        self.config = config or OFSConfig
        self.M = 100000  # Big-M 常数
        
        # 从 problem_dto 提取基本信息
        self.stations = problem_dto.station_list
        self.orders = problem_dto.order_list
        
        print(f"[SP2] 初始化完成: {len(self.stations)} 个工作站, {len(self.orders)} 个订单")

    def solve(
        self,
        active_tasks: List[int],  # u_b from SP1 (激活的任务ID列表)
        task_sku_assignment: Dict[Tuple[int, int, int], int],  # z_{o,s,b} from SP1: {(o,s,b): qty}
        robot_end_times: Dict[int, float],  # T_b^{robot_end} from SP4: {task_id: time}
        time_limit: int = 3600,
        output_flag: bool = True
    ) -> Optional[SP2Variable]:
        """
        求解 SP2 模型
        
        Args:
            active_tasks: 激活的子任务ID列表
            task_sku_assignment: SKU分配结果 {(order_id, sku_id, task_id): quantity}
            robot_end_times: 机器人完成时间 {task_id: completion_time}
            time_limit: 求解时间限制(秒)
            output_flag: 是否显示Gurobi求解过程
            
        Returns:
            SP2Variable 实例，失败返回 None
        """
        
        print(f"\n{'='*60}")
        print(f"开始求解 SP2 - 工作站分配与调度")
        print(f"{'='*60}")
        
        # 1. 数据准备和验证
        if not active_tasks:
            print("⚠️  警告: 没有激活的子任务需要分配")
            return None
        
        B = active_tasks  # 激活的任务列表
        P = list(range(len(self.stations)))  # 工作站索引
        O = list(range(len(self.orders)))  # 订单索引
        
        print(f"📊 问题规模:")
        print(f"   - 激活任务数: {len(B)}")
        print(f"   - 工作站数: {len(P)}")
        print(f"   - 订单数: {len(O)}")
        
        # 2. 预处理：计算任务的处理时间
        task_processing_times = self._compute_task_processing_times(
            active_tasks, 
            task_sku_assignment
        )
        
        # 3. 预处理：构建订单-任务关联关系
        order_task_matrix = self._build_order_task_matrix(
            task_sku_assignment, 
            active_tasks
        )
        
        # 4. 创建 Gurobi 模型
        print("\n🔧 构建 Gurobi 模型...")
        m = gp.Model("SP2_Workstation_Assignment_Scheduling")
        m.setParam('OutputFlag', 1 if output_flag else 0)
        m.setParam('TimeLimit', time_limit)
        m.setParam('MIPGap', 0.01)  # 1% gap
        
        # 5. 决策变量
        print("   添加决策变量...")
        
        # y[b,p]: 任务b分配给工作站p
        y = m.addVars(B, P, vtype=GRB.BINARY, name="y")
        
        # beta[b1,b2,p]: 工作站p上任务b1先于b2
        beta = m.addVars(
            [(b1, b2, p) for b1 in B for b2 in B for p in P if b1 != b2],
            vtype=GRB.BINARY, 
            name="beta"
        )
        
        # 时间变量
        T_arrival = m.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="T_arrival")
        T_start = m.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="T_start")
        T_end = m.addVars(B, vtype=GRB.CONTINUOUS, lb=0, name="T_end")
        
        # 辅助变量
        u_ob = m.addVars(O, B, vtype=GRB.BINARY, name="u_ob")  # 订单o是否在任务b中
        w_o_b1b2 = m.addVars(
            [(o, b1, b2) for o in O for b1 in B for b2 in B if b1 != b2],
            vtype=GRB.BINARY, 
            name="w"
        )
        
        # 目标函数变量
        FT = m.addVar(vtype=GRB.CONTINUOUS, name="FT")
        
        # 6. 目标函数 (eq:obsp1)
        print("   设置目标函数...")
        m.addConstrs((FT >= T_end[b] for b in B), name="makespan")
        m.setObjective(FT, GRB.MINIMIZE)
        
        # 7. 约束条件
        print("   添加约束条件...")
        
        # (C1) 每个激活任务必须分配到恰好一个工作站 (eq:task_assign_ws)
        m.addConstrs(
            (gp.quicksum(y[b, p] for p in P) == 1 for b in B),
            name="C1_task_assignment"
        )
        print(f"   ✓ C1: 任务分配约束 ({len(B)} 个)")
        
        # (C2) 到达时间 >= 机器人完成时间 (eq:ws_arrival)
        for b in B:
            robot_time = robot_end_times.get(b, 0.0)
            for p in P:
                m.addConstr(
                    T_arrival[b] >= robot_time - self.M * (1 - y[b, p]),
                    name=f"C2_arrival_{b}_{p}"
                )
        print(f"   ✓ C2: 到达时间约束 ({len(B) * len(P)} 个)")
        
        # (C3) 开始时间 >= 到达时间 (eq:ws_start_after_arrival)
        m.addConstrs(
            (T_start[b] >= T_arrival[b] for b in B),
            name="C3_start_after_arrival"
        )
        print(f"   ✓ C3: 开始时间约束 ({len(B)} 个)")
        
        # (C4) 缓冲区等待时间限制 (eq:ws_buffer_wait)
        buffer_wait_limit = getattr(self.config, 'BUFFER_WAIT_TIME', 300)
        for b in B:
            for p in P:
                m.addConstr(
                    T_start[b] - T_arrival[b] <= buffer_wait_limit * y[b, p],
                    name=f"C4_buffer_wait_{b}_{p}"
                )
        print(f"   ✓ C4: 缓冲区等待约束 ({len(B) * len(P)} 个)")
        
        # (C5) 结束时间计算 (eq:ws_time)
        T_disassy = getattr(self.config, 'BIN_DISASSEMBLY_TIME', 10)
        for b in B:
            processing_time = task_processing_times.get(b, 0)
            m.addConstr(
                T_end[b] >= T_start[b] + T_disassy + processing_time,
                name=f"C5_end_time_{b}"
            )
        print(f"   ✓ C5: 结束时间约束 ({len(B)} 个)")
        
        # (C6) 同一工作站上的任务顺序约束 (eq:ws_schedule)
        for b1 in B:
            for b2 in B:
                if b1 != b2:
                    for p in P:
                        m.addConstr(
                            T_start[b2] >= T_end[b1] - self.M * (1 - beta[b1, b2, p]),
                            name=f"C6_precedence_{b1}_{b2}_{p}"
                        )
        print(f"   ✓ C6: 任务顺序约束 ({len(B) * (len(B)-1) * len(P)} 个)")
        
        # (C7) 流平衡约束 - 流出 (eq:ws_flow_out)
        for b in B:
            for p in P:
                m.addConstr(
                    gp.quicksum(beta[b, b2, p] for b2 in B if b2 != b) == y[b, p],
                    name=f"C7_flow_out_{b}_{p}"
                )
        print(f"   ✓ C7: 流出约束 ({len(B) * len(P)} 个)")
        
        # (C8) 流平衡约束 - 流入 (eq:ws_flow_in)
        for b in B:
            for p in P:
                m.addConstr(
                    gp.quicksum(beta[b1, b, p] for b1 in B if b1 != b) == y[b, p],
                    name=f"C8_flow_in_{b}_{p}"
                )
        print(f"   ✓ C8: 流入约束 ({len(B) * len(P)} 个)")
        
        # (C9) 链接订单-任务关系 (eq:link_u_ob)
        for o in O:
            for b in B:
                # 如果任务b包含订单o的任何SKU，则 u_ob[o,b] = 1
                if order_task_matrix.get((o, b), 0) > 0:
                    m.addConstr(u_ob[o, b] == 1, name=f"C9_u_ob_{o}_{b}")
                else:
                    m.addConstr(u_ob[o, b] == 0, name=f"C9_u_ob_{o}_{b}_zero")
        print(f"   ✓ C9: 订单-任务链接约束 ({len(O) * len(B)} 个)")
        
        # (C10) 链接 w 变量 (eq:link_w_o_b1_b2)
        for o in O:
            for b1 in B:
                for b2 in B:
                    if b1 != b2:
                        m.addConstr(
                            w_o_b1b2[o, b1, b2] >= u_ob[o, b1] + u_ob[o, b2] - 1,
                            name=f"C10_w_{o}_{b1}_{b2}"
                        )
        print(f"   ✓ C10: w变量链接约束 ({len(O) * len(B) * (len(B)-1)} 个)")
        
        # (C11 & C12) 齐套窗口约束 
        kit_window = getattr(self.config, 'KIT_DELIVERY_WINDOW', 600)
        for o in O:
            for b1 in B:
                for b2 in B:
                    if b1 != b2:
                        # T_end[b1] - T_end[b2] <= kit_window + M*(1 - w)
                        m.addConstr(
                            T_end[b1] - T_end[b2] <= 
                            kit_window + self.M * (1 - w_o_b1b2[o, b1, b2]),
                            name=f"C11_kit_window_a_{o}_{b1}_{b2}"
                        )
                        # T_end[b2] - T_end[b1] <= kit_window + M*(1 - w)
                        m.addConstr(
                            T_end[b2] - T_end[b1] <= 
                            kit_window + self.M * (1 - w_o_b1b2[o, b1, b2]),
                            name=f"C12_kit_window_b_{o}_{b1}_{b2}"
                        )
        print(f"   ✓ C11-C12: 齐套窗口约束 ({2 * len(O) * len(B) * (len(B)-1)} 个)")
        
        # 8. 优化求解
        print(f"\n🚀 开始求解...")
        print(f"   时间限制: {time_limit} 秒")
        m.update()
        m.optimize()
        
        # 9. 解析结果
        return self._parse_solution(
            m, y, beta, T_arrival, T_start, T_end, 
            u_ob, w_o_b1b2, FT, 
            active_tasks, O
        )

    def _compute_task_processing_times(
            self, 
            active_tasks: List[int],
            task_sku_assignment: Dict[Tuple[int, int, int], int]
        ) -> Dict[int, float]:
        """
        计算每个任务的处理时间
        
        处理时间 = Σ(每个SKU的拣选时间)
        
        Args:
            active_tasks: 激活的任务列表
            task_sku_assignment: {(order_id, sku_id, task_id): quantity}
            
        Returns:
            {task_id: processing_time}
        """
        print("\n📐 计算任务处理时间...")
            
        task_times = {}
        pick_time_per_unit = getattr(self.config, 'PICK_TIME_PER_UNIT', 2.0)
            
        for b in active_tasks:
            total_time = 0.0
            sku_count = 0
            
            # 遍历所有分配给任务b的SKU
            for (o, s, task_b), qty in task_sku_assignment.items():
                if task_b == b and qty > 0:
                    # 获取SKU信息
                    sku = self.problem_dto.id_to_sku.get(s)
                    if sku:
                        # 拣选时间 = 数量 × 单位时间
                        pick_time = qty * pick_time_per_unit
                        total_time += pick_time
                        sku_count += 1
            
            task_times[b] = total_time
            print(f"   任务 {b}: {sku_count} 种SKU, 处理时间 = {total_time:.2f}s")
        
        return task_times

    def _build_order_task_matrix(
        self,
        task_sku_assignment: Dict[Tuple[int, int, int], int],
        active_tasks: List[int]
    ) -> Dict[Tuple[int, int], int]:
        """
        构建订单-任务关联矩阵
        
        如果任务b包含订单o的任何SKU，则 matrix[(o,b)] = 1
        
        Args:
            task_sku_assignment: {(order_id, sku_id, task_id): quantity}
            active_tasks: 激活的任务列表
            
        Returns:
            {(order_id, task_id): has_items}
        """
        print("\n🔗 构建订单-任务关联矩阵...")
        
        matrix = {}
        order_set = set()
        
        for (o, s, b), qty in task_sku_assignment.items():
            if b in active_tasks and qty > 0:
                matrix[(o, b)] = 1
                order_set.add(o)
        
        # 确保所有 (o, b) 组合都有值
        for o in order_set:
            for b in active_tasks:
                if (o, b) not in matrix:
                    matrix[(o, b)] = 0
        
        num_associations = sum(1 for v in matrix.values() if v > 0)
        print(f"   共 {num_associations} 个订单-任务关联关系")
        
        return matrix

    def _parse_solution(
        self,
        model: gp.Model,
        y_vars,
        beta_vars,
        T_arrival_vars,
        T_start_vars,
        T_end_vars,
        u_ob_vars,
        w_o_b1b2_vars,
        FT_var,
        active_tasks: List[int],
        orders: List[int]
    ) -> Optional[SP2Variable]:
        """
        解析 Gurobi 求解结果并创建 SP2Variable
        
        Args:
            model: Gurobi 模型
            *_vars: 各决策变量
            active_tasks: 激活的任务列表
            orders: 订单列表
            
        Returns:
            SP2Variable 实例，失败返回 None
        """
        
        print(f"\n{'='*60}")
        print(f"求解完成")
        print(f"{'='*60}")
        
        if model.Status == GRB.OPTIMAL:
            print(f"✅ 状态: OPTIMAL")
            print(f"📊 目标值 (Makespan): {model.ObjVal:.2f}")
            
            # 创建变量容器
            max_task_id = max(active_tasks) if active_tasks else 0
            B_size = max_task_id + 1
            P_size = len(self.stations)
            O_size = len(orders)
            
            sp2_var = SP2Variable(B_size=B_size, P_size=P_size, O_size=O_size)
            
            # 提取解
            try:
                sp2_var.set_solution(
                    y_vars=y_vars,
                    beta_vars=beta_vars,
                    T_arrival_vars=T_arrival_vars,
                    T_start_vars=T_start_vars,
                    T_end_vars=T_end_vars,
                    u_ob_vars=u_ob_vars,
                    w_o_b1b2_vars=w_o_b1b2_vars,
                    obj_value=model.ObjVal,
                    active_tasks=active_tasks
                )
                
                # 打印详细结果
                self._print_solution_summary(sp2_var, active_tasks)
                
                return sp2_var
                
            except Exception as e:
                print(f"❌ 解析解时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        elif model.Status == GRB.TIME_LIMIT:
            print(f"⏱️  状态: TIME_LIMIT")
            print(f"📊 当前目标值: {model.ObjVal:.2f}")
            print(f"📊 最优界: {model.ObjBound:.2f}")
            print(f"📊 Gap: {model.MIPGap*100:.2f}%")
            
            # 即使超时，也尝试提取当前解
            if model.SolCount > 0:
                print("✓ 找到可行解，尝试提取...")
                
                max_task_id = max(active_tasks) if active_tasks else 0
                B_size = max_task_id + 1
                P_size = len(self.stations)
                O_size = len(orders)
                
                sp2_var = SP2Variable(B_size=B_size, P_size=P_size, O_size=O_size)
                
                try:
                    sp2_var.set_solution(
                        y_vars=y_vars,
                        beta_vars=beta_vars,
                        T_arrival_vars=T_arrival_vars,
                        T_start_vars=T_start_vars,
                        T_end_vars=T_end_vars,
                        u_ob_vars=u_ob_vars,
                        w_o_b1b2_vars=w_o_b1b2_vars,
                        obj_value=model.ObjVal,
                        active_tasks=active_tasks
                    )
                    
                    self._print_solution_summary(sp2_var, active_tasks)
                    return sp2_var
                    
                except Exception as e:
                    print(f"❌ 解析解时发生错误: {str(e)}")
                    return None
            else:
                print("❌ 未找到可行解")
                return None
        
        elif model.Status == GRB.INFEASIBLE:
            print(f"❌ 状态: INFEASIBLE (无可行解)")
            print("正在计算IIS (不可行子系统)...")
            
            try:
                model.computeIIS()
                iis_file = "sp2_infeasible_model.ilp"
                model.write(iis_file)
                print(f"已将IIS写入文件: {iis_file}")
                
                # 打印部分冲突约束
                print("\n冲突约束示例:")
                count = 0
                for constr in model.getConstrs():
                    if constr.IISConstr and count < 10:
                        print(f"  - {constr.ConstrName}")
                        count += 1
                        
            except Exception as e:
                print(f"计算IIS时出错: {str(e)}")
            
            return None
        
        elif model.Status == GRB.UNBOUNDED:
            print(f"❌ 状态: UNBOUNDED (无界)")
            return None
        
        else:
            print(f"⚠️  状态: {model.Status} (未知状态)")
            return None

    def _print_solution_summary(self, sp2_var: SP2Variable, active_tasks: List[int]):
        """
        打印解的详细摘要
        
        Args:
            sp2_var: SP2变量容器
            active_tasks: 激活的任务列表
        """
        print("\n" + "="*60)
        print("解决方案摘要")
        print("="*60)
        
        # 1. 任务分配统计
        print("\n📋 任务分配到工作站:")
        for p in range(sp2_var.P_size):
            tasks_at_station = [b for b in active_tasks if sp2_var.get_task_station(b) == p]
            if tasks_at_station:
                print(f"\n  工作站 {p}:")
                print(f"    分配任务数: {len(tasks_at_station)}")
                print(f"    任务列表: {tasks_at_station}")
                
                # 打印该工作站的调度序列
                schedule = sp2_var.get_station_schedule(p)
                if schedule:
                    print(f"    调度序列:")
                    for task_id, arr, start, end in schedule:
                        wait = start - arr
                        proc = end - start
                        print(f"      任务{task_id}: 到达={arr:.1f}, 等待={wait:.1f}, "
                              f"开始={start:.1f}, 处理={proc:.1f}, 结束={end:.1f}")
        
        # 2. 工作站负载平衡
        print(f"\n⚖️  工作站负载:")
        for p in range(sp2_var.P_size):
            workload = sp2_var.station_workloads[p]
            utilization = sp2_var.station_utilization[p]
            idle_time = sp2_var.get_station_idle_time(p)
            print(f"  工作站 {p}: 任务数={workload}, "
                  f"利用率={utilization:.2%}, 空闲时间={idle_time:.1f}s")
        
        # 3. 时间统计
        print(f"\n⏱️  时间统计:")
        print(f"  Makespan (最大完成时间): {sp2_var.max_completion_time:.2f}s")
        
        avg_wait = np.mean([sp2_var.get_task_wait_time(b) for b in active_tasks])
        avg_proc = np.mean([sp2_var.get_task_processing_time(b) for b in active_tasks])
        print(f"  平均等待时间: {avg_wait:.2f}s")
        print(f"  平均处理时间: {avg_proc:.2f}s")
        
        # 4. 齐套窗口验证
        kit_window = getattr(self.config, 'KIT_DELIVERY_WINDOW', 600)
        print(f"\n📦 齐套窗口验证 (限制: {kit_window}s):")
        
        violations = []
        for o in range(sp2_var.O_size):
            earliest, latest = sp2_var.get_order_completion_span(o)
            span = latest - earliest
            is_valid = sp2_var.validate_kitting_window(o, kit_window)
            
            if span > 0:  # 只显示有任务的订单
                status = "✓" if is_valid else "✗"
                print(f"  订单 {o}: {status} 时间跨度={span:.1f}s "
                      f"(开始={earliest:.1f}, 结束={latest:.1f})")
                if not is_valid:
                    violations.append(o)
        
        if violations:
            print(f"\n  ⚠️  警告: {len(violations)} 个订单违反齐套窗口约束")
        else:
            print(f"\n  ✓ 所有订单满足齐套窗口约束")
        
        print("\n" + "="*60)





