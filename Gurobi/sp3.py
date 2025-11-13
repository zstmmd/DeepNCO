

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from entity.tote import Tote


class SP3Variable:
    """
    SP3 (料箱命中) 的决策变量
    
    决策: 为每个任务选择哪些料箱来满足SKU需求
    """
    
    def __init__(self, I_size: int, K_size: int, B_size: int):
        """
        Args:
            I_size: 料箱数量
            K_size: 机器人数量
            B_size: 子任务数量
        """
        # 主要决策变量
        self.x_ikb = np.zeros((I_size, K_size, B_size), dtype=int)  # bin i selected by robot k for task b
        
        # 辅助信息
        self.bin_index_map: Dict[int, int] = {}  # bin_id → index
        self.robot_index_map: Dict[int, int] = {}  # robot_id → index
        
        # 解的质量指标
        self.objective_value: float = 0.0
        self.is_feasible: bool = False
        self.total_bins_used: int = 0
        
        # 规模信息
        self.I_size = I_size
        self.K_size = K_size
        self.B_size = B_size
        
        # 统计信息
        self.robot_bin_loads: List[int] = [0] * K_size  # 每个机器人搬运的料箱数
        self.task_bin_counts: List[int] = [0] * B_size  # 每个任务使用的料箱数
        self.bin_usage: List[int] = [0] * I_size  # 每个料箱被使用次数

    def set_solution(self, 
                     bins: List[Tote],
                     robots: List,
                     x_vars,
                     obj_value: float,
                     active_tasks: List[int]):
        """
        从 Gurobi 求解结果加载变量
        
        Args:
            bins: 料箱列表
            robots: 机器人列表
            x_vars: Gurobi x[i,k,b] 变量
            obj_value: 目标函数值
            active_tasks: 激活的任务列表
        """
        # 建立索引映射
        self.bin_index_map = {bins[i].id: i for i in range(len(bins))}
        self.robot_index_map = {robots[k].robot_id: k for k in range(len(robots))}
        
        # 提取 x_ikb
        for i in range(self.I_size):
            for k in range(self.K_size):
                for b in active_tasks:
                    if b < self.B_size:
                        try:
                            self.x_ikb[i, k, b] = int(x_vars[i, k, b].X > 0.5)
                        except:
                            self.x_ikb[i, k, b] = 0
        
        # 设置解的质量
        self.objective_value = obj_value
        self.is_feasible = True
        self.total_bins_used = int(np.sum(self.x_ikb))
        
        # 计算统计信息
        self._compute_statistics(active_tasks)

    def _compute_statistics(self, active_tasks: List[int]):
        """计算统计信息"""
        # 机器人料箱负载
        for k in range(self.K_size):
            self.robot_bin_loads[k] = int(np.sum(self.x_ikb[:, k, :]))
        
        # 任务料箱数量
        for b in active_tasks:
            if b < self.B_size:
                self.task_bin_counts[b] = int(np.sum(self.x_ikb[:, :, b]))
        
        # 料箱使用频率
        for i in range(self.I_size):
            self.bin_usage[i] = int(np.sum(self.x_ikb[i, :, :]))

    def get_selected_bins(self, robot_id: int, task_id: int) -> List[int]:
        """
        获取机器人k为任务b选择的料箱列表
        
        Args:
            robot_id: 机器人ID (实际ID，非索引)
            task_id: 任务ID
            
        Returns:
            料箱索引列表
        """
        k_idx = self.robot_index_map.get(robot_id, robot_id)
        if task_id >= self.B_size or k_idx >= self.K_size:
            return []
        
        return [i for i in range(self.I_size) if self.x_ikb[i, k_idx, task_id] > 0]

    def get_robot_workload(self, robot_id: int) -> int:
        """
        获取机器人总共需要搬运的料箱数
        
        Args:
            robot_id: 机器人ID
            
        Returns:
            料箱总数
        """
        k_idx = self.robot_index_map.get(robot_id, robot_id)
        if k_idx >= self.K_size:
            return 0
        return self.robot_bin_loads[k_idx]

    def get_task_bins(self, task_id: int) -> List[Tuple[int, int]]:
        """
        获取任务使用的所有料箱及其对应的机器人
        
        Args:
            task_id: 任务ID
            
        Returns:
            [(bin_index, robot_index), ...]
        """
        if task_id >= self.B_size:
            return []
        
        result = []
        for i in range(self.I_size):
            for k in range(self.K_size):
                if self.x_ikb[i, k, task_id] > 0:
                    result.append((i, k))
        return result

    def get_bin_actual_id(self, bin_index: int) -> Optional[int]:
        """
        根据索引获取料箱的实际ID
        
        Args:
            bin_index: 料箱索引
            
        Returns:
            料箱实际ID
        """
        for bin_id, idx in self.bin_index_map.items():
            if idx == bin_index:
                return bin_id
        return None

    def validate_robot_capacity(self, robot_max_capacity: int) -> Dict[int, bool]:
        """
        验证机器人是否超载
        
        Args:
            robot_max_capacity: 机器人最大堆叠容量
            
        Returns:
            {robot_id: is_valid}
        """
        validation = {}
        for robot_id, k_idx in self.robot_index_map.items():
            workload = self.robot_bin_loads[k_idx]
            validation[robot_id] = workload <= robot_max_capacity
        return validation

    def get_most_used_bins(self, top_n: int = 10) -> List[Tuple[int, int]]:
        """
        获取使用最频繁的料箱
        
        Args:
            top_n: 返回前N个
            
        Returns:
            [(bin_index, usage_count), ...] 按使用次数降序
        """
        bin_usage_list = [(i, self.bin_usage[i]) for i in range(self.I_size) if self.bin_usage[i] > 0]
        bin_usage_list.sort(key=lambda x: x[1], reverse=True)
        return bin_usage_list[:top_n]

    def summary(self) -> str:
        """返回变量摘要信息"""
        avg_robot_load = np.mean(self.robot_bin_loads) if self.K_size > 0 else 0.0
        max_robot_load = np.max(self.robot_bin_loads) if self.K_size > 0 else 0
        
        active_bins = sum(1 for u in self.bin_usage if u > 0)
        
        summary_str = f"""
SP3Variable Summary:
====================
Total Bins Used: {self.total_bins_used}
Active Bins (at least once): {active_bins}/{self.I_size}
Objective Value: {self.objective_value:.2f}
Feasible: {self.is_feasible}

Robot Statistics:
-----------------
"""
        for k in range(self.K_size):
            summary_str += f"  Robot {k}: {self.robot_bin_loads[k]} bins\n"
        
        summary_str += f"\nAverage Robot Load: {avg_robot_load:.2f} bins\n"
        summary_str += f"Max Robot Load: {max_robot_load} bins\n"
        
        return summary_str

    def export_selection(self) -> Dict[str, any]:
        """
        导出料箱选择结果为字典格式
        
        Returns:
            包含所有选择信息的字典
        """
        selection_dict = {
            'total_bins_used': self.total_bins_used,
            'objective': self.objective_value,
            'robots': []
        }
        
        for k in range(self.K_size):
            robot_info = {
                'robot_id': k,
                'bin_count': self.robot_bin_loads[k],
                'tasks': []
            }
            
            for b in range(self.B_size):
                bins_for_task = self.get_selected_bins(k, b)
                if bins_for_task:
                    robot_info['tasks'].append({
                        'task_id': b,
                        'bins': bins_for_task
                    })
            
            if robot_info['tasks']:
                selection_dict['robots'].append(robot_info)
        
        return selection_dict


'''
File: solve_sp3.py
Project: OFS_Integrated_Model
Description: 
----------
求解子问题3: 料箱选择 (Bin Selection)
----------
'''

import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Optional, Tuple, Set
from problemDto.ofs_problem_dto import OFSProblemDTO
from solver.sp3_variable import SP3Variable
from config.ofs_config import OFSConfig
from entity.tote import Tote


class SolveSP3:
    """
    求解子问题3: 料箱选择 (Bin Selection)
    
    目标: 为每个任务选择料箱以满足SKU需求，同时考虑机器人容量约束
    
    输入:
        - SP1: z_{o,s,b} (SKU需求分配)
        - SP4: y_{b,k} (任务-机器人分配)
    
    输出:
        - x_{i,k,b}: 料箱i是否被机器人k为任务b选择
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
        self.totes = problem_dto.tote_list
        self.robots = problem_dto.robot_list
        self.skus = problem_dto.skus_list
        
        print(f"[SP3] 初始化完成: {len(self.totes)} 个料箱, "
              f"{len(self.robots)} 个机器人, {len(self.skus)} 种SKU")

    def solve(
        self,
        active_tasks: List[int],  # 激活的任务列表
        task_sku_demand: Dict[Tuple[int, int], int],  # {(task_id, sku_id): quantity} from SP1
        task_robot_assignment: Dict[int, int],  # {task_id: robot_id} from SP4
        time_limit: int = 3600,
        output_flag: bool = True
    ) -> Optional[SP3Variable]:
        """
        求解 SP3 模型
        
        Args:
            active_tasks: 激活的任务ID列表
            task_sku_demand: 任务SKU需求 {(task_id, sku_id): quantity}
            task_robot_assignment: 任务机器人分配 {task_id: robot_id}
            time_limit: 求解时间限制(秒)
            output_flag: 是否显示Gurobi求解过程
            
        Returns:
            SP3Variable 实例，失败返回 None
        """
        
        print(f"\n{'='*60}")
        print(f"开始求解 SP3 - 料箱选择")
        print(f"{'='*60}")
        
        # 1. 数据准备和验证
        if not active_tasks:
            print("⚠️  警告: 没有激活的任务需要处理")
            return None
        
        B = active_tasks  # 激活的任务列表
        I = list(range(len(self.totes)))  # 料箱索引
        K = list(range(len(self.robots)))  # 机器人索引
        S = list(range(len(self.skus)))  # SKU索引
        
        print(f"📊 问题规模:")
        print(f"   - 激活任务数: {len(B)}")
        print(f"   - 料箱数: {len(I)}")
        print(f"   - 机器人数: {len(K)}")
        print(f"   - SKU种类数: {len(S)}")
        
        # 2. 预处理：构建料箱-SKU库存矩阵
        bin_sku_inventory = self._build_bin_sku_inventory()
        
        # 3. 预处理
                # 3. 预处理：验证需求可满足性
        if not self._validate_demand_feasibility(task_sku_demand, bin_sku_inventory):
            print("❌ 错误: SKU需求无法被当前库存满足")
            return None
        
        # 4. 创建 Gurobi 模型
        print("\n🔧 构建 Gurobi 模型...")
        m = gp.Model("SP3_Bin_Selection")
        m.setParam('OutputFlag', 1 if output_flag else 0)
        m.setParam('TimeLimit', time_limit)
        m.setParam('MIPGap', 0.01)  # 1% gap
        
        # 5. 决策变量
        print("   添加决策变量...")
        
        # x[i,k,b]: 料箱i是否被机器人k为任务b选择
        x = m.addVars(I, K, B, vtype=GRB.BINARY, name="x")
        
        # 6. 目标函数：最小化使用的料箱总数
        print("   设置目标函数...")
        # 方案1: 最小化料箱总数
        m.setObjective(
            gp.quicksum(x[i, k, b] for i in I for k in K for b in B),
            GRB.MINIMIZE
        )
        
        # 7. 约束条件
        print("   添加约束条件...")
        
        # (C1) 库存满足约束 (eq:inventory_fulfillment)
        # 对于每个任务b、机器人k、SKU s，选择的料箱必须满足需求
        constraint_count = 0
        for b in B:
            k = task_robot_assignment.get(b)
            if k is None:
                print(f"⚠️  警告: 任务 {b} 未分配机器人，跳过")
                continue
            
            for s in S:
                demand = task_sku_demand.get((b, s), 0)
                if demand > 0:
                    m.addConstr(
                        gp.quicksum(x[i, k, b] * bin_sku_inventory.get((i, s), 0) 
                                   for i in I) >= demand,
                        name=f"C1_inventory_{b}_{k}_{s}"
                    )
                    constraint_count += 1
        
        print(f"   ✓ C1: 库存满足约束 ({constraint_count} 个)")
        
        # (C2) 料箱选择链接约束 (eq:link_x_y_bk)
        # 料箱只能由分配的机器人选择
        constraint_count = 0
        for i in I:
            for b in B:
                assigned_robot = task_robot_assignment.get(b)
                if assigned_robot is not None:
                    # 只有被分配的机器人才能选择料箱
                    for k in K:
                        if k != assigned_robot:
                            m.addConstr(x[i, k, b] == 0, 
                                       name=f"C2_link_{i}_{k}_{b}")
                            constraint_count += 1
        
        print(f"   ✓ C2: 料箱选择链接约束 ({constraint_count} 个)")
        
        # (C3) 机器人堆叠高度约束 (eq:stack_height)
        robot_max_capacity = getattr(self.config, 'ROBOT_CAPACITY', 5)
        for b in B:
            k = task_robot_assignment.get(b)
            if k is not None:
                m.addConstr(
                    gp.quicksum(x[i, k, b] for i in I) <= robot_max_capacity,
                    name=f"C3_stack_height_{b}_{k}"
                )
        
        print(f"   ✓ C3: 机器人堆叠高度约束 ({len(B)} 个)")
        
        # (C4) 料箱唯一性约束 (可选)
        # 同一个料箱不能同时被多个任务使用
        use_uniqueness = getattr(self.config, 'BIN_UNIQUENESS_CONSTRAINT', True)
        if use_uniqueness:
            constraint_count = 0
            for i in I:
                for k in K:
                    m.addConstr(
                        gp.quicksum(x[i, k, b] for b in B) <= 1,
                        name=f"C4_uniqueness_{i}_{k}"
                    )
                    constraint_count += 1
            print(f"   ✓ C4: 料箱唯一性约束 ({constraint_count} 个)")
        
        # (C5) 优先选择顶层料箱 (软约束，通过目标函数权重)
        # 添加惩罚项：选择非顶层料箱有额外成本
        penalty_weight = 0.01  # 小权重，避免影响主要目标
        penalty_terms = []
        for i in I:
            tote = self.totes[i]
            if not tote.is_top:  # 如果不是顶层
                penalty_terms.append(
                    gp.quicksum(x[i, k, b] for k in K for b in B)
                )
        
        if penalty_terms:
            # 修改目标函数加入惩罚项
            m.setObjective(
                gp.quicksum(x[i, k, b] for i in I for k in K for b in B) +
                penalty_weight * gp.quicksum(penalty_terms),
                GRB.MINIMIZE
            )
            print(f"   ✓ C5: 顶层优先软约束已添加")
        
        # 8. 优化求解
        print(f"\n🚀 开始求解...")
        print(f"   时间限制: {time_limit} 秒")
        m.update()
        m.optimize()
        
        # 9. 解析结果
        return self._parse_solution(m, x, active_tasks, I, K)

    def _build_bin_sku_inventory(self) -> Dict[Tuple[int, int], int]:
        """
        构建料箱-SKU库存矩阵
        
        Returns:
            {(bin_index, sku_id): quantity}
        """
        print("\n📦 构建料箱-SKU库存矩阵...")
        
        inventory = {}
        
        for i, tote in enumerate(self.totes):
            for sku in tote.skus_list:
                quantity = tote.sku_quantity_map.get(sku.id, 0)
                if quantity > 0:
                    inventory[(i, sku.id)] = quantity
        
        print(f"   共 {len(inventory)} 个料箱-SKU库存记录")
        
        return inventory

    def _validate_demand_feasibility(
        self,
        task_sku_demand: Dict[Tuple[int, int], int],
        bin_sku_inventory: Dict[Tuple[int, int], int]
    ) -> bool:
        """
        验证SKU需求是否可以被库存满足
        
        Args:
            task_sku_demand: {(task_id, sku_id): quantity}
            bin_sku_inventory: {(bin_index, sku_id): quantity}
            
        Returns:
            是否可行
        """
        print("\n✅ 验证需求可满足性...")
        
        # 统计每个SKU的总需求
        total_demand = {}
        for (task_id, sku_id), qty in task_sku_demand.items():
            total_demand[sku_id] = total_demand.get(sku_id, 0) + qty
        
        # 统计每个SKU的总库存
        total_inventory = {}
        for (bin_idx, sku_id), qty in bin_sku_inventory.items():
            total_inventory[sku_id] = total_inventory.get(sku_id, 0) + qty
        
        # 检查每个SKU
        infeasible_skus = []
        for sku_id, demand in total_demand.items():
            inventory = total_inventory.get(sku_id, 0)
            if inventory < demand:
                infeasible_skus.append((sku_id, demand, inventory))
                print(f"   ❌ SKU {sku_id}: 需求={demand}, 库存={inventory} (不足)")
        
        if infeasible_skus:
            print(f"\n   共 {len(infeasible_skus)} 种SKU库存不足")
            return False
        
        print("   ✓ 所有SKU库存充足")
        return True

    def _parse_solution(
        self,
        model: gp.Model,
        x_vars,
        active_tasks: List[int],
        bins: List[int],
        robots: List[int]
    ) -> Optional[SP3Variable]:
        """
        解析 Gurobi 求解结果并创建 SP3Variable
        
        Args:
            model: Gurobi 模型
            x_vars: 决策变量
            active_tasks: 激活的任务列表
            bins: 料箱索引列表
            robots: 机器人索引列表
            
        Returns:
            SP3Variable 实例，失败返回 None
        """
        
        print(f"\n{'='*60}")
        print(f"求解完成")
        print(f"{'='*60}")
        
        if model.Status == GRB.OPTIMAL:
            print(f"✅ 状态: OPTIMAL")
            print(f"📊 目标值 (料箱总数): {model.ObjVal:.0f}")
            
            # 创建变量容器
            max_task_id = max(active_tasks) if active_tasks else 0
            B_size = max_task_id + 1
            I_size = len(bins)
            K_size = len(robots)
            
            sp3_var = SP3Variable(I_size=I_size, K_size=K_size, B_size=B_size)
            
            # 提取解
            try:
                sp3_var.set_solution(
                    bins=self.totes,
                    robots=self.robots,
                    x_vars=x_vars,
                    obj_value=model.ObjVal,
                    active_tasks=active_tasks
                )
                
                # 打印详细结果
                self._print_solution_summary(sp3_var, active_tasks)
                
                return sp3_var
                
            except Exception as e:
                print(f"❌ 解析解时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        elif model.Status == GRB.TIME_LIMIT:
            print(f"⏱️  状态: TIME_LIMIT")
            print(f"📊 当前目标值: {model.ObjVal:.0f}")
            print(f"📊 最优界: {model.ObjBound:.0f}")
            print(f"📊 Gap: {model.MIPGap*100:.2f}%")
            
            # 即使超时，也尝试提取当前解
            if model.SolCount > 0:
                print("✓ 找到可行解，尝试提取...")
                
                max_task_id = max(active_tasks) if active_tasks else 0
                B_size = max_task_id + 1
                I_size = len(bins)
                K_size = len(robots)
                
                sp3_var = SP3Variable(I_size=I_size, K_size=K_size, B_size=B_size)
                
                try:
                    sp3_var.set_solution(
                        bins=self.totes,
                        robots=self.robots,
                        x_vars=x_vars,
                        obj_value=model.ObjVal,
                        active_tasks=active_tasks
                    )
                    
                    self._print_solution_summary(sp3_var, active_tasks)
                    return sp3_var
                    
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
                iis_file = "sp3_infeasible_model.ilp"
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

