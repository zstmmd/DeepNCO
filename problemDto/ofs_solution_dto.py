from typing import List, Dict
import sys
import math

from problemDto.ofs_problem_dto import OFSProblemDTO
from entity.TaskBatch import TaskBatch
from entity.robot import Robot
from entity.station import Station
from entity.MainBatch import MainBatch


class OFSSolution:
    """
    本类用于存储“免货架堆垛式系统”问题的完整解。
    """
    
    def __init__(self, problem: OFSProblemDTO):
        """
        使用 OFSProblemDTO 初始化一个空的解。
        
        :param problem: 算法要解决的原始问题实例。
        """
        # 1. 基础：保留对原始问题的引用，以便进行评估
        self.problem: OFSProblemDTO = problem
        
        # 2. 存储决策的核心
        
        # 算法生成的“子任务”列表
        self.task_batches: List[TaskBatch] = []
        
        
        # 机器人调度表 (Key: robot_id)
        # Value: 该机器人要执行的 TaskBatch 列表 (按执行顺序)
        self.robot_schedules: Dict[int, List[TaskBatch]] = {
            robot.id: [] for robot in problem.robot_list
        }
        
        # 工作站调度表 (Key: station_id)
        # Value: 该工作站要处理的 TaskBatch 列表 (按加工顺序)
        self.workstation_schedules: Dict[int, List[TaskBatch]] = {
            station.id: [] for station in problem.station_list
        }
        
        # 3. 存储评估结果 (KPIs)
        self.makespan: float = 0.0
        #每个mainbatch的最大完成时间
        self.main_batch_makespan={}
        self.is_feasible: bool = False
        self.kit_window_violation: float = 0.0  # 违反“齐套出库”时间窗的秒数
        self.total_robot_travel_time: float = 0.0 # 机器人总行驶时间
        self.total_robot_dig_time: float = 0.0    # 机器人总移箱时间

    def evaluate(self):
        """
        (核心评估函数)
        根据已填充的决策（TaskBatches 及其时间戳）计算所有关键 KPI。
        
        注意：此函数假定一个“求解器” (Solver) 已经
        1. 创建了 task_batches
        2. 将它们分配到了 self.robot_schedules 和 self.workstation_schedules
        3. 并且计算并填充了每个 TaskBatch 的时间戳字段 
           (如 ws_start_time, ws_end_time)
        """
        
        if not self.task_batches:
            print("警告: 解决方案中没有任务批次 (TaskBatches)，无法评估。")
            return

        min_start_time = math.inf
        max_end_time = 0.0

        # --- 1. 计算最大完工时间 (Makespan) ---
        for task in self.task_batches:
            if task.ws_start_time is None or task.ws_end_time is None:
                # 这是一个错误，表明求解器未正确填充时间
                self.is_feasible = False
                print(f"错误: TaskBatch {task.id} 缺少时间戳，评估失败。")
                return

            # 更新最小/最大时间，用于齐套约束
            if task.ws_start_time < min_start_time:
                min_start_time = task.ws_start_time
            if task.ws_end_time > max_end_time:
                max_end_time = task.ws_end_time

        # 目标函数：最小化最大完工时间
        self.makespan = max_end_time

        # --- 2. 检查全局“齐套出库”约束 ---
        # main_batch: MainBatch = self.problem.main_batch_list #
        
        # # 假设大批次的时间窗是一个“持续时间”
        # # 找到实际的开始/结束时间
        # if hasattr(main_batch, 'kit_window_duration') and main_batch.kit_window_duration:
        #     actual_duration = max_end_time - min_start_time
        #     allowed_duration = main_batch.kit_window_duration
            
        #     if actual_duration <= allowed_duration:
        #         self.is_feasible = True
        #         self.kit_window_violation = 0.0
        #     else:
        #         self.is_feasible = False # 违反齐套约束
        #         self.kit_window_violation = actual_duration - allowed_duration
        # else:
        #     # 如果 MainBatch 没有定义时间窗，我们默认它是可行的
        #     self.is_feasible = True
        #     self.kit_window_violation = 0.0

        # (此处可以添加对机器人总行驶/挖掘时间的计算)
        # (例如，通过 TaskBatch 中的 robot_dig_and_retrieve_time)
        
        # # 填充 MainBatch 的齐套窗口（用于日志记录）
        # main_batch.kit_delivery_window_start = min_start_time
        # main_batch.kit_delivery_window_end = max_end_time


    # def __str__(self) -> str:

    #     self.evaluate()
        
    #     header = "--- OFS 解决方案摘要 ---\n"
        
    #     # 1. 总体 KPI
    #     kpi_str = (
    #         f"目标 (Makespan): {self.makespan:.2f} 秒\n"
    #         f"齐套可行性 (Feasible): {self.is_feasible}\n"
    #         f"齐套违规 (Violation): {self.kit_window_violation:.2f} 秒\n"
    #         f"总任务批次 (TaskBatches): {len(self.task_batches)}\n"
    #     )
        
    #     # 2. 机器人调度
    #     robot_str = "\n--- 机器人调度 (MITA-D) ---\n"
    #     for robot_id, tasks in self.robot_schedules.items():
    #         task_ids = [str(t.id) for t in tasks]
    #         robot_str += f"  机器人 {robot_id}: "
    #         if task_ids:
    #             robot_str += " -> ".join(task_ids) + "\n"
    #         else:
    #             robot_str += "空闲\n"

    #     # 3. 工作站调度
    #     ws_str = "\n--- 工作站调度 (MITA-E) ---\n"
    #     for ws_id, tasks in self.workstation_schedules.items():
    #         task_ids = [str(t.id) for t in tasks]
    #         ws_str += f"  工作站 {ws_id}: "
    #         if task_ids:
    #             ws_str += " -> ".join(task_ids) + "\n"
    #         else:
    #             ws_str += "空闲\n"
                
    #     return header + kpi_str + robot_str + ws_str

# --------------------------------------------------------------------
# 示例用法 (模拟一个求解器填充此 Solution)
# --------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. 导入并创建问题 (使用上一轮的 CreateOFSProblem) ---
    try:
        from create_ofs_problem import CreateOFSProblem
        from entity.MainBatch import MainBatch
        from entity.TaskBatch import TaskBatch #
    except ImportError:
        print("错误: 无法导入 'create_ofs_problem'。")
        print("请确保 create_ofs_problem.py 与此文件位于同一目录或 Python 路径中。")
        # (为防止执行失败，我们在此处定义一个最小化的 DTO 和 TaskBatch 以便演示)
        class MinimalDTO:
            robot_list = [Robot(robot_id=1)]
            station_list = [Station(station_id=1)]
            main_batch = MainBatch("MB_TEST", [])
            main_batch.kit_window_duration = 180.0
        
        problem_dto = MinimalDTO()
        TaskBatch = TaskBatch # 假设它已被正确导入

    # 假设我们用 `CreateOFSProblem` 创建了实例
    # (如果导入成功，使用真实的问题实例)
    if 'CreateOFSProblem' in locals():
        problem_dto = CreateOFSProblem.create_ofs_problem(
            warehouse_length_block_number=2,
            warehouse_width_block_number=2,
            robot_num=1,
            order_num=5,
            skus_num=10,
            tote_num=20,
            station_num=1
        )
        problem_dto.main_batch.kit_window_duration = 300.0 # 设置齐套窗口为5分钟


    # --- 2. 创建一个空的 Solution 对象 ---
    solution = OFSSolution(problem_dto)
    
    print("创建了一个空的解决方案:")
    print(solution)

    # --- 3. (模拟求解器) 填充决策 ---
    
    # 假设求解器创建了 2 个 TaskBatches
    robot = problem_dto.robot_list[0]
    station = problem_dto.station_list[0]
    
    # TaskBatch 1
    tb1 = TaskBatch(task_id=101, main_batch=problem_dto.main_batch)
    tb1.assigned_robot = robot
    tb1.assigned_workstation = station
    # (求解器会计算出时间)
    tb1.ws_start_time = 60.0  # 第60秒开始
    tb1.ws_end_time = 120.0 # 第120秒结束
    
    # TaskBatch 2
    tb2 = TaskBatch(task_id=102, main_batch=problem_dto.main_batch)
    tb2.assigned_robot = robot
    tb2.assigned_workstation = station
    # (求解器会计算出时间)
    tb2.ws_start_time = 125.0 # 第125秒开始 (等待机器人返回)
    tb2.ws_end_time = 190.0 # 第190秒结束
    
    # 决策 1: 订单组批
    solution.task_batches = [tb1, tb2]
    
    # 决策 2, 3, 5, 6: 资源分配和排序
    solution.robot_schedules[robot.id] = [tb1, tb2]
    solution.workstation_schedules[station.id] = [tb1, tb2]
    
    # (决策 4, 6-细节 会在 tb1 和 tb2 的内部被填充，例如)
    # tb1.target_bins = [tote_A, tote_B]
    # tb1.robot_bin_visit_sequence = [tote_B, tote_A]


    # --- 4. 评估并打印填充后的解决方案 ---
    
    print("\n" + "="*30)
    print("填充并评估解决方案:")
    
    # 评估将自动计算 Makespan 和 Feasibility
    print(solution)