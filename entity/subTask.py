# entity/sub_task.py
from dataclasses import dataclass, field
from typing import List, Optional
from entity.order import Order
from entity.SKUs import SKUs
from entity.point import Point
from entity.stack import Stack
from entity.task import Task


@dataclass
class SubTask:
    """
    子任务 (Sub-Task)
    由 SP1 (BOM拆分) 生成，代表一组需要被一起拣选的 SKU 集合。
    """
    id: int
    parent_order: Order  # 归属的父BOM订单
    sku_list: List[SKUs]  # 包含的 SKU 列表
    # --- 自动计算字段 (init=False 表示不需要在构造函数中传参) ---
    # 不同SKU的数量
    sku_quantity: int = field(init=False)
    # 不重复的SKUlist
    unique_sku_list: List[SKUs] = field(init=False)
    # --- 后续阶段决策结果 ---
    assigned_station_id: int = -1  # SP2 决策：分配给哪个工作站
    station_sequence_rank: int = -1  # SP2 决策：在工作站的执行顺位

    # 预计开始处理时间 (用于计算 Makespan)
    estimated_process_start_time: float = 0.0

    '''#sp3 决策：命中的料箱id列表'''
    # 1. 物理分配的料箱ID列表
    assigned_tote_ids: List[int] = field(default_factory=list)

    # 2. 生成的物理任务对象列表 (使用字符串 'Task' 避免循环导入)
    execution_tasks: List['Task'] = field(default_factory=list)

    # 3. 命中的堆垛对象列表 (排重)
    involved_stacks: List['Stack'] = field(default_factory=list)

    # 4. 需要访问的点位对象列表 (对应涉及的堆垛坐标)
    visit_points: List['Point'] = field(default_factory=list)

    assigned_robot_id: int = -1  # SP4 决策：分配给哪个机器人
    # === SP4 运行时缓存字段 (不参与 init) ===
    # 用于存储预处理后的 SubTask 物理入口点 (第一个访问的堆垛坐标)
    _cached_start_pt: Optional[Point] = field(default=None, init=False)

    # 用于存储预处理后的 SubTask 物理出口点 (工作站坐标)
    _cached_end_pt: Optional[Point] = field(default=None, init=False)

    # 用于存储 SubTask 内部执行总耗时 (含移动和操作)
    _cached_duration: float = field(default=0.0, init=False)
    def add_execution_detail(self, task_obj: 'Task', stack_obj: 'Stack'):
        """
        SP3 求解器专用：向 SubTask 注册物理执行细节。
        自动更新 Tasks, Stacks, Points 和 Tote IDs。
        """
        # 1. 添加任务对象
        self.execution_tasks.append(task_obj)

        # 2. 添加堆垛对象 (去重逻辑：如果还没加过这个堆垛)
        # 注意：这里通过 stack_id 判断是否已存在
        if not any(s.stack_id == stack_obj.stack_id for s in self.involved_stacks):
            self.involved_stacks.append(stack_obj)

            # 3. 添加对应的点位 (只有新堆垛才加点位)
            if stack_obj.store_point:
                self.visit_points.append(stack_obj.store_point)

        # 4. 更新料箱 ID 列表 (合并去重)
        # task_obj.target_tote_ids 是本次任务带走的
        current_totes = set(self.assigned_tote_ids)
        new_totes = set(task_obj.target_tote_ids)
        self.assigned_tote_ids = list(current_totes.union(new_totes))
    def __post_init__(self):
        """
        在初始化后自动计算 unique_sku_list 和 sku_quantity
        """
        seen_ids = set()
        self.unique_sku_list = []
        for sku in self.sku_list:
            if sku.id not in seen_ids:
                self.unique_sku_list.append(sku)
                seen_ids.add(sku.id)
        self.sku_quantity = len(self.unique_sku_list)

    def reset_execution_details(self):
        """
        重置/清理 SP3 的执行数据。
        在 SP3 求解开始前调用，防止多次求解导致数据累积。
        """
        self.assigned_tote_ids = []  # 清空分配的料箱
        self.execution_tasks = []  # 清空关联的任务
        self.involved_stacks = []  # 清空涉及的堆垛
        self.visit_points = []  # 清空访问点位
    @property
    def capacity_usage(self) -> int:
        """
        修正：计算容量占用。
        机器人的槽位限制是基于 '料箱数' 或 'SKU种类数' 的。
        假设 1 种 SKU 主要集中在 1 个料箱中，则占用 1 个容量。
        """
        unique_ids = set(s.id for s in self.sku_list)
        return len(unique_ids)

    def confirm_allocation(self, tote_ids: List[int]):
        """
        [New] SP3 求解后调用，确认物理分配的箱子
        """
        self.assigned_tote_ids = list(tote_ids)
    def __str__(self):
        return (f"SubTask(id={self.id}, unique_types={self.sku_quantity})")
