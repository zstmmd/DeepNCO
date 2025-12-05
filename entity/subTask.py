# entity/sub_task.py
from dataclasses import dataclass, field
from typing import List, Optional
from entity.order import Order
from entity.SKUs import SKUs


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
    #sp3 决策：命中的料箱id列表
    assigned_tote_ids: Optional[List[int]] = None
    assigned_robot_id: int = -1  # SP4 决策：分配给哪个机器人

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

    @property
    def capacity_usage(self) -> int:
        """
        修正：计算容量占用。
        机器人的槽位限制是基于 '料箱数' 或 'SKU种类数' 的。
        假设 1 种 SKU 主要集中在 1 个料箱中，则占用 1 个容量。
        """
        unique_ids = set(s.id for s in self.sku_list)
        return len(unique_ids)

    def __str__(self):
        return (f"SubTask(id={self.id}, unique_types={self.sku_quantity})")
