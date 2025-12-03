# entity/sub_task.py
from dataclasses import dataclass
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

    # --- 后续阶段决策结果 ---
    assigned_station_id: int = -1  # SP2 决策：分配给哪个工作站
    station_sequence_id: int = -1  # SP2 决策：在工作站的执行顺位

    # 预计开始处理时间 (用于计算 Makespan)
    estimated_process_start_time: float = 0.0
    #sp3 决策：命中的料箱id列表
    assigned_tote_ids: Optional[List[int]] = None
    assigned_robot_id: int = -1  # SP4 决策：分配给哪个机器人

    @property
    def capacity_usage(self) -> int:
        """计算该任务占用的物理容量 (假设 1 SKU = 1 Slot)"""
        return len(self.sku_list)

    def __str__(self):
        return (f"SubTask(id={self.id}, order={self.parent_order.order_id}, "
                f"skus={len(self.sku_list)})")
