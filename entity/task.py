from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from entity.tote import Tote
from entity.station import Station
from entity.point import Point


@dataclass
class Task:
    """
    Stack Visit
    """
    task_id: int
    sub_task_id: int  # 关联的子任务ID
    # 核心：目标堆垛
    target_stack_id: int
    target_station_id: int
    operation_mode: str  # 'FLIP' or 'SORT'
    robot_id: int = -1  # 分配的机器人ID
    station_sequence_rank: int =0
    # 物理上机器人带走的所有料箱ID (用于计算物理负重、仿真搬运)
    target_tote_ids: List[int] = field(default_factory=list)

    # 真正命中订单需求的料箱ID (用于工作站拣选)
    hit_tote_ids: List[int] = field(default_factory=list)

    # 噪音料箱ID (Target - Hit)，需要工作站剔除并回库
    noise_tote_ids: List[int] = field(default_factory=list)

    # (仅Sort模式) 搬运的物理层级区间 (Bottom, Top)
    sort_layer_range: Optional[Tuple[int, int]] = None

    # 对应 VRP 中的路径次序
    robot_visit_sequence: int = -1

    # 预计到达堆垛的时间
    arrival_time_at_stack: float = 0.0

    # --- 耗时统计 ---
    # 机器人操作耗时 (挖掘/移位 + 抓取)
    robot_service_time: float = 0.0
    # 工作台操作耗时 (剔除噪音箱 + 拣选)
    station_service_time: float = 0.0

    #预计到达工作站的时间 (用于软耦合时间窗检查)
    arrival_time_at_station: float = 0.0

    sku_pick_count: int = 0
    # 开始捡货/处理时间 (取决于 Station 的空闲时间，FCFS 逻辑)
    start_process_time: float = 0.0

    # 结束处理时间 (start + picking_time + noise_handling_time)
    end_process_time: float = 0.0

    # --- 统计时长信息 ---
    # 料箱在工作站的等待时长 (start_process_time - arrival_time_at_station)
    tote_wait_time: float = 0.0

    # 该任务造成的捡货时长 (sku_count * t_pick)
    picking_duration: float = 0.0
    trip_id = 0
    # --- 路径信息 ---
    # 记录该任务对应的机器人具体路径 [(x, y, time), ...]
    # 用于输出到 txt
    detailed_path: List[Tuple[float, float, float]] = field(default_factory=list)
    @property
    def total_load_count(self) -> int:
        return len(self.target_tote_ids)

    @property
    def total_service_time(self) -> float:
        """总耗时 = 机器人耗时 + 工作台耗时"""
        return self.robot_service_time + self.station_service_time

    @property
    def estimated_service_time(self) -> float:
        return self.total_service_time

    @property
    def total_load_count(self) -> int:
        return len(self.target_tote_ids)