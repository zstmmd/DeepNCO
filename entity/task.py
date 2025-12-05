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

    estimated_service_time: float = 0.0  # 预计在这个点消耗的时间(挖掘或移位)

    #预计到达工作站的时间 (用于软耦合时间窗检查)
    arrival_time_at_station: float = 0.0

    @property
    def total_load_count(self) -> int:
        return len(self.target_tote_ids)