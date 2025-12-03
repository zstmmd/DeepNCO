from dataclasses import dataclass, field
from typing import List

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
    # 该任务需要从该堆垛取走的具体料箱
    target_tote_ids: List[int] = field(default_factory=list)
    # 对应 VRP 中的路径次序
    robot_visit_sequence: int = -1

    # 预计到达堆垛的时间
    arrival_time_at_stack: float = 0.0

    #预计到达工作站的时间 (用于软耦合时间窗检查)
    arrival_time_at_station: float = 0.0