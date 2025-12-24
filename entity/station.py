from dataclasses import dataclass, field

from config.ofs_config import OFSConfig
from typing import List
@dataclass
class Station:
    """工作台类"""

    # 工作站当前不可用的截止时间 (即上一个任务结束的时间)
    next_available_time: float = 0.0

    # 累计空闲时长
    total_idle_time: float = 0.0

    # 处理过的任务列表 (用于验证顺序)
    processed_tasks: List = field(default_factory=list)
    def __init__(self, station_id: int = 0):
        """
        :param station_id: 工作台的id
        """
        self.id = station_id
        self.point=None # 工作台所在的点对象
        self.picking_time = OFSConfig.PICKING_TIME  # 工作台的单个SKU的拣选时间
        self.picking_station_buffer = OFSConfig.DEFAULT_PICKING_STATION_BUFFER  # 工作台的分拨墙的上限
        self.task_queue:List['TaskBatch'] = []  # 工作台的任务序列 (用于决策"子任务排序")
    def __repr__(self):
        return f"Station(id={self.id})"

