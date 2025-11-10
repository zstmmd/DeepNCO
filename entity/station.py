from dataclasses import dataclass

from config.ofs_config import OFSConfig
from typing import List
@dataclass
class Station:
    """工作台类"""

    def __init__(self, station_id: int = 0):
        """
        :param station_id: 工作台的id
        """
        self.id = station_id
        self.point=None # 工作台所在的点对象
        self.picking_time = OFSConfig.PICKING_TIME  # 工作台的单个SKU的拣选时间
        self.picking_station_buffer = OFSConfig.DEFAULT_PICKING_STATION_BUFFER  # 工作台的分拨墙的上限
        self.task_queue:List['TaskBatch'] = []  # 工作台的任务序列 (用于决策"子任务排序")



