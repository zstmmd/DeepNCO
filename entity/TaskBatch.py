from entity.warehouseMap import WarehouseMap
from entity.robot import Robot
from entity.order import Order
from entity.SKUs import SKUs
from entity.station import Station
from entity.tote import Tote
from entity.point import Point
from entity.MainBatch import MainBatch
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig
from typing import Dict, List
class TaskBatch:
    """
    “机器人可执行的子任务”。
    这是连接所有决策的核心类，也是算法要优化的主要对象。
    """
    def __init__(self, task_id: int, main_batch: MainBatch):
        self.id = task_id
        self.main_batch = main_batch   # 隶属于哪个大批次

        # --- 决策 1: 订单组批 (聚合的需求) ---
        self.sku_requirements: Dict[SKUs, int] = {}
        self.source_orders: List[Order] = []

        # --- 决策 2 & 5: 分配给机器人和工作站 ---
        self.assigned_robot: Robot = None
        self.assigned_workstation: Station = None

        # --- 决策 4: 命中料箱 ---
        self.target_bins: List[Tote] = []

        # --- 决策 6: 机器人访问排序 (dig与搬运) ---
        # 这是一个有序列表，代表机器人访问料箱堆的顺序
        self.robot_bin_visit_sequence: List[Tote] = []

        # --- 算法的输出：时间 ---
        self.robot_start_time = None           # 机器人出发时间
        self.robot_dig_and_retrieve_time = None  # 挖掘和搬运总耗时
        self.robot_arrival_at_ws_time = None    # 到达工作台时间
        self.ws_start_time = None              # 工作台开始处理时间 (受"子任务排序"影响)
        self.ws_end_time = None                # 工作台完成处理时间

    def get_total_weight(self) -> float:
        """计算此子任务的总重量 (用于校验机器人载重)"""
        total_w = 0.0
        for sku, qty in self.sku_requirements.items():
            total_w += sku.weight * qty
        return total_w