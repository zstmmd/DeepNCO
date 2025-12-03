from typing import List, Dict

from entity.point import Point
from entity.stack import Stack
from entity.warehouseMap import WarehouseMap

from entity.order import Order
from entity.robot import Robot
from entity.tote import Tote
from entity.station import Station
from dataclasses import dataclass
from entity.SKUs import SKUs
@dataclass
class OFSProblemDTO:
    """问题类"""
    
    def __init__(self):
        self.map: WarehouseMap = None  # 地图信息
        self.order_num: int = 0  # 订单数量
        self.order_list: List[Order] = []  # 订单信息

        self.robot_num: int = 0  # 机器人数量
        self.robot_list: List[Robot] = []  # 机器人信息
        self.tote_num: int = 0  # 料箱数量
        self.tote_list: List[Tote] = []  #料箱列表
        self.station_num: int = 0  # 工作站数量
        self.station_list: List[Station] = []  # 工作站列表
        self.skus_num: int = 0  # SKUs数量
        self.skus_list: List[SKUs] = []  # SKUs信息
        self.w: int = 0  # 机器人起点数量
        self.n: int = 0  # 订单点总数
        self.subtask_num: int = 0  # 机器人可执行的子任务数量
        self.subtask_list: List = []  # 机器人可执行的子任务列表
        self.task_num: int = 0  # 可执行任务总数
        self.id_to_tote={} # 料箱id到料箱对象的映射
        self.id_to_sku={} # sku id到sku对象的映射
        self.id_to_order={} # 订单id到订单对象的映射
        self.need_points: List[Point] = []  # 搬运层需要经过的点的列表
        self.node_num: int = 0  # 搬运层需要经过的点总数
        self.stack_list: List[Stack] = []  # 所有堆垛的列表
        self.point_to_stack: Dict[int, Stack] = {}  # Point.idx -> Stack 映射
