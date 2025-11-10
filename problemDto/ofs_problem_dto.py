from typing import List
from entity.warehouseMap import WarehouseMap

from entity.order import Order
from entity.robot import Robot
from entity.tote import Tote
from entity.station import Station
from dataclasses import dataclass
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
        self.station_num: int = 0  # 拣选站数量
        self.station_list: List[Station] = []  # 拣选站信息
        self.skus_num: int = 0  # SKUs数量
        self.skus_list: List[int] = []  # SKUs信息
        self.p1: int = 0  # 订单出库取点数量
        self.d1: int = 0  # 订单出库送点数量
        self.p2: int = 0  # 订单入库取点数量
        self.d2: int = 0  # 订单入库送点数量
        self.w: int = 0  # 机器人起点数量
        self.n: int = 0  # 订单点总数
        self.node_num: int = 0  # 搬运层需要经过的点总数
