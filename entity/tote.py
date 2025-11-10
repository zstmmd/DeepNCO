from typing import List
from SKUs import SKUs
from point import Point
from dataclasses import dataclass
@dataclass
class Tote:
    """料箱类"""

    def __init__(self):
        self.id: int = 0  # 料箱的id
        self.skus_list: List[SKUs] = []   # 料箱包括的SKUs
        self.capacity: List[int] = []  # 料箱中每个SKU的数量
        self.store_point: Point = None  # 料箱的存放坐标
        self.layer: int = 0  # 层高
        self.status: int = 0  # 料箱状态，0表示空闲，1表示使用中

