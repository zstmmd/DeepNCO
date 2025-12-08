from typing import List

from entity.SKUs import SKUs

from entity.point import Point

from dataclasses import dataclass
@dataclass
class Tote:
    """料箱类"""

    def __init__(self,tote_id: int):
        self.id: int = tote_id # 料箱的id
        self.skus_list: List[SKUs] = []   # 料箱包括的SKUs
        self.capacity: List[int] = []  # 料箱中每个SKU的数量
        self.store_point: Point = None  # 料箱的存放坐标
        self.bin_statck: List[int] = []  # 料箱id的堆叠顺序,List[0]是底部, List[-1]是顶部
        self.max_layer:int=0 #所在点的最大层高
        self.is_top:bool=False # 是否为顶部料箱
        self.layer: int = 0  # 层高
        self.status: int = 0  # 料箱状态，0表示空闲，1表示使用中
        self.sku_quantity_map={} # SKU id到数量的映射
    def __repr__(self):
        return f"Tote(id={self.id}, skus={len(self.skus_list)})"

    def __hash__(self):
        # 使用 id 进行哈希，确保可以在 set 中去重
        return hash(self.id)

    def __eq__(self, other):
        # 判断两个 Tote 是否相等，只看 ID
        if isinstance(other, Tote):
            return self.id == other.id
        return False