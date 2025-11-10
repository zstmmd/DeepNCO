
from typing import List
class SKUs:
    """SKUs实体类"""

    def __init__(self, sku_id: int = 0,weight: float = 0):
        self.id = sku_id  # SKUs的id
        self.storeToteList : List[int]=[]  # 存储该SKU的toteID的列表
        self.weight=weight  # SKU的重量
        self.storeQuantityList:List[int]=[] # 存储该SKU在对应tote中的数量列表
        self.tote_quantity_map={} # tote id到数量的映射

    def __str__(self):
        return f"SKUs(id={self.id})"