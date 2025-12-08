
from typing import List, Dict


class SKUs:
    """SKUs实体类"""

    def __init__(self, sku_id: int = 0, weight: float = 0,
                 storeToteList: List[int] = None,
                 storeQuantityList: List[int] = None,
                 tote_quantity_map: Dict[int, int] = None):
        self.id = sku_id  # SKUs的id
        self.weight = weight  # SKU的重量
        self.storeToteList: List[int] = storeToteList if storeToteList is not None else []  # 存储该SKU的toteID的列表
        self.storeQuantityList: List[
            int] = storeQuantityList if storeQuantityList is not None else []  # 存储该SKU在对应tote中的数量列表
        self.tote_quantity_map: Dict[
            int, int] = tote_quantity_map if tote_quantity_map is not None else {}  # tote id到数量的映射

    def __str__(self):
        return f"SKUs(id={self.id})"