from  entity.order import Order
from typing import List
class MainBatch:
    """
    “一个大批次的订单”
    这是问题的输入。
    """
    def __init__(self, main_batch_id: str, orders: List[Order]):
        self.id = main_batch_id
        self.orders = orders
        self.kit_delivery_window= None