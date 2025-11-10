from datetime import datetime
from typing import List


class Order:
    """订单类"""

    def __init__(self):
        self.order_id: str = ""  # 订单ID
        self.order_skus_number: int = 0  # 订单SKUs数量
        self.order_product_id_list: List[int] = []  # 订单SKUs的id列表
        self.order_in_time: datetime = None  # 订单入库时间
        self.order_out_time: datetime = None  # 订单出库时间
        self.status: str = ""  # 订单状态 (例如: "pending", "shipped", "delivered", "canceled")

    def __str__(self):
        return (f"Order(order_id='{self.order_id}', "
                f"order_skus_number={self.order_skus_number}, "
                f"order_product_id_list={self.order_product_id_list}, "
                f"order_in_time={self.order_in_time}, "
                f"order_out_time={self.order_out_time}, "
                f"status='{self.status}')")