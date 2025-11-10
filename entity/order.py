from datetime import datetime
from typing import List
from entity.SKUs import SKUs

class Order:
    """订单类"""

    def __init__(self,order_id: int):
        self.order_id: int = order_id  # 订单的id
        self.order_skus_number: int = 0  # 订单SKUs数量
        self.order_product_id_list: List[int] = []  # 订单SKUs的id列表
        self.order_in_time: datetime = None  # 订单入库时间
        self.order_out_time: datetime = None  # 订单出库时间
        self.batch_id: str = ""  # 订单所属批次ID
        self.unique_sku_list:List[SKUs]=[] # 订单中不同SKU的列表
        self.status: str = ""  # 订单状态 (例如: "pending", "shipped", "delivered", "canceled")

    def __str__(self):
        return (f"Order(order_id='{self.order_id}', "
                f"order_skus_number={self.order_skus_number}, "
                f"order_product_id_list={self.order_product_id_list}, "
                f"order_in_time={self.order_in_time}, "
                f"order_out_time={self.order_out_time}, "
                f"status='{self.status}')")