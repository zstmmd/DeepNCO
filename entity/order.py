from datetime import datetime
from typing import Dict, List

from entity.SKUs import SKUs
from entity.point import Point


class Order:
    """Order/BOM entity."""

    def __init__(self, order_id: int):
        self.order_id: int = order_id
        self.order_skus_number: int = 0
        self.order_product_id_list: List[int] = []
        self.order_in_time: datetime = None
        self.order_out_time: datetime = None

        self.unique_sku_list: List[SKUs] = []

        self.sku_storage_points: List[Point] = []
        self.point_sku_quantity: Dict[int, Dict[int, int]] = {}
        self.status: str = ""
        self.bom_completion_time: float = 0.0

        self.est_sec: float = 0.0
        self.kitting_span_limit_sec: float = 0.0
        self.lst_sec: float = 0.0
        self.total_qty: int = 0
        self.unique_sku_count: int = 0
        self.deadline_buffer_sec: float = 0.0

    def __str__(self):
        return (
            f"Order(order_id='{self.order_id}', "
            f"order_skus_number={self.order_skus_number}, "
            f"order_product_id_list={self.order_product_id_list}, "
            f"est_sec={self.est_sec}, "
            f"kitting_span_limit_sec={self.kitting_span_limit_sec}, "
            f"lst_sec={self.lst_sec}, "
            f"order_in_time={self.order_in_time}, "
            f"order_out_time={self.order_out_time}, "
            f"status='{self.status}')"
        )
