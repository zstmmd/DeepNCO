from typing import Tuple


class Point:
    """节点类"""

    def __init__(self, x: int = 0, y: int = 0, idx: int = 0, node_type: int = 0):
        """
        :param x: Node在仓库布局中的横坐标
        :param y: Node在仓库布局中的纵坐标
        :param idx: Node的编号
        :param node_type: Node的类型 (0: pick in, 1: pick out, 2: aisle, 3: pod)
        """
        self.x = x
        self.y = y
        self.idx = idx
        self.type = node_type
        self.task_type = "0"
        self.service_time = 0.0
        self.start_service_time = 0.0
        self.end_service_time = 0.0

    @staticmethod
    def get_idx_by_xy(map_length: int, x: int, y: int) -> int:
        """
        根据Node的横纵坐标与地图的长度返回Node的编号
        :param map_length: 地图的长度
        :param x: Node的横坐标
        :param y: Node的纵坐标
        :return: Node的编号
        """
        return x + y * map_length

    @staticmethod
    def get_xy_by_idx(map_length: int, idx: int) -> Tuple[int, int]:
        """
        根据Node的编号与地图的长度返回Node的横纵坐标
        :param map_length: 地图的长度
        :param idx: Node的编号
        :return: Node的横纵坐标 (x, y)
        """
        x = idx % map_length
        y = idx // map_length
        return x, y

    def __str__(self):
        return (f"Point(x={self.x}, y={self.y}, idx={self.idx}, "
                f"task_type='{self.task_type}', service_time={self.service_time}, "
                f"start_service_time={self.start_service_time}, "
                f"end_service_time={self.end_service_time}, type={self.type})")