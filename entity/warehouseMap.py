from typing import List

from entity.station import Station
from entity.point import Point


class WarehouseMap:
    """仓库地图类,包含仓库的尺寸、位置之间的距离计算等信息"""

    def __init__(self, warehouse_block_width: int, warehouse_block_length: int,
                 warehouse_block_height: int, warehouse_length_block_number: int,
                 warehouse_width_block_number: int,workstation_num,workstation_rows: int =3):
        """
        构造函数
        :param warehouse_block_width: 仓库块的宽度
        :param warehouse_block_length: 仓库块的长度
        :param warehouse_block_height: 仓库块的高度
        :param warehouse_length_block_number: x方向的块的数量
        :param warehouse_width_block_number: y方向块的数量
        """
        self.warehouse_block_length = warehouse_block_length
        self.warehouse_block_width = warehouse_block_width
        self.warehouse_block_height = warehouse_block_height
        self.warehouse_length_block_number = warehouse_length_block_number
        self.warehouse_width_block_number = warehouse_width_block_number
        self.workstation_rows = workstation_rows
        # 更新：将仓库尺寸计算分为存储区和整个仓库
        self.storage_area_width = (self.warehouse_block_width + 1) * warehouse_width_block_number + 1
        self.warehouse_length = (self.warehouse_block_length + 1) * warehouse_length_block_number + 1
        self.warehouse_width = self.storage_area_width + self.workstation_rows  # 总宽度 = 存储区宽度 + 工作站行数
        self.warehouse_node_number = self.warehouse_width * self.warehouse_length


        # 初始化节点列表
        self.point_list: List[Point] = []  #所有点的列表
        self.pod_list: List[Point] = []    #所有货架点的列表
        self.node_distance_matrix = None
        self.workstation_nums=workstation_num
        self.workStation_list: List[Station] = []  #所有工作站的列表
        self.workPoint: List[Point] = []  #所有工作站对应的点的列表
        self._initialize_nodes()
        self._initialize_node_distance_matrix()

    def _initialize_nodes(self):
        """初始化节点列表"""
        workstation_x_coords = set()
        if self.workstation_nums > 0:
            if self.workstation_nums == 1:
                # 如果只有一个工作站，放在仓库中间
                workstation_x_coords.add((self.warehouse_length - 1) // 2)
            else:
                # 均匀分布多个工作站，覆盖整个仓库宽度
                spacing = (self.warehouse_length - 1) / (self.workstation_nums - 1)
                for k in range(self.workstation_nums):
                    workstation_x_coords.add(round(k * spacing))
        for i in range(self.warehouse_node_number):
            x = i % self.warehouse_length
            y = i // self.warehouse_length

            # 确定节点类型
            if y == 0 and x in workstation_x_coords:
                node_type = 4   # 工作站
            elif y % (self.warehouse_block_width + 1) == 0 or x % (self.warehouse_block_length + 1) == 0:
                # aisle
                node_type = 2
            elif y>= self.workstation_rows:
                # pod
                node_type = 3

            point = Point(x, y, i, node_type)

            # 将type=3的节点加入pod_list
            if node_type == 3:
                self.pod_list.append(point)
                # 新增：将工作站节点加入workPoint列表
            elif node_type == 4:
                self.workPoint.append(point)
            self.point_list.append(point)

    def _initialize_node_distance_matrix(self):
        """初始化节点距离矩阵"""
        if not self.point_list:
            raise ValueError("PointList 不能为空,初始化失败!")

        # 创建距离矩阵
        self.node_distance_matrix = [[0] * self.warehouse_node_number
                                     for _ in range(self.warehouse_node_number)]

        # 计算节点之间的曼哈顿距离
        for i in range(self.warehouse_node_number):
            for j in range(self.warehouse_node_number):
                if i != j:
                    point1 = self.point_list[i]
                    point2 = self.point_list[j]
                    self.node_distance_matrix[i][j] = (
                            abs(point1.x - point2.x) + abs(point1.y - point2.y)
                    )

    def __str__(self):
        return (f"WarehouseMap(warehouse_length={self.warehouse_length}, "
                f"warehouse_width={self.warehouse_width}, "
                f"warehouse_node_number={self.warehouse_node_number}, "
                f"warehouse_block_length={self.warehouse_block_length}, "
                f"warehouse_block_width={self.warehouse_block_width}, "
                f"warehouse_block_height={self.warehouse_block_height}, "
                f"warehouse_length_block_number={self.warehouse_length_block_number}, "
                f"warehouse_width_block_number={self.warehouse_width_block_number}, "
                f"pod_list_size={len(self.pod_list)})")