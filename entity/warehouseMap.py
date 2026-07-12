from typing import List, Tuple

from entity.station import Station
from entity.point import Point

class WarehouseMap:
    """仓库地图类,包含仓库的尺寸、位置之间的距离计算等信息

    经典 picker 块状布局:
      - 工作区(顶部): workstation 独占 y=0 行(均布若干工作站);
        机器人起始行紧随其后, 机器人行与第一行 stack 的 y 间距 = 仓库配置之和
        (length_block_number + width_block_number), 中间为缓冲行.
      - 存储区(下方):
          * 纵向 pick aisle 沿 y 贯穿; x 方向按 "过道 | 货架 货架 | 过道 | 货架 货架 ..."
            排列(双面货架, 两列 stack 夹一条纵向过道);
          * 横向 cross aisle 沿 x 把存储区横切成 width_block_number 个等高块,
            块高 = block_width 行, 块间 1 行 cross aisle; 列向被纵向主通道切成
            length_block_number 段. 每个块尺寸完全一致.
    """

    def __init__(self, warehouse_block_width: int, warehouse_block_length: int,
                 warehouse_block_height: int, warehouse_length_block_number: int,
                 warehouse_width_block_number: int, workstation_num, workstation_rows: int = 3,
                 layout_mode: str = "classic", stack_grid_shape: Tuple[int, int] = (0, 0),
                 storage_gap_rows: int = 0):
        """
        构造函数
        :param warehouse_block_width: 仓库块的宽度(块内 stack 行数)
        :param warehouse_block_length: 仓库块的长度(块内列数, 含 pick aisle 与货架列)
        :param warehouse_block_height: 仓库块的高度(货架层数, 不参与平面坐标)
        :param warehouse_length_block_number: x方向的块的数量
        :param warehouse_width_block_number: y方向块的数量
        """
        self.warehouse_block_length = warehouse_block_length
        self.warehouse_block_width = warehouse_block_width
        self.warehouse_block_height = warehouse_block_height
        self.warehouse_length_block_number = warehouse_length_block_number
        self.warehouse_width_block_number = warehouse_width_block_number
        self.workstation_rows = workstation_rows
        self.layout_mode = str(layout_mode or "classic").lower()
        self.stack_grid_shape = tuple(int(v) for v in (stack_grid_shape or (0, 0)))

        # 经典 picker 块内列结构固定为:
        #   pick | rack | rack | pick | rack | rack | pick
        # 即 3 条 pick aisle + 4 列双面货架。
        # 这里保留原始配置值 warehouse_block_length 以兼容外部配置,
        # 但实际平面布局使用固定块内列数。
        self.block_pick_aisles = 3
        self.block_rack_columns = 4
        self.block_internal_length = self.block_pick_aisles + self.block_rack_columns

        self.robot_row = 1
        if self.layout_mode in {"middle_stack_grid", "stack_grid"}:
            stack_cols = max(1, int(self.stack_grid_shape[0] or warehouse_length_block_number))
            stack_rows = max(1, int(self.stack_grid_shape[1] or warehouse_width_block_number))
            self.stack_grid_shape = (int(stack_cols), int(stack_rows))
            self.work_to_stack_gap = int(storage_gap_rows)
            # station y=0, robot y=1, then fixed empty aisle rows before storage.
            self.storage_start_y = int(self.robot_row + 1 + self.work_to_stack_gap)
            self.work_zone_rows = int(self.storage_start_y)
            self.warehouse_length = int(stack_cols + ((stack_cols + 1) // 2) + 1)
            self.storage_area_width = int(stack_rows)
            self.warehouse_width = int(self.work_zone_rows + self.storage_area_width)
            self.middle_stack_x_positions = self._build_middle_stack_x_positions(stack_cols)
        else:
            # 工作区高度: y=0 工作站行, y=1 机器人行, 机器人行到首行 stack 的 y 间距 = 仓库配置之和.
            # 即首行 stack 位于 y = robot_row(=1) + (length_block_number + width_block_number).
            self.work_to_stack_gap = (self.warehouse_length_block_number
                                      + self.warehouse_width_block_number)
            self.storage_start_y = self.robot_row + self.work_to_stack_gap
            self.work_zone_rows = self.storage_start_y

            # 存储区尺寸: 行方向 = width_block_number 个块(每块 block_width 行) + 块间/首尾 cross aisle.
            # cross aisle 数量 = width_block_number + 1 (块前各 1 行).
            self.storage_area_width = ((self.warehouse_block_width + 1)
                                       * self.warehouse_width_block_number + 1)
            # 列方向 = length_block_number 段, 每段内固定 7 列
            # (3 条 pick aisle + 4 列货架), 段间/首尾 1 列纵向主通道.
            self.warehouse_length = ((self.block_internal_length + 1)
                                     * self.warehouse_length_block_number + 1)
            # 总高度(y) = 工作区行数 + 存储区行数
            self.warehouse_width = self.work_zone_rows + self.storage_area_width
        self.warehouse_node_number = self.warehouse_width * self.warehouse_length

        # 初始化节点列表
        self.point_list: List[Point] = []  # 所有点的列表
        self.pod_list: List[Point] = []    # 所有货架(stack)点的列表
        self.node_distance_matrix = None
        self.workstation_nums = workstation_num
        self.workStation_list: List[Station] = []  # 所有工作站的列表
        self.workPoint: List[Point] = []  # 所有工作站对应的点的列表
        self.id_to_Point = {}  # 点id到点对象的映射
        self._initialize_nodes()
        self._initialize_node_distance_matrix()

    @staticmethod
    def _build_middle_stack_x_positions(stack_cols: int) -> List[int]:
        positions: List[int] = []
        x = 1
        for col in range(max(0, int(stack_cols))):
            positions.append(int(x))
            x += 1
            if col % 2 == 1:
                x += 1
        return positions

    def _column_kind(self, x: int) -> str:
        """返回某列在存储区的角色: 'vert'(纵向主通道) / 'pick'(块内拣选通道) / 'rack'(货架列)."""
        if self.layout_mode in {"middle_stack_grid", "stack_grid"}:
            return "rack" if int(x) in set(getattr(self, "middle_stack_x_positions", []) or []) else "pick"
        seg = self.block_internal_length + 1
        # 主通道: 每段边界(x % seg == 0)
        if x % seg == 0:
            return "vert"
        # 块内列偏移(0 .. block_internal_length-1)
        inner = (x % seg) - 1
        # 块内按 "pick, rack, rack" 周期: 每 3 列首列为 pick aisle, 其余为货架
        if inner % 3 == 0:
            return "pick"
        return "rack"

    def _row_kind(self, y: int) -> str:
        """返回某行的角色: 'work'(工作区) / 'cross'(横向通道) / 'rack'(货架行)."""
        if y < self.work_zone_rows:
            return "work"
        if self.layout_mode in {"middle_stack_grid", "stack_grid"}:
            return "rack"
        ry = y - self.work_zone_rows
        # 存储区行方向按 "cross + block_width 行" 周期, cross 在每块之前
        if ry % (self.warehouse_block_width + 1) == 0:
            return "cross"
        return "rack"

    def _initialize_nodes(self):
        """初始化节点列表(经典 picker 块状布局)"""
        # 工作站 x 坐标: y=0 行均布
        workstation_x_coords = set()
        if self.workstation_nums > 0:
            if self.workstation_nums == 1:
                workstation_x_coords.add((self.warehouse_length - 1) // 2)
            else:
                spacing = (self.warehouse_length - 1) / (self.workstation_nums - 1)
                for k in range(self.workstation_nums):
                    workstation_x_coords.add(round(k * spacing))

        for i in range(self.warehouse_node_number):
            x = i % self.warehouse_length
            y = i // self.warehouse_length

            row = self._row_kind(y)
            if y == 0 and x in workstation_x_coords:
                node_type = 4   # 工作站
            elif row == "work":
                # 工作区(机器人行/缓冲行): 视为通道, 可通行
                node_type = 2
            elif row == "cross":
                node_type = 2   # 横向通道
            else:
                # 存储区货架行: 看列角色
                col = self._column_kind(x)
                if col == "rack":
                    node_type = 3   # 货架(stack)
                else:
                    node_type = 2   # 纵向主通道 / 块内 pick aisle

            point = Point(x, y, i, node_type)

            if node_type == 3:
                self.pod_list.append(point)
            elif node_type == 4:
                self.workPoint.append(point)
            self.point_list.append(point)
        self.id_to_Point = {i: point for i, point in enumerate(self.point_list)}

    def robot_start_x_coords(self, robot_num: int) -> List[int]:
        count = max(0, int(robot_num))
        if count <= 0:
            return []
        if count == 1:
            return [int((self.warehouse_length - 1) // 2)]
        spacing = float(self.warehouse_length - 1) / float(count - 1)
        coords: List[int] = []
        for k in range(count):
            x = int(round(k * spacing))
            if x not in coords:
                coords.append(x)
        while len(coords) < count:
            for x in range(self.warehouse_length):
                if x not in coords:
                    coords.append(int(x))
                    break
        return coords[:count]

    def _initialize_node_distance_matrix(self):
        """初始化节点距离矩阵"""
        if not self.point_list:
            raise ValueError("PointList 不能为空,初始化失败!")

        self.node_distance_matrix = [[0] * self.warehouse_node_number
                                     for _ in range(self.warehouse_node_number)]

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
