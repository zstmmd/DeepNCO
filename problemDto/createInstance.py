import random
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta

import config.ofs_config
from entity.stack import Stack

from entity.warehouseMap import WarehouseMap
from entity.robot import Robot
from entity.order import Order
from entity.SKUs import SKUs
from entity.station import Station
from entity.tote import Tote
from entity.point import Point
import os

import numpy as np
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class CreateOFSProblem:
    @staticmethod
    def generate_problem_by_scale(scale: str = "SMALL", seed: int = OFSConfig.RANDOM_SEED) -> OFSProblemDTO:
        """
        根据规模生成标准算例
        :param scale: "SMALL", "MEDIUM", "LARGE"
        :param seed: 随机种子，用于复现
        """
        # --- 规模参数配置字典 ---
        # 格式: {
        #   "map_size": (len_blocks, width_blocks),
        #   "resources": (robot_num, station_num, tote_num),
        #   "data": (order_num, sku_num),
        #   "bom_complexity": (max_types, max_qty) -> 影响订单BOM的复杂度
        # }
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        configs = {
            "SMALL": {
                "map_size": (4, 4),  # 较小的地图
                "resources": (3, 2, 200),  # 3个机器人, 2个工作站, 200个料箱
                "data": (2, 60),  # 3个BOM, 30种SKU
                "bom_complexity": (20, 5)  # 每个订单最多10种SKU，每种最多5个
            },
            "MEDIUM": {
                "map_size": (8, 6),
                "resources": (8, 4, 800),  # 8个机器人, 4个工作站, 800个料箱
                "data": (10, 100),  # 10个BOM, 50种SKU
                "bom_complexity": (40, 10)
            },
            "LARGE": {
                "map_size": (12, 10),
                "resources": (20, 8, 2000),  # 20个机器人, 8个工作站, 2000个料箱
                "data": (20, 200),  # 20个bom, 100种SKU
                "bom_complexity": (60, 20)
            }
        }

        cfg = configs.get(scale.upper(), configs["SMALL"])

        map_L, map_W = cfg["map_size"]
        rob_n, st_n, tote_n = cfg["resources"]
        ord_n, sku_n = cfg["data"]
        bom_types, bom_qty = cfg["bom_complexity"]

        print(f">>> 生成 [{scale}] 规模实例 | Seed: {seed}")
        print(f"    Map: {map_L}x{map_W} blocks | Robots: {rob_n} | Stations: {st_n}")
        print(f"    Orders: {ord_n} | SKUs: {sku_n} | Totes: {tote_n}")

        return CreateOFSProblem.create_ofs_problem(
            warehouse_length_block_number=map_L,
            warehouse_width_block_number=map_W,
            robot_num=rob_n,
            order_num=ord_n,
            skus_num=sku_n,
            tote_num=tote_n,
            station_num=st_n,
            workstation_rows=3,  # 默认参数
            bom_config=(bom_types, bom_qty)
        )
    @staticmethod
    def create_ofs_problem(
            warehouse_length_block_number: int,
            warehouse_width_block_number: int,
            robot_num: int,
            order_num: int,
            skus_num: int,
            tote_num: int,
            station_num: int,
            workstation_rows: int,
            bom_config: Tuple[int, int] = (10, 5)  # (max_types, max_qty)
    ) -> OFSProblemDTO:
        """
        构造并返回一个 OFSProblemDTO 实例。
        """
       
        ofs_problem_dto = OFSProblemDTO()

        # 1. 创建地图 (使用 warehouseMap.py 的构造函数)
        map_ = WarehouseMap(
            OFSConfig.WAREHOUSE_BLOCK_WIDTH,
            OFSConfig.WAREHOUSE_BLOCK_LENGTH,
            OFSConfig.WAREHOUSE_BLOCK_HEIGHT,
            warehouse_length_block_number,
            warehouse_width_block_number,
            station_num,  # 传递 station_num
            workstation_rows
        )
        ofs_problem_dto.map = map_

        # 2. 创建 SKUs
        skus_list_obj: List[SKUs] = []
        for i in range(skus_num):
            weight = round(random.uniform(0.1, 5.0), 2)
            sku = SKUs(sku_id=i, weight=weight)
            skus_list_obj.append(sku)

        ofs_problem_dto.skus_num = skus_num

        ofs_problem_dto.skus_list = [sku for sku in skus_list_obj]

        sku_map: Dict[int, SKUs] = {sku.id: sku for sku in skus_list_obj}
        ofs_problem_dto.id_to_sku = sku_map
        # 3. 创建机器人

        ofs_problem_dto.robot_num = robot_num
        robots: List[Robot] = []
        map_length = map_.warehouse_length
        robot_start_x = 1
        robot_start_y = 0

        for i in range(robot_num):
            start_point = None
            while robot_start_x < map_length:
                start_idx = Point.get_idx_by_xy(map_length, robot_start_x, robot_start_y)
                if start_idx >= len(map_.point_list): break

                point = map_.point_list[start_idx]

                start_point = point
                robot_start_x += 2
                break

            if start_point is None:
                start_point = map_.point_list[0]  # 回退

            robot = Robot(
                robot_id=i,
                start_point=start_point,
                max_stack_height=OFSConfig.ROBOT_CAPACITY  #
            )
            robots.append(robot)

        ofs_problem_dto.robot_list = robots

        # 4. 创建订单
        ofs_problem_dto.order_num = order_num
        max_sku_types, max_sku_qty = bom_config
        # orders = CreateOFSProblem._generate_orders(
        #     max_sku_types_per_order=10,
        #     max_quantity_per_sku=5,
        #     num_orders=order_num,
        #     num_sku_types=skus_num
        # )
        orders = CreateOFSProblem._generate_orders(
            max_sku_types_per_order=max_sku_types,
            max_quantity_per_sku=max_sku_qty,
            num_orders=order_num,
            skus_list=skus_list_obj  # 传入实体列表以便选择
        )
        ofs_problem_dto.order_list = orders
        ofs_problem_dto.id_to_order={order.order_id:order for order in orders}
        # ----------------------------------------------------

        # 5. 创建料箱(Totes)
        ofs_problem_dto.tote_num = tote_num
        tote_list: List[Tote] = []
        stack_list: List[Stack] = []
        point_to_stack: Dict[int, Stack] = {}
        available_points: List[Point] = sorted(
            list(map_.pod_list), 
            key=lambda p: p.idx
        )
        if not available_points:
            raise ValueError("地图中没有 'pod' (type 3) 节点，无法放置料箱。")
        hot_sku_count = max(1, int(skus_num * 0.2))
        hot_skus = skus_list_obj[:hot_sku_count]  
        cold_skus = skus_list_obj[hot_sku_count:]
        def sample_sku_by_popularity() -> SKUs:
            """按热门度加权采样 SKU"""
            if random.random() < 0.8:  # 80% 概率选热门品
                return random.choice(hot_skus)
            else:
                return random.choice(cold_skus) if cold_skus else random.choice(hot_skus)

        # 关键：定义 3D 堆叠。
        # 结构: Dict[point_idx, List[Tote]]
        # List[0] 是 layer 0 (底部), List[-1] 是顶部
        bin_stacks: Dict[int, List[Tote]] = {}
        # 初始化所有可用存储点的 Stack 对象
        for point in available_points:
            stack = Stack(
                stack_id=point.idx,
                store_point=point,
                max_height=map_.warehouse_block_height
            )
            stack_list.append(stack)
            point_to_stack[point.idx] = stack
        min_redundancy_per_hot_sku = 5  # 每个热门 SKU 至少在 5 个 Stack 中
        desired_tote_count = max(
            tote_num,
            hot_sku_count * min_redundancy_per_hot_sku  # 动态下界
        )
        unassigned_skus = set(skus_list_obj)
        random.shuffle(stack_list)  # 随机化 Stack 顺序
        # 填充料箱逻辑
        # 我们可以遍历 Stack 列表来填充，而不是原来的 while 循环
        # 为了随机性，可以 shuffle stack_list
        random.shuffle(stack_list)
        current_tote_id = 0
        initial_assignment_stacks = []

        for stack in stack_list:
            if not unassigned_skus or current_tote_id >= desired_tote_count:
                break
            
            # 为该 Stack 创建一个 Tote
            tote = Tote(current_tote_id)
            
            # 从未分配集合中选 1-2 个 SKU (确保快速覆盖)
            num_skus = min(2, len(unassigned_skus))
            selected = random.sample(list(unassigned_skus), num_skus)
            
            tote.skus_list = selected
            tote.capacity = [random.randint(15, 50) for _ in selected]
            tote.sku_quantity_map = {sku.id: cap for sku, cap in zip(selected, tote.capacity)}
            
            # 更新 SKU 反向索引
            for sku in selected:
                sku.storeToteList.append(tote.id)
                sku.storeQuantityList.append(tote.sku_quantity_map[sku.id])
                sku.tote_quantity_map[tote.id] = tote.sku_quantity_map[sku.id]
                unassigned_skus.discard(sku)
            
            # 加入 Stack
            stack.add_tote(tote)
            tote_list.append(tote)
            initial_assignment_stacks.append(stack)
            current_tote_id += 1
        # 重新 Shuffle Stack 列表,确保后续填充的随机性
        random.shuffle(stack_list)

        for stack in stack_list:
            if current_tote_id >= desired_tote_count:
                break
            
            # 随机决定该 Stack 的高度 (1 ~ max_height)
            current_height = stack.current_height
            max_additional = stack.max_height - current_height
            
            if max_additional <= 0:
                continue
            
            # 随机决定要填充的层数 (1 到剩余空间)
            layers_to_fill = random.randint(1, max_additional)
            
            for _ in range(layers_to_fill):
                if current_tote_id >= desired_tote_count:
                    break
                
                tote = Tote(current_tote_id)
                
                # ===== 关键修改: 加权随机采样 SKU =====
                num_skus_in_tote = random.randint(3, 5)  # 增加到 3-5 种
                tote_skus: List[SKUs] = []
                
                for _ in range(num_skus_in_tote):
                    attempts = 0
                    while attempts < 10:  # 防止死循环
                        candidate = sample_sku_by_popularity()
                        # 避免重复
                        if candidate not in tote_skus:
                            tote_skus.append(candidate)
                            break
                        attempts += 1
                
                if not tote_skus:
                    continue
                # 设置库存 (热门品给更多库存)
                tote.capacity = []
                for sku in tote_skus:
                    if sku in hot_skus:
                        qty = random.randint(20, 60)  # 热门品库存更多
                    else:
                        qty = random.randint(10, 40)  # 冷门品适中
                    tote.capacity.append(qty)
                
                tote.skus_list = tote_skus
                tote.sku_quantity_map = {sku.id: cap for sku, cap in zip(tote_skus, tote.capacity)}
                
                # 更新 SKU 反向索引
                for sku in tote_skus:
                    sku.storeToteList.append(tote.id)
                    sku.storeQuantityList.append(tote.sku_quantity_map[sku.id])
                    sku.tote_quantity_map[tote.id] = tote.sku_quantity_map[sku.id]
                
                # 加入 Stack
                stack.add_tote(tote)
                tote_list.append(tote)
                current_tote_id += 1

        # 过滤掉空的 Stack (如果没有生成)
        final_stack_list = [s for s in stack_list if s.current_height > 0]
        
        ofs_problem_dto.tote_list = tote_list
        ofs_problem_dto.id_to_tote = {tote.id: tote for tote in tote_list}
        ofs_problem_dto.stack_list = final_stack_list
        ofs_problem_dto.point_to_stack = point_to_stack
         # 6. 创建工作站

        ofs_problem_dto.station_num = station_num
        station_list: List[Station] = []

        if len(map_.workPoint) != station_num:
            print(f"警告: 创建的工作站节点 ({len(map_.workPoint)}) 与请求的数量 ({station_num}) 不匹配。")

        for i, station_point in enumerate(map_.workPoint):
            station = Station(station_id=i)
            station.point = station_point  # 关联工作站与地图上的点
            station_list.append(station)

        ofs_problem_dto.station_list = station_list

        unique_sku_ids: Set[int] = set()
        for order in orders:
            #更新order的unique_sku_list
            unique_ids_in_order = set(order.order_product_id_list)
            order.unique_sku_list = [sku_map[sku_id] for sku_id in unique_ids_in_order]
            unique_sku_ids.update(order.order_product_id_list)
        # 新增：更新所有order需要的sku的存储点point列表和数量
        sku_storepoint_list=set()
        for order in ofs_problem_dto.order_list:
            point_map = {}
            order.point_sku_quantity = {}

            # 遍历订单所需的所有唯一SKU
            for sku in order.unique_sku_list:
                # 遍历存放该SKU的所有tote
                for tote_id in sku.storeToteList:
                    tote = ofs_problem_dto.id_to_tote.get(tote_id)
                    if tote and tote.store_point:
                        point = tote.store_point
                        point_idx = point.idx

                        # 获取该tote中此sku的数量
                        quantity_in_tote = tote.sku_quantity_map.get(sku.id, 0)
                        if quantity_in_tote == 0:
                            continue

                        # 如果是第一次遇到这个存储点，则记录它
                        if point_idx not in point_map:
                            point_map[point_idx] = point
                            sku_storepoint_list.add(point)
                        # 初始化该点的SKU数量字典
                        if point_idx not in order.point_sku_quantity:
                            order.point_sku_quantity[point_idx] = {}

                        # 累加该点上此SKU的总数量（因为同一位置可能有多层tote）
                        order.point_sku_quantity[point_idx][sku.id] = \
                            order.point_sku_quantity[point_idx].get(sku.id, 0) + quantity_in_tote

            # 从point_map中生成不重复的Point对象列表
            order.sku_storage_points = list(point_map.values())
        ofs_problem_dto.need_points = list(sku_storepoint_list)
        ofs_problem_dto.n = len(unique_sku_ids)
        ofs_problem_dto.node_num = len(ofs_problem_dto.need_points)

        return ofs_problem_dto

    @staticmethod
    def _generate_orders(
            max_sku_types_per_order: int,
            max_quantity_per_sku: int,
            num_orders: int,
            skus_list: List[SKUs]
    ) -> List[Order]:
        """
        生成订单 (BOM)
        :param skus_list: 具体的SKU对象列表，用于随机采样
        """
        orders: List[Order] = []

        # 权重分布：假设 ID 越小的 SKU 越热门 (80/20原则模拟)
        # 这里简单模拟：前20%的SKU被选中的概率大一点
        hot_skus_count = max(1, int(len(skus_list) * 0.2))
        hot_skus = skus_list[:hot_skus_count]
        cold_skus = skus_list[hot_skus_count:]

        for i in range(num_orders):
            order = Order(i)

            # 生成时间
            base_time = datetime.now()
            random_minutes = random.randint(0, 480)  # 8小时内
            order.order_in_time = base_time - timedelta(minutes=random_minutes)

            # 确定 BOM 结构 (SKU 种类数)
            # 并非均匀分布，倾向于较小的订单
            num_types = random.randint(max_sku_types_per_order-5, max_sku_types_per_order)

            # 选品逻辑：70%概率混入热门品
            selected_skus = []
            while len(selected_skus) < num_types:
                if random.random() < 0.7 and hot_skus:
                    s = random.choice(hot_skus)
                elif cold_skus:
                    s = random.choice(cold_skus)
                else:
                    s = random.choice(skus_list)

                if s not in selected_skus:
                    selected_skus.append(s)

            # 确定数量并生成 ID 列表
            order_product_id_list = []
            total_qty = 0

            for sku in selected_skus:
                qty = random.randint(1, max_quantity_per_sku)
                order_product_id_list.extend([sku.id] * qty)
                total_qty += qty

            order.order_product_id_list = order_product_id_list
            order.order_skus_number = total_qty
            order.status = "pending"

            orders.append(order)

        return orders


if __name__ == '__main__':

    # 测试生成不同规模的实例
    scales = ["SMALL", "MEDIUM"]

    for scale in scales:
        print(f"\n{'=' * 20} Testing {scale} Scale {'=' * 20}")
        try:
            dto = CreateOFSProblem.generate_problem_by_scale(scale)

            print(f"Success! Generated {scale} instance.")
            print(f"Order 0 BOM sample:")
            if dto.order_list:
                o0 = dto.order_list[0]
                print(f"  ID: {o0.order_id}, Total Items: {o0.order_skus_number}")
                print(f"  SKU IDs: {o0.order_product_id_list}")

            # 验证堆垛
            print(f"Total Stacks: {len(dto.stack_list)}")
            if dto.stack_list:
                s0 = dto.stack_list[0]
                print(f"  Stack {s0.stack_id} Height: {s0.current_height}/{s0.max_height}")

        except Exception as e:
            print(f"Error generating {scale}: {e}")
            import traceback

            traceback.print_exc()