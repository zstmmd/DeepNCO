import random
from typing import List, Dict, Set
from datetime import datetime, timedelta

import config.ofs_config

from entity.warehouseMap import WarehouseMap
from entity.robot import Robot
from entity.order import Order
from entity.SKUs import SKUs
from entity.station import Station
from entity.tote import Tote
from entity.point import Point
from entity.MainBatch import MainBatch
from problemDto.ofs_problem_dto import OFSProblemDTO
from config.ofs_config import OFSConfig


class CreateOFSProblem:

    @staticmethod
    def create_ofs_problem(
            warehouse_length_block_number: int,
            warehouse_width_block_number: int,
            robot_num: int,
            batch_num: int,
            order_num: int,
            skus_num: int,
            tote_num: int,
            station_num: int,
            workstation_rows: int
    ) -> OFSProblemDTO:
        """
        构造并返回一个 OFSProblemDTO 实例。
        """
        random.seed(OFSConfig.RANDOM_SEED)
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
        orders = CreateOFSProblem._generate_orders(
            max_sku_types_per_order=3,
            max_quantity_per_sku=5,
            num_orders=order_num,
            num_sku_types=skus_num
        )
        ofs_problem_dto.order_list = orders

        # ----------------------------------------------------
        # 4.5. 创建 batchnum个MainBatch (大批次)
        # ----------------------------------------------------
        main_batches: List[MainBatch] = []

        if batch_num > 0 and orders:
            random.shuffle(orders)  # 随机打乱订单列表

            # 将订单尽可能均匀地分配到每个批次中
            orders_per_batch = len(orders) // batch_num
            remainder = len(orders) % batch_num
            current_order_idx = 0

            for i in range(batch_num):
                # 为当前批次确定订单数量
                num_in_batch = orders_per_batch + (1 if i < remainder else 0)

                if num_in_batch == 0:
                    continue  # 如果订单数少于批次数，则不创建空批次

                # 获取分配给该批次的订单
                batch_orders = orders[current_order_idx: current_order_idx + num_in_batch]
                current_order_idx += num_in_batch
                main_batch_id = f"MAIN_BATCH_{i}"
                # 创建并配置 MainBatch 实例
                main_batch = MainBatch(f"MAIN_BATCH_{i}",batch_orders)

                for order in batch_orders:
                    order.batch_id = main_batch_id
                # 设置一个随机的齐套窗口时间
                main_batch.kit_delivery_window=OFSConfig.KIT_DELIVERY_WINDOW
                main_batches.append(main_batch)


        ofs_problem_dto.main_batch_list = main_batches

        ofs_problem_dto.main_batch = main_batch

        # 5. 创建料箱(Totes)
        ofs_problem_dto.tote_num = tote_num
        tote_list: List[Tote] = []

        available_points: List[Point] = list(map_.pod_list)
        if not available_points:
            raise ValueError("地图中没有 'pod' (type 3) 节点，无法放置料箱。")
        unassigned_skus = set(skus_list_obj)
        # 关键：定义 3D 堆叠。
        # 结构: Dict[point_idx, List[Tote]]
        # List[0] 是 layer 0 (底部), List[-1] 是顶部
        bin_stacks: Dict[int, List[Tote]] = {}

        current_tote_id = 0
        while current_tote_id < tote_num and available_points:
            point = random.choice(available_points)

            # 随机决定在此处堆叠几层
            stack_height = random.randint(1, map_.warehouse_block_height)

            if point.idx not in bin_stacks:
                bin_stacks[point.idx] = []

            for layer in range(stack_height):
                if current_tote_id >= tote_num: break
                if len(bin_stacks[point.idx]) >= map_.warehouse_block_height:
                    break
                tote = Tote()
                tote.id = current_tote_id

                # 完善的库存填充逻辑
                num_skus_in_tote = random.randint(1, 2)
                tote_skus: List[SKUs] = []

                # 优先从 unassigned_skus 中选择 SKU
                for _ in range(num_skus_in_tote):
                    if unassigned_skus:
                        sku_to_add = unassigned_skus.pop()
                        tote_skus.append(sku_to_add)
                    else:
                        # 所有SKU都已分配过，从全部SKU中随机选择一个（确保不与当前tote中已有的SKU重复）
                        available_for_tote = [s for s in skus_list_obj if s not in tote_skus]
                        if available_for_tote:
                            sku_to_add = random.choice(available_for_tote)
                            tote_skus.append(sku_to_add)
                        else:
                            break  # 没有更多不重复的SKU可选了

                if not tote_skus:
                    continue  # 如果未能为tote分配任何SKU，则不创建此tote

                tote.skus_list = tote_skus
                tote.capacity = [random.randint(10, 50) for _ in tote_skus]
                #tote的sku_quantity_map
                tote.sku_quantity_map={sku.id:cap for sku,cap in zip(tote_skus,tote.capacity)}
                # 更新SKU的storeToteList属性
                for sku_in_tote in tote.skus_list:
                    sku_in_tote.storeToteList.append(tote.id)
                    sku_in_tote.storeQuantityList.append(tote.sku_quantity_map[sku_in_tote.id])
                    sku_in_tote.tote_quantity_map={tote.id:tote.sku_quantity_map[sku_in_tote.id]}
                tote.store_point = point
                tote.layer = layer

                bin_stacks[point.idx].append(tote)
                tote_list.append(tote)
                current_tote_id += 1

                # 如果当前堆叠点已满，从可用点中移除
            if point in available_points and len(bin_stacks[point.idx]) >= map_.warehouse_block_height:
                available_points.remove(point)

         # 新增：检查是否有SKU未被分配
        if unassigned_skus:
            print(f"警告: 料箱数量不足，有 {len(unassigned_skus)} 个SKU未能存放入任何料箱。")

        if current_tote_id < tote_num:
            print(f"警告: 存储点不足或分配提前终止。只创建了 {current_tote_id} / {tote_num} 个料箱。")

        ofs_problem_dto.tote_list = tote_list
        #id2tote映射
        ofs_problem_dto.id_to_tote={tote.id:tote for tote in tote_list}
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

        ofs_problem_dto.n = len(unique_sku_ids)
        ofs_problem_dto.node_num = map_.warehouse_node_number

        return ofs_problem_dto

    @staticmethod
    def _generate_orders(max_sku_types_per_order: int,
                         max_quantity_per_sku: int,
                         num_orders: int,
                         num_sku_types: int) -> List[Order]:
        """
        在 Python 中实现的订单生成器

        """

        orders: List[Order] = []

        sku_types = list(range(num_sku_types))  # [0, 1, ..., skus_num-1]

        for i in range(num_orders):
            order = Order(i)

            base_time = datetime.now()
            random_minutes = random.randint(0, 1440)
            order.order_in_time = base_time - timedelta(minutes=random_minutes)

            sku_types_in_order = random.randint(1, max_sku_types_per_order)

            random.shuffle(sku_types)
            skus_for_this_order = sku_types[:sku_types_in_order]

            order_product_id_list: List[int] = []

            total_skus = 0

            for sku_id in skus_for_this_order:
                quantity = random.randint(1, max_quantity_per_sku)
                order_product_id_list.extend([sku_id] * quantity)
                total_skus += quantity

            order.order_product_id_list = order_product_id_list
            order.order_skus_number = total_skus
            order.status = "pending"

            orders.append(order)

        return orders


# --- 如何使用 (示例) ---
if __name__ == '__main__':

    print("开始创建 OFS 问题实例...")

    problem_dto = CreateOFSProblem.create_ofs_problem(
        warehouse_length_block_number=5,
        warehouse_width_block_number=5,
        robot_num=3,
        batch_num=3,
        order_num=20,
        skus_num=50,
        workstation_rows=3,
        tote_num=200,
        station_num=2
    )

    print("\n--- 问题实例创建完成 ---")
    print(f"地图尺寸 (LxW): {problem_dto.map.warehouse_length} x {problem_dto.map.warehouse_width}")
    print(f"机器人数量: {len(problem_dto.robot_list)}")
    print(f"订单数量: {len(problem_dto.order_list)}")
    print(f"料箱数量 (Totes): {len(problem_dto.tote_list)}")
    print(f"工作站数量: {len(problem_dto.station_list)}")

    # --- 验证 MainBatch 是否已创建 ---
    if hasattr(problem_dto, 'main_batch'):
        print(f"\nMainBatch ID: {problem_dto.main_batch.id}")
        print(f"  > 包含订单数量: {len(problem_dto.main_batch.orders)}")
        print(f"  > 齐套窗口 (开始): {problem_dto.main_batch.kit_delivery_window_start}")  #
    else:
        print("\n错误: MainBatch 未被创建。")