import math
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from config.ofs_config import OFSConfig
from entity.order import Order
from entity.point import Point
from entity.robot import Robot
from entity.SKUs import SKUs
from entity.stack import Stack
from entity.station import Station
from entity.tote import Tote
from entity.warehouseMap import WarehouseMap
from problemDto.ofs_problem_dto import OFSProblemDTO


class CreateOFSProblem:
    @staticmethod
    def generate_problem_by_scale(scale: str = "SMALL", seed: int = OFSConfig.RANDOM_SEED) -> OFSProblemDTO:
        """
        根据规模生成标准算例。
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        configs = {
            "TEST": {"map_size": (2, 4), "resources": (2, 2, 100), "data": (1, 10), "bom_complexity": (10, 1), "exact_bom_sku_count": 10},
            "GUROBI-S1": {"map_size": (2, 4), "resources": (2, 2, 30), "data": (1, 10), "bom_complexity": (10, 1), "exact_bom_sku_count": 10},
            "SMALL": {"map_size": (4, 4), "resources": (2, 2, 200), "data": (2, 60), "bom_complexity": (20, 5)},
            "SMALL2": {"map_size": (4, 4), "resources": (3, 2, 200), "data": (3, 60), "bom_complexity": (25, 5)},
            "SMALL_ZRICH": {"map_size": (4, 4), "resources": (2, 2, 200), "data": (2, 60), "bom_complexity": (20, 5)},
            "SMALL2_ZRICH": {"map_size": (4, 4), "resources": (3, 2, 200), "data": (3, 60), "bom_complexity": (25, 5)},
            "SMALL3": {"map_size": (4, 4), "resources": (3, 2, 200), "data": (3, 70), "bom_complexity": (30, 5)},
            "SMALL_UNEVEN": {"map_size": (4, 4), "resources": (2, 2, 200), "data": (2, 60), "bom_complexity": (24, 6)},
            "SMALL2_UNEVEN": {"map_size": (4, 4), "resources": (3, 2, 200), "data": (3, 60), "bom_complexity": (28, 6)},
            "SMALL3_UNEVEN": {"map_size": (4, 4), "resources": (3, 2, 200), "data": (3, 70), "bom_complexity": (34, 6)},
            "MEDIUM": {"map_size": (8, 6), "resources": (8, 4, 800), "data": (10, 100), "bom_complexity": (40, 10)},
            "LARGE": {"map_size": (12, 10), "resources": (20, 8, 2000), "data": (20, 200), "bom_complexity": (60, 20)},
        }

        scale_upper = str(scale).upper()
        cfg = configs.get(scale_upper, configs["SMALL"])
        imbalance_profile = None
        if scale_upper.endswith("_ZRICH"):
            imbalance_profile = "zrich"
        elif "UNEVEN" in scale_upper:
            imbalance_profile = "uneven"

        map_L, map_W = cfg["map_size"]
        rob_n, st_n, tote_n = cfg["resources"]
        ord_n, sku_n = cfg["data"]
        bom_types, bom_qty = cfg["bom_complexity"]
        exact_bom_sku_count = int(cfg.get("exact_bom_sku_count", 0))

        print(f">>> 生成 [{scale}] 规模实例 | Seed: {seed}")
        print(f"    Map: {map_L}x{map_W} blocks | Robots: {rob_n} | Stations: {st_n}")
        print(f"    Orders: {ord_n} | SKUs: {sku_n} | Totes: {tote_n}")

        problem = CreateOFSProblem.create_ofs_problem(
            warehouse_length_block_number=map_L,
            warehouse_width_block_number=map_W,
            robot_num=rob_n,
            order_num=ord_n,
            skus_num=sku_n,
            tote_num=tote_n,
            station_num=st_n,
            workstation_rows=3,
            bom_config=(bom_types, bom_qty),
            imbalance_profile=imbalance_profile,
            exact_bom_sku_count=exact_bom_sku_count,
        )
        problem.scale_name = scale_upper
        problem.generator_profile = imbalance_profile or "default"
        return problem

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
            bom_config: Tuple[int, int] = (10, 5),
            imbalance_profile: str = None,
            exact_bom_sku_count: int = 0,
    ) -> OFSProblemDTO:
        """
        构造并返回一个 OFSProblemDTO 实例。
        """
        ofs_problem_dto = OFSProblemDTO()
        ofs_problem_dto.generator_profile = str(imbalance_profile or "default")

        map_ = WarehouseMap(
            OFSConfig.WAREHOUSE_BLOCK_WIDTH,
            OFSConfig.WAREHOUSE_BLOCK_LENGTH,
            OFSConfig.WAREHOUSE_BLOCK_HEIGHT,
            warehouse_length_block_number,
            warehouse_width_block_number,
            station_num,
            workstation_rows,
        )
        ofs_problem_dto.map = map_

        skus_list_obj: List[SKUs] = [SKUs(sku_id=i, weight=round(random.uniform(0.1, 5.0), 2)) for i in range(skus_num)]
        ofs_problem_dto.skus_num = skus_num
        ofs_problem_dto.skus_list = [sku for sku in skus_list_obj]
        sku_map: Dict[int, SKUs] = {sku.id: sku for sku in skus_list_obj}
        ofs_problem_dto.id_to_sku = sku_map

        ofs_problem_dto.robot_num = robot_num
        robots: List[Robot] = []
        map_length = map_.warehouse_length
        robot_start_x = 1
        robot_start_y = 0
        for i in range(robot_num):
            start_point = None
            while robot_start_x < map_length:
                start_idx = Point.get_idx_by_xy(map_length, robot_start_x, robot_start_y)
                if start_idx >= len(map_.point_list):
                    break
                start_point = map_.point_list[start_idx]
                robot_start_x += 2
                break
            if start_point is None:
                start_point = map_.point_list[0]
            robots.append(Robot(robot_id=i, start_point=start_point, max_stack_height=OFSConfig.ROBOT_CAPACITY))
        ofs_problem_dto.robot_list = robots

        ofs_problem_dto.order_num = order_num
        max_sku_types, max_sku_qty = bom_config
        zrich_profile: Dict[str, Any] = {}
        if imbalance_profile == "zrich":
            zrich_profile = CreateOFSProblem._build_exponential_sku_profile(skus_list_obj, skew_lambda=0.02)
            orders = CreateOFSProblem._generate_zrich_orders(max_sku_types, max_sku_qty, order_num, skus_list_obj, zrich_profile)
        else:
            orders = CreateOFSProblem._generate_orders(max_sku_types, max_sku_qty, order_num, skus_list_obj, imbalance_profile)
        if int(exact_bom_sku_count) > 0:
            if int(order_num) != 1:
                raise ValueError("exact_bom_sku_count is intended for single-BOM test cases.")
            exact_count = min(int(exact_bom_sku_count), int(len(skus_list_obj)))
            exact_order = orders[0]
            exact_order.order_product_id_list = [int(sku.id) for sku in skus_list_obj[:exact_count]]
            exact_order.order_skus_number = int(exact_count)
            exact_order.status = "pending"
        ofs_problem_dto.order_list = orders
        ofs_problem_dto.id_to_order = {order.order_id: order for order in orders}

        ofs_problem_dto.tote_num = tote_num
        stack_list: List[Stack] = []
        point_to_stack: Dict[int, Stack] = {}
        available_points: List[Point] = sorted(list(map_.pod_list), key=lambda p: p.idx)
        if not available_points:
            raise ValueError("Map has no 'pod' nodes for tote placement.")
        for point in available_points:
            stack = Stack(stack_id=point.idx, store_point=point, max_height=map_.warehouse_block_height)
            stack_list.append(stack)
            point_to_stack[point.idx] = stack

        if imbalance_profile == "zrich":
            tote_list, final_stack_list, redundancy_summary = CreateOFSProblem._build_zrich_inventory(
                stack_list, skus_list_obj, orders, tote_num, zrich_profile, required_distinct_stacks=4
            )
        else:
            tote_list, final_stack_list = CreateOFSProblem._build_default_inventory(stack_list, skus_list_obj, tote_num, imbalance_profile)
            redundancy_summary = CreateOFSProblem._compute_demanded_sku_redundancy_summary(orders, final_stack_list, target_distinct_stacks=0)

        ofs_problem_dto.tote_list = tote_list
        ofs_problem_dto.id_to_tote = {tote.id: tote for tote in tote_list}
        ofs_problem_dto.stack_list = final_stack_list
        ofs_problem_dto.point_to_stack = point_to_stack

        ofs_problem_dto.station_num = station_num
        station_list: List[Station] = []
        if len(map_.workPoint) != station_num:
            print(f"Warning: created station nodes ({len(map_.workPoint)}) != requested ({station_num}).")
        for i, station_point in enumerate(map_.workPoint):
            station = Station(station_id=i)
            station.point = station_point
            station_list.append(station)
        ofs_problem_dto.station_list = station_list

        unique_sku_ids: Set[int] = set()
        for order in orders:
            unique_ids_in_order = sorted(set(order.order_product_id_list))
            order.unique_sku_list = [sku_map[sku_id] for sku_id in unique_ids_in_order]
            unique_sku_ids.update(order.order_product_id_list)

        sku_storepoint_list = set()
        for order in ofs_problem_dto.order_list:
            point_map = {}
            order.point_sku_quantity = {}
            for sku in order.unique_sku_list:
                for tote_id in sku.storeToteList:
                    tote = ofs_problem_dto.id_to_tote.get(tote_id)
                    if tote is None or tote.store_point is None:
                        continue
                    point = tote.store_point
                    point_idx = point.idx
                    quantity_in_tote = tote.sku_quantity_map.get(sku.id, 0)
                    if quantity_in_tote == 0:
                        continue
                    if point_idx not in point_map:
                        point_map[point_idx] = point
                        sku_storepoint_list.add(point)
                    if point_idx not in order.point_sku_quantity:
                        order.point_sku_quantity[point_idx] = {}
                    order.point_sku_quantity[point_idx][sku.id] = order.point_sku_quantity[point_idx].get(sku.id, 0) + quantity_in_tote
            order.sku_storage_points = sorted(point_map.values(), key=lambda p: p.idx)

        ofs_problem_dto.need_points = sorted(sku_storepoint_list, key=lambda p: p.idx)
        ofs_problem_dto.n = len(unique_sku_ids)
        ofs_problem_dto.node_num = len(ofs_problem_dto.need_points)
        ofs_problem_dto.redundancy_summary = dict(redundancy_summary)
        ofs_problem_dto.generator_summary = {
            "inventory_profile": str(imbalance_profile or "default"),
            "demand_profile_lambda": float(zrich_profile.get("lambda", 0.0)) if zrich_profile else 0.0,
            "required_distinct_stacks": int(redundancy_summary.get("target_distinct_stacks", 0)),
            "demanded_sku_count": int(redundancy_summary.get("demanded_sku_count", 0)),
        }
        if imbalance_profile == "zrich" and float(redundancy_summary.get("demanded_sku_ge_target_share", 0.0)) < 1.0 - 1e-9:
            raise ValueError(f"Z-rich redundancy guarantee violated: {redundancy_summary}")
        return ofs_problem_dto

    @staticmethod
    def _build_exponential_sku_profile(skus_list: List[SKUs], skew_lambda: float = 0.02) -> Dict[str, Any]:
        shuffled = [sku for sku in skus_list]
        random.shuffle(shuffled)
        weights: Dict[int, float] = {}
        for rank, sku in enumerate(shuffled):
            weights[int(sku.id)] = float(math.exp(-float(skew_lambda) * float(rank)))
        total_weight = sum(weights.values()) or 1.0
        probabilities = {sku_id: float(weight / total_weight) for sku_id, weight in weights.items()}
        return {
            "lambda": float(skew_lambda),
            "ranking": [int(sku.id) for sku in shuffled],
            "weights": weights,
            "probabilities": probabilities,
        }

    @staticmethod
    def _weighted_unique_sample(candidate_ids: List[int], weight_by_id: Dict[int, float], sample_size: int) -> List[int]:
        available = [int(x) for x in candidate_ids]
        selected: List[int] = []
        while available and len(selected) < max(0, int(sample_size)):
            weights = [max(1e-12, float(weight_by_id.get(int(sku_id), 1.0))) for sku_id in available]
            chosen = int(random.choices(available, weights=weights, k=1)[0])
            selected.append(chosen)
            available.remove(chosen)
        return selected

    @staticmethod
    def _sample_tote_quantities(tote_skus: List[SKUs], weight_by_id: Dict[int, float]) -> Dict[SKUs, int]:
        max_weight = max([float(weight_by_id.get(int(sku.id), 1.0)) for sku in tote_skus] + [1.0])
        quantities: Dict[SKUs, int] = {}
        for sku in tote_skus:
            popularity = float(weight_by_id.get(int(sku.id), 1.0)) / max_weight
            qty_low = max(8, int(round(10 + 15 * popularity)))
            qty_high = max(qty_low + 4, int(round(24 + 36 * popularity)))
            quantities[sku] = int(random.randint(qty_low, qty_high))
        return quantities

    @staticmethod
    def _register_tote(stack: Stack, tote_id: int, sku_quantity_map: Dict[SKUs, int]) -> Tote:
        tote = Tote(int(tote_id))
        tote.skus_list = list(sku_quantity_map.keys())
        tote.capacity = [int(sku_quantity_map[sku]) for sku in tote.skus_list]
        tote.sku_quantity_map = {int(sku.id): int(sku_quantity_map[sku]) for sku in tote.skus_list}
        for sku in tote.skus_list:
            qty = int(tote.sku_quantity_map[sku.id])
            sku.storeToteList.append(tote.id)
            sku.storeQuantityList.append(qty)
            sku.tote_quantity_map[tote.id] = qty
        stack.add_tote(tote)
        return tote

    @staticmethod
    def _build_default_inventory(
            stack_list: List[Stack],
            skus_list: List[SKUs],
            tote_num: int,
            imbalance_profile: str = None,
    ) -> Tuple[List[Tote], List[Stack]]:
        tote_list: List[Tote] = []
        hot_sku_count = max(1, int(len(skus_list) * (0.1 if imbalance_profile == "uneven" else 0.2)))
        hot_skus = skus_list[:hot_sku_count]
        cold_skus = skus_list[hot_sku_count:]

        def sample_sku_by_popularity() -> SKUs:
            if random.random() < 0.8:
                return random.choice(hot_skus)
            return random.choice(cold_skus) if cold_skus else random.choice(hot_skus)

        desired_tote_count = max(tote_num, hot_sku_count * 5)
        unassigned_skus: Dict[int, SKUs] = {int(sku.id): sku for sku in skus_list}
        current_tote_id = 0

        shuffled_stacks = list(stack_list)
        random.shuffle(shuffled_stacks)
        for stack in shuffled_stacks:
            if not unassigned_skus or current_tote_id >= desired_tote_count:
                break
            selected_ids = random.sample(sorted(unassigned_skus.keys()), min(4, len(unassigned_skus)))
            sku_quantity_map = {unassigned_skus[sku_id]: int(random.randint(15, 50)) for sku_id in selected_ids}
            tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
            for sku_id in selected_ids:
                unassigned_skus.pop(sku_id, None)
            current_tote_id += 1

        shuffled_stacks = list(stack_list)
        random.shuffle(shuffled_stacks)
        for stack in shuffled_stacks:
            if current_tote_id >= desired_tote_count:
                break
            max_additional = int(stack.max_height - stack.current_height)
            if max_additional <= 0:
                continue
            layers_to_fill = random.randint(1, max_additional)
            for _ in range(layers_to_fill):
                if current_tote_id >= desired_tote_count:
                    break
                tote_skus: List[SKUs] = []
                for _ in range(random.randint(4, 6)):
                    attempts = 0
                    while attempts < 10:
                        candidate = sample_sku_by_popularity()
                        if candidate not in tote_skus:
                            tote_skus.append(candidate)
                            break
                        attempts += 1
                if not tote_skus:
                    continue
                sku_quantity_map: Dict[SKUs, int] = {}
                for sku in tote_skus:
                    sku_quantity_map[sku] = int(random.randint(20, 60)) if sku in hot_skus else int(random.randint(10, 40))
                tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
                current_tote_id += 1

        final_stack_list = [stack for stack in stack_list if stack.current_height > 0]
        return tote_list, final_stack_list

    @staticmethod
    def _select_spread_stack_ids(
            stack_list: List[Stack],
            reservation_load_by_stack: Dict[int, int],
            required_count: int,
    ) -> List[int]:
        selected: List[Stack] = []
        remaining: Dict[int, Stack] = {int(stack.stack_id): stack for stack in stack_list}
        while remaining and len(selected) < max(0, int(required_count)):
            ranked_rows = []
            for stack in remaining.values():
                load = int(reservation_load_by_stack.get(int(stack.stack_id), 0))
                if not selected:
                    min_distance = 0.0
                    avg_distance = 0.0
                else:
                    distances = [
                        abs(int(stack.store_point.x) - int(other.store_point.x))
                        + abs(int(stack.store_point.y) - int(other.store_point.y))
                        for other in selected
                    ]
                    min_distance = float(min(distances))
                    avg_distance = float(sum(distances) / len(distances))
                ranked_rows.append((load, -min_distance, -avg_distance, random.random(), int(stack.stack_id)))
            ranked_rows.sort()
            chosen_stack = remaining.pop(int(ranked_rows[0][-1]))
            selected.append(chosen_stack)
        return [int(stack.stack_id) for stack in selected]

    @staticmethod
    def _build_zrich_inventory(
            stack_list: List[Stack],
            skus_list: List[SKUs],
            orders: List[Order],
            tote_num: int,
            sku_profile: Dict[str, Any],
            required_distinct_stacks: int = 4,
    ) -> Tuple[List[Tote], List[Stack], Dict[str, Any]]:
        tote_list: List[Tote] = []
        sku_by_id = {int(sku.id): sku for sku in skus_list}
        all_sku_ids = sorted(sku_by_id.keys())
        weight_by_id = dict(sku_profile.get("weights", {}) or {})
        demanded_sku_ids = sorted({
            int(sku_id)
            for order in orders
            for sku_id in getattr(order, "order_product_id_list", []) or []
        })

        reservation_load_by_stack = {int(stack.stack_id): 0 for stack in stack_list}
        reserved_skus_by_stack: Dict[int, List[SKUs]] = {int(stack.stack_id): [] for stack in stack_list}
        reserved_stack_ids_by_sku: Dict[int, List[int]] = {}
        demanded_sorted = sorted(demanded_sku_ids, key=lambda sku_id: (-float(weight_by_id.get(int(sku_id), 0.0)), int(sku_id)))

        for sku_id in demanded_sorted:
            chosen_stack_ids = CreateOFSProblem._select_spread_stack_ids(stack_list, reservation_load_by_stack, required_distinct_stacks)
            reserved_stack_ids_by_sku[int(sku_id)] = list(chosen_stack_ids)
            for stack_id in chosen_stack_ids:
                reserved_skus_by_stack[int(stack_id)].append(sku_by_id[int(sku_id)])
                reservation_load_by_stack[int(stack_id)] = int(reservation_load_by_stack.get(int(stack_id), 0)) + 1

        current_tote_id = 0
        stack_by_id = {int(stack.stack_id): stack for stack in stack_list}
        for stack_id, reserved_skus in sorted(reserved_skus_by_stack.items(), key=lambda item: (-len(item[1]), int(item[0]))):
            stack = stack_by_id[int(stack_id)]
            pending = list(dict.fromkeys(reserved_skus))
            while pending:
                if current_tote_id >= tote_num:
                    raise ValueError("Insufficient tote budget for Z-rich reservation materialization.")
                take_count = min(len(pending), random.randint(1, 4))
                tote_skus = pending[:take_count]
                pending = pending[take_count:]
                target_size = max(len(tote_skus), random.randint(4, 6))
                filler_ids = CreateOFSProblem._weighted_unique_sample(
                    [sku_id for sku_id in all_sku_ids if sku_by_id[sku_id] not in tote_skus],
                    weight_by_id,
                    target_size - len(tote_skus),
                )
                tote_skus.extend([sku_by_id[int(sku_id)] for sku_id in filler_ids])
                sku_quantity_map = CreateOFSProblem._sample_tote_quantities(tote_skus, weight_by_id)
                tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
                current_tote_id += 1

        while current_tote_id < int(tote_num):
            available_stacks = [stack for stack in stack_list if int(stack.current_height) < int(stack.max_height)]
            if not available_stacks:
                break
            available_stacks.sort(key=lambda stack: (int(stack.current_height), random.random(), int(stack.stack_id)))
            candidate_window = available_stacks[:max(1, min(8, len(available_stacks)))]
            stack = random.choice(candidate_window)
            tote_sku_ids = CreateOFSProblem._weighted_unique_sample(all_sku_ids, weight_by_id, random.randint(4, 6))
            if not tote_sku_ids:
                break
            tote_skus = [sku_by_id[int(sku_id)] for sku_id in tote_sku_ids]
            sku_quantity_map = CreateOFSProblem._sample_tote_quantities(tote_skus, weight_by_id)
            tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
            current_tote_id += 1

        final_stack_list = [stack for stack in stack_list if stack.current_height > 0]
        redundancy_summary = CreateOFSProblem._compute_demanded_sku_redundancy_summary(orders, final_stack_list, target_distinct_stacks=required_distinct_stacks)
        redundancy_summary.update({
            "reservation_profile": "zrich",
            "reservation_lambda": float(sku_profile.get("lambda", 0.0)),
            "reserved_placement_count": int(sum(len(v) for v in reserved_stack_ids_by_sku.values())),
            "reserved_sku_count": int(len(reserved_stack_ids_by_sku)),
        })
        return tote_list, final_stack_list, redundancy_summary

    @staticmethod
    def _compute_demanded_sku_redundancy_summary(
            orders: List[Order],
            stack_list: List[Stack],
            target_distinct_stacks: int = 0,
    ) -> Dict[str, Any]:
        demanded_sku_ids = sorted({
            int(sku_id)
            for order in orders
            for sku_id in getattr(order, "order_product_id_list", []) or []
        })
        stack_ids_by_sku: Dict[int, Set[int]] = {}
        for stack in stack_list:
            sku_ids_on_stack: Set[int] = set()
            for tote in getattr(stack, "totes", []) or []:
                for sku in getattr(tote, "skus_list", []) or []:
                    sku_ids_on_stack.add(int(getattr(sku, "id", -1)))
            for sku_id in sku_ids_on_stack:
                stack_ids_by_sku.setdefault(int(sku_id), set()).add(int(stack.stack_id))

        demanded_counts = [len(stack_ids_by_sku.get(int(sku_id), set())) for sku_id in demanded_sku_ids]
        target = max(0, int(target_distinct_stacks))
        if demanded_counts:
            ge_target_count = sum(1 for count in demanded_counts if count >= target) if target > 0 else len(demanded_counts)
            min_count = min(demanded_counts)
            avg_count = float(sum(demanded_counts) / len(demanded_counts))
            max_count = max(demanded_counts)
        else:
            ge_target_count = 0
            min_count = 0
            avg_count = 0.0
            max_count = 0

        return {
            "target_distinct_stacks": int(target),
            "demanded_sku_count": int(len(demanded_sku_ids)),
            "min_distinct_stacks_per_demanded_sku": int(min_count),
            "avg_distinct_stacks_per_demanded_sku": float(avg_count),
            "max_distinct_stacks_per_demanded_sku": int(max_count),
            "demanded_sku_ge_target_count": int(ge_target_count),
            "demanded_sku_ge_target_share": float(ge_target_count / max(1, len(demanded_counts))),
            "distinct_stacks_by_demanded_sku": {
                str(sku_id): int(len(stack_ids_by_sku.get(int(sku_id), set())))
                for sku_id in demanded_sku_ids
            },
        }

    @staticmethod
    def _generate_zrich_orders(
            max_sku_types_per_order: int,
            max_quantity_per_sku: int,
            num_orders: int,
            skus_list: List[SKUs],
            sku_profile: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []
        weight_by_id = dict(sku_profile.get("weights", {}) or {})
        all_sku_ids = [int(sku.id) for sku in skus_list]

        for i in range(num_orders):
            order = Order(i)
            base_time = datetime(2025, 1, 1, 8, 0, 0)
            random_minutes = random.randint(0, 480)
            order.order_in_time = base_time - timedelta(minutes=random_minutes)

            low = max(1, max_sku_types_per_order - 5)
            num_types = random.randint(low, max_sku_types_per_order)
            selected_ids = CreateOFSProblem._weighted_unique_sample(all_sku_ids, weight_by_id, num_types)

            order_product_id_list: List[int] = []
            total_qty = 0
            for sku_id in selected_ids:
                qty = random.randint(1, max_quantity_per_sku)
                order_product_id_list.extend([int(sku_id)] * int(qty))
                total_qty += int(qty)

            order.order_product_id_list = order_product_id_list
            order.order_skus_number = total_qty
            order.status = "pending"
            orders.append(order)
        return orders

    @staticmethod
    def _generate_orders(
            max_sku_types_per_order: int,
            max_quantity_per_sku: int,
            num_orders: int,
            skus_list: List[SKUs],
            imbalance_profile: str = None
    ) -> List[Order]:
        """
        生成订单 (BOM)
        :param skus_list: 具体的 SKU 对象列表，用于随机采样
        """
        orders: List[Order] = []

        hot_skus_count = max(1, int(len(skus_list) * (0.1 if imbalance_profile == "uneven" else 0.2)))
        hot_skus = skus_list[:hot_skus_count]
        cold_skus = skus_list[hot_skus_count:]

        for i in range(num_orders):
            order = Order(i)

            base_time = datetime(2025, 1, 1, 8, 0, 0)
            random_minutes = random.randint(0, 480)
            order.order_in_time = base_time - timedelta(minutes=random_minutes)

            if imbalance_profile == "uneven":
                if random.random() < 0.35:
                    num_types = random.randint(max(4, max_sku_types_per_order - 3), max_sku_types_per_order)
                else:
                    num_types = random.randint(max(1, max_sku_types_per_order // 3), max(2, max_sku_types_per_order // 2))
            else:
                num_types = random.randint(max_sku_types_per_order - 5, max_sku_types_per_order)

            selected_skus = []
            while len(selected_skus) < num_types:
                if random.random() < (0.9 if imbalance_profile == "uneven" else 0.7) and hot_skus:
                    sku = random.choice(hot_skus)
                elif cold_skus:
                    sku = random.choice(cold_skus)
                else:
                    sku = random.choice(skus_list)
                if sku not in selected_skus:
                    selected_skus.append(sku)

            order_product_id_list: List[int] = []
            total_qty = 0

            for sku in selected_skus:
                if imbalance_profile == "uneven" and random.random() < 0.3:
                    qty = random.randint(max(1, max_quantity_per_sku // 2), max_quantity_per_sku + 2)
                else:
                    qty = random.randint(1, max_quantity_per_sku)
                order_product_id_list.extend([int(sku.id)] * int(qty))
                total_qty += int(qty)

            order.order_product_id_list = order_product_id_list
            order.order_skus_number = total_qty
            order.status = "pending"
            orders.append(order)

        return orders


if __name__ == "__main__":
    scales = ["SMALL", "SMALL_ZRICH", "MEDIUM"]

    for scale in scales:
        print(f"\n{'=' * 20} Testing {scale} Scale {'=' * 20}")
        try:
            dto = CreateOFSProblem.generate_problem_by_scale(scale)
            print(f"Success! Generated {scale} instance.")
            if dto.order_list:
                o0 = dto.order_list[0]
                print(f"Order 0: id={o0.order_id}, total_items={o0.order_skus_number}")
                print(f"  SKU IDs: {o0.order_product_id_list}")

            print(f"Total Stacks: {len(dto.stack_list)}")
            if dto.stack_list:
                s0 = dto.stack_list[0]
                print(f"  Stack {s0.stack_id} Height: {s0.current_height}/{s0.max_height}")
            if dto.redundancy_summary:
                print(f"  Redundancy: {dto.redundancy_summary}")

        except Exception as exc:
            print(f"Error generating {scale}: {exc}")
            import traceback

            traceback.print_exc()
