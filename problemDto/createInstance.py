import math
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

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
    def _time_window_rng(base_seed: int, order_id: int) -> random.Random:
        mix = (int(base_seed) * 1000003 + int(order_id) * 9176 + 0x9E3779B9) & 0xFFFFFFFF
        return random.Random(int(mix))

    @staticmethod
    def _assign_order_time_window(order: Order, base_seed: int = 0) -> None:
        est_sec = 0
        unique_sku_count = int(len(set(int(sku_id) for sku_id in (getattr(order, "order_product_id_list", []) or []))))
        total_qty = int(len(getattr(order, "order_product_id_list", []) or []))
        kitting_span_limit_sec = float(unique_sku_count * float(OFSConfig.ORDER_KITTING_SPAN_PER_UNIQUE_SKU_SEC))
        rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed), order_id=int(getattr(order, "order_id", 0)))
        deadline_buffer_sec = float(rng.randint(int(OFSConfig.ORDER_LST_BUFFER_MIN_SEC), int(OFSConfig.ORDER_LST_BUFFER_MAX_SEC)))
        lst_sec = float(
            float(OFSConfig.ORDER_LST_BASE_SEC)
            + float(kitting_span_limit_sec)
            + float(total_qty) * float(OFSConfig.ORDER_LST_PER_QTY_SEC)
            + float(deadline_buffer_sec)
        )
        order.est_sec = float(est_sec)
        order.kitting_span_limit_sec = float(kitting_span_limit_sec)
        order.lst_sec = float(lst_sec)
        order.total_qty = int(total_qty)
        order.unique_sku_count = int(unique_sku_count)
        order.deadline_buffer_sec = float(deadline_buffer_sec)

    @staticmethod
    def _expand_sku_ids_with_quantities(sku_ids: List[int], fixed_qty: int, qty_range: Tuple[int, int], rng: random.Random) -> List[int]:
        expanded: List[int] = []
        use_range = len(qty_range or ()) == 2
        low = int(qty_range[0]) if use_range else int(fixed_qty)
        high = int(qty_range[1]) if use_range else int(fixed_qty)
        low = max(1, int(low))
        high = max(low, int(high))
        for sku_id in sku_ids:
            qty = int(rng.randint(low, high)) if use_range else int(low)
            expanded.extend([int(sku_id)] * int(qty))
        return expanded

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
            "GUROBI-S1": {"map_size": (2, 4), "resources": (2, 2, 30), "data": (1, 10), "bom_complexity": (10, 1), "exact_bom_sku_count": 10, "exact_bom_sku_quantity_range": (3, 5)},
            "GUROBI-S2": {"map_size": (2, 4), "resources": (2, 2, 50), "data": (2, 15), "bom_complexity": (8, 1), "target_stack_count": 3, "exact_order_sku_counts": (6, 7), "exact_order_sku_quantity_range": (3, 5), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (1, 2), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2, "bom_colocated_sku_copy_count": 1, "bom_colocated_chunked_by_stack": True},
            "GUROBI-S3": {"map_size": (2, 4), "resources": (2, 2, 65), "data": (2, 18), "bom_complexity": (8, 1), "target_stack_count": 4, "exact_order_sku_counts": (7, 7), "exact_order_sku_quantity_range": (8, 10), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2, "bom_colocated_sku_copy_count": 1, "bom_colocated_chunked_by_stack": True},
            "GUROBI-S4": {"map_size": (3, 4), "resources": (3, 3, 80), "data": (3, 20), "bom_complexity": (8, 1), "target_stack_count": 5, "exact_order_sku_counts": (6, 7, 7), "exact_order_sku_quantity_range": (7, 9), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (1, 2, 2), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2, "bom_colocated_sku_copy_count": 1, "bom_colocated_chunked_by_stack": True},
            "GUROBI-S5": {"map_size": (3, 4), "resources": (3, 3, 95), "data": (3, 24), "bom_complexity": (8, 1), "target_stack_count": 6, "exact_order_sku_counts": (7, 7, 7), "exact_order_sku_quantity_range": (9, 11), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 2, 2), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2, "bom_colocated_sku_copy_count": 1, "bom_colocated_chunked_by_stack": True},
            "GUROBI-S6": {"map_size": (3, 4), "resources": (3, 3, 100), "data": (4, 25), "bom_complexity": (8, 1), "target_stack_count": 6, "exact_order_sku_counts": (3, 5, 5, 3), "exact_order_sku_quantity_range": (11, 13), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (1, 2, 2, 1), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2, "bom_colocated_sku_copy_count": 1, "bom_colocated_chunked_by_stack": False},
            "GUROBI-S7": {"map_size": (3, 5), "resources": (3, 3, 120), "data": (4, 30), "bom_complexity": (8, 1), "target_stack_count": 22, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (5, 6, 6, 6), "exact_order_sku_quantity_range": (6, 8), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 3, 3, 2), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
            "GUROBI-S8": {"map_size": (3, 5), "resources": (4, 4, 120), "data": (5, 30), "bom_complexity": (8, 1), "target_stack_count": 22, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (5, 6, 6, 6, 6), "exact_order_sku_quantity_range": (9, 11), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
            "GUROBI-S9": {"map_size": (3, 5), "resources": (4, 4, 145), "data": (5, 36), "bom_complexity": (8, 1), "target_stack_count": 26, "inventory_cold_filler_probability": 0.25, "exact_order_sku_counts": (5, 6, 6, 6, 6), "exact_order_sku_quantity_range": (11, 13), "bom_colocated_inventory": True, "bom_colocated_stack_counts": (2, 3, 3, 3, 3), "bom_colocated_disjoint_stack_groups": True, "bom_colocated_support_multiplier": 1.2},
            "GUROBI-SM1": {"map_size": (2, 4), "resources": (1, 1, 12), "data": (1, 8), "bom_complexity": (4, 1), "target_stack_count": 3, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 1, "exact_bom_sku_count": 4, "exact_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM2": {"map_size": (2, 3), "resources": (1, 1, 13), "data": (1, 9), "bom_complexity": (6, 1), "target_stack_count": 4, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 1, "exact_bom_sku_count": 6, "exact_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM3": {"map_size": (2, 4), "resources": (2, 2, 16), "data": (2, 14), "bom_complexity": (6, 1), "target_stack_count": 6, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 1, "exact_disjoint_bom_sku_count": 6, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM4": {"map_size": (2, 4), "resources": (2, 2, 18), "data": (2, 16), "bom_complexity": (6, 1), "target_stack_count": 7, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 1, "exact_disjoint_bom_sku_count": 6, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM5": {"map_size": (3, 4), "resources": (3, 3, 22), "data": (3, 22), "bom_complexity": (6, 1), "target_stack_count": 8, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 1, "exact_disjoint_bom_sku_count": 6, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM6": {"map_size": (3, 4), "resources": (3, 3, 26), "data": (3, 28), "bom_complexity": (8, 1), "target_stack_count": 9, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 1, "exact_disjoint_bom_sku_count": 8, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM7": {"map_size": (3, 5), "resources": (3, 3, 32), "data": (4, 40), "bom_complexity": (8, 1), "target_stack_count": 10, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 2, "exact_disjoint_bom_sku_count": 8, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM8": {"map_size": (3, 5), "resources": (3, 3, 40), "data": (4, 44), "bom_complexity": (8, 1), "target_stack_count": 11, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 2, "exact_disjoint_bom_sku_count": 8, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "GUROBI-SM9": {"map_size": (4, 5), "resources": (4, 3, 50), "data": (5, 60), "bom_complexity": (8, 1), "target_stack_count": 15, "inventory_cold_filler_probability": 0.0, "inventory_initial_unassigned_skus_per_tote": 3, "exact_disjoint_bom_sku_count": 8, "exact_disjoint_bom_sku_quantity_range": (5, 10), "exact_demand_sku_strategy": "redundancy_3_4"},
            "TINY3": {"map_size": (4, 4), "resources": (2, 2, 150), "data": (3, 30), "bom_complexity": (10, 1), "exact_disjoint_bom_sku_count": 10},
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
        exact_shared_bom_sku_count = int(cfg.get("exact_shared_bom_sku_count", 0))
        exact_disjoint_bom_sku_count = int(cfg.get("exact_disjoint_bom_sku_count", 0))
        exact_order_sku_counts = tuple(int(v) for v in (cfg.get("exact_order_sku_counts", ()) or ()))
        exact_remap_existing_bom_skus = bool(cfg.get("exact_remap_existing_bom_skus", False))
        exact_bom_sku_quantity = int(cfg.get("exact_bom_sku_quantity", 1))
        exact_shared_bom_sku_quantity = int(cfg.get("exact_shared_bom_sku_quantity", 1))
        exact_disjoint_bom_sku_quantity = int(cfg.get("exact_disjoint_bom_sku_quantity", 1))
        exact_bom_sku_quantity_range = tuple(cfg.get("exact_bom_sku_quantity_range", ()))
        exact_shared_bom_sku_quantity_range = tuple(cfg.get("exact_shared_bom_sku_quantity_range", ()))
        exact_disjoint_bom_sku_quantity_range = tuple(cfg.get("exact_disjoint_bom_sku_quantity_range", ()))
        exact_order_sku_quantity_range = tuple(cfg.get("exact_order_sku_quantity_range", ()))
        bom_quantity_range = tuple(cfg.get("bom_quantity_range", ()))
        exact_demand_sku_from_tail = bool(cfg.get("exact_demand_sku_from_tail", False))
        exact_demand_sku_strategy = str(cfg.get("exact_demand_sku_strategy", "") or "")
        target_stack_count = int(cfg.get("target_stack_count", 0) or 0)
        inventory_cold_filler_probability = float(cfg.get("inventory_cold_filler_probability", 0.2))
        inventory_initial_unassigned_skus_per_tote = int(cfg.get("inventory_initial_unassigned_skus_per_tote", 4) or 4)
        inventory_max_sku_stack_count = int(cfg.get("inventory_max_sku_stack_count", 3) or 3)
        bom_colocated_inventory = bool(cfg.get("bom_colocated_inventory", False))
        bom_colocated_stack_min = int(cfg.get("bom_colocated_stack_min", 4) or 4)
        bom_colocated_stack_max = int(cfg.get("bom_colocated_stack_max", 6) or 6)
        bom_colocated_stack_counts = tuple(int(v) for v in (cfg.get("bom_colocated_stack_counts", ()) or ()))
        bom_colocated_support_multiplier = float(cfg.get("bom_colocated_support_multiplier", 2.5) or 2.5)
        bom_colocated_disjoint_stack_groups = bool(cfg.get("bom_colocated_disjoint_stack_groups", False))
        bom_colocated_sku_copy_count = int(cfg.get("bom_colocated_sku_copy_count", 2) or 2)
        bom_colocated_chunked_by_stack = bool(cfg.get("bom_colocated_chunked_by_stack", False))

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
            exact_shared_bom_sku_count=exact_shared_bom_sku_count,
            exact_disjoint_bom_sku_count=exact_disjoint_bom_sku_count,
            exact_order_sku_counts=exact_order_sku_counts,
            exact_remap_existing_bom_skus=exact_remap_existing_bom_skus,
            exact_bom_sku_quantity=exact_bom_sku_quantity,
            exact_shared_bom_sku_quantity=exact_shared_bom_sku_quantity,
            exact_disjoint_bom_sku_quantity=exact_disjoint_bom_sku_quantity,
            exact_bom_sku_quantity_range=exact_bom_sku_quantity_range,
            exact_shared_bom_sku_quantity_range=exact_shared_bom_sku_quantity_range,
            exact_disjoint_bom_sku_quantity_range=exact_disjoint_bom_sku_quantity_range,
            exact_order_sku_quantity_range=exact_order_sku_quantity_range,
            bom_quantity_range=bom_quantity_range,
            exact_demand_sku_from_tail=exact_demand_sku_from_tail,
            exact_demand_sku_strategy=exact_demand_sku_strategy,
            target_stack_count=target_stack_count,
            inventory_cold_filler_probability=inventory_cold_filler_probability,
            inventory_initial_unassigned_skus_per_tote=inventory_initial_unassigned_skus_per_tote,
            inventory_max_sku_stack_count=inventory_max_sku_stack_count,
            bom_colocated_inventory=bool(bom_colocated_inventory),
            bom_colocated_stack_min=int(bom_colocated_stack_min),
            bom_colocated_stack_max=int(bom_colocated_stack_max),
            bom_colocated_stack_counts=bom_colocated_stack_counts,
            bom_colocated_disjoint_stack_groups=bool(bom_colocated_disjoint_stack_groups),
            bom_colocated_support_multiplier=float(bom_colocated_support_multiplier),
            bom_colocated_sku_copy_count=int(bom_colocated_sku_copy_count),
            bom_colocated_chunked_by_stack=bool(bom_colocated_chunked_by_stack),
            base_seed=int(seed),
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
            exact_shared_bom_sku_count: int = 0,
            exact_disjoint_bom_sku_count: int = 0,
            exact_order_sku_counts: Tuple[int, ...] = (),
            exact_remap_existing_bom_skus: bool = False,
            exact_bom_sku_quantity: int = 1,
            exact_shared_bom_sku_quantity: int = 1,
            exact_disjoint_bom_sku_quantity: int = 1,
            exact_bom_sku_quantity_range: Tuple[int, int] = (),
            exact_shared_bom_sku_quantity_range: Tuple[int, int] = (),
            exact_disjoint_bom_sku_quantity_range: Tuple[int, int] = (),
            exact_order_sku_quantity_range: Tuple[int, int] = (),
            bom_quantity_range: Tuple[int, int] = (),
            exact_demand_sku_from_tail: bool = False,
            exact_demand_sku_strategy: str = "",
            target_stack_count: int = 0,
            inventory_cold_filler_probability: float = 0.2,
            inventory_initial_unassigned_skus_per_tote: int = 4,
            inventory_max_sku_stack_count: int = 3,
            bom_colocated_inventory: bool = False,
            bom_colocated_stack_min: int = 4,
            bom_colocated_stack_max: int = 6,
            bom_colocated_stack_counts: Tuple[int, ...] = (),
            bom_colocated_disjoint_stack_groups: bool = False,
            bom_colocated_support_multiplier: float = 2.5,
            bom_colocated_sku_copy_count: int = 2,
            bom_colocated_chunked_by_stack: bool = False,
            base_seed: int = OFSConfig.RANDOM_SEED,
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
            orders = CreateOFSProblem._generate_zrich_orders(max_sku_types, max_sku_qty, order_num, skus_list_obj, zrich_profile, qty_range=bom_quantity_range, base_seed=int(base_seed))
        else:
            orders = CreateOFSProblem._generate_orders(max_sku_types, max_sku_qty, order_num, skus_list_obj, imbalance_profile, qty_range=bom_quantity_range, base_seed=int(base_seed))
        defer_exact_demand = bool(str(exact_demand_sku_strategy or "").strip())
        if int(exact_bom_sku_count) > 0 and not defer_exact_demand:
            if int(order_num) != 1:
                raise ValueError("exact_bom_sku_count is intended for single-BOM test cases.")
            exact_count = min(int(exact_bom_sku_count), int(len(skus_list_obj)))
            exact_qty = max(1, int(exact_bom_sku_quantity))
            exact_order = orders[0]
            exact_skus = skus_list_obj[-exact_count:] if bool(exact_demand_sku_from_tail) else skus_list_obj[:exact_count]
            qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 7919, order_id=int(getattr(exact_order, "order_id", 0)))
            exact_order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                [int(sku.id) for sku in exact_skus],
                fixed_qty=exact_qty,
                qty_range=exact_bom_sku_quantity_range,
                rng=qty_rng,
            )
            exact_order.order_skus_number = int(len(exact_order.order_product_id_list))
            CreateOFSProblem._assign_order_time_window(exact_order, base_seed=int(base_seed))
            exact_order.status = "pending"
        if int(exact_disjoint_bom_sku_count) > 0 and not defer_exact_demand:
            exact_count = int(exact_disjoint_bom_sku_count)
            exact_qty = max(1, int(exact_disjoint_bom_sku_quantity))
            if int(order_num) * exact_count > int(len(skus_list_obj)):
                raise ValueError("exact_disjoint_bom_sku_count requires order_num * count <= skus_num.")
            tail_start = int(len(skus_list_obj)) - int(order_num) * exact_count
            for order_idx, order in enumerate(orders):
                start_idx = (tail_start if bool(exact_demand_sku_from_tail) else 0) + int(order_idx) * exact_count
                order_skus = skus_list_obj[start_idx:start_idx + exact_count]
                qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 104729, order_id=int(getattr(order, "order_id", order_idx)))
                order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                    [int(sku.id) for sku in order_skus],
                    fixed_qty=exact_qty,
                    qty_range=exact_disjoint_bom_sku_quantity_range,
                    rng=qty_rng,
                )
                order.order_skus_number = int(len(order.order_product_id_list))
                CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
                order.status = "pending"
        if exact_order_sku_counts and not defer_exact_demand:
            per_order_counts = [max(0, int(v)) for v in exact_order_sku_counts]
            if len(per_order_counts) != int(order_num):
                raise ValueError("exact_order_sku_counts length must match order count.")
            if sum(per_order_counts) > int(len(skus_list_obj)):
                raise ValueError("exact_order_sku_counts requires enough SKU ids.")
            cursor = 0
            for order_idx, order in enumerate(orders):
                take = int(per_order_counts[int(order_idx)])
                order_skus = skus_list_obj[cursor:cursor + take]
                cursor += take
                qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 131071, order_id=int(getattr(order, "order_id", order_idx)))
                order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                    [int(sku.id) for sku in order_skus],
                    fixed_qty=1,
                    qty_range=exact_order_sku_quantity_range,
                    rng=qty_rng,
                )
                order.order_skus_number = int(len(order.order_product_id_list))
                CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
                order.status = "pending"
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
        elif bool(bom_colocated_inventory):
            tote_list, final_stack_list, redundancy_summary = CreateOFSProblem._build_bom_colocated_inventory(
                stack_list=stack_list,
                skus_list=skus_list_obj,
                orders=orders,
                tote_num=int(tote_num),
                target_stack_count=int(target_stack_count),
                stack_min=int(bom_colocated_stack_min),
                stack_max=int(bom_colocated_stack_max),
                stack_counts=tuple(int(v) for v in (bom_colocated_stack_counts or ())),
                disjoint_stack_groups=bool(bom_colocated_disjoint_stack_groups),
                support_multiplier=float(bom_colocated_support_multiplier),
                cold_filler_probability=float(inventory_cold_filler_probability),
                sku_copy_count=int(bom_colocated_sku_copy_count),
                chunked_by_stack=bool(bom_colocated_chunked_by_stack),
            )
        else:
            tote_list, final_stack_list = CreateOFSProblem._build_default_inventory(
                stack_list,
                skus_list_obj,
                tote_num,
                imbalance_profile,
                target_stack_count=int(target_stack_count),
                cold_filler_probability=float(inventory_cold_filler_probability),
                initial_unassigned_skus_per_tote=int(inventory_initial_unassigned_skus_per_tote),
                max_sku_stack_count=int(inventory_max_sku_stack_count),
            )
            if defer_exact_demand:
                redundancy_target = CreateOFSProblem._parse_redundancy_strategy(exact_demand_sku_strategy)
                if redundancy_target is not None:
                    next_tote_id = max([int(getattr(tote, "id", -1)) for tote in tote_list] + [-1]) + 1
                    added_totes = CreateOFSProblem._ensure_exact_demand_redundancy(
                        skus_list=skus_list_obj,
                        stack_list=final_stack_list,
                        exact_bom_sku_count=int(exact_bom_sku_count),
                        exact_disjoint_bom_sku_count=int(exact_disjoint_bom_sku_count),
                        exact_order_sku_counts=tuple(int(v) for v in (exact_order_sku_counts or ())),
                        exact_remap_existing_bom_skus=bool(exact_remap_existing_bom_skus),
                        existing_orders=orders,
                        order_num=int(order_num),
                        target_min=int(redundancy_target[0]),
                        target_max=int(redundancy_target[1]),
                        next_tote_id=int(next_tote_id),
                    )
                    tote_list.extend(added_totes)
                CreateOFSProblem._assign_exact_demands_from_inventory(
                    orders=orders,
                    skus_list=skus_list_obj,
                    stack_list=final_stack_list,
                    exact_bom_sku_count=int(exact_bom_sku_count),
                    exact_shared_bom_sku_count=int(exact_shared_bom_sku_count),
                    exact_disjoint_bom_sku_count=int(exact_disjoint_bom_sku_count),
                    exact_order_sku_counts=tuple(int(v) for v in (exact_order_sku_counts or ())),
                    exact_remap_existing_bom_skus=bool(exact_remap_existing_bom_skus),
                    exact_bom_sku_quantity=int(exact_bom_sku_quantity),
                    exact_shared_bom_sku_quantity=int(exact_shared_bom_sku_quantity),
                    exact_disjoint_bom_sku_quantity=int(exact_disjoint_bom_sku_quantity),
                    exact_bom_sku_quantity_range=exact_bom_sku_quantity_range,
                    exact_shared_bom_sku_quantity_range=exact_shared_bom_sku_quantity_range,
                    exact_disjoint_bom_sku_quantity_range=exact_disjoint_bom_sku_quantity_range,
                    strategy=str(exact_demand_sku_strategy),
                    base_seed=int(base_seed),
                )
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
    def _sku_stack_counts(stack_list: List[Stack], skus_list: List[SKUs]) -> Dict[int, int]:
        counts: Dict[int, int] = {int(sku.id): 0 for sku in skus_list}
        for stack in stack_list or []:
            sku_ids_on_stack: Set[int] = set()
            for tote in getattr(stack, "totes", []) or []:
                sku_ids_on_stack.update(int(sku_id) for sku_id in (getattr(tote, "sku_quantity_map", {}) or {}).keys())
            for sku_id in sku_ids_on_stack:
                counts[int(sku_id)] = int(counts.get(int(sku_id), 0)) + 1
        return counts

    @staticmethod
    def _sku_stack_sets(stack_list: List[Stack], skus_list: List[SKUs]) -> Dict[int, Set[int]]:
        stack_sets: Dict[int, Set[int]] = {int(sku.id): set() for sku in skus_list}
        for stack in stack_list or []:
            stack_id = int(getattr(stack, "stack_id", -1))
            if stack_id < 0:
                continue
            sku_ids_on_stack: Set[int] = set()
            for tote in getattr(stack, "totes", []) or []:
                sku_ids_on_stack.update(int(sku_id) for sku_id in (getattr(tote, "sku_quantity_map", {}) or {}).keys())
            for sku_id in sku_ids_on_stack:
                stack_sets.setdefault(int(sku_id), set()).add(int(stack_id))
        return stack_sets

    @staticmethod
    def _parse_redundancy_strategy(strategy: str) -> Optional[Tuple[int, int]]:
        strategy_norm = str(strategy or "").strip().lower()
        if not strategy_norm.startswith("redundancy_"):
            return None
        parts = strategy_norm.split("_")
        if len(parts) != 3:
            return None
        try:
            target_min = int(parts[1])
            target_max = int(parts[2])
        except ValueError:
            return None
        if target_min <= 0 or target_max < target_min:
            return None
        return target_min, target_max

    @staticmethod
    def _rank_skus_for_exact_demand(skus_list: List[SKUs], stack_list: List[Stack], strategy: str) -> List[int]:
        counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
        rows = []
        for sku in skus_list:
            sku_id = int(sku.id)
            count = int(counts.get(sku_id, 0))
            if count <= 0:
                continue
            rows.append((count, sku_id))
        if not rows:
            return [int(sku.id) for sku in skus_list]
        strategy_norm = str(strategy or "").strip().lower()
        redundancy_target = CreateOFSProblem._parse_redundancy_strategy(strategy_norm)
        if redundancy_target is not None:
            target_min, target_max = redundancy_target
            preferred = [
                (int(count), int(sku_id))
                for count, sku_id in rows
                if int(target_min) <= int(count) <= int(target_max)
            ]
            preferred.sort(key=lambda row: (-int(row[0]), int(row[1])))
            tail = [
                (int(count), int(sku_id))
                for count, sku_id in rows
                if not (int(target_min) <= int(count) <= int(target_max))
            ]
            tail.sort(key=lambda row: (abs(int(row[0]) - int(target_max)), int(row[0]), int(row[1])))
            return [int(sku_id) for _, sku_id in preferred + tail]
        if strategy_norm == "co_located_stack":
            sku_ids_by_stack: Dict[int, Set[int]] = {}
            for stack in stack_list or []:
                stack_sku_ids: Set[int] = set()
                for tote in getattr(stack, "totes", []) or []:
                    stack_sku_ids.update(int(sku_id) for sku_id in (getattr(tote, "sku_quantity_map", {}) or {}).keys())
                if stack_sku_ids:
                    sku_ids_by_stack[int(stack.stack_id)] = stack_sku_ids
            if sku_ids_by_stack:
                chosen_stack_id, chosen_sku_ids = sorted(
                    sku_ids_by_stack.items(),
                    key=lambda item: (-len(item[1]), int(item[0])),
                )[0]
                chosen_set = set(int(sku_id) for sku_id in chosen_sku_ids)
                colocated = sorted(chosen_set)
                tail = [
                    int(sku_id)
                    for _, sku_id in sorted(rows, key=lambda row: (int(row[0]), int(row[1])))
                    if int(sku_id) not in chosen_set
                ]
                return colocated + tail
        if strategy_norm == "mid_low_redundancy":
            positive_counts = sorted(count for count, _ in rows)
            median = positive_counts[len(positive_counts) // 2]
            rows.sort(key=lambda row: (abs(int(row[0]) - int(median)), int(row[0]), int(row[1])))
            return [int(sku_id) for _, sku_id in rows]
        rows.sort(key=lambda row: (int(row[0]), int(row[1])))
        return [int(sku_id) for _, sku_id in rows]

    @staticmethod
    def _assign_exact_demands_from_inventory(
            orders: List[Order],
            skus_list: List[SKUs],
            stack_list: List[Stack],
            exact_bom_sku_count: int = 0,
            exact_shared_bom_sku_count: int = 0,
            exact_disjoint_bom_sku_count: int = 0,
            exact_order_sku_counts: Tuple[int, ...] = (),
            exact_remap_existing_bom_skus: bool = False,
            exact_bom_sku_quantity: int = 1,
            exact_shared_bom_sku_quantity: int = 1,
            exact_disjoint_bom_sku_quantity: int = 1,
            exact_bom_sku_quantity_range: Tuple[int, int] = (),
            exact_shared_bom_sku_quantity_range: Tuple[int, int] = (),
            exact_disjoint_bom_sku_quantity_range: Tuple[int, int] = (),
            strategy: str = "",
            base_seed: int = OFSConfig.RANDOM_SEED,
    ) -> None:
        ranked_sku_ids = CreateOFSProblem._rank_skus_for_exact_demand(skus_list, stack_list, strategy)
        redundancy_target = CreateOFSProblem._parse_redundancy_strategy(strategy)
        per_order_counts = [max(0, int(v)) for v in (exact_order_sku_counts or ())]
        if bool(exact_remap_existing_bom_skus):
            original_ids: List[int] = []
            seen_original: Set[int] = set()
            order_original_sets: Dict[int, List[int]] = {}
            orders_by_original: Dict[int, Set[int]] = {}
            for order in orders:
                order_id = int(getattr(order, "order_id", len(order_original_sets)))
                order_seen: Set[int] = set()
                order_unique_ids: List[int] = []
                for sku_id_raw in getattr(order, "order_product_id_list", []) or []:
                    sku_id = int(sku_id_raw)
                    if sku_id not in seen_original:
                        seen_original.add(int(sku_id))
                        original_ids.append(int(sku_id))
                    if sku_id not in order_seen:
                        order_seen.add(int(sku_id))
                        order_unique_ids.append(int(sku_id))
                    orders_by_original.setdefault(int(sku_id), set()).add(int(order_id))
                order_original_sets[int(order_id)] = list(order_unique_ids)
            required = int(len(original_ids))
            if required > len(ranked_sku_ids):
                raise ValueError("exact_remap_existing_bom_skus requires enough stocked SKU ids.")
            if redundancy_target is not None:
                target_min, target_max = redundancy_target
                counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
                candidate_ids = [
                    int(sku_id)
                    for sku_id in ranked_sku_ids
                    if int(target_min) <= int(counts.get(int(sku_id), 0)) <= int(target_max)
                ]
                if len(candidate_ids) < required:
                    raise ValueError(
                        f"{str(strategy or '').strip().lower()} cannot remap existing BOM SKUs; "
                        f"qualified={len(candidate_ids)} required={required}"
                    )
            else:
                counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
                candidate_ids = list(ranked_sku_ids)
            stack_sets = CreateOFSProblem._sku_stack_sets(stack_list, skus_list)
            assigned_by_order: Dict[int, Set[int]] = {int(order_id): set() for order_id in order_original_sets.keys()}
            used_new: Set[int] = set()
            remap: Dict[int, int] = {}
            process_original_ids = sorted(
                original_ids,
                key=lambda sku_id: (
                    -len(orders_by_original.get(int(sku_id), set())),
                    -max([len(order_original_sets.get(int(order_id), [])) for order_id in orders_by_original.get(int(sku_id), set())] + [0]),
                    int(sku_id),
                ),
            )
            for old_sku_id in process_original_ids:
                affected_orders = sorted(int(order_id) for order_id in orders_by_original.get(int(old_sku_id), set()))
                best_row: Optional[Tuple[float, int, int]] = None
                for new_sku_id in candidate_ids:
                    new_sku_id = int(new_sku_id)
                    if new_sku_id in used_new:
                        continue
                    new_stacks = set(stack_sets.get(int(new_sku_id), set()))
                    if not new_stacks:
                        continue
                    union_after_total = 0
                    marginal_total = 0
                    for order_id in affected_orders:
                        before = set(assigned_by_order.get(int(order_id), set()))
                        after = before | new_stacks
                        union_after_total += int(len(after))
                        marginal_total += int(len(after) - len(before))
                    stack_count = int(counts.get(int(new_sku_id), len(new_stacks)))
                    score = float(union_after_total) + 0.50 * float(marginal_total) + 0.05 * float(stack_count)
                    row = (float(score), int(stack_count), int(new_sku_id))
                    if best_row is None or row < best_row:
                        best_row = row
                if best_row is None:
                    raise ValueError("exact_remap_existing_bom_skus could not find an unused stocked SKU candidate.")
                chosen_new = int(best_row[2])
                remap[int(old_sku_id)] = int(chosen_new)
                used_new.add(int(chosen_new))
                for order_id in affected_orders:
                    assigned_by_order.setdefault(int(order_id), set()).update(stack_sets.get(int(chosen_new), set()))
            for order in orders:
                order.order_product_id_list = [int(remap[int(sku_id)]) for sku_id in (getattr(order, "order_product_id_list", []) or [])]
                order.order_skus_number = int(len(order.order_product_id_list))
                CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
                order.status = "pending"
            return
        if int(exact_bom_sku_count) > 0:
            if len(orders) != 1:
                raise ValueError("exact_bom_sku_count is intended for single-BOM test cases.")
            take = min(int(exact_bom_sku_count), len(ranked_sku_ids))
            if redundancy_target is not None:
                target_min, target_max = redundancy_target
                counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
                bad = [int(sku_id) for sku_id in ranked_sku_ids[:take] if not (int(target_min) <= int(counts.get(int(sku_id), 0)) <= int(target_max))]
                if bad:
                    raise ValueError(f"{str(strategy or '').strip().lower()} cannot satisfy exact_bom_sku_count={take}; bad_sku_ids={bad}")
            order = orders[0]
            qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 7919, order_id=int(getattr(order, "order_id", 0)))
            order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                ranked_sku_ids[:take],
                fixed_qty=max(1, int(exact_bom_sku_quantity)),
                qty_range=exact_bom_sku_quantity_range,
                rng=qty_rng,
            )
            order.order_skus_number = int(len(order.order_product_id_list))
            CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
            order.status = "pending"
            return
        if per_order_counts:
            if len(per_order_counts) != len(orders):
                raise ValueError("exact_order_sku_counts length must match order count.")
            required = int(sum(per_order_counts))
            if required > len(ranked_sku_ids):
                raise ValueError("exact_order_sku_counts requires enough stocked SKU ids.")
            if redundancy_target is not None:
                target_min, target_max = redundancy_target
                counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
                selected = ranked_sku_ids[:required]
                bad = [int(sku_id) for sku_id in selected if not (int(target_min) <= int(counts.get(int(sku_id), 0)) <= int(target_max))]
                if bad:
                    bad_counts = {int(sku_id): int(counts.get(int(sku_id), 0)) for sku_id in bad}
                    raise ValueError(f"{str(strategy or '').strip().lower()} cannot satisfy exact_order_sku_counts={per_order_counts}; bad_sku_counts={bad_counts}")
            cursor = 0
            for order_idx, order in enumerate(orders):
                take = int(per_order_counts[int(order_idx)])
                sku_ids = ranked_sku_ids[cursor:cursor + take]
                cursor += take
                qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 131071, order_id=int(getattr(order, "order_id", order_idx)))
                order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                    sku_ids,
                    fixed_qty=1,
                    qty_range=(),
                    rng=qty_rng,
                )
                order.order_skus_number = int(len(order.order_product_id_list))
                CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
                order.status = "pending"
            return
        shared_count = int(exact_shared_bom_sku_count)
        if shared_count > 0:
            if shared_count > len(ranked_sku_ids):
                raise ValueError("exact_shared_bom_sku_count requires enough stocked SKU ids.")
            sku_ids = ranked_sku_ids[:shared_count]
            for order_idx, order in enumerate(orders):
                qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 65537, order_id=int(getattr(order, "order_id", order_idx)))
                order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                    sku_ids,
                    fixed_qty=max(1, int(exact_shared_bom_sku_quantity)),
                    qty_range=exact_shared_bom_sku_quantity_range,
                    rng=qty_rng,
                )
                order.order_skus_number = int(len(order.order_product_id_list))
                CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
                order.status = "pending"
            return
        exact_count = int(exact_disjoint_bom_sku_count)
        if exact_count <= 0:
            return
        required = len(orders) * exact_count
        if required > len(ranked_sku_ids):
            raise ValueError("exact_disjoint_bom_sku_count requires enough stocked SKU ids.")
        if redundancy_target is not None:
            target_min, target_max = redundancy_target
            counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
            selected = ranked_sku_ids[:required]
            bad = [int(sku_id) for sku_id in selected if not (int(target_min) <= int(counts.get(int(sku_id), 0)) <= int(target_max))]
            if bad:
                bad_counts = {int(sku_id): int(counts.get(int(sku_id), 0)) for sku_id in bad}
                raise ValueError(f"{str(strategy or '').strip().lower()} cannot satisfy exact_disjoint_bom_sku_count={exact_count}; bad_sku_counts={bad_counts}")
        for order_idx, order in enumerate(orders):
            start_idx = int(order_idx) * exact_count
            sku_ids = ranked_sku_ids[start_idx:start_idx + exact_count]
            qty_rng = CreateOFSProblem._time_window_rng(base_seed=int(base_seed) + 104729, order_id=int(getattr(order, "order_id", order_idx)))
            order.order_product_id_list = CreateOFSProblem._expand_sku_ids_with_quantities(
                sku_ids,
                fixed_qty=max(1, int(exact_disjoint_bom_sku_quantity)),
                qty_range=exact_disjoint_bom_sku_quantity_range,
                rng=qty_rng,
            )
            order.order_skus_number = int(len(order.order_product_id_list))
            CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
            order.status = "pending"

    @staticmethod
    def _add_sku_to_stack(stack: Stack, sku: SKUs, tote_id: int, quantity: int = 30) -> Tote:
        return CreateOFSProblem._register_tote(stack, int(tote_id), {sku: int(quantity)})

    @staticmethod
    def _ensure_exact_demand_redundancy(
            skus_list: List[SKUs],
            stack_list: List[Stack],
            exact_bom_sku_count: int,
            exact_disjoint_bom_sku_count: int,
            order_num: int,
            exact_order_sku_counts: Tuple[int, ...] = (),
            exact_remap_existing_bom_skus: bool = False,
            existing_orders: Optional[List[Order]] = None,
            target_min: int = 2,
            target_max: int = 3,
            next_tote_id: int = 0,
    ) -> List[Tote]:
        required = int(exact_bom_sku_count or 0)
        if required <= 0 and bool(exact_remap_existing_bom_skus):
            existing_ids = {
                int(sku_id)
                for order in (existing_orders or [])
                for sku_id in (getattr(order, "order_product_id_list", []) or [])
            }
            required = int(len(existing_ids))
        if required <= 0 and exact_order_sku_counts:
            required = int(sum(max(0, int(v)) for v in exact_order_sku_counts))
        if required <= 0 and int(exact_disjoint_bom_sku_count or 0) > 0:
            required = int(order_num) * int(exact_disjoint_bom_sku_count)
        if required <= 0:
            return []
        stack_pool = [stack for stack in stack_list if int(getattr(stack, "current_height", 0)) < int(getattr(stack, "max_height", 0))]
        if not stack_pool:
            stack_pool = list(stack_list)
        added: List[Tote] = []
        for _ in range(max(1, len(skus_list) * max(1, int(target_min)))):
            counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
            qualified = [
                sku
                for sku in skus_list
                if int(target_min) <= int(counts.get(int(sku.id), 0)) <= int(target_max)
            ]
            if len(qualified) >= int(required):
                return added
            candidates = sorted(
                [
                    (int(target_min) - int(counts.get(int(sku.id), 0)), int(counts.get(int(sku.id), 0)), int(sku.id), sku)
                    for sku in skus_list
                    if int(counts.get(int(sku.id), 0)) < int(target_min)
                ],
                key=lambda row: (int(row[0]), -int(row[1]), int(row[2])),
            )
            if not candidates:
                break
            progressed = False
            for _, _, _, sku in candidates:
                sku_id = int(sku.id)
                present_stack_ids = set()
                for stack in stack_list:
                    for tote in getattr(stack, "totes", []) or []:
                        if sku_id in (getattr(tote, "sku_quantity_map", {}) or {}):
                            present_stack_ids.add(int(stack.stack_id))
                            break
                while len(present_stack_ids) < int(target_min):
                    target_stack = None
                    for stack in sorted(stack_pool, key=lambda item: (int(getattr(item, "current_height", 0)), int(item.stack_id))):
                        if int(stack.stack_id) not in present_stack_ids and int(getattr(stack, "current_height", 0)) < int(getattr(stack, "max_height", 0)):
                            target_stack = stack
                            break
                    if target_stack is None:
                        break
                    tote = CreateOFSProblem._add_sku_to_stack(target_stack, sku, int(next_tote_id), quantity=30)
                    next_tote_id += 1
                    added.append(tote)
                    present_stack_ids.add(int(target_stack.stack_id))
                    progressed = True
                if progressed:
                    break
            if not progressed:
                break
        counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
        qualified_count = sum(1 for sku in skus_list if int(target_min) <= int(counts.get(int(sku.id), 0)) <= int(target_max))
        if qualified_count < int(required):
            raise ValueError(
                f"redundancy_{int(target_min)}_{int(target_max)} has only {qualified_count} qualified SKU ids after repair; required={int(required)}"
            )
        return added

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
    def _build_bom_colocated_inventory(
            stack_list: List[Stack],
            skus_list: List[SKUs],
            orders: List[Order],
            tote_num: int,
            target_stack_count: int = 0,
            stack_min: int = 4,
            stack_max: int = 6,
            stack_counts: Tuple[int, ...] = (),
            disjoint_stack_groups: bool = False,
            support_multiplier: float = 2.5,
            cold_filler_probability: float = 0.25,
            sku_copy_count: int = 2,
            chunked_by_stack: bool = False,
    ) -> Tuple[List[Tote], List[Stack], Dict[str, Any]]:
        active_stacks = (
            CreateOFSProblem._select_spread_stacks_for_inventory(stack_list, int(target_stack_count))
            if int(target_stack_count or 0) > 0
            else list(stack_list)
        )
        active_stacks = list(active_stacks)
        sku_by_id = {int(sku.id): sku for sku in skus_list}
        demanded_sku_ids = sorted({
            int(sku_id)
            for order in (orders or [])
            for sku_id in (getattr(order, "order_product_id_list", []) or [])
        })
        demanded_set = set(demanded_sku_ids)
        cold_skus = [sku for sku in skus_list if int(sku.id) not in demanded_set]
        if not cold_skus:
            cold_skus = list(skus_list)

        tote_list: List[Tote] = []
        current_tote_id = 0
        reserved_load_by_stack: Dict[int, int] = {int(stack.stack_id): 0 for stack in active_stacks}
        bom_stack_groups: Dict[int, List[int]] = {}
        stack_by_id = {int(stack.stack_id): stack for stack in active_stacks}
        home_stacks_by_sku: Dict[int, List[int]] = {}
        reserved_bom_stack_ids: Set[int] = set()
        demand_sku_copy_count = max(1, int(sku_copy_count or 1))
        demand_sku_copy_count = min(2, int(demand_sku_copy_count))
        chunk_size = max(1, int(getattr(OFSConfig, "ROBOT_CAPACITY", 8)) - 2)

        def stack_sort_key(stack: Stack) -> Tuple[int, int, float, int]:
            return (
                int(reserved_load_by_stack.get(int(stack.stack_id), 0)),
                int(getattr(stack, "current_height", 0)),
                random.random(),
                int(stack.stack_id),
            )

        def pick_filler_skus(limit: int) -> List[SKUs]:
            pool = cold_skus if random.random() < max(0.0, min(1.0, float(cold_filler_probability))) else skus_list
            pool = [sku for sku in pool if int(sku.id) not in demanded_set] or list(pool) or list(skus_list)
            take = min(max(1, int(limit)), len(pool))
            return random.sample(pool, take)

        for order in sorted(orders or [], key=lambda item: int(getattr(item, "order_id", 0))):
            order_id = int(getattr(order, "order_id", -1))
            unique_sku_ids = sorted(set(int(sku_id) for sku_id in (getattr(order, "order_product_id_list", []) or [])))
            if not unique_sku_ids:
                continue
            if stack_counts and order_id >= 0 and order_id < len(stack_counts):
                group_size = max(1, int(stack_counts[order_id]))
                group_size = min(group_size, len(active_stacks))
            else:
                group_size = max(int(stack_min), int(math.ceil(float(len(unique_sku_ids)) / 2.0)))
                group_size = min(max(int(stack_min), group_size), max(int(stack_min), int(stack_max)), len(active_stacks))
            preselected_stack_ids = []
            if not bool(disjoint_stack_groups):
                for sku_id in unique_sku_ids:
                    for home_stack_id in home_stacks_by_sku.get(int(sku_id), []):
                        home_stack_id = int(home_stack_id)
                        if home_stack_id >= 0 and home_stack_id not in preselected_stack_ids and home_stack_id in stack_by_id:
                            preselected_stack_ids.append(int(home_stack_id))
            candidate_stacks = [stack_by_id[int(stack_id)] for stack_id in preselected_stack_ids[:group_size]]
            for stack in sorted(active_stacks, key=stack_sort_key):
                if len(candidate_stacks) >= group_size:
                    break
                if int(stack.stack_id) in {int(item.stack_id) for item in candidate_stacks}:
                    continue
                if bool(disjoint_stack_groups) and int(stack.stack_id) in reserved_bom_stack_ids:
                    continue
                candidate_stacks.append(stack)
            if len(candidate_stacks) < group_size:
                for stack in sorted(active_stacks, key=stack_sort_key):
                    if len(candidate_stacks) >= group_size:
                        break
                    if int(stack.stack_id) in {int(item.stack_id) for item in candidate_stacks}:
                        continue
                    candidate_stacks.append(stack)
            bom_stack_groups[int(order_id)] = [int(stack.stack_id) for stack in candidate_stacks]
            if bool(disjoint_stack_groups):
                reserved_bom_stack_ids.update(int(stack.stack_id) for stack in candidate_stacks)
            shared_home_stack_ids = set(int(stack_id) for stack_id in preselected_stack_ids)
            filler_candidate_stacks = [
                stack for stack in candidate_stacks
                if int(stack.stack_id) not in shared_home_stack_ids
            ] or list(candidate_stacks)

            demand_skus_by_stack: Dict[int, List[int]] = defaultdict(list)
            for idx, sku_id in enumerate(unique_sku_ids):
                sku_id = int(sku_id)
                if sku_id in home_stacks_by_sku:
                    target_stack_ids = [
                        int(stack_id)
                        for stack_id in home_stacks_by_sku.get(sku_id, [])
                        if int(stack_id) in stack_by_id
                    ]
                else:
                    target_stack_ids = []
                    for copy_idx in range(demand_sku_copy_count):
                        if bool(chunked_by_stack):
                            stack_idx = min(len(candidate_stacks) - 1, int(idx // chunk_size) + int(copy_idx))
                        else:
                            stack_idx = int(idx + copy_idx) % len(candidate_stacks)
                        stack = candidate_stacks[int(stack_idx)]
                        if int(getattr(stack, "current_height", 0)) >= int(getattr(stack, "max_height", 0)):
                            available_in_group = [
                                item for item in candidate_stacks
                                if int(getattr(item, "current_height", 0)) < int(getattr(item, "max_height", 0))
                            ]
                            if not available_in_group:
                                available_in_group = [
                                    item for item in active_stacks
                                    if int(getattr(item, "current_height", 0)) < int(getattr(item, "max_height", 0))
                                ]
                            if not available_in_group:
                                continue
                            stack = sorted(available_in_group, key=stack_sort_key)[0]
                            if int(stack.stack_id) not in {int(item.stack_id) for item in candidate_stacks}:
                                candidate_stacks.append(stack)
                                bom_stack_groups[int(order_id)] = [int(item.stack_id) for item in candidate_stacks]
                        target_stack_ids.append(int(stack.stack_id))
                    home_stacks_by_sku[sku_id] = list(dict.fromkeys(target_stack_ids))
                for stack_id in list(dict.fromkeys(target_stack_ids))[:demand_sku_copy_count]:
                    stack = stack_by_id.get(int(stack_id))
                    if stack is None:
                        continue
                    if int(getattr(stack, "current_height", 0)) >= int(getattr(stack, "max_height", 0)):
                        continue
                    demand_skus_by_stack[int(stack_id)].append(int(sku_id))

            for stack_id, sku_ids in sorted(demand_skus_by_stack.items(), key=lambda item: (len(item[1]), int(item[0]))):
                stack = stack_by_id.get(int(stack_id))
                if stack is None:
                    continue
                pending = list(dict.fromkeys(int(sku_id) for sku_id in sku_ids))
                while pending and int(getattr(stack, "current_height", 0)) < int(getattr(stack, "max_height", 0)):
                    take = min(len(pending), chunk_size if bool(chunked_by_stack) else random.randint(1, 3))
                    chunk = pending[:take]
                    pending = pending[take:]
                    sku_quantity_map = {
                        sku_by_id[int(sku_id)]: int(random.randint(20, 60))
                        for sku_id in chunk
                        if int(sku_id) in sku_by_id
                    }
                    if not sku_quantity_map:
                        continue
                    tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
                    current_tote_id += 1
                    reserved_load_by_stack[int(stack.stack_id)] = int(reserved_load_by_stack.get(int(stack.stack_id), 0)) + 1

            target_support = int(math.ceil(float(len(unique_sku_ids)) * max(1.0, float(support_multiplier))))
            filler_needed = max(0, int(target_support) - int(len(unique_sku_ids)))
            for filler_idx in range(filler_needed):
                stack = filler_candidate_stacks[int(filler_idx) % len(filler_candidate_stacks)]
                if int(getattr(stack, "current_height", 0)) >= int(getattr(stack, "max_height", 0)):
                    continue
                filler_skus = pick_filler_skus(random.randint(1, 3))
                sku_quantity_map = {sku: int(random.randint(8, 35)) for sku in filler_skus}
                tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
                current_tote_id += 1
                reserved_load_by_stack[int(stack.stack_id)] = int(reserved_load_by_stack.get(int(stack.stack_id), 0)) + 1

        bom_stack_id_set = {
            int(stack_id)
            for stack_ids in bom_stack_groups.values()
            for stack_id in stack_ids
        }
        while current_tote_id < int(tote_num):
            available = [
                stack for stack in active_stacks
                if int(getattr(stack, "current_height", 0)) < int(getattr(stack, "max_height", 0))
                and int(stack.stack_id) not in bom_stack_id_set
            ]
            if not available:
                break
            stack = sorted(available, key=lambda item: (int(reserved_load_by_stack.get(int(item.stack_id), 0)), random.random(), int(item.stack_id)))[0]
            filler_skus = pick_filler_skus(random.randint(2, 5))
            sku_quantity_map = {sku: int(random.randint(8, 35)) for sku in filler_skus}
            tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
            current_tote_id += 1
            reserved_load_by_stack[int(stack.stack_id)] = int(reserved_load_by_stack.get(int(stack.stack_id), 0)) + 1

        final_stack_list = [stack for stack in active_stacks if int(getattr(stack, "current_height", 0)) > 0]
        summary = CreateOFSProblem._compute_demanded_sku_redundancy_summary(orders, final_stack_list, target_distinct_stacks=0)
        summary.update(
            {
                "inventory_mode": "bom_colocated",
                "bom_stack_groups": {int(k): list(v) for k, v in bom_stack_groups.items()},
                "target_support_multiplier": float(support_multiplier),
                "cold_filler_probability": float(cold_filler_probability),
            }
        )
        return tote_list, final_stack_list, summary

    @staticmethod
    def _build_default_inventory(
            stack_list: List[Stack],
            skus_list: List[SKUs],
            tote_num: int,
            imbalance_profile: str = None,
            target_stack_count: int = 0,
            cold_filler_probability: float = 0.2,
            initial_unassigned_skus_per_tote: int = 4,
            max_sku_stack_count: int = 3,
    ) -> Tuple[List[Tote], List[Stack]]:
        tote_list: List[Tote] = []
        if int(target_stack_count or 0) > 0 and int(target_stack_count) < len(stack_list):
            stack_list = CreateOFSProblem._select_spread_stacks_for_inventory(stack_list, int(target_stack_count))
        if float(cold_filler_probability) <= 0.0:
            hot_sku_count = 1
        else:
            hot_sku_count = max(1, int(len(skus_list) * (0.1 if imbalance_profile == "uneven" else 0.2)))
        hot_skus = skus_list[:hot_sku_count]
        cold_skus = skus_list[hot_sku_count:]

        def sample_sku_by_popularity() -> SKUs:
            hot_probability = min(1.0, max(0.0, 1.0 - float(cold_filler_probability)))
            if random.random() < hot_probability:
                return random.choice(hot_skus)
            return random.choice(cold_skus) if cold_skus else random.choice(hot_skus)

        desired_tote_count = max(tote_num, hot_sku_count * 5)
        unassigned_skus: Dict[int, SKUs] = {int(sku.id): sku for sku in skus_list}
        current_tote_id = 0

        if int(target_stack_count or 0) > 0:
            for stack in list(stack_list):
                if current_tote_id >= desired_tote_count:
                    break
                selected_source = sorted(unassigned_skus.keys())
                if selected_source:
                    take_unassigned = max(1, int(initial_unassigned_skus_per_tote or 4))
                    selected_ids = random.sample(selected_source, min(take_unassigned, len(selected_source)))
                    sku_quantity_map = {unassigned_skus[sku_id]: int(random.randint(15, 50)) for sku_id in selected_ids}
                    for sku_id in selected_ids:
                        unassigned_skus.pop(sku_id, None)
                else:
                    if float(cold_filler_probability) <= 0.0:
                        current_stack_sku_ids = set()
                        for tote in getattr(stack, "totes", []) or []:
                            current_stack_sku_ids.update(int(sku_id) for sku_id in (getattr(tote, "sku_quantity_map", {}) or {}).keys())
                        stack_counts = CreateOFSProblem._sku_stack_counts(stack_list, skus_list)
                        filler_pool = [
                            sku
                            for sku in skus_list
                            if int(stack_counts.get(int(sku.id), 0)) < int(max_sku_stack_count) and int(sku.id) not in current_stack_sku_ids
                        ]
                        if not filler_pool:
                            filler_pool = [
                                sku
                                for sku in skus_list
                                if int(sku.id) not in current_stack_sku_ids
                            ] or list(skus_list)
                    else:
                        filler_pool = skus_list
                    selected_skus = random.sample(filler_pool, min(4, len(filler_pool)))
                    sku_quantity_map = {sku: int(random.randint(10, 40)) for sku in selected_skus}
                tote_list.append(CreateOFSProblem._register_tote(stack, current_tote_id, sku_quantity_map))
                current_tote_id += 1

        shuffled_stacks = list(stack_list)
        random.shuffle(shuffled_stacks)
        for stack in shuffled_stacks:
            if not unassigned_skus or current_tote_id >= desired_tote_count:
                break
            take_unassigned = max(1, int(initial_unassigned_skus_per_tote or 4))
            selected_ids = random.sample(sorted(unassigned_skus.keys()), min(take_unassigned, len(unassigned_skus)))
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
    def _select_spread_stacks_for_inventory(stack_list: List[Stack], required_count: int) -> List[Stack]:
        required = max(0, int(required_count))
        if required <= 0 or required >= len(stack_list):
            return list(stack_list)
        remaining: Dict[int, Stack] = {int(stack.stack_id): stack for stack in stack_list}
        selected: List[Stack] = []
        while remaining and len(selected) < required:
            rows = []
            for stack in remaining.values():
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
                rows.append((-min_distance, -avg_distance, int(stack.stack_id)))
            rows.sort()
            selected.append(remaining.pop(int(rows[0][-1])))
        return sorted(selected, key=lambda stack: int(stack.stack_id))

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
            qty_range: Tuple[int, int] = (),
            base_seed: int = OFSConfig.RANDOM_SEED,
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
            use_range = len(qty_range or ()) == 2
            qty_low = max(1, int(qty_range[0])) if use_range else 1
            qty_high = max(qty_low, int(qty_range[1])) if use_range else int(max_quantity_per_sku)
            for sku_id in selected_ids:
                qty = random.randint(qty_low, qty_high)
                order_product_id_list.extend([int(sku_id)] * int(qty))
                total_qty += int(qty)

            order.order_product_id_list = order_product_id_list
            order.order_skus_number = total_qty
            CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
            order.status = "pending"
            orders.append(order)
        return orders

    @staticmethod
    def _generate_orders(
            max_sku_types_per_order: int,
            max_quantity_per_sku: int,
            num_orders: int,
            skus_list: List[SKUs],
            imbalance_profile: str = None,
            qty_range: Tuple[int, int] = (),
            base_seed: int = OFSConfig.RANDOM_SEED,
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
            use_range = len(qty_range or ()) == 2
            qty_low = max(1, int(qty_range[0])) if use_range else 1
            qty_high = max(qty_low, int(qty_range[1])) if use_range else int(max_quantity_per_sku)

            for sku in selected_skus:
                if use_range:
                    qty = random.randint(qty_low, qty_high)
                elif imbalance_profile == "uneven" and random.random() < 0.3:
                    qty = random.randint(max(1, max_quantity_per_sku // 2), max_quantity_per_sku + 2)
                else:
                    qty = random.randint(1, max_quantity_per_sku)
                order_product_id_list.extend([int(sku.id)] * int(qty))
                total_qty += int(qty)

            order.order_product_id_list = order_product_id_list
            order.order_skus_number = total_qty
            CreateOFSProblem._assign_order_time_window(order, base_seed=int(base_seed))
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
