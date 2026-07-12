import unittest
import math
from collections import defaultdict
from types import SimpleNamespace

import gurobipy as gp

from config.ofs_config import OFSConfig
from problemDto.createInstance import CreateOFSProblem
from Gurobi.global_xyzu import GlobalXYZUConfig, GlobalXYZUSolver, RankAwareGlobalTimeCalculator, RouteNodeSpec, RouteTaskSpec


class GlobalXYZUSolverTests(unittest.TestCase):
    def test_generate_gurobi_s1_scale(self):
        problem = CreateOFSProblem.generate_problem_by_scale("Gurobi-s1", seed=42)
        self.assertEqual(problem.scale_name, "GUROBI-S1")
        self.assertEqual(problem.order_num, 1)
        self.assertEqual(problem.robot_num, 2)
        self.assertEqual(problem.station_num, 2)
        self.assertEqual(problem.tote_num, 30)
        self.assertEqual(len(problem.order_list), 1)
        self.assertEqual(len(problem.order_list[0].order_product_id_list), 10)
        self.assertEqual(len(set(problem.order_list[0].order_product_id_list)), 10)

    def test_generate_gurobi_s1_to_s9_scale_ladder(self):
        expected = {
            "GUROBI-S1": ((2, 4), 2, 2, 30, 1, 10),
            "GUROBI-S2": ((2, 4), 2, 2, 50, 2, 15),
            "GUROBI-S3": ((3, 4), 3, 3, 80, 3, 20),
            "GUROBI-S4": ((3, 4), 3, 3, 100, 4, 25),
            "GUROBI-S5": ((3, 5), 4, 4, 120, 5, 30),
            "GUROBI-S6": ((4, 5), 5, 4, 150, 6, 40),
            "GUROBI-S7": ((4, 6), 5, 5, 180, 7, 45),
            "GUROBI-S8": ((4, 6), 6, 6, 200, 8, 50),
            "GUROBI-S9": ((5, 6), 7, 6, 240, 9, 60),
        }
        for scale, (map_size, robots, stations, totes, orders, skus) in expected.items():
            with self.subTest(scale=scale):
                problem = CreateOFSProblem.generate_problem_by_scale(scale, seed=42)
                self.assertEqual(problem.scale_name, scale)
                self.assertEqual(
                    (
                        int(problem.map.warehouse_length_block_number),
                        int(problem.map.warehouse_width_block_number),
                    ),
                    map_size,
                )
                self.assertEqual(problem.robot_num, robots)
                self.assertEqual(problem.station_num, stations)
                self.assertEqual(problem.tote_num, totes)
                self.assertEqual(problem.order_num, orders)
                self.assertEqual(problem.skus_num, skus)
                self.assertGreater(len(problem.stack_list), 0)

    def test_generate_tiny3_scale_has_three_disjoint_10sku_boms(self):
        problem = CreateOFSProblem.generate_problem_by_scale("TINY3", seed=42)
        self.assertEqual(problem.scale_name, "TINY3")
        self.assertEqual(problem.order_num, 3)
        self.assertEqual(problem.robot_num, 2)
        self.assertEqual(problem.station_num, 2)
        self.assertEqual(problem.skus_num, 30)
        self.assertEqual(problem.tote_num, 150)
        self.assertEqual(len(problem.order_list), 3)

        order_sku_sets = []
        for order in problem.order_list:
            sku_ids = [int(sku_id) for sku_id in order.order_product_id_list]
            sku_set = set(sku_ids)
            self.assertEqual(len(sku_ids), 10)
            self.assertEqual(len(sku_set), 10)
            order_sku_sets.append(sku_set)

        self.assertEqual(len(set().union(*order_sku_sets)), 30)
        for i in range(len(order_sku_sets)):
            for j in range(i + 1, len(order_sku_sets)):
                self.assertFalse(order_sku_sets[i].intersection(order_sku_sets[j]))

    def test_slot_upper_bound_uses_heuristic_slack(self):
        cfg = GlobalXYZUConfig(slot_slack_per_order=2)
        bound = GlobalXYZUSolver._slot_upper_bound(unique_sku_count=10, heuristic_subtask_count=3, cap_limit=6, cfg=cfg)
        self.assertEqual(bound, 3)

        relaxed_cfg = GlobalXYZUConfig(slot_slack_per_order=2, enable_tight_slot_upper_bound=False)
        relaxed_bound = GlobalXYZUSolver._slot_upper_bound(unique_sku_count=10, heuristic_subtask_count=3, cap_limit=6, cfg=relaxed_cfg)
        self.assertEqual(relaxed_bound, 5)

    def test_integrated_u_route_config_defaults_enabled(self):
        cfg = GlobalXYZUConfig()
        self.assertTrue(cfg.integrate_u_route)
        self.assertTrue(cfg.route_arc_prune)
        self.assertTrue(cfg.u_same_slot_same_robot)
        self.assertTrue(cfg.warm_start_use_sp4)
        self.assertIsNone(cfg.route_big_m_time)
        self.assertEqual(float(cfg.bom_arrival_window_sec), 60.0)
        self.assertTrue(cfg.enable_global_arrival_workload_lb)
        self.assertTrue(cfg.enable_route_time_window_arc_prune)
        self.assertTrue(cfg.enable_route_load_interval_arc_prune)
        self.assertFalse(cfg.enable_route_service_sec_cuts)

    def test_effective_order_span_limit_uses_dynamic_bom_window(self):
        cfg = GlobalXYZUConfig(bom_arrival_window_sec=60.0)
        order = SimpleNamespace(unique_sku_count=8)
        self.assertAlmostEqual(
            GlobalXYZUSolver._effective_order_span_limit_sec(order, cfg),
            120.0,
        )
        self.assertAlmostEqual(
            OFSConfig.effective_bom_arrival_window_sec(cfg.bom_arrival_window_sec, order.unique_sku_count),
            120.0,
        )

    def test_rank_aware_calculator_respects_explicit_zero_pick_count(self):
        class DummyTask:
            sku_pick_count = 0
            hit_tote_ids = [1, 2]

        self.assertEqual(RankAwareGlobalTimeCalculator._task_pick_count(DummyTask()), 0)

    def test_allocate_pick_count_consumes_remaining(self):
        problem = CreateOFSProblem.generate_problem_by_scale("SMALL", seed=42)
        solver = GlobalXYZUSolver()
        warm = solver._build_warm_start(problem, GlobalXYZUConfig(warm_start_sp4_time_limit_sec=3))
        hit_totes = []
        for rows in warm.subtask_by_order.values():
            for st in rows:
                for task in getattr(st, "execution_tasks", []) or []:
                    if getattr(task, "hit_tote_ids", None):
                        hit_totes = list(task.hit_tote_ids)
                        break
                if hit_totes:
                    break
            if hit_totes:
                break
        self.assertTrue(hit_totes)
        tote = problem.id_to_tote[int(hit_totes[0])]
        first_sku_id = int(next(iter(tote.sku_quantity_map.keys())))
        remaining = defaultdict(int)
        remaining[first_sku_id] = 3
        picked = solver._allocate_pick_count(problem, remaining, [int(hit_totes[0])])
        self.assertGreaterEqual(picked, 1)
        self.assertLessEqual(remaining[first_sku_id], 2)

    def test_dynamic_time_big_m_uses_warm_start_bound(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(warm_start_use_sp4=False)
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        value, diag = solver._compute_dynamic_time_big_m(prepared, cfg, {(0, 1): 12.0}, {})
        self.assertGreater(value, 0.0)
        self.assertLess(value, float(cfg.big_m_time))
        self.assertIn(diag.get("time_big_m_source"), {"warm_start_dynamic", "3x_warm_start_route_end", "3x_warm_start_makespan"})
        self.assertGreater(float(diag.get("time_big_m_warm_makespan", 0.0)), 0.0)
        self.assertIn("route_node_time_ub", diag)
        self.assertIn("route_arc_time_m", diag)
        self.assertIn("route_node_time_ub_max", diag)
        self.assertIn("route_arc_time_m_max", diag)

    def test_prepare_filters_candidate_stacks_and_totes_by_demand_sku(self):
        problem = CreateOFSProblem.generate_problem_by_scale("Gurobi-s1", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(warm_start_use_sp4=False)
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        order_id = int(problem.order_list[0].order_id)
        demand_sku_ids = set(int(sku_id) for sku_id in problem.order_list[0].order_product_id_list)
        demand_hit_totes = list(prepared["demand_hit_totes_by_order"][order_id])
        support_totes = list(prepared["support_totes_by_order"][order_id])
        candidate_stacks = set(int(stack_id) for stack_id in prepared["candidate_stacks_by_order"][order_id])
        warm_stacks = {
            int(getattr(task, "target_stack_id", -1))
            for st in warm.subtask_by_order.get(order_id, [])
            for task in (getattr(st, "execution_tasks", []) or [])
            if int(getattr(task, "target_stack_id", -1)) >= 0
        }
        demand_hit_stacks = {int(prepared["tote_to_stack"][int(tote_id)]) for tote_id in demand_hit_totes}

        self.assertTrue(demand_hit_totes)
        self.assertTrue(support_totes)
        self.assertTrue(warm_stacks.issubset(candidate_stacks))
        self.assertTrue(candidate_stacks.issubset(demand_hit_stacks | warm_stacks))
        for tote_id in demand_hit_totes:
            tote = problem.id_to_tote[int(tote_id)]
            self.assertTrue(any(int(sku_id) in demand_sku_ids for sku_id in tote.sku_quantity_map.keys()))
        self.assertTrue(
            {int(prepared["tote_to_stack"][int(tote_id)]) for tote_id in support_totes}.issubset(candidate_stacks)
        )

    def test_rebuild_warm_route_continuous_start(self):
        route_tasks = {
            0: RouteTaskSpec(task_key=0, slot_id=10, stack_id=100, station_id=0, pickup_node=1, delivery_node=2),
        }
        route_nodes = {
            0: RouteNodeSpec(0, "start", -1, -1, -1, -1, 0.0, 0.0),
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "delivery", 0, 10, 100, 0, 2.0, 0.0),
            5: RouteNodeSpec(5, "end", -1, -1, -1, -1, 0.0, 0.0, robot_id=0),
            6: RouteNodeSpec(6, "start", -1, -1, -1, -1, 0.0, 0.0, robot_id=1),
            7: RouteNodeSpec(7, "end", -1, -1, -1, -1, 0.0, 0.0, robot_id=1),
        }
        rebuilt = GlobalXYZUSolver._rebuild_warm_route_continuous_start(
            selected_route_rows=[
                {
                    "slot_id": 10,
                    "route_key": 0,
                    "task_id": 0,
                    "robot_id": 0,
                    "warm_stack_arrival": 7.0,
                    "service_time": 5.0,
                    "load": 2,
                }
            ],
            robot_ids=[0, 1],
            route_start_nodes={0: 0, 1: 6},
            route_end_nodes={0: 5, 1: 7},
            route_tasks=route_tasks,
            route_nodes=route_nodes,
            route_tau={(0, 1): 4.0, (1, 2): 6.0, (2, 5): 3.0, (6, 7): 0.0},
            route_arc_keys={(0, 1), (1, 2), (2, 5), (6, 7)},
            robot_capacity=8,
            route_arc_prune=True,
        )
        self.assertTrue(rebuilt["ok"])
        self.assertEqual(rebuilt["missing_arc_count"], 0)
        self.assertEqual(rebuilt["capacity_violation_count"], 0)
        self.assertAlmostEqual(rebuilt["route_time_start"][1], 4.0)
        self.assertAlmostEqual(rebuilt["route_time_start"][2], 15.0)
        self.assertAlmostEqual(rebuilt["route_time_start"][5], 18.0)
        self.assertAlmostEqual(rebuilt["route_finish_start"][0], 18.0)
        self.assertAlmostEqual(rebuilt["route_finish_start"][1], 0.0)
        self.assertAlmostEqual(rebuilt["route_load_start"][1], 2.0)
        self.assertAlmostEqual(rebuilt["route_load_start"][2], 0.0)
        self.assertEqual(rebuilt["pass_x_start"][(1, 0)], 1.0)
        self.assertAlmostEqual(rebuilt["slot_arrival_lower"][10], 15.0)

    def test_rebuild_warm_route_continuous_start_keeps_cross_slot_batch_when_prune_disabled(self):
        route_tasks = {
            0: RouteTaskSpec(task_key=0, slot_id=10, stack_id=100, station_id=0, pickup_node=1, delivery_node=2, estimated_load=2),
            1: RouteTaskSpec(task_key=1, slot_id=11, stack_id=101, station_id=0, pickup_node=3, delivery_node=4, estimated_load=2),
        }
        route_nodes = {
            0: RouteNodeSpec(0, "start", -1, -1, -1, -1, 0.0, 0.0),
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "delivery", 0, 10, 100, 0, 2.0, 0.0),
            3: RouteNodeSpec(3, "pickup", 1, 11, 101, 0, 3.0, 0.0),
            4: RouteNodeSpec(4, "delivery", 1, 11, 101, 0, 4.0, 0.0),
            5: RouteNodeSpec(5, "end", -1, -1, -1, -1, 0.0, 0.0),
        }
        rebuilt = GlobalXYZUSolver._rebuild_warm_route_continuous_start(
            selected_route_rows=[
                {"slot_id": 10, "route_key": 0, "task_id": 0, "robot_id": 0, "trip_id": 0, "station_id": 0, "robot_visit_sequence": 0, "warm_stack_arrival": 1.0, "warm_station_arrival": 10.0, "service_time": 5.0, "load": 2},
                {"slot_id": 11, "route_key": 1, "task_id": 1, "robot_id": 0, "trip_id": 0, "station_id": 0, "robot_visit_sequence": 1, "warm_stack_arrival": 2.0, "warm_station_arrival": 10.0, "service_time": 7.0, "load": 2},
            ],
            robot_ids=[0],
            route_start_nodes={0: 0},
            route_end_nodes={0: 5},
            route_tasks=route_tasks,
            route_nodes=route_nodes,
            route_tau={(0, 1): 1.0, (1, 3): 2.0, (3, 2): 3.0, (2, 4): 4.0, (4, 5): 5.0},
            route_arc_keys={(0, 1), (1, 3), (3, 2), (2, 4), (4, 5)},
            robot_capacity=8,
            route_arc_prune=False,
        )
        self.assertTrue(rebuilt["ok"])
        self.assertIn((1, 3), rebuilt["route_arc_start"])
        self.assertIn((3, 2), rebuilt["route_arc_start"])
        self.assertNotIn((1, 2), rebuilt["route_arc_start"])
        self.assertAlmostEqual(rebuilt["route_finish_start"][0], 27.0)

    def test_route_arc_allowed_enforces_station_affinity_for_cross_slot(self):
        route_tasks = {
            0: RouteTaskSpec(task_key=0, slot_id=10, stack_id=100, station_id=0, pickup_node=1, delivery_node=2, estimated_load=2),
            1: RouteTaskSpec(task_key=1, slot_id=11, stack_id=101, station_id=1, pickup_node=3, delivery_node=4, estimated_load=2),
        }
        route_nodes = {
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "delivery", 0, 10, 100, 0, 2.0, 0.0),
            3: RouteNodeSpec(3, "pickup", 1, 11, 101, 1, 3.0, 0.0),
            4: RouteNodeSpec(4, "delivery", 1, 11, 101, 1, 4.0, 0.0),
        }
        self.assertFalse(
            GlobalXYZUSolver._route_arc_allowed(
                1, 3, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=True, robot_capacity=8
            )
        )
        self.assertFalse(
            GlobalXYZUSolver._route_arc_allowed(
                2, 4, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=True, robot_capacity=8
            )
        )
        self.assertTrue(
            GlobalXYZUSolver._route_arc_allowed(
                2, 3, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=True, robot_capacity=8
            )
        )

        route_tasks[1] = RouteTaskSpec(task_key=1, slot_id=11, stack_id=101, station_id=0, pickup_node=3, delivery_node=4, estimated_load=2)
        route_nodes[3] = RouteNodeSpec(3, "pickup", 1, 11, 101, 0, 3.0, 0.0)
        route_nodes[4] = RouteNodeSpec(4, "delivery", 1, 11, 101, 0, 4.0, 0.0)
        self.assertTrue(
            GlobalXYZUSolver._route_arc_allowed(
                1, 3, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=True, robot_capacity=8
            )
        )
        self.assertTrue(
            GlobalXYZUSolver._route_arc_allowed(
                1, 4, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=True, robot_capacity=8
            )
        )
        self.assertTrue(
            GlobalXYZUSolver._route_arc_allowed(
                2, 4, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=True, robot_capacity=8
            )
        )

    def test_route_arc_allowed_enforces_capacity_bound_for_pickup_pickup(self):
        route_tasks = {
            0: RouteTaskSpec(task_key=0, slot_id=10, stack_id=100, station_id=0, pickup_node=1, delivery_node=2, estimated_load=5),
            1: RouteTaskSpec(task_key=1, slot_id=11, stack_id=101, station_id=0, pickup_node=3, delivery_node=4, estimated_load=4),
        }
        route_nodes = {
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "delivery", 0, 10, 100, 0, 2.0, 0.0),
            3: RouteNodeSpec(3, "pickup", 1, 11, 101, 0, 3.0, 0.0),
            4: RouteNodeSpec(4, "delivery", 1, 11, 101, 0, 4.0, 0.0),
        }
        self.assertFalse(
            GlobalXYZUSolver._route_arc_allowed(
                1, 3, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=False, robot_capacity=8
            )
        )
        self.assertTrue(
            GlobalXYZUSolver._route_arc_allowed(
                1, 4, route_nodes=route_nodes, route_tasks=route_tasks, route_arc_prune=False, robot_capacity=8
            )
        )

    def test_route_resource_prune_drops_time_window_arcs_but_keeps_protected(self):
        route_tasks = {
            0: RouteTaskSpec(task_key=0, slot_id=10, stack_id=100, station_id=0, pickup_node=1, delivery_node=2, estimated_load=1),
            1: RouteTaskSpec(task_key=1, slot_id=11, stack_id=101, station_id=0, pickup_node=3, delivery_node=4, estimated_load=1),
        }
        route_nodes = {
            0: RouteNodeSpec(0, "start", -1, -1, -1, -1, 0.0, 0.0, robot_id=0),
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "delivery", 0, 10, 100, 0, 2.0, 0.0),
            3: RouteNodeSpec(3, "pickup", 1, 11, 101, 0, 3.0, 0.0),
            4: RouteNodeSpec(4, "delivery", 1, 11, 101, 0, 4.0, 0.0),
            9: RouteNodeSpec(9, "end", -1, -1, -1, -1, 0.0, 0.0, robot_id=0),
        }
        route_tau = {
            (0, 1): 1.0,
            (0, 3): 1.0,
            (1, 2): 3.0,
            (3, 4): 3.0,
            (2, 9): 1.0,
            (4, 9): 1.0,
            (1, 3): 100.0,
            (2, 3): 100.0,
        }
        pruned, diag = GlobalXYZUSolver._prune_route_arcs_by_resource_bounds(
            route_nodes=route_nodes,
            route_tasks=route_tasks,
            route_arcs=list(route_tau.keys()),
            route_tau=route_tau,
            route_start_nodes={0: 0},
            route_end_nodes={0: 9},
            pickup_service_lb_by_node={1: 1.0, 3: 1.0},
            pickup_service_ub_by_node={1: 1.0, 3: 1.0},
            slot_time_ub=10.0,
            robot_capacity=8,
            enable_time_window=True,
            enable_load_interval=False,
            protected_arcs={(2, 3)},
        )
        self.assertNotIn((1, 3), pruned)
        self.assertIn((2, 3), pruned)
        self.assertEqual(int(diag["u_time_window_pruned_arc_count"]), 1)
        self.assertEqual(int(diag["u_protected_arc_kept_after_prune_count"]), 1)

    def test_route_arc_knn_prunes_pickup_neighbors_but_preserves_delivery_arcs(self):
        route_nodes = {
            0: RouteNodeSpec(0, "start", -1, -1, -1, -1, 0.0, 0.0),
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "pickup", 1, 11, 101, 0, 2.0, 0.0),
            3: RouteNodeSpec(3, "pickup", 2, 12, 102, 0, 3.0, 0.0),
            4: RouteNodeSpec(4, "pickup", 3, 13, 103, 0, 4.0, 0.0),
            5: RouteNodeSpec(5, "pickup", 4, 14, 104, 0, 5.0, 0.0),
            6: RouteNodeSpec(6, "pickup", 5, 15, 105, 0, 6.0, 0.0),
            7: RouteNodeSpec(7, "pickup", 6, 16, 106, 0, 7.0, 0.0),
            20: RouteNodeSpec(20, "delivery", 0, 10, 100, 0, 0.0, 1.0),
            21: RouteNodeSpec(21, "delivery", 1, 11, 101, 0, 0.0, 2.0),
            99: RouteNodeSpec(99, "end", -1, -1, -1, -1, 0.0, 0.0),
        }
        route_arcs = []
        route_tau = {}
        for pickup_id in range(1, 8):
            route_arcs.append((0, pickup_id))
            route_tau[(0, pickup_id)] = float(pickup_id)
            route_arcs.append((pickup_id, 20))
            route_tau[(pickup_id, 20)] = 1.0
            route_arcs.append((pickup_id, 21))
            route_tau[(pickup_id, 21)] = 2.0
            route_arcs.append((20, pickup_id))
            route_tau[(20, pickup_id)] = float(10 + pickup_id)
        for pickup_id in range(2, 8):
            route_arcs.append((1, pickup_id))
            route_tau[(1, pickup_id)] = float(pickup_id) + 0.25
        route_arcs.extend([(20, 99), (21, 99), (20, 21)])
        route_tau[(20, 99)] = 1.0
        route_tau[(21, 99)] = 1.0
        route_tau[(20, 21)] = 1.0

        pruned_arcs, diag = GlobalXYZUSolver._prune_route_arcs_by_knn(
            route_nodes=route_nodes,
            route_arcs=route_arcs,
            route_tau=route_tau,
            route_start_node=0,
            pickup_neighbor_limit=5,
        )
        start_pickup_arcs = [(i, j) for i, j in pruned_arcs if i == 0 and route_nodes[j].kind == "pickup"]
        self.assertGreater(diag["u_knn_pruned_arc_count"], 0)
        self.assertGreaterEqual(diag["u_legal_arc_count_before_knn"], diag["u_arc_count_after_knn"])
        self.assertEqual(len(start_pickup_arcs), 7)
        self.assertIn((1, 20), pruned_arcs)
        self.assertIn((1, 21), pruned_arcs)
        self.assertEqual(len([(i, j) for i, j in pruned_arcs if i == 1 and route_nodes[j].kind == "pickup"]), 5)
        for pickup_id in range(1, 8):
            self.assertIn((0, pickup_id), pruned_arcs)
        for pickup_id in range(1, 8):
            self.assertTrue(
                any(
                    j == pickup_id and route_nodes[i].kind in {"start", "delivery"}
                    for i, j in pruned_arcs
                )
            )

    def test_route_transition_knn_prunes_delivery_pickup_but_keeps_protected_and_inbound(self):
        route_nodes = {
            0: RouteNodeSpec(0, "start", -1, -1, -1, -1, 0.0, 0.0),
            1: RouteNodeSpec(1, "pickup", 0, 10, 100, 0, 1.0, 0.0),
            2: RouteNodeSpec(2, "pickup", 1, 11, 101, 0, 2.0, 0.0),
            3: RouteNodeSpec(3, "pickup", 2, 12, 102, 0, 3.0, 0.0),
            20: RouteNodeSpec(20, "delivery", 0, 10, 100, 0, 0.0, 1.0),
            21: RouteNodeSpec(21, "delivery", 1, 11, 101, 0, 0.0, 2.0),
            99: RouteNodeSpec(99, "end", -1, -1, -1, -1, 0.0, 0.0),
        }
        route_arcs = [(0, 1), (0, 2), (0, 3), (20, 1), (20, 2), (20, 3), (21, 1), (21, 2), (21, 3), (20, 99), (21, 99)]
        route_tau = {arc: float(index + 1) for index, arc in enumerate(route_arcs)}
        protected = {(20, 3)}
        pruned_arcs, diag = GlobalXYZUSolver._prune_route_arcs_by_knn(
            route_nodes=route_nodes,
            route_arcs=route_arcs,
            route_tau=route_tau,
            route_start_node=0,
            pickup_neighbor_limit=1,
            protected_arcs=protected,
            prune_delivery_pickup=True,
        )
        self.assertIn((20, 3), pruned_arcs)
        self.assertEqual(len([(i, j) for i, j in pruned_arcs if i == 21 and route_nodes[j].kind == "pickup"]), 1)
        self.assertTrue(bool(diag["u_transition_knn_prune_enabled"]))
        for pickup_id in (1, 2, 3):
            self.assertTrue(any(j == pickup_id for _i, j in pruned_arcs))

    def test_rebuild_warm_slot_continuous_start_projects_arrival(self):
        rebuilt = GlobalXYZUSolver._rebuild_warm_slot_continuous_start(
            active_slot_rows=[(1, 0, 0), (2, 0, 1), (3, 0, 2)],
            slot_arrival_lower={1: 10.0, 2: 5.0, 3: 6.0},
            slot_unit_count={1: 2, 2: 1, 3: 1},
            slot_noise_count={1: 0, 2: 0, 3: 0},
            picking_time=3.0,
            move_extra_tote_time=1.0,
            route_end_max=0.0,
        )
        self.assertAlmostEqual(rebuilt["arrival_start"][1], 10.0)
        self.assertAlmostEqual(rebuilt["arrival_start"][2], 10.0)
        self.assertAlmostEqual(rebuilt["arrival_start"][3], 10.0)
        self.assertAlmostEqual(rebuilt["start_start"][1], 10.0)
        self.assertAlmostEqual(rebuilt["finish_start"][1], 16.0)
        self.assertAlmostEqual(rebuilt["start_start"][2], 16.0)
        self.assertAlmostEqual(rebuilt["finish_start"][2], 19.0)
        self.assertAlmostEqual(rebuilt["start_start"][3], 19.0)
        self.assertAlmostEqual(rebuilt["finish_start"][3], 22.0)
        self.assertAlmostEqual(rebuilt["model_cmax"], 22.0)

    def test_swap_first_two_robot_ids_by_path_duration(self):
        remapped_rows, robot_duration, robot_id_map, swapped = GlobalXYZUSolver._swap_first_two_robot_ids_by_path_duration(
            selected_route_rows=[
                {"slot_id": 10, "robot_id": 0, "warm_station_arrival": 8.0},
                {"slot_id": 11, "robot_id": 1, "warm_station_arrival": 15.0},
            ],
            robot_ids=[0, 1],
        )
        self.assertTrue(swapped)
        self.assertEqual(robot_duration[0], 8.0)
        self.assertEqual(robot_duration[1], 15.0)
        self.assertEqual(robot_id_map[0], 1)
        self.assertEqual(robot_id_map[1], 0)
        self.assertEqual(remapped_rows[0]["robot_id"], 1)
        self.assertEqual(remapped_rows[1]["robot_id"], 0)

    def test_route_finish_lb_matches_total_service_plus_max_dist(self):
        route_finish_lb = GlobalXYZUSolver._evaluate_route_finish_lb_from_visits(
            pickup_nodes=[1, 3],
            robot_ids=[0, 1],
            service_lb_by_pickup={1: 5.0, 3: 7.0},
            dist_lb_by_pickup={1: 11.0, 3: 13.0},
            route_visit_solution={(1, 0): 1.0, (3, 0): 1.0, (1, 1): 0.0, (3, 1): 0.0},
        )
        self.assertAlmostEqual(route_finish_lb[0], 25.0)
        self.assertAlmostEqual(route_finish_lb[1], 0.0)

    def test_build_model_contains_strengthened_no_relay_bounds(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(warm_start_use_sp4=False, integrate_u_route=True)
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        model = gp.Model("xyzu_bounds_test")
        model.Params.OutputFlag = 0
        payload = solver._build_model(model, prepared, cfg)
        model.update()
        constr_names = {constr.ConstrName for constr in model.getConstrs()}
        gen_constr_names = {gen_constr.GenConstrName for gen_constr in model.getGenConstrs()}
        self.assertTrue(any(name.startswith("ActiveStackNeedsLoad_") for name in constr_names))
        self.assertTrue(any(name.startswith("PairNeedsLoad_") for name in constr_names))
        self.assertIn("Global_Robot_Service_Only_Bound", constr_names)
        self.assertIn("Global_Robot_TravelService_Capacity_Bound", constr_names)
        self.assertTrue(any(name.startswith("RouteFinishTaskLB_") for name in constr_names))
        self.assertTrue(any(name.startswith("StationArrivalClockBind_") for name in constr_names))
        self.assertFalse(any(name.startswith("ArrivalSeq_") for name in constr_names))
        self.assertTrue(any(name.startswith("StationArrivalClockEq_") for name in gen_constr_names))
        self.assertTrue(any(name.startswith("StationStartAfterPrev_") for name in gen_constr_names))
        self.assertTrue(any(name.startswith("OrderArrivalLBLink_") for name in gen_constr_names))
        self.assertTrue(any(name.startswith("RoutePickupBeforeDelivery_") for name in gen_constr_names))
        self.assertTrue(any(name.startswith("RouteLoadLB_") for name in gen_constr_names))
        self.assertNotIn("station_use", payload)
        self.assertIsNotNone(payload["station_arrival_clock"])
        self.assertIsNotNone(payload["station_finish_clock"])
        self.assertIsNotNone(payload["order_arrival_lb"])
        self.assertIsNotNone(payload["order_arrival_ub"])
        self.assertTrue(payload["route_node_time_ub"])
        self.assertTrue(payload["route_arc_time_m"])
        self.assertIn("pass_x", payload)
        first_spec = next(iter(payload["route_tasks"].values()))
        pickup_ub = float(payload["route_node_time_ub"][int(first_spec.pickup_node)])
        delivery_ub = float(payload["route_node_time_ub"][int(first_spec.delivery_node)])
        self.assertLess(pickup_ub, delivery_ub)
        self.assertLessEqual(delivery_ub, float(payload["slot_time_ub"]))
        self.assertEqual(
            int(payload["diagnostics"].get("auto_max_rank", 0)),
            int(math.ceil(float(len(prepared["slots"])) / max(1, len(problem.station_list))) + 4),
        )
        self.assertEqual(int(payload["max_rank"]), int(payload["diagnostics"].get("effective_max_rank", 0)))
        self.assertGreaterEqual(
            int(payload["diagnostics"].get("u_legal_arc_count_before_knn", 0)),
            int(payload["diagnostics"].get("u_arc_count_after_knn", 0)),
        )
        self.assertGreaterEqual(float(payload["diagnostics"].get("global_robot_capacity_trip_lb", 0.0) or 0.0), float(payload["diagnostics"].get("global_robot_service_only_lb", 0.0) or 0.0))

    def test_build_model_disables_bom_arrival_window_when_nonpositive(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(warm_start_use_sp4=False, integrate_u_route=True, bom_arrival_window_sec=0.0)
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        model = gp.Model("xyzu_no_bom_window_test")
        model.Params.OutputFlag = 0
        solver._build_model(model, prepared, cfg)
        model.update()
        constr_names = {constr.ConstrName for constr in model.getConstrs()}
        self.assertFalse(any(name.startswith("OrderArrivalWindow_") for name in constr_names))

    def test_build_model_adds_route_lb_probe_cuts_when_enabled(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(
            warm_start_use_sp4=False,
            integrate_u_route=True,
            enable_route_incident_travel_lb=True,
            enable_route_pair_service_travel_lb=True,
            enable_route_slot_stack_count_lb=True,
            enable_route_finish_cmax_lb=True,
            enable_global_arrival_workload_lb=True,
        )
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        model = gp.Model("xyzu_route_lb_probe_test")
        model.Params.OutputFlag = 0
        payload = solver._build_model(model, prepared, cfg)
        model.update()
        constr_names = {constr.ConstrName for constr in model.getConstrs()}
        route_task_count = len(payload["route_tasks"])
        robot_count = len(payload["robot_ids"])
        slot_count = len(prepared["slots"])

        self.assertEqual(int(payload["diagnostics"].get("route_incident_travel_lb_count", 0)), robot_count)
        self.assertEqual(int(payload["diagnostics"].get("route_pair_service_travel_lb_count", 0)), route_task_count * robot_count)
        self.assertEqual(int(payload["diagnostics"].get("route_slot_stack_count_lb_count", 0)), slot_count)
        self.assertEqual(int(payload["diagnostics"].get("route_finish_cmax_lb_count", 0)), robot_count)
        self.assertEqual(int(payload["diagnostics"].get("global_arrival_workload_lb_count", 0)), 1)
        self.assertTrue(any(name.startswith("RouteIncidentTravelLB_") for name in constr_names))
        self.assertTrue(any(name.startswith("RoutePairServiceTravelLB_") for name in constr_names))
        self.assertTrue(any(name.startswith("SlotStackCapacityCoverLB_") for name in constr_names))
        self.assertTrue(any(name.startswith("RouteFinishCmaxLB_") for name in constr_names))
        self.assertIn("GlobalArrivalWorkloadLB", constr_names)

    def test_build_model_splits_hit_domain_from_support_totes(self):
        problem = CreateOFSProblem.generate_problem_by_scale("SMALL", seed=2)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(warm_start_use_sp4=False, integrate_u_route=True)
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        order_id = -1
        extra_tote_id = -1
        for current_order_id, support_totes in prepared["support_totes_by_order"].items():
            extra_totes = sorted(set(support_totes) - set(prepared["demand_hit_totes_by_order"].get(current_order_id, [])))
            if extra_totes:
                order_id = int(current_order_id)
                extra_tote_id = int(extra_totes[0])
                break
        self.assertGreaterEqual(order_id, 0)
        self.assertGreaterEqual(extra_tote_id, 0)

        model = gp.Model("xyzu_domain_split_test")
        model.Params.OutputFlag = 0
        payload = solver._build_model(model, prepared, cfg)
        model.update()
        for slot_id in prepared["slot_ids_by_order"][order_id]:
            self.assertIn((int(slot_id), int(extra_tote_id)), payload["carry"])
            self.assertIn((int(slot_id), int(extra_tote_id)), payload["noise"])
            self.assertNotIn((int(slot_id), int(extra_tote_id)), payload["hit"])
            self.assertNotIn((int(slot_id), int(extra_tote_id)), payload["flip_hit"])

    def test_build_model_keeps_larger_explicit_max_rank(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(warm_start_use_sp4=False, integrate_u_route=True, max_rank=99)
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        model = gp.Model("xyzu_explicit_rank_test")
        model.Params.OutputFlag = 0
        payload = solver._build_model(model, prepared, cfg)
        self.assertEqual(int(payload["max_rank"]), 99)
        self.assertEqual(int(payload["diagnostics"].get("effective_max_rank", 0)), 99)

    def test_apply_warm_start_rebuilds_continuous_starts(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(
            time_limit_sec=5,
            mip_gap=0.2,
            candidate_stack_topk=2,
            max_candidate_stacks_per_order=12,
            warm_start_sp4_time_limit_sec=3,
            integrate_u_route=True,
            route_arc_prune=False,
            enable_sp4_fallback=False,
        )
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        try:
            model = gp.Model("warm_start_apply_test")
        except gp.GurobiError as exc:
            self.skipTest(f"Gurobi unavailable for warm-start apply test: {exc}")
        model.Params.OutputFlag = 0
        payload = solver._build_model(model, prepared, cfg)
        diagnostics = solver._apply_warm_start(payload, prepared, warm)
        self.assertTrue(diagnostics.get("warm_start_continuous_time_start"))
        self.assertTrue(diagnostics.get("warm_start_route_rebuild_ok"))
        self.assertTrue(diagnostics.get("warm_start_slot_time_rebuild_ok"))
        self.assertTrue(diagnostics.get("warm_start_mip_start_ready"))
        self.assertTrue(diagnostics.get("warm_start_slot_lex_checked"))
        self.assertEqual(int(diagnostics.get("warm_start_slot_load_lex_violation_count", -1)), 0)
        self.assertEqual(int(diagnostics.get("warm_start_slot_station_lex_violation_count", -1)), 0)
        self.assertIn("warm_start_robot_id_swapped", diagnostics)
        self.assertIn("warm_start_robot_id_map", diagnostics)
        self.assertIn("warm_start_robot_path_duration", diagnostics)
        self.assertGreater(float(diagnostics.get("warm_start_model_cmax", 0.0) or 0.0), 0.0)
        slot_time_rows = list(diagnostics.get("warm_start_slot_times") or [])
        self.assertTrue(slot_time_rows)
        self.assertTrue(any(float(row.get("arrival", 0.0) or 0.0) > 0.0 for row in slot_time_rows))

    def test_warm_start_slot_lex_order_canonicalizes_load_and_station_rank(self):
        def _sku(sku_id):
            return SimpleNamespace(id=int(sku_id))

        def _task(tote_ids, arrival):
            return SimpleNamespace(
                target_tote_ids=list(tote_ids),
                hit_tote_ids=[],
                arrival_time_at_station=float(arrival),
            )

        rows = [
            SimpleNamespace(
                id=1,
                assigned_station_id=0,
                station_sequence_rank=0,
                sku_list=[_sku(101)],
                execution_tasks=[_task([10], 5.0)],
            ),
            SimpleNamespace(
                id=2,
                assigned_station_id=1,
                station_sequence_rank=9,
                sku_list=[_sku(101), _sku(102)],
                execution_tasks=[_task([10, 11], 1.0)],
            ),
            SimpleNamespace(
                id=3,
                assigned_station_id=1,
                station_sequence_rank=1,
                sku_list=[_sku(102), _sku(103)],
                execution_tasks=[_task([11, 12], 100.0)],
            ),
        ]
        profiles = GlobalXYZUSolver._canonical_warm_slot_profiles(
            order_id=1,
            warm_rows=rows,
            slot_ids=[20, 21, 22],
            units_by_order_sku={(1, 101): ["u1"], (1, 102): ["u2"], (1, 103): ["u3"]},
            tote_sku_qty={(10, 101): 1, (11, 102): 1, (12, 103): 1},
        )
        self.assertEqual([int(row["slot_id"]) for row in profiles], [20, 21, 22])
        self.assertEqual([int(row["start_load"]) for row in profiles], [2, 2, 1])
        self.assertEqual([int(row["subtask_id"]) for row in profiles], [3, 2, 1])

        active_rows = [
            (int(row["slot_id"]), int(row["station_id"]), int(row["rank"]))
            for row in profiles
        ]
        reranked_rows = GlobalXYZUSolver._lex_aware_station_rank_rows(
            active_slot_rows=active_rows,
            slot_ids_by_order={1: [20, 21, 22]},
            slot_start_load_by_slot={int(row["slot_id"]): int(row["start_load"]) for row in profiles},
            slot_arrival_lower={20: 100.0, 21: 1.0, 22: 5.0},
        )
        self.assertIn((20, 1, 0), reranked_rows)
        self.assertIn((21, 1, 1), reranked_rows)

        def _var(value):
            return SimpleNamespace(Start=float(value))

        a = {20: _var(1.0), 21: _var(1.0), 22: _var(1.0)}
        sku_use = {}
        for profile in profiles:
            for sku_id in profile["start_sku_ids"]:
                sku_use[(1, int(sku_id), int(profile["slot_id"]))] = _var(1.0)
        y = {(slot_id, station_id, rank): _var(1.0) for slot_id, station_id, rank in reranked_rows}
        diagnostics = GlobalXYZUSolver._validate_slot_lex_starts(
            a=a,
            sku_use=sku_use,
            y=y,
            slot_ids_by_order={1: [20, 21, 22]},
            unique_skus_by_order={1: [101, 102, 103]},
            station_ids=[0, 1],
            max_rank=3,
        )
        self.assertTrue(diagnostics.get("warm_start_slot_lex_checked"))
        self.assertEqual(int(diagnostics.get("warm_start_slot_load_lex_violation_count", -1)), 0)
        self.assertEqual(int(diagnostics.get("warm_start_slot_station_lex_violation_count", -1)), 0)

    def test_apply_warm_start_does_not_set_pair_activate_without_route_cover(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        solver = GlobalXYZUSolver()
        cfg = GlobalXYZUConfig(
            time_limit_sec=5,
            mip_gap=0.2,
            candidate_stack_topk=2,
            max_candidate_stacks_per_order=12,
            warm_start_sp4_time_limit_sec=3,
            integrate_u_route=True,
            warm_start_use_sp4=False,
            enable_sp4_fallback=False,
        )
        warm = solver._build_warm_start(problem, cfg)
        prepared = solver._prepare(problem, cfg, warm)
        model = gp.Model("warm_start_pair_guard_test")
        model.Params.OutputFlag = 0
        payload = solver._build_model(model, prepared, cfg)

        broken_tuple = None
        for order_id, slot_ids in prepared["slot_ids_by_order"].items():
            warm_rows = list(warm.subtask_by_order.get(int(order_id), []))
            warm_rows.sort(key=lambda row: int(getattr(row, "id", -1)))
            if not warm_rows or not slot_ids:
                continue
            slot_id = int(slot_ids[0])
            st = warm_rows[0]
            station_id = int(getattr(st, "assigned_station_id", -1))
            first_task = next(iter(getattr(st, "execution_tasks", []) or []), None)
            if first_task is None:
                continue
            stack_id = int(getattr(first_task, "target_stack_id", -1))
            candidate = (slot_id, stack_id, station_id)
            if candidate in payload["route_task_by_tuple"]:
                broken_tuple = candidate
                del payload["route_task_by_tuple"][candidate]
                break

        self.assertIsNotNone(broken_tuple)
        diagnostics = solver._apply_warm_start(payload, prepared, warm)
        model.update()
        self.assertIn("missing_route_task", str(diagnostics.get("warm_start_u_skipped_reason", "")))
        pair_var = payload["pair_activate"][broken_tuple]
        self.assertAlmostEqual(float(pair_var.Start), 0.0)

    def test_global_xyzu_solver_small_smoke(self):
        problem = CreateOFSProblem.generate_problem_by_scale("SMALL", seed=42)
        solver = GlobalXYZUSolver()
        result = solver.solve(
            problem,
            GlobalXYZUConfig(
                time_limit_sec=5,
                mip_gap=0.2,
                candidate_stack_topk=2,
                max_candidate_stacks_per_order=12,
                warm_start_sp4_time_limit_sec=3,
                u_route_use_mip=False,
            ),
        )
        self.assertIn(result.status, {"OPTIMAL", "TIME_LIMIT", "WARM_START_FALLBACK", "TIME_VERIFY_MISMATCH"})
        self.assertGreater(result.subtask_count, 0)
        self.assertGreater(result.task_count, 0)
        self.assertGreater(float(problem.global_makespan), 0.0)
        self.assertTrue(any(int(getattr(st, "assigned_station_id", -1)) >= 0 for st in (problem.subtask_list or [])))
        self.assertIn("u_integrated_route_used", result.diagnostics)
        self.assertIn("u_fallback_reason", result.diagnostics)
        self.assertIn("gurobi_solve_time_sec", result.diagnostics)
        self.assertIn("relay_tote_count", result.diagnostics)
        self.assertEqual(int(result.diagnostics.get("relay_tote_count", 0)), 0)
        self.assertEqual(result.diagnostics.get("warm_start_sp2_mode"), "heuristic")

    def test_global_xyzu_solver_small2_smoke(self):
        problem = CreateOFSProblem.generate_problem_by_scale("SMALL2", seed=42)
        solver = GlobalXYZUSolver()
        result = solver.solve(
            problem,
            GlobalXYZUConfig(
                time_limit_sec=5,
                mip_gap=0.2,
                candidate_stack_topk=2,
                max_candidate_stacks_per_order=12,
                warm_start_sp4_time_limit_sec=3,
                u_route_use_mip=False,
            ),
        )
        self.assertIn(result.status, {"OPTIMAL", "TIME_LIMIT", "WARM_START_FALLBACK", "TIME_VERIFY_MISMATCH"})
        self.assertGreater(result.subtask_count, 0)
        self.assertGreater(result.task_count, 0)
        self.assertGreater(float(problem.global_makespan), 0.0)
        self.assertTrue(result.station_schedule)
        self.assertIn("u_candidate_task_count", result.diagnostics)
        self.assertIn("u_arc_count", result.diagnostics)
        self.assertIn("gurobi_solve_time_sec", result.diagnostics)
        self.assertEqual(int(result.diagnostics.get("relay_tote_count", 0)), 0)

    def test_global_xyzu_solver_gurobi_s1_smoke(self):
        problem = CreateOFSProblem.generate_problem_by_scale("Gurobi-s1", seed=42)
        solver = GlobalXYZUSolver()
        result = solver.solve(
            problem,
            GlobalXYZUConfig(
                time_limit_sec=5,
                mip_gap=0.2,
                candidate_stack_topk=2,
                max_candidate_stacks_per_order=12,
                warm_start_sp4_time_limit_sec=3,
                warm_start_use_sp4=False,
                u_route_use_mip=False,
            ),
        )
        self.assertIn(result.status, {"OPTIMAL", "TIME_LIMIT", "WARM_START_FALLBACK", "TIME_VERIFY_MISMATCH"})
        self.assertGreater(result.subtask_count, 0)
        self.assertGreater(result.task_count, 0)
        self.assertGreater(float(problem.global_makespan), 0.0)
        self.assertIn("demand_hit_tote_count_by_order", result.diagnostics)
        self.assertIn("support_tote_count_by_order", result.diagnostics)
        self.assertIn("u_legal_arc_count_before_knn", result.diagnostics)
        self.assertIn("u_arc_count_after_knn", result.diagnostics)
        self.assertEqual(int(result.diagnostics.get("relay_tote_count", 0)), 0)

    def test_global_xyzu_one_bom_10sku_2station_2robot_case(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        self.assertEqual(problem.order_num, 1)
        self.assertEqual(len(problem.order_list), 1)
        self.assertEqual(len(set(problem.order_list[0].order_product_id_list)), 10)
        self.assertEqual(problem.order_list[0].order_skus_number, 10)
        self.assertEqual(problem.station_num, 2)
        self.assertEqual(problem.robot_num, 2)
        self.assertEqual(problem.scale_name, "TEST")

        result = GlobalXYZUSolver().solve(
            problem,
            GlobalXYZUConfig(
                time_limit_sec=5,
                mip_gap=0.2,
                candidate_stack_topk=2,
                max_candidate_stacks_per_order=12,
                warm_start_sp4_time_limit_sec=3,
                integrate_u_route=True,
                warm_start_use_sp4=False,
                enable_sp4_fallback=False,
            ),
        )
        self.assertIn(result.status, {"OPTIMAL", "TIME_LIMIT", "WARM_START_FALLBACK", "TIME_VERIFY_MISMATCH"})
        self.assertGreater(result.subtask_count, 0)
        self.assertGreater(result.task_count, 0)
        self.assertGreater(float(problem.global_makespan), 0.0)
        self.assertIn("u_integrated_route_used", result.diagnostics)
        self.assertIn("u_candidate_task_count", result.diagnostics)
        self.assertEqual(int(result.diagnostics.get("relay_tote_count", 0)), 0)
        slot_arrivals = []
        for st in getattr(problem, "subtask_list", []) or []:
            arrivals = [float(getattr(task, "arrival_time_at_station", 0.0) or 0.0) for task in getattr(st, "execution_tasks", []) or []]
            if arrivals:
                slot_arrivals.append(max(arrivals))
        self.assertTrue(slot_arrivals)
        self.assertLessEqual(max(slot_arrivals) - min(slot_arrivals), 60.0 + 1e-6)

    def test_global_xyzu_one_bom_window_can_be_disabled(self):
        problem = CreateOFSProblem.generate_problem_by_scale("test", seed=42)
        result = GlobalXYZUSolver().solve(
            problem,
            GlobalXYZUConfig(
                time_limit_sec=5,
                mip_gap=0.2,
                candidate_stack_topk=2,
                max_candidate_stacks_per_order=12,
                warm_start_sp4_time_limit_sec=3,
                integrate_u_route=True,
                warm_start_use_sp4=False,
                enable_sp4_fallback=False,
                bom_arrival_window_sec=0.0,
            ),
        )
        self.assertIn(result.status, {"OPTIMAL", "TIME_LIMIT", "WARM_START_FALLBACK", "TIME_VERIFY_MISMATCH"})
        self.assertGreater(result.subtask_count, 0)
        self.assertGreater(result.task_count, 0)


if __name__ == "__main__":
    unittest.main()
