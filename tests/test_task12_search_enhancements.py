import unittest
from types import SimpleNamespace

from Gurobi.resource_time_alns.fixgurobi_evaluator import FixGurobiEvaluator
from Gurobi.resource_time_alns.operators_y import y_plan_destroy_critical_load_rebalance
from Gurobi.resource_time_alns.route_edge_audit import (
    allowed_route_edges_from_global_payload,
    audit_fixed_route_edges,
)
from Gurobi.resource_time_alns.state import ResourceConfig, ResourceSubtask, WorkUnitInfo, ZTaskDescriptor
from Gurobi.sp4 import SP4_Robot_Router


class Task12SearchEnhancementTests(unittest.TestCase):
    def _config_for_y_rebalance(self) -> ResourceConfig:
        work_units = {
            "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
            "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 2),
            "2:20:0": WorkUnitInfo("2:20:0", 2, 20, 0, 3),
        }
        subtasks = {
            1: ResourceSubtask(
                1,
                1,
                ("1:10:0",),
                0,
                0,
                [ZTaskDescriptor(1, 7, "FLIP", (101,), (101,), (), None, station_service_time=1.0, sku_pick_count=1)],
                ("st_1",),
            ),
            2: ResourceSubtask(
                2,
                1,
                ("1:11:0",),
                0,
                2,
                [ZTaskDescriptor(2, 8, "FLIP", (102,), (102,), (), None, station_service_time=9.0, sku_pick_count=3)],
                ("st_2",),
            ),
            3: ResourceSubtask(
                3,
                2,
                ("2:20:0",),
                1,
                0,
                [ZTaskDescriptor(3, 9, "FLIP", (103,), (103,), (), None, station_service_time=1.0, sku_pick_count=1)],
                ("st_3",),
            ),
        }
        return ResourceConfig(work_units, subtasks, {1: 2, 2: 1}, 4, 4).rebuild_indices()

    def test_y_critical_load_rebalance_prioritizes_heavy_tail_subtask(self):
        config = self._config_for_y_rebalance()
        opt = SimpleNamespace(cfg=SimpleNamespace(resource_y_critical_load_degree_cap=2))

        plan = y_plan_destroy_critical_load_rebalance(opt, config, None, degree=1)

        self.assertTrue(plan["success"])
        self.assertEqual(plan["source_station_ids"], [0])
        self.assertEqual(plan["target_station_hint"], 1)
        self.assertEqual(sorted(plan["released_subtasks"].keys()), [2])
        self.assertEqual(config.subtasks[2].station_id, 0)

    def test_route_sequence_audit_accepts_exact_task_and_node_replay(self):
        route_tasks = {
            0: [
                {"task_id": 11, "subtask_id": 1, "order_id": 1, "stack_id": 7, "station_id": 0, "arrival_stack": 1.0, "arrival_station": 5.0},
                {"task_id": 12, "subtask_id": 2, "order_id": 1, "stack_id": 8, "station_id": 0, "arrival_stack": 6.0, "arrival_station": 9.0},
            ]
        }
        route_nodes = {
            0: [
                {"kind": "pickup", "task_id": 11, "subtask_id": 1, "order_id": 1, "stack_id": 7, "station_id": 0, "time": 1.0},
                {"kind": "delivery", "task_id": 11, "subtask_id": 1, "order_id": 1, "stack_id": 7, "station_id": 0, "time": 5.0},
                {"kind": "pickup", "task_id": 12, "subtask_id": 2, "order_id": 1, "stack_id": 8, "station_id": 0, "time": 6.0},
                {"kind": "delivery", "task_id": 12, "subtask_id": 2, "order_id": 1, "stack_id": 8, "station_id": 0, "time": 9.0},
            ]
        }

        audit = FixGurobiEvaluator._route_sequence_audit(route_tasks, route_nodes)

        self.assertTrue(audit["ok"])
        self.assertEqual(audit["checked_task_count"], 2)

    def test_route_sequence_audit_rejects_node_order_drift(self):
        route_tasks = {
            0: [
                {"task_id": 11, "subtask_id": 1, "order_id": 1, "stack_id": 7, "station_id": 0, "arrival_stack": 1.0, "arrival_station": 5.0},
                {"task_id": 12, "subtask_id": 2, "order_id": 1, "stack_id": 8, "station_id": 0, "arrival_stack": 6.0, "arrival_station": 9.0},
            ]
        }
        route_nodes = {
            0: [
                {"kind": "pickup", "task_id": 11, "subtask_id": 1, "order_id": 1, "stack_id": 7, "station_id": 0, "time": 1.0},
                {"kind": "pickup", "task_id": 12, "subtask_id": 2, "order_id": 1, "stack_id": 8, "station_id": 0, "time": 6.0},
                {"kind": "delivery", "task_id": 11, "subtask_id": 1, "order_id": 1, "stack_id": 7, "station_id": 0, "time": 5.0},
                {"kind": "delivery", "task_id": 12, "subtask_id": 2, "order_id": 1, "stack_id": 8, "station_id": 0, "time": 9.0},
            ]
        }

        audit = FixGurobiEvaluator._route_sequence_audit(route_tasks, route_nodes)

        self.assertFalse(audit["ok"])
        self.assertIn("node_sequence_order_mismatch", audit["reason"])

    def test_fixed_payload_merges_structure_seed_stacks_by_order(self):
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 7, "FLIP", (101,), (101,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 1},
            next_subtask_id=2,
            next_task_id=2,
            metadata={"structure_seed_stack_ids_by_order": {1: [8, 9]}},
        ).rebuild_indices()
        evaluator = FixGurobiEvaluator(SimpleNamespace(cfg=SimpleNamespace(fixgurobi_fix_used_stack_ids=False)))

        payload = evaluator._fixed_payload(config, scope="XYZ")

        self.assertEqual(payload["forced_candidate_stacks_by_order"][1], [7, 8, 9])

    def test_route_edge_audit_rejects_full_global_missing_edge(self):
        payload = {
            "slots": [SimpleNamespace(slot_id=10, order_id=1, local_index=0)],
            "route_tasks": {
                0: SimpleNamespace(task_key=0, slot_id=10, stack_id=100, station_id=0),
            },
            "route_nodes": {
                0: SimpleNamespace(kind="start", task_key=-1, node_id=0),
                1: SimpleNamespace(kind="pickup", task_key=0, node_id=1),
                2: SimpleNamespace(kind="delivery", task_key=0, node_id=2),
                3: SimpleNamespace(kind="end", task_key=-1, node_id=3),
            },
            "route_tau": {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0},
        }
        allowed = allowed_route_edges_from_global_payload(payload)
        fixed_nodes = {
            0: [
                {"kind": "pickup", "order_id": 1, "local_slot_index": 0, "stack_id": 100, "station_id": 0},
                {"kind": "delivery", "order_id": 1, "local_slot_index": 0, "stack_id": 100, "station_id": 0},
            ]
        }
        self.assertTrue(audit_fixed_route_edges(allowed, route_node_sequence=fixed_nodes)["ok"])

        drift_nodes = {
            0: [
                {"kind": "delivery", "order_id": 1, "local_slot_index": 0, "stack_id": 100, "station_id": 0},
                {"kind": "pickup", "order_id": 1, "local_slot_index": 0, "stack_id": 100, "station_id": 0},
            ]
        }
        audit = audit_fixed_route_edges(allowed, route_node_sequence=drift_nodes)
        self.assertFalse(audit["ok"])
        self.assertGreaterEqual(audit["missing_edge_count"], 1)

    def test_sp4_global_route_graph_sync_uses_order_local_slot_key(self):
        router = SP4_Robot_Router.__new__(SP4_Robot_Router)
        task = SimpleNamespace(sub_task_id=42, target_stack_id=100, target_station_id=0)
        nodes_info = [
            (SimpleNamespace(x=0, y=0), None, "depot"),
            (SimpleNamespace(x=1, y=0), task, "pickup"),
            (SimpleNamespace(x=2, y=0), task, "delivery"),
        ]
        allowed = {
            (("start",), ("node", "pickup", 7, 0, 100, 0)),
            (("node", "pickup", 7, 0, 100, 0), ("node", "delivery", 7, 0, 100, 0)),
            (("node", "delivery", 7, 0, 100, 0), ("end",)),
        }

        kept, diag = router._build_lkh_arcs_from_global_route_graph(
            nodes_info,
            allowed,
            subtask_slot_lookup={42: (7, 0)},
        )

        self.assertEqual(kept, {(0, 1), (1, 2), (2, 0)})
        self.assertEqual(diag["source"], "global_route_tau")


if __name__ == "__main__":
    unittest.main()
