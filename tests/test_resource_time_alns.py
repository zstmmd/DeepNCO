import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Gurobi.resource_time_alns.operators_x import x_finalize_insert_or_new_group
from Gurobi.resource_time_alns.operators_z import z_destroy_mode_window
from Gurobi.resource_time_alns.operators_z import _repair_window
from Gurobi.resource_time_alns.operators_z import _sort_plan_within_capacity
from Gurobi.resource_time_alns.operators_z import validate_z_assignment
from Gurobi.resource_time_alns.operators_z import apply_joint_colocated_sort_postprocess
from Gurobi.resource_time_alns.state import ResourceConfig, ResourceSubtask, WorkUnitInfo, ZTaskDescriptor
from Gurobi.resource_time_alns.surrogate import ResourceSurrogateScorer


class ResourceTimeALNSTests(unittest.TestCase):
    def _joint_sort_opt(self):
        tote_101 = SimpleNamespace(id=101, sku_quantity_map={10: 1})
        tote_102 = SimpleNamespace(id=102, sku_quantity_map={11: 1})
        tote_103 = SimpleNamespace(id=103, sku_quantity_map={99: 1})
        stack = SimpleNamespace(id=7, totes=[tote_101, tote_102, tote_103])
        problem = SimpleNamespace(
            point_to_stack={7: stack},
            id_to_tote={101: tote_101, 102: tote_102, 103: tote_103},
            order_list=[SimpleNamespace(order_id=1)],
            skus_list=[SimpleNamespace(id=10), SimpleNamespace(id=11)],
        )
        return SimpleNamespace(
            cfg=SimpleNamespace(resource_z_candidate_stack_topk=5),
            problem=problem,
            _z_build_plan_from_hits=lambda *_args, **_kwargs: {
                "valid": True,
                "target_stack_id": 7,
                "operation_mode": "SORT",
                "target_tote_ids": [101, 102],
                "hit_tote_ids": [101, 102],
                "noise_tote_ids": [],
                "sort_layer_range": (0, 1),
                "station_service_time": 2.0,
                "robot_service_time": 1.0,
            },
            _z_best_insertion_detour=lambda *_args, **_kwargs: 0.0,
        )

    def _base_config(self) -> ResourceConfig:
        work_units = {
            "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
            "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 2),
        }
        subtasks = {
            1: ResourceSubtask(1, 1, ("1:10:0", "1:11:0"), 0, 0, [], ("st_1",)),
            2: ResourceSubtask(2, 1, ("1:12:0",), 1, 0, [], ("st_2",)),
        }
        return ResourceConfig(
            work_units=work_units,
            subtasks=subtasks,
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=10,
        ).rebuild_indices()

    def test_x_finalize_can_create_new_group_when_all_existing_groups_full(self):
        config = self._base_config()
        new_id = x_finalize_insert_or_new_group(
            config=config,
            order_id=1,
            work_unit_id="1:12:0",
            scorer=lambda *_: 0.0,
            origin_group_ids=("st_1",),
            prefer_new_group=False,
        )
        self.assertEqual(new_id, 3)
        self.assertIn(3, config.subtasks)
        self.assertEqual(config.subtasks[3].work_unit_ids, ("1:12:0",))

    def test_z_destroy_mode_window_releases_a_window_not_a_single_point(self):
        descriptors = [
            ZTaskDescriptor(1, 10, "SORT", (1, 2), (1,), (2,), (0, 1)),
            ZTaskDescriptor(2, 11, "SORT", (3, 4), (3,), (4,), (0, 1)),
            ZTaskDescriptor(3, 12, "FLIP", (5,), (5,), (), None),
            ZTaskDescriptor(4, 13, "FLIP", (6,), (6,), (), None),
        ]
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, descriptors, ("st_1",))},
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=5,
        ).rebuild_indices()
        opt = SimpleNamespace(cfg=SimpleNamespace(resource_z_window_size=3))
        ctx = z_destroy_mode_window(opt, config, None, 1)
        self.assertTrue(ctx["success"])
        self.assertGreaterEqual(len(ctx["removed_window"]), 2)

    def test_residual_decay_monotonic_and_zero_after_period(self):
        opt = SimpleNamespace(
            cfg=SimpleNamespace(
                seed=42,
                resource_real_eval_period=6,
                resource_residual_half_life=3.0,
                resource_residual_uncertainty_cap=80.0,
                resource_surrogate_trust_radius=0.35,
            ),
            best=SimpleNamespace(z=100.0),
        )
        scorer = ResourceSurrogateScorer(opt)
        values = [scorer._residual_decay(step) for step in range(0, 7)]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertEqual(values[6], 0.0)

    def test_joint_sort_repair_rejects_multi_subtask_windows(self):
        config = self._base_config()
        ctx = {
            "success": True,
            "windows": [
                {"subtask_id": 1, "preserved_before": [], "preserved_after": [], "removed_window": [], "seed_stack_ids": []},
                {"subtask_id": 2, "preserved_before": [], "preserved_after": [], "removed_window": [], "seed_stack_ids": []},
            ],
        }
        payload = _repair_window(
            opt=SimpleNamespace(),
            config=config,
            ctx=ctx,
            strategy="z_repair_joint_sort_colocated_flip",
            allow_fallback=False,
            rng=None,
        )
        self.assertFalse(payload["success"])
        self.assertEqual(payload.get("reason"), "joint_sort_requires_single_subtask_window")

    def test_sort_plan_capacity_guard_rejects_overload(self):
        overloaded_plan = {"operation_mode": "SORT", "target_tote_ids": list(range(100, 109))}
        self.assertFalse(_sort_plan_within_capacity(overloaded_plan))
        flip_plan = {"operation_mode": "FLIP", "target_tote_ids": list(range(100, 109))}
        self.assertTrue(_sort_plan_within_capacity(flip_plan))

    def test_joint_sort_repair_failure_rolls_back_original_assignment(self):
        config = self._base_config()
        original = [ZTaskDescriptor(1, 10, "FLIP", (1,), (1,), (), None)]
        config.subtasks[1].z_tasks = list(original)
        ctx = {
            "success": True,
            "windows": [
                {
                    "subtask_id": 1,
                    "preserved_before": [],
                    "preserved_after": [],
                    "removed_window": list(original),
                    "seed_stack_ids": [10],
                }
            ],
        }
        with patch("Gurobi.resource_time_alns.operators_z._rebuild_window", return_value=(False, [], {"reason": "invalid_assignment"})):
            payload = _repair_window(
                opt=SimpleNamespace(),
                config=config,
                ctx=ctx,
                strategy="z_repair_joint_sort_colocated_flip",
                allow_fallback=False,
                rng=None,
            )
        self.assertFalse(payload["success"])
        self.assertEqual(payload.get("reason"), "invalid_assignment")
        self.assertEqual(config.subtasks[1].z_tasks[0].signature(), original[0].signature())

    def test_joint_colocated_sort_postprocess_merges_same_stack_flip_group(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0", "1:11:0"),
                    0,
                    0,
                    [
                        ZTaskDescriptor(1, 7, "FLIP", (101,), (101,), (), None),
                        ZTaskDescriptor(2, 7, "FLIP", (102,), (102,), (), None),
                    ],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=3,
        ).rebuild_indices()
        opt = self._joint_sort_opt()
        tote_pos = {101: 0, 102: 1}
        opt._z_build_plan_from_hits = lambda _temp_subtask, _dummy_task, _stack_id, hit_ids, _mode, _ignore: {
            "valid": True,
            "target_stack_id": 7,
            "operation_mode": "SORT",
            "target_tote_ids": list(hit_ids),
            "hit_tote_ids": list(hit_ids),
            "noise_tote_ids": [],
            "sort_layer_range": (min(tote_pos[int(tid)] for tid in hit_ids), max(tote_pos[int(tid)] for tid in hit_ids)),
            "station_service_time": 2.0,
            "robot_service_time": 1.0,
        }
        with patch("Gurobi.resource_time_alns.operators_z._build_temp_subtask", return_value=SimpleNamespace(add_execution_detail=lambda *args, **kwargs: None)):
            updated, stats = apply_joint_colocated_sort_postprocess(opt, config, max_groups=1)
        self.assertEqual(len(updated.subtasks[1].z_tasks), 1)
        merged = updated.subtasks[1].z_tasks[0]
        self.assertEqual(merged.mode, "SORT")
        self.assertEqual(merged.sort_layer_range, (0, 1))
        self.assertTrue(updated.metadata.get("joint_colocated_sort_postprocess"))
        self.assertEqual(stats["applied"], 1.0)

    def test_joint_colocated_sort_postprocess_rejects_capacity_overflow(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
                "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 1),
                "1:13:0": WorkUnitInfo("1:13:0", 1, 13, 0, 1),
                "1:14:0": WorkUnitInfo("1:14:0", 1, 14, 0, 1),
                "1:15:0": WorkUnitInfo("1:15:0", 1, 15, 0, 1),
                "1:16:0": WorkUnitInfo("1:16:0", 1, 16, 0, 1),
                "1:17:0": WorkUnitInfo("1:17:0", 1, 17, 0, 1),
                "1:18:0": WorkUnitInfo("1:18:0", 1, 18, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0", "1:11:0", "1:12:0", "1:13:0", "1:14:0", "1:15:0", "1:16:0", "1:17:0", "1:18:0"),
                    0,
                    0,
                    [
                        ZTaskDescriptor(1, 7, "FLIP", (101, 102, 103, 104, 105), (101, 102, 103, 104, 105), (), None),
                        ZTaskDescriptor(2, 7, "FLIP", (106, 107, 108, 109), (106, 107, 108, 109), (), None),
                    ],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=3,
        ).rebuild_indices()
        opt = self._joint_sort_opt()
        opt._z_build_plan_from_hits = lambda *_args, **_kwargs: {
            "valid": True,
            "target_stack_id": 7,
            "operation_mode": "SORT",
            "target_tote_ids": list(range(101, 110)),
            "hit_tote_ids": list(range(101, 110)),
            "noise_tote_ids": [],
            "sort_layer_range": (0, 8),
            "station_service_time": 2.0,
            "robot_service_time": 1.0,
        }
        with patch("Gurobi.resource_time_alns.operators_z._build_temp_subtask", return_value=SimpleNamespace(add_execution_detail=lambda *args, **kwargs: None)):
            updated, stats = apply_joint_colocated_sort_postprocess(opt, config, max_groups=1)
        self.assertEqual(len(updated.subtasks[1].z_tasks), 2)
        self.assertEqual(stats["applied"], 0.0)
        self.assertGreaterEqual(stats["rejected_capacity"] + stats["rejected_interval_illegal"], 1.0)

    def test_joint_colocated_sort_postprocess_rejects_noise(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0", "1:11:0"),
                    0,
                    0,
                    [
                        ZTaskDescriptor(1, 7, "FLIP", (101,), (101,), (), None),
                        ZTaskDescriptor(2, 7, "FLIP", (102,), (102,), (), None),
                    ],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=3,
        ).rebuild_indices()
        opt = self._joint_sort_opt()
        opt._z_build_plan_from_hits = lambda *_args, **_kwargs: {
            "valid": True,
            "target_stack_id": 7,
            "operation_mode": "SORT",
            "target_tote_ids": [101, 102, 103],
            "hit_tote_ids": [101, 102],
            "noise_tote_ids": [103],
            "sort_layer_range": (0, 2),
            "station_service_time": 2.0,
            "robot_service_time": 1.0,
        }
        with patch("Gurobi.resource_time_alns.operators_z._build_temp_subtask", return_value=SimpleNamespace(add_execution_detail=lambda *args, **kwargs: None)):
            updated, stats = apply_joint_colocated_sort_postprocess(opt, config, max_groups=1)
        self.assertEqual(len(updated.subtasks[1].z_tasks), 1)
        self.assertEqual(updated.subtasks[1].z_tasks[0].target_tote_ids, (101, 102))
        self.assertEqual(updated.subtasks[1].z_tasks[0].noise_tote_ids, ())
        self.assertEqual(stats["applied"], 1.0)

    def test_validate_z_assignment_accepts_exact_style_sort_targets(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0", "1:11:0"),
                    0,
                    0,
                    [],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=3,
        ).rebuild_indices()
        tote_101 = SimpleNamespace(id=101, sku_quantity_map={10: 1})
        tote_102 = SimpleNamespace(id=102, sku_quantity_map={11: 1})
        tote_103 = SimpleNamespace(id=103, sku_quantity_map={99: 1})
        stack = SimpleNamespace(id=7, totes=[tote_103, tote_101, tote_102])
        opt = SimpleNamespace(
            problem=SimpleNamespace(
                point_to_stack={7: stack},
                id_to_tote={101: tote_101, 102: tote_102, 103: tote_103},
            )
        )
        descriptor = ZTaskDescriptor(
            1,
            7,
            "SORT",
            (101, 102),
            (101, 102),
            (),
            (0, 2),
        )
        self.assertTrue(validate_z_assignment(opt, config, config.subtasks[1], [descriptor]))

    def test_joint_colocated_sort_postprocess_can_merge_across_subtasks(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 2),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [ZTaskDescriptor(1, 7, "FLIP", (101,), (101,), (), None, station_service_time=6.0, robot_service_time=6.0)], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), 0, 1, [ZTaskDescriptor(2, 7, "FLIP", (102,), (102,), (), None, station_service_time=6.0, robot_service_time=6.0)], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=3,
        ).rebuild_indices()
        opt = self._joint_sort_opt()
        tote_pos = {101: 0, 102: 1}
        opt._z_build_plan_from_hits = lambda _temp_subtask, _dummy_task, _stack_id, hit_ids, _mode, _ignore: {
            "valid": True,
            "target_stack_id": 7,
            "operation_mode": "SORT",
            "target_tote_ids": list(hit_ids),
            "hit_tote_ids": list(hit_ids),
            "noise_tote_ids": [],
            "sort_layer_range": (min(tote_pos[int(tid)] for tid in hit_ids), max(tote_pos[int(tid)] for tid in hit_ids)),
            "station_service_time": 2.0,
            "robot_service_time": 1.0,
        }
        with patch("Gurobi.resource_time_alns.operators_z._build_temp_subtask", return_value=SimpleNamespace(add_execution_detail=lambda *args, **kwargs: None)):
            updated, stats = apply_joint_colocated_sort_postprocess(opt, config, max_groups=1)
        self.assertEqual(updated.subtasks[1].z_tasks[0].mode, "SORT")
        self.assertEqual(updated.subtasks[2].z_tasks[0].mode, "SORT")
        self.assertEqual(updated.subtasks[1].z_tasks[0].target_tote_ids, (101,))
        self.assertEqual(updated.subtasks[2].z_tasks[0].target_tote_ids, (102,))
        self.assertEqual(stats["candidate_groups"], 1.0)
        self.assertEqual(stats["applied"], 1.0)

    def test_joint_colocated_sort_postprocess_can_reassign_hits_within_stack(self):
        tote_101 = SimpleNamespace(id=101, sku_quantity_map={10: 1})
        tote_102 = SimpleNamespace(id=102, sku_quantity_map={11: 1})
        tote_103 = SimpleNamespace(id=103, sku_quantity_map={10: 1})
        stack = SimpleNamespace(id=7, totes=[tote_103, tote_101, tote_102])
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 2),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [ZTaskDescriptor(1, 7, "FLIP", (101,), (101,), (), None, robot_service_time=6.0)], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), 0, 1, [ZTaskDescriptor(2, 7, "FLIP", (102,), (102,), (), None, robot_service_time=6.0)], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=3,
        ).rebuild_indices()
        opt = SimpleNamespace(
            cfg=SimpleNamespace(resource_z_candidate_stack_topk=5, resource_joint_colocated_sort_candidate_limit=12),
            problem=SimpleNamespace(
                point_to_stack={7: stack},
                id_to_tote={101: tote_101, 102: tote_102, 103: tote_103},
                order_list=[SimpleNamespace(order_id=1)],
                skus_list=[SimpleNamespace(id=10), SimpleNamespace(id=11)],
            ),
            _z_best_insertion_detour=lambda *_args, **_kwargs: 0.0,
        )
        tote_pos = {103: 0, 101: 1, 102: 2}
        service_by_hits = {(101,): 5.0, (102,): 5.0, (103,): 1.0}
        opt._z_build_plan_from_hits = lambda _temp_subtask, _dummy_task, _stack_id, hit_ids, _mode, _ignore: {
            "valid": True,
            "target_stack_id": 7,
            "operation_mode": "SORT",
            "target_tote_ids": list(hit_ids),
            "hit_tote_ids": list(hit_ids),
            "noise_tote_ids": [],
            "sort_layer_range": (min(tote_pos[int(tid)] for tid in hit_ids), max(tote_pos[int(tid)] for tid in hit_ids)),
            "station_service_time": 0.0,
            "robot_service_time": float(service_by_hits.get(tuple(hit_ids), 2.0)),
        }
        with patch("Gurobi.resource_time_alns.operators_z._build_temp_subtask", return_value=SimpleNamespace(add_execution_detail=lambda *args, **kwargs: None)):
            updated, stats = apply_joint_colocated_sort_postprocess(opt, config, max_groups=1)
        self.assertEqual(updated.subtasks[1].z_tasks[0].target_tote_ids, (103,))
        self.assertEqual(updated.subtasks[2].z_tasks[0].target_tote_ids, (102,))
        self.assertEqual(stats["applied"], 1.0)


if __name__ == "__main__":
    unittest.main()
