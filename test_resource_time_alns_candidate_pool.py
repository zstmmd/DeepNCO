import os
import unittest
from types import SimpleNamespace

from Gurobi.resource_time_alns.engine import ResourceTimeALNSEngine
from Gurobi.resource_time_alns.reporting import write_resource_time_candidates_csv
from Gurobi.resource_time_alns.state import ResourceConfig, ResourceSubtask, UpperEvalResult, WorkUnitInfo, ZTaskDescriptor


class ResourceTimeALNSCandidatePoolTests(unittest.TestCase):
    def _config(self, subtask_id: int, station_id: int, station_rank: int, stack_id: int) -> ResourceConfig:
        work_unit_id = f"1:{subtask_id}:0"
        return ResourceConfig(
            work_units={
                work_unit_id: WorkUnitInfo(work_unit_id, 1, 10 + subtask_id, 0, subtask_id),
            },
            subtasks={
                subtask_id: ResourceSubtask(
                    subtask_id,
                    1,
                    (work_unit_id,),
                    station_id,
                    station_rank,
                    [ZTaskDescriptor(subtask_id, stack_id, "FLIP", (stack_id,), (stack_id,), (), None)],
                    (f"st_{subtask_id}",),
                )
            },
            capacity_limits={1: 3},
            next_subtask_id=subtask_id + 1,
            next_task_id=subtask_id + 1,
        ).rebuild_indices()

    def _eval_result(self, f_raw: float, f_cal: float) -> UpperEvalResult:
        return UpperEvalResult(
            Sx=1.0,
            Sy=2.0,
            Sz=3.0,
            F_raw=float(f_raw),
            F_cal=float(f_cal),
            duplicate_tote_count=0,
            duplicate_tote_penalty=0.0,
        )

    def test_generate_x_candidate_pool_respects_uniqueness_and_attempt_cap(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(resource_candidate_pool_max_attempts=5)
        engine.current_config = self._config(99, 0, 0, 99)
        engine.action_signature_history = {"X": [], "Y": [], "Z": []}
        engine.action_signature_seen = {"X": set(), "Y": set(), "Z": set()}
        engine._action_signature_known = lambda _layer, _sig: False
        engine._remember_action_signature = lambda _layer, _sig: None

        config_a = self._config(1, 0, 0, 10)
        config_b = self._config(2, 1, 0, 20)
        config_c = self._config(3, 1, 1, 30)
        rows = iter(
            [
                {
                    "destroy_operator": "x_destroy_spatial_outliers",
                    "repair_operator": "x_repair_affinity_pack",
                    "candidate_signature_tuple": config_a.validation_signature(),
                    "duplicate_tote_count": 0,
                },
                {
                    "destroy_operator": "x_destroy_spatial_outliers",
                    "repair_operator": "x_repair_affinity_pack",
                    "candidate_signature_tuple": config_a.validation_signature(),
                    "duplicate_tote_count": 0,
                },
                {
                    "destroy_operator": "x_destroy_low_consolidation",
                    "repair_operator": "x_repair_route_span_min",
                    "candidate_signature_tuple": config_b.validation_signature(),
                    "duplicate_tote_count": 0,
                },
                None,
                {
                    "destroy_operator": "x_destroy_over_capacity_release",
                    "repair_operator": "x_repair_regret2_new_group",
                    "candidate_signature_tuple": config_c.validation_signature(),
                    "duplicate_tote_count": 0,
                },
            ]
        )
        pairs = iter(
            [
                ("x_destroy_spatial_outliers", "x_repair_affinity_pack"),
                ("x_destroy_spatial_outliers", "x_repair_affinity_pack"),
                ("x_destroy_low_consolidation", "x_repair_route_span_min"),
                ("x_destroy_group_boundary_release", "x_repair_template_preserve"),
                ("x_destroy_over_capacity_release", "x_repair_regret2_new_group"),
            ]
        )

        engine._sample_operator_pair = lambda _layer: next(pairs)
        engine._build_x_exact_candidate = lambda *_args: next(rows)
        engine._select_best_candidate = ResourceTimeALNSEngine._select_best_candidate.__get__(engine, ResourceTimeALNSEngine)

        pool = ResourceTimeALNSEngine._generate_x_candidate_pool(engine, 1, 2, 3)
        self.assertEqual(pool["attempt_count"], 5)
        self.assertEqual(pool["generated_count"], 4)
        self.assertEqual(pool["unique_count"], 3)
        self.assertEqual(pool["exact_count"], 3)
        self.assertEqual(len(pool["rows"]), 3)
        self.assertEqual(len({row["candidate_signature_tuple"] for row in pool["rows"]}), 3)

    def test_select_best_candidate_uses_configured_tie_breaks(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        rows = [
            {
                "destroy_operator": "z_destroy_noise_window",
                "repair_operator": "z_repair_same_stack_window",
                "candidate_signature": "sig_c",
                "fallback_used": True,
                "F_raw": 10.0,
                "F_cal": 10.0,
            },
            {
                "destroy_operator": "z_destroy_noise_window",
                "repair_operator": "z_repair_bounded_detour_window",
                "candidate_signature": "sig_b",
                "fallback_used": False,
                "F_raw": 9.5,
                "F_cal": 10.0,
            },
            {
                "destroy_operator": "z_destroy_detour_window",
                "repair_operator": "z_repair_same_stack_window",
                "candidate_signature": "sig_a",
                "fallback_used": False,
                "F_raw": 9.5,
                "F_cal": 9.0,
            },
        ]
        best = ResourceTimeALNSEngine._select_best_candidate(engine, rows)
        self.assertEqual(best["candidate_signature"], "sig_a")
        self.assertFalse(best["selected_for_sa"])
        self.assertEqual(best["candidate_rank"], 1)
        selected_count = sum(1 for row in rows if row.get("selected_for_sa"))
        self.assertEqual(selected_count, 0)

    def test_candidate_csv_writer_outputs_expected_columns(self):
        tmp = os.path.join(os.getcwd(), "test_output_tmp_candidate_pool")
        os.makedirs(tmp, exist_ok=True)
        try:
            path = write_resource_time_candidates_csv(
                tmp,
                [
                    {
                        "iter": 1,
                        "layer": "Z",
                        "candidate_stage": "rough",
                        "candidate_rank": 1,
                        "destroy_operator": "z_destroy_noise_window",
                        "repair_operator": "z_repair_same_stack_window",
                        "fallback_used": False,
                        "projection_mode": "",
                        "projection_repaired_subtask_count": 0,
                        "F_raw": 10.0,
                        "F_cal": 9.0,
                        "duplicate_tote_count": 0,
                        "duplicate_tote_penalty": 0.0,
                        "candidate_signature": "sig_a",
                        "selected_for_sa": True,
                    }
                ],
            )
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("candidate_rank", text)
            self.assertIn("candidate_stage", text)
            self.assertIn("selected_for_sa", text)
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)

    def test_clone_for_layer_only_clones_touched_subtasks(self):
        config = ResourceConfig(
            work_units={
                "1:a:0": WorkUnitInfo("1:a:0", 1, 11, 0, 1),
                "1:b:0": WorkUnitInfo("1:b:0", 1, 12, 0, 2),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:a:0",), 0, 0, [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)]),
                2: ResourceSubtask(2, 1, ("1:b:0",), 1, 0, [ZTaskDescriptor(2, 200, "FLIP", (200,), (200,), (), None)]),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=3,
        )
        cloned_y = config.clone_for_layer("Y", [1])
        self.assertIs(cloned_y.subtasks[2], config.subtasks[2])
        self.assertIsNot(cloned_y.subtasks[1], config.subtasks[1])
        self.assertIs(cloned_y.subtasks[1].z_tasks[0], config.subtasks[1].z_tasks[0])

        cloned_z = config.clone_for_layer("Z", [2])
        self.assertIs(cloned_z.subtasks[1], config.subtasks[1])
        self.assertIsNot(cloned_z.subtasks[2], config.subtasks[2])
        self.assertIsNot(cloned_z.subtasks[2].z_tasks[0], config.subtasks[2].z_tasks[0])


if __name__ == "__main__":
    unittest.main()
