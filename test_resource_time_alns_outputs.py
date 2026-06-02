import os
import unittest
from types import SimpleNamespace

import experiments.run_small_small2_200_resource_time as batch_script
from Gurobi.tra import TRAOptimizer, TRARunConfig
from Gurobi.resource_time_alns.reporting import (
    write_resource_time_best_runtime_txt,
    write_resource_time_iters_csv,
)


class ResourceTimeALNSOutputTests(unittest.TestCase):
    def test_reporting_writes_csv_and_runtime_txt(self):
        rows = [{
            "iter": 1,
            "focus": "X",
            "selected_resource_layer": "X",
            "destroy_operator": "x_destroy_spatial_outliers",
            "repair_operator": "x_repair_affinity_pack",
            "fallback_repair_used": False,
            "projection_mode": "",
            "projection_repaired_subtask_count": 0,
            "local_obj": 10.0,
            "Sx": 1.0,
            "Sy": 2.0,
            "Sz": 3.0,
            "F_raw": 10.0,
            "F_cal": 9.5,
            "residual_hat": -0.5,
            "residual_decay_alpha": 0.5,
            "residual_conf_alpha": 1.0,
            "local_accept": True,
            "global_eval_triggered": True,
            "validation_trigger": "periodic",
            "validated_makespan": 12.0,
            "catastrophic_rollback": False,
            "z": 12.0,
            "best_z": 12.0,
            "improved": True,
            "sa_temperature": 5.0,
            "sa_accept_prob": 1.0,
            "iter_runtime_sec": 0.25,
            "global_eval_time_sec": 0.10,
            "lkh_call_count": 1,
        }]
        opt = SimpleNamespace(
            cfg=SimpleNamespace(search_scheme="resource_time_alns", scale="SMALL", seed=42),
            best=SimpleNamespace(iter_id=1, z=12.0),
        )
        run_stats = {
            "best_validated_makespan": 12.0,
            "run_total_time_sec": 3.5,
            "global_eval_count": 1,
            "lkh_call_count": 1,
            "lkh_budget_consumed_by_rollback": 0,
            "layer_runtime_sec_by_name": {"X": 1.0, "Y": 0.0, "Z": 0.0, "U": 0.1},
        }
        tmp = os.path.join(os.getcwd(), "test_output_tmp_resource_time")
        os.makedirs(tmp, exist_ok=True)
        try:
            csv_path = write_resource_time_iters_csv(tmp, rows)
            txt_path = write_resource_time_best_runtime_txt(tmp, opt, run_stats)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(txt_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("destroy_operator", text)
            self.assertIn("best_z", text)
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("best_z=12.000000", text)
            self.assertIn("run_total_time_sec=3.500000", text)
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)

    def test_best_solution_audit_flags_missing_sku_and_invalid_z(self):
        task1 = SimpleNamespace(
            task_id=1,
            sub_task_id=1,
            target_station_id=-1,
            station_sequence_rank=-1,
            target_tote_ids=[1],
            hit_tote_ids=[2],
            noise_tote_ids=[2],
            robot_id=-1,
        )
        task2 = SimpleNamespace(
            task_id=2,
            sub_task_id=1,
            target_station_id=0,
            station_sequence_rank=0,
            target_tote_ids=[1],
            hit_tote_ids=[1],
            noise_tote_ids=[],
            robot_id=-1,
        )
        subtask = SimpleNamespace(
            id=1,
            assigned_station_id=-1,
            station_sequence_rank=-1,
            execution_tasks=[task1, task2],
        )
        fake = SimpleNamespace(
            problem=SimpleNamespace(subtask_list=[subtask], global_makespan=12.0),
            best=SimpleNamespace(z=10.0),
        )
        fake._compute_solution_coverage = lambda: {
            "coverage_ok": False,
            "unmet_sku_total": 2,
            "unmet_subtask_count": 1,
            "subtasks": [{"subtask_id": 1, "unmet_sku_units": 2, "unmet_skus": {10: 2}}],
        }
        fake._evaluate_bom_arrival_window = lambda: {"enabled": False, "window_sec": 0.0, "feasible": True, "violating_order_count": 0, "violations": []}
        fake._evaluate_order_time_window_metrics = lambda: {"span_overrun_total": 0.0, "orders": []}
        audit = TRAOptimizer._build_best_solution_audit(fake, 12.0, verification_result={"failures": ["manual_failure"]})
        self.assertTrue(audit["missing_sku_hit"])
        self.assertGreater(audit["invalid_z_task_count"], 0)
        self.assertGreater(audit["duplicate_tote_use_count"], 0)
        self.assertGreater(audit["unassigned_robot_task_count"], 0)
        self.assertTrue(audit["has_unreasonable_solution"])

    def test_layer_augmented_is_retired(self):
        opt = TRAOptimizer(TRARunConfig(search_scheme="layer_augmented", export_best_solution=False, write_iteration_logs=False))
        opt.problem = SimpleNamespace()
        opt.best = SimpleNamespace()
        opt.work = SimpleNamespace()
        opt.precheck_aborted = False
        with self.assertRaises(NotImplementedError):
            opt.run()

    def test_resource_time_run_stats_uses_synced_counters_and_layer_counts(self):
        fake = SimpleNamespace(
            iter_log=[
                {"selected_resource_layer": "X", "local_accept": False, "validation_trigger": "", "coverage_hard_reject": False},
                {"selected_resource_layer": "Y", "local_accept": True, "validation_trigger": "periodic", "coverage_hard_reject": True, "exact_eval_cache_hit_count": 3},
                {"selected_resource_layer": "Z", "local_accept": True, "validation_trigger": "", "coverage_hard_reject": False, "exact_eval_cache_hit_count": 5},
            ],
            layer_names=["X", "Y", "Z"],
            _ensure_log_dir=lambda: "tmp",
            run_start_time_sec=1.0,
            run_total_time_sec=2.0,
            _runtime_elapsed_sec=lambda: 2.0,
            layer_runtime_sec_by_name={"X": 1.0, "Y": 0.5, "Z": 0.25, "U": 0.1},
            layer_trial_count_by_name={"X": 1.0, "Y": 1.0, "Z": 1.0},
            global_eval_count=4,
            best=SimpleNamespace(z=99.0),
            coverage_hard_reject_count=7,
            x_failure_decapitation_count=2,
            stop_reason="best_z_no_change_50",
            cfg=SimpleNamespace(resource_real_eval_period=8),
            operator_stats={},
            _timing_breakdown_payload=lambda: {},
        )
        payload = TRAOptimizer._resource_time_run_stats_payload(fake)
        self.assertEqual(payload["coverage_hard_reject_count"], 7)
        self.assertEqual(payload["layer_selected_count_by_name"], {"X": 1, "Y": 1, "Z": 1})
        self.assertEqual(payload["layer_accepted_count_by_name"], {"X": 0, "Y": 1, "Z": 1})
        self.assertEqual(payload["resource_real_eval_period"], 8)
        self.assertEqual(payload["exact_eval_cache_hit_count"], 5)
        self.assertEqual(payload["kitting_span_hard_reject_count"], 0)

    def test_batch_summary_keeps_resource_real_eval_period_and_layer_counts(self):
        rows = [
            {
                "scale": "SMALL",
                "status": "ok",
                "best_z": 90.0,
                "runtime_sec": 12.0,
                "initial_makespan": 100.0,
                "improvement_ratio": 0.1,
                "initial_task_count": 10,
                "initial_subtask_count": 5,
                "initial_station_loads": {"0": 5.0},
                "initial_robot_path_lengths": {"0": 3.0},
                "initial_robot_path_length_total": 3.0,
                "bom_unique_sku_counts": [2, 3],
                "bom_unique_sku_total": 5,
                "resource_real_eval_period": 8,
                "coverage_hard_reject_count": 4,
                "exact_eval_cache_hit_count": 9,
                "x_failure_decapitation_count": 2,
                "stop_reason": "best_z_no_change_50",
                "layer_selected_x": 12,
                "layer_selected_y": 88,
                "layer_selected_z": 100,
                "layer_accepted_x": 2,
                "layer_accepted_y": 20,
                "layer_accepted_z": 24,
                "coverage_ok": True,
                "makespan_consistent": True,
                "has_unreasonable_solution": False,
            }
        ]
        summary_rows = batch_script._summarize(rows)
        self.assertEqual(len(summary_rows), 1)
        row = summary_rows[0]
        self.assertEqual(row["resource_real_eval_period"], 8)
        self.assertEqual(row["coverage_hard_reject_count"], 4)
        self.assertEqual(row["layer_selected_x"], 12)
        self.assertEqual(row["layer_accepted_z"], 24)


if __name__ == "__main__":
    unittest.main()
