import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from config.ofs_config import OFSConfig
from Gurobi.resource_time_alns.engine import ResourceTimeALNSEngine
from Gurobi.resource_time_alns.operators_y import (
    Y_DESTROY_OPERATORS,
    _plan_y_assignments,
    _station_choice_cost,
    y_destroy_heavy_robot_tail,
    y_destroy_initial_idle_head,
    y_destroy_max_tardiness_blocker,
)
from Gurobi.resource_time_alns.operators_z import (
    Z_REPAIR_OPERATORS,
    _candidate_stack_ids,
    _descriptor_from_plan,
    _normalize_joint_sort_plan,
    _predict_station_queues,
    _rough_route_feasibility,
    build_full_z_assignment,
    validate_z_assignment,
    validate_z_assignment_detail,
    z_destroy_spread_hotspot_window,
)
from Gurobi.resource_time_alns.state import (
    OperatorArm,
    ResourceConfig,
    ResourceSubtask,
    UpperEvalResult,
    ValidatedIncumbent,
    WorkUnitInfo,
    ZTaskDescriptor,
)
from Gurobi.resource_time_alns.surrogate import ResourceSurrogateScorer
from Gurobi.resource_time_alns.utils import pick_soft_greedy_min
from Gurobi.resource_time_alns.validator import ResourceValidator
from Gurobi.sp4 import SP4_Robot_Router, SP4RoutingInfeasibleError
from Gurobi.tra import TRAOptimizer


class ResourceTimeALNSRuntimeOptimizationTests(unittest.TestCase):
    def _build_snapshot_subtask(
        self,
        subtask_id,
        station_id,
        rank,
        assigned_robot_id,
        task_rows,
    ):
        return SimpleNamespace(
            id=int(subtask_id),
            assigned_station_id=int(station_id),
            station_sequence_rank=int(rank),
            assigned_robot_id=int(assigned_robot_id),
            execution_tasks=list(task_rows),
        )

    def test_validation_signature_distinguishes_x_structure_change(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
                "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 2),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0", "1:11:0"), 0, 0, [], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:12:0",), 1, 0, [], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=10,
        ).rebuild_indices()
        sig_before = config.validation_signature()
        config.subtasks[1].work_unit_ids = ("1:10:0",)
        config.subtasks[2].work_unit_ids = ("1:11:0", "1:12:0")
        self.assertNotEqual(sig_before, config.validation_signature())

    def test_should_validate_skips_periodic_same_config(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(resource_real_eval_period=8, no_improve_limit=3)
        engine.last_validation_iter = 0
        engine.last_validation_f_raw = 100.0
        engine.best_f_raw = 100.0
        engine.no_improve_rounds = 10
        signature = ("same",)
        engine.last_validated_signature = signature
        reason = ResourceTimeALNSEngine._should_validate(engine, 8, SimpleNamespace(F_raw=105.0), signature)
        self.assertEqual(reason, "periodic_skip_same_config")

    def test_destroy_mu_uses_three_tiers(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(
            resource_destroy_mu_base=0.10,
            resource_destroy_mu_medium=0.20,
            resource_destroy_mu_heavy=0.35,
            resource_destroy_mu_medium_trigger=30,
            resource_heavy_destroy_trigger=50,
        )
        engine.no_improve_rounds = 10
        self.assertEqual(ResourceTimeALNSEngine._current_destroy_mu(engine), (0.10, False, "base"))
        engine.no_improve_rounds = 35
        self.assertEqual(ResourceTimeALNSEngine._current_destroy_mu(engine), (0.20, False, "medium"))
        engine.no_improve_rounds = 55
        self.assertEqual(ResourceTimeALNSEngine._current_destroy_mu(engine), (0.35, True, "heavy"))

    def test_z_candidate_stack_ids_truncates_extra_candidates(self):
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (1,), (1,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        all_stack_ids = [100, 200, 201, 202, 203, 204, 205, 206]
        stack_xy = {
            100: (0.0, 0.0),
            200: (1.0, 0.0),
            201: (2.0, 0.0),
            202: (3.0, 0.0),
            203: (4.0, 0.0),
            204: (5.0, 0.0),
            205: (6.0, 0.0),
            206: (7.0, 0.0),
        }
        opt = SimpleNamespace(
            cfg=SimpleNamespace(resource_z_candidate_stack_topk=5),
            _x_candidate_stack_ids_for_sku=lambda _sku: list(all_stack_ids),
            _stack_xy=lambda sid: stack_xy.get(int(sid)),
            _xy_manhattan=lambda a, b: abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1])),
            _z_best_insertion_detour=lambda sid: float(int(sid)),
        )
        rows = _candidate_stack_ids(opt, config, config.subtasks[1], seed_stack_ids=[999])
        self.assertEqual(rows[:2], [999, 100])
        self.assertEqual(set(rows[2:]), {200, 201, 202, 203, 204})
        self.assertEqual(len(rows[2:]), 5)

    def test_z_best_insertion_detour_uses_cache(self):
        fake = TRAOptimizer.__new__(TRAOptimizer)
        fake._z_detour_cache = {}
        fake.anchor_reference = {"robot_insertion_windows": {0: [(1, 2)]}}
        fake._warehouse_distance_scale = lambda: 1.0
        fake._stack_xy = lambda sid: {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (1.0, 0.0)}.get(int(sid))
        calls = {"n": 0}

        def _manhattan(left, right):
            calls["n"] += 1
            return abs(float(left[0]) - float(right[0])) + abs(float(left[1]) - float(right[1]))

        fake._xy_manhattan = _manhattan
        first = TRAOptimizer._z_best_insertion_detour(fake, 3)
        after_first = int(calls["n"])
        second = TRAOptimizer._z_best_insertion_detour(fake, 3)
        self.assertEqual(first, second)
        self.assertEqual(int(calls["n"]), after_first)

    def test_empty_candidate_failure_penalizes_and_applies_layer_cooldown(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(
            resource_empty_candidate_reward=-2.0,
            resource_empty_candidate_layer_cooldown=3,
            resource_layer_base_weight_x=0.10,
            resource_layer_base_weight_y=0.45,
            resource_layer_base_weight_z=0.45,
            resource_component_weight_x=1.0,
            resource_component_weight_y=1.0,
            resource_component_weight_z=1.0,
            resource_layer_score_epsilon=0.05,
            resource_force_rotate_threshold=20,
            resource_layer_explore_eps=0.0,
            resource_stagnation_boost=0.15,
        )
        engine.operator_arms = {
            "X": {
                "destroy": {"dx": OperatorArm(name="dx")},
                "repair": {"rx": OperatorArm(name="rx")},
            },
            "Y": {
                "destroy": {"dy": OperatorArm(name="dy")},
                "repair": {"ry": OperatorArm(name="ry")},
            },
            "Z": {
                "destroy": {"dz": OperatorArm(name="dz")},
                "repair": {"rz": OperatorArm(name="rz")},
            },
        }
        engine.layer_exec_since_update = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_last_update_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_failure_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_dynamic_multiplier = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.layer_ema_improve = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.layer_stagnation = {"X": 0, "Y": 0, "Z": 0}
        engine.last_selected_layer = ""
        engine.current_eval = SimpleNamespace(Sx=1.0, Sy=1.0, Sz=1.0)
        import random
        engine.rng = random.Random(0)

        penalized = ResourceTimeALNSEngine._apply_empty_candidate_failure(engine, "X", [("dx", "rx")], 10)
        self.assertTrue(penalized)
        self.assertEqual(engine.operator_arms["X"]["destroy"]["dx"].pending_rewards, [-2.0])
        self.assertEqual(engine.operator_arms["X"]["repair"]["rx"].pending_rewards, [-2.0])
        self.assertEqual(engine._current_layer_cooldown_remaining("X", 11), 3)

        layer, _ = ResourceTimeALNSEngine._select_layer(engine, 11)
        self.assertNotEqual(layer, "X")

    def test_exact_eval_cache_reuses_structural_state(self):
        opt = SimpleNamespace(
            best=SimpleNamespace(z=100.0),
            cfg=SimpleNamespace(
                seed=42,
                resource_component_weight_x=1.0,
                resource_component_weight_y=1.0,
                resource_component_weight_z=1.0,
                resource_duplicate_tote_penalty=100000.0,
                resource_use_surrogate_calibrator=False,
                resource_real_eval_period=8,
                resource_residual_half_life=3.0,
                resource_residual_uncertainty_cap=20.0,
                resource_surrogate_trust_radius=0.35,
            ),
            problem=SimpleNamespace(station_list=[0, 1]),
        )
        scorer = ResourceSurrogateScorer(opt)
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        calls = {"n": 0}
        original = scorer._compute_structural_state

        def wrapped(cfg):
            calls["n"] += 1
            return original(cfg)

        scorer._compute_structural_state = wrapped
        first = scorer.evaluate(config, iterations_since_last_validation=1, distance_to_last_validated=0.0)
        second = scorer.evaluate(config, iterations_since_last_validation=4, distance_to_last_validated=0.0)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(bool(first.metadata.get("used_exact_eval_cache", False)))
        self.assertTrue(bool(second.metadata.get("used_exact_eval_cache", False)))
        self.assertEqual(int(second.metadata.get("exact_eval_cache_hit_count", 0)), 1)

    def test_empty_candidate_round_does_not_count_as_effective_iteration(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        self.assertFalse(ResourceTimeALNSEngine._counts_as_effective_iteration(engine, {"generated_count": 0}))
        self.assertTrue(ResourceTimeALNSEngine._counts_as_effective_iteration(engine, {"generated_count": 1}))

    def test_surrogate_rejects_unmet_coverage_with_infinite_score(self):
        tote = SimpleNamespace(id=100, sku_quantity_map={999: 1})
        opt = SimpleNamespace(
            best=SimpleNamespace(z=100.0),
            cfg=SimpleNamespace(
                seed=42,
                resource_component_weight_x=1.0,
                resource_component_weight_y=1.0,
                resource_component_weight_z=1.0,
                resource_duplicate_tote_penalty=100000.0,
                resource_use_surrogate_calibrator=False,
                resource_real_eval_period=8,
                resource_residual_half_life=3.0,
                resource_residual_uncertainty_cap=20.0,
                resource_surrogate_trust_radius=0.35,
            ),
            problem=SimpleNamespace(station_list=[0], id_to_tote={100: tote}),
        )
        scorer = ResourceSurrogateScorer(opt)
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        result = scorer.evaluate(config)
        self.assertEqual(result.unmet_sku_total, 1)
        self.assertFalse(result.coverage_feasible)
        self.assertEqual(result.F_raw, float("inf"))
        self.assertEqual(result.F_cal, float("inf"))

    def test_stagnation_increment_uses_empty_and_cache_rules(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(
            resource_empty_candidate_stagnation_increment=0.0,
            resource_cache_hit_stagnation_increment=0.2,
        )
        self.assertEqual(ResourceTimeALNSEngine._stagnation_increment(engine, False, False, False), 0.0)
        self.assertEqual(ResourceTimeALNSEngine._stagnation_increment(engine, True, True, False), 0.2)
        self.assertEqual(ResourceTimeALNSEngine._stagnation_increment(engine, True, False, False), 1.0)
        self.assertEqual(ResourceTimeALNSEngine._stagnation_increment(engine, True, False, True), 0.0)

    def test_x_failure_state_decapitates_layer_multiplier_and_applies_failure_cooldown(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(
            resource_layer_fail_threshold=3,
            resource_layer_fail_multiplier=0.1,
            resource_layer_fail_cooldown=10,
        )
        engine.consecutive_fail_count = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_dynamic_multiplier = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.layer_failure_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.x_failure_decapitation_count = 0

        ResourceTimeALNSEngine._update_failure_state(engine, "X", False, False, 5)
        ResourceTimeALNSEngine._update_failure_state(engine, "X", False, False, 6)
        ResourceTimeALNSEngine._update_failure_state(engine, "X", False, False, 7)
        self.assertEqual(engine.layer_dynamic_multiplier["X"], 0.1)
        self.assertEqual(engine.layer_failure_cooldown_until_iter["X"], 17)
        self.assertEqual(engine.x_failure_decapitation_count, 1)

        ResourceTimeALNSEngine._update_failure_state(engine, "Y", True, True, 8)
        self.assertEqual(engine.layer_dynamic_multiplier["X"], 1.0)
        self.assertEqual(engine.layer_failure_cooldown_until_iter["X"], 0)

    def test_exact_cache_hits_raise_adaptive_destroy_bonus(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace(
            resource_adaptive_destroy_cache_hit_trigger=3,
            resource_adaptive_destroy_bonus_step=0.05,
            resource_adaptive_destroy_bonus_cap=0.20,
        )
        engine.consecutive_exact_cache_hit_count = 0
        engine.adaptive_destroy_bonus = 0.0
        ResourceTimeALNSEngine._update_exact_cache_funnel(engine, True, False)
        ResourceTimeALNSEngine._update_exact_cache_funnel(engine, True, False)
        ResourceTimeALNSEngine._update_exact_cache_funnel(engine, True, False)
        self.assertEqual(engine.adaptive_destroy_bonus, 0.05)
        self.assertEqual(engine.consecutive_exact_cache_hit_count, 0)
        ResourceTimeALNSEngine._update_exact_cache_funnel(engine, False, True)
        self.assertEqual(engine.adaptive_destroy_bonus, 0.0)

    def test_validator_hard_rejects_unassigned_robot_tasks(self):
        tote = SimpleNamespace(id=100, sku_quantity_map={10: 1})
        opt = SimpleNamespace(
            problem=SimpleNamespace(id_to_tote={100: tote}, subtask_list=[]),
            _run_sp4=lambda: None,
            evaluate=lambda: 123.0,
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda *args, **kwargs: "snapshot",
        )
        validator = ResourceValidator(opt)
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()

        def stub_materialize(_config):
            opt.problem.subtask_list = [
                SimpleNamespace(
                    id=1,
                    assigned_robot_id=-1,
                    execution_tasks=[SimpleNamespace(task_id=1, sub_task_id=1, robot_id=-1)],
                )
            ]

        validator.materialize = stub_materialize
        result = validator.validate(config, 1)
        self.assertEqual(result["hard_reject_reason"], "unassigned_robot_task_hard_reject")
        self.assertEqual(result["unassigned_robot_task_count"], 1)
        self.assertEqual(result["makespan"], float("inf"))

    def test_validator_includes_conflict_summary_for_unassigned_tasks(self):
        tote = SimpleNamespace(id=100, sku_quantity_map={10: 1})
        opt = SimpleNamespace(
            cfg=SimpleNamespace(resource_conflict_time_bucket_sec=20.0),
            problem=SimpleNamespace(id_to_tote={100: tote}, subtask_list=[]),
            sp4=SimpleNamespace(last_conflict_summary={"reason_code": "sp4_lkh_infeasible", "resource_type": "robot_capacity", "station_ids": [3]}),
            _run_sp4=lambda: None,
            evaluate=lambda: 123.0,
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda *args, **kwargs: "snapshot",
        )
        validator = ResourceValidator(opt)
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    2,
                    [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()

        def stub_materialize(_config):
            opt.problem.subtask_list = [
                SimpleNamespace(
                    id=1,
                    assigned_robot_id=-1,
                    execution_tasks=[SimpleNamespace(task_id=1, sub_task_id=1, robot_id=-1)],
                )
            ]

        validator.materialize = stub_materialize
        result = validator.validate(config, 1)
        self.assertEqual(result["hard_reject_reason"], "unassigned_robot_task_hard_reject")
        self.assertEqual(result["conflict_summary"]["reason_code"], "sp4_lkh_infeasible")
        self.assertIn(1, result["conflict_summary"]["failed_subtask_ids"])
        self.assertIn(100, result["conflict_summary"]["stack_ids"])

    def test_validator_converts_sp4_route_exception_into_hard_reject(self):
        tote = SimpleNamespace(id=100, sku_quantity_map={10: 1})
        summary = {
            "reason_code": "sp4_lkh_no_solution",
            "failed_task_ids": [1, 2],
            "failed_subtask_ids": [1],
            "station_ids": [0],
            "stack_ids": [100],
            "strategy_name": "PATH_CHEAPEST_ARC",
            "time_limit_seconds": 30,
        }
        opt = SimpleNamespace(
            problem=SimpleNamespace(id_to_tote={100: tote}, subtask_list=[]),
            _run_sp4=lambda *args, **kwargs: (_ for _ in ()).throw(SP4RoutingInfeasibleError(summary)),
            evaluate=lambda: 123.0,
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda *args, **kwargs: "snapshot",
        )
        validator = ResourceValidator(opt)
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)], ("st_1",))
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        validator.materialize = lambda _config: None
        result = validator.validate(config, 1)
        self.assertEqual(result["hard_reject_reason"], "unassigned_robot_task_hard_reject")
        self.assertEqual(result["conflict_summary"]["reason_code"], "sp4_lkh_no_solution")
        self.assertEqual(result["unassigned_robot_task_count"], 2)

    def test_sp4_solve_uses_greedy_fallback_after_route_failure(self):
        router = SP4_Robot_Router.__new__(SP4_Robot_Router)
        router.problem = SimpleNamespace(robot_list=[])
        router.last_infeasible_reason = ""
        router.last_conflict_summary = {}
        router._solve_mip_pdp_v2 = lambda _sub_tasks: ({}, {})
        router._solve_LKH = lambda *args, **kwargs: (_ for _ in ()).throw(SP4RoutingInfeasibleError({"reason_code": "sp4_lkh_no_solution"}))
        router._greedy_fallback_route = lambda valid_tasks, **kwargs: ({1: 12.0}, {0: 0})
        result_times, result_assign = SP4_Robot_Router.solve(
            router,
            [SimpleNamespace(execution_tasks=[1])],
            use_mip=False,
            enable_greedy_fallback=True,
            raise_on_no_solution=True,
        )
        self.assertEqual(result_times, {1: 12.0})
        self.assertEqual(result_assign, {0: 0})

    def test_sp4_solve_raises_when_no_fallback_is_enabled(self):
        router = SP4_Robot_Router.__new__(SP4_Robot_Router)
        router.problem = SimpleNamespace(robot_list=[])
        router.last_infeasible_reason = ""
        router.last_conflict_summary = {}
        router._solve_mip_pdp_v2 = lambda _sub_tasks: ({}, {})
        router._solve_LKH = lambda *args, **kwargs: (_ for _ in ()).throw(SP4RoutingInfeasibleError({"reason_code": "sp4_lkh_no_solution"}))
        with self.assertRaises(SP4RoutingInfeasibleError):
            SP4_Robot_Router.solve(
                router,
                [SimpleNamespace(execution_tasks=[1])],
                use_mip=False,
                enable_greedy_fallback=False,
                raise_on_no_solution=True,
            )

    def test_sp4_same_subtask_vehicle_mode_switches_correctly(self):
        router = SP4_Robot_Router.__new__(SP4_Robot_Router)
        self.assertTrue(router._should_enforce_same_subtask_vehicle(2, "strict", 2))
        self.assertTrue(router._should_enforce_same_subtask_vehicle(2, "conditional", 2))
        self.assertFalse(router._should_enforce_same_subtask_vehicle(3, "conditional", 2))
        self.assertFalse(router._should_enforce_same_subtask_vehicle(2, "relaxed", 2))

    def test_sp4_strategy_enum_supports_requested_fast_strategies(self):
        router = SP4_Robot_Router.__new__(SP4_Robot_Router)
        self.assertIsNotNone(router._strategy_enum("PATH_CHEAPEST_ARC"))
        self.assertIsNotNone(router._strategy_enum("SAVINGS"))
        self.assertIsNotNone(router._strategy_enum("PARALLEL_CHEAPEST_INSERTION"))

    def test_y_station_choice_cost_penalizes_hot_route(self):
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        stack = SimpleNamespace(store_point=point(0, 0))
        station0 = SimpleNamespace(point=point(0, 4))
        station1 = SimpleNamespace(point=point(10, 0))
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:10:0",), 0, 1, [ZTaskDescriptor(2, 100, "FLIP", (100,), (100,), (), None)], ("st_2",)),
                3: ResourceSubtask(3, 1, ("1:10:0",), -1, -1, [ZTaskDescriptor(3, 100, "FLIP", (100,), (100,), (), None)], ("st_3",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=4,
            next_task_id=4,
        ).rebuild_indices()
        opt = SimpleNamespace(
            cfg=SimpleNamespace(
                resource_y_heat_grid_size=2.0,
                resource_y_time_bucket_sec=20.0,
                resource_y_heat_penalty_weight=3.0,
                resource_y_window_overlap_weight=1.0,
                resource_y_nearby_robot_budget_weight=0.5,
                resource_y_conflict_station_penalty=12.0,
                resource_y_conflict_time_bucket_penalty=8.0,
                resource_y_conflict_heat_bonus=4.0,
            ),
            problem=SimpleNamespace(
                station_list=[station0, station1],
                point_to_stack={100: stack},
                robot_list=[SimpleNamespace(start_point=point(0, 4)), SimpleNamespace(start_point=point(10, 0))],
            ),
        )
        loads = {0: 2.0, 1: 0.0}
        hot_cost = _station_choice_cost(opt, config, config.subtasks[3], 0, loads)
        cold_cost = _station_choice_cost(opt, config, config.subtasks[3], 1, loads)
        self.assertGreater(hot_cost, cold_cost)

    def test_y_destroy_initial_idle_head_releases_anchor_and_robot_prefix(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 2),
                "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 3),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), 1, 0, [], ("st_2",)),
                3: ResourceSubtask(3, 1, ("1:12:0",), 1, 1, [], ("st_3",)),
            },
            capacity_limits={1: 3},
            next_subtask_id=4,
            next_task_id=4,
        ).rebuild_indices()
        t = lambda tid, rid, arr_stack, arr_station, start, end: SimpleNamespace(
            task_id=int(tid),
            robot_id=int(rid),
            arrival_time_at_stack=float(arr_stack),
            arrival_time_at_station=float(arr_station),
            start_process_time=float(start),
            end_process_time=float(end),
        )
        snapshot = SimpleNamespace(
            subtask_state=[
                self._build_snapshot_subtask(1, 0, 0, 0, [t(1, 0, 5, 12, 12, 18)]),
                self._build_snapshot_subtask(2, 1, 0, 1, [t(2, 1, 8, 60, 60, 66)]),
                self._build_snapshot_subtask(3, 1, 1, 1, [t(3, 1, 2, 20, 20, 26)]),
            ]
        )
        opt = SimpleNamespace(work=snapshot, best=None, cfg=SimpleNamespace())
        result = y_destroy_initial_idle_head(opt, config, None, 2)
        self.assertTrue(result["success"])
        self.assertEqual(result["trigger_reason"], "initial_idle_head")
        self.assertEqual(result["source_station_ids"], [1])
        self.assertEqual(result["source_robot_ids"], [1])
        self.assertEqual(set(result["released_subtasks"].keys()), {2, 3})

    def test_y_destroy_initial_idle_head_falls_back_without_snapshot(self):
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [], ("st_1",))},
            capacity_limits={1: 1},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        opt = SimpleNamespace(work=None, best=None, cfg=SimpleNamespace(resource_soft_greedy_topk=1, resource_soft_greedy_noise=0.0))
        result = y_destroy_initial_idle_head(opt, config, None, 1)
        self.assertTrue(result["success"])
        self.assertEqual(set(result["released_subtasks"].keys()), {1})

    def test_y_destroy_max_tardiness_blocker_releases_target_and_blockers(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 2),
                "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 3),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), 0, 1, [], ("st_2",)),
                3: ResourceSubtask(3, 1, ("1:12:0",), 1, 0, [], ("st_3",)),
            },
            capacity_limits={1: 3},
            next_subtask_id=4,
            next_task_id=4,
        ).rebuild_indices()
        t = lambda tid, rid, arr_stack, arr_station, start, end: SimpleNamespace(
            task_id=int(tid),
            robot_id=int(rid),
            arrival_time_at_stack=float(arr_stack),
            arrival_time_at_station=float(arr_station),
            start_process_time=float(start),
            end_process_time=float(end),
        )
        snapshot = SimpleNamespace(
            subtask_state=[
                self._build_snapshot_subtask(1, 0, 0, 1, [t(1, 1, 4, 40, 40, 70)]),
                self._build_snapshot_subtask(2, 0, 1, 1, [t(2, 1, 10, 75, 75, 120)]),
                self._build_snapshot_subtask(3, 1, 0, 0, [t(3, 0, 3, 20, 20, 30)]),
            ]
        )
        opt = SimpleNamespace(work=snapshot, best=None, cfg=SimpleNamespace())
        result = y_destroy_max_tardiness_blocker(opt, config, None, 2)
        self.assertTrue(result["success"])
        self.assertEqual(result["trigger_reason"], "max_tardiness_blocker")
        self.assertEqual(result["target_subtask_ids"], [2])
        self.assertEqual(result["source_station_ids"], [0])
        self.assertEqual(result["source_robot_ids"], [1])
        self.assertEqual(set(result["released_subtasks"].keys()), {1, 2})

    def test_y_destroy_operators_include_new_targeted_variants(self):
        self.assertIn("y_destroy_initial_idle_head", Y_DESTROY_OPERATORS)
        self.assertIn("y_destroy_max_tardiness_blocker", Y_DESTROY_OPERATORS)
        self.assertIn("y_destroy_heavy_robot_tail", Y_DESTROY_OPERATORS)

    def test_y_destroy_heavy_robot_tail_releases_latest_heavy_robot_tasks(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 2),
                "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 3),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), 1, 0, [], ("st_2",)),
                3: ResourceSubtask(3, 1, ("1:12:0",), 1, 1, [], ("st_3",)),
            },
            capacity_limits={1: 3},
            next_subtask_id=4,
            next_task_id=4,
        ).rebuild_indices()
        t = lambda tid, rid, arr_stack, arr_station, start, end: SimpleNamespace(
            task_id=int(tid),
            robot_id=int(rid),
            arrival_time_at_stack=float(arr_stack),
            arrival_time_at_station=float(arr_station),
            start_process_time=float(start),
            end_process_time=float(end),
        )
        snapshot = SimpleNamespace(
            subtask_state=[
                self._build_snapshot_subtask(1, 0, 0, 0, [t(1, 0, 4, 20, 20, 30)]),
                self._build_snapshot_subtask(2, 1, 0, 1, [t(2, 1, 10, 60, 60, 90)]),
                self._build_snapshot_subtask(3, 1, 1, 1, [t(3, 1, 30, 100, 100, 130)]),
            ]
        )
        opt = SimpleNamespace(work=snapshot, best=None, cfg=SimpleNamespace())
        result = y_destroy_heavy_robot_tail(opt, config, None, 1)
        self.assertTrue(result["success"])
        self.assertEqual(result["trigger_reason"], "heavy_robot_unload")
        self.assertEqual(result["source_robot_ids"], [1])
        self.assertEqual(set(result["released_subtasks"].keys()), {3})

    def test_z_gurobi_like_sort_and_joint_sort_have_boosted_arms(self):
        self.assertIn("z_repair_gurobi_like_sort", Z_REPAIR_OPERATORS)
        self.assertIn("z_repair_joint_sort_colocated_flip", Z_REPAIR_OPERATORS)
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        arms = ResourceTimeALNSEngine._init_operator_arms(engine)
        self.assertGreaterEqual(arms["Z"]["repair"]["z_repair_gurobi_like_sort"].weight, 3.0)
        self.assertGreaterEqual(arms["Z"]["repair"]["z_repair_sort_range_shrink_first"].weight, 2.0)
        self.assertGreaterEqual(arms["Z"]["repair"]["z_repair_joint_sort_colocated_flip"].weight, 2.0)
        self.assertGreaterEqual(arms["Y"]["destroy"]["y_destroy_heavy_robot_tail"].weight, 2.0)

    def test_select_layer_consumes_forced_yz_queue_before_scores(self):
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = SimpleNamespace()
        engine.forced_layer_queue = deque(["Y", "Z"])
        engine._available_layers = lambda _iter_id: ["X", "Y", "Z"]
        first, first_forced = ResourceTimeALNSEngine._select_layer(engine, 1)
        second, second_forced = ResourceTimeALNSEngine._select_layer(engine, 2)
        self.assertEqual((first, first_forced), ("Y", True))
        self.assertEqual((second, second_forced), ("Z", True))
        self.assertEqual(len(engine.forced_layer_queue), 0)

    def test_predict_station_queues_respects_rank_order(self):
        descriptor_short = ZTaskDescriptor(1, 100, "FLIP", (1,), (1,), (), None, station_service_time=4.0, sku_pick_count=1)
        descriptor_long = ZTaskDescriptor(2, 101, "FLIP", (2,), (2,), (), None, station_service_time=10.0, sku_pick_count=1)
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [descriptor_long], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), 0, 1, [descriptor_short], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=3,
        ).rebuild_indices()
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        problem = SimpleNamespace(
            station_list=[SimpleNamespace(point=point(10, 0))],
            robot_list=[SimpleNamespace(id=0, start_point=point(0, 0))],
            point_to_stack={
                100: SimpleNamespace(store_point=point(1, 0), totes=[SimpleNamespace(id=11), SimpleNamespace(id=12)]),
                101: SimpleNamespace(store_point=point(2, 0)),
            },
        )
        opt = SimpleNamespace(problem=problem, best_validated=SimpleNamespace(snapshot=None), work=None)
        queue_ctx = _predict_station_queues(opt, config)
        waits = queue_ctx["expected_wait_times"]
        self.assertAlmostEqual(waits[1], 0.0)
        self.assertGreater(waits[2], 0.0)

    def test_build_full_z_assignment_prefers_cleaner_plan_when_station_wait_is_high(self):
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (1,), (1,), (), None, station_service_time=20.0, sku_pick_count=1)],
                    ("st_1",),
                ),
                2: ResourceSubtask(2, 1, ("1:11:0",), 0, 1, [], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=10,
        ).rebuild_indices()
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        problem = SimpleNamespace(
            station_list=[SimpleNamespace(point=point(10, 0))],
            robot_list=[SimpleNamespace(id=0, start_point=point(0, 0))],
            point_to_stack={
                100: SimpleNamespace(store_point=point(1, 0)),
                101: SimpleNamespace(store_point=point(2, 0)),
            },
            id_to_tote={
                11: SimpleNamespace(sku_quantity_map={11: 1}),
                12: SimpleNamespace(sku_quantity_map={11: 1}),
            },
        )
        cfg = SimpleNamespace(
            resource_z_queue_wait_threshold_sec=1.0,
            resource_z_queue_wait_multiplier=2.0,
            resource_z_queue_noise_weight=5.0,
            resource_z_queue_multistack_weight=0.8,
            resource_z_queue_sort_weight=4.0,
            resource_z_queue_station_service_weight=0.0,
            resource_z_mode_switch_penalty=1.0,
            resource_z_contention_time_bucket_sec=20.0,
            resource_y_heat_grid_size=4.0,
            resource_z_sequence_feasibility_cap=180.0,
            resource_stack_concurrency_limit=3,
            resource_tote_concurrency_limit=3,
            resource_choke_point_budget=3,
            resource_z_stack_contention_penalty=1.0,
            resource_z_tote_contention_penalty=1.0,
            resource_z_choke_point_penalty=1.0,
            resource_z_conflict_stack_penalty=1.0,
            resource_z_conflict_tote_penalty=1.0,
            resource_z_conflict_time_bucket_penalty=1.0,
            resource_z_route_feasibility_penalty=0.1,
            z_arrival_shift_soft_cap=999.0,
            z_wait_overflow_soft_cap=999.0,
            z_route_tail_soft_cap=999.0,
            z_route_gap_soft_cap=999.0,
            resource_soft_greedy_topk=1,
            resource_soft_greedy_noise=0.0,
        )
        opt = SimpleNamespace(
            cfg=cfg,
            problem=problem,
            best_validated=SimpleNamespace(snapshot=None),
            work=None,
            _x_candidate_stack_ids_for_sku=lambda sku_id: [100, 101],
            _z_best_insertion_detour=lambda stack_id: {100: 8.0, 101: 8.0}[int(stack_id)],
            _stack_xy=lambda stack_id: {100: (1.0, 0.0), 101: (2.0, 0.0)}[int(stack_id)],
            _xy_manhattan=lambda lhs, rhs: abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1])),
        )

        def fake_summary(_temp_subtask, stack_id, _task_ids):
            return {"hit_tote_ids": [11, 12]}

        def fake_plan(_temp_subtask, _dummy_task, stack_id, _hit_ids, mode, _task_ids):
            if int(stack_id) == 100:
                return {
                    "valid": True,
                    "target_stack_id": 100,
                    "operation_mode": "SORT",
                    "target_tote_ids": [11, 12],
                    "hit_tote_ids": [11],
                    "noise_tote_ids": [12],
                    "sort_layer_range": (0, 1),
                    "station_service_time": 2.0,
                }
            return {
                "valid": True,
                "target_stack_id": 101,
                "operation_mode": "FLIP",
                "target_tote_ids": [11],
                "hit_tote_ids": [11],
                "noise_tote_ids": [],
                "station_service_time": 2.0,
            }

        opt._z_stack_summary = fake_summary
        opt._z_build_plan_from_hits = fake_plan
        success, assignment, _meta = build_full_z_assignment(opt, config, 2, strategy="fallback", allow_fallback=True, external_used_totes=set(), rng=None)
        self.assertTrue(success)
        self.assertEqual([descriptor.stack_id for descriptor in assignment], [101])
        self.assertEqual([descriptor.mode for descriptor in assignment], ["FLIP"])

    def test_plan_y_assignments_prefers_starving_station_for_first_batch(self):
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
                "1:12:0": WorkUnitInfo("1:12:0", 1, 12, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [ZTaskDescriptor(1, 100, "FLIP", (1,), (1,), (), None, station_service_time=3.0, sku_pick_count=1)], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), -1, -1, [ZTaskDescriptor(2, 101, "FLIP", (2,), (2,), (), None, station_service_time=3.0, sku_pick_count=1)], ("st_2",)),
                3: ResourceSubtask(3, 1, ("1:12:0",), -1, -1, [ZTaskDescriptor(3, 102, "FLIP", (3,), (3,), (), None, station_service_time=3.0, sku_pick_count=1)], ("st_3",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=4,
            next_task_id=4,
        ).rebuild_indices()
        problem = SimpleNamespace(
            station_list=[SimpleNamespace(point=point(4, 0)), SimpleNamespace(point=point(20, 0))],
            robot_list=[SimpleNamespace(id=0, start_point=point(0, 0)), SimpleNamespace(id=1, start_point=point(18, 0))],
            point_to_stack={
                100: SimpleNamespace(store_point=point(3, 0)),
                101: SimpleNamespace(store_point=point(19, 0)),
                102: SimpleNamespace(store_point=point(18, 0)),
            },
        )
        cfg = SimpleNamespace(
            resource_y_heat_grid_size=4.0,
            resource_y_time_bucket_sec=20.0,
            resource_y_heat_penalty_weight=0.0,
            resource_y_window_overlap_weight=0.0,
            resource_y_nearby_robot_budget_weight=0.0,
            resource_y_conflict_station_penalty=0.0,
            resource_y_conflict_time_bucket_penalty=0.0,
            resource_y_conflict_heat_bonus=0.0,
            resource_y_triangle_projected_finish_weight=0.0,
            resource_y_triangle_arrival_weight=1.0,
            resource_y_starvation_bonus=100.0,
            resource_y_wave1_enable=True,
            resource_y_wave1_sort_penalty=8.0,
            resource_y_wave1_pick_weight=0.75,
            resource_y_first_batch_station_penalty=100.0,
            resource_y_first_batch_robot_penalty=50.0,
            resource_y_first_batch_starvation_bonus=20.0,
            resource_soft_greedy_topk=1,
            resource_soft_greedy_noise=0.0,
        )
        opt = SimpleNamespace(
            cfg=cfg,
            problem=problem,
            work=None,
            best_validated=SimpleNamespace(snapshot=None),
            _x_candidate_stack_ids_for_sku=lambda sku_id: {11: [101], 12: [102]}.get(int(sku_id), []),
            _stack_xy=lambda stack_id: {100: (3.0, 0.0), 101: (19.0, 0.0), 102: (18.0, 0.0)}.get(int(stack_id)),
        )
        plan = _plan_y_assignments(opt, config, {2: (-1, -1), 3: (-1, -1)}, "y_repair_earliest_finish", rng=None)
        self.assertTrue(plan["success"])
        assigned_stations = {int(meta["station_id"]) for meta in plan["assignments"].values()}
        self.assertEqual(assigned_stations, {0, 1})
        assigned_robots = {int(meta["selected_robot_id"]) for meta in plan["assignments"].values()}
        self.assertEqual(assigned_robots, {0, 1})

    def test_plan_y_assignments_wave1_minimax_avoids_selfish_robot_grab(self):
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), -1, -1, [], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), -1, -1, [], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=3,
        ).rebuild_indices()
        cfg = SimpleNamespace(
            resource_y_heat_grid_size=4.0,
            resource_y_time_bucket_sec=20.0,
            resource_y_heat_penalty_weight=0.0,
            resource_y_window_overlap_weight=0.0,
            resource_y_nearby_robot_budget_weight=0.0,
            resource_y_conflict_station_penalty=0.0,
            resource_y_conflict_time_bucket_penalty=0.0,
            resource_y_conflict_heat_bonus=0.0,
            resource_y_triangle_projected_finish_weight=0.0,
            resource_y_triangle_arrival_weight=1.0,
            resource_y_starvation_bonus=0.0,
            resource_y_wave1_enable=True,
            resource_y_wave1_sort_penalty=0.0,
            resource_y_wave1_pick_weight=0.0,
            resource_y_first_batch_station_penalty=0.0,
            resource_y_first_batch_robot_penalty=0.0,
            resource_y_first_batch_starvation_bonus=0.0,
            resource_soft_greedy_topk=1,
            resource_soft_greedy_noise=0.0,
        )
        problem = SimpleNamespace(
            station_list=[SimpleNamespace(point=point(0, 0)), SimpleNamespace(point=point(10, 0))],
            robot_list=[SimpleNamespace(id=0, start_point=point(0, 0)), SimpleNamespace(id=1, start_point=point(10, 0))],
            point_to_stack={100: SimpleNamespace(store_point=point(1, 0)), 101: SimpleNamespace(store_point=point(9, 0))},
        )
        opt = SimpleNamespace(
            cfg=cfg,
            problem=problem,
            work=None,
            best_validated=SimpleNamespace(snapshot=None),
            _x_candidate_stack_ids_for_sku=lambda sku_id: {10: [100], 11: [101]}.get(int(sku_id), []),
            _stack_xy=lambda stack_id: {100: (1.0, 0.0), 101: (9.0, 0.0)}.get(int(stack_id)),
        )
        matrix = {
            (1, 0, 0): 1.0,
            (1, 0, 1): 9.0,
            (1, 1, 0): 20.0,
            (1, 1, 1): 9.0,
            (2, 0, 0): 2.0,
            (2, 0, 1): 50.0,
            (2, 1, 0): 50.0,
            (2, 1, 1): 10.0,
        }
        with patch("Gurobi.resource_time_alns.operators_y._robot_station_arrival_cost", side_effect=lambda _opt, _cfg, row, station_id, robot_id, robot_frontier=None: matrix[(int(row.subtask_id), int(station_id), int(robot_id))]):
            plan = _plan_y_assignments(opt, config, {1: (-1, -1), 2: (-1, -1)}, "y_repair_earliest_finish", rng=None)
        self.assertTrue(plan["success"])
        self.assertEqual(int(plan["assignments"][1]["station_id"]), 1)
        self.assertEqual(int(plan["assignments"][1]["selected_robot_id"]), 1)
        self.assertEqual(int(plan["assignments"][2]["station_id"]), 0)
        self.assertEqual(int(plan["assignments"][2]["selected_robot_id"]), 0)

    def test_plan_y_assignments_wave1_heavy_task_gets_premium_combo(self):
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        light = ZTaskDescriptor(1, 100, "FLIP", (1,), (1,), (), None, station_service_time=1.0, sku_pick_count=1)
        heavy = ZTaskDescriptor(2, 101, "SORT", (2,), (2,), (), None, station_service_time=1.0, sku_pick_count=1)
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), -1, -1, [light], ("st_1",)),
                2: ResourceSubtask(2, 1, ("1:11:0",), -1, -1, [heavy], ("st_2",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=3,
            next_task_id=3,
        ).rebuild_indices()
        cfg = SimpleNamespace(
            resource_y_heat_grid_size=4.0,
            resource_y_time_bucket_sec=20.0,
            resource_y_heat_penalty_weight=0.0,
            resource_y_window_overlap_weight=0.0,
            resource_y_nearby_robot_budget_weight=0.0,
            resource_y_conflict_station_penalty=0.0,
            resource_y_conflict_time_bucket_penalty=0.0,
            resource_y_conflict_heat_bonus=0.0,
            resource_y_triangle_projected_finish_weight=0.0,
            resource_y_triangle_arrival_weight=1.0,
            resource_y_starvation_bonus=0.0,
            resource_y_wave1_enable=True,
            resource_y_wave1_sort_penalty=15.0,
            resource_y_wave1_pick_weight=0.0,
            resource_y_first_batch_station_penalty=0.0,
            resource_y_first_batch_robot_penalty=0.0,
            resource_y_first_batch_starvation_bonus=0.0,
            resource_soft_greedy_topk=1,
            resource_soft_greedy_noise=0.0,
        )
        problem = SimpleNamespace(
            station_list=[SimpleNamespace(point=point(0, 0)), SimpleNamespace(point=point(10, 0))],
            robot_list=[SimpleNamespace(id=0, start_point=point(0, 0)), SimpleNamespace(id=1, start_point=point(10, 0))],
            point_to_stack={100: SimpleNamespace(store_point=point(1, 0)), 101: SimpleNamespace(store_point=point(9, 0))},
        )
        opt = SimpleNamespace(
            cfg=cfg,
            problem=problem,
            work=None,
            best_validated=SimpleNamespace(snapshot=None),
            _x_candidate_stack_ids_for_sku=lambda sku_id: {10: [100], 11: [101]}.get(int(sku_id), []),
            _stack_xy=lambda stack_id: {100: (1.0, 0.0), 101: (9.0, 0.0)}.get(int(stack_id)),
        )
        matrix = {
            (1, 0, 0): 11.0,
            (1, 0, 1): 20.0,
            (1, 1, 0): 20.0,
            (1, 1, 1): 1.0,
            (2, 0, 0): 10.0,
            (2, 0, 1): 30.0,
            (2, 1, 0): 30.0,
            (2, 1, 1): 11.0,
        }
        with patch("Gurobi.resource_time_alns.operators_y._robot_station_arrival_cost", side_effect=lambda _opt, _cfg, row, station_id, robot_id, robot_frontier=None: matrix[(int(row.subtask_id), int(station_id), int(robot_id))]):
            plan = _plan_y_assignments(opt, config, {1: (-1, -1), 2: (-1, -1)}, "y_repair_earliest_finish", rng=None)
        self.assertTrue(plan["success"])
        self.assertEqual(int(plan["assignments"][2]["station_id"]), 0)
        self.assertEqual(int(plan["assignments"][2]["selected_robot_id"]), 0)

    def _build_sort_validation_fixture(self):
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        totes = [
            SimpleNamespace(id=100, sku_quantity_map={10: 1}),
            SimpleNamespace(id=101, sku_quantity_map={99: 1}),
            SimpleNamespace(id=102, sku_quantity_map={11: 1}),
        ]
        stack = SimpleNamespace(store_point=point(0, 0), totes=totes)
        station = SimpleNamespace(point=point(0, 0))
        config = ResourceConfig(
            work_units={
                "1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1),
                "1:11:0": WorkUnitInfo("1:11:0", 1, 11, 0, 1),
            },
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0", "1:11:0"), 0, 0, [], ("st_1",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        cfg = SimpleNamespace(
            resource_y_heat_grid_size=4.0,
            resource_z_contention_time_bucket_sec=20.0,
            resource_z_sequence_feasibility_cap=9999.0,
            resource_z_mode_switch_penalty=0.0,
            resource_z_stack_contention_penalty=0.0,
            resource_z_tote_contention_penalty=0.0,
            resource_z_choke_point_penalty=0.0,
            resource_z_route_feasibility_penalty=0.0,
            resource_stack_concurrency_limit=10,
            resource_tote_concurrency_limit=10,
            resource_choke_point_budget=10,
        )
        opt = SimpleNamespace(
            cfg=cfg,
            problem=SimpleNamespace(
                point_to_stack={100: stack},
                station_list=[station],
                id_to_tote={int(tote.id): tote for tote in totes},
            ),
            _z_best_insertion_detour=lambda _sid: 0.0,
        )
        return opt, config

    def test_z_sort_plan_canonicalizes_range_target_and_noise(self):
        opt, config = self._build_sort_validation_fixture()
        plan = {
            "operation_mode": "SORT",
            "target_stack_id": 100,
            "target_tote_ids": [100, 102],
            "hit_tote_ids": [100, 102],
            "noise_tote_ids": [],
            "sort_layer_range": (0, 2),
            "station_service_time": 0.0,
            "robot_service_time": 0.0,
        }
        descriptor = _descriptor_from_plan(opt, config.subtasks[1], plan, 1, 2)
        self.assertEqual(tuple(descriptor.target_tote_ids), (100, 101, 102))
        self.assertEqual(tuple(descriptor.hit_tote_ids), (100, 102))
        self.assertEqual(tuple(descriptor.noise_tote_ids), (101,))
        self.assertEqual(float(descriptor.station_service_time), float(getattr(OFSConfig, "MOVE_EXTRA_TOTE_TIME", 1.0)))
        ok, reason, _ = validate_z_assignment_detail(opt, config, config.subtasks[1], [descriptor])
        self.assertTrue(ok, reason)

    def test_z_validate_assignment_reports_sort_detail_reasons(self):
        opt, config = self._build_sort_validation_fixture()
        subtask = config.subtasks[1]
        overlap = [ZTaskDescriptor(1, 100, "SORT", (100, 101, 102), (100, 102), (101, 102), (0, 2), station_service_time=1.0, robot_service_time=1.0, sku_pick_count=2)]
        ok, reason, meta = validate_z_assignment_detail(opt, config, subtask, overlap)
        self.assertFalse(ok)
        self.assertEqual(reason, "hit_noise_overlap")
        self.assertEqual(meta["descriptor_index"], 0)

        non_contiguous = [ZTaskDescriptor(1, 100, "SORT", (100, 102), (100, 102), (), (0, 2), station_service_time=1.0, robot_service_time=1.0, sku_pick_count=2)]
        ok, reason, meta = validate_z_assignment_detail(opt, config, subtask, non_contiguous)
        self.assertFalse(ok)
        self.assertEqual(reason, "sort_target_not_contiguous")
        self.assertEqual(meta["expected_target_tote_ids"], [100, 101, 102])

        descriptor = ZTaskDescriptor(1, 100, "SORT", (100, 101, 102), (100, 102), (101,), (0, 2), station_service_time=1.0, robot_service_time=1.0, sku_pick_count=2)
        ok, reason, meta = validate_z_assignment_detail(opt, config, subtask, [descriptor], external_used_totes={101})
        self.assertFalse(ok)
        self.assertEqual(reason, "duplicate_or_blocked_tote")
        self.assertEqual(meta["tote_id"], 101)

    def test_z_joint_sort_normalize_keeps_continuous_range_and_noise(self):
        opt, config = self._build_sort_validation_fixture()
        plan = {
            "operation_mode": "SORT",
            "target_stack_id": 100,
            "target_tote_ids": [100, 102],
            "hit_tote_ids": [100, 102],
            "noise_tote_ids": [],
            "sort_layer_range": (0, 2),
            "station_service_time": 0.0,
            "robot_service_time": 0.0,
        }
        normalized = _normalize_joint_sort_plan(opt, plan, [100, 102])
        self.assertEqual(normalized["target_tote_ids"], [100, 101, 102])
        self.assertEqual(normalized["hit_tote_ids"], [100, 102])
        self.assertEqual(normalized["noise_tote_ids"], [101])
        descriptor = _descriptor_from_plan(opt, config.subtasks[1], normalized, 1, 2)
        ok, reason, _ = validate_z_assignment_detail(opt, config, config.subtasks[1], [descriptor])
        self.assertTrue(ok, reason)

    def test_z_choke_is_soft_not_validation_failure(self):
        opt, config = self._build_sort_validation_fixture()
        opt.cfg.resource_choke_point_budget = 1
        config.subtasks[2] = ResourceSubtask(
            2,
            1,
            ("1:10:0",),
            0,
            1,
            [ZTaskDescriptor(2, 100, "FLIP", (100,), (100,), (), None, station_service_time=1.0, robot_service_time=1.0, sku_pick_count=1)],
            ("st_2",),
        )
        config.rebuild_indices()
        descriptor = ZTaskDescriptor(1, 100, "SORT", (100, 101, 102), (100, 102), (101,), (0, 2), station_service_time=1.0, robot_service_time=1.0, sku_pick_count=2)
        ok, _penalty, meta = _rough_route_feasibility(opt, config, config.subtasks[1], [descriptor])
        self.assertTrue(ok)
        self.assertGreater(int(meta.get("choke_over", 0)), 0)

    def test_z_destroy_spread_hotspot_window_targets_loaded_station(self):
        opt, config = self._build_sort_validation_fixture()
        config.subtasks[1].station_id = 0
        config.subtasks[1].z_tasks = [
            ZTaskDescriptor(1, 100, "SORT", (100, 101, 102), (100, 102), (101,), (0, 2), station_service_time=20.0, robot_service_time=1.0, sku_pick_count=2)
        ]
        config.subtasks[2] = ResourceSubtask(
            2,
            1,
            ("1:10:0",),
            0,
            1,
            [ZTaskDescriptor(2, 100, "FLIP", (100,), (100,), (), None, station_service_time=20.0, robot_service_time=1.0, sku_pick_count=1)],
            ("st_2",),
        )
        config.subtasks[3] = ResourceSubtask(
            3,
            1,
            ("1:11:0",),
            1,
            0,
            [ZTaskDescriptor(3, 100, "FLIP", (102,), (102,), (), None, station_service_time=1.0, robot_service_time=1.0, sku_pick_count=1)],
            ("st_3",),
        )
        config.rebuild_indices()
        ctx = z_destroy_spread_hotspot_window(opt, config, rng=None, degree=1)
        self.assertTrue(ctx["success"])
        self.assertIn(int(ctx["subtask_id"]), {1, 2})

    def test_tail_guard_rejects_robot_tail_and_active_robot_regression(self):
        cfg = SimpleNamespace(resource_tail_guard_enabled=True, resource_tail_guard_ratio=1.05)
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = cfg
        task = lambda rid, arr, end: SimpleNamespace(robot_id=rid, arrival_time_at_stack=arr, arrival_time_at_station=arr, end_process_time=end)
        inc_snapshot = SimpleNamespace(
            subtask_state=[
                SimpleNamespace(id=1, execution_tasks=[task(0, 100.0, 110.0)]),
                SimpleNamespace(id=2, execution_tasks=[task(1, 90.0, 100.0)]),
            ]
        )
        cand_snapshot = SimpleNamespace(
            subtask_state=[
                SimpleNamespace(id=1, execution_tasks=[task(0, 120.0, 130.0)]),
            ]
        )
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [], ("st_1",)),
                2: ResourceSubtask(2, 2, ("1:10:0",), 0, 1, [], ("st_2",)),
            },
            capacity_limits={},
            next_subtask_id=3,
            next_task_id=1,
        )
        engine.best_validated = ValidatedIncumbent(config=config, makespan=100.0, iter_id=0, snapshot=inc_snapshot)
        reason, meta = engine._tail_guard_reason({"snapshot": cand_snapshot}, config)
        self.assertEqual(reason, "latest_robot_finish_regression")
        self.assertEqual(int(meta["candidate_active_robot_count"]), 1)

    def test_z_diversification_operator_weight_floor_is_preserved(self):
        cfg = SimpleNamespace(resource_z_diversification_weight_floor=1.25)
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.cfg = cfg
        engine.operator_arms = {
            "Z": {
                "destroy": {"z_destroy_spread_hotspot_window": OperatorArm("z_destroy_spread_hotspot_window", weight=0.01)},
                "repair": {"z_repair_spread_region_balance": OperatorArm("z_repair_spread_region_balance", weight=0.01)},
            }
        }
        engine._apply_operator_weight_floors()
        self.assertGreaterEqual(engine.operator_arms["Z"]["destroy"]["z_destroy_spread_hotspot_window"].weight, 1.25)
        self.assertGreaterEqual(engine.operator_arms["Z"]["repair"]["z_repair_spread_region_balance"].weight, 1.25)

    def test_z_validate_assignment_rejects_sequence_feasibility_over_cap(self):
        point = lambda x, y: SimpleNamespace(x=float(x), y=float(y))
        tote = SimpleNamespace(id=100, sku_quantity_map={10: 1})
        stack = SimpleNamespace(store_point=point(0, 0), totes=[tote])
        station = SimpleNamespace(point=point(20, 0))
        config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [], ("st_1",)),
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        opt = SimpleNamespace(
            cfg=SimpleNamespace(
                resource_y_heat_grid_size=4.0,
                resource_z_contention_time_bucket_sec=20.0,
                resource_z_sequence_feasibility_cap=1.0,
                resource_z_mode_switch_penalty=6.0,
                resource_z_stack_contention_penalty=6.0,
                resource_z_tote_contention_penalty=10.0,
                resource_z_choke_point_penalty=4.0,
                resource_z_route_feasibility_penalty=20.0,
                resource_stack_concurrency_limit=2,
                resource_tote_concurrency_limit=1,
                resource_choke_point_budget=2,
            ),
            problem=SimpleNamespace(point_to_stack={100: stack}, station_list=[station], id_to_tote={100: tote}),
            _z_best_insertion_detour=lambda _sid: 100.0,
        )
        descriptors = [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None, station_service_time=2.0, robot_service_time=2.0, sku_pick_count=1)]
        self.assertFalse(validate_z_assignment(opt, config, config.subtasks[1], descriptors))

    def test_engine_does_not_update_best_on_validator_hard_reject(self):
        import random

        base_config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        candidate_config = base_config.clone()
        candidate_config.subtasks[1].station_rank = 1
        base_eval = UpperEvalResult(Sx=1.0, Sy=1.0, Sz=1.0, F_raw=100.0, F_cal=100.0)
        candidate_eval = UpperEvalResult(
            Sx=1.0,
            Sy=0.8,
            Sz=0.7,
            F_raw=90.0,
            F_cal=90.0,
            metadata={"used_exact_eval_cache": False, "exact_eval_cache_hit_count": 0},
        )

        class FakeScorer:
            def evaluate(self, config, **kwargs):
                if config.validation_signature() == base_config.validation_signature():
                    return base_eval
                return candidate_eval

            def update_with_validation(self, *_args, **_kwargs):
                raise AssertionError("validation hard reject should not update calibrator")

        restore_calls = []
        cfg = SimpleNamespace(
            max_iters=1,
            resource_sa_cooling=0.95,
            resource_sa_reheat_factor=1.25,
            resource_candidate_pool_log=False,
            resource_stop_if_best_z_no_change_rounds=50,
        )
        opt = SimpleNamespace(
            cfg=cfg,
            best="best_snapshot",
            work="best_snapshot",
            work_z=100.0,
            iter_log=[],
            candidate_iter_log=[],
            layer_runtime_sec_by_name={"X": 0.0, "Y": 0.0, "Z": 0.0, "U": 0.0},
            layer_trial_count_by_name={"X": 0.0, "Y": 0.0, "Z": 0.0},
            global_eval_count=0,
            stop_reason="",
            run_total_time_sec=0.0,
            _runtime_elapsed_sec=lambda: 0.0,
            _clear_z_detour_cache=lambda: None,
            _write_logs=lambda: None,
            restore_snapshot=lambda snap: restore_calls.append(snap),
            operator_stats={},
        )
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.opt = opt
        engine.cfg = cfg
        engine.rng = random.Random(0)
        engine.validator = SimpleNamespace(
            validate=lambda config, iter_id: {
                "makespan": float("inf"),
                "snapshot": None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "unassigned_robot_task_hard_reject",
                "unmet_sku_total": 0,
                "unassigned_robot_task_count": 1,
                "unassigned_robot_tasks": [{"task_id": 1, "subtask_id": 1}],
                "lkh_call_count": 1,
            }
        )
        engine.scorer = FakeScorer()
        engine.current_config = base_config.clone()
        engine.current_eval = base_eval
        engine.best_validated = ValidatedIncumbent(
            config=base_config.clone(),
            makespan=100.0,
            iter_id=0,
            snapshot="best_snapshot",
        )
        engine.last_validated_config = base_config.clone()
        engine.last_validated_signature = base_config.validation_signature()
        engine.last_validation_iter = 0
        engine.last_validation_f_raw = 100.0
        engine.recent_validated_makespans = [100.0]
        engine.temperature = 10.0
        engine.layer_ema_improve = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.layer_stagnation = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        engine.layer_exec_since_update = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_last_update_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_failure_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_dynamic_multiplier = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.consecutive_fail_count = {"X": 0, "Y": 0, "Z": 0}
        engine.last_selected_layer = ""
        engine.no_improve_rounds = 0.0
        engine.no_best_z_change_rounds = 0.0
        engine.validated_best_no_change_rounds = 0
        engine.best_f_raw = 100.0
        engine.consecutive_exact_cache_hit_count = 0
        engine.adaptive_destroy_bonus = 0.0
        engine.coverage_hard_reject_count = 0
        engine.x_failure_decapitation_count = 0
        engine.lkh_call_count = 0
        engine.lkh_budget_consumed_by_rollback = 0
        engine.operator_arms = {"X": {"destroy": {}, "repair": {}}, "Y": {"destroy": {}, "repair": {}}, "Z": {"destroy": {}, "repair": {}}}
        engine.action_signature_history = {"X": [], "Y": [], "Z": []}
        engine.action_signature_seen = {"X": set(), "Y": set(), "Z": set()}
        engine.joint_colocated_sort_postprocess_stats = {
            "triggered": 0.0,
            "candidate_groups": 0.0,
            "submitted": 0.0,
            "applied": 0.0,
            "makespan_improvement": 0.0,
            "rejected_capacity": 0.0,
            "rejected_interval_illegal": 0.0,
            "rejected_noise": 0.0,
            "rejected_eval_not_better": 0.0,
            "rejected_validation": 0.0,
            "rejected_target_conflict": 0.0,
        }
        engine._refresh_operator_stats_payload = lambda: None
        engine._accumulate_joint_postprocess_stats = lambda *_args, **_kwargs: None
        engine._select_layer = lambda iter_id: ("X", False)
        engine._current_destroy_mu = lambda: (0.1, False, "base")
        engine._effective_destroy_budget = lambda layer, mu: 1
        engine._apply_pair_rewards = lambda *_args, **_kwargs: False
        engine._sa_accept = lambda *_args, **_kwargs: (True, 1.0, 10.0)
        engine._record_reward = lambda *_args, **_kwargs: None
        engine._maybe_update_weights = lambda *_args, **_kwargs: None
        engine._update_layer_progress = lambda *_args, **_kwargs: None
        engine._update_failure_state = lambda *_args, **_kwargs: None
        engine._update_exact_cache_funnel = lambda *_args, **_kwargs: None
        engine._weight_snapshot = lambda *_args, **_kwargs: {}
        engine._current_layer_cooldown_remaining = lambda *_args, **_kwargs: 0
        engine._current_failure_cooldown_remaining = lambda *_args, **_kwargs: 0
        engine._generate_x_candidate_pool = lambda *_args, **_kwargs: {
            "rows": [],
            "selected": {
                "destroy_operator": "x_destroy",
                "repair_operator": "x_repair",
                "fallback_used": False,
                "projection_mode": "",
                "projection_repaired_subtask_count": 0,
                "candidate_eval": candidate_eval,
                "candidate_payload": {"config": candidate_config},
                "F_raw": float(candidate_eval.F_raw),
                "F_cal": float(candidate_eval.F_cal),
                "candidate_rank": 1,
            },
            "penalized_pairs": [],
            "coverage_hard_reject_count": 0,
            "target_size": 1,
            "generated_count": 1,
            "unique_count": 1,
            "exact_count": 1,
            "attempt_count": 1,
        }

        result = ResourceTimeALNSEngine.run(engine)
        self.assertEqual(result, 100.0)
        self.assertEqual(engine.best_validated.makespan, 100.0)
        self.assertEqual(opt.best, "best_snapshot")
        self.assertTrue(restore_calls)
        self.assertEqual(opt.iter_log[0]["candidate_hard_reject_reason"], "unassigned_robot_task_hard_reject")
        self.assertFalse(opt.iter_log[0]["local_accept"])

    def test_engine_accepts_reentry_repaired_candidate(self):
        import random

        base_config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={1: ResourceSubtask(1, 1, ("1:10:0",), 0, 0, [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)], ("st_1",))},
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        repaired_config = base_config.clone()
        repaired_config.subtasks[1].station_rank = 2
        base_eval = UpperEvalResult(Sx=1.0, Sy=1.0, Sz=1.0, F_raw=100.0, F_cal=100.0)
        repaired_eval = UpperEvalResult(Sx=0.8, Sy=0.8, Sz=0.8, F_raw=95.0, F_cal=95.0)

        cfg = SimpleNamespace(
            max_iters=1,
            resource_sa_cooling=0.95,
            resource_sa_reheat_factor=1.25,
            resource_candidate_pool_log=False,
            resource_stop_if_best_z_no_change_rounds=50,
            resource_conflict_local_reentry_enabled=True,
        )
        opt = SimpleNamespace(
            cfg=cfg,
            best="best_snapshot",
            work="best_snapshot",
            work_z=100.0,
            iter_log=[],
            candidate_iter_log=[],
            layer_runtime_sec_by_name={"X": 0.0, "Y": 0.0, "Z": 0.0, "U": 0.0},
            layer_trial_count_by_name={"X": 0.0, "Y": 0.0, "Z": 0.0},
            global_eval_count=0,
            stop_reason="",
            run_total_time_sec=0.0,
            _runtime_elapsed_sec=lambda: 0.0,
            _clear_z_detour_cache=lambda: None,
            _write_logs=lambda: None,
            restore_snapshot=lambda snap: None,
            operator_stats={},
        )
        engine = ResourceTimeALNSEngine.__new__(ResourceTimeALNSEngine)
        engine.opt = opt
        engine.cfg = cfg
        engine.rng = random.Random(0)
        engine.validator = SimpleNamespace(
            validate=lambda config, iter_id: {
                "makespan": 88.0 if config.validation_signature() == repaired_config.validation_signature() else float("inf"),
                "snapshot": "repaired_snapshot" if config.validation_signature() == repaired_config.validation_signature() else None,
                "coverage_hard_reject": False,
                "hard_reject_reason": "" if config.validation_signature() == repaired_config.validation_signature() else "unassigned_robot_task_hard_reject",
                "conflict_summary": {"failed_subtask_ids": [1], "station_ids": [0], "stack_ids": [100], "time_bucket_ids": [0], "resource_type": "robot_capacity"},
                "unmet_sku_total": 0,
                "unassigned_robot_task_count": 0 if config.validation_signature() == repaired_config.validation_signature() else 1,
                "unassigned_robot_tasks": [] if config.validation_signature() == repaired_config.validation_signature() else [{"task_id": 1, "subtask_id": 1}],
                "lkh_call_count": 1,
            }
        )
        engine.scorer = SimpleNamespace(
            evaluate=lambda config, **kwargs: repaired_eval if config.validation_signature() == repaired_config.validation_signature() else base_eval,
            update_with_validation=lambda *_args, **_kwargs: None,
        )
        engine.current_config = base_config.clone()
        engine.current_eval = base_eval
        engine.best_validated = ValidatedIncumbent(config=base_config.clone(), makespan=100.0, iter_id=0, snapshot="best_snapshot")
        engine.last_validated_config = base_config.clone()
        engine.last_validated_signature = base_config.validation_signature()
        engine.last_validation_iter = 0
        engine.last_validation_f_raw = 100.0
        engine.recent_validated_makespans = [100.0]
        engine.temperature = 10.0
        engine.layer_ema_improve = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.layer_stagnation = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        engine.layer_exec_since_update = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_last_update_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_failure_cooldown_until_iter = {"X": 0, "Y": 0, "Z": 0}
        engine.layer_dynamic_multiplier = {"X": 1.0, "Y": 1.0, "Z": 1.0}
        engine.consecutive_fail_count = {"X": 0, "Y": 0, "Z": 0}
        engine.last_selected_layer = ""
        engine.no_improve_rounds = 0.0
        engine.no_best_z_change_rounds = 0.0
        engine.validated_best_no_change_rounds = 0
        engine.best_f_raw = 100.0
        engine.consecutive_exact_cache_hit_count = 0
        engine.adaptive_destroy_bonus = 0.0
        engine.coverage_hard_reject_count = 0
        engine.x_failure_decapitation_count = 0
        engine.lkh_call_count = 0
        engine.lkh_budget_consumed_by_rollback = 0
        engine.operator_arms = {"X": {"destroy": {}, "repair": {}}, "Y": {"destroy": {}, "repair": {}}, "Z": {"destroy": {}, "repair": {}}}
        engine.action_signature_history = {"X": [], "Y": [], "Z": []}
        engine.action_signature_seen = {"X": set(), "Y": set(), "Z": set()}
        engine.joint_colocated_sort_postprocess_stats = {
            "triggered": 0.0, "candidate_groups": 0.0, "submitted": 0.0, "applied": 0.0, "makespan_improvement": 0.0,
            "rejected_capacity": 0.0, "rejected_interval_illegal": 0.0, "rejected_noise": 0.0, "rejected_eval_not_better": 0.0,
            "rejected_validation": 0.0, "rejected_target_conflict": 0.0,
        }
        engine._refresh_operator_stats_payload = lambda: None
        engine._accumulate_joint_postprocess_stats = lambda *_args, **_kwargs: None
        engine._select_layer = lambda iter_id: ("X", False)
        engine._current_destroy_mu = lambda: (0.1, False, "base")
        engine._effective_destroy_budget = lambda layer, mu: 1
        engine._apply_pair_rewards = lambda *_args, **_kwargs: False
        engine._sa_accept = lambda *_args, **_kwargs: (True, 1.0, 10.0)
        engine._record_reward = lambda *_args, **_kwargs: None
        engine._maybe_update_weights = lambda *_args, **_kwargs: None
        engine._update_layer_progress = lambda *_args, **_kwargs: None
        engine._update_failure_state = lambda *_args, **_kwargs: None
        engine._update_exact_cache_funnel = lambda *_args, **_kwargs: None
        engine._weight_snapshot = lambda *_args, **_kwargs: {}
        engine._current_layer_cooldown_remaining = lambda *_args, **_kwargs: 0
        engine._current_failure_cooldown_remaining = lambda *_args, **_kwargs: 0
        engine._attempt_constrained_reentry = lambda *args, **kwargs: (repaired_config.clone(), {
            "makespan": 88.0,
            "snapshot": "repaired_snapshot",
            "coverage_hard_reject": False,
            "hard_reject_reason": "",
            "conflict_summary": {},
            "unmet_sku_total": 0,
            "unassigned_robot_task_count": 0,
            "unassigned_robot_tasks": [],
            "lkh_call_count": 1,
        })
        engine._generate_x_candidate_pool = lambda *_args, **_kwargs: {
            "rows": [],
            "selected": {
                "destroy_operator": "x_destroy",
                "repair_operator": "x_repair",
                "fallback_used": False,
                "projection_mode": "",
                "projection_repaired_subtask_count": 0,
                "candidate_eval": base_eval,
                "candidate_payload": {"config": base_config.clone()},
                "F_raw": float(base_eval.F_raw),
                "F_cal": float(base_eval.F_cal),
                "candidate_rank": 1,
            },
            "penalized_pairs": [],
            "coverage_hard_reject_count": 0,
            "target_size": 1,
            "generated_count": 1,
            "unique_count": 1,
            "exact_count": 1,
            "attempt_count": 1,
        }
        result = ResourceTimeALNSEngine.run(engine)
        self.assertEqual(result, 88.0)
        self.assertEqual(engine.best_validated.makespan, 88.0)
        self.assertEqual(opt.best, "repaired_snapshot")

    def test_engine_invalid_initial_incumbent_is_not_treated_as_validated_best(self):
        base_config = ResourceConfig(
            work_units={"1:10:0": WorkUnitInfo("1:10:0", 1, 10, 0, 1)},
            subtasks={
                1: ResourceSubtask(
                    1,
                    1,
                    ("1:10:0",),
                    0,
                    0,
                    [ZTaskDescriptor(1, 100, "FLIP", (100,), (100,), (), None)],
                    ("st_1",),
                )
            },
            capacity_limits={1: 2},
            next_subtask_id=2,
            next_task_id=2,
        ).rebuild_indices()
        opt = SimpleNamespace(
            cfg=SimpleNamespace(
                seed=42,
                resource_sa_init_temp=20.0,
            ),
            best=SimpleNamespace(z=56.0),
            work=None,
            work_z=56.0,
            candidate_iter_log=[],
            stop_reason="",
        )
        fake_eval = UpperEvalResult(Sx=1.0, Sy=1.0, Sz=1.0, F_raw=100.0, F_cal=100.0)
        with patch("Gurobi.resource_time_alns.engine.build_initial_resource_config", return_value=base_config.clone()), \
             patch.object(ResourceValidator, "validate", return_value={
                 "makespan": float("inf"),
                 "snapshot": None,
                 "coverage_hard_reject": False,
                 "hard_reject_reason": "unassigned_robot_task_hard_reject",
                 "unmet_sku_total": 0,
                 "unassigned_robot_task_count": 1,
                 "unassigned_robot_tasks": [{"task_id": 1, "subtask_id": 1}],
                 "lkh_call_count": 1,
             }), \
             patch.object(ResourceSurrogateScorer, "evaluate", return_value=fake_eval):
            engine = ResourceTimeALNSEngine(opt)
        self.assertEqual(engine.best_validated.makespan, float("inf"))
        self.assertIsNone(engine.best_validated.snapshot)
        self.assertEqual(engine.recent_validated_makespans, [])
        self.assertEqual(engine.current_eval.metadata.get("initial_hard_reject_reason"), "unassigned_robot_task_hard_reject")

    def test_soft_greedy_pick_never_leaves_topk(self):
        import random

        cfg = SimpleNamespace(resource_soft_greedy_topk=3, resource_soft_greedy_noise=0.05)
        scored = [(1.0, "a"), (2.0, "b"), (3.0, "c"), (4.0, "d"), (5.0, "e")]
        picks = {
            pick_soft_greedy_min(random.Random(seed), scored, cfg, score_getter=lambda item: item[0])[1]
            for seed in range(50)
        }
        self.assertTrue(picks.issubset({"a", "b", "c"}))
        self.assertNotIn("d", picks)
        self.assertNotIn("e", picks)


if __name__ == "__main__":
    unittest.main()
