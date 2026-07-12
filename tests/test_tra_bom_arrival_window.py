import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import sys
import types

if "ortools.constraint_solver" not in sys.modules:
    ortools_mod = types.ModuleType("ortools")
    constraint_solver_mod = types.ModuleType("ortools.constraint_solver")
    routing_enums_mod = types.ModuleType("ortools.constraint_solver.routing_enums_pb2")
    pywrapcp_mod = types.ModuleType("ortools.constraint_solver.pywrapcp")
    routing_enums_mod.FirstSolutionStrategy = SimpleNamespace(PARALLEL_CHEAPEST_INSERTION=0)
    routing_enums_mod.LocalSearchMetaheuristic = SimpleNamespace(GUIDED_LOCAL_SEARCH=0)
    pywrapcp_mod.RoutingIndexManager = object
    pywrapcp_mod.RoutingModel = object
    pywrapcp_mod.DefaultRoutingSearchParameters = lambda: SimpleNamespace()
    sys.modules["ortools"] = ortools_mod
    sys.modules["ortools.constraint_solver"] = constraint_solver_mod
    sys.modules["ortools.constraint_solver.routing_enums_pb2"] = routing_enums_mod
    sys.modules["ortools.constraint_solver.pywrapcp"] = pywrapcp_mod

from config.ofs_config import OFSConfig
from Gurobi.resource_time_alns.fixgurobi_evaluator import FixGurobiEvaluator
from Gurobi.resource_time_alns.validator import ResourceValidator
from Gurobi.sp2 import SP2_Station_Assigner
from Gurobi.tra import TRAOptimizer, TRARunConfig


def _make_task(task_id: int, subtask_id: int, arrival: float, robot_id: int = 0):
    return SimpleNamespace(
        task_id=int(task_id),
        sub_task_id=int(subtask_id),
        arrival_time_at_station=float(arrival),
        robot_id=int(robot_id),
        order_time_window_ub_sec=0.0,
    )


def _make_subtask(subtask_id: int, order_id: int, arrivals, unique_sku_count: int = 0, kitting_span_limit_sec: float = 0.0):
    order = SimpleNamespace(
        order_id=int(order_id),
        unique_sku_count=int(unique_sku_count),
        kitting_span_limit_sec=float(kitting_span_limit_sec),
    )
    tasks = [_make_task(task_id=subtask_id * 10 + idx, subtask_id=subtask_id, arrival=arrival) for idx, arrival in enumerate(arrivals, start=1)]
    return SimpleNamespace(
        id=int(subtask_id),
        parent_order=order,
        execution_tasks=tasks,
        assigned_robot_id=0 if tasks else -1,
        assigned_station_id=-1,
        station_sequence_rank=-1,
        estimated_process_start_time=0.0,
        sku_list=[object()],
        kitting_span_limit_sec=float(kitting_span_limit_sec),
        order_anchor_start_sec=0.0,
        order_time_window_lb_sec=0.0,
        order_time_window_ub_sec=0.0,
    )


class FakeConfig:
    def coverage_summary(self, _tote_map):
        return {
            "coverage_ok": True,
            "unmet_sku_total": 0,
            "unmet_subtask_count": 0,
            "subtasks": [],
        }


class TRABomArrivalWindowTests(unittest.TestCase):
    def _build_optimizer(self, window_sec: float, subtasks):
        opt = TRAOptimizer(TRARunConfig(scale="TEST", bom_arrival_window_sec=float(window_sec)))
        order_by_id = {}
        for st in subtasks:
            order = getattr(st, "parent_order", None)
            if order is not None:
                order_by_id[int(getattr(order, "order_id", -1))] = order
        opt.problem = SimpleNamespace(
            subtask_list=list(subtasks),
            station_list=[],
            robot_list=[],
            order_list=list(order_by_id.values()),
            task_list=[task for st in subtasks for task in getattr(st, "execution_tasks", [])],
            global_makespan=0.0,
        )
        return opt

    def test_effective_bom_arrival_window_scales_with_unique_sku_count(self):
        self.assertAlmostEqual(float(OFSConfig.effective_bom_arrival_window_sec(60.0, 3)), 60.0)
        self.assertAlmostEqual(float(OFSConfig.effective_bom_arrival_window_sec(60.0, 8)), 120.0)

    def test_bom_arrival_window_passes_within_limit(self):
        opt = self._build_optimizer(
            60.0,
            [
                _make_subtask(1, 100, [10.0]),
                _make_subtask(2, 100, [65.0]),
            ],
        )
        result = opt._evaluate_bom_arrival_window()
        self.assertTrue(result["enabled"])
        self.assertTrue(result["feasible"])
        self.assertEqual(int(result["violating_order_count"]), 0)

    def test_bom_arrival_window_rejects_exceeding_limit(self):
        opt = self._build_optimizer(
            60.0,
            [
                _make_subtask(1, 100, [10.0], unique_sku_count=3),
                _make_subtask(2, 100, [71.0], unique_sku_count=3),
            ],
        )
        result = opt._evaluate_bom_arrival_window()
        self.assertFalse(result["feasible"])
        self.assertEqual(int(result["violating_order_count"]), 1)
        self.assertAlmostEqual(float(result["violations"][0]["span_sec"]), 61.0)
        self.assertAlmostEqual(float(result["violations"][0]["effective_window_sec"]), 60.0)

    def test_bom_arrival_window_allows_large_bom_dynamic_window(self):
        opt = self._build_optimizer(
            60.0,
            [
                _make_subtask(1, 100, [10.0], unique_sku_count=8),
                _make_subtask(2, 100, [123.0], unique_sku_count=8),
            ],
        )
        result = opt._evaluate_bom_arrival_window()
        self.assertTrue(result["feasible"])
        self.assertEqual(int(result["violating_order_count"]), 0)
        self.assertAlmostEqual(float(result["orders"][0]["span_sec"]), 113.0)
        self.assertAlmostEqual(float(result["orders"][0]["effective_window_sec"]), 120.0)

    def test_bom_arrival_window_disabled_when_nonpositive(self):
        opt = self._build_optimizer(
            0.0,
            [
                _make_subtask(1, 100, [10.0]),
                _make_subtask(2, 100, [500.0]),
            ],
        )
        result = opt._evaluate_bom_arrival_window()
        self.assertFalse(result["enabled"])
        self.assertTrue(result["feasible"])

    def test_bom_arrival_window_single_subtask_always_passes(self):
        opt = self._build_optimizer(60.0, [_make_subtask(1, 100, [200.0])])
        result = opt._evaluate_bom_arrival_window()
        self.assertTrue(result["feasible"])
        self.assertEqual(int(result["violating_order_count"]), 0)

    def test_bom_arrival_window_missing_arrivals_does_not_fail(self):
        opt = self._build_optimizer(
            60.0,
            [
                _make_subtask(1, 100, []),
                _make_subtask(2, 100, [25.0]),
            ],
        )
        result = opt._evaluate_bom_arrival_window()
        self.assertTrue(result["feasible"])
        self.assertEqual(int(result["violating_order_count"]), 0)

    def test_validator_rejects_candidate_when_bom_arrival_window_violated(self):
        problem = SimpleNamespace(subtask_list=[], id_to_tote={})
        opt = SimpleNamespace(
            problem=problem,
            cfg=SimpleNamespace(bom_arrival_window_sec=60.0),
            _run_sp4=lambda: None,
            evaluate=lambda: 123.0,
            _evaluate_order_time_window_metrics=lambda: {},
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda makespan, iter_id, lightweight=True: {"makespan": makespan, "iter_id": iter_id},
        )
        opt._evaluate_bom_arrival_window = lambda: TRAOptimizer._evaluate_bom_arrival_window(
            SimpleNamespace(
                cfg=opt.cfg,
                problem=opt.problem,
                _subtask_arrival_from_tasks=lambda: {1: 10.0, 2: 75.0},
                _effective_bom_arrival_window_for_order=lambda order: OFSConfig.effective_bom_arrival_window_sec(
                    getattr(opt.cfg, "bom_arrival_window_sec", 0.0), getattr(order, "unique_sku_count", 0)
                ),
            )
        )
        validator = ResourceValidator(opt)
        validator.materialize = lambda config: setattr(
            opt,
            "problem",
            SimpleNamespace(
                subtask_list=[_make_subtask(1, 100, [10.0], unique_sku_count=3), _make_subtask(2, 100, [75.0], unique_sku_count=3)],
                order_list=[SimpleNamespace(order_id=100, unique_sku_count=3)],
                id_to_tote={},
            ),
        )
        result = validator.validate(FakeConfig(), iter_id=1)
        self.assertEqual(str(result["hard_reject_reason"]), "bom_arrival_window_hard_reject")
        self.assertIsNone(result["snapshot"])
        self.assertEqual(int(result["bom_arrival_window_violating_order_count"]), 1)

    def test_validator_allows_large_bom_candidate_with_dynamic_window(self):
        problem = SimpleNamespace(subtask_list=[], id_to_tote={})
        opt = SimpleNamespace(
            problem=problem,
            cfg=SimpleNamespace(bom_arrival_window_sec=60.0),
            _run_sp4=lambda: None,
            evaluate=lambda: 123.0,
            _evaluate_order_time_window_metrics=lambda: {},
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda makespan, iter_id, lightweight=True: {"makespan": makespan, "iter_id": iter_id},
        )
        opt._evaluate_bom_arrival_window = lambda: TRAOptimizer._evaluate_bom_arrival_window(
            SimpleNamespace(
                cfg=opt.cfg,
                problem=opt.problem,
                _subtask_arrival_from_tasks=lambda: {1: 10.0, 2: 123.0},
                _effective_bom_arrival_window_for_order=lambda order: OFSConfig.effective_bom_arrival_window_sec(
                    getattr(opt.cfg, "bom_arrival_window_sec", 0.0), getattr(order, "unique_sku_count", 0)
                ),
            )
        )
        validator = ResourceValidator(opt)
        validator.materialize = lambda config: setattr(
            opt,
            "problem",
            SimpleNamespace(
                subtask_list=[_make_subtask(1, 100, [10.0], unique_sku_count=8), _make_subtask(2, 100, [123.0], unique_sku_count=8)],
                order_list=[SimpleNamespace(order_id=100, unique_sku_count=8)],
                id_to_tote={},
            ),
        )
        result = validator.validate(FakeConfig(), iter_id=1)
        self.assertEqual(str(result["hard_reject_reason"]), "")
        self.assertIsNotNone(result["snapshot"])
        self.assertEqual(int(result["bom_arrival_window_violating_order_count"]), 0)

    def test_validator_allows_candidate_when_bom_arrival_window_disabled(self):
        problem = SimpleNamespace(subtask_list=[], id_to_tote={})
        opt = SimpleNamespace(
            problem=problem,
            cfg=SimpleNamespace(bom_arrival_window_sec=0.0),
            _run_sp4=lambda: None,
            evaluate=lambda: 123.0,
            _evaluate_order_time_window_metrics=lambda: {},
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda makespan, iter_id, lightweight=True: {"makespan": makespan, "iter_id": iter_id},
        )
        opt._evaluate_bom_arrival_window = lambda: TRAOptimizer._evaluate_bom_arrival_window(
            SimpleNamespace(
                cfg=opt.cfg,
                problem=opt.problem,
                _subtask_arrival_from_tasks=lambda: {1: 10.0, 2: 500.0},
                _effective_bom_arrival_window_for_order=lambda order: OFSConfig.effective_bom_arrival_window_sec(
                    getattr(opt.cfg, "bom_arrival_window_sec", 0.0), getattr(order, "unique_sku_count", 0)
                ),
            )
        )
        validator = ResourceValidator(opt)
        validator.materialize = lambda config: setattr(
            opt,
            "problem",
            SimpleNamespace(
                subtask_list=[_make_subtask(1, 100, [10.0]), _make_subtask(2, 100, [500.0])],
                order_list=[SimpleNamespace(order_id=100, unique_sku_count=0)],
                id_to_tote={},
            ),
        )
        result = validator.validate(FakeConfig(), iter_id=1)
        self.assertEqual(str(result["hard_reject_reason"]), "")
        self.assertIsNotNone(result["snapshot"])

    def test_validator_rejects_candidate_when_kitting_span_violated(self):
        problem = SimpleNamespace(subtask_list=[], id_to_tote={})
        opt = SimpleNamespace(
            problem=problem,
            cfg=SimpleNamespace(bom_arrival_window_sec=60.0),
            _run_sp4=lambda: None,
            evaluate=lambda: 123.0,
            _evaluate_bom_arrival_window=lambda: {"feasible": True, "violating_order_count": 0, "violations": []},
            _evaluate_order_time_window_metrics=lambda: {"span_overrun_total": 5.0, "orders": [{"order_id": 100, "span_overrun": 5.0}]},
            _harvest_station_start_times=lambda: None,
            _update_beta_from_station=lambda: None,
            snapshot=lambda makespan, iter_id, lightweight=True: {"makespan": makespan, "iter_id": iter_id},
        )
        validator = ResourceValidator(opt)
        validator.materialize = lambda config: setattr(
            opt,
            "problem",
            SimpleNamespace(subtask_list=[_make_subtask(1, 100, [10.0], kitting_span_limit_sec=30.0)], order_list=[SimpleNamespace(order_id=100)], id_to_tote={}),
        )
        result = validator.validate(FakeConfig(), iter_id=1)
        self.assertEqual(str(result["hard_reject_reason"]), "kitting_span_hard_reject")
        self.assertIsNone(result["snapshot"])

    def test_validator_cache_cfg_defaults_do_not_require_cfg_fields(self):
        validator = ResourceValidator(SimpleNamespace(cfg=SimpleNamespace()))
        self.assertFalse(validator._validation_cache_enabled())
        self.assertEqual(validator._validation_cache_size(), 1024)

    def test_fixgurobi_warm_start_fallback_rejects_time_window_overrun_first(self):
        result = SimpleNamespace(
            status="WARM_START_FALLBACK",
            objective=100.0,
            diagnostics={
                "warm_start_model_cmax": 100.0,
                "warm_start_total_span_overrun": 2.0,
                "warm_start_total_deadline_overrun": 0.0,
            },
        )
        self.assertEqual(FixGurobiEvaluator._objective_from_result(result), float("inf"))

    def test_sp2_initial_heuristic_keeps_same_order_task_within_window(self):
        station0 = SimpleNamespace(id=0)
        station1 = SimpleNamespace(id=1)
        unrelated = _make_subtask(1, 999, [], kitting_span_limit_sec=0.0)
        unrelated.sku_list = [object(), object(), object(), object()]
        first = _make_subtask(2, 100, [], kitting_span_limit_sec=10.0)
        first.sku_list = [object()]
        second = _make_subtask(3, 100, [], kitting_span_limit_sec=10.0)
        second.sku_list = [object()]
        problem = SimpleNamespace(station_list=[station0, station1], subtask_list=[unrelated, first, second])
        solver = SP2_Station_Assigner(problem)
        solver.solve_initial_heuristic()
        self.assertEqual(int(first.assigned_station_id), 1)
        self.assertEqual(int(second.assigned_station_id), 1)
        self.assertAlmostEqual(float(first.order_anchor_start_sec), 0.0)
        self.assertAlmostEqual(float(second.order_time_window_ub_sec), 10.0)
        self.assertLessEqual(float(second.estimated_process_start_time), float(second.order_time_window_ub_sec))

    def test_sp4_order_window_latest_by_task_uses_kitting_window_upper_bound(self):
        st = _make_subtask(1, 100, [0.0], kitting_span_limit_sec=30.0)
        st.estimated_process_start_time = 50.0
        st.order_time_window_ub_sec = 40.0
        for task in st.execution_tasks:
            task.order_time_window_ub_sec = 45.0
        opt = TRAOptimizer(TRARunConfig(scale="TEST"))
        opt.problem = SimpleNamespace(subtask_list=[st])
        latest_by_task = opt._build_sp4_order_window_latest_by_task()
        self.assertEqual(latest_by_task[int(st.execution_tasks[0].task_id)], 40.0)

    def test_initialize_marks_initial_incumbent_infeasible_when_bom_arrival_window_violated(self):
        opt = TRAOptimizer(TRARunConfig(scale="TEST", seed=42, bom_arrival_window_sec=60.0))
        fake_problem = SimpleNamespace(
            subtask_list=[_make_subtask(1, 100, [10.0], unique_sku_count=3), _make_subtask(2, 100, [75.0], unique_sku_count=3)],
            station_list=[],
            robot_list=[],
            order_list=[SimpleNamespace(order_id=100, unique_sku_count=3)],
            task_list=[],
            global_makespan=0.0,
        )
        tmp_dir = tempfile.mkdtemp()

        opt._set_seed = lambda seed: None
        opt._ensure_log_dir = lambda: tmp_dir
        opt._rebuild_solvers = lambda: None
        opt._run_sp1 = lambda: None
        opt._run_sp2_initial = lambda: None
        opt._run_sp3 = lambda: None
        opt._run_sp4 = lambda: None
        opt.evaluate = lambda: 123.0
        opt._compute_solution_coverage = lambda: {"coverage_ok": True, "unmet_sku_total": 0, "unmet_subtask_count": 0, "subtasks": []}
        opt._harvest_station_start_times = lambda: None
        opt._update_beta_from_station = lambda: None
        opt.snapshot = lambda z, iter_id, lightweight=True: SimpleNamespace(z=float(z), iter_id=int(iter_id), seed=int(opt.cfg.seed))
        opt._refresh_runtime_cache = lambda z: None

        with mock.patch("Gurobi.tra.CreateOFSProblem.generate_problem_by_scale", return_value=fake_problem), mock.patch(
            "Gurobi.tra.init_resource_time_runtime_state", return_value=None
        ):
            opt.initialize()

        self.assertTrue(opt.best.z == float("inf"))
        self.assertAlmostEqual(float(opt.work.z), 123.0)

    def test_initialize_marks_initial_incumbent_infeasible_when_kitting_span_violated(self):
        opt = TRAOptimizer(TRARunConfig(scale="TEST", seed=42, bom_arrival_window_sec=0.0))
        fake_problem = SimpleNamespace(
            subtask_list=[
                _make_subtask(1, 100, [10.0], kitting_span_limit_sec=30.0),
                _make_subtask(2, 100, [75.0], kitting_span_limit_sec=30.0),
            ],
            station_list=[],
            robot_list=[],
            order_list=[SimpleNamespace(order_id=100, unique_sku_count=3, kitting_span_limit_sec=30.0)],
            task_list=[],
            global_makespan=0.0,
        )
        tmp_dir = tempfile.mkdtemp()

        opt._set_seed = lambda seed: None
        opt._ensure_log_dir = lambda: tmp_dir
        opt._rebuild_solvers = lambda: None
        opt._run_sp1 = lambda: None
        opt._run_sp2_initial = lambda: None
        opt._run_sp3 = lambda: None
        opt._run_sp4 = lambda: None
        opt.evaluate = lambda: 123.0
        opt._compute_solution_coverage = lambda: {"coverage_ok": True, "unmet_sku_total": 0, "unmet_subtask_count": 0, "subtasks": []}
        opt._harvest_station_start_times = lambda: None
        opt._update_beta_from_station = lambda: None
        opt.snapshot = lambda z, iter_id, lightweight=True: SimpleNamespace(z=float(z), iter_id=int(iter_id), seed=int(opt.cfg.seed))
        opt._refresh_runtime_cache = lambda z: None

        with mock.patch("Gurobi.tra.CreateOFSProblem.generate_problem_by_scale", return_value=fake_problem), mock.patch(
            "Gurobi.tra.init_resource_time_runtime_state", return_value=None
        ):
            opt.initialize()

        self.assertTrue(opt.best.z == float("inf"))
        self.assertEqual(str(opt.iter_log[-1].get("candidate_hard_reject_reason", "")), "kitting_span_hard_reject")


if __name__ == "__main__":
    unittest.main()
