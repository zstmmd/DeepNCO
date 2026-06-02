import unittest

from BPC.branching import BranchAndPriceTree, first_fractional_assignment
from BPC.master import RestrictedMasterProblem
from BPC.models import BPCCertificate, BPCRouteColumn, BPCRouteTask
from BPC.pricing import LabelSettingPricer, validate_column


class BPCFrameworkTests(unittest.TestCase):
    def _tasks(self):
        return [
            BPCRouteTask(0, 10, 1, 1, 7, 0, (1.0, 0.0), (2.0, 0.0), 1.0, 1),
            BPCRouteTask(1, 11, 2, 1, 8, 0, (2.0, 0.0), (3.0, 0.0), 1.0, 1),
        ]

    def test_pricing_generates_capacity_and_time_feasible_columns(self):
        tasks = self._tasks()
        pricer = LabelSettingPricer(robot_id=0, start_xy=(0.0, 0.0), robot_capacity=2)
        result = pricer.price(tasks, dual_task_cover={0: 100.0, 1: 100.0}, time_limit_sec=5.0, max_labels=1000)
        self.assertTrue(result.exact)
        self.assertGreater(len(result.columns), 0)
        by_key = {task.task_key: task for task in tasks}
        ok, reason = validate_column(result.columns[0], by_key, robot_capacity=2)
        self.assertTrue(ok, reason)

    def test_master_active_task_coverage(self):
        tasks = self._tasks()
        columns = [
            BPCRouteColumn(0, 0, (0,), (0,), {0: 2.0}, 3.0, 2.0, 1.0),
            BPCRouteColumn(1, 0, (1,), (1,), {1: 3.0}, 4.0, 3.0, 1.0),
        ]
        rmp = RestrictedMasterProblem(tasks, columns)
        coverage = rmp.active_task_coverage({0: 1.0, 1: 1.0})
        self.assertEqual(coverage[0], 1.0)
        self.assertEqual(coverage[1], 1.0)

    def test_branch_pair_constraints_are_complementary(self):
        tree = BranchAndPriceTree()
        root = tree.pop()
        left, right = tree.branch_on_task_robot(root, task_key=3, robot_id=1)
        self.assertEqual(left.fixed_task_robot[3], 1)
        self.assertTrue(right.forbidden_task_robot[(3, 1)])
        self.assertEqual(tree.open_count, 2)

    def test_first_fractional_assignment(self):
        self.assertEqual(first_fractional_assignment({(1, 0): 0.0, (2, 0): 0.5}), (2, 0))
        self.assertIsNone(first_fractional_assignment({(1, 0): 1.0, (2, 0): 0.0}))

    def test_exact_certificate_requires_closed_gap_and_exact_pricing(self):
        cert = BPCCertificate.evaluate(
            incumbent_found=True,
            integer_solution=True,
            all_nodes_closed=True,
            pricing_exact=False,
            no_negative_reduced_cost=True,
            open_nodes=0,
            upper_bound=10.0,
            lower_bound=10.0,
        )
        self.assertFalse(cert.exact)
        self.assertEqual(cert.reason, "pricing_not_exact")
        cert = BPCCertificate.evaluate(
            incumbent_found=True,
            integer_solution=True,
            all_nodes_closed=True,
            pricing_exact=True,
            no_negative_reduced_cost=True,
            open_nodes=0,
            upper_bound=10.0,
            lower_bound=10.0,
        )
        self.assertTrue(cert.exact)


if __name__ == "__main__":
    unittest.main()
