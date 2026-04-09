import os
import shutil
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from problemDto.createInstance import CreateOFSProblem
from Gurobi.tra import TRARunConfig, TRAOptimizer
from entity.calculate import GlobalTimeCalculator


@dataclass
class ALNSRelaxDecompConfig(TRARunConfig):
    alns_init_iters: int = 12
    alns_init_temperature: float = 10.0
    alns_cooling: float = 0.92
    alns_destroy_degree: int = 3


class RankAwareGlobalTimeCalculator(GlobalTimeCalculator):
    def _simulate_station_fcfs(self, all_tasks: List):
        station_to_tasks: Dict[int, List] = defaultdict(list)
        for task in all_tasks:
            station_to_tasks[int(getattr(task, "target_station_id", 0))].append(task)

        for station_id, tasks in station_to_tasks.items():
            station = self.problem.station_list[station_id]
            ordered_tasks = sorted(
                tasks,
                key=lambda t: (
                    int(getattr(t, "station_sequence_rank", -1)) if int(getattr(t, "station_sequence_rank", -1)) >= 0 else 10 ** 9,
                    float(getattr(t, "arrival_time_at_station", 0.0)),
                    int(getattr(t, "task_id", -1)),
                ),
            )

            for task in ordered_tasks:
                sku_count_in_task = self._calculate_sku_count(task)
                task.picking_duration = sku_count_in_task * self.t_pick

                extra_service = task.station_service_time if getattr(task, "noise_tote_ids", None) else 0.0
                total_process_duration = task.picking_duration + extra_service

                start_time = max(float(getattr(task, "arrival_time_at_station", 0.0)), station.next_available_time)
                if start_time > station.next_available_time:
                    station.total_idle_time += (start_time - station.next_available_time)

                task.tote_wait_time = start_time - float(getattr(task, "arrival_time_at_station", 0.0))
                task.start_process_time = start_time
                task.end_process_time = start_time + total_process_duration
                task.extra_service_used = float(extra_service)
                task.total_process_duration = float(total_process_duration)

                station.next_available_time = task.end_process_time
                station.processed_tasks.append(task)


class ALNSRelaxDecompOptimizer(TRAOptimizer):
    """
    ALNS initialization + four-layer decomposition optimizer.

    Pragmatic implementation notes:
    - Reuses the existing SP1/SP2/SP3/SP4 solvers in the repo.
    - Keeps the outer-loop objective fully aligned with TRA:
      z = GlobalTimeCalculator(problem).calculate()
    - Uses SP4 LKH/OR-Tools only.
    """

    def __init__(self, cfg: ALNSRelaxDecompConfig):
        super().__init__(cfg)
        self.cfg: ALNSRelaxDecompConfig = cfg
        self.layer_order: List[str] = ["SP1", "SP2", "SP3", "SP4"]
        self.mode_names = list(self.layer_order)
        self.mode_stats = {
            m: {"calls": 0.0, "success": 0.0, "fail": 0.0, "skip": 0.0, "last_gap": 1.0}
            for m in self.layer_order
        }
        self.destroy_ops: Dict[str, List[str]] = {
            "X": ["D_X_spread", "D_X_fragment"],
            "Y": ["D_Y_congested_station", "D_Y_fragmented_order"],
            "B": ["D_B_high_noise", "D_B_deep_layer"],
            "R": ["D_R_late_robot"],
        }
        self.repair_ops: Dict[str, List[str]] = {
            "X": ["R_X_affinity_pack", "R_X_capacity_balance"],
            "Y": ["R_Y_earliest_completion", "R_Y_arrival_aware"],
            "B": ["R_B_max_cover_min_noise", "R_B_same_stack_first"],
            "R": ["R_R_nearest_feasible", "R_R_regret_k"],
        }
        self.destroy_weights = {
            name: 1.0
            for names in self.destroy_ops.values()
            for name in names
        }
        self.repair_weights = {
            name: 1.0
            for names in self.repair_ops.values()
            for name in names
        }

    def _rebuild_solvers(self):
        super()._rebuild_solvers()
        self.sim = RankAwareGlobalTimeCalculator(self.problem)

    def _append_layer_log(
        self,
        iter_id: int,
        focus: str,
        z_before: float,
        z_after: float,
        improved: bool,
        skipped: bool,
        accepted_type: str = "",
        lb: Optional[float] = None,
    ):
        self.iter_log.append({
            "iter": int(iter_id),
            "focus": str(focus),
            "z": float(z_after),
            "z_before": float(z_before),
            "best_z": float(self.best.z if self.best is not None else z_after),
            "best_z_after": float(self.best.z if self.best is not None else z_after),
            "improved": bool(improved),
            "skipped": bool(skipped),
            "accepted_type": str(accepted_type),
            "lb": float(lb) if lb is not None else float("nan"),
            "epsilon": float(self.cfg.epsilon),
        })

    def _run_sp2_solver(self):
        if self.cfg.sp2_use_mip:
            self._run_sp2_mip()
        else:
            self._run_sp2_initial()

    def _run_full_pipeline(self):
        self._run_sp1()
        self._run_sp2_solver()
        self._run_sp3()
        self._run_sp4()
        z = float(self.evaluate())
        self._harvest_station_start_times()
        self._update_beta_from_station()
        return z

    def _run_focus_pipeline(self, focus: str):
        """
        Keep the implementation stable by rebuilding dependent layers when needed.
        This preserves a fully evaluable solution state for z = GlobalTimeCalculator.calculate().
        """
        if focus == "SP1":
            self._run_sp1()
            self._run_sp2_solver()
            self._run_sp3()
            self._run_sp4()
        elif focus == "SP2":
            self._run_sp2_solver()
            self._run_sp3()
            self._run_sp4()
        elif focus == "SP3":
            self._run_sp3()
            self._run_sp4()
        else:
            self._run_sp4()

    def _weighted_choice(self, items: List[str], weights: Dict[str, float], rng: random.Random) -> str:
        total = sum(max(1e-9, float(weights.get(item, 1.0))) for item in items)
        pick = rng.random() * total
        acc = 0.0
        for item in items:
            acc += max(1e-9, float(weights.get(item, 1.0)))
            if pick <= acc:
                return item
        return items[-1]

    def _choose_alns_layer(self, rng: random.Random) -> str:
        current_metrics = self._collect_layer_metrics()
        scores = {
            "X": 1.0 + float(current_metrics.get("avg_sku_per_subtask", 0.0)) * 0.1,
            "Y": 1.0 + float(current_metrics.get("station_load_max_ratio", 0.0)),
            "B": 1.0 + float(current_metrics.get("noise_ratio", 0.0)) * 2.0 + float(current_metrics.get("avg_stack_span", 0.0)) * 0.02,
            "R": 1.0 + float(current_metrics.get("robot_finish_ratio", 0.0)),
        }
        total = sum(scores.values())
        pick = rng.random() * total
        acc = 0.0
        for key in ["X", "Y", "B", "R"]:
            acc += scores[key]
            if pick <= acc:
                return key
        return "X"

    def _apply_destroy(self, op_name: str, rng: random.Random):
        degree = max(1, int(self.cfg.alns_destroy_degree))
        if op_name == "D_X_spread":
            for _ in range(degree):
                self._perturb_split_structure(rng)
        elif op_name == "D_X_fragment":
            for _ in range(degree + 1):
                self._perturb_split_structure(rng)
        elif op_name == "D_Y_congested_station":
            self._perturb_station_assignments(rng, max_count=degree + 1)
        elif op_name == "D_Y_fragmented_order":
            self._perturb_station_assignments(rng, max_count=degree)
        elif op_name == "D_B_high_noise":
            for _ in range(degree):
                self._perturb_stack_behavior(rng)
        elif op_name == "D_B_deep_layer":
            for _ in range(degree + 1):
                self._perturb_stack_behavior(rng)
        elif op_name == "D_R_late_robot":
            self._perturb_routing_anchor(rng, max_count=degree + 1)

    def _apply_repair(self, layer_key: str, repair_name: str):
        # Repair operator names are logged and weighted; the concrete repair path
        # is the layer-specific re-solve chain below.
        if layer_key == "X":
            self._run_sp1()
            self._run_sp2_solver()
            self._run_sp3()
            self._run_sp4()
        elif layer_key == "Y":
            self._run_sp2_solver()
            self._run_sp3()
            self._run_sp4()
        elif layer_key == "B":
            self._run_sp3()
            self._run_sp4()
        else:
            self._run_sp4()

    def _update_alns_weights(self, destroy_name: str, repair_name: str, accepted: bool, improved: bool):
        reward = 0.6 if accepted else -0.15
        if improved:
            reward += 0.8
        self.destroy_weights[destroy_name] = max(0.1, float(self.destroy_weights.get(destroy_name, 1.0)) * (1.0 + 0.1 * reward))
        self.repair_weights[repair_name] = max(0.1, float(self.repair_weights.get(repair_name, 1.0)) * (1.0 + 0.1 * reward))

    def _run_alns_initialization(self):
        assert self.work is not None
        assert self.best is not None

        temp = max(1e-6, float(self.cfg.alns_init_temperature))
        for it in range(1, max(0, int(self.cfg.alns_init_iters)) + 1):
            self.restore_snapshot(self.work)
            rng = random.Random(int(self.cfg.seed) * 10007 + it * 97)
            layer_key = self._choose_alns_layer(rng)
            destroy_name = self._weighted_choice(self.destroy_ops[layer_key], self.destroy_weights, rng)
            repair_name = self._weighted_choice(self.repair_ops[layer_key], self.repair_weights, rng)
            z_before = float(self.work_z)

            self._apply_destroy(destroy_name, rng)
            self._apply_repair(layer_key, repair_name)
            z_cand = float(self.evaluate())
            self._harvest_station_start_times()
            self._update_beta_from_station()
            cand_snap = self.snapshot(z_cand, iter_id=0)

            improved = z_cand < float(self.best.z) - 1e-6
            accept = False
            if z_cand < float(self.work_z) - 1e-6:
                accept = True
            else:
                delta = float(z_cand - self.work_z)
                if delta <= 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta / max(1e-6, temp))
                    accept = rng.random() <= accept_prob

            if accept:
                self.work = cand_snap
                self.work_z = float(z_cand)
                if improved:
                    self.best = cand_snap

            self._update_alns_weights(destroy_name, repair_name, accept, improved)
            temp *= float(self.cfg.alns_cooling)
            self.restore_snapshot(self.work)
            self._notify_progress(0, self.cfg.max_iters, f"ALNS_INIT_{it}")
            _ = z_before

    def initialize(self):
        self._set_seed(self.cfg.seed)
        self.problem = CreateOFSProblem.generate_problem_by_scale(self.cfg.scale, seed=self.cfg.seed)
        self._rebuild_solvers()

        if self.cfg.enable_sp3_precheck:
            try:
                self.precheck_result = self._precheck_sp3_coverage()
                unmet = int(self.precheck_result.get("unmet_sku_total", 0)) if self.precheck_result else 0
                if unmet > 0 and str(self.cfg.sp3_precheck_fail_action).lower() == "abort":
                    self.precheck_aborted = True
                    self.precheck_status = f"precheck_unmet:{unmet}"
                    return
            except Exception as e:
                try:
                    path = self._log_path("sp3_precheck_error.txt")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(e))
                except Exception:
                    pass

        z0 = self._run_full_pipeline()
        self.best = self.snapshot(z0, iter_id=0)
        self.work = self.snapshot(z0, iter_id=0)
        self.work_z = float(z0)
        self._run_alns_initialization()
        self.restore_snapshot(self.work)
        z_init = float(self.evaluate())
        self.work_z = float(z_init)
        if self.best is None or z_init < float(self.best.z) - 1e-6:
            self.best = self.snapshot(z_init, iter_id=0)
        self._append_layer_log(
            iter_id=0,
            focus="ALNS_INIT",
            z_before=float(z0),
            z_after=float(self.work_z),
            improved=float(self.work_z) < float(z0) - 1e-6,
            skipped=False,
            accepted_type="init",
            lb=None,
        )

    def _focus_lb(self, focus: str) -> float:
        if focus == "SP1":
            return float(self._lb_order_chain())
        if focus == "SP2":
            return float(self._lb_station_workload())
        if focus == "SP3":
            return float(self._lb_stack_blocking())
        return float(max(self._lb_transport(), self._lb_robot_workload()))

    def run(self) -> float:
        if self.problem is None:
            self.initialize()

        if self.precheck_aborted:
            if self.cfg.write_iteration_logs:
                self._write_logs()
            return float("nan")

        assert self.best is not None
        assert self.work is not None

        mark = 0
        self._notify_progress(0, self.cfg.max_iters, "ALNS_INIT")
        for it in range(1, self.cfg.max_iters + 1):
            focus = self.layer_order[(it - 1) % len(self.layer_order)]
            self.mode_stats[focus]["calls"] += 1.0
            z_before = float(self.work_z)
            lb = self._focus_lb(focus)
            self.mode_stats[focus]["last_gap"] = float((self.best.z - lb) / self.best.z) if self.best.z > 1e-9 else 0.0

            try:
                self.restore_snapshot(self.work)
                self._run_focus_pipeline(focus)
                z = float(self.evaluate())
                self._harvest_station_start_times()
                self._update_beta_from_station()
                cand_snap = self.snapshot(z, iter_id=it)
            except Exception:
                self.mode_stats[focus]["fail"] += 1.0
                self.restore_snapshot(self.work)
                self._append_layer_log(
                    iter_id=it,
                    focus=focus,
                    z_before=z_before,
                    z_after=z_before,
                    improved=False,
                    skipped=True,
                    accepted_type="error",
                    lb=lb,
                )
                mark += 1
                if mark >= self.cfg.no_improve_limit:
                    break
                continue

            improved_best = z < float(self.best.z) - 1e-6
            improved_work = z < float(self.work_z) - 1e-6
            accepted_type = "reject"
            if improved_best:
                self.best = cand_snap
                self.work = cand_snap
                self.work_z = float(z)
                self.mode_stats[focus]["success"] += 1.0
                self.mode_stats[focus]["fail"] = 0.0
                accepted_type = "strong"
                mark = 0
            elif improved_work:
                self.work = cand_snap
                self.work_z = float(z)
                self.mode_stats[focus]["success"] += 1.0
                self.mode_stats[focus]["fail"] = 0.0
                accepted_type = "improve_work"
                mark += 1
            else:
                weak_ok = z <= float(self.work_z) * (1.0 + float(self.cfg.weak_accept_eta))
                if weak_ok:
                    self.work = cand_snap
                    self.work_z = float(z)
                    self.mode_stats[focus]["success"] += 1.0
                    self.mode_stats[focus]["fail"] = 0.0
                    accepted_type = "weak"
                else:
                    self.mode_stats[focus]["fail"] += 1.0
                mark += 1

            self.restore_snapshot(self.work)
            self._append_layer_log(
                iter_id=it,
                focus=focus,
                z_before=z_before,
                z_after=float(self.work_z),
                improved=improved_best,
                skipped=False,
                accepted_type=accepted_type,
                lb=lb,
            )
            self._notify_progress(it, self.cfg.max_iters, focus)
            if mark >= self.cfg.no_improve_limit:
                break

        if self.cfg.write_iteration_logs:
            self._write_logs()
        if self.cfg.export_best_solution:
            self.export_best()
        return float(self.best.z)

    def export_best_to(self, out_dir: str):
        assert self.best is not None
        self._set_seed(self.best.seed)
        self.restore_snapshot(self.best)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        calc = RankAwareGlobalTimeCalculator(self.problem)
        z = float(calc.calculate())
        calc.calculate_and_export(out_dir)
        self._verify_makespan_breakdown(out_dir)
        self._write_best_solution_summary(out_dir, z)
        self._write_best_solution_dump(out_dir, z)
